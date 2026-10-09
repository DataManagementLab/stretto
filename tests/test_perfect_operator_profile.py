"""Tests for the ``Perfect*`` operators acting as a label source during profiling.

The base `PhysicalOperator.profile` passes ``labels=None`` to `run_outside_db`, but the
perfect operators require a `LabelsDefinition`, so they read it off the observation's
logical plan step. This lets them sit at the end of a step's candidate list as the thing
`Profiler.get_labels` derives labels from.

The saturated +-1000 matrix shape is the contract: `Profiler.get_labels` hard-argmaxes it
into the boolean ``keep_labels`` vector the optimizer measures precision and recall
against.
"""

import asyncio
from pathlib import Path

import pandas as pd
import pytest

try:
    import torch

    from reasondb.evaluation.benchmark import LabelsDefinition
    from reasondb.operators.perfect_operators.label_lookup import (
        MissingLabelsError,
        index_column_names,
        lookup_labels,
    )
    from reasondb.operators.perfect_operators.perfect_extract import PerfectExtract
    from reasondb.operators.perfect_operators.perfect_filter import PerfectFilter
    from reasondb.reasoning.observation import FilterOnComputedDataObservation
except ImportError:  # pragma: no cover - optional deps
    pytest.skip("operator deps not installed", allow_module_level=True)


def _labels_csv(tmp_path: Path) -> Path:
    path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "_index_artworks": [0, 1, 2, 3],
            "religious": [1, 0, 1, 0],
            "century": ["16", "17", "16", "18"],
        }
    ).to_csv(path, index=False)
    return path


def _input_data(row_ids=(0, 1, 2, 3)) -> pd.DataFrame:
    return pd.DataFrame(
        {"title": [f"t{i}" for i in row_ids]},
        index=pd.MultiIndex.from_arrays([list(row_ids)], names=["_index_artworks"]),
    )


# --- lookup_labels -------------------------------------------------------------


def test_lookup_returns_labels_in_input_order(tmp_path):
    labels = LabelsDefinition(_labels_csv(tmp_path), "religious", ["artworks"])
    result = lookup_labels(labels, _input_data((2, 0, 3)), "PerfectFilter")
    assert result.tolist() == [1, 1, 0]


def test_index_column_names_sorts_base_tables():
    assert index_column_names(["reports", "players"]) == [
        "_index_players",
        "_index_reports",
    ]


def test_missing_row_names_the_file_and_the_gap(tmp_path):
    """A partial label file must fail loudly. Silently treating an uncovered tuple as a
    negative is indistinguishable from a real negative label."""
    labels = LabelsDefinition(_labels_csv(tmp_path), "religious", ["artworks"])
    with pytest.raises(MissingLabelsError) as excinfo:
        lookup_labels(labels, _input_data((0, 99)), "PerfectFilter")
    message = str(excinfo.value)
    assert "labels.csv" in message
    assert "99" in message
    assert "1 of 2" in message


def test_missing_column_lists_what_is_available(tmp_path):
    labels = LabelsDefinition(_labels_csv(tmp_path), "not_a_column", ["artworks"])
    with pytest.raises(MissingLabelsError) as excinfo:
        lookup_labels(labels, _input_data(), "PerfectFilter")
    assert "not_a_column" in str(excinfo.value)
    assert "religious" in str(excinfo.value)


def test_missing_file_is_reported_before_pandas_sees_it(tmp_path):
    labels = LabelsDefinition(tmp_path / "absent.csv", "religious", ["artworks"])
    with pytest.raises(MissingLabelsError, match="does not exist"):
        lookup_labels(labels, _input_data(), "PerfectFilter")


# --- cost ----------------------------------------------------------------------


def test_a_label_operator_reports_no_cost(tmp_path):
    """What a human label costs is a modeling assumption this repo has no basis to make,
    and a made-up per-label price would look like a measurement in the results. The
    measurable quantity is the label *count* -- see `ProfilingOutput.n_labels_requested`.

    Zero also matters mechanically: since `profile()` runs these operators, any
    per-row cost they report (e.g. a large sentinel) would reach the optimizer's cost
    accounting."""
    labels = LabelsDefinition(_labels_csv(tmp_path), "religious", ["artworks"])
    op = PerfectFilter(quality=1.0, fake_cost=0.0)
    _, _, cost = _run(
        op.profile(
            inputs=["artworks"],
            database_state=None,
            observation=_observation(labels),
            llm_parameters={},
            sample=None,
            data_sample=[_input_data()],
            logger=_NullLogger(),
        )
    )
    assert (cost.runtime, cost.monetary_cost, cost.fake_cost) == (0.0, 0.0, 0.0)


def test_perfect_operators_are_label_only():
    assert PerfectFilter(quality=1.0, fake_cost=0.0).is_label_only is True
    assert PerfectExtract(quality=1.0, fake_cost=0.0).is_label_only is True


# --- profile() -----------------------------------------------------------------


class _FakeLogicalStep:
    """Only what `FilterOnComputedDataObservation.__init__` and `profile` read: the
    validation flag and the labels. Building a real `LogicalFilter` would drag in a live
    `IntermediateState` for no extra coverage."""

    validated = True

    def __init__(self, labels):
        self._labels = labels

    def get_labels(self):
        return self._labels


class _FakeHiddenColumns:
    """`FilterOnComputedDataObservation.__init__` only dereferences `.hidden_column`;
    nothing on the profile path touches it further."""

    hidden_column = None


def _observation(labels):
    """A real `FilterOnComputedDataObservation`, so the test exercises the actual
    `transform_input` the profile path calls -- including that `random_ids=None` keeps
    the gold-mixing branch inert."""
    return FilterOnComputedDataObservation(
        hidden_columns=_FakeHiddenColumns(),
        logical_plan_step=_FakeLogicalStep(labels),
        quality=1.0,
    )


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_filter_profile_emits_saturated_matrix_matching_the_csv(tmp_path):
    labels = LabelsDefinition(_labels_csv(tmp_path), "religious", ["artworks"])
    observation = _observation(labels)
    op = PerfectFilter(quality=1.0, fake_cost=0.0)

    data = _input_data()
    out_tuples, matrix, cost = _run(
        op.profile(
            inputs=["artworks"],
            database_state=None,
            observation=observation,
            llm_parameters={},
            sample=None,
            data_sample=[data],
            logger=_NullLogger(),
        )
    )

    assert list(out_tuples.index) == list(data.index)
    assert matrix.shape == (4, 1, 3)
    # KEEP wins exactly where the CSV says 1, DISCARD elsewhere, UNSURE never.
    decisions = torch.argmax(matrix, dim=2).reshape(-1)
    assert decisions.tolist() == [0, 1, 0, 1]
    assert (matrix[:, 0, 2] == -1000).all()
    assert cost.runtime == 0.0  # see test_a_label_operator_reports_no_cost


def test_filter_profile_propagates_missing_labels(tmp_path):
    """Must raise rather than degrade -- `Profiler.profile_level` only catches `Mistake`,
    and its fallback fabricates all-KEEP labels."""
    labels = LabelsDefinition(_labels_csv(tmp_path), "religious", ["artworks"])
    op = PerfectFilter(quality=1.0, fake_cost=0.0)
    with pytest.raises(MissingLabelsError):
        _run(
            op.profile(
                inputs=["artworks"],
                database_state=None,
                observation=_observation(labels),
                llm_parameters={},
                sample=None,
                data_sample=[_input_data((0, 1, 77))],
                logger=_NullLogger(),
            )
        )


class _NullLogger:
    def __truediv__(self, _other):
        return self

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def debug(self, *_args, **_kwargs):
        pass
