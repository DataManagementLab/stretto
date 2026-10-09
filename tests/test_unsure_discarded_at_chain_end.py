"""Loss and execution must agree on what an unresolved "unsure" verdict means.

A logical operator runs as a cascade of tiers, each re-deciding only what the previous
one flagged unsure. The rule everywhere is: **an unsure verdict that no further tier of
the same logical step will re-decide is a DISCARD.**

The differentiable model (`get_final_keep_probabilities` returns only the KEEP mass) and
the in-database path (`SqlQuery.get_cond_list_select` hardens a step's last condition to
`> threshold_upper` when finalized) both follow this rule. On the run-outside path,
`transform_input` returns everything above the *lower* threshold, keep and unsure alike,
so the rows still unsure at the terminal tier must be discarded explicitly. These tests
pin that.

When the last tier is the gold operator, whose thresholds are pinned at
`DEFAULT_THRESHOLD_LOWER == DEFAULT_THRESHOLD_UPPER == 0.0` (a zero-width unsure band),
no row can end unsure. Once the last tier is an operator whose thresholds the optimizer
tunes, the case is reachable.
"""

import pandas as pd
import pytest

try:
    from reasondb.operators.filter.text_qa_filter import (
        DEFAULT_THRESHOLD_LOWER,
        DEFAULT_THRESHOLD_UPPER,
    )
    from reasondb.query_plan.optimized_physical_plan import MultiModalTunedPipeline
except ImportError:  # pragma: no cover - optional deps
    pytest.skip("optimizer deps not installed", allow_module_level=True)


class _FakeLogicalStep:
    def __init__(self, identifier):
        self.identifier = identifier


class _FakeStep:
    def __init__(self, identifier):
        self.logical_plan_step = _FakeLogicalStep(identifier)


class _NullLogger:
    def __truediv__(self, _other):
        return self

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def _flags(identifier, values):
    return pd.DataFrame(
        {identifier: values, "_random_id": [0.1] * len(values)},
        index=range(len(values)),
    )


def _data(n):
    return pd.DataFrame({"x": list(range(n))}, index=range(n))


# --- which tier is terminal ----------------------------------------------------


def test_only_the_last_tier_of_a_logical_step_is_terminal():
    steps = [_FakeStep("filter_a"), _FakeStep("filter_a"), _FakeStep("filter_b")]
    is_last = MultiModalTunedPipeline.is_last_tier_for_its_logical_step
    assert is_last(0, steps) is False  # a later tier will re-decide its unsure rows
    assert is_last(1, steps) is True
    assert is_last(2, steps) is True


def test_interleaved_cascades_are_tracked_per_logical_step():
    """The reorderer may interleave tiers of different logical operators, so "terminal"
    is per identifier, not simply the last index."""
    steps = [_FakeStep("a"), _FakeStep("b"), _FakeStep("a"), _FakeStep("b")]
    is_last = MultiModalTunedPipeline.is_last_tier_for_its_logical_step
    assert [is_last(i, steps) for i in range(4)] == [False, False, True, True]


# --- the discard ---------------------------------------------------------------


def test_unsure_rows_are_dropped_at_the_terminal_tier():
    step = _FakeStep("filter_a")
    data, flags = MultiModalTunedPipeline.discard_unsure(
        step=step,
        output_data=_data(4),
        sure_mask=_flags("filter_a", [True, False, True, False]),
        logger=_NullLogger(),
    )
    assert list(data.index) == [0, 2]
    assert list(flags.index) == [0, 2]


def test_a_fully_decided_step_is_untouched():
    step = _FakeStep("filter_a")
    original = _data(3)
    data, flags = MultiModalTunedPipeline.discard_unsure(
        step=step,
        output_data=original,
        sure_mask=_flags("filter_a", [True, True, True]),
        logger=_NullLogger(),
    )
    assert data is original
    assert len(flags) == 3


def test_a_step_with_no_flag_column_is_untouched():
    """Not every operator contributes a verdict column (extracts, transforms); absence
    must not be read as "everything is unsure"."""
    step = _FakeStep("extract_a")
    original = _data(3)
    data, _ = MultiModalTunedPipeline.discard_unsure(
        step=step,
        output_data=original,
        sure_mask=_flags("some_other_step", [True, False, True]),
        logger=_NullLogger(),
    )
    assert data is original


def test_the_discard_matches_what_the_loss_scores():
    """The loss counts only the KEEP mass, so an unsure row is a false negative there.
    Keeping it at execution would make the same row a *positive* -- the two disagreeing
    in opposite directions on precision."""
    kept_by_execution = MultiModalTunedPipeline.discard_unsure(
        step=_FakeStep("f"),
        output_data=_data(3),
        sure_mask=_flags("f", [True, False, True]),
        logger=_NullLogger(),
    )[0]
    scored_as_kept_by_the_loss = [0, 2]  # unsure mass is not in Decision.KEEP
    assert list(kept_by_execution.index) == scored_as_kept_by_the_loss


# --- the gold tier's zero-width unsure band -------------------------------------


def test_the_gold_operators_unsure_band_has_zero_width():
    """The gold operator as last tier can never be unsure, because its thresholds are
    pinned at equal defaults."""
    assert DEFAULT_THRESHOLD_LOWER == DEFAULT_THRESHOLD_UPPER
