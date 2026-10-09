"""Deriving a benchmark's filter stats from a ``--precompute`` file, without models.

The recording already holds the gold model's answer to every pool filter, keyed by the
question its pinned config produced. Installing that file as the *replay* store and
running the single-filter queries therefore reproduces the matrix exactly: the operators
apply their own thresholds and answer conversion, and every model call becomes a lookup.

Two properties make it safe to run ahead of a coordinator task:

- an answer the file does not hold raises, rather than falling back to a live model, so
  an incomplete recording is reported instead of half-used;
- once the matrix and the query set are written, the coordinator enumerates no
  filter-stats job for that dataset, which is what makes this worth running.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reasondb.backends.simulate_store import SimulateStore
from reasondb.coordinator.producers import filter_stats_jobs as fsj
from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.evaluation import filter_stats as fs
from reasondb.evaluation.benchmark import RandomBenchmark
from reasondb.query_plan.logical_plan import LogicalFilter
from reasondb.query_plan.query import (
    OperatorOption,
    OperatorPlaceholder,
    QueryShape,
)

SHAPE = QueryShape(
    OperatorPlaceholder(
        LogicalFilter,
        inputs=[VirtualTableIdentifier("rows")],
        output=VirtualTableIdentifier("output"),
    ),
)

#: Row ids each pool filter keeps, as the fake executor reports them.
KEPT = {0: [0, 1], 1: [1, 2]}


@pytest.fixture(autouse=True)
def _no_leaked_store():
    yield
    SimulateStore.set_precompute(None)
    SimulateStore.set_simulate(None)


def _benchmark_cls(tmp_path, name="fake_random"):
    options = [OperatorOption(LogicalFilter, "{rows.text} matches p%d" % i) for i in KEPT]

    class FakeRandom(RandomBenchmark):
        @classmethod
        def name(cls):
            return name

        @classmethod
        def dir(cls):
            return tmp_path / name

        @classmethod
        def filter_stats_dir(cls, split):
            return tmp_path / "stats" / name / str(split)

        @property
        def has_ground_truth(self):
            return False

        @staticmethod
        def urls():
            return {}

        @staticmethod
        def download(split):
            raise AssertionError("not used")

        @classmethod
        def _load_database(cls, split):
            return object()

        @classmethod
        def _get_query_shapes(cls):
            return [SHAPE]

        @classmethod
        def _get_operator_options(cls):
            return options

        @classmethod
        def _single_filter_shape(cls):
            return SHAPE

    return FakeRandom


class _ReplayExecutor:
    """Stands in for the gold labeler, and asserts it ran under a replay store."""

    def __init__(self, plan):
        self.plan = plan
        self.saw_simulate_store = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_benchmark(self, queries, **kwargs):
        self.saw_simulate_store = SimulateStore.get_simulate()
        # A materialized result carries one `_index_<table>` level per base table, so
        # pandas yields a tuple per row even for a single table - and numpy integers
        # inside it. The payload has to survive json.dump anyway.
        results = {
            query.query: pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    [(np.int64(row),) for row in KEPT[i]], names=["_index_rows"]
                )
            )
            for i, (_key, _option, query) in enumerate(self.plan)
        }
        return type(
            "Result", (), {"results": results, "tuned_pipelines": {}, "costs": {}}
        )()


def _prepared(tmp_path, monkeypatch, name="fake_random"):
    cls = _benchmark_cls(tmp_path, name)
    executor = _ReplayExecutor(cls.single_filter_plan())

    def build(benchmark, *, replay_only=False):
        executor.replay_only = replay_only
        return executor

    monkeypatch.setattr(fs, "build_filter_stats_executor", build)
    store_path = tmp_path / f"{name}.json"
    SimulateStore().save(store_path)
    return cls, executor, store_path


def test_the_matrix_is_written_into_the_precompute_file(tmp_path, monkeypatch):
    cls, _executor, store_path = _prepared(tmp_path, monkeypatch)

    payload = fs.derive_filter_stats_from_recordings(cls, "dev", store_path)

    assert payload["keys"][""]["overlap_matrix"] == [[1, 1, 0], [0, 1, 1]]
    reloaded = SimulateStore.load(store_path)
    assert reloaded.get_filter_stats("fake_random", "dev") == payload


def test_the_payload_is_json_serialisable(tmp_path, monkeypatch):
    """It is written straight into the store's JSON, so a numpy integer or a raw index
    tuple anywhere in it fails the whole derivation *after* every query has run."""
    cls, _executor, store_path = _prepared(tmp_path, monkeypatch)

    payload = fs.derive_filter_stats_from_recordings(cls, "dev", store_path)

    json.dumps(payload)  # raises TypeError on numpy scalars or tuples
    # Row ids identify a result row per base table, so each is a list of ints.
    assert payload["keys"][""]["row_ids"] == [[0], [1], [2]]


def test_it_runs_against_the_replay_store_so_no_model_is_contacted(tmp_path, monkeypatch):
    """Installing the file as the replay store is what turns every model call into a
    lookup - and what makes a missing answer raise instead of reaching a server."""
    cls, executor, store_path = _prepared(tmp_path, monkeypatch)

    fs.derive_filter_stats_from_recordings(cls, "dev", store_path)

    assert executor.saw_simulate_store is not None
    assert executor.replay_only is True
    # And it is cleared afterwards, so nothing later in the process replays by accident.
    assert SimulateStore.get_simulate() is None


def test_a_replay_only_suite_prepares_against_no_embedding_server():
    """`PlanConfigurator.prepare` prepares every operator in the suite, and the image
    similarity operator embeds its images locally instead of replaying them - so leaving
    it in would demand a live server for a pass that never selects it."""
    from reasondb.interface.config import get_default_configurator
    from reasondb.operators.filter.image_embed_filter import ImageSimilarityFilter

    full = get_default_configurator()
    lean = fs.without_live_embedding_operators(full)

    assert any(isinstance(op, ImageSimilarityFilter) for op in full.physical_operators)
    assert not any(isinstance(op, ImageSimilarityFilter) for op in lean.physical_operators)
    # Everything else survives: the gold operator must still be there to be selected.
    dropped = len(list(full.physical_operators)) - len(list(lean.physical_operators))
    assert dropped == 1


def test_the_payload_names_the_filters_it_covers(tmp_path, monkeypatch):
    cls, _executor, store_path = _prepared(tmp_path, monkeypatch)

    payload = fs.derive_filter_stats_from_recordings(cls, "dev", store_path)

    assert payload["source"] == "recordings"
    assert payload["covered_filters"][""] == sorted(
        "{rows.text} matches p%d" % i for i in KEPT
    )


def test_a_second_run_is_a_no_op(tmp_path, monkeypatch):
    """Safe to re-run over a directory of stores without redrawing query sets."""
    cls, executor, store_path = _prepared(tmp_path, monkeypatch)
    first = fs.derive_filter_stats_from_recordings(cls, "dev", store_path)

    executor.saw_simulate_store = None
    again = fs.derive_filter_stats_from_recordings(cls, "dev", store_path)

    assert again == first
    assert executor.saw_simulate_store is None, "it should not have executed anything"


def test_force_recomputes_an_existing_matrix(tmp_path, monkeypatch):
    cls, executor, store_path = _prepared(tmp_path, monkeypatch)
    fs.derive_filter_stats_from_recordings(cls, "dev", store_path)

    executor.saw_simulate_store = None
    fs.derive_filter_stats_from_recordings(cls, "dev", store_path, force=True)

    assert executor.saw_simulate_store is not None


def test_a_missing_recording_surfaces_rather_than_reaching_a_model(tmp_path, monkeypatch):
    """The replay store raises on a lookup miss; that error must reach the caller."""
    cls, _executor, store_path = _prepared(tmp_path, monkeypatch)

    def explode(benchmark, *, replay_only=False):
        class _Boom:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute_benchmark(self, queries, **kwargs):
                raise RuntimeError(
                    "Simulate mode: missing precomputed result for model=gold"
                )

        return _Boom()

    monkeypatch.setattr(fs, "build_filter_stats_executor", explode)

    with pytest.raises(RuntimeError, match="missing precomputed result"):
        fs.derive_filter_stats_from_recordings(cls, "dev", store_path)

    assert SimulateStore.get_simulate() is None
    assert SimulateStore.load(store_path).get_filter_stats("fake_random", "dev") is None


def test_pinning_the_query_set_writes_queries_json(tmp_path, monkeypatch):
    cls, _executor, store_path = _prepared(tmp_path, monkeypatch)
    payload = fs.derive_filter_stats_from_recordings(cls, "dev", store_path)

    n = fs.pin_query_set(cls, "dev", payload)

    written = cls.benchmark_dir() / "dev" / "queries.json"
    assert written.exists()
    assert len(json.loads(written.read_text())) == n > 0


def test_afterwards_the_coordinator_enumerates_no_filter_stats_job(
    tmp_path, monkeypatch
):
    """The point of running it: the whole first stage disappears from the task."""
    cls, _executor, store_path = _prepared(tmp_path, monkeypatch)
    payload = fs.derive_filter_stats_from_recordings(cls, "dev", store_path)
    fs.write_stats_file(cls.filter_stats_dir("dev"), payload)
    fs.pin_query_set(cls, "dev", payload)

    monkeypatch.setattr(fsj, "BENCHMARKS", {"fake_random": cls})
    import argparse

    args = argparse.Namespace(
        benchmarks=["fake_random"], split="dev",
        precompute={"fake_random": store_path}, simulate=None,
    )
    assert fsj.enumerate_filter_stats_jobs("t1", tmp_path, args, "run_benchmark") == []


def test_without_the_query_set_the_job_is_still_enumerated(tmp_path, monkeypatch):
    """Both halves matter: the matrix makes sampling possible, the pinned file makes
    every later job run the same queries."""
    cls, _executor, store_path = _prepared(tmp_path, monkeypatch)
    payload = fs.derive_filter_stats_from_recordings(cls, "dev", store_path)
    fs.write_stats_file(cls.filter_stats_dir("dev"), payload)

    monkeypatch.setattr(fsj, "BENCHMARKS", {"fake_random": cls})
    import argparse

    args = argparse.Namespace(
        benchmarks=["fake_random"], split="dev",
        precompute={"fake_random": store_path}, simulate=None,
    )
    assert len(fsj.enumerate_filter_stats_jobs("t1", tmp_path, args, "run_benchmark")) == 1
