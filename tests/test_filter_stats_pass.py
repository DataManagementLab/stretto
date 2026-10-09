"""The pass produces three artifacts, and the store is the one that must stay consistent.

`stats.json` and `queries.json` are derived: both can be rebuilt from the store, which is
rewritten whole by a single `save`. The store cannot be rebuilt from them, because it also
holds the pinned prompts and the gold responses the matrix was computed from. So the store
is written first, and a crash in between leaves the authoritative artifact intact.

Under `--simulate` the pass executes nothing at all: it replays the recorded matrix. That
is what lets a simulate worker - which has no KV servers and no OPENAI_API_KEY - run the
phase-0 job at all. A store without the bucket must therefore fail loudly rather than fall
back to computing one, since computing needs exactly what that worker lacks.
"""

import json

import pandas as pd
import pytest

from reasondb.backends.simulate_store import SimulateStore
from reasondb.evaluation import filter_stats as fs
from reasondb.database.indentifier import VirtualTableIdentifier
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

#: Row ids each predicate keeps. p0 and p1 overlap on row 1; p2 is disjoint from both.
KEPT = {0: [0, 1], 1: [1, 2], 2: [3]}


class _FakeDatabase:
    external_tables = []


def _benchmark_cls(tmp_path):
    options = [
        OperatorOption(LogicalFilter, "{rows.text} matches p%d" % i) for i in KEPT
    ]

    class FakeRandom(RandomBenchmark):
        @classmethod
        def name(cls):
            return "fake_random"

        @classmethod
        def dir(cls):
            return tmp_path / "fake_random"

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
            return _FakeDatabase()

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


class _FakeExecutor:
    """Stands in for the gold labeler: returns the rows each predicate keeps."""

    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_benchmark(self, queries, **kwargs):
        self.calls += 1
        results = {
            query.query: pd.DataFrame(index=pd.Index(KEPT[i], name="_index_rows"))
            for i, (_key, _option, query) in enumerate(self.plan)
        }
        return type(
            "Result", (), {"results": results, "tuned_pipelines": {}, "costs": {}}
        )()


@pytest.fixture(autouse=True)
def _no_leaked_store():
    yield
    SimulateStore.set_precompute(None)
    SimulateStore.set_simulate(None)


def _run(tmp_path, monkeypatch, cls, **kwargs):
    executor = _FakeExecutor(cls.single_filter_plan())
    monkeypatch.setattr(fs, "build_filter_stats_executor", lambda benchmark: executor)
    summary = fs.run_filter_stats_pass(
        cls,
        "dev",
        simulate=False,
        store_path=tmp_path / "fake.json",
        stats_dir=tmp_path / "stats",
        **kwargs,
    )
    return executor, summary


def test_the_matrix_lands_in_the_store_and_in_the_inspection_copy(tmp_path, monkeypatch):
    cls = _benchmark_cls(tmp_path)
    _executor, _summary = _run(tmp_path, monkeypatch, cls)

    store = SimulateStore.load(tmp_path / "fake.json")
    payload = store.get_filter_stats("fake_random", "dev")
    assert payload is not None
    matrix = payload["keys"][""]["overlap_matrix"]
    # Columns are the union of kept rows: [0, 1, 2, 3].
    assert payload["keys"][""]["row_ids"] == [0, 1, 2, 3]
    assert matrix == [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]]

    on_disk = json.loads((tmp_path / "stats" / "stats.json").read_text())
    assert on_disk["keys"][""]["overlap_matrix"] == matrix


def test_the_store_is_installed_while_the_pass_runs_and_cleared_after(tmp_path, monkeypatch):
    """Without an installed precompute store `_pin_operator_config` is a no-op and the
    backends record nothing - the pass would build its matrix from un-pinned prompts,
    which is exactly the drift it exists to remove."""
    cls = _benchmark_cls(tmp_path)
    seen = {}

    executor = _FakeExecutor(cls.single_filter_plan())
    original = executor.execute_benchmark

    def spy(queries, **kwargs):
        seen["store"] = SimulateStore.get_precompute()
        return original(queries, **kwargs)

    executor.execute_benchmark = spy
    monkeypatch.setattr(fs, "build_filter_stats_executor", lambda benchmark: executor)

    fs.run_filter_stats_pass(
        cls, "dev", simulate=False,
        store_path=tmp_path / "fake.json", stats_dir=tmp_path / "stats",
    )

    assert seen["store"] is not None
    assert SimulateStore.get_precompute() is None


def test_the_pass_pins_the_query_set_it_just_made_possible(tmp_path, monkeypatch):
    cls = _benchmark_cls(tmp_path)
    _executor, summary = _run(tmp_path, monkeypatch, cls)

    written = cls.benchmark_dir() / "dev" / "queries.json"
    assert written.exists()
    assert len(json.loads(written.read_text())) == summary["n_queries"] > 0


def test_a_simulate_pass_replays_the_matrix_and_executes_nothing(tmp_path, monkeypatch):
    cls = _benchmark_cls(tmp_path)
    _run(tmp_path, monkeypatch, cls)
    (cls.benchmark_dir() / "dev" / "queries.json").unlink()

    SimulateStore.set_simulate(SimulateStore.load(tmp_path / "fake.json"))
    monkeypatch.setattr(
        fs, "build_filter_stats_executor",
        lambda benchmark: pytest.fail("simulate must not build an executor"),
    )

    summary = fs.run_filter_stats_pass(
        cls, "dev", simulate=True, store_path=None, stats_dir=tmp_path / "stats2",
    )

    assert summary["n_queries"] > 0
    assert (cls.benchmark_dir() / "dev" / "queries.json").exists()


def test_a_simulate_pass_against_a_pre_bucket_store_fails_loudly(tmp_path, monkeypatch):
    """A precompute JSON may hold no filter-stats matrix. Falling back to
    computing one would need the model servers a simulate worker never started."""
    cls = _benchmark_cls(tmp_path)
    SimulateStore.set_simulate(SimulateStore())

    with pytest.raises(RuntimeError, match="no filter stats recorded"):
        fs.run_filter_stats_pass(
            cls, "dev", simulate=True, store_path=None, stats_dir=tmp_path / "stats",
        )


def test_the_executor_results_cache_is_off_by_default(tmp_path, monkeypatch):
    """A cache hit returns the previous run's rows without calling a model, so the matrix
    would be rebuilt from stale results and the store would come out empty - a silent
    no-op that looks exactly like success. Recording passes must opt in explicitly."""
    cls = _benchmark_cls(tmp_path)
    seen = {}

    executor = _FakeExecutor(cls.single_filter_plan())
    original = executor.execute_benchmark

    def spy(queries, **kwargs):
        seen["results_cache_dir"] = kwargs.get("results_cache_dir")
        return original(queries, **kwargs)

    executor.execute_benchmark = spy
    monkeypatch.setattr(fs, "build_filter_stats_executor", lambda benchmark: executor)

    fs.run_filter_stats_pass(
        cls, "dev", simulate=False,
        store_path=tmp_path / "fake.json", stats_dir=tmp_path / "stats",
    )

    assert seen["results_cache_dir"] is None
