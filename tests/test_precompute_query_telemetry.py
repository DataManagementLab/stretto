"""A precompute pass must move the same progress bar every other pass moves.

`precompute_progress` is read only by the monitor's Precompute tab. Every *query*
counter - the per-worker bar, the coordinator's whole-experiment bar
(`JobDB.job_progress` joined with `queries_done_in_job`), the queries/hour behind the
ETA - increments on `query_end` and nothing else, so `run_precompute` must emit query
events too, or a precompute task would sit at 0% for its entire run.

These query rows are tagged `role="precompute"`: this pass runs every candidate
operator over the full table and measures nothing, so it must not be poolable into a
sweep's operator and tuple statistics - the same mistake `role` was introduced to stop
for labelling passes.
"""

import time

import pytest

from reasondb.backends.simulate_store import SimulateStore
from reasondb.evaluation.precompute import PRECOMPUTE_ROLE, run_precompute
from reasondb.monitor.collector import Collector


@pytest.fixture
def collector():
    c = Collector(jsonl_path=None)
    c.install()
    try:
        yield c
    finally:
        c.close()


def _drain(collector):
    for _ in range(200):
        if collector.snapshot_run()["queue_depth"] == 0:
            break
        time.sleep(0.01)
    time.sleep(0.05)
    return collector


class _Query:
    def __init__(self, text: str) -> None:
        self.query = text


class _Benchmark:
    def __init__(self, *queries: str) -> None:
        self.queries = [_Query(q) for q in queries]


class _Executor:
    """Records one text-QA answer per query, unless told this query has nothing new.

    That is the whole behaviour under test: a query all of whose operators are already
    in the store adds no entries, which is what `cached` must reflect.
    """

    def __init__(self, novel) -> None:
        self.novel = novel
        self.seen = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def precompute_query(self, query, index: int) -> None:
        self.seen.append(query.query)
        if not self.novel(query.query):
            return
        SimulateStore.get_precompute().record_text_qa(
            model_id="llama",
            question=query.query,
            context="ctx",
            response="yes",
            log_odds=0.0,
            runtime=0.1,
            effective_compression_ratio=0.0,
            materialized_compression_ratio=0.0,
            vanilla=False,
        )


def _run(tmp_path, benchmark, executor, **kwargs):
    return run_precompute(
        benchmark, executor, tmp_path / "store.json", progress_label="job-pre", **kwargs
    )


def test_every_query_reports_a_start_and_an_end(tmp_path, collector):
    """The events the query counters are built from. Without them the bar cannot move."""
    _run(tmp_path, _Benchmark("q1", "q2", "q3"), _Executor(lambda _q: True))

    rows = _drain(collector).snapshot_aggregates()["query_times"]
    assert [r["query"] for r in rows] == ["q1", "q2", "q3"]
    assert [r["query_index"] for r in rows] == [0, 1, 2]


def test_the_worker_bar_advances_within_the_pass(tmp_path, collector):
    """`queries_done_in_job` is what `coordinator/app.py` joins onto the job queue.

    It counts `query_end` events, so it must advance during a precompute pass.
    """
    _run(tmp_path, _Benchmark("q1", "q2", "q3"), _Executor(lambda _q: True))

    run = _drain(collector).snapshot_run()
    assert run["queries_done"] == 3
    assert run["queries_done_in_flight"] == 3


def test_rows_are_tagged_as_a_precompute_pass(tmp_path, collector):
    """Not "sweep": this pass runs every candidate operator and measures nothing."""
    _run(tmp_path, _Benchmark("q1"), _Executor(lambda _q: True))

    rows = _drain(collector).snapshot_aggregates()["query_times"]
    assert [r["role"] for r in rows] == [PRECOMPUTE_ROLE]
    assert PRECOMPUTE_ROLE not in ("sweep", "label")


def test_a_query_that_added_nothing_is_reported_as_cached(tmp_path, collector):
    """The resumed-pass case, and why a flat store-growth chart is not a stalled run.

    Queries whose operators are already recorded cost no inference and grow the store by
    nothing; they are still progress, and the bar must count them.
    """
    _run(tmp_path, _Benchmark("new", "skipped"), _Executor(lambda q: q == "new"))

    rows = _drain(collector).snapshot_aggregates()["query_times"]
    assert [(r["query"], r["cached"]) for r in rows] == [("new", False), ("skipped", True)]


def test_the_precompute_tab_still_gets_its_own_progress(tmp_path, collector):
    """The store-growth event is unchanged - the query rows are additional, not a swap."""
    _run(tmp_path, _Benchmark("q1", "q2"), _Executor(lambda _q: True))

    progress = _drain(collector).snapshot_aggregates()["precompute_progress"]
    assert [p["query_index"] for p in progress] == [0, 1]
    assert [p["n_text_qa"] for p in progress] == [1, 2]
    assert {p["n_queries"] for p in progress} == {2}


def test_debug_query_narrows_the_denominator_too(tmp_path, collector):
    """`n_queries` on the events must describe the queries this pass will actually run,
    or the bar is drawn against a total it can never reach."""
    _run(
        tmp_path,
        _Benchmark("q1", "q2", "q3"),
        _Executor(lambda _q: True),
        debug_query="q2",
    )

    agg = _drain(collector).snapshot_aggregates()
    assert [p["n_queries"] for p in agg["precompute_progress"]] == [1]
    assert [r["query"] for r in agg["query_times"]] == ["q2"]
