"""The aggregates must describe the whole run, not a prefix of it.

Two properties, both invisible on a single-machine run and both essential on the
distributed sweep the dashboard exists to watch:

1. Every producer calls ``record_benchmark_start``, so `benchmark` and `split` are set
   on every operator bucket, query row and search-space row; otherwise a
   multi-benchmark sweep would be indistinguishable in every panel.

2. The aggregates are not capped. At roughly 35 operator buckets and 70 queries per job,
   a sweep of over a thousand jobs needs tens of thousands of buckets and query rows;
   any single-machine-sized cap would silently discard most of them.
"""

import time

import pytest

from reasondb.monitor import collector as monitor
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
    for _ in range(500):
        if collector.snapshot_run()["queue_depth"] == 0:
            break
        time.sleep(0.01)
    time.sleep(0.05)
    return collector.snapshot_aggregates()


# ── benchmark / split ────────────────────────────────────────────────────────


def test_benchmark_and_split_reach_every_operator_bucket(collector):
    monitor.record_benchmark_start(benchmark="movie_random", split="dev", n_queries=70)
    monitor.record_executor_start(executor="storage_step0", role="sweep")
    monitor.record_query_start(executor="storage_step0", role="sweep", query="q", query_index=0)
    monitor.record_operator_run(
        operator="TextQaFilter-x", operation_class="TextQaFilter", seconds=1.0,
        n_input_rows=10, runtime=1.0, monetary_cost=0.0, fake_cost=0.0, phase="execution",
    )

    (bucket,) = _drain(collector)["operator_buckets"]
    assert bucket["benchmark"] == "movie_random"
    assert bucket["split"] == "dev"


def test_benchmark_start_is_per_worker(collector):
    """Two workers running different benchmarks must not overwrite each other."""
    monitor.record_benchmark_start(benchmark="movie_random", split="dev", n_queries=70, worker_id="w1")
    monitor.record_benchmark_start(benchmark="artwork_random", split="test", n_queries=80, worker_id="w2")
    for worker, executor in (("w1", "a"), ("w2", "b")):
        monitor.record_executor_start(executor=executor, role="sweep", worker_id=worker)
        monitor.record_query_start(executor=executor, role="sweep", query="q", query_index=0, worker_id=worker)
        monitor.record_operator_run(
            operator="TextQaFilter-x", operation_class="TextQaFilter", seconds=1.0,
            n_input_rows=10, runtime=1.0, monetary_cost=0.0, fake_cost=0.0,
            phase="execution", worker_id=worker,
        )

    by_worker = {b["worker_id"]: b for b in _drain(collector)["operator_buckets"]}
    assert by_worker["w1"]["benchmark"] == "movie_random"
    assert by_worker["w1"]["split"] == "dev"
    assert by_worker["w2"]["benchmark"] == "artwork_random"
    assert by_worker["w2"]["split"] == "test"


def test_every_producer_actually_calls_benchmark_start():
    """Every producer's ``run_job`` calls ``record_benchmark_start``.

    A unit test of the collector cannot catch a missing call site - the dimensions
    would simply be None everywhere - so this asserts the call site exists.

    Checked against every *registered* producer rather than a hand-kept list, and against
    the module that actually defines each ``run_job`` rather than the one the registry
    names: the experiment wrappers in front of ``parameter_sweep`` delegate ``run_job``
    to it verbatim, so the call site lives one module over. A wrapper that grew its own
    ``run_job`` and forgot the call would still fail here.

    ``functools.partial`` is unwrapped first, because that is the second way a wrapper
    delegates: ``label_reference`` binds ``run_benchmark.run_job``'s ``producer_name``.
    Without unwrapping, ``getmodule`` resolves a partial to ``functools`` and the
    assertion would be about the standard library rather than about the producer.
    """
    import functools
    import inspect

    from reasondb.coordinator.producers import PRODUCERS

    for name, producer in sorted(PRODUCERS.items()):
        run_job = producer.run_job
        while isinstance(run_job, functools.partial):
            run_job = run_job.func
        source = inspect.getsource(inspect.getmodule(run_job))
        assert "record_benchmark_start(" in source, (
            f"{name}'s run_job does not emit benchmark_start, so benchmark/split will "
            "be None on every row this producer's jobs record."
        )


# ── Unbounded aggregates ─────────────────────────────────────────────────────


def test_operator_buckets_are_not_capped(collector):
    """Past any single-machine-sized cap: a 1 260-job sweep needs tens of thousands."""
    monitor.record_benchmark_start(benchmark="b", split="dev", n_queries=1)
    for i in range(4_200):
        monitor.record_executor_start(executor=f"step{i}", role="sweep")
        monitor.record_operator_run(
            operator="TextQaFilter-x", operation_class="TextQaFilter", seconds=1.0,
            n_input_rows=1, runtime=1.0, monetary_cost=0.0, fake_cost=0.0, phase="execution",
        )

    agg = _drain(collector)
    assert len(agg["operator_buckets"]) == 4_200
    assert agg["operator_buckets_dropped"] == 0


def test_query_rows_are_not_trimmed(collector):
    """Query rows feed the accuracy panels, so none may be silently trimmed."""
    monitor.record_executor_start(executor="e", role="sweep")
    for i in range(1_200):
        monitor.record_query_start(executor="e", role="sweep", query=f"q{i}", query_index=i)
        monitor.record_query_end(
            executor="e", role="sweep", query=f"q{i}", query_index=i, cached=False,
            n_rows=1, component_times={"end_to_end": 1.0}, tuned_pipeline=[],
        )

    assert len(_drain(collector)["query_times"]) == 1_200


def test_aggregate_sizes_report_what_is_held(collector):
    """Aggregate sizes are reported rather than capped: growth is visible, not enforced."""
    monitor.record_executor_start(executor="e", role="sweep")
    monitor.record_query_start(executor="e", role="sweep", query="q", query_index=0)
    monitor.record_operator_run(
        operator="TextQaFilter-x", operation_class="TextQaFilter", seconds=1.0,
        n_input_rows=10, runtime=1.0, monetary_cost=0.0, fake_cost=0.0, phase="execution",
    )
    monitor.record_query_end(
        executor="e", role="sweep", query="q", query_index=0, cached=False, n_rows=1,
        component_times={"end_to_end": 1.0}, tuned_pipeline=[],
    )
    _drain(collector)

    sizes = collector.snapshot_run()["aggregate_sizes"]
    assert sizes["operator_buckets"] == 1
    assert sizes["query_times"] == 1
    assert sizes["estimated_mb"] >= 0.0
    assert set(sizes) >= {
        "operator_buckets", "query_times", "query_metrics", "query_details",
        "search_space", "estimated_mb",
    }
