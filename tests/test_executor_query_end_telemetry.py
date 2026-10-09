"""``Executor.execute_benchmark`` must forward ``tuned_pipeline`` on every
``query_end`` telemetry event - both the cache-hit and freshly-executed paths - since
the Run tab's per-query "Analyze" panel (which plan was picked, compared across every
configuration in a sweep) reads it straight off that event (see
``reasondb.monitor.collector._Aggregates``'s ``query_times``).

Only the cache-hit path is driven end to end here: it's the one genuinely testable
without faking the reasoning/optimization machinery (``execute_benchmark``'s cached
branch returns before touching any of that), and it's also where a tuple-unpacking
ordering bug would most plausibly hide.
"""

import pandas as pd
import pytest

from reasondb.executor import CostSummary, Executor
from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector
from reasondb.query_plan.physical_operator import ProfilingCost
from reasondb.query_plan.query import Query


class _NoopComponent:
    def set_database(self, database):
        pass


@pytest.fixture(autouse=True)
def clean_sink():
    yield
    if monitor.get_collector() is not None:
        monitor.get_collector().close()


def _drain(collector: Collector, timeout: float = 2.0) -> None:
    import time

    deadline = time.time() + timeout
    while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
        time.sleep(0.01)


def test_cached_query_end_carries_the_cached_tuned_pipeline(tmp_path):
    executor = Executor(
        database=_NoopComponent(),
        reasoner=_NoopComponent(),
        optimizer=_NoopComponent(),
        configurator=_NoopComponent(),
        name="test-executor",
    )
    cache_dir = tmp_path / "cache"
    cached_pipeline = ["[]", '[{"operator": "TextQaFilter-cr0.5"}]']
    executor.cache_result(
        results_cache_dir=cache_dir,
        query_str="Extract [title] from {r.text}",
        guarantees=(),
        results=pd.DataFrame({"title": ["a"]}),
        costs=CostSummary(
            execution_cost=ProfilingCost(runtime=1.0, monetary_cost=0.0),
            tuning_cost=ProfilingCost(runtime=0.5, monetary_cost=0.0),
            component_times={"execution": 1.0},
        ),
        tuned_pipelines=cached_pipeline,
        logger=executor.logger,
    )

    collector = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        result = executor.execute_benchmark(
            [Query("Extract [title] from {r.text}")],
            results_cache_dir=cache_dir,
        )
        _drain(collector)
        events = collector.events_since(since=0, limit=100)["events"]
        query_end_events = [e for e in events if e["type"] == "query_end"]
        assert len(query_end_events) == 1
        assert query_end_events[0]["data"]["cached"] is True
        assert query_end_events[0]["data"]["tuned_pipeline"] == cached_pipeline
        # And the returned BenchmarkResult itself carries the same pipeline - the
        # telemetry event isn't inventing a value that diverges from what's returned.
        assert result.tuned_pipelines["Extract [title] from {r.text}"] == cached_pipeline
    finally:
        collector.close()
