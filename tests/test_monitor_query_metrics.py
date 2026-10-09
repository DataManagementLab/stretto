"""``evaluate()`` is the only place per-query accuracy exists.

Precision/recall are scored against labels *after* a run, so nothing during execution
can report them - which is why the dashboard's accuracy panels are fed from here rather
than from ``query_end``. Emitting at this one point covers every caller: run_benchmark,
the storage/runtime and sample-size sweeps, and the coordinator's merge pass.
"""

import pandas as pd

from reasondb.evaluation.row_signature import RowSignature
import pytest

from reasondb.executor import CostSummary
from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector
from reasondb.query_plan.physical_operator import ProfilingCost


@pytest.fixture(autouse=True)
def clean_sink():
    yield
    if monitor.get_collector() is not None:
        monitor.get_collector().close()


def _cost():
    c = CostSummary(ProfilingCost(1.0, 0.0), ProfilingCost(0.5, 0.0))
    c.component_times = {"end_to_end": 2.0, "execution": 1.0}
    return c


def test_evaluate_emits_query_metrics_per_guarantee(tmp_path):
    from reasondb.evaluation.evaluation import evaluate

    labels = {"q1": RowSignature.of(pd.DataFrame({"title": ["a", "b", "c"]}))}
    preds = {"q1": {
        (0.7, 0.7): RowSignature.of(pd.DataFrame({"title": ["a", "b"]})),
        (0.9, 0.9): RowSignature.of(pd.DataFrame({"title": ["a", "b", "c"]})),
    }}
    costs = {"q1": {(0.7, 0.7): _cost(), (0.9, 0.9): _cost()}}

    c = Collector(jsonl_path=None).install()
    try:
        evaluate("movie_random", "optim_global", preds, labels, costs,
                 debug_root=tmp_path / "dbg")
        import time
        deadline = time.time() + 3
        while c.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
            time.sleep(0.01)
        rows = c.snapshot_aggregates()["query_metrics"]
        assert len(rows) == 2, rows
        assert {(r["precision_target"], r["recall_target"]) for r in rows} == {(0.7, 0.7), (0.9, 0.9)}
        for r in rows:
            assert r["executor"] == "optim_global"
            assert r["benchmark"] == "movie_random"
            assert r["precision_achieved"] is not None
            assert r["recall_achieved"] is not None
    finally:
        c.close()


def test_evaluate_works_unmonitored(tmp_path):
    """The recorder must be a no-op with no collector installed."""
    from reasondb.evaluation.evaluation import evaluate

    labels = {"q1": RowSignature.of(pd.DataFrame({"title": ["a", "b"]}))}
    preds = {"q1": RowSignature.of(pd.DataFrame({"title": ["a"]}))}
    costs = {"q1": _cost()}
    df = evaluate("b", "a", preds, labels, costs, debug_root=tmp_path / "dbg")
    assert len(df) == 1
