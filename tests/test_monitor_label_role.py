"""A labelling pass must be distinguishable from a measured sweep point.

`collect_labels` runs a full execution over every query to produce the ground truth a
sweep is scored against. It emits exactly the telemetry a real sweep point does, so
without a tag it is pooled into every operator, timing and tuple statistic, where it
can account for a large share of all tuples processed and queries run.

The tag is `role`, stamped by `Executor` onto every event it emits, because the two
alternatives do not work:

- The executor *name* ("silver"/"gold") is a display string. Matching statistics on it
  would silently drop a sweep executor legitimately named that way.
- The job's `spec["kind"]` cannot see it either: a labelling pass also runs *inside*
  sweep jobs when the shared label cache is cold, so only a tag inside the event's own
  payload is exact.
"""

import time

import pytest

from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import CONFIG_DIMENSIONS, Collector


@pytest.fixture
def collector():
    c = Collector(jsonl_path=None)
    c.install()
    try:
        yield c
    finally:
        c.close()


def _drain(collector):
    """Let the drain thread fold what has been queued, then read the aggregates."""
    for _ in range(200):
        if collector.snapshot_run()["queue_depth"] == 0:
            break
        time.sleep(0.01)
    time.sleep(0.05)
    return collector.snapshot_aggregates()


def _operator_run(**overrides):
    payload = dict(
        operator="TextQaFilter-llama-cr0.5",
        operation_class="TextQaFilter",
        seconds=1.0,
        n_input_rows=100,
        runtime=1.0,
        monetary_cost=0.0,
        fake_cost=0.0,
        phase="execution",
        worker_id="w1",
        job_id="job-s0",
    )
    payload.update(overrides)
    monitor.record_operator_run(**payload)


def test_role_is_a_config_dimension():
    assert "role" in CONFIG_DIMENSIONS


def test_a_label_pass_inside_a_sweep_job_is_still_tagged_as_labelling(collector):
    """The case job_id cannot express, and the reason the tag exists.

    Both passes run under the same job; only the executor_start in between says which
    is which.
    """
    monitor.record_executor_start(executor="silver", role="label", worker_id="w1", job_id="job-s0")
    monitor.record_query_start(
        executor="silver", role="label", query="q1", query_index=0, worker_id="w1", job_id="job-s0"
    )
    _operator_run(n_input_rows=100)

    monitor.record_executor_start(
        executor="storage_step0", role="sweep", precision_guarantee=0.7,
        recall_guarantee=0.7, worker_id="w1", job_id="job-s0",
    )
    monitor.record_query_start(
        executor="storage_step0", role="sweep", query="q1", query_index=0,
        worker_id="w1", job_id="job-s0",
    )
    _operator_run(n_input_rows=200)

    buckets = _drain(collector)["operator_buckets"]
    by_role = {b["role"]: b for b in buckets}

    # Same operator, same job - two buckets, because the pass differs.
    assert set(by_role) == {"label", "sweep"}
    assert by_role["label"]["input_rows"] == 100
    assert by_role["sweep"]["input_rows"] == 200
    assert by_role["label"]["job_id"] == by_role["sweep"]["job_id"] == "job-s0"


def test_query_rows_carry_the_role(collector):
    monitor.record_executor_start(executor="silver", role="label", worker_id="w1", job_id="j")
    monitor.record_query_start(
        executor="silver", role="label", query="q1", query_index=0, worker_id="w1", job_id="j"
    )
    monitor.record_query_end(
        executor="silver", role="label", query="q1", query_index=0, cached=False,
        n_rows=5, component_times={"end_to_end": 1.0}, tuned_pipeline=[], worker_id="w1", job_id="j",
    )

    rows = _drain(collector)["query_times"]
    assert [r["role"] for r in rows] == ["label"]


def test_kv_inference_is_attributed_to_the_pass_that_caused_it(collector):
    """KV inference buckets are split by role as well.

    Without this, a labelling pass's model calls are pooled with the sweep's and there
    is no dimension that could separate them.
    """
    def kv(n_items):
        monitor.record_kv_inference({
            "server": "text", "model_name": "llama", "path": "kv", "n_items": n_items,
            "server_elapsed_s": 1.0, "client_elapsed_s": 1.0, "worker_id": "w1", "job_id": "j",
        })

    monitor.record_executor_start(executor="silver", role="label", worker_id="w1", job_id="j")
    kv(10)
    monitor.record_executor_start(executor="storage_step0", role="sweep", worker_id="w1", job_id="j")
    kv(20)

    by_role = {k["role"]: k for k in _drain(collector)["kv"]}
    assert set(by_role) == {"label", "sweep"}
    assert by_role["label"]["n_items"] == 10
    assert by_role["sweep"]["n_items"] == 20


def test_an_event_before_any_executor_start_keeps_a_null_role(collector):
    """An operator_run that arrives before any executor_start has an unknown role.

    It must not be dropped, and must not be silently counted as a sweep point - it is
    genuinely unknown, and the dashboard shows it as such.
    """
    _operator_run(worker_id="w-fresh", job_id="j")

    buckets = _drain(collector)["operator_buckets"]
    assert len(buckets) == 1
    assert buckets[0]["role"] is None


def test_role_is_not_defaulted_when_a_producer_omits_it(collector):
    """A forgotten role must surface as unknown, not be absorbed as "sweep".

    Defaulting would turn a producer bug into a silent mislabel of exactly the kind
    this field exists to prevent.
    """
    monitor.record_executor_start(executor="mystery", worker_id="w1", job_id="j")
    monitor.record_query_start(
        executor="mystery", query="q1", query_index=0, worker_id="w1", job_id="j"
    )
    _operator_run(worker_id="w1", job_id="j")

    assert _drain(collector)["operator_buckets"][0]["role"] is None


# ── Where the tag comes from ─────────────────────────────────────────────────


def _executor(optimizer, **kwargs):
    """An Executor with the collaborators stubbed - only `role` is under test."""
    from reasondb.executor import Executor

    class _Stub:
        def set_database(self, _db):
            pass

    from reasondb.utils.logging import FileLogger

    return Executor(
        database=_Stub(), reasoner=_Stub(), optimizer=optimizer, configurator=_Stub(),
        name="x", logger=FileLogger(), **kwargs,
    )


def test_role_is_derived_from_the_optimizer():
    """Deriving it means no caller can forget - what makes a pass a labelling pass is
    that it runs a LabelOptimizer."""
    from reasondb.optimizer.gd_optimizer import GradientDescentOptimizer, OptimizationConfig
    from reasondb.optimizer.label_optimizer import LabelOptimizer

    assert _executor(LabelOptimizer()).role == "label"
    assert _executor(GradientDescentOptimizer(OptimizationConfig())).role == "sweep"


def test_an_explicit_role_overrides_the_derivation():
    """For measuring a LabelOptimizer run as an approach in its own right."""
    from reasondb.optimizer.label_optimizer import LabelOptimizer

    assert _executor(LabelOptimizer(), role="sweep").role == "sweep"


def test_an_unknown_role_is_rejected():
    from reasondb.optimizer.label_optimizer import LabelOptimizer

    with pytest.raises(AssertionError, match="sweep.*label"):
        _executor(LabelOptimizer(), role="bronze")
