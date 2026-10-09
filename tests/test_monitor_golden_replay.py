"""Replay a known telemetry stream and pin every number that comes out.

The fixture is synthetic rather than a capture of a real run, for three reasons: real
sidecars live under ``benchmark_results/**/_monitor/`` which is gitignored, so CI could
never rely on one; a fixture the tests own is independent of the telemetry format of
any particular recording; and it can be built to contain exactly the awkward cases
rather than whatever a run happened to produce.

Those cases, all of which occur in real multi-worker runs:

- a labelling pass in a job of its own, *and* one inside a sweep job - which is why the
  job id cannot identify a labelling pass and the ``role`` tag has to;
- an ``operator_run`` that arrives before its worker's first ``executor_start``, so its
  role and executor are genuinely unknown;
- a ``TraditionalFilter`` run carrying no model or compression block at all, which is
  correct for an operator with no KV backend;
- the same operator, model and compression ratio used by both a labelling pass and a
  sweep point, so they can only be told apart by role.
"""

import json
import time
from pathlib import Path

import pytest

from reasondb.monitor.collector import Collector

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "telemetry-golden.jsonl"


@pytest.fixture(scope="module")
def aggregates():
    collector = Collector(jsonl_path=None, validation="strict")
    collector.install()
    try:
        for line in FIXTURE.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            collector.put(event["type"], event["data"], at=event["t"])
        for _ in range(500):
            if collector.snapshot_run()["queue_depth"] == 0:
                break
            time.sleep(0.01)
        time.sleep(0.1)
        snapshot = collector.snapshot_aggregates()
        run = collector.snapshot_run()
    finally:
        # strict: this also asserts the whole fixture satisfies EVENT_SCHEMA.
        collector.close()
    return snapshot, run


def _buckets(aggregates):
    return aggregates[0]["operator_buckets"]


def test_the_fixture_validates_cleanly(aggregates):
    """The module fixture closes the collector in strict mode, so reaching here at all
    means every event carried the keys the aggregates read."""
    _snapshot, run = aggregates
    assert run["validation_failures"] == 0


def test_operator_work_splits_by_pass(aggregates):
    """The headline number: how much of the measured work is actually labelling."""
    buckets = _buckets(aggregates)
    tuples_by_role = {}
    for b in buckets:
        tuples_by_role[b["role"]] = tuples_by_role.get(b["role"], 0) + b["input_rows"]

    # 2 label passes x 2 queries x 2 operators x 100 rows = 800
    # 1 sweep pass  x 2 queries x 2 operators x 200 rows = 800
    # 1 orphan run before any executor_start            =  10
    assert tuples_by_role == {"label": 800, "sweep": 800, None: 10}

    total = sum(tuples_by_role.values())
    labelling_share = tuples_by_role["label"] / total
    assert 0.49 < labelling_share < 0.50, (
        "roughly half this fixture's measured tuples are labelling - the same shape as "
        "the real run, where excluding them changes every operator panel"
    )


def test_a_label_pass_inside_a_sweep_job_is_still_labelling(aggregates):
    """The case the job id cannot express: one job, both kinds of pass."""
    in_sweep_job = [
        b for b in _buckets(aggregates)
        if b["job_id"] == "t-storage_sweep-movie_random-s0-p0.7-r0.7"
    ]
    assert {b["role"] for b in in_sweep_job} == {"label", "sweep"}

    # Filtering that job by `kind == "step"` would therefore keep labelling work, which
    # is exactly why statistics are filtered on `role` instead.
    labelling = [b for b in in_sweep_job if b["role"] == "label"]
    assert sum(b["input_rows"] for b in labelling) == 400


def test_an_orphan_run_is_not_attributed_to_a_job_it_may_not_belong_to(aggregates):
    """Its payload named a job, and the collector still records no job for it.

    A worker stamps `job_id` on a whole forwarded batch at flush time, not per event
    (see scripts/run_worker.py), so a job id on an event that arrived before the
    worker's first executor_start is not evidence it belongs to that job. Recording
    `None` is the honest answer; guessing would put unattributable work inside a job's
    totals.
    """
    (orphan,) = [b for b in _buckets(aggregates) if b["role"] is None]
    assert orphan["job_id"] is None


def test_the_untagged_run_survives(aggregates):
    """Neither known to be labelling nor known to be a sweep point - it must not be
    silently counted as either, and must not be dropped."""
    orphans = [b for b in _buckets(aggregates) if b["role"] is None]
    assert len(orphans) == 1
    assert orphans[0]["executor"] is None
    assert orphans[0]["input_rows"] == 10


def test_benchmark_and_split_reach_every_tagged_bucket(aggregates):
    """Every role-tagged bucket carries the benchmark and split from benchmark_start."""
    tagged = [b for b in _buckets(aggregates) if b["role"] is not None]
    assert tagged
    assert {b["benchmark"] for b in tagged} == {"movie_random"}
    assert {b["split"] for b in tagged} == {"dev"}


def test_an_operator_with_no_kv_backend_keeps_its_own_bucket(aggregates):
    """Correct behaviour, not a defect: it has no model, so it reports none."""
    plain = [b for b in _buckets(aggregates) if b["operation_class"] == "TraditionalFilter"]
    assert plain
    for bucket in plain:
        assert bucket["model_name"] is None
        assert bucket["cr_label"] == "n/a"


def test_the_same_operator_is_split_by_pass_not_merged(aggregates):
    """Same operator, model and ratio in a labelling pass and a sweep point."""
    kv_ops = [
        b for b in _buckets(aggregates)
        if b["operation_class"] == "TextQaFilter" and b["role"] is not None
    ]
    assert {b["role"] for b in kv_ops} == {"label", "sweep"}
    assert {b["model_name"] for b in kv_ops} == {"meta-llama/Llama-3.1-70B-Instruct"}
    assert {b["cr_label"] for b in kv_ops} == {"cr0.5"}


def test_inference_volume_is_attributed_to_the_pass_that_caused_it(aggregates):
    """KV buckets read the worker context, so labelling model calls are not pooled
    with the sweep's."""
    by_role = {}
    for k in aggregates[0]["kv"]:
        by_role[k["role"]] = by_role.get(k["role"], 0) + k["n_items"]
    assert by_role == {"label": 400, "sweep": 400}


def test_nothing_is_discarded(aggregates):
    """The aggregates are unbounded; these counters confirm nothing was discarded."""
    snapshot, run = aggregates
    assert snapshot["operator_buckets_dropped"] == 0
    assert snapshot["dropped_records"] == 0
    assert run["dropped_events"] == 0


def test_query_rows_and_sizes_are_complete(aggregates):
    snapshot, run = aggregates
    # 3 passes x 2 queries
    assert len(snapshot["query_times"]) == 6
    assert run["aggregate_sizes"]["query_times"] == 6
    assert run["aggregate_sizes"]["operator_buckets"] == len(_buckets(aggregates))
