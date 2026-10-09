"""Per-job scoring: a finished job's accuracy reaches the monitor as soon as its labels
exist, and not one moment before.

The behaviour under test is a scheduling one, not an arithmetic one: scoring each job as
it finishes, rather than once after *every* job in the task is terminal, keeps the
dashboard's guarantee panel filling during a sweep. These tests pin the three properties
that make scoring-as-you-go safe: it waits for real labels, it never reports the same measurement twice, and silver
and gold advance independently.

Most of them monkeypatch ``evaluate`` to record the calls, because the decision under
test is *which* (job, label set) pairs get evaluated with *which* shards - not the
metric maths, which ``MetricsManager`` owns. The last test wires the real thing through
a real ``Collector`` to prove the emitted row carries what the dashboard groups on.
"""

import pickle
import time
from pathlib import Path

import pandas as pd
import pytest
from conftest import cached_answer

from reasondb.coordinator.db import JobDB
from reasondb.coordinator.models import Job, Worker
from reasondb.coordinator.producers import run_benchmark as rbp
from reasondb.coordinator.producers import shards as shards_mod
from reasondb.coordinator.producers.shards import manifest_base, relative_manifest
from reasondb.coordinator.scoring import JobScorer
from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector


@pytest.fixture(autouse=True)
def clean_sink():
    yield
    if monitor.get_collector() is not None:
        monitor.get_collector().close()


def _drain(collector: Collector, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
        time.sleep(0.01)


class _Harness:
    """A task whose jobs complete one at a time, leaving the rest genuinely pending.

    Jobs are added with an explicit ``priority`` and finished in that order, because
    ``claim_next_job`` hands out the lowest priority first and there is no API to
    complete a job without claiming it - a helper that claimed its way to a target
    would silently complete everything it passed over, which is exactly the state
    ("this label job is not done yet") these tests need to hold.
    """

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.db = JobDB(tmp_path / "coord.db")
        self.db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))

    def add(self, job_id, kind, name, benchmark="bench_a", queries=("q1", "q2"), priority=0):
        out_dir = self.tmp_path / f"job_{job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # A shard's `results` is a manifest into the executor's result cache, so each
        # answer needs a file to point at.
        answers = {
            q: cached_answer(out_dir / "cache", q, pd.DataFrame({"a": [1]}))
            for q in queries
        }
        if kind == "approach":
            results = {q: {(0.7, 0.7): answers[q]} for q in queries}
            costs = {q: {(0.7, 0.7): _cost()} for q in queries}
            tracks = {q: {(0.7, 0.7): ["[]"]} for q in queries}
        else:
            results = {q: answers[q] for q in queries}
            costs = {q: _cost() for q in queries}
            tracks = {q: ["[]"] for q in queries}
        with open(out_dir / "shard.pkl", "wb") as f:
            pickle.dump(
                {"benchmark": benchmark, "split": "dev", "kind": kind, "name": name,
                 "results": relative_manifest(results, manifest_base(out_dir)),
                 "pipeline_tracks": tracks, "costs": costs},
                f,
            )
        job = Job(
            job_id=job_id, task_id="t1", producer="run_benchmark", benchmark=benchmark,
            split="dev",
            spec={"kind": kind, "name": name,
                  "guarantee": [0.7, 0.7] if kind == "approach" else None},
            required_capabilities=[], output_dir=str(out_dir), max_attempts=1,
            priority=priority,
        )
        self.db.enqueue_job(job)
        return job

    def finish(self, job_id):
        """Complete the next job the scheduler offers, asserting it is the expected one."""
        claimed = self.db.claim_next_job("w1", "both")
        assert claimed is not None, f"no job to claim when finishing {job_id}"
        assert claimed.job_id == job_id, (
            f"expected to finish {job_id}, but the scheduler offered {claimed.job_id} "
            "- check the priorities this test assigned."
        )
        self.db.start_job(claimed.job_id, "w1")
        self.db.complete_job(claimed.job_id, "w1", {})


def _cost():
    from reasondb.executor import CostSummary
    from reasondb.query_plan.physical_operator import ProfilingCost

    return CostSummary(
        execution_cost=ProfilingCost(runtime=1.0, monetary_cost=0.0),
        tuning_cost=ProfilingCost(runtime=0.5, monetary_cost=0.0),
        component_times={"end_to_end": 2.0, "execution": 1.0},
    )


@pytest.fixture
def calls(monkeypatch):
    """Capture evaluate() calls instead of computing metrics."""
    seen = []

    def fake_evaluate(benchmark, approach, preds, labels, costs, **kwargs):
        seen.append({
            "benchmark": benchmark, "approach": approach,
            "queries": sorted(preds), "context": kwargs.get("telemetry_context"),
        })
        return pd.DataFrame({"precision": [1.0]})

    monkeypatch.setattr(rbp, "evaluate", fake_evaluate)
    return seen


def test_a_job_is_scored_once_its_labels_are_done(tmp_path, calls):
    h = _Harness(tmp_path)
    h.add("j-label-silver", "label", "silver", priority=0)
    h.add("j-approach", "approach", "optim_global", priority=1)
    h.finish("j-label-silver")
    h.finish("j-approach")

    assert JobScorer("t1", h.db).pass_once() == 1
    (call,) = calls
    assert call["approach"] == "optim_global"
    assert call["queries"] == ["q1", "q2"]
    # No worker_id: completing a job nulls `claimed_by` to release the lease, so the
    # machine that ran it is gone by the time the job is scorable.
    assert call["context"] == {"labels": "silver", "job_id": "j-approach"}


def test_a_job_that_beats_its_labels_waits_and_is_scored_later(tmp_path, calls):
    """The requirement: an approach job can finish before the (expensive) silver pass.
    Nothing is reported for it until real labels exist - and then it is, without
    needing the job to be re-run or the coordinator restarted."""
    h = _Harness(tmp_path)
    h.add("j-approach", "approach", "optim_global", priority=0)
    h.add("j-label-silver", "label", "silver", priority=1)
    h.finish("j-approach")

    scorer = JobScorer("t1", h.db)
    assert scorer.pass_once() == 0
    assert calls == []

    h.finish("j-label-silver")
    assert scorer.pass_once() == 1
    assert [c["context"]["labels"] for c in calls] == ["silver"]


def test_repeated_passes_do_not_rescore(tmp_path, calls):
    """Every pass sees the same finished jobs; a measurement must reach the
    append-only query_metrics list exactly once."""
    h = _Harness(tmp_path)
    h.add("j-label-silver", "label", "silver", priority=0)
    h.add("j-approach", "approach", "optim_global", priority=1)
    h.finish("j-label-silver")
    h.finish("j-approach")

    scorer = JobScorer("t1", h.db)
    assert scorer.pass_once() == 1
    assert scorer.pass_once() == 0
    assert scorer.pass_once() == 0
    assert len(calls) == 1


def test_gold_landing_after_silver_scores_only_gold(tmp_path, calls):
    """Silver is a full pass of the best model, gold reads the ground-truth files, so
    they finish far apart. The later one must not drag the earlier one along again."""
    h = _Harness(tmp_path)
    h.add("j-label-silver", "label", "silver", priority=0)
    h.add("j-approach", "approach", "optim_global", priority=1)
    h.add("j-label-gold", "label", "gold", priority=2)
    h.finish("j-label-silver")
    h.finish("j-approach")

    scorer = JobScorer("t1", h.db)
    scorer.pass_once()
    assert [c["context"]["labels"] for c in calls] == ["silver"]

    h.finish("j-label-gold")
    scorer.pass_once()
    assert [c["context"]["labels"] for c in calls] == ["silver", "gold"]


def test_labels_of_another_benchmark_do_not_score_this_job(tmp_path, calls):
    h = _Harness(tmp_path)
    h.add("j-label-other", "label", "silver", benchmark="bench_b", priority=0)
    h.add("j-approach", "approach", "optim_global", benchmark="bench_a", priority=1)
    h.finish("j-label-other")
    h.finish("j-approach")

    assert JobScorer("t1", h.db).pass_once() == 0
    assert calls == []


def test_a_failing_score_does_not_break_the_pass(tmp_path, monkeypatch):
    """This runs on the coordinator's main loop, which also drives lease sweeps and the
    final merge; a bad shard must cost one job's accuracy, not the sweep."""
    def boom(*a, **k):
        raise RuntimeError("bad shard")

    monkeypatch.setattr(rbp, "evaluate", boom)
    h = _Harness(tmp_path)
    h.add("j-label-silver", "label", "silver", priority=0)
    h.add("j-approach", "approach", "optim_global", priority=1)
    h.finish("j-label-silver")
    h.finish("j-approach")

    assert JobScorer("t1", h.db).pass_once() == 0


def test_only_the_queries_both_shards_share_are_scored(tmp_path, calls):
    """evaluate() iterates the labels and indexes predictions by query, so a query in
    one and not the other raises rather than being skipped. Score the overlap."""
    h = _Harness(tmp_path)
    h.add("j-label-silver", "label", "silver", queries=("q1", "q2", "q3"), priority=0)
    h.add("j-approach", "approach", "optim_global", queries=("q1", "q2"), priority=1)
    h.finish("j-label-silver")
    h.finish("j-approach")

    JobScorer("t1", h.db).pass_once()
    (call,) = calls
    assert call["queries"] == ["q1", "q2"]


def test_the_emitted_metric_carries_its_label_set_and_job(tmp_path):
    """End to end through the real evaluate() and a real Collector: the row the Run tab
    reads must say which job produced it and which labels scored it, or accuracy cannot
    be grouped by job and silver/gold are averaged together."""
    h = _Harness(tmp_path)
    h.add("j-label-silver", "label", "silver", priority=0)
    h.add("j-approach", "approach", "optim_global", priority=1)
    h.finish("j-label-silver")
    h.finish("j-approach")

    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        assert JobScorer("t1", h.db).pass_once() == 1
        _drain(c)
        rows = c.snapshot_aggregates()["query_metrics"]
        assert sorted(r["query"] for r in rows) == ["q1", "q2"]
        assert {r["labels"] for r in rows} == {"silver"}
        assert {r["job_id"] for r in rows} == {"j-approach"}
        assert {r["executor"] for r in rows} == {"optim_global"}
    finally:
        c.close()


def test_merge_does_not_re_emit_what_the_scorer_already_reported(tmp_path, monkeypatch):
    """The merge evaluates the same (query, guarantee) pairs a second time to write the
    CSVs. query_metrics is append-only, so if it also emitted, every query's accuracy
    would appear twice on the dashboard."""
    from reasondb.coordinator.merge import merge_task

    h = _Harness(tmp_path)
    h.add("j-label-silver", "label", "silver", priority=0)
    h.add("j-approach", "approach", "optim_global", priority=1)
    h.finish("j-label-silver")
    h.finish("j-approach")

    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        written = merge_task("t1", h.db)
        _drain(c)
        assert any(p.name == "silver_metrics.csv" for p in written["run_benchmark"])
        assert c.snapshot_aggregates()["query_metrics"] == []
    finally:
        c.close()


def _parameter_sweep_task(tmp_path, *, label_spec):
    """A finished parameter_sweep step job and its label job, as the producer writes them.

    ``label_spec`` is the label job's spec, naming the label set either as ``name`` or as
    ``label_set``; both must score.
    """
    from reasondb.coordinator.producers import parameter_sweep as srp

    db = JobDB(tmp_path / "coord.db")
    db.register_worker(Worker(worker_id="w1", task_id="t2", capability="both"))
    jobs = []
    for job_id, kind, spec, priority in (
        ("j-labels", "label", label_spec, 0),
        ("j-s0", "step", {"kind": "step", "step_idx": 0, "guarantee": [0.7, 0.7]}, 1),
    ):
        out_dir = tmp_path / f"job_{job_id}"
        job = Job(
            job_id=job_id, task_id="t2", producer=srp.PRODUCER_NAME,
            benchmark="bench_a", split="dev", spec=spec, required_capabilities=[],
            output_dir=str(out_dir), max_attempts=1, priority=priority,
        )
        if kind == "label":
            shards_mod.write_shard(job, {
                "kind": "label", "name": "silver",
                "results": {"q1": cached_answer(
                    out_dir / "cache", "q1", pd.DataFrame({"a": [1]})
                )},
            })
        else:
            shards_mod.write_shard(job, {
                "kind": "step", "name": "storage_step0_tunetrue_nNone",
                "label_set": "silver",
                "results": {"q1": {(0.7, 0.7): cached_answer(
                    out_dir / "cache", "q1", pd.DataFrame({"a": [1]})
                )}},
                "costs": {"q1": {(0.7, 0.7): _cost()}},
            })
        db.enqueue_job(job)
        jobs.append(job)
    for job in jobs:
        claimed = db.claim_next_job("w1", "both")
        db.start_job(claimed.job_id, "w1")
        db.complete_job(claimed.job_id, "w1", {})
    return db


def test_a_parameter_sweep_step_is_scored_from_its_shards(tmp_path, monkeypatch):
    """The storage sweep scores like run_benchmark: in the coordinator, off pickles.

    No worker collects labels (that would replay the shared label cache through a full
    executor loop once per job), so this path is what brings a finished step's accuracy
    to the dashboard.
    """
    from reasondb.coordinator.producers import parameter_sweep as srp

    seen = []
    monkeypatch.setattr(
        shards_mod, "evaluate",
        lambda **kw: seen.append(kw) or pd.DataFrame({"precision": [1.0]}),
    )
    db = _parameter_sweep_task(tmp_path, label_spec={"kind": "label", "name": "silver"})

    assert JobScorer("t2", db).pass_once() == 1
    (call,) = seen
    assert call["approach_name"] == "storage_step0_tunetrue_nNone"
    assert call["telemetry_context"] == {"labels": "silver", "job_id": "j-s0"}


def test_a_parameter_sweep_task_enqueued_before_scoring_existed_still_scores(
    tmp_path, monkeypatch
):
    """Label jobs that spell the label set ``label_set`` rather than ``name`` still
    score, rather than leaving every step waiting on labels that are done."""
    from reasondb.coordinator.producers import parameter_sweep as srp

    seen = []
    monkeypatch.setattr(
        shards_mod, "evaluate",
        lambda **kw: seen.append(kw) or pd.DataFrame({"precision": [1.0]}),
    )
    db = _parameter_sweep_task(tmp_path, label_spec={"kind": "label", "label_set": "silver"})

    assert JobScorer("t2", db).pass_once() == 1
    assert len(seen) == 1


def test_a_label_job_that_cannot_name_its_set_on_the_spec_is_read_from_its_shard(tmp_path):
    """`sample_size` label jobs carry no label name in their spec.

    `label_set_for` needs a benchmark *instance* to read `has_ground_truth`, and
    enumeration deliberately builds none (`count_queries` exists so counting jobs does
    not stand up a DuckDB). The job that ran it knows, and wrote it into its shard.
    """
    from reasondb.coordinator.producers import shards as shards_mod
    from reasondb.coordinator.scoring import _label_dirs_by_benchmark

    job = Job(
        job_id="j-labels", task_id="t3", producer="sample_size", benchmark="bench_a",
        split="dev", spec={"kind": "label"}, required_capabilities=[],
        output_dir=str(tmp_path / "job_labels"),
    )
    shards_mod.write_shard(job, {
        "kind": "label", "name": "gold",
        "results": {"q1": cached_answer(
            Path(job.output_dir) / "cache", "q1", pd.DataFrame({"a": [1]})
        )},
    })

    assert _label_dirs_by_benchmark([job]) == {"bench_a": [("gold", job.output_dir)]}


def test_a_restarted_scorer_does_not_rescore_what_the_previous_one_scored(tmp_path, calls):
    """The restart case: the memory of what has been scored lives in the job database.

    Held only in the scorer's own process, it would be lost on restart, and a resumed
    coordinator would re-score its whole task, re-emitting accuracy the previous run had
    already reported.
    """
    h = _Harness(tmp_path)
    h.add("j-label-silver", "label", "silver", priority=0)
    h.add("j-approach", "approach", "optim_global", priority=1)
    h.finish("j-label-silver")
    h.finish("j-approach")

    assert JobScorer("t1", h.db).pass_once() == 1
    assert len(calls) == 1

    # A brand new scorer over the same database - i.e. the coordinator restarted.
    assert JobScorer("t1", h.db).pass_once() == 0
    assert len(calls) == 1, "a restart must not re-score a job the task already scored"


def test_the_marker_is_written_per_job_and_label_set(tmp_path, calls):
    h = _Harness(tmp_path)
    h.add("j-label-silver", "label", "silver", priority=0)
    h.add("j-label-gold", "label", "gold", priority=1)
    h.add("j-approach", "approach", "optim_global", priority=2)
    for job_id in ("j-label-silver", "j-label-gold", "j-approach"):
        h.finish(job_id)

    JobScorer("t1", h.db).pass_once()
    assert sorted(h.db.list_scored("t1")) == [
        ("j-approach", "gold"),
        ("j-approach", "silver"),
    ]


def test_a_requeued_job_is_scored_again_once_it_has_re_run(tmp_path, calls):
    """Persisting the marker must not make a re-run job unscorable.

    Requeued the way an operator does it by hand - state back to pending, attempts
    reset - which is a path no coordinator code takes. It still has to be *claimed* to
    run again, and that is where the marker is cleared, so every route back to running
    is covered by the one rule.
    """
    h = _Harness(tmp_path)
    h.add("j-label-silver", "label", "silver", priority=0)
    h.add("j-approach", "approach", "optim_global", priority=1)
    h.finish("j-label-silver")
    h.finish("j-approach")
    JobScorer("t1", h.db).pass_once()
    assert len(calls) == 1

    h.db._conn.execute(
        "UPDATE jobs SET state = 'pending', attempt = 0 WHERE job_id = 'j-approach'"
    )
    assert h.db.list_scored("t1") == [("j-approach", "silver")]
    h.finish("j-approach")
    assert h.db.list_scored("t1") == [], "claiming a job un-scores it"

    assert JobScorer("t1", h.db).pass_once() == 1
    assert len(calls) == 2, "the retry's own results must be scored"
