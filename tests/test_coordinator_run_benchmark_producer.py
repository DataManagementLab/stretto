"""``coordinator.producers.run_benchmark`` wiring - same monkeypatched, GPU-free
approach as the other producer test files (see
``test_coordinator_parameter_sweep_producer.py``'s docstring).
"""

import argparse
import types

import pandas as pd
from pathlib import Path

from conftest import cached_answer
import pytest

from reasondb.coordinator.db import JobDB
from reasondb.coordinator.merge import merge_task
from reasondb.coordinator.models import JOB_DONE, Job, Worker, WorkerContext
from reasondb.coordinator.producers import run_benchmark as rbp
from reasondb.evaluation.benchmark import RandomBenchmark
from reasondb.coordinator.producers.run_benchmark import _build_executor as _real_build_executor


def _args(**overrides):
    base = dict(
        benchmarks=["fake_bench"],
        split="dev",
        use_indexes=False,
        human_labels=False,
        precision_guarantees=[0.7],
        recall_guarantees=[0.7],
        all_guarantee_combinations=False,
        cost_type="runtime",
        debug_query=None,
        simulate=None,
        precompute=None,
        select_executors=[],
        skip_executors=[],
    )
    base.update(overrides)
    return argparse.Namespace(**base)


#: Every producer's ``enumerate_jobs`` asks the benchmark how many queries a job will
#: run (``Benchmark.query_count``) and puts the answer on the job spec, so the monitor
#: can size a progress bar over the whole experiment. The fakes answer that too.
FAKE_QUERY_COUNT = 4


def _fake_benchmark(**attrs):
    return types.SimpleNamespace(
        query_count=lambda debug_query=None: FAKE_QUERY_COUNT, **attrs
    )


def _fake_benchmark_class(benchmark):
    """Enumeration asks the *class* for the count (it reads the pinned
    ``queries.json``), and loads without queries so it cannot generate them.

    A real class, not a SimpleNamespace: the producer branches on
    ``issubclass(cls, RandomBenchmark)`` to decide whether loading would generate a
    query set as a side effect.
    """

    class FakeBenchmark(RandomBenchmark):
        @classmethod
        def name(cls):
            return "fake_bench"

        @property
        def has_ground_truth(self):
            return benchmark.has_ground_truth

        @staticmethod
        def urls():
            return {}

        @staticmethod
        def download(split):
            return benchmark

        @classmethod
        def load(cls, split):
            return benchmark

        @classmethod
        def load_without_queries(cls, split):
            return benchmark

        @classmethod
        def count_queries(cls, split, debug_query=None):
            return FAKE_QUERY_COUNT

        @classmethod
        def _load_database(cls, split):
            return benchmark.database

        @classmethod
        def _get_query_shapes(cls):
            return []

        @classmethod
        def _get_operator_options(cls):
            return []

        @classmethod
        def _single_filter_shape(cls):
            return {}

    return FakeBenchmark


@pytest.fixture(autouse=True)
def _patch_deps(monkeypatch):
    fake_benchmark = _fake_benchmark(has_ground_truth=True, database=object())
    monkeypatch.setattr(rbp, "ALL_BENCHMARKS", {"fake_bench": _fake_benchmark_class(fake_benchmark)})
    monkeypatch.setattr(
        rbp, "enumerate_filter_stats_jobs", lambda task_id, out, args, producer: []
    )
    monkeypatch.setattr(
        rbp, "get_default_configurator", lambda use_indexes, use_human_labels: object()
    )
    monkeypatch.setattr(rbp, "get_label_configurator", lambda: object())
    monkeypatch.setattr(rbp, "build_reasoner", lambda configurator: object())

    class FakeExecutorCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(rbp, "_build_executor", lambda *a, **k: FakeExecutorCtx())


def test_enumerate_jobs_produces_approach_and_label_jobs(tmp_path):
    jobs = rbp.enumerate_jobs("t1", tmp_path, _args())
    kinds = [j.spec["kind"] for j in jobs]

    # 5 approaches x 1 guarantee + 2 labels (gold included: has_ground_truth=True)
    assert kinds.count("approach") == 5
    assert kinds.count("label") == 2
    assert {j.spec["name"] for j in jobs if j.spec["kind"] == "approach"} == set(rbp.EXECUTOR_NAMES)
    assert {j.spec["name"] for j in jobs if j.spec["kind"] == "label"} == {"silver", "gold"}


def test_enumerate_jobs_skips_gold_label_without_ground_truth(tmp_path, monkeypatch):
    fake_benchmark = _fake_benchmark(has_ground_truth=False)
    monkeypatch.setattr(rbp, "ALL_BENCHMARKS", {"fake_bench": _fake_benchmark_class(fake_benchmark)})

    jobs = rbp.enumerate_jobs("t1", tmp_path, _args())
    label_names = {j.spec["name"] for j in jobs if j.spec["kind"] == "label"}
    assert label_names == {"silver"}


def test_select_executors_narrows_the_approach_jobs(tmp_path):
    jobs = rbp.enumerate_jobs(
        "t1", tmp_path, _args(select_executors=["optim_global", "lotus"])
    )
    assert {j.spec["name"] for j in jobs if j.spec["kind"] == "approach"} == {
        "optim_global",
        "lotus",
    }
    # Labels are chosen by --labels, not by executor selection: an approach job
    # cannot be scored without them.
    assert {j.spec["name"] for j in jobs if j.spec["kind"] == "label"} == {
        "silver",
        "gold",
    }


def test_skip_executors_removes_only_those(tmp_path):
    jobs = rbp.enumerate_jobs("t1", tmp_path, _args(skip_executors=["abacus"]))
    names = {j.spec["name"] for j in jobs if j.spec["kind"] == "approach"}
    assert names == set(rbp.EXECUTOR_NAMES) - {"abacus"}


def test_unknown_executor_selection_is_rejected(tmp_path):
    """A typo must fail loudly, not silently enumerate zero approach jobs."""
    with pytest.raises(AssertionError, match="optim_gobal"):
        rbp.enumerate_jobs("t1", tmp_path, _args(select_executors=["optim_gobal"]))

    # A fixed-ratio configurator name is not one of the five and is rejected too.
    with pytest.raises(AssertionError, match="kv70B05"):
        rbp.enumerate_jobs("t1", tmp_path, _args(select_executors=["kv70B05"]))


def test_enumerate_jobs_multiplies_approach_jobs_by_guarantees(tmp_path):
    jobs = rbp.enumerate_jobs("t1", tmp_path, _args(precision_guarantees=[0.5, 0.9], recall_guarantees=[0.5, 0.9]))
    approach_jobs = [j for j in jobs if j.spec["kind"] == "approach"]
    assert len(approach_jobs) == 10  # 5 approaches x 2 guarantees


def test_run_job_approach_writes_pickled_shard(tmp_path, monkeypatch):
    def fake_collect_all(**kwargs):
        # collect_* return where each answer was cached, not the answer itself.
        return (
            {"q1": {(0.7, 0.7): cached_answer(
                Path(kwargs["out_dir"]) / "cache", "q1", pd.DataFrame({"a": [1]})
            )}},
            {"q1": {(0.7, 0.7): ["[]"]}},  # a "track" is a list of JSON-string sections
            {"q1": {(0.7, 0.7): object()}},
        )

    monkeypatch.setattr(rbp, "collect_results_all_guarantees", fake_collect_all)

    job = Job(
        job_id="t1-run_benchmark-fake_bench-optim_global-p0.7-r0.7",
        task_id="t1", producer="run_benchmark", benchmark="fake_bench", split="dev",
        spec={"kind": "approach", "name": "optim_global", "guarantee": [0.7, 0.7],
              "simulate": False, "simulate_paths": None, "use_indexes": False,
              "cost_type": "runtime", "debug_query": None, "n_queries": FAKE_QUERY_COUNT},
        required_capabilities=["embedding", "text_kv", "image_kv"],
        output_dir=str(tmp_path / "job_0"),
    )
    result = rbp.run_job(job, WorkerContext(device="cpu", worker_id="w1", capability="both"))

    assert result.success is True, result.error
    shard = tmp_path / "job_0" / "shard.pkl"
    assert shard.is_file()

    import pickle
    with open(shard, "rb") as f:
        data = pickle.load(f)
    assert data["benchmark"] == "fake_bench"
    assert data["kind"] == "approach"
    assert data["name"] == "optim_global"


def test_run_job_label_uses_collect_result_no_guarantees(tmp_path, monkeypatch):
    calls = []

    def fake_collect_no_guarantees(**kwargs):
        calls.append(kwargs["executor_name"])
        answer = cached_answer(
            Path(kwargs["out_dir"]) / "cache", "q1", pd.DataFrame({"a": [1]})
        )
        return {"q1": answer}, {"q1": ["[]"]}, {"q1": object()}

    monkeypatch.setattr(rbp, "collect_result_no_guarantees", fake_collect_no_guarantees)

    job = Job(
        job_id="t1-run_benchmark-fake_bench-label-silver",
        task_id="t1", producer="run_benchmark", benchmark="fake_bench", split="dev",
        spec={"kind": "label", "name": "silver", "guarantee": None, "simulate": False,
              "simulate_paths": None, "use_indexes": False, "cost_type": "runtime", "debug_query": None, "n_queries": FAKE_QUERY_COUNT},
        required_capabilities=["embedding", "text_kv", "image_kv"],
        output_dir=str(tmp_path / "job_label"),
    )
    result = rbp.run_job(job, WorkerContext(device="cpu", worker_id="w1", capability="both"))

    assert result.success is True, result.error
    assert calls == ["silver"]


def test_run_job_rejects_unknown_kind(tmp_path):
    job = Job(
        job_id="j1", task_id="t1", producer="run_benchmark", benchmark="fake_bench", split="dev",
        spec={"kind": "bogus", "name": "optim_global", "use_indexes": False, "cost_type": "runtime"},
        required_capabilities=[], output_dir=str(tmp_path / "job_0"),
    )
    result = rbp.run_job(job, WorkerContext(device="cpu", worker_id="w1", capability="both"))
    assert result.success is False
    assert "kind" in result.error.lower() or "bogus" in result.error


def test_build_executor_rejects_unknown_name():
    # _real_build_executor was imported before the autouse fixture patches
    # rbp._build_executor (a module-attribute patch, which doesn't affect an
    # already-bound name imported separately) - this calls the genuine function.
    with pytest.raises(AssertionError):
        _real_build_executor(
            "not-a-real-executor", database=None, reasoner=None,
            default_configurator=None, label_configurator=None,
            cost_type=None, device="cpu", logger_=None,
        )


def test_merge_groups_by_benchmark_and_writes_metrics(tmp_path, monkeypatch):
    # **kwargs, not a fixed positional signature: merge also passes
    # record_telemetry=False, because coordinator.scoring already reported these rows
    # per job. See tests/test_coordinator_scoring.py for that half.
    monkeypatch.setattr(
        rbp, "evaluate",
        lambda benchmark_name, approach, preds, labels, costs, **kwargs: pd.DataFrame(
            {"precision": [1.0]}
        ),
    )

    db = JobDB(tmp_path / "coord.db")
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))

    def _make_job(job_id, benchmark, kind, name, guarantee=None):
        out_dir = tmp_path / "t1" / f"job_{job_id}"
        out_dir.mkdir(parents=True)
        # The manifest is relative to the task dir and the files have to exist: the
        # merge reads each answer out of the cache the shard points at.
        cached_answer(out_dir / "cache", "q1", pd.DataFrame({"a": [1]}))
        import pickle
        shard = {
            "benchmark": benchmark, "split": "dev", "kind": kind, "name": name,
            "results": (
                {"q1": {(0.7, 0.7): f"job_{job_id}/cache/q1.sig.pkl"}}
                if kind == "approach"
                else {"q1": f"job_{job_id}/cache/q1.sig.pkl"}
            ),
            "pipeline_tracks": ({"q1": {(0.7, 0.7): ["[]"]}} if kind == "approach" else {"q1": ["[]"]}),
            "costs": ({"q1": {(0.7, 0.7): object()}} if kind == "approach" else {"q1": object()}),
        }
        with open(out_dir / "shard.pkl", "wb") as f:
            pickle.dump(shard, f)
        job = Job(
            job_id=job_id, task_id="t1", producer="run_benchmark", benchmark=benchmark, split="dev",
            spec={"kind": kind, "name": name, "guarantee": guarantee},
            required_capabilities=[], output_dir=str(out_dir), max_attempts=1,
        )
        db.enqueue_job(job)
        claimed = db.claim_next_job("w1", "both")
        db.start_job(claimed.job_id, "w1")
        db.complete_job(claimed.job_id, "w1", {})

    _make_job("j-label-silver", "bench_a", "label", "silver")
    _make_job("j-approach", "bench_a", "approach", "optim_global", guarantee=[0.7, 0.7])

    assert {j.state for j in db.list_jobs("t1")} == {JOB_DONE}
    written = merge_task("t1", db)

    assert "run_benchmark" in written
    silver_path = next(p for p in written["run_benchmark"] if p.name == "silver_metrics.csv")
    assert silver_path.is_file()
    yaml_path = next(p for p in written["run_benchmark"] if p.name == "pipeline_tracks.yaml")
    assert yaml_path.is_file()


def test_enumerate_jobs_stamp_every_spec_with_an_exact_query_count(tmp_path):
    """The monitor sums this over the whole queue to draw one progress bar across the
    experiment (see ``JobDB.job_progress``); a job without it degrades that bar to
    counting jobs instead of queries."""
    jobs = rbp.enumerate_jobs("t1", tmp_path, _args())
    assert jobs
    assert all(j.spec["n_queries"] == FAKE_QUERY_COUNT for j in jobs)
