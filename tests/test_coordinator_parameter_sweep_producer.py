"""``coordinator.producers.parameter_sweep`` wiring, and ``coordinator.merge``'s
dispatch - exercised with ``prepare_sweep``/``run_state`` monkeypatched to fake,
GPU/server-free stand-ins.

A genuine end-to-end run needs real KV caches on disk, real embedding servers and a
real benchmark. What's tested here instead: that ``enumerate_jobs`` turns a sweep plan
into the right job specs (capabilities, output dirs, priorities), that ``run_job``
calls through to ``prepare_sweep``/``run_state`` with a job-exclusive ``output_dir``
and writes its own shard rather than a shared CSV, and that ``merge_task`` folds only
``done`` jobs' shards together, grouped by benchmark - the two properties that keep
concurrent jobs from racing on shared outputs (see the module docstrings in
``producers/parameter_sweep.py`` and ``executor.py``).
"""

import argparse
import json
import pickle
import types
from pathlib import Path

import pandas as pd
import pytest

from conftest import cached_answer, fake_slot_map

from reasondb.coordinator.db import JobDB
from reasondb.coordinator.merge import merge_task
from reasondb.coordinator.models import JOB_DONE, JOB_FAILED, Job, Worker, WorkerContext
from reasondb.coordinator.producers import parameter_sweep as srp
from reasondb.coordinator.producers import shards as shards_mod


def _args(**overrides):
    base = dict(
        benchmarks=["fake_bench"],
        split="dev",
        use_indexes=False,
        text_small_model="small-text-model",
        text_large_model="large-text-model",
        image_small_model="small-image-model",
        image_large_model="large-image-model",
        press_name="expected_attention",
        precision_guarantees=[0.7],
        recall_guarantees=[0.7],
        all_guarantee_combinations=False,
        cost_type="runtime",
        debug_query=None,
        device="cpu",
        simulate=None,
        precompute=None,
        output_dir=None,
        tune_parameters=None,
        sample_sizes=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


#: Every producer's ``enumerate_jobs`` asks the benchmark how many queries a job will
#: run (``Benchmark.query_count``) and puts the answer on the job spec, so the monitor
#: can size a progress bar over the whole experiment. The fakes answer that too.
FAKE_QUERY_COUNT = 4


def _fake_database(text=True, image=False, audio=False):
    """Only what ``capabilities_for_benchmark`` reads: which modalities the benchmark's
    columns declare. A text-only fake, so a job over it must not demand an image model."""
    return types.SimpleNamespace(
        external_tables=[
            types.SimpleNamespace(
                name="t",
                text_columns=["t.body"] if text else [],
                image_columns=["t.pic"] if image else [],
                audio_columns=["t.clip"] if audio else [],
            )
        ]
    )


def _fake_benchmark(**attrs):
    attrs.setdefault("has_ground_truth", False)
    attrs.setdefault("name", lambda: "fake_bench")
    # The precompute path builds a real Executor around benchmark.database; every
    # other test fakes run_state/run_point before it gets that far.
    attrs.setdefault("database", _fake_database())
    return types.SimpleNamespace(
        query_count=lambda debug_query=None: FAKE_QUERY_COUNT, **attrs
    )


def _fake_benchmark_class(benchmark=None):
    return types.SimpleNamespace(
        name=lambda: "fake_bench",
        load=lambda split: benchmark or _fake_benchmark(),
        load_without_queries=lambda split: benchmark or _fake_benchmark(),
        count_queries=lambda split, debug_query=None: FAKE_QUERY_COUNT,
    )


def _steps(jobs):
    """The sweep-point jobs, i.e. everything but the benchmark's label job."""
    return [j for j in jobs if j.spec.get("kind") not in ("label", "filter_stats")]


def _fake_states():
    """Two steps: one text-only, one text+image - exercises the capability inference."""
    return [
        ({"text_small": [0.0]}, 1000),
        ({"text_small": [0.5], "image_small": [0.0]}, 500),
    ]


@pytest.fixture(autouse=True)
def _patch_prepare_sweep(monkeypatch):
    def fake_prepare_sweep(benchmark, args):
        states = _fake_states()
        return srp.psweep.SweepPrep(states, {}, {}, [], fake_slot_map(states), {})

    monkeypatch.setattr(srp.psweep, "prepare_sweep", fake_prepare_sweep)
    monkeypatch.setattr(srp, "BENCHMARKS", {"fake_bench": _fake_benchmark_class()})
    # The phase-0 job has its own tests (test_coordinator_filter_stats_jobs.py); a fake
    # benchmark is not a RandomBenchmark, so it would emit nothing here regardless.
    monkeypatch.setattr(
        srp, "enumerate_filter_stats_jobs", lambda task_id, out, args, producer: []
    )


def test_enumerate_jobs_produces_one_job_per_step_and_guarantee(tmp_path):
    args = _args(precision_guarantees=[0.5, 0.9], recall_guarantees=[0.5, 0.9])
    jobs = srp.enumerate_jobs("t1", tmp_path, args)

    steps = _steps(jobs)
    assert len(steps) == 4  # 2 steps x 2 guarantees
    assert {j.spec["step_idx"] for j in steps} == {0, 1}
    assert {tuple(j.spec["guarantee"]) for j in steps} == {(0.5, 0.5), (0.9, 0.9)}

    # Exactly one labelling pass per benchmark, handed out before any sweep point.
    (label_job,) = [j for j in jobs if j.spec.get("kind") == "label"]
    assert label_job.priority == -1
    assert all(j.priority >= 0 for j in steps)
    assert label_job.spec["label_set"] == "silver"  # the fake has no ground truth


def test_enumerate_jobs_infers_required_capabilities_from_active_slots(tmp_path):
    jobs = srp.enumerate_jobs("t1", tmp_path, _args())
    by_step = {j.spec["step_idx"]: j for j in _steps(jobs)}

    assert set(by_step[0].required_capabilities) == {"embedding", "text_kv"}
    assert set(by_step[1].required_capabilities) == {"embedding", "text_kv", "image_kv"}

    # The labeler ignores the sweep's storage states (it runs the vanilla operators), so
    # it needs the union over every step - not just the modalities step 0 materializes.
    (label_job,) = [j for j in jobs if j.spec.get("kind") == "label"]
    assert set(label_job.required_capabilities) == {"embedding", "text_kv", "image_kv"}


def test_enumerate_jobs_gives_each_job_its_own_output_dir(tmp_path):
    jobs = srp.enumerate_jobs("t1", tmp_path, _args())
    dirs = [j.output_dir for j in jobs]
    assert len(set(dirs)) == len(dirs)  # no collisions
    assert all(str(tmp_path) in d and "t1" in d for d in dirs)


def test_run_job_writes_its_own_shard_not_a_shared_file(tmp_path, monkeypatch):
    captured = {}

    def fake_run_state(**kwargs):
        captured.update(kwargs)
        rows = [
            {"benchmark": "fake_bench", "step": kwargs["step_idx"], "query": "q1",
             "precision_guarantee": 0.7, "recall_guarantee": 0.7, "storage_gb": 1.0},
        ]
        # run_state returns where each answer was cached, not the answer.
        answer = cached_answer(
            Path(kwargs["output_dir"]) / "cache", "q1", pd.DataFrame({"a": [1]})
        )
        return rows, {"q1": {(0.7, 0.7): answer}}, {"q1": {(0.7, 0.7): object()}}

    monkeypatch.setattr(srp.psweep, "run_state", fake_run_state)

    job = Job(
        job_id="t1-parameter_sweep-fake_bench-s0-p0.7-r0.7",
        task_id="t1",
        producer="parameter_sweep",
        benchmark="fake_bench",
        split="dev",
        spec={
            "step_idx": 0,
            "guarantee": [0.7, 0.7],
            "tune_parameters": True,
            "sample_size": None,
            "sweep_to_gold": False,
            "simulate": False,
            "simulate_paths": None,
            "use_indexes": False,
            "cost_type": "runtime",
            "press_name": "expected_attention",
            "text_small_model": "small-text-model",
            "text_large_model": "large-text-model",
            "image_small_model": "small-image-model",
            "image_large_model": "large-image-model",
            "debug_query": None,
            "n_queries": FAKE_QUERY_COUNT,
        },
        required_capabilities=["embedding", "text_kv"],
        output_dir=str(tmp_path / "job_0"),
    )
    worker = WorkerContext(device="cpu", worker_id="w1", capability="both")

    result = srp.run_job(job, worker)

    assert result.success is True
    assert result.result_summary["n_rows"] == 1
    # run_state got THIS job's own output_dir, not a shared one.
    assert captured["output_dir"] == tmp_path / "job_0"
    # It is told which labeler will score it - that is a column on every row - but it is
    # not handed labels, and does not collect any: scoring happens off the shards.
    assert captured["label_set"] == "silver"
    assert "labels" not in captured
    rows_path = tmp_path / "job_0" / "rows.parquet"
    assert rows_path.is_file()
    assert len(pd.read_parquet(rows_path)) == 1
    # ...and the predictions/costs scoring needs, beside them.
    with open(tmp_path / "job_0" / "shard.pkl", "rb") as f:
        shard = pickle.load(f)
    assert (shard["kind"], shard["benchmark"], shard["label_set"]) == (
        "step", "fake_bench", "silver",
    )
    assert list(shard["results"]) == ["q1"]


def test_run_job_out_of_range_step_fails_gracefully(monkeypatch, tmp_path):
    job = Job(
        job_id="j1", task_id="t1", producer="parameter_sweep", benchmark="fake_bench",
        split="dev",
        spec={
            "step_idx": 99, "guarantee": [0.7, 0.7], "simulate": False, "simulate_paths": None,
            "tune_parameters": True, "sample_size": None, "sweep_to_gold": False,
            "use_indexes": False, "cost_type": "runtime", "press_name": "expected_attention",
            "text_small_model": "s", "text_large_model": "l", "image_small_model": "s",
            "image_large_model": "l", "debug_query": None,
            "n_queries": FAKE_QUERY_COUNT,
        },
        required_capabilities=["embedding"], output_dir=str(tmp_path / "job_99"),
    )
    result = srp.run_job(job, WorkerContext(device="cpu", worker_id="w1", capability="both"))
    assert result.success is False
    assert "out of range" in result.error


def test_merge_task_only_folds_in_done_jobs(tmp_path):
    db = JobDB(tmp_path / "coord.db")
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))

    def _make_job(job_id, benchmark, step, will_succeed):
        out_dir = tmp_path / "t1" / f"job_{job_id}"
        out_dir.mkdir(parents=True)
        job = Job(
            job_id=job_id, task_id="t1", producer="parameter_sweep", benchmark=benchmark,
            split="dev",
            spec={"step_idx": step, "guarantee": [0.7, 0.7], "tune_parameters": True, "sample_size": None, "sweep_to_gold": False,},
            required_capabilities=["embedding"], output_dir=str(out_dir), max_attempts=1,
        )
        db.enqueue_job(job)
        claimed = db.claim_next_job("w1", "both")
        db.start_job(claimed.job_id, "w1")
        if will_succeed:
            pd.DataFrame(
                [{"benchmark": benchmark, "split": "dev", "step": step, "query": "q",
                  "precision_guarantee": 0.7, "recall_guarantee": 0.7}]
            ).to_parquet(out_dir / "rows.parquet", index=False)
            db.complete_job(claimed.job_id, "w1", {"n_rows": 1})
        else:
            db.fail_job(claimed.job_id, "w1", "boom")  # max_attempts=1 -> straight to failed
        return job_id

    _make_job("j-done-0", "bench_a", 0, will_succeed=True)
    _make_job("j-done-1", "bench_a", 1, will_succeed=True)
    _make_job("j-failed", "bench_a", 2, will_succeed=False)

    assert {j.state for j in db.list_jobs("t1")} == {JOB_DONE, JOB_FAILED}

    written = merge_task("t1", db)
    assert "parameter_sweep" in written
    merged_csv = next(p for p in written["parameter_sweep"] if p.suffix == ".csv")
    df = pd.read_csv(merged_csv)
    assert len(df) == 2  # only the two done jobs' rows
    assert set(df["step"]) == {0, 1}


def test_enumerate_jobs_stamp_every_spec_with_an_exact_query_count(tmp_path):
    """See the identically-named test in test_coordinator_run_benchmark_producer.py."""
    jobs = srp.enumerate_jobs("t1", tmp_path, _args())
    assert jobs
    assert all(j.spec["n_queries"] == FAKE_QUERY_COUNT for j in jobs)


def test_label_job_fills_the_shared_cache_and_writes_no_shard(tmp_path, monkeypatch):
    """A label job contributes labels, not sweep rows. It must write no rows.parquet -
    merge() concatenates whatever shards it finds, and an empty/absent one there would
    either break the concat or inject a bogus row into the merged CSV."""
    calls = []

    def fake_collect_labels(benchmark, out_dir, label_set, **kwargs):
        calls.append((Path(out_dir), label_set))
        # `collect_labels` returns where each answer was cached, not the answers.
        return {"q1": cached_answer(Path(out_dir) / f"cache_{label_set}",
                                    "q1", pd.DataFrame({"a": [1]}))}

    monkeypatch.setattr(srp, "collect_labels", fake_collect_labels)

    job = Job(
        job_id="t1-storage_runtime-fake_bench-labels", task_id="t1",
        producer="parameter_sweep", benchmark="fake_bench", split="dev",
        spec={"kind": "label", "label_set": "silver", "guarantee": None,
              "tune_parameters": True, "sample_size": None, "sweep_to_gold": False,
              "simulate": False, "simulate_paths": [], "use_indexes": False,
              "cost_type": "runtime", "press_name": "expected_attention",
              "text_small_model": "s", "text_large_model": "l",
              "image_small_model": "s", "image_large_model": "l", "debug_query": None, "n_queries": FAKE_QUERY_COUNT},
        required_capabilities=["embedding", "text_kv"],
        output_dir=str(tmp_path / "sweep" / "job_labels"),
    )
    result = srp.run_job(job, WorkerContext(device="cpu", worker_id="w1", capability="both"))

    assert result.success is True, result.error
    assert result.result_summary["label_set"] == "silver"
    assert not (tmp_path / "sweep" / "job_labels" / "rows.parquet").exists()
    # Beside the job dirs, not inside this one: every step job of the benchmark reads
    # the same cache, and it has to outlive whichever job filled it.
    (cache_dir, _), = calls
    assert cache_dir == tmp_path / "sweep" / "label_cache" / "fake_bench"


def test_only_the_label_job_collects_labels(tmp_path, monkeypatch):
    """A step job runs its own queries and nothing else.

    Collecting labels in a step job would replay every query through a full
    ``execute_benchmark`` loop once per step job. Scoring happens in
    ``score_job``/``merge``, which read the label shard directly; nothing in a worker
    collects labels but the label job.
    """
    dirs = []
    monkeypatch.setattr(
        srp, "collect_labels",
        lambda benchmark, out_dir, label_set, **kw: (
            dirs.append(Path(out_dir)) or {"q1": pd.DataFrame({"a": [1]})}
        ),
    )
    monkeypatch.setattr(
        srp.psweep, "run_state",
        lambda **kw: ([], {}, {}),
    )

    common = dict(
        task_id="t1", producer="parameter_sweep", benchmark="fake_bench", split="dev",
        required_capabilities=["embedding"],
    )
    base_spec = {
        "tune_parameters": True, "sample_size": None, "sweep_to_gold": False,
        "simulate": False, "simulate_paths": [], "use_indexes": False,
        "cost_type": "runtime", "press_name": "expected_attention",
        "text_small_model": "s", "text_large_model": "l", "image_small_model": "s",
        "image_large_model": "l", "debug_query": None, "n_queries": FAKE_QUERY_COUNT,
    }
    worker = WorkerContext(device="cpu", worker_id="w1", capability="both")
    srp.run_job(Job(
        job_id="t1-labels", spec={"kind": "label", "label_set": "silver", **base_spec},
        output_dir=str(tmp_path / "sweep" / "job_labels"), **common,
    ), worker)
    srp.run_job(Job(
        job_id="t1-s0", spec={"kind": "step", "step_idx": 0, "guarantee": [0.7, 0.7], **base_spec},
        output_dir=str(tmp_path / "sweep" / "job_s0"), **common,
    ), worker)

    assert dirs == [tmp_path / "sweep" / "label_cache" / "fake_bench"], (
        "the step job must not collect labels; only the label job does"
    )


# ── The optimizer axes (tune_parameters x sample_size) ───────────────────────


def test_job_ids_name_every_axis_including_at_its_default(tmp_path):
    """A job id says what the point is without the reader knowing the defaults."""
    jobs = srp.enumerate_jobs("t1", tmp_path, _args())
    step_ids = [j.job_id for j in _steps(jobs)]

    assert step_ids == [
        f"t1-{srp.PRODUCER_NAME}-fake_bench-s{i}-p0.7-r0.7-optim_global-tunetrue-nNone"
        for i in range(len(step_ids))
    ]


def test_unswept_axes_still_land_on_the_spec(tmp_path):
    for job in _steps(srp.enumerate_jobs("t1", tmp_path, _args())):
        assert job.spec["tune_parameters"] is True
        assert job.spec["sample_size"] is None


def test_axes_multiply_the_sweep_and_tag_the_job_ids(tmp_path):
    jobs = srp.enumerate_jobs(
        "t1",
        tmp_path,
        _args(tune_parameters=["true", "false"], sample_sizes=[30, 150]),
    )
    steps = _steps(jobs)
    n_states = len({j.spec["step_idx"] for j in steps})

    assert len(steps) == n_states * 2 * 2  # states x tune x sample_size
    assert {(j.spec["tune_parameters"], j.spec["sample_size"]) for j in steps} == {
        (True, 30), (True, 150), (False, 30), (False, 150),
    }
    # Every combination is named in full, and every id stays unique.
    assert len({j.job_id for j in steps}) == len(steps)
    assert any(j.job_id.endswith("-tunefalse-n30") for j in steps)


def test_priority_still_follows_the_storage_step_only(tmp_path):
    """The optimizer axes must not disturb the expensive->cheap walk."""
    steps = _steps(
        srp.enumerate_jobs(
            "t1", tmp_path, _args(tune_parameters=["true", "false"], sample_sizes=[30])
        )
    )
    for job in steps:
        assert job.priority == job.spec["step_idx"]


def test_bad_tune_parameters_value_is_rejected(tmp_path):
    with pytest.raises(AssertionError, match="true/false"):
        srp.enumerate_jobs("t1", tmp_path, _args(tune_parameters=["maybe"]))


def test_bad_sample_size_is_rejected(tmp_path):
    with pytest.raises(AssertionError, match="positive integers"):
        srp.enumerate_jobs("t1", tmp_path, _args(sample_sizes=[0]))


def test_repeated_axis_values_are_deduplicated(tmp_path):
    """`--tune-parameters true true` is a typo, not a request to run twice."""
    steps = _steps(
        srp.enumerate_jobs(
            "t1", tmp_path, _args(tune_parameters=["true", "true"], sample_sizes=[30, 30])
        )
    )
    assert len({j.job_id for j in steps}) == len(steps)
    assert {(j.spec["tune_parameters"], j.spec["sample_size"]) for j in steps} == {(True, 30)}


def test_run_job_rejects_a_spec_missing_an_axis(tmp_path, monkeypatch):
    """A required axis key is not defaulted - a missing one is a bug, not a legacy case.

    Defaulting here would silently run the point at some other axis value and name its
    results cache accordingly, so the wrong cached results would be replayed.
    """
    monkeypatch.setattr(srp, "collect_labels", lambda *a, **k: {})

    spec = {
        "kind": "step",
        "step_idx": 0,
        "guarantee": [0.7, 0.7],
        "tune_parameters": True,
        "sample_size": None,
        "sweep_to_gold": False,
        "simulate": False,
        "simulate_paths": None,
        "use_indexes": False,
        "cost_type": "runtime",
        "press_name": "expected_attention",
        "text_small_model": "s",
        "text_large_model": "l",
        "image_small_model": "s",
        "image_large_model": "l",
        "debug_query": None,
        "n_queries": FAKE_QUERY_COUNT,
    }
    for missing in ("tune_parameters", "sample_size", "sweep_to_gold", "simulate_paths"):
        partial = {k: v for k, v in spec.items() if k != missing}
        job = Job(
            job_id=f"j-no-{missing}", task_id="t1", producer=srp.PRODUCER_NAME,
            benchmark="fake_bench", split="dev", spec=partial,
            required_capabilities=["embedding"], output_dir=str(tmp_path / missing),
        )
        result = srp.run_job(
            job, WorkerContext(device="cpu", worker_id="w1", capability="both")
        )
        assert result.success is False, f"{missing} was silently defaulted"
        assert missing in result.error


def test_the_state_plan_travels_on_every_spec(tmp_path):
    """The worker re-runs `prepare_sweep` and indexes the result by `step_idx`, so it has
    to plan the same states the coordinator did - and the plan name (`--state-plan`) is
    what says which.
    """
    jobs = srp.enumerate_jobs("t1", tmp_path, _args(state_plan="full"))

    assert {j.spec["state_plan"] for j in jobs} == {"full"}


def test_the_state_plan_is_not_part_of_a_job_id(tmp_path):
    """A task runs one plan, so the id does not carry it - and must not start to without
    a migration: the id is the results-cache key a resumed task looks itself up by."""
    plain = [j.job_id for j in _steps(srp.enumerate_jobs("t1", tmp_path, _args()))]
    named = [
        j.job_id
        for j in _steps(srp.enumerate_jobs("t1", tmp_path, _args(state_plan="greedy")))
    ]

    assert plain == named


def test_point_name_separates_the_results_cache_per_axis_combination():
    """The cache key is only (query, guarantees); the directory name is the rest.

    Two sweep points sharing a name would replay each other's cached results.
    """
    from reasondb.evaluation.parameter_sweep import step_point_name

    names = {
        step_point_name(0, approach, tune, n)
        for approach in ("optim_global", "lotus")
        for tune in (True, False)
        for n in (None, 30, 150)
    }
    assert len(names) == 12
    # Every axis is named, so the string is readable without knowing the defaults.
    assert step_point_name(3, "optim_global", True, None) == "step3_optim_global_tunetrue_nNone"
    assert step_point_name(3, "lotus", False, 30) == "step3_lotus_tunefalse_n30"


def test_merged_layout_matches_the_plot_script_glob(tmp_path):
    """merge() must write where scripts/plot_sweep.py actually looks.

    That script globs ``<output-dir>/*/<split>/<producer>.csv``; without the
    ``<split>`` level no ``--output-dirs`` value could match a merged sweep.
    """
    task_root = tmp_path / "t1"
    job_dir = task_root / "job_0"
    job_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"benchmark": "fake_bench", "split": "dev", "step": 0, "query": "q",
          "precision_guarantee": 0.7, "recall_guarantee": 0.7, "storage_gb": 1.0}]
    ).to_parquet(job_dir / "rows.parquet", index=False)

    written = srp.merge("t1", [str(job_dir)])

    merged_root = task_root / "merged"
    assert sorted(merged_root.glob(f"*/dev/{srp.PRODUCER_NAME}.csv")) == [
        merged_root / "fake_bench" / "dev" / f"{srp.PRODUCER_NAME}.csv"
    ]
    assert merged_root / "fake_bench" / "dev" / f"{srp.PRODUCER_NAME}.csv" in written


def test_merge_keeps_the_axes_in_the_output(tmp_path):
    """The two optimizer axes must survive into the merged CSV, or the sweep is unreadable."""
    task_root = tmp_path / "t1"
    rows = []
    for tune in (True, False):
        for n in (30, 150):
            rows.append({
                "benchmark": "fake_bench", "split": "dev", "step": 0,
                "tune_parameters": tune, "sample_size": n, "query": "q",
                "precision_guarantee": 0.7, "recall_guarantee": 0.7, "storage_gb": 1.0,
            })
    job_dir = task_root / "job_0"
    job_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(job_dir / "rows.parquet", index=False)

    srp.merge("t1", [str(job_dir)])

    out = pd.read_csv(task_root / "merged" / "fake_bench" / "dev" / f"{srp.PRODUCER_NAME}.csv")
    assert set(out["tune_parameters"]) == {True, False}
    assert set(out["sample_size"]) == {30, 150}


# ── --precompute as a job kind ───────────────────────────────────────────────


def _precompute_args(**overrides):
    base = dict(precompute={"fake_bench": Path("out.json")})
    base.update(overrides)
    return _args(**base)


def test_precompute_mode_enumerates_only_precompute_jobs(tmp_path):
    """A precompute task must not also enqueue sweep points.

    A sweep job's spec has to carry its --simulate path at enumeration time, and that
    file does not exist until this task has recorded it - hence two separate tasks.
    """
    jobs = srp.enumerate_jobs("t1", tmp_path, _precompute_args())
    assert {j.spec["kind"] for j in jobs} == {"precompute"}


def test_precompute_is_one_job_per_benchmark(tmp_path):
    """Not split by query range, deliberately.

    Precomputed work is keyed by (operator, expression, base tables) rather than by
    query, so one pass over a benchmark already skips every expression it has seen.
    Separate shards get separate stores and cannot reuse each other's work, so the
    overlap would be recomputed - with real model calls - once per shard.
    """
    jobs = srp.enumerate_jobs("t1", tmp_path, _precompute_args())
    assert len(jobs) == 1
    (job,) = jobs
    assert job.spec["n_queries"] == FAKE_QUERY_COUNT
    assert job.benchmark == "fake_bench"


def test_precompute_writes_the_file_the_operator_named(tmp_path):
    """Read as well as written: an existing file supplies the resume markers and the
    pinned operator configs, so a re-run skips what it did and re-asks the same
    questions."""
    (job,) = srp.enumerate_jobs(
        "t1", tmp_path, _precompute_args(precompute={"fake_bench": tmp_path / "mine.json"})
    )
    assert job.spec["precompute_path"] == str(tmp_path / "mine.json")


def test_one_dataset_cannot_be_precomputed_twice(tmp_path):
    """SimulateStore rewrites its whole file per query, so two jobs on one path is a
    lost-update race. A mapping makes that unrepresentable - a repeated benchmark is
    one key."""
    (job,) = srp.enumerate_jobs(
        "t1", tmp_path, _precompute_args(benchmarks=["fake_bench", "fake_bench"])
    )
    assert job.spec["precompute_path"] == "out.json"


def test_precompute_jobs_are_never_treated_as_simulate(tmp_path):
    """scheduler.worker_can_run short-circuits on a truthy spec['simulate'].

    A simulate-capability worker only starts the embedding servers, so letting it
    claim a precompute job would mean recording responses with no model behind them.
    """
    from reasondb.coordinator.scheduler import worker_can_run

    (job,) = srp.enumerate_jobs("t1", tmp_path, _precompute_args())
    assert job.spec["simulate"] is False
    assert not worker_can_run("simulate", job.required_capabilities, job_is_simulate=False)
    assert worker_can_run("both", job.required_capabilities, job_is_simulate=False)


def test_precompute_asks_only_for_the_modalities_the_benchmark_uses(tmp_path):
    """A text-only dataset must not demand an image worker.

    Asking for embedding + text_kv + image_kv on every job would let a text-only
    recording be claimed only by a worker that also holds an idle image model.
    """
    from reasondb.coordinator.scheduler import worker_can_run

    (job,) = srp.enumerate_jobs("t1", tmp_path, _precompute_args())
    assert set(job.required_capabilities) == {"embedding", "text_kv"}
    assert worker_can_run("text", job.required_capabilities, job_is_simulate=False)
    assert not worker_can_run("image", job.required_capabilities, job_is_simulate=False)


def _use_mixed_modality_benchmark(monkeypatch):
    """A fake dataset with both a text and an image column - ecommerce's shape, and the
    only one the split flag does anything to."""
    monkeypatch.setattr(
        srp,
        "BENCHMARKS",
        {
            "fake_bench": _fake_benchmark_class(
                _fake_benchmark(database=_fake_database(text=True, image=True))
            )
        },
    )


def test_precompute_is_unsplit_unless_asked(tmp_path, monkeypatch):
    """``--split-both-capability-datasets`` is opt-in: without it a mixed-modality
    dataset stays one job writing the mapped file."""
    _use_mixed_modality_benchmark(monkeypatch)
    jobs = srp.enumerate_jobs("t1", tmp_path, _precompute_args())

    assert len(jobs) == 1
    assert jobs[0].spec["precompute_modality"] is None
    assert jobs[0].spec["precompute_path"] == "out.json"
    assert set(jobs[0].required_capabilities) == {"embedding", "text_kv", "image_kv"}


def test_split_gives_a_mixed_dataset_one_job_per_modality(tmp_path, monkeypatch):
    """Unsplit, a text+image recording matches only a worker holding all four KV servers,
    and its text half queues behind its image half on the same GPUs."""
    _use_mixed_modality_benchmark(monkeypatch)
    jobs = srp.enumerate_jobs(
        "t1", tmp_path, _precompute_args(split_both_capability_datasets=True)
    )

    assert [j.spec["precompute_modality"] for j in jobs] == ["text", "image"]
    # Distinct ids and directories, or the two halves would share a results cache and a
    # log root.
    assert len({j.job_id for j in jobs}) == 2
    assert len({j.output_dir for j in jobs}) == 2
    # Never the mapped path: a store rewrites its whole file after every query.
    assert [j.spec["precompute_path"] for j in jobs] == ["out.text.json", "out.image.json"]
    assert all(j.spec["precompute_base_path"] == "out.json" for j in jobs)
    assert [set(j.required_capabilities) for j in jobs] == [
        {"embedding", "text_kv"},
        {"embedding", "image_kv"},
    ]
    # Everything that is a property of the dataset rather than of a half is copied
    # verbatim into both.
    assert {j.spec["n_queries"] for j in jobs} == {FAKE_QUERY_COUNT}
    assert {j.phase for j in jobs} == {1}
    assert {j.spec["simulate"] for j in jobs} == {False}


def test_split_leaves_a_single_modality_dataset_alone(tmp_path):
    """So the flag can be left on for a --precompute naming ecommerce alongside the
    text-only datasets."""
    jobs = srp.enumerate_jobs(
        "t1", tmp_path, _precompute_args(split_both_capability_datasets=True)
    )
    assert len(jobs) == 1
    assert jobs[0].spec["precompute_path"] == "out.json"
    assert jobs[0].spec["precompute_modality"] is None


def test_a_precompute_task_has_nothing_to_merge(tmp_path):
    """One job per dataset writes the operator's own file, so there is exactly one store
    and it is already where they asked for it. Copying it under merged/ would create a
    second multi-GB artifact that later runs would have to choose between.
    """
    task_root = tmp_path / "t1"
    job_dir = task_root / "job_0"
    job_dir.mkdir(parents=True)

    assert srp.merge("t1", [str(job_dir)]) == []
    assert not (task_root / "merged").exists()


def _precompute_job(tmp_path, **spec_overrides):
    spec = {
        "kind": "precompute",
        "simulate": False,
        "simulate_paths": None,
        "sweep_to_gold": False,
        "use_indexes": False,
        "cost_type": "runtime",
        "press_name": "expected_attention",
        "text_small_model": "small-text-model",
        "text_large_model": "large-text-model",
        "image_small_model": "small-image-model",
        "image_large_model": "large-image-model",
        "debug_query": None,
        "guarantee": None,
        "precompute_path": str(tmp_path / "job" / "precompute.json"),
        "n_queries": FAKE_QUERY_COUNT,
    }
    spec.update(spec_overrides)
    return Job(
        job_id="j-precompute", task_id="t1", producer=srp.PRODUCER_NAME,
        benchmark="fake_bench", split="dev", spec=spec,
        required_capabilities=["embedding", "text_kv", "image_kv"],
        output_dir=str(tmp_path / "job"),
    )


def test_precompute_job_runs_the_whole_benchmark_and_leaves_merge_a_note(
    tmp_path, monkeypatch
):
    seen = {}

    def fake_run_precompute(benchmark, executor, output_path, **kwargs):
        seen.update(kwargs)
        seen["output_path"] = Path(output_path)
        return types.SimpleNamespace(counts=lambda: {"text_qa": 3})

    monkeypatch.setattr(srp, "run_precompute", fake_run_precompute)
    monkeypatch.setattr(srp.psweep, "build_precompute_configurator", lambda *a, **k: object())
    monkeypatch.setattr(srp, "build_reasoner", lambda c: object())
    monkeypatch.setattr(srp, "build_approach_executor", lambda *a, **k: object())

    job = _precompute_job(tmp_path)
    result = srp.run_job(job, WorkerContext(device="cpu", worker_id="w1", capability="both"))

    assert result.success, result.error
    # No query range: the whole benchmark, so the store's own reuse applies.
    assert "query_slice" not in seen
    # Straight into the mapped file. No shard, and so no shard_meta.json: with one job
    # per dataset there is nothing left for merge() to gather.
    assert seen["output_path"] == tmp_path / "job" / "precompute.json"
    assert not (tmp_path / "job" / "shard_meta.json").exists()


def test_a_split_half_records_one_modality_into_its_own_file(tmp_path, monkeypatch):
    """The worker half of the split: the skip set is live *while* the pass runs (the env
    var is parsed at import, long before a job is claimed), the store written is the
    sibling rather than the mapped file, and a manifest is left for merge() to find the
    other half by."""
    from reasondb.utils import precompute_modalities

    monkeypatch.setattr(precompute_modalities, "PRECOMPUTE_SKIP_MODALITIES", frozenset())
    seen = {}

    def fake_run_precompute(benchmark, executor, output_path, **kwargs):
        seen["skipping"] = set(precompute_modalities.PRECOMPUTE_SKIP_MODALITIES)
        seen["output_path"] = Path(output_path)
        return types.SimpleNamespace(counts=lambda: {"text_qa": 3})

    monkeypatch.setattr(srp, "run_precompute", fake_run_precompute)
    monkeypatch.setattr(srp.psweep, "build_precompute_configurator", lambda *a, **k: object())
    monkeypatch.setattr(srp, "build_reasoner", lambda c: object())
    monkeypatch.setattr(srp, "build_approach_executor", lambda *a, **k: object())

    base = tmp_path / "job" / "precompute.json"
    base.parent.mkdir(parents=True)
    base.write_text(
        json.dumps(
            {
                "text_qa": {},
                "vision": {},
                "precomputed_ops": ["TextQaFilter|expr|_T0"],
                "operator_configs": {"TextQaFilter|expr|_T0": {"question_template": "q"}},
                "filter_stats": {},
            }
        )
    )
    job = _precompute_job(
        tmp_path,
        precompute_path=str(tmp_path / "job" / "precompute.text.json"),
        precompute_modality="text",
        precompute_skip_modalities=["image", "audio"],
        precompute_base_path=str(base),
        precompute_split_parts=2,
    )

    result = srp.run_job(job, WorkerContext(device="cpu", worker_id="w1", capability="text"))

    assert result.success, result.error
    assert seen["skipping"] == {"image", "audio"}
    assert seen["output_path"] == tmp_path / "job" / "precompute.text.json"
    # Restored, or the worker's next job would silently record nothing.
    assert precompute_modalities.PRECOMPUTE_SKIP_MODALITIES == frozenset()
    # Seeded from the mapped file: without the resume markers a top-up half re-records
    # everything the dataset already has.
    seeded = json.loads((tmp_path / "job" / "precompute.text.json").read_text())
    assert seeded["precomputed_ops"] == ["TextQaFilter|expr|_T0"]
    manifest = json.loads((tmp_path / "job" / "precompute_split.json").read_text())
    assert manifest["modality"] == "text"
    assert manifest["n_parts"] == 2


def test_precompute_job_refuses_a_simulate_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(srp, "run_precompute", lambda *a, **k: None)
    job = _precompute_job(tmp_path, simulate=True)
    result = srp.run_job(job, WorkerContext(device="cpu", worker_id="w1", capability="both"))
    assert result.success is False
    assert "mutually exclusive" in result.error


# ── Scoring off the shards (aligned with producers/run_benchmark.py) ─────────


def _write_pair(tmp_path, *, label_benchmark="fake_bench"):
    """A finished step job's dir and a finished label job's dir, both with shards."""
    step_dir, label_dir = tmp_path / "job_s0", tmp_path / "job_labels"
    common = dict(
        task_id="t1", producer="parameter_sweep", split="dev",
        required_capabilities=["embedding"],
    )
    step_job = Job(job_id="t1-s0", benchmark="fake_bench", spec={},
                   output_dir=str(step_dir), **common)
    label_job = Job(job_id="t1-labels", benchmark=label_benchmark, spec={},
                    output_dir=str(label_dir), **common)
    shards_mod.write_shard(step_job, {
        "kind": "step", "name": "storage_step0_tunetrue_nNone", "label_set": "silver",
        "results": {"q1": {(0.7, 0.7): cached_answer(
            step_dir / "cache", "q1", pd.DataFrame({"a": [1]})
        )}},
        "costs": {"q1": {(0.7, 0.7): object()}},
    })
    shards_mod.write_shard(label_job, {
        "kind": "label", "name": "silver",
        "results": {"q1": cached_answer(
            label_dir / "cache", "q1", pd.DataFrame({"a": [1]})
        )},
    })
    return step_job, label_job


def test_score_job_scores_a_finished_step_against_the_label_shard(tmp_path, monkeypatch):
    """The dashboard's accuracy panel fills job by job, from pickles, in the coordinator.

    No worker is involved and no query is replayed - which is the whole point of moving
    scoring out of the job.
    """
    seen = {}

    def fake_evaluate(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"precision": [1.0]})

    monkeypatch.setattr(shards_mod, "evaluate", fake_evaluate)
    step_job, label_job = _write_pair(tmp_path)

    assert srp.score_job(step_job, [label_job.output_dir]) == ["silver"]
    # The early emitter: score_job reports, merge stays silent (query_metrics is
    # append-only, so two emitters would double every accuracy row).
    assert seen["record_telemetry"] is True
    assert seen["telemetry_context"] == {"labels": "silver", "job_id": "t1-s0"}
    assert seen["approach_name"] == "storage_step0_tunetrue_nNone"


def test_score_job_stays_silent_until_its_labels_land(tmp_path, monkeypatch):
    """A step job routinely finishes before the silver pass it is scored against.

    Returning [] means `coordinator.scoring` retries on its next pass; emitting accuracy
    scored against absent labels would be worse than showing none.
    """
    monkeypatch.setattr(shards_mod, "evaluate", lambda **kw: pytest.fail("scored with no labels"))
    step_job, _ = _write_pair(tmp_path)

    assert srp.score_job(step_job, []) == []


def test_score_job_ignores_a_label_shard_from_another_benchmark(tmp_path, monkeypatch):
    """merge() hands every job dir over at once, so the pairing has to be by benchmark."""
    monkeypatch.setattr(shards_mod, "evaluate", lambda **kw: pytest.fail("scored across benchmarks"))
    step_job, label_job = _write_pair(tmp_path, label_benchmark="other_bench")

    assert srp.score_job(step_job, [label_job.output_dir]) == []


def test_merge_fills_achieved_columns_without_re_emitting_telemetry(tmp_path, monkeypatch):
    """A step job writes achieved_* empty - it does not hold the labels - and merge
    fills them, silently, because score_job already reported those same rows."""
    seen = {}

    def fake_evaluate(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame(
            {"precision": [0.8], "recall": [0.6], "f1_score": [0.7]},
            index=pd.MultiIndex.from_tuples([("storage_step0_tunetrue_nNone", "q1", 0.7, 0.7)]),
        )

    monkeypatch.setattr(shards_mod, "evaluate", fake_evaluate)
    step_job, label_job = _write_pair(tmp_path)
    pd.DataFrame([{
        "benchmark": "fake_bench", "split": "dev", "step": 0, "query": "q1",
        "precision_guarantee": 0.7, "recall_guarantee": 0.7,
        "achieved_precision": None, "achieved_recall": None, "achieved_f1": None,
    }]).to_parquet(Path(step_job.output_dir) / "rows.parquet", index=False)

    srp.merge("t1", [step_job.output_dir, label_job.output_dir])

    assert seen["record_telemetry"] is False
    merged = pd.read_csv(tmp_path / "merged" / "fake_bench" / "dev" / f"{srp.PRODUCER_NAME}.csv")
    assert merged["achieved_precision"].tolist() == [0.8]
    assert merged["achieved_f1"].tolist() == [0.7]


# ---------------------------------------------------------------------------
# Query shape statistics on the merged rows
# ---------------------------------------------------------------------------


def _rows_for(step_job, queries=("q1",), achieved=None, extra=None):
    """The rows.parquet a step job writes, one row per query at one guarantee pair."""
    rows = []
    for query in queries:
        row = {
            "benchmark": "fake_bench", "split": "dev", "step": 0, "query": query,
            "precision_guarantee": 0.7, "recall_guarantee": 0.7,
            "achieved_precision": achieved, "achieved_recall": achieved,
            "achieved_f1": achieved,
        }
        row.update(extra or {})
        rows.append(row)
    pd.DataFrame(rows).to_parquet(
        Path(step_job.output_dir) / "rows.parquet", index=False
    )


def _merged(tmp_path):
    return pd.read_csv(
        tmp_path / "merged" / "fake_bench" / "dev" / f"{srp.PRODUCER_NAME}.csv"
    )


def _no_metrics(**kwargs):
    return None


def test_merge_stamps_query_shape_statistics_on_every_row(tmp_path, monkeypatch):
    monkeypatch.setattr(shards_mod, "evaluate", _no_metrics)
    monkeypatch.setattr(
        shards_mod, "query_stats_for", lambda *_: {"q1": {"num_semops": 3}}
    )
    step_job, label_job = _write_pair(tmp_path)
    _rows_for(step_job)

    srp.merge("t1", [step_job.output_dir, label_job.output_dir])

    assert _merged(tmp_path)["num_semops"].tolist() == [3]


def test_a_query_the_scorer_never_reached_still_gets_its_complexity_bucket(
    tmp_path, monkeypatch
):
    """Why this is not part of fill_achieved.

    The runtime figures and --figures breakdown need no labels at all, so a row whose
    label shard never landed must still carry the bucket it belongs in.
    """
    monkeypatch.setattr(shards_mod, "evaluate", _no_metrics)
    monkeypatch.setattr(
        shards_mod, "query_stats_for", lambda *_: {"q1": {"num_semops": 3}}
    )
    step_job, label_job = _write_pair(tmp_path)
    _rows_for(step_job)

    srp.merge("t1", [step_job.output_dir, label_job.output_dir])

    merged = _merged(tmp_path)
    assert merged["achieved_precision"].isna().all(), "nothing scored this row"
    assert merged["num_semops"].tolist() == [3]


def test_a_benchmark_without_query_statistics_adds_no_columns(tmp_path, monkeypatch):
    """A fixed benchmark, or one whose set is not pinned, gets no statistics columns."""
    monkeypatch.setattr(shards_mod, "evaluate", _no_metrics)
    monkeypatch.setattr(shards_mod, "query_stats_for", lambda *_: {})
    step_job, label_job = _write_pair(tmp_path)
    _rows_for(step_job)

    srp.merge("t1", [step_job.output_dir, label_job.output_dir])

    assert "num_semops" not in _merged(tmp_path).columns


def test_only_declared_statistics_become_columns(tmp_path, monkeypatch):
    """A merged CSV spans six benchmarks; a stray key would be empty on five of them."""
    monkeypatch.setattr(shards_mod, "evaluate", _no_metrics)
    monkeypatch.setattr(
        shards_mod,
        "query_stats_for",
        lambda *_: {"q1": {"num_semops": 2, "shape_name": "filter_then_extract"}},
    )
    step_job, label_job = _write_pair(tmp_path)
    _rows_for(step_job)

    srp.merge("t1", [step_job.output_dir, label_job.output_dir])

    merged = _merged(tmp_path)
    assert merged["num_semops"].tolist() == [2]
    assert "shape_name" not in merged.columns


def test_a_stamped_row_is_not_overwritten_by_a_re_merge(tmp_path, monkeypatch):
    """The writer may stamp these itself; a later re-merge must not clobber it."""
    monkeypatch.setattr(shards_mod, "evaluate", _no_metrics)
    monkeypatch.setattr(
        shards_mod, "query_stats_for", lambda *_: {"q1": {"num_semops": 3}}
    )
    step_job, label_job = _write_pair(tmp_path)
    _rows_for(step_job, extra={"num_semops": 9})

    srp.merge("t1", [step_job.output_dir, label_job.output_dir])

    assert _merged(tmp_path)["num_semops"].tolist() == [9]


def test_a_query_missing_from_the_pinned_set_keeps_an_empty_bucket(tmp_path, monkeypatch):
    """`query_stats` omits a query whose shape declared nothing; it must not raise."""
    monkeypatch.setattr(shards_mod, "evaluate", _no_metrics)
    monkeypatch.setattr(
        shards_mod, "query_stats_for", lambda *_: {"q1": {"num_semops": 3}}
    )
    step_job, label_job = _write_pair(tmp_path)
    _rows_for(step_job, queries=("q1", "q2"))

    srp.merge("t1", [step_job.output_dir, label_job.output_dir])

    merged = _merged(tmp_path).set_index("query")["num_semops"]
    assert merged["q1"] == 3
    assert pd.isna(merged["q2"])
