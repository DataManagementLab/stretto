"""``run_benchmark`` records what ``run_benchmark`` replays.

``parameter_sweep``'s precompute job records against
``build_precompute_configurator``, whose operator set is derived from the KV caches
materialized on disk. This producer's sweep builds ``get_default_configurator``. The two
coincide only when every default baseline happens to be materialized, so replaying a
storage recording here could fail as a `--simulate` cache miss. Recording with the same
configurator the sweep uses guarantees coverage.

The other half of the coverage question is the executor. It does not matter:
``Executor._precompute_pipeline`` loops over ``step.operators`` - every candidate of
every step - rather than the one an optimizer would choose, so a single pass serves all
five approaches plus the silver label pass, all of which share the configurator.
"""

import argparse
import types
from pathlib import Path

import pytest

from reasondb.coordinator.models import WorkerContext
from reasondb.coordinator.producers import run_benchmark as rbp
from reasondb.evaluation.benchmark import RandomBenchmark

FAKE_QUERIES = ["q0", "q1", "q2"]


def _args(**overrides):
    base = dict(
        benchmarks=["fake_bench"],
        split="dev",
        use_indexes=False,
        precision_guarantees=[0.7],
        recall_guarantees=[0.7],
        all_guarantee_combinations=False,
        cost_type="runtime",
        debug_query=None,
        simulate=None,
        precompute={"fake_bench": Path("fake.json")},
        select_executors=[],
        skip_executors=[],
        labels=["silver", "gold"],
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _fake_benchmark():
    return types.SimpleNamespace(
        name=lambda: "fake_bench",
        has_ground_truth=False,
        database=object(),
        queries=list(FAKE_QUERIES),
        query_count=lambda debug_query=None: len(FAKE_QUERIES),
    )


def _fake_benchmark_class(benchmark, text=True, image=False):
    class FakeBenchmark(RandomBenchmark):
        @classmethod
        def name(cls):
            return "fake_bench"

        @property
        def has_ground_truth(self):
            return False

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
            return types.SimpleNamespace(
                has_ground_truth=False,
                database=types.SimpleNamespace(
                    external_tables=[
                        types.SimpleNamespace(
                            name="t",
                            text_columns=["t.body"] if text else [],
                            image_columns=["t.pic"] if image else [],
                            audio_columns=[],
                        )
                    ]
                ),
            )

        @classmethod
        def count_queries(cls, split, debug_query=None):
            return len(FAKE_QUERIES)

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
def _patch(monkeypatch):
    benchmark = _fake_benchmark()
    monkeypatch.setattr(
        rbp, "ALL_BENCHMARKS", {"fake_bench": _fake_benchmark_class(benchmark)}
    )
    monkeypatch.setattr(
        rbp, "enumerate_filter_stats_jobs", lambda task_id, out, args, producer: []
    )
    return benchmark


def test_precompute_mode_enumerates_no_sweep_points(tmp_path):
    """An approach job's spec carries its --simulate path at enumeration time, and that
    file does not exist until this task has written it - hence two separate tasks."""
    jobs = rbp.enumerate_jobs("t1", tmp_path, _args())

    assert {j.spec["kind"] for j in jobs} == {"precompute"}


def test_one_job_per_dataset_writing_the_mapped_file(tmp_path):
    jobs = rbp.enumerate_jobs(
        "t1", tmp_path, _args(precompute={"fake_bench": tmp_path / "mine.json"})
    )

    assert len(jobs) == 1
    (job,) = jobs
    assert job.spec["precompute_path"] == str(tmp_path / "mine.json")
    assert job.spec["n_queries"] == len(FAKE_QUERIES)
    assert job.phase == 1


def test_a_repeated_benchmark_still_gets_one_job(tmp_path):
    """Two jobs would collide on job_id and race the same store; the mapping is a dict,
    so the duplicate cannot be expressed."""
    jobs = rbp.enumerate_jobs(
        "t1", tmp_path, _args(benchmarks=["fake_bench", "fake_bench"])
    )
    assert len(jobs) == 1


def test_it_asks_only_for_the_modalities_the_benchmark_uses(tmp_path):
    (job,) = rbp.enumerate_jobs("t1", tmp_path, _args())
    assert set(job.required_capabilities) == {"embedding", "text_kv"}


def test_split_gives_a_mixed_dataset_one_job_per_modality(tmp_path, monkeypatch):
    """This producer records the curated benchmarks, and ``ecommerce_curated`` is
    mixed-modality too - so ``--split-both-capability-datasets`` has to reach here and not
    only ``parameter_sweep``."""
    monkeypatch.setattr(
        rbp,
        "ALL_BENCHMARKS",
        {"fake_bench": _fake_benchmark_class(_fake_benchmark(), text=True, image=True)},
    )
    jobs = rbp.enumerate_jobs(
        "t1",
        tmp_path,
        _args(
            precompute={"fake_bench": tmp_path / "mine.json"},
            split_both_capability_datasets=True,
        ),
    )

    assert [j.spec["precompute_modality"] for j in jobs] == ["text", "image"]
    assert [j.spec["precompute_path"] for j in jobs] == [
        str(tmp_path / "mine.text.json"),
        str(tmp_path / "mine.image.json"),
    ]
    assert [set(j.required_capabilities) for j in jobs] == [
        {"embedding", "text_kv"},
        {"embedding", "image_kv"},
    ]
    assert len({j.job_id for j in jobs}) == len({j.output_dir for j in jobs}) == 2


def test_split_leaves_a_single_modality_dataset_alone(tmp_path):
    (job,) = rbp.enumerate_jobs(
        "t1", tmp_path, _args(split_both_capability_datasets=True)
    )
    assert job.spec["precompute_modality"] is None
    assert job.spec["precompute_path"] == "fake.json"


def test_a_precompute_job_is_never_handed_to_a_simulate_worker(tmp_path):
    """A simulate worker starts only the embedding servers, so letting it claim this
    would mean recording responses with no model behind them."""
    from reasondb.coordinator.scheduler import worker_can_run

    (job,) = rbp.enumerate_jobs("t1", tmp_path, _args())
    assert job.spec["simulate"] is False
    assert not worker_can_run("simulate", job.required_capabilities, job_is_simulate=False)


def test_selecting_one_executor_does_not_narrow_what_is_recorded(tmp_path):
    """The whole point: every approach shares get_default_configurator, and
    _precompute_pipeline runs every candidate operator rather than an optimizer's pick.
    So `--select-executors optim_global` records exactly what `lotus` will later need.
    """
    all_five = rbp.enumerate_jobs("t1", tmp_path, _args())
    just_one = rbp.enumerate_jobs(
        "t1", tmp_path, _args(select_executors=["optim_global"])
    )

    assert [j.spec for j in all_five] == [j.spec for j in just_one]


def test_the_recording_pass_uses_the_configurator_the_sweep_uses(tmp_path, monkeypatch):
    """Not build_precompute_configurator: that one's operator set comes from the caches
    materialized on disk, which is a different question from what this sweep configures.
    """
    seen = {}

    monkeypatch.setattr(
        rbp, "get_default_configurator",
        lambda use_indexes: seen.setdefault("use_indexes", use_indexes) or object(),
    )
    monkeypatch.setattr(rbp, "get_label_configurator", lambda: object())
    monkeypatch.setattr(rbp, "build_reasoner", lambda c: object())
    monkeypatch.setattr(
        rbp, "_build_executor",
        lambda name, *a, **k: seen.setdefault("executor", name) or object(),
    )

    def fake_run_precompute(benchmark, executor, output_path, **kwargs):
        seen["output_path"] = Path(output_path)
        seen["kwargs"] = kwargs
        return types.SimpleNamespace(counts=lambda: {"n_text_qa": 3})

    monkeypatch.setattr(rbp, "run_precompute", fake_run_precompute)

    (job,) = rbp.enumerate_jobs(
        "t1", tmp_path, _args(precompute={"fake_bench": tmp_path / "out.json"})
    )
    result = rbp.run_job(job, WorkerContext(device="cpu", worker_id="w1", capability="both"))

    assert result.success, result.error
    assert seen["use_indexes"] is False
    assert seen["executor"] == "optim_global"
    assert seen["output_path"] == tmp_path / "out.json"
    # The whole benchmark, unsharded: the store's (operator, expression) reuse applies
    # across every query rather than being recomputed per shard.
    assert "query_slice" not in seen["kwargs"]


def test_use_indexes_is_carried_into_the_recording(tmp_path, monkeypatch):
    """The operator identifiers differ between indexed and direct materialization, so a
    recording made under one setting misses under the other."""
    seen = {}
    monkeypatch.setattr(
        rbp, "get_default_configurator",
        lambda use_indexes: seen.setdefault("use_indexes", use_indexes) or object(),
    )
    monkeypatch.setattr(rbp, "get_label_configurator", lambda: object())
    monkeypatch.setattr(rbp, "build_reasoner", lambda c: object())
    monkeypatch.setattr(rbp, "_build_executor", lambda *a, **k: object())
    monkeypatch.setattr(
        rbp, "run_precompute",
        lambda *a, **k: types.SimpleNamespace(counts=lambda: {}),
    )

    (job,) = rbp.enumerate_jobs("t1", tmp_path, _args(use_indexes=True))
    rbp.run_job(job, WorkerContext(device="cpu", worker_id="w1", capability="both"))

    assert seen["use_indexes"] is True


def test_a_precompute_job_marked_simulate_fails_rather_than_recording_nothing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rbp, "run_precompute", lambda *a, **k: None)
    (job,) = rbp.enumerate_jobs("t1", tmp_path, _args())
    job.spec["simulate"] = True

    result = rbp.run_job(job, WorkerContext(device="cpu", worker_id="w1", capability="both"))

    assert result.success is False
    assert "mutually exclusive" in result.error
