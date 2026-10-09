"""The phase-0 job all three producers prepend.

Enumeration happens before anything has run, which is the whole difficulty: a
``RandomBenchmark``'s query set is *sampled from* the filter stats, so at enumeration
time there is no query set, and asking for one would generate a randomly-sampled set and
dump it - making the wrong set authoritative. So the count on this job comes from the
class-level operator pool, and every other job of the benchmark is enqueued with
``n_queries: None`` and back-filled when this one completes.
"""

import argparse
import json
import types
from pathlib import Path

import pytest

from reasondb.backends.simulate_store import SimulateStore
from reasondb.coordinator.producers import filter_stats_jobs as fsj
from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.evaluation.benchmark import RandomBenchmark
from reasondb.query_plan.logical_plan import LogicalFilter
from reasondb.query_plan.query import (
    OperatorOption,
    OperatorPlaceholder,
    QueryShape,
)

SHAPE = QueryShape(
    OperatorPlaceholder(
        LogicalFilter,
        inputs=[VirtualTableIdentifier("rows")],
        output=VirtualTableIdentifier("output"),
    ),
)


def _benchmark_cls(tmp_path, n_options=3, name="fake_random"):
    options = [
        OperatorOption(LogicalFilter, "{rows.text} matches p%d" % i)
        for i in range(n_options)
    ]

    class FakeRandom(RandomBenchmark):
        @classmethod
        def name(cls):
            return name

        @classmethod
        def filter_stats_dir(cls, split):
            return tmp_path / "stats" / name / str(split)

        @property
        def has_ground_truth(self):
            return False

        @staticmethod
        def urls():
            return {}

        @staticmethod
        def download(split):
            raise AssertionError("not used")

        @classmethod
        def _load_database(cls, split):
            # Only what capabilities_for_benchmark reads: a text-only benchmark.
            return types.SimpleNamespace(
                external_tables=[
                    types.SimpleNamespace(
                        name="rows",
                        text_columns=["rows.text"],
                        image_columns=[],
                        audio_columns=[],
                    )
                ]
            )

        @classmethod
        def _get_query_shapes(cls):
            return [SHAPE]

        @classmethod
        def _get_operator_options(cls):
            return options

        @classmethod
        def _single_filter_shape(cls):
            return SHAPE

    return FakeRandom


def _args(**overrides):
    base = dict(benchmarks=["fake_random"], split="dev", precompute=None, simulate=None)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def registry(monkeypatch, tmp_path):
    cls = _benchmark_cls(tmp_path)
    monkeypatch.setattr(fsj, "BENCHMARKS", {"fake_random": cls})
    return cls


def test_one_phase_zero_job_per_random_benchmark(registry, tmp_path):
    args = _args(precompute={"fake_random": tmp_path / "fake.json"})
    jobs = fsj.enumerate_filter_stats_jobs("t1", tmp_path, args, "parameter_sweep")

    assert len(jobs) == 1
    (job,) = jobs
    assert job.phase == 0
    assert job.spec["kind"] == "filter_stats"
    assert job.spec["precompute_path"] == str(tmp_path / "fake.json")


def test_the_count_comes_from_the_pool_not_from_a_query_set(registry, tmp_path):
    """Exact at enumeration, with no database and no queries - which is the point: this
    is the one job whose size is knowable before the query set exists."""
    args = _args(precompute={"fake_random": tmp_path / "fake.json"})
    (job,) = fsj.enumerate_filter_stats_jobs("t1", tmp_path, args, "parameter_sweep")

    assert job.spec["n_queries"] == 3
    assert not (registry.benchmark_dir() / "dev" / "queries.json").exists()


def test_a_benchmark_that_is_already_set_up_is_skipped(registry, tmp_path, monkeypatch):
    stats_dir = registry.filter_stats_dir("dev")
    stats_dir.mkdir(parents=True)
    (stats_dir / "stats.json").write_text(json.dumps({"keys": {}}))
    monkeypatch.setattr(fsj, "has_pinned_queries", lambda cls, split: True)

    args = _args(precompute={"fake_random": tmp_path / "fake.json"})
    assert fsj.enumerate_filter_stats_jobs("t1", tmp_path, args, "parameter_sweep") == []


def test_stats_on_disk_without_a_pinned_query_set_still_gets_a_job(registry, tmp_path):
    """Both halves matter: the matrix is what makes sampling possible, the pinned file
    is what makes every later job execute the same queries."""
    stats_dir = registry.filter_stats_dir("dev")
    stats_dir.mkdir(parents=True)
    (stats_dir / "stats.json").write_text(json.dumps({"keys": {}}))

    args = _args(precompute={"fake_random": tmp_path / "fake.json"})
    assert len(fsj.enumerate_filter_stats_jobs("t1", tmp_path, args, "parameter_sweep")) == 1


def test_nothing_is_enqueued_once_the_matrix_and_the_query_set_both_exist(
    registry, tmp_path, monkeypatch
):
    store = SimulateStore()
    store.record_filter_stats("fake_random", "dev", {"keys": {}})
    path = tmp_path / "fake.json"
    store.save(path)
    monkeypatch.setattr(fsj, "has_pinned_queries", lambda cls, split: True)

    args = _args(simulate={"fake_random": path})
    assert fsj.enumerate_filter_stats_jobs("t1", tmp_path, args, "parameter_sweep") == []


def test_a_recorded_matrix_without_a_pinned_query_set_still_gets_a_job(
    registry, tmp_path
):
    """A simulate run on a fresh machine has the matrix but not the file drawn from it.
    Letting every worker regenerate would be deterministic only if their code and stats
    agreed. The job replays and writes it instead,
    touching no model.
    """
    store = SimulateStore()
    store.record_filter_stats("fake_random", "dev", {"keys": {}})
    path = tmp_path / "fake.json"
    store.save(path)

    (job,) = fsj.enumerate_filter_stats_jobs(
        "t1", tmp_path, _args(simulate={"fake_random": path}), "parameter_sweep"
    )
    assert job.spec["simulate"] is True
    assert job.required_capabilities == ["embedding"]


def test_simulating_against_a_pre_bucket_store_fails_at_enumeration(registry, tmp_path):
    """Computing the matrix needs the model servers, and a --capability simulate worker
    never started any. Failing here costs seconds; failing at claim time costs however
    long the task had been running.
    """
    path = tmp_path / "old.json"
    SimulateStore().save(path)
    args = _args(simulate={"fake_random": path})

    with pytest.raises(SystemExit, match="carries no filter stats"):
        fsj.enumerate_filter_stats_jobs("t1", tmp_path, args, "parameter_sweep")


def test_a_recording_job_asks_only_for_the_modalities_it_uses(registry, tmp_path):
    """The pass runs the gold operators over this benchmark's filters, so it needs that
    benchmark's servers - not both families. A text-only dataset asking for image_kv
    would require every worker to be --capability both.
    """
    args = _args(precompute={"fake_random": tmp_path / "fake.json"})
    (job,) = fsj.enumerate_filter_stats_jobs("t1", tmp_path, args, "parameter_sweep")

    assert job.spec["simulate"] is False
    assert set(job.required_capabilities) == {"embedding", "text_kv"}


def test_a_non_random_benchmark_needs_no_stats(monkeypatch, tmp_path):
    """A fixed benchmark carries a hand-written query list; nothing is sampled."""
    monkeypatch.setattr(fsj, "BENCHMARKS", {"fixed_bench": type("Fixed", (), {})})
    args = _args(benchmarks=["fixed_bench"], precompute={"fixed_bench": Path("x.json")})

    assert fsj.enumerate_filter_stats_jobs("t1", tmp_path, args, "parameter_sweep") == []


def test_a_benchmark_listed_twice_gets_one_job(registry, tmp_path):
    """Two jobs would collide on job_id and race the same store."""
    args = _args(
        benchmarks=["fake_random", "fake_random"],
        precompute={"fake_random": tmp_path / "fake.json"},
    )
    jobs = fsj.enumerate_filter_stats_jobs("t1", tmp_path, args, "parameter_sweep")

    assert len(jobs) == 1
