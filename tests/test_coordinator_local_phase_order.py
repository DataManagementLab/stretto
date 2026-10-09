"""``--local`` has no queue, so the SQL barrier does nothing for it.

Distributed runs get their ordering from ``db.claim_next_job``. ``run_local`` just walks
the list ``enumerate_jobs`` returned, so it must not rely on producers listing phase-0
jobs first. It has to sort, and it has to skip work whose prerequisite failed, because a sweep whose
filter stats never landed has no query set to sweep over and would fail confusingly
deeper in.
"""

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import scripts.run_coordinator as rc
except ImportError:  # pragma: no cover - deps not installed
    pytest.skip("coordinator deps not installed", allow_module_level=True)

from reasondb.coordinator.models import Job, JobResult


def _job(job_id, phase, priority=0, tmp_path=Path("/tmp")):
    return Job(
        job_id=job_id,
        task_id="t1",
        producer="storage_runtime",
        benchmark="movie_random",
        split="dev",
        spec={"simulate": False},
        required_capabilities=["embedding"],
        output_dir=str(tmp_path / job_id),
        phase=phase,
        priority=priority,
        created_at=0,
    )


def _install(monkeypatch, jobs, failing=()):
    ran = []

    def run_job(job, worker):
        ran.append(job.job_id)
        if job.job_id in failing:
            return JobResult(success=False, error="boom")
        return JobResult(success=True)

    monkeypatch.setattr(
        rc, "get_producer",
        lambda name: types.SimpleNamespace(
            enumerate_jobs=lambda task_id, out, args: jobs,
            run_job=run_job,
            merge=lambda task_id, dirs: [],
        ),
    )
    return ran


def _args(tmp_path):
    return types.SimpleNamespace(
        producer="parameter_sweep", task_id="t1", output_dir=tmp_path, device="cpu"
    )


def test_phase_zero_runs_first_even_when_enumerated_last(monkeypatch, tmp_path):
    jobs = [
        _job("sweep-a", phase=1, tmp_path=tmp_path),
        _job("stats", phase=0, tmp_path=tmp_path),
        _job("sweep-b", phase=1, tmp_path=tmp_path),
    ]
    ran = _install(monkeypatch, jobs)

    rc.run_local(_args(tmp_path))

    assert ran[0] == "stats"
    assert set(ran) == {"stats", "sweep-a", "sweep-b"}


def test_priority_still_orders_within_a_phase(monkeypatch, tmp_path):
    """Labels are enqueued at priority -1 so the shared cache is warm before the sweep;
    phases must not flatten that."""
    jobs = [
        _job("step", phase=1, priority=0, tmp_path=tmp_path),
        _job("labels", phase=1, priority=-1, tmp_path=tmp_path),
    ]
    ran = _install(monkeypatch, jobs)

    rc.run_local(_args(tmp_path))

    assert ran == ["labels", "step"]


def test_a_failed_phase_skips_everything_behind_it(monkeypatch, tmp_path):
    jobs = [
        _job("stats", phase=0, tmp_path=tmp_path),
        _job("sweep", phase=1, tmp_path=tmp_path),
    ]
    ran = _install(monkeypatch, jobs, failing={"stats"})

    rc.run_local(_args(tmp_path))

    assert ran == ["stats"]


def test_a_failure_does_not_skip_its_own_phase(monkeypatch, tmp_path):
    """Two datasets' stats are independent; one failing must not cancel the other, which
    may still be the one the operator cares about."""
    jobs = [
        _job("stats-movie", phase=0, tmp_path=tmp_path),
        _job("stats-artwork", phase=0, tmp_path=tmp_path),
    ]
    ran = _install(monkeypatch, jobs, failing={"stats-movie"})

    rc.run_local(_args(tmp_path))

    assert ran == ["stats-movie", "stats-artwork"]
