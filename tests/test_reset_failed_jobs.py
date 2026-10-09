"""``scripts/reset_failed_jobs.py``: which database it opens, and when it declines.

The reset arithmetic itself is ``JobDB.reset_failed_jobs`` and is covered in
``test_coordinator_db.py``. What is script-shaped, and what is tested here, is everything
around it: resolving a task id to a queue on disk (a task id is *not* required to be one
the cluster config knows), the live-coordinator guard, and the two ways it declines
without changing anything.
"""

import argparse
import sqlite3
import time
from pathlib import Path

import pytest

reset_failed_jobs = pytest.importorskip("scripts.reset_failed_jobs")

from reasondb.coordinator.db import JobDB
from reasondb.coordinator.models import JOB_FAILED, JOB_RUNNING, Job


def _seed(db_dir: Path, task_id: str, states=(JOB_FAILED,)) -> JobDB:
    """A task's queue on disk, with one job per state. Returns the open connection, so a
    test can play the live coordinator that still holds one."""
    db = JobDB(db_dir / "coordinator.db")
    for index, state in enumerate(states):
        db.enqueue_job(
            Job(
                job_id=f"{task_id}-{index}",
                task_id=task_id,
                producer="parameter_sweep",
                benchmark="movie_random",
                split="dev",
                spec={"kind": "precompute"},
                required_capabilities=["embedding"],
                output_dir=str(db_dir / "job"),
                state=state,
                attempt=3,
                created_at=time.time(),
                error="boom",
            )
        )
    return db


def _reset(db_dir: Path, task_id: str, **kwargs) -> int:
    return reset_failed_jobs.reset_task(
        task_id,
        db_dir / "coordinator.db",
        apply=kwargs.pop("apply", True),
        force=kwargs.pop("force", False),
        benchmarks=kwargs.pop("benchmarks", None),
        job_ids=kwargs.pop("job_ids", None),
    )


# ── Which database it opens ──────────────────────────────────────────────────


def test_a_task_id_the_config_never_heard_of_resolves_to_the_default_path(tmp_path):
    """The cluster config is a source of *overrides*, not the list of acceptable task
    ids: a hand-launched run's queue is at <results-root>/<task-id> like any other."""
    assert reset_failed_jobs.db_path_for("my_task", tmp_path, overrides={}) == (
        tmp_path / "my_task" / "coordinator.db"
    )


def test_a_config_override_wins_over_the_default_path(tmp_path):
    overrides = {"abl01": tmp_path / "elsewhere"}
    assert reset_failed_jobs.db_path_for("abl01", tmp_path, overrides) == (
        tmp_path / "elsewhere" / "coordinator.db"
    )


def test_an_explicit_output_dir_wins_over_both(tmp_path):
    """The escape hatch for a run that named its own --output-dir and is in no config."""
    overrides = {"abl01": tmp_path / "elsewhere"}
    assert reset_failed_jobs.db_path_for(
        "abl01", tmp_path, overrides, output_dir=tmp_path / "scratch"
    ) == (tmp_path / "scratch" / "coordinator.db")


def test_output_dir_overrides_survive_a_config_that_is_not_there(tmp_path):
    """A task need not come from a config, so an absent one is not fatal - it simply
    contributes no overrides and every task resolves to the default path."""
    assert reset_failed_jobs.output_dir_overrides(tmp_path / "nope.yaml") == {}


def test_output_dir_is_refused_for_more_than_one_task(tmp_path, monkeypatch):
    """One directory holds one task's queue, so two ids pointed at it is a typo either
    way. --all is the same statement, and is checked explicitly for the config that
    happens to hold a single experiment."""
    monkeypatch.setattr("sys.argv", ["reset_failed_jobs.py", "a", "b", "--output-dir", str(tmp_path)])
    with pytest.raises(AssertionError, match="--output-dir names one task"):
        reset_failed_jobs.main()


def test_a_missing_database_is_skipped_not_fatal(tmp_path, caplog):
    """One unreachable task must not take the other task ids on the command line with
    it - and the message has to name the two flags that move the search."""
    with caplog.at_level("WARNING"):
        assert _reset(tmp_path, "nosuchtask") == 0
    assert "--results-root" in caplog.text and "--output-dir" in caplog.text


# ── When it declines ─────────────────────────────────────────────────────────


def test_a_task_that_looks_live_is_refused_without_force(tmp_path, caplog):
    db = _seed(tmp_path, "t1", states=(JOB_FAILED, JOB_RUNNING))
    try:
        with caplog.at_level("ERROR"):
            assert _reset(tmp_path, "t1") == 0
        assert "Stop it first" in caplog.text
        assert db.task_summary("t1")["failed"] == 1  # untouched
    finally:
        db.close()


def test_force_resets_a_task_that_looks_live(tmp_path):
    db = _seed(tmp_path, "t1", states=(JOB_FAILED, JOB_RUNNING))
    try:
        assert _reset(tmp_path, "t1", force=True) == 1
        summary = db.task_summary("t1")
        assert (summary["failed"], summary["pending"]) == (0, 1)
    finally:
        db.close()


def test_a_live_connection_sees_the_reset_without_reopening_anything(tmp_path):
    """Which is what makes running this against a live coordinator meaningful at all:
    ``claim_next_job`` reads the table on every claim, so a reset job is claimable
    immediately - no restart, and nothing cached in the coordinator to invalidate."""
    db = _seed(tmp_path, "t1")
    try:
        assert db.task_summary("t1")["failed"] == 1
        _reset(tmp_path, "t1")
        assert db.task_summary("t1")["pending"] == 1
    finally:
        db.close()


def test_a_locked_database_changes_nothing_and_says_so(tmp_path, caplog, monkeypatch):
    """The ordinary outcome of a coordinator holding the write lock past sqlite3's 5s
    busy timeout. The selection and the write share one transaction, so the task is
    exactly as it was and re-running is the whole fix - which the message has to say,
    rather than a traceback ending in ``sqlite3.OperationalError``.

    The lock is raised rather than really held: ``db.py``'s module-level ``_LOCK``
    serializes JobDB calls *within a process*, so a test that held a real cursor here
    would deadlock on that instead of ever reaching sqlite - which is a property of
    driving both halves from one process, not of the script, which is always its own.
    """
    db = _seed(tmp_path, "t1")

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(JobDB, "reset_failed_jobs", locked)
    try:
        with caplog.at_level("ERROR"):
            assert _reset(tmp_path, "t1") == 0
        assert "database is locked" in caplog.text
        assert "run this again" in caplog.text
        assert db.task_summary("t1")["failed"] == 1
    finally:
        db.close()


def test_a_dry_run_writes_nothing(tmp_path):
    db = _seed(tmp_path, "t1")
    try:
        assert _reset(tmp_path, "t1", apply=False) == 1
        assert db.task_summary("t1")["failed"] == 1
    finally:
        db.close()
