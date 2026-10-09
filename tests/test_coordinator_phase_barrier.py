"""A phase is a barrier, unlike ``priority``, which is only a hint.

Some work cannot be run in any order: a ``RandomBenchmark`` samples its queries from the
filter stats, so the phase-0 job that computes them has to *finish* before a phase-1 job
can execute a single multi-operator query. Priority cannot express that - it is documented
as advisory precisely because with several workers a high-priority job can still be
running while a low-priority one starts (see ``coordinator/scoring.py``).

The gate lives in ``claim_next_job``'s SQL rather than in ``scheduler.pick_next_job`` so
it is evaluated inside the claiming transaction; two workers reading "phase 0 is nearly
done" must not both jump to phase 1.
"""

import sqlite3

import pytest

from reasondb.coordinator.db import JobDB
from reasondb.coordinator.models import JOB_FAILED, JOB_PENDING, Job, Worker


def _job(job_id, phase=0, task_id="t1", **kw):
    defaults = dict(
        job_id=job_id,
        task_id=task_id,
        producer="storage_runtime",
        benchmark="movie_random",
        split="dev",
        spec={"simulate": False},
        required_capabilities=["embedding"],
        output_dir="/tmp/job",
        created_at=0,
        phase=phase,
    )
    defaults.update(kw)
    return Job(**defaults)


@pytest.fixture()
def db(tmp_path):
    d = JobDB(tmp_path / "coord.db")
    d.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    yield d
    d.close()


def test_a_later_phase_is_unclaimable_until_the_earlier_one_is_terminal(db):
    db.enqueue_job(_job("stats", phase=0))
    db.enqueue_job(_job("sweep", phase=1))

    claimed = db.claim_next_job("w1", "both")
    assert claimed is not None and claimed.job_id == "stats"
    # The phase-1 job exists and this worker can run it - it is simply not available.
    assert db.claim_next_job("w1", "both") is None

    db.start_job("stats", "w1")
    db.complete_job("stats", "w1", {})

    unblocked = db.claim_next_job("w1", "both")
    assert unblocked is not None and unblocked.job_id == "sweep"


def test_the_barrier_waits_for_every_job_of_the_phase(db):
    """One benchmark's stats finishing does not unblock another's sweep: the phase is
    the unit, so a task spanning datasets pins its whole query set before sweeping."""
    db.enqueue_job(_job("stats-movie", phase=0))
    db.enqueue_job(_job("stats-artwork", phase=0))
    db.enqueue_job(_job("sweep", phase=1))

    first = db.claim_next_job("w1", "both")
    db.complete_job(first.job_id, "w1", {})

    assert db.claim_next_job("w1", "both").phase == 0
    assert db.claim_next_job("w1", "both") is None


def test_a_failed_but_retrying_job_keeps_the_barrier_down(db):
    """``fail_job`` returns a job to pending while attempts remain. Treating that as
    'phase over' would let the sweep start against stats that may still arrive."""
    db.enqueue_job(_job("stats", phase=0, max_attempts=3))
    db.enqueue_job(_job("sweep", phase=1))

    db.claim_next_job("w1", "both")
    db.fail_job("stats", "w1", "server died")

    assert db.get_job("stats").state == JOB_PENDING
    claimed = db.claim_next_job("w1", "both")
    assert claimed is not None and claimed.job_id == "stats"


def test_two_workers_cannot_both_jump_the_barrier(db):
    """The candidate query and the claim share one transaction, so a worker that claims
    the last phase-0 job cannot have a second worker read the barrier as already lifted."""
    db.register_worker(Worker(worker_id="w2", task_id="t1", capability="both"))
    db.enqueue_job(_job("stats", phase=0))
    db.enqueue_job(_job("sweep-a", phase=1))
    db.enqueue_job(_job("sweep-b", phase=1))

    first = db.claim_next_job("w1", "both")
    second = db.claim_next_job("w2", "both")

    assert first.job_id == "stats"
    assert second is None


def test_phase_zero_only_tasks_behave_exactly_as_before(db):
    """A producer that emits no phase lands every job at phase 0, so a task with no
    barrier schedules as if phases did not exist."""
    db.enqueue_job(_job("a", phase=0))
    db.enqueue_job(_job("b", phase=0))

    assert db.claim_next_job("w1", "both") is not None
    assert db.claim_next_job("w1", "both") is not None


def test_a_dead_phase_cascades_so_the_task_can_terminate(db):
    """Without the cascade, a failed phase-0 job strands every later job in pending:
    all_terminal never turns true, the coordinator never merges, and every worker spins
    in its no-job backoff until someone notices."""
    db.enqueue_job(_job("stats", phase=0, max_attempts=1))
    db.enqueue_job(_job("sweep-a", phase=1))
    db.enqueue_job(_job("sweep-b", phase=2))

    db.claim_next_job("w1", "both")
    db.fail_job("stats", "w1", "no model server")
    assert db.get_job("stats").state == JOB_FAILED
    assert db.task_summary("t1")["all_terminal"] is False

    assert db.fail_blocked_jobs("t1") == 2

    assert db.task_summary("t1")["all_terminal"] is True
    assert "blocked: phase 0" in db.get_job("sweep-a").error


def test_the_cascade_holds_off_while_the_phase_can_still_recover(db):
    """One failed job does not doom the phase if a sibling is still running - it may
    produce exactly the artifact the later phase needs."""
    db.register_worker(Worker(worker_id="w2", task_id="t1", capability="both"))
    db.enqueue_job(_job("stats-a", phase=0, max_attempts=1))
    db.enqueue_job(_job("stats-b", phase=0))
    db.enqueue_job(_job("sweep", phase=1))

    db.claim_next_job("w1", "both")
    db.claim_next_job("w2", "both")
    db.fail_job("stats-a", "w1", "boom")

    assert db.fail_blocked_jobs("t1") == 0
    assert db.get_job("sweep").state == JOB_PENDING


def test_the_cascade_leaves_a_healthy_task_alone(db):
    db.enqueue_job(_job("stats", phase=0))
    db.enqueue_job(_job("sweep", phase=1))

    assert db.fail_blocked_jobs("t1") == 0


def test_the_summary_says_where_the_barrier_is_and_what_waits_behind_it(db):
    db.enqueue_job(_job("stats", phase=0))
    db.enqueue_job(_job("sweep", phase=1))

    summary = db.task_summary("t1")
    assert summary["phase"] == 0 and summary["blocked"] == 1

    db.claim_next_job("w1", "both")
    db.complete_job("stats", "w1", {})

    summary = db.task_summary("t1")
    assert summary["phase"] == 1 and summary["blocked"] == 0


def test_an_existing_database_gains_the_column_and_keeps_its_jobs(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` does nothing to an existing table, and a coordinator
    resumes a task without re-enumerating - so without a migration every read of a
    pre-existing task's jobs would raise on the missing column.
    """
    path = tmp_path / "old.db"
    db = JobDB(path)
    db.enqueue_job(_job("legacy", phase=0))
    db.close()

    conn = sqlite3.connect(str(path))
    # Simulate a database written before phases existed. The index has to go first;
    # sqlite refuses to drop a column an index still references.
    conn.execute("DROP INDEX idx_jobs_task_phase")
    conn.execute("ALTER TABLE jobs DROP COLUMN phase")
    conn.commit()
    conn.close()

    reopened = JobDB(path)
    try:
        job = reopened.get_job("legacy")
        assert job is not None and job.phase == 0
    finally:
        reopened.close()
