"""``coordinator.db``/``coordinator.scheduler`` state machine, driven directly.

Pure SQLite + ``tmp_path``, no Flask, no HTTP: fast, and this is the layer where a
state-machine bug would be most costly on a long-running sweep.
"""

import time

import pytest

from reasondb.coordinator.db import JobDB
from reasondb.coordinator.models import (
    JOB_DONE,
    JOB_FAILED,
    JOB_PENDING,
    Job,
    Worker,
)


def _job(task_id="t1", job_id="", spec=None, required=("text_kv", "embedding"), **kw):
    defaults = dict(
        job_id=job_id,
        task_id=task_id,
        producer="storage_runtime",
        benchmark="movie_random",
        split="dev",
        spec=spec or {"step_idx": 0, "simulate": False},
        required_capabilities=list(required),
        output_dir="/tmp/job",
        created_at=0,
    )
    defaults.update(kw)
    return Job(**defaults)


@pytest.fixture()
def db(tmp_path):
    d = JobDB(tmp_path / "coord.db")
    yield d
    d.close()


def test_capability_matching_only_serves_jobs_a_worker_can_run(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="embedding-only"))
    db.register_worker(Worker(worker_id="w2", task_id="t1", capability="both"))
    db.enqueue_job(_job(spec={"step_idx": 0, "simulate": False}))
    db.enqueue_job(_job(spec={"step_idx": 1, "simulate": True}))

    # embedding-only can only run the simulate job, never the real-KV one.
    claimed = db.claim_next_job("w1", "embedding-only")
    assert claimed is not None and claimed.spec["simulate"] is True
    assert db.claim_next_job("w1", "embedding-only") is None

    # "both" can run whatever's left.
    claimed2 = db.claim_next_job("w2", "both")
    assert claimed2 is not None and claimed2.spec["simulate"] is False


def test_complete_job_marks_done_and_clears_worker(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    db.enqueue_job(_job())
    job = db.claim_next_job("w1", "both")
    db.start_job(job.job_id, "w1")
    db.complete_job(job.job_id, "w1", {"rows": 5})

    got = db.get_job(job.job_id)
    assert got.state == JOB_DONE
    assert got.result_summary == {"rows": 5}
    worker = db.get_worker("w1")
    assert worker.current_job_id is None


def test_fail_job_requeues_until_max_attempts_then_fails(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    db.enqueue_job(_job(max_attempts=3))

    for expected_state in (JOB_PENDING, JOB_PENDING, JOB_FAILED):
        job = db.claim_next_job("w1", "both")
        assert job is not None, f"expected a job to reclaim before reaching {expected_state}"
        db.start_job(job.job_id, "w1")
        db.fail_job(job.job_id, "w1", "boom")
        got = db.get_job(job.job_id)
        assert got.state == expected_state

    # Once failed, it's terminal - no more claims.
    assert db.claim_next_job("w1", "both") is None


def test_lease_sweep_requeues_stale_claimed_jobs(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    db.enqueue_job(_job())
    job = db.claim_next_job("w1", "both")
    db.start_job(job.job_id, "w1")

    with db._cursor() as cur:
        cur.execute(
            "UPDATE jobs SET heartbeat_at = ? WHERE job_id = ?",
            (time.time() - 1000, job.job_id),
        )

    swept = db.sweep_expired_leases(
        "t1", job_heartbeat_timeout_s=60, worker_heartbeat_timeout_s=999999
    )
    assert swept["requeued_jobs"] == 1

    got = db.get_job(job.job_id)
    assert got.state == JOB_PENDING
    assert got.claimed_by is None


def test_lease_sweep_marks_stale_workers_dead(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    with db._cursor() as cur:
        cur.execute(
            "UPDATE workers SET last_heartbeat_at = ? WHERE worker_id = ?",
            (time.time() - 1000, "w1"),
        )
    swept = db.sweep_expired_leases(
        "t1", job_heartbeat_timeout_s=999999, worker_heartbeat_timeout_s=60
    )
    assert swept["workers_marked_dead"] == 1
    assert db.get_worker("w1").status == "dead"


def test_worker_restart_with_same_id_is_recognized_not_new(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="text", pid=111))
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="text", pid=222))
    workers = db.list_workers("t1")
    assert len(workers) == 1
    assert workers[0].pid == 222
    assert workers[0].status == "alive"


def test_worker_cannot_reregister_under_a_different_task_id(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="text"))
    with pytest.raises(AssertionError):
        db.register_worker(Worker(worker_id="w1", task_id="t2", capability="text"))


def test_heartbeat_touches_current_job(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    db.enqueue_job(_job())
    job = db.claim_next_job("w1", "both")
    db.start_job(job.job_id, "w1")

    before = db.get_job(job.job_id).heartbeat_at
    time.sleep(0.01)
    current_job_id = db.heartbeat_worker("w1")
    after = db.get_job(job.job_id).heartbeat_at

    assert current_job_id == job.job_id
    assert after > before


def test_task_summary_counts_and_all_terminal(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    db.enqueue_job(_job(max_attempts=1))
    db.enqueue_job(_job(max_attempts=1))

    summary = db.task_summary("t1")
    assert summary["total"] == 2 and summary["pending"] == 2 and not summary["all_terminal"]

    j1 = db.claim_next_job("w1", "both")
    db.start_job(j1.job_id, "w1")
    db.complete_job(j1.job_id, "w1", None)

    j2 = db.claim_next_job("w1", "both")
    db.start_job(j2.job_id, "w1")
    db.fail_job(j2.job_id, "w1", "boom")  # max_attempts=1 -> straight to failed

    summary = db.task_summary("t1")
    assert summary["done"] == 1 and summary["failed"] == 1 and summary["all_terminal"]


def _fail_one(db, worker="w1", capability="both"):
    """Claim, start and fail the next job until it is terminal. Returns it."""
    job = None
    while True:
        claimed = db.claim_next_job(worker, capability)
        assert claimed is not None
        job = claimed
        db.start_job(job.job_id, worker)
        db.fail_job(job.job_id, worker, "boom")
        if db.get_job(job.job_id).state == JOB_FAILED:
            return db.get_job(job.job_id)


def test_reset_failed_jobs_clears_the_attempt_counter_not_just_the_state(db):
    """The counter is the point: a job put back to ``pending`` with ``attempt`` still at
    ``max_attempts`` is handed out once and fails terminally on the first hiccup."""
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    db.enqueue_job(_job(max_attempts=3))
    failed = _fail_one(db)
    assert failed.attempt == 3

    reset = db.reset_failed_jobs("t1")
    assert [j.job_id for j in reset] == [failed.job_id]
    # Returned as it *was*, so a caller can report what it put back and why it failed.
    assert reset[0].state == JOB_FAILED and reset[0].error == "boom"

    got = db.get_job(failed.job_id)
    assert got.state == JOB_PENDING and got.attempt == 0
    assert got.claimed_by is None and got.finished_at is None and got.error is None

    # And it really is claimable again, with its full retry budget back.
    assert db.claim_next_job("w1", "both") is not None


def test_reset_failed_jobs_leaves_done_and_pending_jobs_alone(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    db.enqueue_job(_job(max_attempts=1))
    db.enqueue_job(_job(max_attempts=1))
    db.enqueue_job(_job(max_attempts=1))

    done = db.claim_next_job("w1", "both")
    db.start_job(done.job_id, "w1")
    db.complete_job(done.job_id, "w1", None)
    _fail_one(db)

    assert len(db.reset_failed_jobs("t1")) == 1
    summary = db.task_summary("t1")
    assert summary["done"] == 1 and summary["failed"] == 0 and summary["pending"] == 2


def test_reset_failed_jobs_dry_run_selects_without_writing(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    db.enqueue_job(_job(max_attempts=1))
    failed = _fail_one(db)

    preview = db.reset_failed_jobs("t1", dry_run=True)
    assert [j.job_id for j in preview] == [failed.job_id]
    assert db.get_job(failed.job_id).state == JOB_FAILED

    assert [j.job_id for j in db.reset_failed_jobs("t1")] == [j.job_id for j in preview]


def test_reset_failed_jobs_narrows_by_benchmark_and_task(db):
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    db.register_worker(Worker(worker_id="w2", task_id="t2", capability="both"))
    db.enqueue_job(_job(benchmark="movie_random", max_attempts=1))
    db.enqueue_job(_job(benchmark="artwork_random_medium", max_attempts=1))
    db.enqueue_job(_job(task_id="t2", max_attempts=1))
    _fail_one(db)
    _fail_one(db)
    _fail_one(db, worker="w2")

    reset = db.reset_failed_jobs("t1", benchmarks=["movie_random"])
    assert [j.benchmark for j in reset] == ["movie_random"]
    # Another task's failure in the same database is not this task's business.
    assert db.task_summary("t2")["failed"] == 1


def test_job_progress_weights_states_by_query_count(db):
    """The dashboard's headline bar is measured in queries, not jobs: a 40-query job and
    a 4-query job are not the same fraction of an experiment. Producers stamp an exact
    ``n_queries`` on every spec (see ``Benchmark.query_count``) and this is where it is
    summed per state."""
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    db.enqueue_job(_job(spec={"step_idx": 0, "n_queries": 10}, max_attempts=1))
    db.enqueue_job(_job(spec={"step_idx": 1, "n_queries": 4}, max_attempts=1))
    db.enqueue_job(_job(spec={"step_idx": 2, "n_queries": 6}, max_attempts=1))

    claimed = db.claim_next_job("w1", "both")
    db.start_job(claimed.job_id, "w1")
    db.complete_job(claimed.job_id, "w1", None)
    running = db.claim_next_job("w1", "both")
    db.start_job(running.job_id, "w1")

    progress = db.job_progress("t1")
    queries = progress["queries"]
    assert queries["total"] == 20
    assert queries["done"] == 10  # the completed job's queries, not "1 of 3 jobs"
    assert queries["assigned"] == 4  # in flight on a worker
    assert queries["pending"] == 6  # nobody has taken it
    assert queries["exact"] is True
    assert len(progress["jobs"]) == 3
    assert {j["state"] for j in progress["jobs"]} == {"done", "running", "pending"}


def test_job_progress_falls_back_to_one_unit_per_job_without_counts(db):
    """Job specs without ``n_queries`` still render a bar, just a coarser one - flagged
    by ``exact: False`` so the UI can say so."""
    db.enqueue_job(_job(spec={"step_idx": 0}))
    db.enqueue_job(_job(spec={"step_idx": 1}))

    queries = db.job_progress("t1")["queries"]
    assert queries["total"] == 2 and queries["pending"] == 2
    assert queries["exact"] is False


def test_the_spec_carries_the_benchmark_on_the_way_in_and_back_out(db):
    """The benchmark is a configuration axis, so it belongs in the spec beside the others.

    Whether it is *on the spec* rather than only beside it decides one thing: the
    run-configuration panel's linkage rule compares axes within one row, so a dataset
    supplied separately can never be found to be tied to anything, although the operator
    set is a function of the benchmark's modality. Set in ``Job.__post_init__``, so it holds for
    every producer, and on the way back out of the database and off the wire too.
    """
    job = _job(spec={"step_idx": 0})
    assert job.spec["benchmark"] == "movie_random"

    db.enqueue_job(job)
    db.register_worker(Worker(worker_id="w1", task_id="t1", capability="both"))
    claimed = db.claim_next_job("w1", "both")
    assert claimed.spec["benchmark"] == "movie_random"
    assert claimed.to_json()["spec"]["benchmark"] == "movie_random"

    # A stored spec without a benchmark: `from_row` supplies it, so `/api/jobs` is
    # correct without a migration.
    with db._cursor() as cur:
        cur.execute("UPDATE jobs SET spec_json = ?", ('{"step_idx": 0}',))
    assert db.list_jobs("t1")[0].spec["benchmark"] == "movie_random"
