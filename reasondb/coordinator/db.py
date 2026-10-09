"""SQLite-backed job queue and worker registry. The only module that touches sqlite3.

Deliberately single-writer: only the coordinator process ever opens this database, and
it's opened from one thread at a time per connection (Flask's request thread and the
lease-sweep background thread each get their own connection via :meth:`JobDB.connect`,
serialized by ``check_same_thread=False`` + a module-level lock - see ``_LOCK``). This
sidesteps SQLite's well-known multi-writer-over-NFS fragility: there is exactly one
writer, ever, by construction (workers only ever talk HTTP to the coordinator, never
touch this file). ``journal_mode=DELETE`` (not WAL) is used on purpose - WAL's
``-wal``/``-shm`` shared-memory files are the part most likely to misbehave on an
NFS-like shared filesystem, and DELETE-mode's plain rollback journal is the
conservative, well-tested choice for this workload's write volume (job-state
transitions, not per-telemetry-event writes).
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from reasondb.coordinator.models import (
    JOB_CLAIMED,
    JOB_DONE,
    JOB_FAILED,
    JOB_PENDING,
    JOB_RUNNING,
    WORKER_ALIVE,
    WORKER_DEAD,
    Job,
    Worker,
)
from reasondb.coordinator.scheduler import pick_next_job

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id              TEXT PRIMARY KEY,
    task_id             TEXT NOT NULL,
    producer            TEXT NOT NULL,
    benchmark           TEXT NOT NULL,
    split               TEXT NOT NULL,
    spec_json           TEXT NOT NULL,
    required_caps_json  TEXT NOT NULL,
    output_dir          TEXT NOT NULL,
    priority            INTEGER NOT NULL DEFAULT 0,
    phase               INTEGER NOT NULL DEFAULT 0,
    state               TEXT NOT NULL DEFAULT 'pending'
                          CHECK (state IN ('pending','claimed','running','done','failed')),
    claimed_by          TEXT,
    attempt             INTEGER NOT NULL DEFAULT 0,
    max_attempts        INTEGER NOT NULL DEFAULT 3,
    created_at          REAL NOT NULL,
    claimed_at          REAL,
    heartbeat_at        REAL,
    finished_at         REAL,
    error               TEXT,
    result_summary_json TEXT
);
-- Which (job, label set) pairs the scorer has already turned into metrics.
--
-- Persisted (rather than kept in memory) so a restarted coordinator does not re-score
-- every finished job or re-emit accuracy telemetry that was already reported.
--
-- Cleared by claim_next_job, so a re-run job is scored again; see the note there.
CREATE TABLE IF NOT EXISTS scored_jobs (
    job_id     TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    label_name TEXT NOT NULL,
    scored_at  REAL NOT NULL,
    PRIMARY KEY (job_id, label_name)
);
CREATE TABLE IF NOT EXISTS workers (
    worker_id         TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL,
    capability        TEXT NOT NULL,
    hostname          TEXT,
    device            TEXT,
    worker_dir        TEXT,
    pid               INTEGER,
    status            TEXT NOT NULL DEFAULT 'alive' CHECK (status IN ('alive','dead')),
    registered_at     REAL NOT NULL,
    last_heartbeat_at REAL NOT NULL,
    current_job_id    TEXT,
    meta_json         TEXT
);
"""

#: Indexes, applied *after* the migrations below rather than as part of ``SCHEMA``.
#: An index over a migrated column cannot be created until that column exists, and on a
#: database written before the migration it does not - so creating them together fails
#: to open exactly the old databases the migration is for.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_jobs_task_state ON jobs(task_id, state, priority);
CREATE INDEX IF NOT EXISTS idx_jobs_task_phase ON jobs(task_id, state, phase);
CREATE INDEX IF NOT EXISTS idx_jobs_claimed_by ON jobs(claimed_by);
CREATE INDEX IF NOT EXISTS idx_workers_task ON workers(task_id, status);
"""

#: ``column -> DDL`` for columns that may be missing from an existing database.
#:
#: ``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already exists, and a
#: coordinator resumes an existing ``coordinator.db`` without re-enumerating (see
#: ``run_coordinator``), so such columns are added explicitly. Defaults are neutral:
#: jobs without a phase land in phase 0, i.e. no barrier at all.
_JOBS_MIGRATIONS = {
    "phase": "ALTER TABLE jobs ADD COLUMN phase INTEGER NOT NULL DEFAULT 0",
}

# One process-wide lock serializing all writers to the sqlite3 connection. Flask
# serves requests on multiple threads (``threaded=True``, see monitor/server.py); the
# lease-sweep runs on its own background thread. sqlite3 connections aren't safe to
# share across threads without either this or a connection-per-thread pool - a single
# lock is simpler and the write volume here (job state transitions) is nowhere near
# where lock contention would matter.
_LOCK = threading.Lock()


class JobDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        with _LOCK:
            self._conn.execute("PRAGMA journal_mode=DELETE")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._migrate_jobs()
            self._conn.executescript(INDEXES)

    def _migrate_jobs(self) -> None:
        existing = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        for column, ddl in _JOBS_MIGRATIONS.items():
            if column not in existing:
                logger.info("JobDB: migrating %s - adding jobs.%s", self.db_path, column)
                self._conn.execute(ddl)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with _LOCK:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE")
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    # ── jobs: writes ──────────────────────────────────────────────────────────

    def enqueue_job(self, job: Job) -> None:
        if not job.job_id:
            job.job_id = f"{job.task_id}-{job.producer}-{uuid.uuid4().hex[:12]}"
        if not job.created_at:
            job.created_at = time.time()
        row = job.to_row()
        with self._cursor() as cur:
            cur.execute(
                f"INSERT INTO jobs ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                list(row.values()),
            )

    def enqueue_jobs(self, jobs: Sequence[Job]) -> None:
        for job in jobs:
            self.enqueue_job(job)

    def claim_next_job(self, worker_id: str, worker_capability: str) -> Optional[Job]:
        """Atomically pick and claim one pending job this worker's capability can run.

        Filters to this worker's ``task_id`` implicitly via a join on the workers
        table row that ``register_worker`` already created - a worker can only ever
        claim jobs for the task it registered under.

        The phase barrier is applied here, in SQL, rather than in ``scheduler.py``:
        whether a job is *available* is a property of the queue as a whole, and the
        subquery has to be evaluated in the same transaction as the claim or two workers
        can both read "phase 0 is finishing" and both jump to phase 1. Capability
        matching stays in the scheduler because it is a property of the worker, and is
        the only part that varies per caller.
        """
        with self._cursor() as cur:
            worker_row = cur.execute(
                "SELECT task_id FROM workers WHERE worker_id = ?", (worker_id,)
            ).fetchone()
            assert worker_row is not None, (
                f"Worker {worker_id!r} must POST /api/workers/register before claiming jobs."
            )
            task_id = worker_row["task_id"]
            candidates = [
                Job.from_row(r)
                for r in cur.execute(
                    # MIN over non-terminal jobs (which includes pending): while any
                    # phase-0 job is unfinished the barrier sits at 0, and it lifts to 1
                    # the moment the last one reaches done or failed.
                    "SELECT * FROM jobs WHERE task_id = ? AND state = ? "
                    "  AND phase <= (SELECT COALESCE(MIN(phase), 0) FROM jobs "
                    "                WHERE task_id = ? AND state NOT IN (?, ?)) "
                    "ORDER BY phase ASC, priority ASC, created_at ASC",
                    (task_id, JOB_PENDING, task_id, JOB_DONE, JOB_FAILED),
                ).fetchall()
            ]
            job = pick_next_job(candidates, worker_capability)
            if job is None:
                return None
            now = time.time()
            cur.execute(
                "UPDATE jobs SET state = ?, claimed_by = ?, claimed_at = ?, "
                "heartbeat_at = ?, attempt = attempt + 1 WHERE job_id = ?",
                (JOB_CLAIMED, worker_id, now, now, job.job_id),
            )
            # A job about to run again has not been scored - whatever it produced before
            # is about to be replaced. In the same transaction as the claim so the two
            # cannot disagree, and here rather than in fail_job/requeue because *every*
            # path back to running goes through a claim: a retry after a crash, a
            # lease-sweep reclaim, and a manual `UPDATE jobs SET state='pending'` alike.
            cur.execute("DELETE FROM scored_jobs WHERE job_id = ?", (job.job_id,))
            cur.execute(
                "UPDATE workers SET current_job_id = ? WHERE worker_id = ?",
                (job.job_id, worker_id),
            )
            job.state, job.claimed_by, job.claimed_at, job.heartbeat_at = (
                JOB_CLAIMED,
                worker_id,
                now,
                now,
            )
            job.attempt += 1
            return job

    def start_job(self, job_id: str, worker_id: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE jobs SET state = ?, heartbeat_at = ? "
                "WHERE job_id = ? AND claimed_by = ? AND state = ?",
                (JOB_RUNNING, time.time(), job_id, worker_id, JOB_CLAIMED),
            )
            if cur.rowcount == 0:
                # Not a hard error: the lease sweep can legitimately have already
                # reclaimed this job out from under a slow-to-report worker - a real
                # distributed-systems race, not a violated invariant. Logged so it's
                # visible, not silently swallowed.
                logger.warning(
                    "start_job(%s, %s): 0 rows updated - job not claimed by this "
                    "worker (already reclaimed by a lease-sweep timeout?).",
                    job_id, worker_id,
                )

    def backfill_n_queries(
        self, task_id: str, benchmark: str, split: str, n_queries: int
    ) -> int:
        """Stamp the real query count on jobs enqueued before it was knowable.

        A random benchmark's query set does not exist until its phase-0 filter-stats job
        draws it, so every later job of that benchmark is enumerated with
        ``n_queries: None`` - enumeration must not generate the queries itself, or the
        set it invented would become the authoritative one. The monitor then weights each
        such job as a single unit, which makes a 40-query job look like a 4-query one.

        Read-modify-write in Python rather than SQLite's JSON1: this runs once per
        benchmark per task over a handful of rows, and ``db.py`` would otherwise depend
        on a compile-time-optional extension.
        """
        updated = 0
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT job_id, spec_json FROM jobs WHERE task_id = ? AND benchmark = ? "
                "AND split = ?",
                (task_id, benchmark, split),
            ).fetchall()
            for row in rows:
                spec = json.loads(row["spec_json"])
                if spec.get("n_queries") is not None or spec.get("kind") == "filter_stats":
                    continue
                spec["n_queries"] = n_queries
                cur.execute(
                    "UPDATE jobs SET spec_json = ? WHERE job_id = ?",
                    (json.dumps(spec), row["job_id"]),
                )
                updated += 1
        if updated:
            logger.info(
                "Task %s: back-filled n_queries=%d onto %d %s/%s job(s).",
                task_id, n_queries, updated, benchmark, split,
            )
        return updated

    def complete_job(
        self, job_id: str, worker_id: str, result_summary: Optional[Dict[str, Any]]
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE jobs SET state = ?, finished_at = ?, result_summary_json = ?, "
                "claimed_by = NULL WHERE job_id = ? AND claimed_by = ?",
                (JOB_DONE, time.time(), json.dumps(result_summary), job_id, worker_id),
            )
            if cur.rowcount == 0:
                logger.warning(
                    "complete_job(%s, %s): 0 rows updated - job not claimed by this "
                    "worker (already reclaimed by a lease-sweep timeout?).",
                    job_id, worker_id,
                )
            cur.execute(
                "UPDATE workers SET current_job_id = NULL "
                "WHERE worker_id = ? AND current_job_id = ?",
                (worker_id, job_id),
            )
            job_row = cur.execute(
                "SELECT task_id, benchmark, split, spec_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()

        # Outside the cursor above: backfill_n_queries opens its own transaction, and
        # _cursor's lock is not reentrant.
        n_queries = (result_summary or {}).get("n_queries")
        if job_row is not None and n_queries is not None:
            if json.loads(job_row["spec_json"]).get("kind") == "filter_stats":
                self.backfill_n_queries(
                    job_row["task_id"],
                    job_row["benchmark"],
                    job_row["split"],
                    int(n_queries),
                )

    def fail_job(self, job_id: str, worker_id: str, error: str) -> None:
        """``running``/``claimed`` -> ``pending`` if attempts remain, else ``failed``."""
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT attempt, max_attempts FROM jobs WHERE job_id = ? AND claimed_by = ?",
                (job_id, worker_id),
            ).fetchone()
            if row is None:
                return
            next_state = JOB_PENDING if row["attempt"] < row["max_attempts"] else JOB_FAILED
            claimed_by = None if next_state == JOB_PENDING else worker_id
            cur.execute(
                "UPDATE jobs SET state = ?, error = ?, claimed_by = ?, "
                "finished_at = CASE WHEN ? = ? THEN ? ELSE finished_at END "
                "WHERE job_id = ?",
                (
                    next_state,
                    error,
                    claimed_by,
                    next_state,
                    JOB_FAILED,
                    time.time(),
                    job_id,
                ),
            )
            cur.execute(
                "UPDATE workers SET current_job_id = NULL "
                "WHERE worker_id = ? AND current_job_id = ?",
                (worker_id, job_id),
            )

    def mark_job_scored(self, job_id: str, label_name: str) -> None:
        """Record that ``job_id`` has been scored against ``label_name``.

        ``OR REPLACE`` rather than ``OR IGNORE`` so a re-score refreshes ``scored_at``:
        the marker means "scored, as of then", and a stale timestamp on a job that was
        re-run and re-scored would misdescribe it.
        """
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO scored_jobs (job_id, label_name, scored_at) "
                "VALUES (?, ?, ?)",
                (job_id, label_name, time.time()),
            )

    def list_scored(self, task_id: str) -> List[Tuple[str, str]]:
        """Every ``(job_id, label_name)`` already scored for ``task_id``.

        Joined against ``jobs`` rather than filtered in Python because the markers carry
        no task of their own - one coordinator database holds one task today, but nothing
        in the schema says so, and the scorer asks per task.
        """
        with self._cursor() as cur:
            return [
                (row["job_id"], row["label_name"])
                for row in cur.execute(
                    "SELECT s.job_id, s.label_name FROM scored_jobs s "
                    "JOIN jobs j ON j.job_id = s.job_id WHERE j.task_id = ?",
                    (task_id,),
                ).fetchall()
            ]

    def reset_failed_jobs(
        self,
        task_id: str,
        job_ids: Optional[Sequence[str]] = None,
        benchmarks: Optional[Sequence[str]] = None,
        dry_run: bool = False,
    ) -> List[Job]:
        """Put ``failed`` jobs back to ``pending`` with their attempt count at zero.

        ``failed`` is terminal only because the attempts ran out: ``fail_job`` returns a
        job to ``pending`` while ``attempt < max_attempts`` and gives up after that, and
        ``fail_blocked_jobs`` fails everything behind a phase that never settled. Neither
        is a statement that the work cannot succeed - a dead server or a full disk
        land here too - so re-running the task means clearing the counter, not
        just the state: flipping ``state`` alone hands the job out once and fails it again
        on the first hiccup.

        ``error``/``finished_at``/``claimed_*``/``heartbeat_at`` are cleared with it, so a
        reset job is indistinguishable from one that never ran. The ``scored_jobs`` markers
        are deliberately *not* touched here - ``claim_next_job`` deletes them as it hands a
        job out, which is the one place every route back to running passes through.

        Blocked jobs come back with everything else, and that is why the selection is
        all-failed-by-default: reset a phase-1 job while the phase-0 job it waits on is
        still ``failed`` and the coordinator's ``fail_blocked_jobs`` simply fails it again.
        ``job_ids``/``benchmarks`` narrow it for the case where that is understood.

        ``dry_run`` runs the selection and skips the write, so a preview cannot describe a
        different set of jobs than the write would touch.

        :returns: the jobs as they were *before* the reset, so a caller can report what it
            put back and why each had failed.
        """
        query = "SELECT * FROM jobs WHERE task_id = ? AND state = ?"
        params: List[Any] = [task_id, JOB_FAILED]
        if job_ids is not None:
            assert job_ids, "reset_failed_jobs: job_ids was empty; pass None to select every failed job."
            query += f" AND job_id IN ({','.join('?' * len(job_ids))})"
            params.extend(job_ids)
        if benchmarks is not None:
            assert benchmarks, "reset_failed_jobs: benchmarks was empty; pass None for every benchmark."
            query += f" AND benchmark IN ({','.join('?' * len(benchmarks))})"
            params.extend(benchmarks)
        query += " ORDER BY phase ASC, priority ASC, created_at ASC"
        with self._cursor() as cur:
            jobs = [Job.from_row(row) for row in cur.execute(query, params).fetchall()]
            if jobs and not dry_run:
                cur.executemany(
                    "UPDATE jobs SET state = ?, attempt = 0, claimed_by = NULL, "
                    "claimed_at = NULL, heartbeat_at = NULL, finished_at = NULL, "
                    "error = NULL WHERE job_id = ?",
                    [(JOB_PENDING, job.job_id) for job in jobs],
                )
        if jobs and not dry_run:
            logger.info(
                "Task %s: reset %d failed job(s) to pending with attempt=0.",
                task_id, len(jobs),
            )
        return jobs

    def fail_blocked_jobs(self, task_id: str) -> int:
        """Fail everything waiting on a phase that will never complete.

        Without this a task hangs rather than ends: a phase-0 job that exhausts its
        attempts leaves every later job ``pending`` forever, so ``all_terminal`` never
        turns true, the coordinator never merges, and every worker spins in its no-job
        backoff. The jobs are not runnable either - a sweep whose filter stats failed has
        no query set to sweep over.

        Only fires once the blocking phase is *settled*: ``fail_job`` returns a job to
        ``pending`` until its attempts run out, so a job still retrying keeps the barrier
        down and its dependents waiting, which is the intended behavior.

        :returns: how many jobs were failed.
        """
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT MIN(phase) AS p FROM jobs WHERE task_id = ? AND state = ?",
                (task_id, JOB_FAILED),
            ).fetchone()
            failed_phase = row["p"] if row else None
            if failed_phase is None:
                return 0
            unsettled = cur.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE task_id = ? AND phase <= ? "
                "AND state NOT IN (?, ?)",
                (task_id, failed_phase, JOB_DONE, JOB_FAILED),
            ).fetchone()["n"]
            if unsettled:
                return 0
            cur.execute(
                "UPDATE jobs SET state = ?, finished_at = ?, error = ? "
                "WHERE task_id = ? AND phase > ? AND state = ?",
                (
                    JOB_FAILED,
                    time.time(),
                    f"blocked: phase {failed_phase} did not complete",
                    task_id,
                    failed_phase,
                    JOB_PENDING,
                ),
            )
            n = cur.rowcount
        if n:
            logger.warning(
                "Task %s: phase %s did not complete; failed %d blocked job(s) so the "
                "task can terminate rather than wait forever.",
                task_id, failed_phase, n,
            )
        return n

    def current_phase(self, task_id: str) -> int:
        """The phase jobs are claimable at right now - the barrier's position."""
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT COALESCE(MIN(phase), 0) AS p FROM jobs "
                "WHERE task_id = ? AND state NOT IN (?, ?)",
                (task_id, JOB_DONE, JOB_FAILED),
            ).fetchone()
        return int(row["p"]) if row else 0

    # ── jobs: reads ───────────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._cursor() as cur:
            row = cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return Job.from_row(row) if row else None

    def list_jobs(self, task_id: str, state: Optional[str] = None) -> List[Job]:
        with self._cursor() as cur:
            if state:
                rows = cur.execute(
                    "SELECT * FROM jobs WHERE task_id = ? AND state = ? "
                    "ORDER BY priority ASC, created_at ASC",
                    (task_id, state),
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT * FROM jobs WHERE task_id = ? ORDER BY priority ASC, created_at ASC",
                    (task_id,),
                ).fetchall()
            return [Job.from_row(r) for r in rows]

    def task_summary(self, task_id: str) -> Dict[str, Any]:
        with self._cursor() as cur:
            counts = {
                r["state"]: r["n"]
                for r in cur.execute(
                    "SELECT state, COUNT(*) AS n FROM jobs WHERE task_id = ? GROUP BY state",
                    (task_id,),
                ).fetchall()
            }
            total = sum(counts.values())
            workers_alive = cur.execute(
                "SELECT COUNT(*) AS n FROM workers WHERE task_id = ? AND status = ?",
                (task_id, WORKER_ALIVE),
            ).fetchone()["n"]
            workers_dead = cur.execute(
                "SELECT COUNT(*) AS n FROM workers WHERE task_id = ? AND status = ?",
                (task_id, WORKER_DEAD),
            ).fetchone()["n"]
            phase = cur.execute(
                "SELECT COALESCE(MIN(phase), 0) AS p FROM jobs "
                "WHERE task_id = ? AND state NOT IN (?, ?)",
                (task_id, JOB_DONE, JOB_FAILED),
            ).fetchone()["p"]
            blocked = cur.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE task_id = ? AND state = ? "
                "AND phase > ?",
                (task_id, JOB_PENDING, phase),
            ).fetchone()["n"]
        return {
            "task_id": task_id,
            "total": total,
            # Where the barrier sits, and how many pending jobs are behind it. Without
            # these "nothing is running and nothing is finishing" is indistinguishable
            # from a stuck queue.
            "phase": int(phase),
            "blocked": blocked,
            "pending": counts.get(JOB_PENDING, 0),
            "claimed": counts.get(JOB_CLAIMED, 0),
            "running": counts.get(JOB_RUNNING, 0),
            "done": counts.get(JOB_DONE, 0),
            "failed": counts.get(JOB_FAILED, 0),
            "workers_alive": workers_alive,
            "workers_dead": workers_dead,
            "all_terminal": total > 0
            and (counts.get(JOB_DONE, 0) + counts.get(JOB_FAILED, 0)) == total,
        }

    def job_progress(self, task_id: str) -> Dict[str, Any]:
        """Every job with the query count it will run, plus query-level state totals.

        Jobs are what the queue schedules, but queries are what a run is measured in and
        what takes the time - a 40-query job and a 4-query job are not the same fifth of
        an experiment. Producers put an exact ``n_queries`` on every job's spec (see
        ``Benchmark.query_count``), so summing it per state gives the monitor a progress
        bar over the whole experiment: finished queries, queries inside a job some
        worker has taken, and queries nobody has started.

        Jobs without an ``n_queries`` report ``None`` and are counted as one unit each.
        """
        jobs = self.list_jobs(task_id)
        rows: List[Dict[str, Any]] = []
        done_q = assigned_q = pending_q = failed_q = 0
        for job in jobs:
            n_queries = job.spec.get("n_queries")
            weight = int(n_queries) if isinstance(n_queries, int) and n_queries > 0 else 1
            if job.state == JOB_DONE:
                done_q += weight
            elif job.state == JOB_FAILED:
                failed_q += weight
            elif job.state in (JOB_CLAIMED, JOB_RUNNING):
                assigned_q += weight
            else:
                pending_q += weight
            rows.append(
                {
                    "job_id": job.job_id,
                    "state": job.state,
                    "producer": job.producer,
                    "benchmark": job.benchmark,
                    "split": job.split,
                    "spec": job.spec,
                    "n_queries": n_queries,
                    "priority": job.priority,
                    "phase": job.phase,
                    "attempt": job.attempt,
                    "max_attempts": job.max_attempts,
                    "claimed_by": job.claimed_by,
                    "created_at": job.created_at,
                    "claimed_at": job.claimed_at,
                    "heartbeat_at": job.heartbeat_at,
                    "finished_at": job.finished_at,
                    "error": job.error,
                    "output_dir": job.output_dir,
                }
            )
        return {
            "task_id": task_id,
            "jobs": rows,
            "queries": {
                "total": done_q + failed_q + assigned_q + pending_q,
                "done": done_q,
                "failed": failed_q,
                "assigned": assigned_q,
                "pending": pending_q,
                # True when every job carried a real count, i.e. the totals above are
                # query counts rather than a mix of counts and one-per-job fallbacks.
                "exact": all(
                    isinstance(j.spec.get("n_queries"), int) for j in jobs
                ) and bool(jobs),
            },
        }

    # ── workers ───────────────────────────────────────────────────────────────

    def register_worker(self, worker: Worker) -> None:
        """Upsert: a worker restarting with the same ``worker_id`` is recognized as
        "back", not new (see ``models.Worker`` docstring)."""
        now = time.time()
        worker.registered_at = worker.registered_at or now
        worker.last_heartbeat_at = now
        worker.status = WORKER_ALIVE
        row = worker.to_row()
        with self._cursor() as cur:
            existing = cur.execute(
                "SELECT task_id FROM workers WHERE worker_id = ?", (worker.worker_id,)
            ).fetchone()
            if existing is not None:
                assert existing["task_id"] == worker.task_id, (
                    f"Worker {worker.worker_id!r} previously registered under task "
                    f"{existing['task_id']!r}; refusing to reassign it to "
                    f"{worker.task_id!r} (409: use a different --worker-id)."
                )
            cur.execute(
                "INSERT INTO workers ({cols}) VALUES ({ph}) "
                "ON CONFLICT(worker_id) DO UPDATE SET "
                "capability=excluded.capability, hostname=excluded.hostname, "
                "device=excluded.device, worker_dir=excluded.worker_dir, "
                "pid=excluded.pid, status=excluded.status, "
                "last_heartbeat_at=excluded.last_heartbeat_at, meta_json=excluded.meta_json".format(
                    cols=",".join(row), ph=",".join("?" * len(row))
                ),
                list(row.values()),
            )

    def heartbeat_worker(self, worker_id: str) -> Optional[str]:
        """Bump the worker's heartbeat and, if it has a current job, that job's too.

        Returns the worker's ``current_job_id`` (or ``None``). This is the *only*
        heartbeat path - there is deliberately no separate per-job heartbeat route, so
        there's exactly one mechanism to keep in sync.
        """
        now = time.time()
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT current_job_id FROM workers WHERE worker_id = ?", (worker_id,)
            ).fetchone()
            assert row is not None, f"Unknown worker_id {worker_id!r}; register first."
            cur.execute(
                "UPDATE workers SET last_heartbeat_at = ?, status = ? WHERE worker_id = ?",
                (now, WORKER_ALIVE, worker_id),
            )
            current_job_id = row["current_job_id"]
            if current_job_id is not None:
                cur.execute(
                    "UPDATE jobs SET heartbeat_at = ? WHERE job_id = ? AND claimed_by = ?",
                    (now, current_job_id, worker_id),
                )
            return current_job_id

    def get_worker(self, worker_id: str) -> Optional[Worker]:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT * FROM workers WHERE worker_id = ?", (worker_id,)
            ).fetchone()
            return Worker.from_row(row) if row else None

    def list_workers(self, task_id: str) -> List[Worker]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM workers WHERE task_id = ? ORDER BY registered_at ASC",
                (task_id,),
            ).fetchall()
            return [Worker.from_row(r) for r in rows]

    # ── lease sweep ───────────────────────────────────────────────────────────

    def sweep_expired_leases(
        self,
        task_id: str,
        job_heartbeat_timeout_s: float,
        worker_heartbeat_timeout_s: float,
    ) -> Dict[str, int]:
        """Requeue jobs whose claiming worker's heartbeat has gone stale, and mark
        stale workers dead. Idempotent; safe to call on a timer from a background
        thread (see ``scripts/run_coordinator.py``).
        """
        now = time.time()
        with self._cursor() as cur:
            stale_jobs = cur.execute(
                "SELECT job_id, claimed_by, attempt, max_attempts FROM jobs "
                "WHERE task_id = ? AND state IN (?, ?) AND heartbeat_at < ?",
                (task_id, JOB_CLAIMED, JOB_RUNNING, now - job_heartbeat_timeout_s),
            ).fetchall()
            requeued = 0
            for row in stale_jobs:
                next_state = (
                    JOB_PENDING if row["attempt"] < row["max_attempts"] else JOB_FAILED
                )
                cur.execute(
                    "UPDATE jobs SET state = ?, claimed_by = ?, error = ? WHERE job_id = ?",
                    (
                        next_state,
                        None if next_state == JOB_PENDING else row["claimed_by"],
                        "lease expired",
                        row["job_id"],
                    ),
                )
                requeued += 1
            dead = cur.execute(
                "UPDATE workers SET status = ? "
                "WHERE task_id = ? AND status = ? AND last_heartbeat_at < ?",
                (
                    WORKER_DEAD,
                    task_id,
                    WORKER_ALIVE,
                    now - worker_heartbeat_timeout_s,
                ),
            )
        return {"requeued_jobs": requeued, "workers_marked_dead": dead.rowcount}
