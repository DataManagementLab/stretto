"""Coordinator HTTP surface: job queue + worker registry routes, layered onto the
monitor dashboard.

``create_coordinator_app`` calls ``reasondb.monitor.server.create_app`` to get the
base Flask app - ``/``, ``/static/*``, ``/status``, ``/api/run``, ``/api/events``,
``/api/aggregates``, ``/api/results/*`` - which here describes the coordinator's own
``Collector`` (fed via ``/api/ingest``, see ``reasondb.coordinator.ingest``), and
registers the job/worker routes onto that same app object. The routes are a thin
layer over the database: all policy lives in ``db.py`` (state machine) and
``scheduler.py`` (capability matching).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request

from reasondb.coordinator.db import JobDB
from reasondb.coordinator.ingest import ingest_events
from reasondb.coordinator.models import Worker
from reasondb.monitor.collector import Collector
from reasondb.monitor.server import create_app as create_monitor_app

logger = logging.getLogger(__name__)


def create_coordinator_app(
    task_id: str,
    job_db: JobDB,
    collector: Collector,
    result_roots: List[Path],
    run_info: Optional[Dict[str, Any]] = None,
) -> Flask:
    app = create_monitor_app(collector, result_roots, run_info)

    def _require(body: Optional[dict], *fields: str):
        body = dict(body or {})
        missing = [f for f in fields if f not in body]
        if missing:
            return body, (jsonify({"error": f"missing required field(s): {missing}"}), 400)
        return body, None

    # ── worker registry ─────────────────────────────────────────────────────

    @app.post("/api/workers/register")
    def api_worker_register():
        body, err = _require(
            request.get_json(silent=True), "worker_id", "task_id", "capability"
        )
        if err:
            return err
        if body["task_id"] != task_id:
            return jsonify(
                {
                    "error": (
                        f"this coordinator is running task_id={task_id!r}; worker "
                        f"registered for {body['task_id']!r}. Pass the right --task-id."
                    )
                }
            ), 409
        try:
            job_db.register_worker(
                Worker(
                    worker_id=body["worker_id"],
                    task_id=body["task_id"],
                    capability=body["capability"],
                    hostname=body.get("hostname"),
                    device=body.get("device"),
                    worker_dir=body.get("worker_dir"),
                    pid=body.get("pid"),
                    meta=body.get("meta") or {},
                )
            )
        except AssertionError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"ok": True, "worker_id": body["worker_id"]})

    @app.post("/api/workers/<worker_id>/heartbeat")
    def api_worker_heartbeat(worker_id: str):
        current_job_id = job_db.heartbeat_worker(worker_id)
        return jsonify({"ok": True, "current_job_id": current_job_id})

    @app.get("/api/workers")
    def api_workers_list():
        workers = job_db.list_workers(request.args.get("task_id", task_id))
        return jsonify({"workers": [w.to_json() for w in workers]})

    # ── job queue ────────────────────────────────────────────────────────────

    @app.post("/api/jobs/claim")
    def api_jobs_claim():
        body, err = _require(request.get_json(silent=True), "worker_id", "capability")
        if err:
            return err
        job = job_db.claim_next_job(body["worker_id"], body["capability"])
        return jsonify({"job": job.to_json() if job else None})

    @app.post("/api/jobs/<job_id>/start")
    def api_jobs_start(job_id: str):
        body, err = _require(request.get_json(silent=True), "worker_id")
        if err:
            return err
        job_db.start_job(job_id, body["worker_id"])
        return jsonify({"ok": True})

    @app.post("/api/jobs/<job_id>/complete")
    def api_jobs_complete(job_id: str):
        body, err = _require(request.get_json(silent=True), "worker_id")
        if err:
            return err
        job_db.complete_job(job_id, body["worker_id"], body.get("result_summary"))
        return jsonify({"ok": True})

    @app.post("/api/jobs/<job_id>/fail")
    def api_jobs_fail(job_id: str):
        body, err = _require(request.get_json(silent=True), "worker_id", "error")
        if err:
            return err
        job_db.fail_job(job_id, body["worker_id"], body["error"])
        return jsonify({"ok": True})

    @app.get("/api/jobs")
    def api_jobs_list():
        jobs = job_db.list_jobs(
            request.args.get("task_id", task_id), state=request.args.get("state")
        )
        return jsonify({"jobs": [j.to_json() for j in jobs]})

    @app.get("/api/task/<tid>/summary")
    def api_task_summary(tid: str):
        return jsonify(job_db.task_summary(tid))

    @app.get("/api/task/<tid>/progress")
    def api_task_progress(tid: str):
        """Job-queue state joined with what the workers have actually reported.

        The queue knows how many queries each job covers and who holds it; the
        collector knows how far into its current job each worker has got. Neither alone
        can say "x of y queries done across the whole experiment", which is the one
        number the dashboard leads with - so the join happens here, once, rather than
        being reassembled in the browser from two endpoints that can disagree.
        """
        payload = job_db.job_progress(tid)
        payload["summary"] = job_db.task_summary(tid)

        # Per-worker progress within the job it currently holds.
        by_worker = {
            w.get("worker_id"): w
            for w in collector.snapshot_run().get("workers", [])
            if w.get("worker_id")
        }
        in_progress = 0
        for row in payload["jobs"]:
            reported = by_worker.get(row["claimed_by"])
            if reported is None or reported.get("job_id") != row["job_id"]:
                row["queries_done"] = 0
                continue
            row["queries_done"] = reported.get("queries_done_in_job") or 0
            if row["state"] in ("claimed", "running"):
                in_progress += row["queries_done"]
        # Queries finished inside jobs that are still assigned: they belong on the
        # "done" side of the bar even though their job has not completed yet.
        queries = payload["queries"]
        queries["in_progress_done"] = min(in_progress, queries["assigned"])
        queries["assigned"] = max(0, queries["assigned"] - queries["in_progress_done"])
        queries["done"] = queries["done"] + queries["in_progress_done"]
        return jsonify(payload)

    # ── telemetry ingest ─────────────────────────────────────────────────────

    @app.post("/api/ingest")
    def api_ingest():
        body, err = _require(request.get_json(silent=True), "worker_id", "events")
        if err:
            return err
        accepted = ingest_events(
            collector, body["worker_id"], body.get("job_id"), body["events"]
        )
        return jsonify({"ok": True, "accepted": accepted})

    return app


def run_lease_sweep_forever(
    job_db: JobDB,
    task_id: str,
    stop_event,
    interval_s: float = 15.0,
    job_heartbeat_timeout_s: float = 300.0,
    worker_heartbeat_timeout_s: float = 120.0,
) -> None:
    """Background-thread body: periodically requeue stale jobs / mark dead workers, and
    fail whatever is stuck behind a phase that will never complete.

    Conservative defaults: a worker heartbeats every ~15s, so a 120s worker-dead
    threshold tolerates several missed beats before giving up on it;
    a 300s job-heartbeat threshold tolerates the same plus doesn't punish a job that's
    genuinely mid-step with no natural progress event to piggyback on.

    The blocked-job cascade rides along here rather than getting its own thread: it is
    one query, it must run *after* the sweep that may have just requeued the last retry
    of a blocking job, and this thread already treats its own failure as non-fatal.
    """
    while not stop_event.is_set():
        try:
            result = job_db.sweep_expired_leases(
                task_id, job_heartbeat_timeout_s, worker_heartbeat_timeout_s
            )
            if result["requeued_jobs"] or result["workers_marked_dead"]:
                logger.info(
                    "Coordinator: lease sweep requeued %d job(s), marked %d worker(s) dead.",
                    result["requeued_jobs"],
                    result["workers_marked_dead"],
                )
            job_db.fail_blocked_jobs(task_id)
        except Exception as exc:  # never let the sweep thread's death stall the task
            logger.warning("Coordinator: lease sweep failed: %s", exc)
        stop_event.wait(interval_s)
