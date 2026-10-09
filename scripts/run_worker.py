"""Worker for a distributed benchmark sweep: registers with a coordinator, starts the
local backend servers its ``--capability`` implies, then loops claiming and running jobs
until the task drains - at which point it moves on to the next ``--tasks`` entry, and
exits once the last one has drained.

``--worker-id`` is stable across restarts - passed explicitly on the CLI (not
generated) so a crashed worker can be relaunched with the same id and be recognized as
"back" rather than a brand-new worker (see ``reasondb.coordinator.models.Worker``).

Example
-------
    # Backgrounds itself and survives an SSH disconnect on its own (see
    # logging_utils.daemonize) - no nohup/redirection/mkdir needed on the command line.
    python scripts/run_worker.py \\
        --tasks sweep01=http://coordinator-a:5099 sweep02=http://coordinator-b:5099 \\
        --worker-id worker-03 --capability embedding-only --device cuda:0 &

One coordinator process serves exactly one task, so a task and its URL are named
together. The list is served in order but *cycled* rather than drained one entry at a
time: a worker leaves a task the moment the coordinator has nothing it can claim, and
comes back to it on the next pass. This keeps workers busy at the tail of a task, where
the queue empties long before the last straggler job finishes. Leaving early is only safe
*because* it comes back: a straggler can fail and requeue, and jobs behind a phase
barrier become claimable when it lifts. A coordinator that is not up yet reports no work
and is simply revisited (connection errors are retried forever; an HTTP error - a 409 for
the wrong task id, say - still kills the worker loudly). The worker exits when every task
reports all-terminal.

--worker-dir is a *template* resolved per task, defaulting to
benchmark_results/{task_id}/workers/{worker_id} (nested under the same tree as the
coordinator's own --output-dir default, so every worker's artifacts show up alongside the
task's job outputs). Each task therefore gets its own worker.log and telemetry sidecar,
and the worker re-points both when it switches. Override it to put a worker's
sidecar/logs/server artifacts on local disk instead, e.g. if I/O-heavy backend-server
logs shouldn't hit the shared filesystem.

What is *not* per task: --capability, --use-indexes, --kv-cache-pin-gb and --device, and
with them the backend servers, which start once (under the first task's directory) and
are shared by every task in the list - restarting them per task would throw away a warm 70B KV cache
between sweeps. Every task in one list must therefore want the same servers; an
--use-indexes mismatch in particular is not detectable from here.
"""

import argparse
import logging
import os
import socket
import threading
import time
from pathlib import Path

import requests

from reasondb.coordinator.capabilities import (
    DEFAULT_READY_TIMEOUT_S,
    PinnedKVReleaseFailed,
    ServerIndexModeMismatch,
    release_pinned_kv_if_dataset_changed,
    start_capability_servers,
    wait_until_ready,
)
from reasondb.coordinator.logging_utils import daemonize, redirect_log
from reasondb.coordinator.models import (
    WORKER_CAPABILITY_CHOICES,
    Job,
    JobResult,
    TaskTarget,
    WorkerContext,
    parse_task_targets,
)
from reasondb.coordinator.producers import get_producer
from reasondb.monitor.collector import (
    Collector,
    default_sidecar_path,
    record_job_spec,
    set_current_job,
)

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 15.0
TELEMETRY_FLUSH_INTERVAL_S = 2.0
NO_JOB_BACKOFF_S = 5.0
INGEST_BATCH_LIMIT = 500

DEFAULT_WORKER_DIR_TEMPLATE = "benchmark_results/{task_id}/workers/{worker_id}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tasks", type=str, nargs="+", required=True, metavar="TASK_ID=URL",
        help="Coordinators to serve, in order; the worker moves to the next one as soon "
        "as the current task drains and exits after the last. "
        "e.g. --tasks sweep01=http://coordinator-host:5099",
    )
    parser.add_argument("--worker-id", type=str, required=True, help="Stable across restarts - pass the same value to reconnect after a crash.")
    parser.add_argument("--capability", type=str, required=True, choices=WORKER_CAPABILITY_CHOICES)
    parser.add_argument(
        "--worker-dir", type=str, default=DEFAULT_WORKER_DIR_TEMPLATE,
        help="Template for the directory this worker's own sidecar/logs/server artifacts "
        f"live under, resolved per task. Placeholders: {{task_id}}, {{worker_id}}. Default: "
        f"{DEFAULT_WORKER_DIR_TEMPLATE}. The backend servers are started once, under the "
        "first task's directory, and shared by every task in the list.",
    )
    parser.add_argument("--device", type=str, required=True, help="e.g. cpu, cuda:0. No default, so a worker never silently falls back to cpu.")
    parser.add_argument("--use-indexes", action="store_true")
    parser.add_argument(
        "--kv-cache-pin-gb", type=float, default=None, metavar="GB",
        help="RAM budget per backend server for the KV caches an -in-memory operator "
        "pins at prepare() - exported as KV_CACHE_PIN_GB to the start_servers_*.sh this "
        "worker runs, so every server it starts gets this much (not this much between "
        "them). 0 = disk-served operators only, which makes an -in-memory operator fail "
        "loudly at setup() rather than quietly read from disk. Omitted leaves the "
        "variable to the environment the servers inherit; the cluster config sets it for "
        "a fleet (workers.kv_cache_pin_gb).",
    )
    parser.add_argument(
        "--skip-server-start", action="store_true",
        help="Skip the server-start step entirely, including its readiness wait "
        "(debugging). Not needed just because the servers are up: start_capability_"
        "servers already reuses a live set rather than launching a second one.",
    )
    parser.add_argument(
        "--server-ready-timeout-s", type=float, default=DEFAULT_READY_TIMEOUT_S,
        help="How long to wait for this capability's servers to answer /status before "
        "giving up and exiting. A cold 70B KV server pre-compresses a cache for every "
        f"row before it answers, so the default is generous ({DEFAULT_READY_TIMEOUT_S:.0f}s).",
    )
    return parser


#: Gateway statuses that mean "nothing answered on the other side yet", not "the
#: coordinator said no" (e.g. a proxy in front of a coordinator that is still starting
#: up). Retrying these lets the whole fleet be launched before the coordinators exist.
RETRY_STATUS_CODES = frozenset({502, 503, 504})


def _request(method, url: str, **kwargs) -> dict:
    """One coordinator call, retrying *only* the errors that mean "not reachable yet".

    A connection error, a timeout or a gateway status is transient by assumption: the next
    coordinator in ``--tasks`` may not have been started yet, or the current one is
    restarting. Every other ``HTTPError`` is fatal - a 409 says this worker is registered
    for a different task, and spinning on that would hide the misconfiguration instead of
    failing on it.
    """
    while True:
        try:
            r = method(url, **kwargs)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.warning("Worker: %s unreachable (%s); retrying in %.0fs.", url, exc, NO_JOB_BACKOFF_S)
            time.sleep(NO_JOB_BACKOFF_S)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRY_STATUS_CODES:
                raise
            logger.warning("Worker: %s answered %s (not up yet?); retrying in %.0fs.", url, status, NO_JOB_BACKOFF_S)
            time.sleep(NO_JOB_BACKOFF_S)


def _post(url: str, json_body: dict, timeout: float = 30.0) -> dict:
    return _request(requests.post, url, json=json_body, timeout=timeout)


def _get(url: str, timeout: float = 10.0) -> dict:
    return _request(requests.get, url, timeout=timeout)


def register(target: TaskTarget, args: argparse.Namespace, worker_dir: Path) -> None:
    _post(
        f"{target.coordinator_url}/api/workers/register",
        {
            "worker_id": args.worker_id,
            "task_id": target.task_id,
            "capability": args.capability,
            "hostname": socket.gethostname(),
            "device": args.device,
            "worker_dir": str(worker_dir),
            "pid": os.getpid(),
        },
    )
    logger.info("Worker %s registered for task %s at %s.", args.worker_id, target.task_id, target.coordinator_url)


class _WorkerState:
    """Mutable holder shared between the claim/run loop and the background threads: the
    coordinator being served right now (it changes when a task drains), whether this
    worker has registered with it yet, and the job active right now, so a flushed
    telemetry batch can be tagged with it."""

    def __init__(self) -> None:
        self.coordinator_url: str = ""
        self.job_id: "str | None" = None
        #: Cleared while switching task, set again once registered - a heartbeat sent to a
        #: coordinator this worker has not registered with yet is a 500 on its side.
        self.registered = threading.Event()


def heartbeat_loop(state: _WorkerState, worker_id: str, stop_event: threading.Event) -> None:
    """One thread for the worker's whole life: it reads the *current* coordinator off
    ``state`` each tick, so a task switch redirects heartbeats without a restart."""
    while not stop_event.is_set():
        if not state.registered.wait(HEARTBEAT_INTERVAL_S):
            continue
        try:
            # Deliberately not _post: this loop has its own interval and must stay
            # responsive to stop_event, so it never sits inside _post's retry sleep.
            requests.post(
                f"{state.coordinator_url}/api/workers/{worker_id}/heartbeat", json={}, timeout=10.0
            ).raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Worker: heartbeat failed (%s); will retry.", exc)
        stop_event.wait(HEARTBEAT_INTERVAL_S)


def _forward_once(state: _WorkerState, worker_id: str, collector: Collector, since: int) -> int:
    """Forward whatever the collector has produced since ``since``; return the new cursor.

    Deliberately not _post: telemetry is best-effort, so an unreachable coordinator must
    cost one warning and a retry on the next tick, not a blocking retry loop.
    """
    try:
        payload = collector.events_since(since=since, limit=INGEST_BATCH_LIMIT)
        events = payload["events"]
        if events:
            requests.post(
                f"{state.coordinator_url}/api/ingest",
                json={"worker_id": worker_id, "job_id": state.job_id, "events": events},
                timeout=10.0,
            ).raise_for_status()
            since = payload["next_seq"]
    except requests.RequestException as exc:
        logger.warning("Worker: telemetry forward failed (%s); will retry.", exc)
    except Exception as exc:  # never let telemetry forwarding take the worker down
        logger.warning("Worker: telemetry forward error: %s", exc)
    return since


class _TaskSession:
    """One task's per-visit-invariant state, kept across revisits.

    A worker leaves a task as soon as it has nothing to claim and comes back later, so a
    task is entered many times. Both of these must survive across visits:

    * the **collector**, because a new one would start a new ``run_id`` and append it to
      the same sidecar, turning one worker's telemetry into a pile of pseudo-runs;
    * its **cursor**, because ``Collector.events_since`` is per collector and restarting
      it at 0 would re-forward every event to ``/api/ingest`` on each revisit -
      ``optimizer_solves`` and ``query_metrics`` are append-only and nothing downstream
      deduplicates.
    """

    def __init__(self, worker_dir: Path, worker_id: str) -> None:
        self.worker_dir = worker_dir
        self.collector = Collector(
            jsonl_path=default_sidecar_path(worker_dir, f"worker-{worker_id}")
        )
        self.since = 0


def telemetry_forward_loop(
    state: _WorkerState,
    worker_id: str,
    collector: Collector,
    stop_event: threading.Event,
    session: "_TaskSession | None" = None,
) -> None:
    """Periodically forward new local events to the coordinator's /api/ingest.

    One thread per *visit* to a task, because it is bound to that task's collector: it
    always ends with a final forward *after* ``stop_event`` is set, so the caller can
    close that collector and open the next task's without losing the tail (and without
    sharing the cursor across threads).

    The cursor lives on ``session`` rather than in this frame, because a worker leaves a
    task as soon as it has nothing claimable and revisits it later: a thread starting
    again from 0 would re-forward everything the previous visit already sent, and
    ``/api/ingest`` deduplicates nothing.

    Job attribution does not depend on when a batch happens to flush: each event is
    stamped with the job that produced it, as it is produced
    (``reasondb.monitor.collector.set_current_job``, called at the job boundaries in
    ``claim_run_loop``). ``state.job_id`` is still sent alongside the batch, but the
    coordinator only applies it to events carrying no job of their own - see
    ``reasondb.coordinator.ingest``. This keeps a batch that straddles a job boundary from
    attributing the previous job's tail to the next job. Job state itself does not come
    from telemetry - that is /api/jobs/*.
    """
    since = session.since if session is not None else 0
    while True:
        since = _forward_once(state, worker_id, collector, since)
        if session is not None:
            session.since = since
        if stop_event.is_set():
            return
        stop_event.wait(TELEMETRY_FLUSH_INTERVAL_S)


def claim_run_loop(
    target: TaskTarget, args: argparse.Namespace, state: _WorkerState
) -> "tuple[bool, bool]":
    """Claim and run this task's jobs; return ``(drained, ran_anything)``.

    Returns as soon as the coordinator has nothing this worker can claim, rather than
    waiting for the jobs other workers are still running, so an idle worker can start the
    next task's jobs instead of polling through the tail of this one.

    ``drained`` is the coordinator's own ``all_terminal`` (done + failed == total, with
    total > 0), so it is never true mid-flight, and a task whose coordinator has not
    enumerated its jobs yet reports total == 0 - which comes back as "not drained, no
    work", so ``main`` moves on and comes back rather than skipping the entry for good.

    Not drained is therefore *not* the same as finished with: a straggler can fail and
    requeue, and pending jobs behind a phase barrier become claimable the moment the
    barrier lifts. Both are why ``main`` revisits every task that has not drained, and
    why leaving early is only safe together with that.
    """
    coordinator_url = target.coordinator_url
    worker_ctx = WorkerContext(device=args.device, worker_id=args.worker_id, capability=args.capability)
    ran_anything = False
    while True:
        claimed = _post(
            f"{coordinator_url}/api/jobs/claim",
            {"worker_id": args.worker_id, "capability": args.capability},
        )["job"]
        if claimed is None:
            summary = _get(f"{coordinator_url}/api/task/{target.task_id}/summary")
            if summary["all_terminal"]:
                logger.info("Worker %s: task %s fully drained.", args.worker_id, target.task_id)
                return True, ran_anything
            # `.get` because this is a log line: a coordinator that does not report a
            # counter should not be able to kill a worker over it.
            logger.info(
                "Worker %s: task %s has nothing claimable (%s pending, %s blocked behind "
                "phase %s, %s running elsewhere); moving on.",
                args.worker_id,
                target.task_id,
                summary.get("pending"),
                summary.get("blocked"),
                summary.get("phase"),
                (summary.get("claimed") or 0) + (summary.get("running") or 0),
            )
            return False, ran_anything
        ran_anything = True

        job = Job(**claimed)
        state.job_id = job.job_id
        # Every event produced from here on carries this job, whenever it is flushed.
        set_current_job(job.job_id)
        # Recorded per job so the sweep axes survive into the sidecar, and are ingested
        # up to the coordinator with the rest of this job's events.
        record_job_spec(job.job_id, job.spec)

        _post(f"{coordinator_url}/api/jobs/{job.job_id}/start", {"worker_id": args.worker_id})
        logger.info("Worker %s: running job %s (%s step=%s).", args.worker_id, job.job_id, job.benchmark, job.spec.get("step_idx"))
        producer = get_producer(job.producer)
        try:
            # Pinned KV caches are never evicted, so a warm server would accumulate one
            # dataset's column after another. Released here rather than at job end: this is
            # the first moment the previous dataset is known to be finished with. Placed
            # after the /start POST so a failure is reported as *this job* failing, below;
            # letting it propagate would kill the worker instead.
            release_pinned_kv_if_dataset_changed(args.capability, job.benchmark, job.split)
        except PinnedKVReleaseFailed as exc:
            logger.error("Worker %s: job %s: %s", args.worker_id, job.job_id, exc)
            result = JobResult.from_exception(exc)
        else:
            result = producer.run_job(job, worker_ctx)
        if result.success:
            _post(
                f"{coordinator_url}/api/jobs/{job.job_id}/complete",
                {"worker_id": args.worker_id, "result_summary": result.result_summary},
            )
            logger.info("Worker %s: job %s done (%s).", args.worker_id, job.job_id, result.result_summary)
        else:
            _post(
                f"{coordinator_url}/api/jobs/{job.job_id}/fail",
                {"worker_id": args.worker_id, "error": result.error},
            )
            logger.warning("Worker %s: job %s failed: %s", args.worker_id, job.job_id, result.error)
        state.job_id = None
        set_current_job(None)


def resolve_worker_dir(template: str, task_id: str, worker_id: str) -> Path:
    """Fill in ``--worker-dir``'s placeholders for one task."""
    try:
        return Path(template.format(task_id=task_id, worker_id=worker_id))
    except (KeyError, IndexError) as exc:
        raise SystemExit(
            f"--worker-dir {template!r}: unknown placeholder {exc}. Only {{task_id}} and "
            "{worker_id} are substituted; write a literal brace as {{ or }}."
        )


def resolve_targets(args: argparse.Namespace) -> "list[TaskTarget]":
    try:
        targets = parse_task_targets(args.tasks)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if len(targets) > 1 and "{task_id}" not in args.worker_dir:
        # Every task would write its log and telemetry sidecar over the previous task's:
        # default_sidecar_path derives its filename from the worker id alone.
        raise SystemExit(
            f"--worker-dir {args.worker_dir!r} has no {{task_id}} placeholder, so all "
            f"{len(targets)} tasks would share one directory and overwrite each other's "
            f"telemetry sidecar. Default: {DEFAULT_WORKER_DIR_TEMPLATE}."
        )
    return targets


def main() -> None:
    args = build_parser().parse_args()
    targets = resolve_targets(args)
    worker_dir = resolve_worker_dir(args.worker_dir, targets[0].task_id, args.worker_id)

    # This worker's own file - under its own --worker-dir, so no two workers (even
    # sharing a machine/cwd) ever write to the same log file. Backgrounds this
    # process against SIGHUP and points stdout/stderr here too - no shell
    # `> file 2>&1`/`mkdir -p` needed (see logging_utils.daemonize). Done before
    # server startup, so even a startup failure is captured. Each later task re-points
    # this at its own directory (redirect_log below).
    daemonize(worker_dir / "logging" / "worker.log")

    if not args.skip_server_start:
        try:
            # Once for the whole worker, under the first task's directory: the servers
            # are shared by every task in --tasks, and restarting them per task would
            # throw away a warm KV cache between sweeps.
            start_capability_servers(
                args.capability, worker_dir, args.use_indexes, kv_cache_pin_gb=args.kv_cache_pin_gb
            )
        except ServerIndexModeMismatch as exc:
            # Exit rather than register: every job this worker claimed would either fail
            # on a missing relative index or silently re-prefill every ratio, and the
            # running servers may belong to another run, so stopping them is not ours
            # to decide.
            logger.error("Worker %s: %s", args.worker_id, exc)
            raise SystemExit(1)
        ready = wait_until_ready(args.capability, timeout_s=args.server_ready_timeout_s)
        if not ready:
            logger.error("Worker %s: capability %r servers never became ready; exiting.", args.worker_id, args.capability)
            raise SystemExit(1)

    state = _WorkerState()
    stop_event = threading.Event()
    heartbeat = threading.Thread(
        target=heartbeat_loop, args=(state, args.worker_id, stop_event), daemon=True
    )
    heartbeat.start()

    # One session per task, created on first visit and kept until the worker exits: a
    # task is entered many times, and both the collector and its ingest cursor have
    # to survive that (see _TaskSession).
    sessions: "dict[str, _TaskSession]" = {}
    #: Where fd 1/2 already point, so the log is re-pointed only when the task changes.
    current_log = worker_dir / "logging" / "worker.log"
    # Tasks still worth returning to, in the order given. A task leaves this list only
    # when the coordinator reports it all-terminal; "nothing claimable right now" keeps
    # it, because a straggler can fail and requeue and a phase barrier can lift.
    remaining = list(targets)
    total = len(targets)
    try:
        while remaining:
            progressed = False
            for target in list(remaining):
                session = sessions.get(target.task_id)
                if session is None:
                    session = _TaskSession(
                        resolve_worker_dir(args.worker_dir, target.task_id, args.worker_id),
                        args.worker_id,
                    )
                    sessions[target.task_id] = session
                # Only when it actually moves: a worker revisits a task whenever it has
                # nothing to claim, so avoid re-pointing fd 1/2 on every pass.
                task_log = session.worker_dir / "logging" / "worker.log"
                if task_log != current_log:
                    redirect_log(task_log)
                    current_log = task_log
                # Point the background threads at this task before either can fire.
                state.registered.clear()
                state.coordinator_url = target.coordinator_url
                # Legal to re-install because close() uninstalls the previous one.
                session.collector.install()
                task_stop = threading.Event()
                telemetry = threading.Thread(
                    target=telemetry_forward_loop,
                    args=(state, args.worker_id, session.collector, task_stop, session),
                    daemon=True,
                )
                telemetry.start()
                try:
                    register(target, args, session.worker_dir)
                    state.registered.set()
                    drained, ran_anything = claim_run_loop(target, args, state)
                finally:
                    # Stop the forwarder (it flushes once more on its way out) before
                    # uninstalling the collector this visit wrote to. Not closed: the
                    # next visit reuses it.
                    state.job_id = None
                    set_current_job(None)
                    task_stop.set()
                    telemetry.join(timeout=TELEMETRY_FLUSH_INTERVAL_S * 5)
                    session.collector.uninstall()
                progressed = progressed or ran_anything
                if drained:
                    remaining.remove(target)
                    logger.info(
                        "Worker %s: task %s drained (%d/%d).",
                        args.worker_id, target.task_id, total - len(remaining), total,
                    )
            if remaining and not progressed:
                # A whole cycle with nothing claimable anywhere: every task is waiting on
                # another worker's job, a phase barrier, or a coordinator still starting
                # up. Back off before going round again rather than spinning through
                # register/claim for each of them.
                time.sleep(NO_JOB_BACKOFF_S)
        logger.info("Worker %s: all %d task(s) drained; exiting.", args.worker_id, total)
    except KeyboardInterrupt:
        logger.info("Worker %s: shutting down.", args.worker_id)
    finally:
        stop_event.set()
        for session in sessions.values():
            session.collector.close()


if __name__ == "__main__":
    main()
