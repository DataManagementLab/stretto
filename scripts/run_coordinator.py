"""Coordinator for a distributed benchmark sweep: enumerates the sweep's jobs once,
persists them in a SQLite queue, serves them to workers over HTTP, and merges results
when the task drains.

Example
-------
    # Distributed: start the coordinator in the background - it survives an SSH
    # disconnect on its own (see logging_utils.daemonize, no nohup/redirection/mkdir
    # needed on the command line) - then on each worker machine (see
    # scripts/run_worker.py) name this task and host in --tasks, e.g.
    # --tasks sweep01=http://this-host:5099 (a worker can queue several).
    python scripts/run_coordinator.py --task-id sweep01 --host 0.0.0.0 \\
        --producer parameter_sweep --benchmarks movie_random \\
        --precision-guarantees 0.7 --recall-guarantees 0.7 --simulate movie.json &

    # Local (single machine, no Flask/SQLite/workers - fastest debug/test path):
    python scripts/run_coordinator.py --task-id smoke --local \\
        --producer parameter_sweep \\
        --benchmarks movie_random --simulate movie.json --device cpu

--output-dir defaults to benchmark_results/<task-id> (a shared-filesystem path) -
override it only if you want this task's directory somewhere else.

Every producer is wired into this one parser rather than producer-conditional
subparsers: their producer-specific flags (``--text-small-model``/``--press-name``/
``--tune-parameters``/``--sample-sizes``/``--approaches`` for the ``parameter_sweep``
family, ``--labels``/``--select-executors`` for run_benchmark) don't collide with each
other or with the shared flags every producer's ``enumerate_jobs``/``run_job`` reads
(``--benchmarks``, guarantees, ``--cost-type``, etc.), so unconditionally adding every
group keeps this simple at the cost of ``--help`` showing flags irrelevant to whichever
``--producer`` you picked.

``parameter_sweep`` is the engine; ``sample_size``, ``operator_count`` and ``tuning``
are thin producers in front of it, each fixing the axes it is not studying. A flag one
of them fixes is *rejected* rather than ignored when you pass it, with a message naming
the producer that does sweep that axis - so picking the experiment by name and picking
it by flag can never disagree.

Every axis flag defaults to a single point (``--sample-sizes`` to the optimizer's own
budget, ``--approaches`` to ``optim_global``, ``--tune-parameters`` to ``true``), so
widening the sweep is always something you asked for rather than something a shared
default did to you.

"""

import argparse
import logging
import threading
import time
from pathlib import Path


from reasondb.coordinator.app import create_coordinator_app, run_lease_sweep_forever
from reasondb.coordinator.capabilities import (
    PinnedKVReleaseFailed,
    release_pinned_kv_if_dataset_changed,
)
from reasondb.coordinator.db import JobDB
from reasondb.coordinator.logging_utils import daemonize
from reasondb.coordinator.merge import merge_task
from reasondb.coordinator.models import JobResult, WorkerContext
from reasondb.coordinator.cli import DEFAULT_BENCHMARKS, build_parser
from reasondb.coordinator.producers import get_producer
from reasondb.coordinator.scoring import JobScorer
from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS
from reasondb.monitor.collector import (
    Collector,
    default_sidecar_path,
    make_run_id,
    record_job_spec,
    set_current_job,
)
from reasondb.monitor.server import MonitorServer, find_free_port, resolve_port
from reasondb.utils.benchmark_args import resolve_precompute_simulate

logger = logging.getLogger(__name__)

# The parser lives in the package (reasondb/coordinator/cli.py), not here:
# scripts/generate_experiment_report.py has to resolve the same flags to describe what
# each experiment sweeps, and a script may not import another script.


def run_local(args: argparse.Namespace) -> None:
    """Enumerate + run every job in-process, then merge. No coordinator DB, no HTTP."""
    producer = get_producer(args.producer)
    jobs = producer.enumerate_jobs(args.task_id, args.output_dir, args)
    # The DB-level barrier does nothing here - there is no queue, just this list. Sorting
    # by phase is what makes --local obey the same ordering as a distributed run, rather
    # than depending on producers happening to emit phase-0 jobs first.
    jobs = [
        job
        for _index, job in sorted(
            enumerate(jobs), key=lambda pair: (pair[1].phase, pair[1].priority, pair[0])
        )
    ]
    logger.info("[local] %d job(s) to run for task %s.", len(jobs), args.task_id)
    worker = WorkerContext(device=str(args.device), worker_id="local", capability="both")

    output_dirs = []
    failures = []
    failed_phases = set()
    for i, job in enumerate(jobs):
        if any(phase < job.phase for phase in failed_phases):
            # Same rule as db.fail_blocked_jobs: a sweep whose filter stats failed has no
            # query set to sweep over, so running it would fail confusingly later.
            failures.append((job.job_id, "blocked: an earlier phase failed"))
            logger.warning("[local] job %s skipped: an earlier phase failed.", job.job_id)
            continue
        Path(job.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("[local] [%d/%d] phase=%d %s step=%s guarantee=%s", i + 1, len(jobs), job.phase, job.benchmark, job.spec.get("step_idx"), job.spec.get("guarantee"))
        # A --local run has no forwarder to tag batches, so without this its telemetry
        # carries no job at all and the dashboard cannot separate one job from the next.
        set_current_job(job.job_id)
        # The sweep axes (step_idx, sample_size, adaptive_sampling) live only in the
        # spec, and a --local run has no coordinator API to serve them from.
        record_job_spec(job.job_id, job.spec)
        try:
            # See claim_run_loop: pins are never evicted, so they are dropped when the
            # dataset changes. Caught rather than raised so one job's failure leaves the
            # loop running, as a failing run_job already does.
            release_pinned_kv_if_dataset_changed(worker.capability, job.benchmark, job.split)
            result = producer.run_job(job, worker)
        except PinnedKVReleaseFailed as exc:
            result = JobResult.from_exception(exc)
        finally:
            set_current_job(None)
        if result.success:
            output_dirs.append(job.output_dir)
        else:
            failures.append((job.job_id, result.error))
            failed_phases.add(job.phase)
            logger.warning("[local] job %s failed: %s", job.job_id, result.error)

    written = producer.merge(args.task_id, output_dirs)
    logger.info("[local] merged %d job(s) -> %s", len(output_dirs), [str(p) for p in written])
    if failures:
        logger.warning("[local] %d job(s) failed and were excluded: %s", len(failures), failures)


def run_coordinator(args: argparse.Namespace) -> None:
    job_db = JobDB(args.output_dir / "coordinator.db")

    summary = job_db.task_summary(args.task_id)
    if summary["total"] == 0:
        producer = get_producer(args.producer)
        jobs = producer.enumerate_jobs(args.task_id, args.output_dir, args)
        job_db.enqueue_jobs(jobs)
        logger.info("Coordinator: enumerated and enqueued %d job(s) for task %s.", len(jobs), args.task_id)
    else:
        logger.info(
            "Coordinator: task %s already has %d job(s) on disk (resuming, not re-enumerating).",
            args.task_id, summary["total"],
        )

    if args.merge_now:
        merge_task(args.task_id, job_db)
        job_db.close()
        return

    run_id = make_run_id()
    collector = Collector(
        jsonl_path=default_sidecar_path(args.output_dir, run_id)
    ).install()

    # Build the coordinator-extended app *before* binding, then drive MonitorServer
    # directly (rather than monitor.server.start_server, which builds its own
    # collector-only app internally) - the app a MonitorServer serves is fixed for
    # its thread's lifetime once .start() runs, so it must be the right one from the
    # first request, not swapped in after the fact.
    app = create_coordinator_app(
        task_id=args.task_id,
        job_db=job_db,
        collector=collector,
        result_roots=[args.output_dir],
        run_info={"run_id": run_id, "task_id": args.task_id},
    )
    port = find_free_port(resolve_port(args.port), host=args.host)
    server = MonitorServer(app, port, host=args.host).start() if port is not None else None
    if server is not None:
        logger.info("Coordinator: dashboard/API at %s (task_id=%s)", server.url, args.task_id)
        print(f"[coordinator] dashboard: {server.url}  task_id={args.task_id}", flush=True)
    else:
        logger.warning("Coordinator: HTTP server failed to bind; workers cannot connect.")

    stop_event = threading.Event()
    sweep_thread = threading.Thread(
        target=run_lease_sweep_forever,
        args=(job_db, args.task_id, stop_event, args.lease_sweep_interval_s,
              args.job_heartbeat_timeout_s, args.worker_heartbeat_timeout_s),
        daemon=True,
    )
    sweep_thread.start()

    # Scores each finished job as soon as its labels land, so the dashboard's accuracy
    # panel fills during the sweep rather than only at the merge below. Driven off this
    # same loop rather than its own thread: it is cheap (unpickle + vectorized metrics),
    # and a job that is not scorable yet is simply retried on the next tick.
    scorer = JobScorer(args.task_id, job_db)

    merged = False
    try:
        while True:
            time.sleep(5.0)
            scorer.pass_once()
            summary = job_db.task_summary(args.task_id)
            if summary["all_terminal"] and not merged:
                logger.info("Coordinator: task %s complete (%s). Merging.", args.task_id, summary)
                merge_task(args.task_id, job_db)
                merged = True
                print(f"[coordinator] task {args.task_id} complete; dashboard stays up for inspection.", flush=True)
            # Keep serving after the merge so the dashboard stays available for
            # inspection; stop with Ctrl-C.
    except KeyboardInterrupt:
        logger.info("Coordinator: shutting down.")
    finally:
        stop_event.set()
        collector.close()
        job_db.close()


#: Producers that implement ``--precompute``. Other producers would ignore the flag and
#: run an ordinary, non-recording sweep, so the combination is rejected up front.
_PRECOMPUTE_PRODUCERS = ("parameter_sweep", "run_benchmark")


def main() -> None:
    args = build_parser(description=__doc__).parse_args()
    resolve_precompute_simulate(args, ALL_BENCHMARKS, DEFAULT_BENCHMARKS)
    assert args.precompute is None or args.producer in _PRECOMPUTE_PRODUCERS, (
        f"--producer {args.producer} does not implement --precompute; use "
        f"{' or '.join(_PRECOMPUTE_PRODUCERS)}."
    )
    # Rejected rather than ignored, for the reason _PRECOMPUTE_PRODUCERS exists.
    # run_benchmark's precompute already records exactly `get_default_configurator`'s
    # suite, so it has nothing to narrow.
    assert args.precompute_states == "all" or (
        args.precompute is not None and args.producer == "parameter_sweep"
    ), (
        "--precompute-states narrows what a --producer parameter_sweep --precompute "
        f"pass records; it means nothing for --producer {args.producer}"
        f"{' without --precompute' if args.precompute is None else ''}."
    )
    # Same reason: the split only shapes how a --precompute pass is enumerated. Both precompute producers implement it - `ecommerce_curated` is
    # recorded through run_benchmark, and is mixed-modality too.
    assert not args.split_both_capability_datasets or (
        args.precompute is not None and args.producer in _PRECOMPUTE_PRODUCERS
    ), (
        "--split-both-capability-datasets splits a --precompute pass over a "
        "mixed-modality dataset into one job per modality; it means nothing for "
        f"--producer {args.producer}"
        f"{' without --precompute' if args.precompute is None else ''}."
    )
    assert args.task_id, "--task-id must be non-empty."
    assert not args.local or args.device is not None, (
        "--local runs jobs in-process as its own worker and needs --device "
        "(e.g. cpu or cuda:0); distributed mode (the default) doesn't take one at "
        "all - only the workers you launch separately do."
    )
    if args.output_dir is None:
        args.output_dir = Path("benchmark_results") / args.task_id
    # Shared-FS log, next to this task's SQLite DB and JSONL telemetry sidecar (both
    # already live under output_dir). Backgrounds this process against SIGHUP and
    # points stdout/stderr here too - no shell `> file 2>&1`/`mkdir -p` needed, just
    # `python scripts/run_coordinator.py ... &` (see logging_utils.daemonize).
    daemonize(args.output_dir / "logging" / "coordinator.log")
    if args.local:
        run_local(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
