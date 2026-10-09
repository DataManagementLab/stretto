"""Put a task's ``failed`` jobs back on the queue, attempt counter and all.

A job is ``failed`` only because its attempts ran out: ``fail_job`` returns it to
``pending`` while ``attempt < max_attempts`` and gives up after that, and
``fail_blocked_jobs`` fails everything sitting behind a phase that never settled.
Neither says the work cannot succeed - an unavailable model server or a full disk
land there too - so re-running a task means clearing the *counter*, not only the
state. This sets ``attempt`` to 0 and clears the error, the claim and the finish time
along with the state.

Nothing else is needed to make a restarted coordinator pick them up:
``run_coordinator`` re-enumerates only when the task has *no* jobs on disk, and
otherwise resumes from this same database - so the reset jobs are simply pending
work the moment it comes back up.

The input is a **task id** - the coordinator's own ``--task-id``, which is also the
directory its results live in: the database is
``<results-root>/<task-id>/coordinator.db``, mirroring ``run_coordinator.py``'s own
``--output-dir`` default. Any task id resolves that way, whether or not the cluster
config has heard of it; a hand-launched ``--task-id my_task`` is reset exactly like
``abl01``. The config is consulted for **one** thing and is optional: an experiment that
overrides ``--output-dir`` in its ``args:`` has that value read back out of the same
config the launchers read, so its results are found where they really are. ``--all``
is the one flag that needs the config, since "every experiment" is a question only it
can answer.

A task that is neither in the config nor at the default path - a run whose
``--output-dir`` was named on the command line - is named with ``--output-dir`` here
too, one task at a time.

Stop the coordinator first. ``coordinator/db.py`` is single-writer by construction
(only the coordinator process ever opens the file), and this script is a second
writer; it refuses to run against a task that looks live unless ``--force``. What that
does and does not protect you from is spelled out under "Running it live" below.

Examples
--------
    # What would be reset (default: nothing is written)
    python scripts/reset_failed_jobs.py abl01

    # Do it
    python scripts/reset_failed_jobs.py abl01 --apply

    # Every experiment in scripts/cluster.yaml
    python scripts/reset_failed_jobs.py --all --apply

    # A task the config has never heard of, at the default path
    python scripts/reset_failed_jobs.py my_task --apply

    # ... and one whose results are somewhere else entirely
    python scripts/reset_failed_jobs.py my_task --output-dir /path/to/results/my_task --apply

    # One benchmark's failures only - see the note about blocked jobs below
    python scripts/reset_failed_jobs.py base01 --benchmarks movie_random --apply

Running it live
---------------
Physically it is safe: SQLite's own file locking serializes the two processes, the
selection and the write share one ``BEGIN IMMEDIATE`` transaction, and ``failed`` is
terminal for everyone but this script - so nothing can be lost or half-written, and the
worst case is ``database is locked`` after the 5s busy timeout, i.e. run it again. The
single-writer design is about a *shared* filesystem, where SQLite's locking is the part
that cannot be trusted; on an NFS-mounted ``--results-root``, stop the coordinator.

Logically there are two things to know. A live coordinator picks the reset jobs up
**without a restart** - ``claim_next_job`` reads the table on every claim - so the
restart the closing message suggests is only needed for a coordinator that is already
down. And the blocked-job hazard below stops being a hazard-if-you-later-restart and
becomes immediate: ``fail_blocked_jobs`` runs on the lease-sweep thread every
``--lease-sweep-interval-s``, so a narrowed reset that leaves an earlier phase failed is
undone within seconds rather than at the next launch.

Narrowing with ``--benchmarks``/``--job-ids`` has one hazard worth knowing: a
phase-1 job reset while the phase-0 job it waits on is still ``failed`` gets failed
again by the coordinator's ``fail_blocked_jobs`` as soon as it starts. Resetting
everything (the default) cannot hit that, since it leaves no failed job behind.
"""

import argparse
import logging
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from reasondb.coordinator.cluster import DEFAULT_CONFIG_PATH, load_cluster_config
from reasondb.coordinator.db import JobDB
from reasondb.coordinator.models import Job

logger = logging.getLogger(__name__)

#: Same literal ``run_coordinator.py`` defaults its ``--output-dir`` to.
DEFAULT_RESULTS_ROOT = Path("benchmark_results")


def output_dir_overrides(config_path: Path) -> Dict[str, Path]:
    """``task_id -> --output-dir`` for the experiments that override it.

    Read from the cluster config rather than guessed, so a task whose results do not
    live at the default path is found here for the same reason the launchers find it.
    A missing or unparseable config is not fatal, and a task id absent from a config
    that parses fine is not an error either: the config is a source of *overrides*, not
    the list of task ids this script will accept. Either way the default path is used,
    which is where a hand-launched run's results actually are.
    """
    try:
        config = load_cluster_config(config_path)
    except Exception as error:  # unreadable, absent, or invalid - all non-fatal here
        logger.debug("Not reading output-dir overrides from %s: %s", config_path, error)
        return {}
    overrides: Dict[str, Path] = {}
    for experiment in config.experiments:
        argv = experiment.argv()
        if "--output-dir" in argv:
            index = argv.index("--output-dir")
            if index + 1 < len(argv):
                overrides[experiment.task_id] = Path(argv[index + 1])
    return overrides


def config_task_ids(config_path: Path) -> List[str]:
    """Every task id in the cluster config, in the order it lists them."""
    return [experiment.task_id for experiment in load_cluster_config(config_path).experiments]


def db_path_for(
    task_id: str,
    results_root: Path,
    overrides: Dict[str, Path],
    output_dir: Optional[Path] = None,
) -> Path:
    """Where this task's queue lives, most specific source first.

    ``--output-dir`` on this script's own command line, then the cluster config's
    override for this experiment, then ``<results-root>/<task-id>`` - the default
    ``run_coordinator.py`` itself uses, and therefore the answer for any task nobody
    said anything else about.
    """
    if output_dir is not None:
        return output_dir / "coordinator.db"
    return overrides.get(task_id, results_root / task_id) / "coordinator.db"


def summarize(jobs: Sequence[Job]) -> List[str]:
    """One ``<count>x <benchmark> <producer>`` line per group, most frequent first."""
    counts = Counter((job.benchmark, job.producer) for job in jobs)
    return [
        f"{count:5d}x {benchmark} ({producer})"
        for (benchmark, producer), count in counts.most_common()
    ]


def error_lines(jobs: Sequence[Job], limit: int = 5) -> List[str]:
    """The distinct errors these jobs failed with, most frequent first.

    Truncated per line and capped in number: a stack trace in ``error`` is what a real
    failure looks like, and the point here is to see *whether* they all failed the same
    way before putting them back - not to read them.
    """
    counts = Counter(
        ((job.error or "").strip().splitlines() or ["<no error recorded>"])[-1] for job in jobs
    )
    lines = [f"{count:5d}x {error[:160]}" for error, count in counts.most_common(limit)]
    if len(counts) > limit:
        lines.append(f"      ... and {len(counts) - limit} more distinct error(s)")
    return lines


def reset_task(
    task_id: str,
    db_path: Path,
    apply: bool,
    force: bool,
    benchmarks: Optional[Sequence[str]],
    job_ids: Optional[Sequence[str]],
) -> int:
    """Reset one task's failed jobs. Returns how many were (or would be) reset.

    A ``database is locked`` is reported as itself rather than as a traceback: it is the
    ordinary outcome of a coordinator writing at the same moment, it leaves this database
    exactly as it was (the selection and the write share one transaction), and the fix is
    to run the command again.
    """
    if not db_path.is_file():
        logger.warning(
            "%-8s no coordinator database at %s - skipping. A task's queue lives at "
            "<results-root>/<task-id>/coordinator.db; point --results-root at the tree "
            "it ran under, or --output-dir straight at the task's own directory if the "
            "run named one.",
            task_id, db_path,
        )
        return 0

    try:
        job_db = JobDB(db_path)
    except sqlite3.OperationalError as error:
        logger.error("%-8s %s: %s. Nothing was changed; try again.", task_id, db_path, error)
        return 0

    try:
        summary = job_db.task_summary(task_id)
        if summary["total"] == 0:
            logger.warning(
                "%-8s %s holds no jobs for this task id - is it the right one?",
                task_id, db_path,
            )
            return 0
        live = summary["claimed"] + summary["running"]
        if live and not force:
            logger.error(
                "%-8s %d job(s) are claimed/running and %d worker(s) alive: the "
                "coordinator looks live, and this database has exactly one writer. "
                "Stop it first, or pass --force if you know it is down.",
                task_id, live, summary["workers_alive"],
            )
            return 0

        jobs = job_db.reset_failed_jobs(
            task_id,
            job_ids=list(job_ids) if job_ids else None,
            benchmarks=list(benchmarks) if benchmarks else None,
            dry_run=not apply,
        )
        verb = "Reset" if apply else "Would reset"
        # The counts are the state *before* the reset, and say so - they are what the
        # selection was made from, and a "pending" that already counted the reset jobs
        # would read as if nothing had moved.
        logger.info(
            "%-8s %s %d job(s). Before: %d total, %d done, %d pending, %d failed.",
            task_id, verb, len(jobs), summary["total"],
            summary["done"], summary["pending"], summary["failed"],
        )
        for line in summarize(jobs):
            logger.info("           %s", line)
        for line in error_lines(jobs):
            logger.info("           %s", line)
        return len(jobs)
    except sqlite3.OperationalError as error:
        # Almost always "database is locked": a live coordinator held the write lock for
        # longer than sqlite3's 5s busy timeout. The transaction rolled back, so this
        # task is untouched and re-running is the whole fix.
        logger.error(
            "%-8s %s. Nothing was changed - the write is one transaction, so it either "
            "happens or it does not. A coordinator writing at the same moment is the "
            "usual cause; run this again, or stop it first.",
            task_id, error,
        )
        return 0
    finally:
        job_db.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "task_ids",
        nargs="*",
        metavar="TASK_ID",
        help="Any coordinator --task-id: the cluster config's (base01, abl01, ...) or a "
        "hand-launched one. Its queue is read from <results-root>/<task-id>/"
        "coordinator.db unless the config or --output-dir says otherwise.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Every experiment in --config, instead of naming task ids.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_PATH),
        help="Only two things are read from it: --all's task list, and the --output-dir "
        "an experiment overrides. A task id it does not mention is not an error.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Where a task's directory lives (default: {DEFAULT_RESULTS_ROOT}), "
        "mirroring run_coordinator.py's own --output-dir default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="The task's own directory, i.e. run_coordinator.py's --output-dir, for a "
        "run that named one and is not in --config. One task id at a time.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        metavar="NAME",
        help="Only failed jobs of these benchmarks. See the note on blocked jobs.",
    )
    parser.add_argument(
        "--job-ids",
        nargs="+",
        metavar="JOB_ID",
        help="Only these job ids. See the note on blocked jobs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write. Without it nothing is modified and the selection is only printed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reset even when jobs are claimed/running, i.e. when the coordinator "
        "looks live. Only when you know it is down.",
    )
    args = parser.parse_args()

    assert bool(args.task_ids) != bool(args.all), (
        "Name the task ids to reset, or pass --all for every experiment in "
        f"{args.config} - not both, and not neither."
    )
    task_ids = config_task_ids(args.config) if args.all else args.task_ids
    assert not (args.job_ids and len(task_ids) > 1), (
        "--job-ids selects within one task; name a single task id."
    )
    # One directory holds one task's queue, so pointing two task ids at it would either
    # mean the same database twice or a typo. --all is the same statement, louder.
    assert not (args.output_dir and (args.all or len(task_ids) > 1)), (
        "--output-dir names one task's own directory; name a single task id (or drop it "
        "and let each task resolve to <results-root>/<task-id>, which is what --all "
        "needs). Also true of a one-experiment --all, hence the explicit check."
    )

    overrides = output_dir_overrides(args.config)
    total = 0
    for task_id in task_ids:
        total += reset_task(
            task_id,
            db_path_for(task_id, args.results_root, overrides, args.output_dir),
            apply=args.apply,
            force=args.force,
            benchmarks=args.benchmarks,
            job_ids=args.job_ids,
        )

    if not args.apply:
        logger.info(
            "Dry run - nothing written. Re-run with --apply to reset %d job(s).", total
        )
    elif total:
        logger.info(
            "Reset %d job(s) across %d task(s). Nothing else is needed: a coordinator "
            "still up claims them on its next claim (it reads this table every time), "
            "and a stopped one resumes rather than re-enumerating when it comes back, "
            "since the task has jobs on disk.",
            total, len(task_ids),
        )


if __name__ == "__main__":
    main()
