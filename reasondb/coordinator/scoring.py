"""Per-job scoring: report each finished job's accuracy as soon as its labels exist.

This lets the dashboard's guarantee-satisfaction panel fill in while a sweep runs,
rather than only after :func:`reasondb.coordinator.merge.merge_task` (which runs once
*every* job in the task is terminal). A job is exactly the axis the panel groups by
(one benchmark, one executor, one guarantee pair), the coordinator process installs a
``Collector``, and the Run tab polls ``query_metrics`` live.

This module owns the one question the producers cannot answer for themselves - *which
finished jobs are scorable yet* - and drives ``producer.score_job`` for those. Two
properties matter:

- **Labels are found from the job queue, not from a job's spec.** The label jobs for a
  benchmark are simply this task's ``spec["kind"] == "label"`` jobs, so nothing has to
  be baked into an approach job at enumeration time.
- **A job that cannot be scored yet is skipped, not failed.** Label jobs are enqueued
  at ``priority=-1`` so they are handed out first, but priority is not ordering: with
  several workers, an approach job can finish while the (expensive) silver pass is
  still running. Those jobs are simply retried on the next pass, which is the whole
  delay mechanism - no accuracy is shown for them until real labels back it.

Which ``(job, label set)`` pairs have been scored is persisted in the job database
(see :meth:`JobDB.mark_job_scored`), so a restarted coordinator does not re-score and
re-emit metrics for jobs it already reported.

Only ``scripts/run_coordinator.py``'s distributed path drives this. ``--local`` installs
no ``Collector`` at all (it is the no-Flask/no-SQLite debug path), so there is nothing
for a scorer to report to there; the merge still writes its CSVs.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from reasondb.coordinator.db import JobDB
from reasondb.coordinator.models import JOB_DONE, Job
from reasondb.coordinator.producers import PRODUCERS
from reasondb.coordinator.producers.shards import label_name_from_shard

logger = logging.getLogger(__name__)


class JobScorer:
    """Scores this task's finished jobs, one pass at a time.

    Keyed on ``(job_id, label_name)`` rather than ``job_id``: silver and gold finish far
    apart, so a job scored against silver must still be scored against gold when that
    lands, without re-emitting the silver rows.
    """

    def __init__(self, task_id: str, job_db: JobDB) -> None:
        self.task_id = task_id
        self.job_db = job_db
        #: ``(job_id, label_name)`` pairs already turned into metrics, loaded from the
        #: job database so a restarted coordinator does not re-score (and re-emit
        #: telemetry for) its whole task.
        self._scored: Set[Tuple[str, str]] = set(job_db.list_scored(task_id))
        #: Last line logged, so a long label pass does not repeat the same "awaiting"
        #: message every few seconds for as long as it runs.
        self._last_line: str = ""

    def pass_once(self) -> int:
        """Score every newly-scorable job. Returns how many jobs emitted anything.

        Never raises. A malformed shard or an unreadable label directory must not take
        down the coordinator loop that also drives lease sweeps and the final merge -
        the job is simply left unscored and retried on the next pass, and if the cause
        is permanent the merge still writes the CSVs at the end.
        """
        try:
            jobs = self.job_db.list_jobs(self.task_id, state=JOB_DONE)
        except Exception as exc:  # pragma: no cover - defensive, see docstring
            logger.warning("[scoring] task %s: could not list jobs: %s", self.task_id, exc)
            return 0

        label_dirs = _label_dirs_by_benchmark(jobs)
        scored_jobs = 0
        awaiting: List[str] = []
        for job in jobs:
            producer = PRODUCERS.get(job.producer)
            if producer is None or producer.score_job is None:
                continue
            if job.spec.get("kind") == "label":
                continue
            dirs = [
                output_dir
                for label_name, output_dir in label_dirs.get(job.benchmark, [])
                if (job.job_id, label_name) not in self._scored
            ]
            if not dirs:
                if not label_dirs.get(job.benchmark):
                    awaiting.append(job.benchmark)
                continue
            try:
                names = producer.score_job(job, dirs)
            except Exception as exc:
                logger.warning("[scoring] job %s could not be scored: %s", job.job_id, exc)
                continue
            for name in names:
                self._scored.add((job.job_id, name))
                # Persisted as each one lands, not batched at the end of the pass, so a
                # coordinator killed mid-pass does not redo what it had already scored.
                self.job_db.mark_job_scored(job.job_id, name)
            if names:
                scored_jobs += 1

        if scored_jobs or awaiting:
            waiting = (
                f"; {len(awaiting)} awaiting labels (none done yet for "
                f"{', '.join(sorted(set(awaiting)))})"
                if awaiting
                else ""
            )
            line = f"[scoring] task {self.task_id}: scored {scored_jobs} job(s){waiting}"
            if line != self._last_line:
                logger.info("%s", line)
                self._last_line = line
        return scored_jobs


def _label_dirs_by_benchmark(jobs: List[Job]) -> Dict[str, List[Tuple[str, str]]]:
    """``benchmark -> [(label name, output dir)]`` over the finished label jobs.

    Three spellings, in order. ``name`` is what ``run_benchmark`` and ``parameter_sweep``
    put on the spec. ``label_set`` is an alternative spelling of the same key. Failing
    both, the shard the job wrote says so
    itself - ``sample_size`` cannot name its label set at enumeration time, because
    ``label_set_for`` needs a benchmark instance and enumeration builds none.
    """
    by_benchmark: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for job in jobs:
        if job.spec.get("kind") != "label":
            continue
        name = (
            job.spec.get("name")
            or job.spec.get("label_set")
            or label_name_from_shard(job.output_dir)
        )
        if isinstance(name, str) and name:
            by_benchmark[job.benchmark].append((name, job.output_dir))
    return by_benchmark
