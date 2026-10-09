"""Final-merge trigger: once every job in a task is terminal, fold each producer's
per-job output shards into one result file per benchmark.

Called by ``scripts/run_coordinator.py`` when ``JobDB.task_summary(...)['all_terminal']``
turns true (automatic-on-completion), or manually via that same script's ``--merge-now``
flag to check progress mid-sweep without waiting for stragglers - both paths go through
:func:`merge_task` so they can't drift.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from reasondb.coordinator.db import JobDB
from reasondb.coordinator.models import JOB_DONE
from reasondb.coordinator.producers import get_producer

logger = logging.getLogger(__name__)


def merge_task(task_id: str, job_db: JobDB) -> Dict[str, List[Path]]:
    """Merge every completed job's output for ``task_id``, grouped by producer.

    Only ``done`` jobs contribute a shard - a ``failed`` job (attempts exhausted) has
    no valid ``rows.parquet`` to fold in and is left out of the merged result rather
    than failing the whole merge; ``scripts/run_coordinator.py`` logs failed job ids
    separately so they're not silently lost from view.
    """
    jobs = job_db.list_jobs(task_id, state=JOB_DONE)
    by_producer: Dict[str, List[str]] = defaultdict(list)
    for job in jobs:
        by_producer[job.producer].append(job.output_dir)

    written: Dict[str, List[Path]] = {}
    for producer_name, output_dirs in by_producer.items():
        producer = get_producer(producer_name)
        paths = producer.merge(task_id, output_dirs)
        written[producer_name] = paths
        logger.info(
            "Coordinator: merged %d %s job(s) for task %s -> %s",
            len(output_dirs),
            producer_name,
            task_id,
            [str(p) for p in paths],
        )

    failed = job_db.list_jobs(task_id, state="failed")
    if failed:
        logger.warning(
            "Coordinator: %d job(s) for task %s exhausted their attempts and are "
            "excluded from the merge: %s",
            len(failed),
            task_id,
            [j.job_id for j in failed],
        )
    return written
