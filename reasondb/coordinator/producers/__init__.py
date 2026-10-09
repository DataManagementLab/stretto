"""Registry: producer name -> its (enumerate_jobs, run_job, merge[, score_job]) tuple.

``scripts/run_coordinator.py``/``scripts/run_worker.py`` look producers up by name here
rather than importing a specific producer module directly, so adding one never requires
touching the coordinator/worker core.

Two shapes live here. ``run_benchmark`` wraps its own script's sweep-point logic, since its
executor construction differs from the rest. ``parameter_sweep`` is an
*engine* - one cross-product over state x guarantee x approach x tuning x sample size x
adaptive sampling x reordering - with nine named experiment producers in front of it
(``baselines``, ``sample_size``, ``operator_count``, ``kv_operator``, ``tuning``,
``ablation``, ``adaptive_sampling``, ``reordering``, ``reorder_only``),
each fixing the axes its experiment is not studying. See ``producers/experiments.py`` for why those are producers rather than
documented flag combinations.
"""

from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional

from reasondb.coordinator.models import Job, JobResult, WorkerContext
from reasondb.coordinator.producers import (
    ablation,
    adaptive_sampling,
    baselines,
    kv_operator,
    label_reference,
    operator_count,
    parameter_sweep,
    reorder_only,
    reordering,
    run_benchmark,
    sample_size,
    tuning,
)


class Producer(NamedTuple):
    enumerate_jobs: Callable[..., List[Job]]
    run_job: Callable[[Job, WorkerContext], JobResult]
    merge: Callable[[str, List[str]], List[Path]]
    #: Score one finished job against the label jobs' output dirs, reporting its
    #: accuracy to the monitor as soon as those labels exist, and return the label set
    #: names it scored (see ``coordinator.scoring``). Optional because it only makes
    #: sense for a producer that has separate label jobs at all.
    score_job: Optional[Callable[[Job, List[str]], List[str]]] = None


def _producer(module) -> Producer:
    return Producer(
        enumerate_jobs=module.enumerate_jobs,
        run_job=module.run_job,
        merge=module.merge,
        score_job=getattr(module, "score_job", None),
    )


PRODUCERS: Dict[str, Producer] = {
    module.PRODUCER_NAME: _producer(module)
    for module in (
        parameter_sweep,
        baselines,
        sample_size,
        operator_count,
        kv_operator,
        tuning,
        ablation,
        adaptive_sampling,
        reordering,
        reorder_only,
        run_benchmark,
        # The third shape: a wrapper in front of `run_benchmark` rather than of
        # `parameter_sweep`, because the benchmarks carrying per-tuple ground truth are
        # fixed query sets and the sweep engine only resolves RANDOM_BENCHMARKS.
        label_reference,
    )
}


def get_producer(name: str) -> Producer:
    producer = PRODUCERS.get(name)
    assert producer is not None, (
        f"Unknown producer {name!r}; expected one of {sorted(PRODUCERS)}."
    )
    return producer
