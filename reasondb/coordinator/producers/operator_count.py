"""Experiment: what does a larger operator search space buy, and what does it cost?

The greedy storage walk *is* the operator-count axis. In the default (no-index) serving
mode every retained compression level has its own dedicated cache and its own operator,
so each step of ``plan_greedy_states`` removes exactly one baseline from the search
space; the storage footprint the sweep reports and the number of operators the optimizer
can choose between fall together, step for step.

This producer runs that walk **to gold**: past the greedy planner's usual stop (one
compressed cache per slot) to the state that materializes nothing at all and leaves only
the vanilla operators - which are exactly the gold operators the profiler derives its
labels from. So the curve spans the entire axis, from every baseline on disk down to the
gold model alone, and the far end is the honest "no compression at all" baseline rather
than an arbitrary cheapest-compressed point.

Where the walk *starts* is not this producer's choice either: ``prepare_sweep`` derives
step 0 from the caches materialized on disk, never from ``get_default_configurator``. The
sweep therefore covers the full search space independently of the default suite - which
is what makes the default suite locatable *on* this curve rather than being the curve's
own left edge.

The other axes are held at one point each so the steps stay comparable: parameter tuning
on, one sample size, one optimizer (widen with ``--approaches`` to put a baseline on the
same axis). ``--sample-sizes`` is left open rather than pinned, because a reader who wants
this curve at a different profiling budget wants exactly that and nothing else to change.
"""

import argparse
from pathlib import Path
from typing import List

from reasondb.coordinator.models import Job
from reasondb.coordinator.producers import parameter_sweep as engine
from reasondb.coordinator.producers.experiments import (
    reject_pinned,
    wrapper_enumerate,
    wrapper_merge,
)

PRODUCER_NAME = "operator_count"

#: One point, and it is ``DEFAULT_SAMPLE_SIZE`` - what the optimizer actually draws when
#: nobody sweeps this axis, so the operator-count curve describes a deployed
#: configuration rather than a cheaper one nobody runs. It is also the top of the
#: ``sample_size`` producer's own grid, so the two curves still share an anchor: a reader
#: can find this sweep's profiling budget on that experiment's x-axis instead of having
#: to assume the difference between them does not matter.
DEFAULT_SAMPLE_SIZES = [100]


def enumerate_jobs(
    task_id: str, output_root: Path, args: argparse.Namespace
) -> List[Job]:
    reject_pinned(
        args, PRODUCER_NAME, "tune_parameters", "--tune-parameters",
        swept_by="tuning",
        because="holds parameter tuning on so the operator count is the only variable",
    )
    reject_pinned(
        args, PRODUCER_NAME, "state_plan", "--state-plan",
        swept_by="baselines",
        because="is the state axis itself - it walks every state down to gold",
    )
    return wrapper_enumerate(
        task_id,
        output_root,
        args,
        producer_name=PRODUCER_NAME,
        state_plan="greedy_to_gold",
        sample_sizes=DEFAULT_SAMPLE_SIZES,
        tune_parameters=["true"],
    )


run_job = engine.run_job
score_job = engine.score_job
merge = wrapper_merge(PRODUCER_NAME)
