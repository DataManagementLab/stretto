"""Experiment: what does a larger profiling sample buy, over the default operator suite?

One axis swept - the absolute number of rows ``UniformSampler`` draws to estimate
operator quality during tuning - against a fixed search space, a fixed optimizer and
fixed tuning behaviour. A larger sample should sharpen the optimizer's estimates at the
cost of a longer tuning phase, with diminishing returns; where the returns stop
diminishing is the whole question, so the grid is geometric rather than uniform.

**The operator suite is the default one**, via ``state_plan="default"``: whatever
``get_default_configurator`` activates, restricted to what this benchmark has
materialized (see ``evaluation.parameter_sweep.plan_default_state``). That is deliberate
and is why there is no "profile N operators" flag here. The question "how big a sample
does the optimizer need" only has a useful answer relative to a search space someone
would actually deploy, and the default suite is that space by definition. It also means
moving what this experiment profiles is one edit to ``DEFAULT_*_ACTIVE_DIRECT``, which
moves the default suite, ``run_benchmark`` and this experiment together rather than
letting them drift.

To compare the baselines along this axis, widen ``--approaches``.
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

PRODUCER_NAME = "sample_size"

#: Geometric-ish, four points over the range where the knee is expected. Deliberately not
#: a uniform grid: doubling from 10 to 25 changes the estimate far more than doubling from
#: 50 to 100, and a uniform grid spends most of its budget on the flat part of the curve.
DEFAULT_SAMPLE_SIZES = [10, 25, 50, 100]


def enumerate_jobs(
    task_id: str, output_root: Path, args: argparse.Namespace
) -> List[Job]:
    reject_pinned(
        args, PRODUCER_NAME, "sweep_to_gold", "--sweep-to-gold",
        swept_by="operator_count",
        because="profiles the default operator suite at one fixed state",
    )
    reject_pinned(
        args, PRODUCER_NAME, "tune_parameters", "--tune-parameters",
        swept_by="tuning",
        because="holds parameter tuning on so the sample size is the only variable",
    )
    reject_pinned(
        args, PRODUCER_NAME, "state_plan", "--state-plan",
        swept_by="baselines",
        because="measures the sample-size curve at the default operator suite",
    )
    return wrapper_enumerate(
        task_id,
        output_root,
        args,
        producer_name=PRODUCER_NAME,
        state_plan="default",
        sample_sizes=DEFAULT_SAMPLE_SIZES,
        tune_parameters=["true"],
    )


run_job = engine.run_job
score_job = engine.score_job
merge = wrapper_merge(PRODUCER_NAME)
