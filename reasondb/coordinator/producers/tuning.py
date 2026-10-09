"""Experiment: does tuning operator parameters make compression operators redundant?

One axis swept - ``OptimizationConfig.tune_parameters`` on versus off - with everything
else held where the ``sample_size`` producer holds it: the default operator suite, one
profiling budget, one optimizer. With tuning off the optimizer may only *choose*
operators, with their thresholds frozen at their declared defaults; with it on it can
also move those thresholds and mix in the gold model. If the two arms land in the same
place, the search space is doing the work; if the tuned arm holds its guarantees with a
cheaper operator mix, the thresholds are.

Pairing it with ``--producer operator_count`` answers the sharper version of the
question - whether tuning *substitutes* for operators, rather than whether it helps at
one point - but that is two axes crossed and costs twice the sweep, so it is deliberately
not what this producer does.

``--approaches`` is pinned to ``optim_global`` here, and this is the one wrapper where
that is a real constraint rather than a simplification: ``lotus`` and ``abacus`` have no
parameter-tuning phase at all, so ``tune_parameters=False`` is meaningless for them and
``build_approach_executor`` refuses it. A "tuning off" row attributed to ``abacus`` would
be asserting something about the run that is not true.
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

PRODUCER_NAME = "tuning"

#: The same single point ``operator_count`` uses, and the top of ``sample_size``'s grid,
#: so all three experiments meet at one profiling budget - and that budget is
#: ``DEFAULT_SAMPLE_SIZE``, what the optimizer draws when nobody sweeps this axis.
DEFAULT_SAMPLE_SIZES = [100]


def enumerate_jobs(
    task_id: str, output_root: Path, args: argparse.Namespace
) -> List[Job]:
    reject_pinned(
        args, PRODUCER_NAME, "sweep_to_gold", "--sweep-to-gold",
        swept_by="operator_count",
        because="compares the two tuning arms over the default operator suite",
    )
    reject_pinned(
        args, PRODUCER_NAME, "approaches", "--approaches",
        swept_by="baselines",
        because=(
            "runs optim_global alone - lotus and abacus have no parameter-tuning "
            "phase, so there is no 'tuning off' arm for them to have"
        ),
    )
    reject_pinned(
        args, PRODUCER_NAME, "state_plan", "--state-plan",
        swept_by="baselines",
        because="compares the two tuning arms at the default operator suite",
    )
    return wrapper_enumerate(
        task_id,
        output_root,
        args,
        producer_name=PRODUCER_NAME,
        state_plan="default",
        sample_sizes=DEFAULT_SAMPLE_SIZES,
        tune_parameters=["true", "false"],
        approaches=["optim_global"],
    )


run_job = engine.run_job
score_job = engine.score_job
merge = wrapper_merge(PRODUCER_NAME)
