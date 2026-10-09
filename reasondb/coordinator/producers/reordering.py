"""Experiment: what does reordering the selected operators buy the full system?

One axis swept - ``OptimizationConfig.reorder``, on versus off - over the default
operator suite, with everything else held where the ``tuning`` and ``sample_size``
producers hold it: one optimizer, one profiling budget, parameter tuning on. Two arms:

=====  ============  ============================================================
arm    ``reorder``   what it is
=====  ============  ============================================================
1      true          ``Stretto`` - the DP reorderer places the chosen operators
2      false         ``Stretto, no reordering`` - the plan runs as it was built
=====  ============  ============================================================

Arm 2 takes the **whole feature** off, not merely the final permutation:
``GradientDescentOptimizer.get_reorderer`` returns ``NoOpReorderer`` *and*
``compute_reordered_cascade_order`` returns ``None``, so the differentiable cost model
costs every cascade as if it saw the full input. Leaving the cost model order-aware would
have arm 2's optimizer select operators for an execution order it then does not get, and
the gap between the arms would partly measure a cost model lying to one of them rather
than reordering itself.

**What the arms should and should not differ in.** Reordering changes the cost of a plan,
not its answers, so the two arms' ``achieved_precision``/``achieved_recall`` should track
each other closely and the gap should land in ``execution_runtime_s``. That is why this
preset keeps the target-met figure rather than dropping it the way ``ablation`` does: here
it is a genuine check, not a tautology. It is not an exact identity - the optimizer solves
against a different cost model in arm 2 and may land on a different operator mix - which
is the honest reading of "what reordering buys the system", as opposed to permuting one
fixed plan two ways.

``--approaches`` is pinned to ``optim_global`` for the same reason ``tuning`` pins it:
``lotus`` has no reordering step at all and ``abacus`` reorders through its own
``BasicReorderer``, which this flag does not reach, so a ``reorder=false`` row attributed
to either would assert something about the run that is not true. ``no_optim`` *does* have
a reordering step this flag reaches, but it is a pushdown ordering rather than the DP -
that comparison is ``--producer reorder_only``, which isolates it properly.

    python scripts/run_coordinator.py --local --producer reordering --task-id abl02 \\
      --benchmarks movie_random --simulate movie_random=movie_precompute_kv.json \\
      --precision-guarantees 0.5 0.7 0.9 --recall-guarantees 0.5 0.7 0.9
    python scripts/plot_sweep.py --experiment abl02
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

PRODUCER_NAME = "reordering"

#: The same single point ``tuning`` and ``operator_count`` use, and the top of
#: ``sample_size``'s grid - ``DEFAULT_SAMPLE_SIZE``, what the optimizer draws when nobody
#: sweeps that axis. Reordering is measured at a profiling budget someone would deploy,
#: and the four experiments still meet at one point on it.
DEFAULT_SAMPLE_SIZES = [100]


def enumerate_jobs(
    task_id: str, output_root: Path, args: argparse.Namespace
) -> List[Job]:
    reject_pinned(
        args, PRODUCER_NAME, "sweep_to_gold", "--sweep-to-gold",
        swept_by="operator_count",
        because="compares the two reordering arms over the default operator suite",
    )
    reject_pinned(
        args, PRODUCER_NAME, "approaches", "--approaches",
        swept_by="baselines",
        because=(
            "runs optim_global alone - lotus never reorders and abacus reorders "
            "through a path this axis does not reach"
        ),
    )
    reject_pinned(
        args, PRODUCER_NAME, "state_plan", "--state-plan",
        swept_by="baselines",
        because="compares the two reordering arms at the default operator suite",
    )
    reject_pinned(
        args, PRODUCER_NAME, "tune_parameters", "--tune-parameters",
        swept_by="tuning",
        because="holds parameter tuning on so reordering is the only variable",
    )
    return wrapper_enumerate(
        task_id,
        output_root,
        args,
        producer_name=PRODUCER_NAME,
        state_plan="default",
        sample_sizes=DEFAULT_SAMPLE_SIZES,
        tune_parameters=["true"],
        approaches=["optim_global"],
        reorder=["true", "false"],
    )


run_job = engine.run_job
score_job = engine.score_job
merge = wrapper_merge(PRODUCER_NAME)
