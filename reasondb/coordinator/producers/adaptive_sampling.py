"""Experiment: does spending the profiling budget in rounds beat drawing it at once?

Two arms per benchmark, over the default operator suite with tuning on and
``optim_global`` as the only optimizer:

=====  ===============  ====================  ============================================
arm    ``sample_size``  ``adaptive_sampling``  what the optimizer does
=====  ===============  ====================  ============================================
1      100              off                   one sample of ``DEFAULT_SAMPLE_SIZE`` rows,
                                              every candidate profiled on all of them
2      160              on                    20 / 40 / 80 / 160 accumulated, stopping as
                                              soon as more rows stop paying for themselves,
                                              and dropping candidates no feasible restart
                                              picked between rounds
=====  ===============  ====================  ============================================

Arm 1 is the configuration Stretto ships: it is the same point ``baselines`` and
``tuning`` hold, and ``sample_size``'s grid ends one point past it, so a row here can be
located on all three of those experiments. Arm 2 is the whole adaptive apparatus at once
- the growing schedule *and* the operator pruning that only exists between rounds
(``OptimizationConfig.prune_unpicked_operators``, on by default and inert without
``adaptive_sampling``, so it needs no flag of its own here).

**The gap between the arms has two causes, and that is the deliberate shape of it.** Arm
2 draws a larger budget *and* draws it differently, so a win is "the adaptive
configuration beats the deployed one", not "rounds beat a single shot at equal budget".
The question this producer answers is the first one. Decomposing it costs exactly one
more point - a 160-row single-shot job, which is ``--producer sample_size --sample-sizes
160`` - and that point is deliberately not enumerated here, because paying for it on
every benchmark x guarantee pair is a different experiment's budget.

**Neither number is written down twice.** Arm 1's budget is
``sampler.DEFAULT_SAMPLE_SIZE`` itself, so moving the deployed budget moves this arm with
it, and arm 2's is derived from ``OptimizationConfig.first_round_rows`` by the doubling
the what-if grid already assumes: each round draws the sample it already has, so
``first_round_rows * 2 ** (rounds - 1)`` is the budget a whole number of rounds reaches.
20 x 2^3 = 160 is that ladder at ``ADAPTIVE_ROUNDS`` = 4. A hardcoded 160 would silently
become a ragged schedule (a clipped last round, as the default budget of 100 already is)
the moment either constant moved.

**Two arms out of a 2x2, narrowed at enumeration.** The engine crosses
``--sample-sizes`` with ``--adaptive-sampling``, which also offers 100-adaptive and
160-single-shot; both are dropped here, the same way ``ablation`` drops the two points of
its cross that are not arms. Their job ids and results-cache directories differ from the
arms' (``_axis_suffix``/``step_point_name`` spell out both axes), so nothing the two arms
run is shared with a point this producer declined to enumerate.

``--precompute`` is rejected by ``scripts/run_coordinator.py`` for every wrapper
producer; record with ``--producer parameter_sweep``. Nothing here needs a top-up: both
arms profile the *default* suite, which is a search space every store of any age covers,
and how many rows an optimizer draws does not change which operators the recording is
keyed on.

    python scripts/run_coordinator.py --local --producer adaptive_sampling \\
      --task-id adapt01 --benchmarks movie_random \\
      --simulate movie_random=movie_precompute_kv.json \\
      --precision-guarantees 0.7 --recall-guarantees 0.7
"""

import argparse
from pathlib import Path
from typing import List, Tuple

from reasondb.coordinator.models import Job
from reasondb.coordinator.producers import parameter_sweep as engine
from reasondb.coordinator.producers.experiments import (
    reject_pinned,
    wrapper_enumerate,
    wrapper_merge,
)
from reasondb.optimizer.gd_optimizer import OptimizationConfig
from reasondb.optimizer.sampler import DEFAULT_SAMPLE_SIZE

PRODUCER_NAME = "adaptive_sampling"

#: Rounds the adaptive arm is sized to draw. Four is what `max_sampling_rounds` derives
#: from the budget below, by construction rather than by coincidence - the budget is
#: computed from it.
ADAPTIVE_ROUNDS = 4

#: The adaptive arm's row budget: the schedule 20 / 40 / 80 / 160, reached by a whole
#: number of doubling rounds rather than by clipping the last one. Derived, so that
#: moving `first_round_rows` moves the schedule instead of breaking it.
ADAPTIVE_SAMPLE_SIZE = OptimizationConfig.first_round_rows * 2 ** (ADAPTIVE_ROUNDS - 1)

#: `(sample_size, adaptive_sampling)` per arm, in the order they are read. Arm 1 defers
#: to `DEFAULT_SAMPLE_SIZE` rather than copying it: the claim "arm 1 is what Stretto
#: ships" is only true while it follows that constant.
ARMS: List[Tuple[int, bool]] = [
    (DEFAULT_SAMPLE_SIZE, False),
    (ADAPTIVE_SAMPLE_SIZE, True),
]


def enumerate_jobs(
    task_id: str, output_root: Path, args: argparse.Namespace
) -> List[Job]:
    """The engine's jobs, narrowed from the sample-size x adaptive cross to the arms.

    The two dropped points are real sweep points, not invalid ones - they are simply a
    different experiment (the equal-budget comparison, see the module docstring). Dropped
    at enumeration rather than left in, so that every row of ``adaptive_sampling.csv``
    belongs to one of the two arms this producer is a claim about.
    """
    reject_pinned(
        args, PRODUCER_NAME, "sweep_to_gold", "--sweep-to-gold",
        swept_by="operator_count",
        because="compares the two sampling arms over the default operator suite",
    )
    reject_pinned(
        args, PRODUCER_NAME, "tune_parameters", "--tune-parameters",
        swept_by="tuning",
        because="holds parameter tuning on so the sampling schedule is the variable",
    )
    reject_pinned(
        args, PRODUCER_NAME, "state_plan", "--state-plan",
        swept_by="baselines",
        because="compares the two sampling arms at the default operator suite",
    )
    reject_pinned(
        args, PRODUCER_NAME, "approaches", "--approaches",
        swept_by="baselines",
        because=(
            "runs optim_global alone - lotus and abacus draw one sample and have no "
            "round to reconsider it in, so there is no adaptive arm for them to have"
        ),
    )
    reject_pinned(
        args, PRODUCER_NAME, "sample_sizes", "--sample-sizes",
        swept_by="sample_size",
        because="pins one budget per arm: the deployed one, and the adaptive schedule's",
    )
    reject_pinned(
        args, PRODUCER_NAME, "adaptive_sampling", "--adaptive-sampling",
        swept_by="parameter_sweep",
        because="pins the two arms it is built from, one per sampling mode",
    )

    arms = set(ARMS)
    jobs: List[Job] = []
    for job in wrapper_enumerate(
        task_id,
        output_root,
        args,
        producer_name=PRODUCER_NAME,
        state_plan="default",
        sample_sizes=[size for size, _ in ARMS],
        adaptive_sampling=[str(adaptive).lower() for _, adaptive in ARMS],
        tune_parameters=["true"],
        approaches=["optim_global"],
    ):
        spec = job.spec
        # Only step jobs carry the two axes; filter_stats and label jobs pass through.
        if spec.get("kind") == "step":
            if (spec["sample_size"], spec["adaptive_sampling"]) not in arms:
                continue
        jobs.append(job)
    return jobs


run_job = engine.run_job
score_job = engine.score_job
merge = wrapper_merge(PRODUCER_NAME)
