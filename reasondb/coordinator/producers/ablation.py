"""Experiment: what does each layer of Stretto buy, stripped off one at a time?

Three arms per benchmark, each sharing a search space with its neighbour so that every
gap has exactly one cause:

=====  ========  ================  ==============================  ====================
arm    ``step``  ``approach``      search space                    optimizer
=====  ========  ================  ==============================  ====================
1      0         ``optim_global``  the default suite               gradient descent
2      1         ``optim_global``  vanilla only, no KV operators   gradient descent
3      1         ``no_optim``      vanilla only, no KV operators   none
=====  ========  ================  ==============================  ====================

Arm 1 -> arm 2 removes the *KV-compressed operators*. Arm 2 -> arm 3 removes the
*optimizer*, leaving the plan that runs the highest-quality operator at every step. Both
axes already existed and are merely pinned here: the search space is the state axis
(``evaluation.parameter_sweep.plan_ablation_states``, whose second state is the same
vanilla-only one the greedy walk terminates on) and the optimizer is the approach axis
(``no_optim``, i.e. ``LabelOptimizer`` measured as an approach rather than run as a
labelling pass).

The vanilla-only **state** keeps both model sizes, so what arm 1 -> arm 2 takes away is
compression and nothing else. Without that it would hold a single LLM operator per
modality: the optimizer would have nothing to cascade from, and the step would remove the
small model along with the compressed caches, so the gap would conflate "no KV
compression" with "no cheap proxy at all".

Arm 3 sits in that same state and **never executes the small model** - ``LabelOptimizer``
takes each step's ``get_last_executable_operator_index()``, which is the large model's
vanilla operator whether or not the small one is in the space. That is what makes
arm 2 -> arm 3 a controlled step: the two arms share a search space, so they also share
the configurator's cardinality probe
(``potentially_run_outside_db(op_idx=chosen_operator_idx)``, which runs before any
optimizer and would be forced to gold in a one-operator space). A narrower space for arm 3
would put a configuration-cost difference into the gap. Arm 3 having a cheap operator
available and not taking it *is* the measurement.

The small models' ``vanilla=True`` spec rows come from the default suite
(``build_toolbox(include_small_model_vanilla=True)``), so arms 1 and 2 differ in the
KV-cached operators and nothing else. ``state_includes_small_model_vanilla`` answers False
for the greedy walks alone, keeping their terminal state at one operator per modality.

**This is a cost experiment, and arm 3's accuracy is a tautology.** Every benchmark the
sweep engine resolves is a ``RandomBenchmark`` with no ground truth, so ``label_set_for``
scores against *silver* - and a silver pass is ``LabelOptimizer`` over the default suite,
which picks the same vanilla operator arm 3 picks. Arm 3's ``achieved_*`` columns are
therefore 1.0 on every row by construction. That is the accuracy ceiling and a
determinism check on the replay, not a measurement; the comparison is
``total_runtime_s``. Arms 1 and 2 carry real accuracy numbers against silver.

**Arm 2 -> arm 3 is not structurally monotone**, though it is a fair fight: arm 2 has
the uncompressed small model to cascade from (plus ``TraditionalFilter`` and
``ImageSimilarityFilter``), so it can escalate only the tuples that need the large one -
but it pays a profiling cost first, and on a query whose predicate the small model cannot
answer it pays that for nothing. Arm 1 -> arm 2 is the gap with a floor under it.

**Why two states rather than one.** Arms 1 and 2 are both ``optim_global``, and a job's
id, its output directory and ``step_point_name`` (the executor name *and* the
results-cache directory) are keyed on ``step_idx`` and the approach. Distinguishing the
two arms by the state plan alone would have them share all three and replay each other's
cached results, so the ablation gets its own two-state plan instead of two single-state
ones.

``--precompute`` is rejected by ``scripts/run_coordinator.py`` for every wrapper
producer; record with ``--producer parameter_sweep``. One recording covers all three
arms: ``build_precompute_configurator`` takes the union of every materialized level
*and* passes ``include_vanilla=True``, ``_precompute_pipeline`` runs every candidate
operator rather than an optimizer's pick, and the LLM-derived operator configs are pinned
by interface *name*, which a vanilla-only toolbox exposes identically to the full one.

One exception: a store recorded against a search space without the small models' vanilla
operators does not hold them. Top it up by re-running its precompute job before replaying
this experiment - the resume markers are keyed per operator, so it records only what is
missing, on the small models' servers alone.

    python scripts/run_coordinator.py --local --producer ablation --task-id abl01 \\
      --benchmarks movie_random --simulate movie_random=movie_precompute_kv.json \\
      --precision-guarantees 0.7 --recall-guarantees 0.7
    python scripts/plot_sweep.py --experiment abl01
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
from reasondb.evaluation.parameter_sweep import ABLATION_VANILLA_STEP

PRODUCER_NAME = "ablation"

#: The two approaches the three arms draw from. `no_optim` is narrowed to one arm below;
#: `optim_global` supplies two, one per state.
APPROACHES = ["optim_global", "no_optim"]

#: Whether to enumerate arm 3 once per benchmark instead of once per guarantee pair.
#:
#: `LabelOptimizer` ignores guarantees entirely, so the engine's guarantee cross produces
#: byte-identical runs of the most expensive plan in the system - a full vanilla-70B pass
#: over every query, the same work the benchmark's silver label job is already doing. At
#: the usual three zipped guarantee pairs that is two passes saved per benchmark.
#:
#: The cost is that `ablation.csv` carries arm-3 rows at one guarantee pair only, so
#: anything faceting on the guarantee axis has to fan them back out;
#: `scripts/plot_sweep.py` does. Flip this to False to have the engine enumerate the
#: arm normally and get a complete CSV instead.
COLLAPSE_GUARANTEE_AXIS = True


def enumerate_jobs(
    task_id: str, output_root: Path, args: argparse.Namespace
) -> List[Job]:
    """The engine's jobs, narrowed from the 2x2 cross to the three arms.

    The engine crosses states with approaches, which would give a fourth point:
    ``no_optim`` at the *default* state. It is not a fourth arm - ``LabelOptimizer``
    picks the vanilla operator whichever baselines are materialized, so it would run the
    same plan as arm 3 and differ only in what the configurator's cardinality probe
    costs. Dropped rather than reported, so that "step 1 is the vanilla-only state" stays
    true of every row that carries it.
    """
    reject_pinned(
        args, PRODUCER_NAME, "sweep_to_gold", "--sweep-to-gold",
        swept_by="operator_count",
        because="visits the default suite and the vanilla-only state, and nothing between",
    )
    reject_pinned(
        args, PRODUCER_NAME, "approaches", "--approaches",
        swept_by="parameter_sweep",
        because="pins the two approaches its three arms are built from",
    )
    reject_pinned(
        args, PRODUCER_NAME, "state_plan", "--state-plan",
        swept_by="baselines",
        because="is the state axis itself - its arms are the default suite and the "
                "vanilla-only state",
    )
    reject_pinned(
        args, PRODUCER_NAME, "tune_parameters", "--tune-parameters",
        swept_by="tuning",
        because="holds parameter tuning on so the search space and the optimizer are "
                "the only variables",
    )
    reject_pinned(
        args, PRODUCER_NAME, "sample_sizes", "--sample-sizes",
        swept_by="sample_size",
        because="includes an arm that draws no profiling sample at all",
    )

    jobs: List[Job] = []
    seen_no_optim = set()
    for job in wrapper_enumerate(
        task_id,
        output_root,
        args,
        producer_name=PRODUCER_NAME,
        state_plan="ablation",
        approaches=list(APPROACHES),
        tune_parameters=["true"],
    ):
        spec = job.spec
        # Only step jobs carry an approach; filter_stats and label jobs pass through.
        if spec.get("kind") == "step" and spec["approach"] == "no_optim":
            if spec["step_idx"] != ABLATION_VANILLA_STEP:
                continue
            if COLLAPSE_GUARANTEE_AXIS:
                key = (job.benchmark, job.split)
                if key in seen_no_optim:
                    continue
                seen_no_optim.add(key)
        jobs.append(job)
    return jobs


run_job = engine.run_job
score_job = engine.score_job
merge = wrapper_merge(PRODUCER_NAME)
