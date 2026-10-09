"""Experiment: what does ONE KV-compressed operator cost on disk, and what does it buy?

The two storage experiments either side of this one answer a coarser question. The greedy
walk of ``operator_count`` prices a whole search space at a time, so every point on its
curve is several caches and a total footprint, and the walk never visits a state holding
exactly one. ``ablation`` prices the deployed suite against no compression at all. Neither
can say whether a *particular* compressed operator is worth materializing, which is the
question a deployment actually asks: there is a disk budget, and a grid of
(model size x compression ratio) to spend it on.

Every point here is the gold model plus one compressed operator, over the grid
``operator_count`` sweeps, plus one further point that is the vanilla operators alone:

=====  ========================  ==========================================  ============
step   search space              footprint                                   reference
=====  ========================  ==========================================  ============
0      both sizes' vanilla ops   0 bytes                                     yes
1..n   gold + one compressed op  exactly that one cache                      no
=====  ========================  ==========================================  ============

Two things follow from a state holding one level and nothing else:

- ``storage_bytes`` is that operator's own footprint rather than a sum, and
  ``storage_bytes / cache_entries`` is what it costs per row, which is comparable across
  benchmarks.
- a runtime difference against step 0 has one cache behind it, so "beneficial" is a
  statement about an operator instead of about a menu.

Step 0 is in the same task rather than read off ``abl01`` deliberately. The speedup is
paired per (query, guarantee) - the same query set, the same replay - and needs no
cross-task check that two runs drew the same queries. That the two are nevertheless the
same search space (``plan_kv_operator_states`` and ``plan_ablation_states`` build it from
the same gate) makes ``abl01`` a determinism cross-check on this one, the way ``adapt01``'s
first arm duplicates ``base01``'s.

**What the gap carries.** Step 0 keeps both model sizes' vanilla operators and the later
steps keep only the large one, so step 0 -> step k adds a KV operator AND removes the
uncompressed small model. It prices a two-operator cascade against the vanilla suite, not
the marginal value of one operator; ``plan_kv_operator_states`` says how to get the other
reading.

**One per modality: ``--state-plan kv_operator_pairs``**. On a multimodal
benchmark the gap above is lopsided: a state holding only a text cache also loses the small
*image* model and gets nothing back for it, so ecommerce's text caches price "and image
falls back to gold" rather than the cache. ``plan_kv_operator_pairs_states`` holds one level
in each modality instead, matched by model size and compression rank, which makes every
modality gold plus one compressed operator. On a single-modality benchmark its states are
this plan's, so it suffices to name only the multimodal store in ``--simulate``:

    python scripts/run_coordinator.py --local --producer kv_operator --task-id <task-id> \\
      --state-plan kv_operator_pairs \\
      --simulate ecommerce_random_large=ecomm_precompute_kv.json

**Adding rather than swapping: ``--state-plan kv_operator_marginal``** (``kvop01``). The gap
above prices a cascade against the vanilla suite, because step 0 keeps both model sizes and
the later steps keep only the large one. This plan visits the same matched-pair states with
``state_includes_small_model_vanilla`` answering True at every step, so each state is step
0's suite *plus* one compressed operator per modality and the gap is what adding a KV
operator buys - the comparison ``abl01`` makes for the whole suite at once, one cache at a
time. It runs every benchmark, since the reading is not about multimodality:

    python scripts/run_coordinator.py --local --producer kv_operator --task-id kvop01 \\
      --state-plan kv_operator_marginal --simulate <every store>

**Direct mode only**: under ``--use-indexes`` one materialized level serves every
effective ratio above it, so a state would expose a menu of operators rather than one.

The remaining axes are held at one point each so the operators stay comparable: parameter
tuning on, one sample size (``DEFAULT_SAMPLE_SIZE``, so this describes a deployed
configuration), and the whole-pipeline optimizer. ``--approaches`` and ``--sample-sizes``
are defaulted rather than pinned, as in ``operator_count``, so the grid can be rerun
against another optimizer or profiling budget.

Needs the same recordings ``operator_count`` needs and nothing beyond them. Every state is
a level that walk also visits, so a store that serves ops01 serves this - and a store
narrowed with ``--precompute-states default``/``ablation`` serves neither.

    python scripts/run_coordinator.py --local --producer kv_operator --task-id <task-id> \\
      --benchmarks movie_random --simulate movie_random=movie_precompute_kv.json \\
      --precision-guarantees 0.7 --recall-guarantees 0.7
    python scripts/plot_sweep.py --experiment <task-id>
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
from reasondb.evaluation.parameter_sweep import KV_OPERATOR_PLANS

PRODUCER_NAME = "kv_operator"

#: One point, ``DEFAULT_SAMPLE_SIZE``: the budget the optimizer draws when that axis is
#: not swept. Shared with ``operator_count`` so the two storage experiments
#: describe the same profiling behaviour and can be read against each other.
DEFAULT_SAMPLE_SIZES = [100]

#: The whole-pipeline optimizer alone. Defaulted rather than pinned: the grid is worth
#: measuring against another approach, and nothing about the states depends on which one
#: searches them.
APPROACHES = ["optim_global"]


def enumerate_jobs(
    task_id: str, output_root: Path, args: argparse.Namespace
) -> List[Job]:
    # The one open choice on the state axis is how many compressed operators a state
    # holds: one in all (`kv_operator`) or one per modality (`kv_operator_pairs`, and
    # `kv_operator_marginal`, i.e. kvop01). Any other plan belongs to a different experiment.
    plan = getattr(args, "state_plan", None)
    if plan not in KV_OPERATOR_PLANS:
        reject_pinned(
            args, PRODUCER_NAME, "state_plan", "--state-plan",
            swept_by="baselines",
            because="is the state axis itself - one compressed operator per state (or per "
                    f"modality: {sorted(KV_OPERATOR_PLANS)}), over the whole materialized "
                    "grid",
        )
    reject_pinned(
        args, PRODUCER_NAME, "tune_parameters", "--tune-parameters",
        swept_by="tuning",
        because="holds parameter tuning on so the compressed operator is the only "
                "variable",
    )
    reject_pinned(
        args, PRODUCER_NAME, "sweep_to_gold", "--sweep-to-gold",
        swept_by="operator_count",
        because="visits every materialized level one at a time rather than walking down "
                "to gold",
    )
    return wrapper_enumerate(
        task_id,
        output_root,
        args,
        producer_name=PRODUCER_NAME,
        state_plan="kv_operator",
        sample_sizes=DEFAULT_SAMPLE_SIZES,
        tune_parameters=["true"],
        approaches=list(APPROACHES),
    )


run_job = engine.run_job
score_job = engine.score_job
merge = wrapper_merge(PRODUCER_NAME)
