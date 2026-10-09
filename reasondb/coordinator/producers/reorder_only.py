"""Experiment: what does reordering alone buy, with no operator selection at all?

The ``ablation`` producer prices the optimizer as a whole and ``reordering`` prices Step 4
inside the full system. Neither isolates reordering from operator selection, because in
both the optimizer is also choosing and tuning operators. This one does, by running an
optimizer that cannot choose: both arms take each step's last executable operator with its
declared default parameters, at ``--state-plan gold``.

=====  ====================  ==================  ============  =============================
arm    ``approach``          ``tune_parameters`` ``reorder``   what it is
=====  ====================  ==================  ============  =============================
1      ``no_optim``          true (inert)        false         gold everywhere, given order
2      ``no_optim_reorder``  true (inert)        true          gold everywhere, DP-reordered
=====  ====================  ==================  ============  =============================

Both arms build the *same plan*; ``ReorderOnlyOptimizer`` then profiles it and orders it
with the DP, so the gap between them is operator reordering and the profiling it needs,
and nothing else. That is the claim the paper's reordering step makes, measured on its
own rather than inside a bundle.

**Why arm 2 is its own optimizer rather than ``optim_global`` at a one-operator state.**
The gold state does not mean "every step has exactly one candidate": it empties the KV
levels, which leaves one *LLM* operator per modality, but the non-KV operators stay in the
toolbox at every state, so a semantic step still offers two or three - ``TraditionalFilter`` beside the 70B vanilla
on a text filter; ``TraditionalFilter``, ``ImageSimilarityFilter`` and the 72B vanilla on
an artwork image filter. ``GradientDescentOptimizer`` with ``tune_parameters=False`` does
not stop choosing between those: ``OptimizationConfig.__post_init__`` gives the whole step
budget to CHOOSE_OPERATORS precisely because there are no parameters left to tune. Such
an arm 2 could cascade from a cheap proxy that arm 1 cannot, and the gap would carry an
operator-selection win attributed to the reorderer.

**Arm 2 pays a profiling cost arm 1 does not**, and that is not an artifact to be
corrected away: the DP orders by per-tuple cost and selectivity, so somebody has to
measure them. The breakdown figure keeps ``tuning_runtime_s`` and ``execution_runtime_s``
as separate phases, so a reader sees the execution win *and* what buying it cost. On a
query set where the ordering has nothing to find, arm 2 is therefore expected to lose -
which is the honest shape of this measurement, not a bug in it.

**``tune_parameters`` is inert on both arms** and is pinned ``true`` on both, because
neither optimizer has a tuning phase and ``build_approach_executor`` rejects ``false`` for
an approach with nothing to disable. The column therefore does not differ between the
arms: ``approach`` and ``reorder`` are the only two that do.

**Both arms' accuracy is a tautology.** Every benchmark the sweep engine resolves is a
``RandomBenchmark`` with no ground truth, so rows are scored against *silver* - and a
silver pass is ``LabelOptimizer`` over the default suite, which picks the same gold
operator both arms pick. Their ``achieved_*`` columns are 1.0 by construction, and
reordering cannot change an answer anyway. This is a cost experiment; the comparison is
``total_runtime_s`` and its phases. A row that is *not* 1.0 means the arms differ in more
than their order and is worth chasing rather than reporting.

``--sample-sizes`` is rejected for the reason ``ablation`` rejects it: arm 1 draws no
profiling sample at all, so an explicit value would be attributed to a run that never
took one. Arm 2 keeps ``DEFAULT_SAMPLE_SIZE`` - it is the budget the DP's cost and
selectivity estimates come out of, and sweeping it is ``--producer sample_size``'s job.

``--precompute`` is rejected by ``scripts/run_coordinator.py`` for every wrapper producer;
record with ``--producer parameter_sweep``. Nothing new needs recording for this
experiment - the gold state's operators are the ones the ``ablation`` producer's
``no_optim`` arm already replays.

    python scripts/run_coordinator.py --local --producer reorder_only --task-id abl03 \\
      --benchmarks movie_random --simulate movie_random=movie_precompute_kv.json \\
      --precision-guarantees 0.5 0.7 0.9 --recall-guarantees 0.5 0.7 0.9
    python scripts/plot_sweep.py --experiment abl03
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

PRODUCER_NAME = "reorder_only"

#: ``(approach, tune_parameters, reorder)`` for the two arms, in reading order. Kept in
#: step with ``evaluation.sweep_frames``'s ``REORDER_ONLY_ARMS``, which labels exactly
#: these combinations.
#:
#: Each arm is enumerated on its own rather than filtered out of one 2x2x2 cross, because
#: the cross does not exist to be filtered: ``_assert_axes_are_compatible`` rejects
#: ``--approaches no_optim`` alongside ``--tune-parameters false`` at enumeration time,
#: and rightly so - it is the guard that stops a worker discovering hours in that
#: ``LabelOptimizer`` has no tuning phase to disable. Enumerating per arm asks the engine
#: only for points that are real.
ARMS = (
    ("no_optim", True, False),
    ("no_optim_reorder", True, True),
)

#: The approaches that never read a guarantee, hence the arms :data:`COLLAPSE_GUARANTEE_AXIS`
#: enumerates once per benchmark rather than once per pair. Both of this producer's arms
#: are in it: neither optimizer has a target to solve against, so the guarantee cross
#: would run the same pass three times. Kept in step with
#: ``sweep_frames.GUARANTEE_BLIND_APPROACHES``, which fans the rows back out at plot time.
GUARANTEE_BLIND_APPROACHES = frozenset({"no_optim", "no_optim_reorder"})

#: Whether to enumerate the guarantee-blind arms once per benchmark instead of once per
#: guarantee pair.
#:
#: Same argument as ``ablation.COLLAPSE_GUARANTEE_AXIS``, and here it applies to *both*
#: arms: neither optimizer reads a guarantee, so the guarantee cross produces
#: byte-identical runs of a full gold pass over every query. At three zipped guarantee
#: pairs that is four passes saved per benchmark rather than two - and the one it saves on
#: arm 2 is the expensive one, since that arm profiles as well. The cost is that
#: ``reorder_only.csv`` carries every row at one guarantee pair only;
#: ``scripts/plot_sweep.py`` fans them back out (``fan_out_guarantee_blind``).
COLLAPSE_GUARANTEE_AXIS = True


def enumerate_jobs(
    task_id: str, output_root: Path, args: argparse.Namespace
) -> List[Job]:
    """One engine enumeration per arm in :data:`ARMS`, merged on job id.

    Still the engine's own enumeration, so a job's spec, its id, its capabilities and the
    scoring path stay identical to every other producer's - only the axes differ. What
    the per-arm call costs is that the benchmark-level jobs (phase-0 ``filter_stats`` and
    the shared ``label`` pass) come back once per arm; their ids carry no axis, so the
    second copy is dropped by id rather than by kind, which stays right if the engine
    ever grows a third benchmark-level job.

    The combinations that are not arms are not enumerated at all: ``no_optim`` with
    reordering on is the pushdown ordering (``--producer reordering`` is where reordering
    is compared properly), and ``optim_global`` with reordering off at a one-operator
    state is a plan no phase of the optimizer can change - arm 1 with a profiling bill
    attached.
    """
    reject_pinned(
        args, PRODUCER_NAME, "sweep_to_gold", "--sweep-to-gold",
        swept_by="operator_count",
        because="pins the single gold state, where every step has one candidate",
    )
    reject_pinned(
        args, PRODUCER_NAME, "approaches", "--approaches",
        swept_by="parameter_sweep",
        because="pins the two approaches its arms are built from",
    )
    reject_pinned(
        args, PRODUCER_NAME, "state_plan", "--state-plan",
        swept_by="baselines",
        because=(
            "runs at the gold state, where the optimizer has nothing left to choose "
            "and ordering is the only remaining degree of freedom"
        ),
    )
    reject_pinned(
        args, PRODUCER_NAME, "tune_parameters", "--tune-parameters",
        swept_by="tuning",
        because="holds tuning off so ordering is the only variable between its arms",
    )
    reject_pinned(
        args, PRODUCER_NAME, "reorder", "--reorder",
        swept_by="reordering",
        because="is the reordering axis itself - its two arms are its two values",
    )
    reject_pinned(
        args, PRODUCER_NAME, "sample_sizes", "--sample-sizes",
        swept_by="sample_size",
        because="includes an arm that draws no profiling sample at all",
    )

    jobs: List[Job] = []
    seen_job_ids = set()
    seen_collapsed = set()
    for approach, tune, reorder in ARMS:
        for job in wrapper_enumerate(
            task_id,
            output_root,
            args,
            producer_name=PRODUCER_NAME,
            state_plan="gold",
            approaches=[approach],
            tune_parameters=[str(tune).lower()],
            reorder=[str(reorder).lower()],
        ):
            if job.job_id in seen_job_ids:
                continue
            # Only step jobs carry the optimizer axes; filter_stats and label jobs are
            # per benchmark and are already deduplicated by the check above.
            if (
                COLLAPSE_GUARANTEE_AXIS
                and job.spec.get("kind") == "step"
                and job.spec["approach"] in GUARANTEE_BLIND_APPROACHES
            ):
                # Per arm, not per benchmark: the two arms are different runs and both
                # need one job each.
                key = (job.benchmark, job.split, job.spec["approach"])
                if key in seen_collapsed:
                    continue
                seen_collapsed.add(key)
            seen_job_ids.add(job.job_id)
            jobs.append(job)
    return jobs


run_job = engine.run_job
score_job = engine.score_job
merge = wrapper_merge(PRODUCER_NAME)
