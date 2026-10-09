"""Experiment: how does Stretto compare with the baselines, at the configuration it ships?

One axis swept - which optimizer runs - over the default operator suite, one profiling
budget and parameter tuning on:

=====================  ==================================================================
``optim_global``       ``GradientDescentOptimizer`` in COMBO mode: operator choice and
                       tuning parameters searched jointly across the whole pipeline,
                       subject to the precision/recall guarantees.
``lotus``              ``LotusOptimizer``: cheapest proxy operators, then the gold model.
``abacus``             ``ParetoCascades``: Abacus-style per-step cascades.
=====================  ==================================================================

This is the headline comparison, and the reason it is a producer rather than three flags
is that "default settings" is a claim about four other axes, not one. ``--approaches
optim_global lotus abacus`` on the bare engine sweeps every *greedy state* as well, which
is a different experiment (``operator_count``) whose walk runs right past this one; the
default suite sits at the right-hand end of that curve - the walk's last cached state plus
the small models' vanilla operators - and nowhere near its left edge, so the comparison has
to name the state plan to be the comparison anyone means by it.

``DEFAULT_STATE_PLAN`` is that name - whatever ``get_default_configurator`` activates,
restricted to what the benchmark has materialized
(``evaluation.parameter_sweep.plan_default_state``). It is the same point ``sample_size``
and ``tuning`` hold, so all three experiments' arms are directly comparable and moving the
deployed suite (``DEFAULT_*_ACTIVE_DIRECT``) moves them together.

The plan is **defaulted rather than pinned**, like ``--approaches`` below and for the same
reason: a reader who wants this comparison at a different operator suite wants exactly
that and nothing else to change, and making them fork a producer to get it would put the
combination back in a lab notebook. What the producer keeps is the part of its claim that
is structural rather than conventional - ``--state-plan`` must name a *single-state* plan
(``SINGLE_STATE_PLANS``). A greedy walk here would enumerate every state, merge them into
one ``baselines.csv`` and present a curve as a configuration, which is exactly what the
paragraph above says this producer exists to prevent; it is refused with a pointer to
``operator_count``. ``mode01`` in ``scripts/cluster.yaml`` is the other single-state
plan in use: the optimizer-mode comparison over ``full``, every operator the benchmark
has materialized.

``--sample-sizes`` is left open rather than pinned, as in ``operator_count``: a reader who
wants this comparison at a different profiling budget wants exactly that and nothing else
to change. Its default here is ``DEFAULT_SAMPLE_SIZE``, which is also the top of
``sample_size``'s grid - so a baseline gap read here can be located on that curve rather
than assumed independent of it.

``--approaches`` is defaulted rather than pinned, so narrowing to one baseline is a flag
and not a fork. Widening it to ``no_optim`` is possible but is the ablation's third arm
(``--producer ablation``), where it is measured against a search space chosen to make the
gap mean one thing.

Widening it to ``optim_local``/``optim_shift_budget`` is the other case, and is a
comparison this producer is exactly right for: those are the same
``GradientDescentOptimizer`` in another ``GlobalOptimizationMode``, so what the gap
measures is the *scope of the search* - per-step, per-step with a shifted budget, or the
whole pipeline jointly - with the operator suite, the profiling budget and the guarantees
all held where the headline comparison holds them (``mode01`` in ``scripts/cluster.yaml``
runs it over the ``full`` state).
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
from reasondb.evaluation import parameter_sweep as psweep

PRODUCER_NAME = "baselines"

#: The three optimizers the comparison is between, in the order they are read.
DEFAULT_APPROACHES = ["optim_global", "lotus", "abacus"]

#: The same single point ``operator_count`` and ``tuning`` use, and the top of
#: ``sample_size``'s grid: ``DEFAULT_SAMPLE_SIZE``, what the optimizer draws when nobody
#: sweeps the axis. All three baselines consume a profiling budget, so this is one number
#: they are all held at rather than a knob only Stretto feels.
DEFAULT_SAMPLE_SIZES = [100]

#: The operator set this comparison is run at unless ``--state-plan`` names another one:
#: the deployed suite, which is what makes it the headline comparison rather than a point
#: on ``operator_count``'s curve.
DEFAULT_STATE_PLAN = "default"


def enumerate_jobs(
    task_id: str, output_root: Path, args: argparse.Namespace
) -> List[Job]:
    reject_pinned(
        args, PRODUCER_NAME, "sweep_to_gold", "--sweep-to-gold",
        swept_by="operator_count",
        because="compares the approaches over the default operator suite alone",
    )
    reject_pinned(
        args, PRODUCER_NAME, "tune_parameters", "--tune-parameters",
        swept_by="tuning",
        because=(
            "holds tuning on so the optimizer is the only variable - and lotus and "
            "abacus have no tuning phase to switch off"
        ),
    )
    state_plan = getattr(args, "state_plan", None) or DEFAULT_STATE_PLAN
    # A walk here would be operator_count's experiment wearing this producer's name: it
    # would enumerate every greedy state, write them all to `baselines.csv` as one
    # comparison, and report the left edge of a curve as the deployed configuration.
    # Refused rather than reinterpreted, the same way reject_pinned refuses a pinned axis.
    assert state_plan in psweep.SINGLE_STATE_PLANS, (
        f"--state-plan {state_plan} is a walk over several operator sets, and the "
        f"{PRODUCER_NAME} producer compares the approaches at one of them "
        f"(single-state plans: {sorted(psweep.SINGLE_STATE_PLANS)}). "
        "Use --producer operator_count to sweep that axis."
    )
    return wrapper_enumerate(
        task_id,
        output_root,
        args,
        producer_name=PRODUCER_NAME,
        state_plan=state_plan,
        sample_sizes=DEFAULT_SAMPLE_SIZES,
        tune_parameters=["true"],
        approaches=list(DEFAULT_APPROACHES),
    )


run_job = engine.run_job
score_job = engine.score_job
merge = wrapper_merge(PRODUCER_NAME)
