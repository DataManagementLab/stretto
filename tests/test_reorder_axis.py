"""The ``reorder`` sweep axis: the optimizer's Step 4, turned off.

Reordering is on by default, and two of these tests are about *names* rather than
behaviour. A job id and a results-cache directory are keyed on
the sweep point's axes, and the result cache is keyed only by ``(query, guarantees)``
underneath - so a naming scheme that spelled ``reorder=True`` out would rename every point
already recorded under ``benchmark_results/``, and every one of those replays would miss.
The axis is therefore suffixed only at its *off* value, and the two places that spell it
(``_axis_suffix`` for the job id, ``step_point_name`` for the executor name and the cache
directory) must agree, or two different sweep points would share a cache and replay each
other's answers - which is precisely the comparison the axis exists to make.

What is not covered here: a full ``tune_pipeline`` with reordering off, which needs a real
Profiler/TuningPipeline/Database. The seam between "the config says off" and "no reordering
happens" is ``get_reorderer``, and that is checked directly.
"""

import argparse
import asyncio

import pytest

from reasondb.coordinator.producers import parameter_sweep as producer
from reasondb.evaluation import parameter_sweep as psweep

try:
    import torch

    from reasondb.optimizer.gd_optimizer import GradientDescentOptimizer, OptimizationConfig
    from reasondb.optimizer.label_optimizer import LabelOptimizer
    from reasondb.optimizer.reorderer import DPReorderer, NoOpReorderer, Reorderer
    from reasondb.query_plan.physical_operator import CostType
except ImportError:  # pragma: no cover - same guard the sibling optimizer tests use
    pytest.skip("optimizer deps not installed", allow_module_level=True)


def _optimizer(**overrides) -> GradientDescentOptimizer:
    return GradientDescentOptimizer(
        OptimizationConfig(
            cost_type=CostType.RUNTIME, device=torch.device("cpu"), **overrides
        )
    )


class _FakePipeline:
    """Records the swaps ``Reorderer.reorder_steps`` performs on it."""

    def __init__(self, n: int):
        self.plan_steps = list(range(n))
        self.swaps = []

    def switch_nodes(self, i, j, database):
        self.swaps.append((i, j))
        self.plan_steps[i], self.plan_steps[j] = self.plan_steps[j], self.plan_steps[i]


# ── The reorderer itself ─────────────────────────────────────────────────────


def test_no_op_reorderer_performs_no_swaps():
    pipeline = _FakePipeline(4)

    result = NoOpReorderer().reorder(
        per_operator_and_sample_costs=[1.0, 2.0, 3.0, 4.0],
        pipeline=pipeline,
        dependencies=[set(), set(), set(), set()],
        selectivities=None,
        duplication_factors={},
        input_sizes={},
        database=object(),
        logger=None,
    )

    assert result is pipeline
    assert pipeline.swaps == [], "the identity permutation must move nothing"
    assert pipeline.plan_steps == [0, 1, 2, 3]


def test_reorder_steps_still_moves_things():
    """The control for the test above: an order that is not the identity does swap."""
    pipeline = _FakePipeline(3)

    Reorderer.reorder_steps(pipeline, [2, 0, 1], object())

    assert pipeline.swaps, "a real permutation must reach switch_nodes"


# ── The config knob ──────────────────────────────────────────────────────────


def test_get_reorderer_is_the_dp_by_default():
    assert isinstance(_optimizer().get_reorderer(), DPReorderer)


def test_get_reorderer_is_a_no_op_when_the_axis_is_off():
    assert isinstance(_optimizer(reorder=False).get_reorderer(), NoOpReorderer)


def test_reorder_off_also_takes_the_cost_model_order_blind():
    """Half-off would be worse than either arm.

    With ``order_aware_cost`` left on, the CHOOSE_PARAMETERS phase would price every
    cascade against the order the DP *would* have picked, and then the executor would run
    the plan in a different one - so the gap the ablation measures would carry a cost model
    lying to one arm as well as the reordering itself.
    """
    optimizer = _optimizer(reorder=False)
    assert optimizer.optimizer_config.order_aware_cost is True

    order = asyncio.run(
        optimizer.compute_reordered_cascade_order(
            profiler=None,
            pipeline=None,
            guarantees=[],
            config=None,
            sample=None,
            profiling_output=None,
            intermediate_state=None,
            duplication_factors={},
            input_sizes={},
            logger=None,
        )
    )

    assert order is None


def test_label_optimizer_keeps_its_pushdown_ordering_by_default():
    assert LabelOptimizer().reorder_steps_enabled is True
    assert LabelOptimizer(reorder=False).reorder_steps_enabled is False


# ── Naming: the axis must not rename any point already recorded ──────────────


def test_job_id_suffix_is_unchanged_when_reordering_is_on():
    assert producer._axis_suffix("optim_global", True, 100) == producer._axis_suffix(
        "optim_global", True, 100, False, True
    )


def test_job_id_suffix_spells_out_only_the_off_value():
    on = producer._axis_suffix("optim_global", True, 100, False, True)
    off = producer._axis_suffix("optim_global", True, 100, False, False)

    assert off == on + "-noreorder"


def test_point_name_is_unchanged_when_reordering_is_on():
    assert psweep.step_point_name(0, "optim_global", True, 100) == psweep.step_point_name(
        0, "optim_global", True, 100, False, True
    )


def test_point_name_separates_the_two_arms():
    """Two points that shared a name would share a results-cache directory, and the
    un-reordered arm would replay the reordered arm's answers."""
    on = psweep.step_point_name(0, "optim_global", True, 100, False, True)
    off = psweep.step_point_name(0, "optim_global", True, 100, False, False)

    assert on != off
    assert off == on + "_noreorder"


# ── Resolving the flag ───────────────────────────────────────────────────────


def test_resolve_reorder_defaults_to_on():
    assert producer.resolve_reorder(argparse.Namespace()) == [True]
    assert producer.resolve_reorder(argparse.Namespace(reorder=None)) == [True]


def test_resolve_reorder_takes_strings_and_bools_and_dedupes():
    assert producer.resolve_reorder(argparse.Namespace(reorder=["true", "false"])) == [
        True,
        False,
    ]
    assert producer.resolve_reorder(argparse.Namespace(reorder=[False, False])) == [False]


def test_resolve_reorder_rejects_anything_else():
    with pytest.raises(AssertionError, match="takes true/false"):
        producer.resolve_reorder(argparse.Namespace(reorder=["maybe"]))


# ── Which approaches the off value is meaningful for ─────────────────────────


@pytest.mark.parametrize("approach", ["lotus", "abacus", "no_optim_reorder"])
def test_reorder_off_is_rejected_for_the_optimizers_it_cannot_reach(approach):
    """``lotus`` never reorders, ``abacus`` reorders through its own ``BasicReorderer``,
    and ``no_optim_reorder`` *is* the reordering - turning it off there is ``no_optim``,
    the other arm of that experiment. A row labelled ``reorder=False`` for any of the
    three would describe a run that is not what happened. Caught at enumeration rather
    than on a worker hours in."""
    with pytest.raises(AssertionError, match="--reorder false"):
        producer._assert_axes_are_compatible(
            ["optim_global", approach], [True], False, [False], [None], [True, False]
        )


def test_reorder_off_is_allowed_for_no_optim():
    """It has a reordering step this flag does reach - ``LabelOptimizer.reorder``, the
    pushdown ordering the ``reorder_only`` experiment's floor arm turns off."""
    producer._assert_axes_are_compatible(
        ["no_optim", "optim_global"], [True], False, [False], [None], [True, False]
    )
