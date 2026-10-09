"""Tests for the Abacus (ParetoCascades) baseline optimizer's search algorithm.

Covers the memo-table / rule-based plan-space search (`GroupTree`, `Task`,
`Rule` subclasses, driven through `ParetoCascades._optimize_from_stats`) and
the pareto-frontier utilities built on top of it (`get_best_operators`,
`get_best_operators_pareto_points`). These operate on plain
`{(cascade_id, level, operator_id): stats}` dicts and a plain dependency
dict, so they're exercised directly without needing a real `TuningPipeline`
or `ProfilingOutput`.

Also covers `abacus_optimize` itself -- the thin wrapper that computes
operator stats and rephrases index-based `pipeline.dependencies` into
(cascade_id, level)-keyed dependencies before delegating to
`_optimize_from_stats`.

Not covered: `get_operator_stats` itself (needs a full `ProfilingOutput`/
`TuningPipeline`), and `tune_pipeline`/reordering into a `TunedPipeline`
(needs a `Database`/`IntermediateState`).
"""

from types import SimpleNamespace

import pytest
import torch

from reasondb.optimizer.baselines.abacus_optimizer import (
    ParetoCascades,
    PhysicalExpression,
    VectorizedStats,
)
from reasondb.query_plan.physical_operator import CostType


def _cascades() -> ParetoCascades:
    return ParetoCascades(cost_type=CostType.RUNTIME)


def _op_stats(precision, recall, cost, selectivity):
    return {
        "precision": precision,
        "recall": recall,
        "cost": cost,
        "selectivity": selectivity,
    }


# --- plan-space search (reordering + dominance) ----------------------------


def test_independent_operators_reordered_to_cheaper_execution_order():
    """(0,0) and (0,1) have no dependency on each other. Running (0,0) first
    is cheaper (10 + 0.5*100=60) than running (0,1) first (100 + 0.9*10=109),
    and both leave precision/recall at 1.0, so only the (0,0)-first plan
    should survive on the pareto frontier."""
    cascades = _cascades()
    op_stats = {
        (0, 0, 0): _op_stats(1.0, 1.0, cost=10, selectivity=0.5),
        (0, 1, 0): _op_stats(1.0, 1.0, cost=100, selectivity=0.9),
    }
    dependencies = {(0, 0): set(), (0, 1): set()}

    pareto = cascades._optimize_from_stats(op_stats, dependencies)

    assert len(pareto.op_keys) == 1
    physical_ids = tuple(pe.physical_operator_id for pe in pareto.op_keys[0])
    assert physical_ids == ((0, 0, 0), (0, 1, 0))
    assert pareto.get_metric_data("cost").item() == pytest.approx(60.0)


def test_dependency_forces_execution_order_even_when_costlier():
    """Same shape as above but with costs/selectivities swapped so the
    unconstrained optimum would run (0,1) first (cost 60) -- except (0,1) is
    declared to depend on (0,0), which must rule that ordering out entirely,
    leaving only the costlier (0,0)-first plan (cost 109)."""
    cascades = _cascades()
    op_stats = {
        (0, 0, 0): _op_stats(1.0, 1.0, cost=100, selectivity=0.9),
        (0, 1, 0): _op_stats(1.0, 1.0, cost=10, selectivity=0.5),
    }
    dependencies = {(0, 0): set(), (0, 1): {(0, 0)}}

    pareto = cascades._optimize_from_stats(op_stats, dependencies)

    assert len(pareto.op_keys) == 1
    physical_ids = tuple(pe.physical_operator_id for pe in pareto.op_keys[0])
    assert physical_ids == ((0, 0, 0), (0, 1, 0))
    assert pareto.get_metric_data("cost").item() == pytest.approx(109.0)


def test_dominated_physical_candidate_is_pruned():
    """Two candidate implementations of the same logical operator: (0,0,1)
    beats (0,0,0) on every metric (higher precision, higher recall, lower
    cost), so it should dominate (0,0,0) off the frontier entirely."""
    cascades = _cascades()
    op_stats = {
        (0, 0, 0): _op_stats(0.90, 0.90, cost=50, selectivity=0.5),
        (0, 0, 1): _op_stats(0.99, 0.99, cost=10, selectivity=0.5),
    }
    dependencies = {(0, 0): set()}

    pareto = cascades._optimize_from_stats(op_stats, dependencies)

    assert len(pareto.op_keys) == 1
    (winner,) = pareto.op_keys[0]
    assert winner.physical_operator_id == (0, 0, 1)
    assert pareto.get_metric_data("cost").item() == pytest.approx(10.0)


# --- abacus_optimize wrapper: stats + dependency rephrasing ----------------


def test_abacus_optimize_rephrases_index_dependencies_before_delegating(monkeypatch):
    """abacus_optimize should turn pipeline.dependencies (a list of sets of
    *step indexes*) into the (cascade_id, level)-keyed dependency dict
    `_optimize_from_stats` expects, using pipeline.steps_in_order_with_ids
    for the index -> (cascade_id, level) mapping -- then produce the same
    result as calling `_optimize_from_stats` directly with the equivalent
    dependency dict (this is the same scenario as the dependency-ordering
    test above, driven through the public entry point instead)."""
    cascades = _cascades()
    op_stats = {
        (0, 0, 0): _op_stats(1.0, 1.0, cost=100, selectivity=0.9),
        (0, 1, 0): _op_stats(1.0, 1.0, cost=10, selectivity=0.5),
    }
    monkeypatch.setattr(
        ParetoCascades,
        "get_operator_stats",
        lambda self, profiling_output, pipeline, level: op_stats,
    )
    pipeline = SimpleNamespace(
        steps_in_order_with_ids=[(0, 0, None), (0, 1, None)],
        dependencies=[set(), {0}],  # step 1 ((0,1)) depends on step 0 ((0,0))
    )

    pareto = cascades.abacus_optimize(profiling_output=object(), pipeline=pipeline)

    assert len(pareto.op_keys) == 1
    physical_ids = tuple(pe.physical_operator_id for pe in pareto.op_keys[0])
    assert physical_ids == ((0, 0, 0), (0, 1, 0))
    assert pareto.get_metric_data("cost").item() == pytest.approx(109.0)


# --- get_best_operators / get_best_operators_pareto_points -----------------


def _make_pareto(candidates):
    """candidates: list of (physical_operator_id, precision, recall, cost)."""
    op_keys = [
        (PhysicalExpression(frozenset(), pid, {}),) for pid, _, _, _ in candidates
    ]
    data = torch.tensor(
        [[-precision, -recall, cost] for _, precision, recall, cost in candidates]
    )
    selectivities = torch.full((len(candidates),), 0.5)
    return VectorizedStats(
        data=data,
        op_keys=op_keys,
        metric_keys=["precision", "recall", "cost"],
        selectivities=selectivities,
    )


def test_get_best_operators_picks_cheapest_among_those_meeting_targets():
    cascades = _cascades()
    pareto = _make_pareto(
        [
            ((0, 0, 0), 0.95, 0.95, 50),  # meets targets, not cheapest
            ((0, 0, 1), 0.99, 0.99, 80),  # meets targets, priciest
            ((0, 0, 2), 0.80, 0.99, 10),  # cheapest overall, fails precision
            ((0, 0, 3), 0.92, 0.91, 40),  # meets targets, cheapest of those
        ]
    )

    operators = cascades.get_best_operators(
        pareto, precision_target=0.9, recall_target=0.9
    )

    assert operators == ((0, 0, 3),)


def test_get_best_operators_falls_back_to_cheapest_when_none_meet_target():
    """If no candidate meets the guarantee, the (~mask)*100_000 penalty is
    applied uniformly, so argmin still just picks the globally cheapest
    candidate rather than raising."""
    cascades = _cascades()
    pareto = _make_pareto(
        [
            ((0, 0, 0), 0.5, 0.5, 10),
            ((0, 0, 1), 0.4, 0.4, 20),
        ]
    )

    operators = cascades.get_best_operators(
        pareto, precision_target=0.99, recall_target=0.99
    )

    assert operators == ((0, 0, 0),)


def test_get_best_operators_pareto_points_centers_on_target_by_cost_order():
    cascades = _cascades()
    # Costs strictly increasing 10..50; index 0 fails the precision target,
    # so index 1 (cost=20) is the cheapest qualifying candidate and becomes
    # the center point neighbors are chosen around.
    pareto = _make_pareto(
        [
            ((0, 0, 0), 0.50, 0.99, 10),  # fails precision target
            ((0, 0, 1), 0.95, 0.95, 20),  # target: cheapest that qualifies
            ((0, 0, 2), 0.95, 0.95, 30),
            ((0, 0, 3), 0.95, 0.95, 40),
            ((0, 0, 4), 0.95, 0.95, 50),
        ]
    )

    plans = cascades.get_best_operators_pareto_points(
        pareto, precision_target=0.9, recall_target=0.9, num_points=3
    )

    # offsets [0, -1, +1] around the cost-sorted target position -- note the
    # -1 neighbor (index 0) doesn't itself meet the target, by design (see
    # get_best_operators_pareto_points's docstring).
    assert plans == [((0, 0, 1),), ((0, 0, 0),), ((0, 0, 2),)]


def test_get_best_operators_pareto_points_skips_out_of_range_neighbors():
    """With the target at the very start of the cost order (position 0
    after sorting), the '-1' neighbor offset would go out of range and
    should be skipped rather than raising or wrapping around."""
    cascades = _cascades()
    pareto = _make_pareto(
        [
            ((0, 0, 0), 0.95, 0.95, 10),  # target: cheapest, meets it
            ((0, 0, 1), 0.95, 0.95, 20),
            ((0, 0, 2), 0.95, 0.95, 30),
        ]
    )

    plans = cascades.get_best_operators_pareto_points(
        pareto, precision_target=0.9, recall_target=0.9, num_points=3
    )

    # offsets [0, -1, +1]; -1 is out of range (target already at position 0)
    # and is dropped, leaving only the target and its one pricier neighbor.
    assert plans == [((0, 0, 0),), ((0, 0, 1),)]
