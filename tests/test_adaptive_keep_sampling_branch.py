"""The branch adaptive sampling exists to take: feasible now, cheaper if we sample more.

`post_optimization_check` has three outcomes, and this file covers the middle one:

    if not meets_targets[...]:  -> keep optimizing, the plan does not work yet
    if not ended:               -> the plan works, but a larger sample is predicted
                                   cheaper end to end, and a round remains to draw it
    else:                       -> ship it

The middle branch is the entire point of the feature, but the loop tests mock
``gd_optimize`` out wholesale and the grid test arranges slot 0 to win, so neither
reaches it. These tests drive the *real* `post_optimization_check` with the two heavy
stages stubbed, the way `test_monitor_optimizer_solve.py` does, and assert on which of the three
outcomes comes back.
"""

from types import SimpleNamespace

import pytest

try:
    import torch

    from reasondb.optimizer.base_optimizer import PipelineSearchSpace
    from reasondb.optimizer.gd_optimizer import (
        VIOLATION_LOSS_MULTIPLIER,
        DifferentiableConfig,
        GradientDescentOptimizer,
        OptimizationConfig,
        OptimizationLoss,
    )
    from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
except ImportError:  # pragma: no cover - optional deps
    pytest.skip("optimizer deps not installed", allow_module_level=True)


class _Logger:
    """`post_optimization_check` only calls `.info`/`.warning`."""

    def info(self, *args, **kwargs):
        pass

    warning = info

    def __truediv__(self, _name):
        return self


def _search_space() -> "PipelineSearchSpace":
    space = PipelineSearchSpace()
    space.add_operator_choice(
        step_id=0, cascade_id=0, level=0, operators=list(range(2))  # type: ignore[arg-type]
    )
    return space


def _check(
    *,
    per_budget_costs,
    feasible_per_budget,
    rounds_remaining,
    sample_frac=0.5,
    monkeypatch,
):
    """Run the real check with `num_budgets` slots and a hand-built loss.

    `per_budget_costs[b]` is the winning restart's per-tuple execution cost at budget
    slot `b`, and `feasible_per_budget[b]` whether that slot's best restart met the
    targets. Returns the `termination_criterion_met` flag.
    """
    num_budgets = len(per_budget_costs)
    optimizer_config = OptimizationConfig(device=torch.device("cpu"))
    optimizer = GradientDescentOptimizer(optimizer_config)
    config = DifferentiableConfig(
        search_space=_search_space(),
        rng=torch.Generator().manual_seed(0),
        num_initializations=4,
        num_budgets_to_test=num_budgets,
        num_methods=1,
        num_gold_mixing_params=[],
        batch_size=20,
    )
    config.init(optimizer_config)
    config.remaining_budget = 10_000

    # Jobs are budget-major: slot b owns [b * num_init, (b+1) * num_init). Within each
    # slot make restart 0 the winner, so `best_index` is 0 everywhere and the per-slot
    # numbers below are the ones the argmin actually reads.
    n_init = config.num_initializations
    violation = torch.full((config.num_jobs,), 0.5)
    costs = torch.ones(config.num_jobs, 1)
    for b, (cost, ok) in enumerate(zip(per_budget_costs, feasible_per_budget)):
        violation[b * n_init] = 0.0 if ok else 0.5
        costs[b * n_init] = cost
    loss = OptimizationLoss(
        precision_violation=violation * VIOLATION_LOSS_MULTIPLIER,
        recall_violation=torch.zeros(config.num_jobs),
        costs=costs,
        max_costs=torch.ones(config.num_jobs, 1),
    )

    fake_pass = SimpleNamespace(
        get_operator_received_data=lambda job_index: {},
        compute_selectivities=lambda cascade_id, job_index, profiling_output: {},
    )
    monkeypatch.setattr(optimizer, "simulate_all_cascades", lambda **kw: [fake_pass])
    monkeypatch.setattr(optimizer, "compute_loss", lambda **kw: loss)

    zero_cost = SimpleNamespace(get_cost=lambda cost_type: 0.0)
    met, _used, _best, _received, _sel = optimizer.post_optimization_check(
        profiler=None,
        pipeline=None,
        guarantees=[PrecisionGuarantee(0.8), RecallGuarantee(0.8)],
        config=config,
        profiling_output=SimpleNamespace(total_cost_per_sample=zero_cost),
        profiling_cost_so_far=zero_cost,
        sample_size=20,
        sample_frac=sample_frac,
        level=0,
        logger=_Logger(),
        report=False,
        rounds_remaining=rounds_remaining,
    )
    return met


# ── keep-sampling decisions ─────────────────────────────────────────────────────


def test_a_cheaper_larger_sample_keeps_sampling(monkeypatch):
    """Feasible at slot 0, but slot 2 is cheaper and a round remains: do not stop."""
    met = _check(
        per_budget_costs=[100.0, 90.0, 80.0],
        feasible_per_budget=[True, True, True],
        rounds_remaining=2,
        monkeypatch=monkeypatch,
    )
    assert met is False


def test_it_stops_when_the_current_sample_is_already_cheapest(monkeypatch):
    met = _check(
        per_budget_costs=[80.0, 90.0, 100.0],
        feasible_per_budget=[True, True, True],
        rounds_remaining=2,
        monkeypatch=monkeypatch,
    )
    assert met is True


def test_the_last_round_ships_the_plan_it_has(monkeypatch):
    """A cheaper slot with no round left to draw it is not a reason to defer -- doing so
    drops the caller into the highest-quality-operator fallback."""
    met = _check(
        per_budget_costs=[100.0, 80.0, 70.0],
        feasible_per_budget=[True, True, True],
        rounds_remaining=0,
        monkeypatch=monkeypatch,
    )
    assert met is True


def test_an_exhausted_table_also_ships(monkeypatch):
    """`sample_frac == 1.0` means the sample *is* the table, so `remaining` is zero."""
    met = _check(
        per_budget_costs=[100.0, 80.0],
        feasible_per_budget=[True, True],
        rounds_remaining=3,
        sample_frac=1.0,
        monkeypatch=monkeypatch,
    )
    assert met is True


def test_a_cheaper_but_infeasible_slot_is_not_chased(monkeypatch):
    """An infeasible plan often runs almost nothing, so it looks free. Deferring to it
    would spend a round reaching for a plan that does not meet the targets."""
    met = _check(
        per_budget_costs=[100.0, 0.1],
        feasible_per_budget=[True, False],
        rounds_remaining=3,
        monkeypatch=monkeypatch,
    )
    assert met is True


def test_an_infeasible_current_sample_keeps_optimizing(monkeypatch):
    met = _check(
        per_budget_costs=[100.0, 90.0],
        feasible_per_budget=[False, True],
        rounds_remaining=1,
        monkeypatch=monkeypatch,
    )
    assert met is False


def test_the_single_slot_shape_still_ships(monkeypatch):
    """The non-adaptive configuration: one budget slot, nothing to defer to."""
    met = _check(
        per_budget_costs=[42.0],
        feasible_per_budget=[True],
        rounds_remaining=0,
        monkeypatch=monkeypatch,
    )
    assert met is True
