"""Adaptive sampling is off by default, and off means *nothing changes*.

The claim is not just
"the flag defaults to False" but that the whole chain hanging off it collapses: the
what-if grid is all-zero, so the bound extrapolation is a no-op and the optimizer-solve
term is zero; there is one budget slot, so the cost argmin is trivially slot 0; the
sampler draws its own budget in one shot; and no operator is ever pruned.

If any assertion here fails, non-adaptive runs have silently changed behavior.
"""

import pytest
import torch

from reasondb.optimizer.base_optimizer import PipelineSearchSpace
from reasondb.optimizer.gd_optimizer import (
    MAX_BUDGET_SLOTS,
    DifferentiableConfig,
    GradientDescentOptimizer,
    OptimizationConfig,
)
from reasondb.optimizer.sampler import DEFAULT_SAMPLE_SIZE


def _differentiable_config(config: OptimizationConfig) -> DifferentiableConfig:
    """A config over an empty search space -- enough for the job-axis arithmetic."""
    return DifferentiableConfig(
        PipelineSearchSpace(),
        rng=torch.Generator().manual_seed(0),
        num_initializations=config.num_initializations,
        num_budgets_to_test=config.num_budgets_to_test,
        num_methods=1,
        num_gold_mixing_params=[1],
        batch_size=config.batch_size,
    )


def test_the_one_switch_is_off_by_default():
    config = OptimizationConfig()
    assert config.adaptive_sampling is False


def test_the_derived_properties_collapse():
    """Everything the loop needs derives from `adaptive_sampling` being off."""
    config = OptimizationConfig()
    assert config.batch_size is None
    assert config.num_budgets_to_test == 1
    assert config.max_sampling_rounds == 1


def test_the_sampler_draws_the_whole_budget_in_one_go():
    sampler = GradientDescentOptimizer(OptimizationConfig()).get_sampler()
    # No per-round batch, because there are no rounds: the sampler draws the whole
    # `sample_size` at once. Both are plain counts -- the table's length never enters.
    assert sampler.batch_size is None
    assert sampler.sample_size == OptimizationConfig().sample_size == DEFAULT_SAMPLE_SIZE


def test_the_what_if_grid_proposes_nothing():
    """`config.batch_size or 0` is what reaches `get_what_if` at the call sites."""
    grid = GradientDescentOptimizer.get_what_if(
        batch_size=OptimizationConfig().batch_size or 0,
        num_jobs=1,
        num_budgets_to_test=OptimizationConfig().num_budgets_to_test,
        max_remaining=1_000_000,
        device=torch.device("cpu"),
    )
    assert grid.tolist() == [0]


def test_one_budget_slot_makes_the_cost_argmin_trivially_zero():
    """With a single slot, "is a bigger sample cheaper?" can only answer "no"."""
    total_cost = torch.tensor([[42.0]])
    meets_targets = torch.tensor([[True]])
    argmin = int(
        total_cost[0].masked_fill(~meets_targets[0], float("inf")).argmin().item()
    )
    assert argmin == 0


def test_an_infeasible_single_slot_is_still_reported_rather_than_chased():
    """masked_fill leaves an all-infeasible row at inf; argmin must not crash or wander."""
    total_cost = torch.tensor([[42.0]])
    meets_targets = torch.tensor([[False]])
    masked = total_cost[0].masked_fill(~meets_targets[0], float("inf"))
    assert int(masked.argmin().item()) == 0


def test_the_solve_cost_estimate_is_zero_before_any_solve():
    optimizer = GradientDescentOptimizer(OptimizationConfig())
    assert optimizer._solve_cost_estimate(0.0) == 0.0


def test_the_solve_cost_is_runtime_only():
    cost = OptimizationConfig().optimizer_solve_cost(2.5)
    assert cost.runtime == 2.5
    # A GD solve on your own GPU has no dollar price in this repo's model, and
    # `fake_cost` is a synthetic per-operator unit with no time interpretation.
    assert cost.monetary_cost == 0.0
    assert cost.fake_cost == 0.0


def test_the_non_adaptive_batch_size_never_gates_the_loop():
    optimizer = GradientDescentOptimizer(OptimizationConfig())
    batch = optimizer._next_batch_size(round_index=0, rows_drawn=0, total_cap=None)
    assert batch > 0


# ── and when it is switched on ──────────────────────────────────────────────────


def test_switching_it_on_arms_the_apparatus():
    config = OptimizationConfig(adaptive_sampling=True)
    assert config.batch_size == config.first_round_rows
    assert config.num_budgets_to_test > 1
    assert config.max_sampling_rounds > 1


def test_default_budget_round_schedule():
    """The default budget derives the expected round count and what-if width.

    100 rows starting at 20, each round drawing the sample it already has, is four
    rounds -- 20 / 20 / 40 / 20 drawn, 20 / 40 / 80 / 100 accumulated. The last round is
    clipped by the budget rather than doubling into it.

    The what-if axis is four slots wide, not five: the grid is only ever built *after* a
    round has been drawn, so at most three rounds remain and a fifth slot could only
    duplicate the fourth.
    """
    config = OptimizationConfig(adaptive_sampling=True)
    assert (config.sample_size, config.first_round_rows) == (100, 20)
    assert config.max_sampling_rounds == 4
    assert config.num_budgets_to_test == 4


def test_the_batch_grows_geometrically_and_respects_the_total_cap():
    optimizer = GradientDescentOptimizer(
        OptimizationConfig(adaptive_sampling=True, sample_size=160, first_round_rows=20)
    )
    # Each round draws the sample it already has, so the total doubles every round.
    assert optimizer._next_batch_size(0, rows_drawn=0, total_cap=160) == 20
    assert optimizer._next_batch_size(1, rows_drawn=20, total_cap=160) == 20
    assert optimizer._next_batch_size(2, rows_drawn=40, total_cap=160) == 40
    assert optimizer._next_batch_size(3, rows_drawn=80, total_cap=160) == 80
    # Spent: the loop reads this as "budget exhausted" and stops.
    assert optimizer._next_batch_size(4, rows_drawn=160, total_cap=160) == 0


def test_the_last_round_is_clipped_to_whatever_the_budget_still_allows():
    """A budget that is not a clean power of two still lands exactly on it."""
    optimizer = GradientDescentOptimizer(
        OptimizationConfig(adaptive_sampling=True, sample_size=100, first_round_rows=20)
    )
    assert optimizer._next_batch_size(3, rows_drawn=80, total_cap=100) == 20


def test_the_total_cap_is_just_the_sample_size():
    """An adaptive run spends the same row budget as the run it is compared against."""
    optimizer = GradientDescentOptimizer(
        OptimizationConfig(adaptive_sampling=True, sample_size=160)
    )
    assert optimizer._total_sample_cap() == 160
    # Uncapped when there are no rounds to spread the budget over.
    assert (
        GradientDescentOptimizer(OptimizationConfig())._total_sample_cap() is None
    )


# ── the derivation ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sample_size,first_round_rows,rounds,schedule",
    [
        (160, 20, 4, [20, 20, 40, 80]),
        (100, 20, 4, [20, 20, 40, 20]),
        (80, 20, 3, [20, 20, 40]),
        (40, 20, 2, [20, 20]),
        (20, 20, 1, [20]),
        (160, 40, 3, [40, 40, 80]),
        (160, 80, 2, [80, 80]),
    ],
)
def test_the_round_count_is_the_length_of_the_schedule(
    sample_size, first_round_rows, rounds, schedule
):
    """`max_sampling_rounds` is derived, so it must equal what the sampler does.

    A separate round count could only ever disagree with the budget.
    """
    config = OptimizationConfig(
        adaptive_sampling=True,
        sample_size=sample_size,
        first_round_rows=first_round_rows,
    )
    optimizer = GradientDescentOptimizer(config)
    drawn, actual = 0, []
    for r in range(config.max_sampling_rounds):
        batch = optimizer._next_batch_size(r, rows_drawn=drawn, total_cap=sample_size)
        if batch <= 0:
            break
        actual.append(batch)
        drawn += batch
    assert actual == schedule
    assert config.max_sampling_rounds == rounds
    assert sum(actual) == sample_size
    # And the loop has nothing left to draw once the budget is spent.
    assert optimizer._next_batch_size(rounds, drawn, sample_size) <= 0


def test_the_what_if_width_starts_at_one_slot_per_round():
    """The widest the axis ever needs to be: after round 0 there are `rounds - 1` still
    to come, plus "draw nothing more"."""
    for sample_size, expected in [(20, 1), (40, 2), (80, 3), (160, 4)]:
        config = OptimizationConfig(
            adaptive_sampling=True, sample_size=sample_size, first_round_rows=20
        )
        assert config.num_budgets_to_test == config.max_sampling_rounds
        assert config.num_budgets_to_test == expected


def test_the_width_narrows_as_the_rounds_run_down():
    """Surplus slots do not go unused -- they clamp onto the remaining budget and become
    duplicates, so a fixed width spends restarts re-deciding a priced hypothesis."""
    config = OptimizationConfig(adaptive_sampling=True)
    optimizer = GradientDescentOptimizer(config)
    cap, drawn, widths, jobs = optimizer._total_sample_cap(), 0, [], []
    diff = _differentiable_config(config)
    for r in range(config.max_sampling_rounds):
        drawn += optimizer._next_batch_size(r, drawn, cap)
        rounds_remaining = config.max_sampling_rounds - (r + 1)
        diff.resize_budgets(min(1 + rounds_remaining, MAX_BUDGET_SLOTS))
        widths.append(diff.num_budgets)
        jobs.append(diff.num_jobs)
        # Every slot prices something distinct and nothing beyond the budget.
        grid = GradientDescentOptimizer.get_what_if(
            batch_size=max(optimizer._next_batch_size(r + 1, drawn, cap), 0),
            num_jobs=diff.num_budgets,
            num_budgets_to_test=diff.num_budgets,
            max_remaining=10**9,
            max_extra=max(cap - drawn, 0),
            device=torch.device("cpu"),
        ).tolist()
        assert len(set(grid)) == diff.num_budgets, (grid, diff.num_budgets)
        assert max(grid) <= max(cap - drawn, 0)
    assert widths == [4, 3, 2, 1]
    # 2560 against the 4096 a fixed width would have solved: a 37.5% saving.
    assert sum(jobs) == 2560 and 4 * max(jobs) == 4096


def test_resizing_keeps_the_pruning_mask():
    """The whole reason the width is resized rather than the config rebuilt: rebuilding
    reruns `build_lookup_structures`, which clears the mask that says which operators have
    stopped being profiled."""
    config = OptimizationConfig(adaptive_sampling=True)
    diff = _differentiable_config(config)
    diff.pruning_mask[:] = True
    before = diff.pruning_mask.clone()
    assert diff.resize_budgets(2) is True
    assert diff.num_budgets == 2
    assert torch.equal(diff.pruning_mask, before)
    # Idempotent, and never narrower than one slot.
    assert diff.resize_budgets(2) is False
    diff.resize_budgets(0)
    assert diff.num_budgets == 1


def test_the_what_if_width_is_capped_so_a_big_budget_cannot_multiply_the_job_count():
    config = OptimizationConfig(
        adaptive_sampling=True, sample_size=10_000, first_round_rows=10
    )
    assert config.max_sampling_rounds == 11  # the loop still walks them all
    assert config.num_budgets_to_test == MAX_BUDGET_SLOTS  # but the grid does not widen


def test_a_first_round_larger_than_the_budget_is_rejected():
    with pytest.raises(AssertionError, match="first_round_rows"):
        OptimizationConfig(adaptive_sampling=True, sample_size=50, first_round_rows=100)
