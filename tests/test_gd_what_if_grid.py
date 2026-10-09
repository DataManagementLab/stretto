"""The what-if budget grid: how many extra profiling rows each budget slot proposes.

Slot k means "sample for k more rounds". Each round draws the sample it already has, so
k more rounds multiply it by ``2 ** k`` and add ``(2 ** k - 1)`` times the current one --
which is what ``batch_size`` carries here. The slots are therefore the sample sizes the
loop can actually reach, not a ladder past them.

Slot 0 is the load-bearing one. It means "draw nothing more, ship what was measured",
and because it is zero the bound computed at that slot is the exact one -- which is what
the achieved guarantee is reported from. ``2 ** 0 - 1`` is zero, so the formula gives
that rather than a guard having to, independent of integer vs. float arithmetic.
"""

import torch

from reasondb.optimizer.gd_optimizer import GradientDescentOptimizer


CPU = torch.device("cpu")


def _grid(batch_size, num_budgets, max_remaining=10_000, max_extra=None, num_jobs=None):
    return GradientDescentOptimizer.get_what_if(
        batch_size=batch_size,
        num_jobs=num_jobs if num_jobs is not None else num_budgets,
        num_budgets_to_test=num_budgets,
        max_remaining=max_remaining,
        max_extra=max_extra,
        device=CPU,
    )


def test_slot_zero_proposes_nothing_whatever_the_parameters():
    for batch_size in (0, 1, 7, 10, 1000):
        for num_budgets in (1, 2, 5, 9):
            grid = _grid(batch_size, num_budgets)
            assert float(grid[0]) == 0.0, (batch_size, num_budgets)


def test_each_slot_is_k_more_rounds_of_doubling():
    # k more rounds take the sample from n to n * 2**k, i.e. add (2**k - 1) * n.
    assert _grid(10, 5).tolist() == [0, 10, 30, 70, 150]
    assert _grid(3, 4).tolist() == [0, 3, 9, 21]


def test_the_slots_are_the_reachable_sample_sizes():
    """With 20 rows drawn, the loop can reach 40, 80, 160 -- and the grid says so."""
    drawn = 20
    assert [drawn + x for x in _grid(drawn, 4).tolist()] == [20, 40, 80, 160]


def test_the_grid_is_non_decreasing():
    grid = _grid(10, 6)
    assert all(b >= a for a, b in zip(grid.tolist(), grid.tolist()[1:]))


def test_the_table_size_clamps_it():
    # Only 25 rows are left, so no slot may propose more than 25.
    assert _grid(10, 5, max_remaining=25).tolist() == [0, 10, 25, 25, 25]


def test_the_remaining_budget_clamps_it_too():
    # The table has plenty of rows but the sampler is only allowed 15 more. Proposing
    # +80 would make the optimizer defer to a sample it is not permitted to draw.
    assert _grid(10, 5, max_remaining=10_000, max_extra=15).tolist() == [
        0,
        10,
        15,
        15,
        15,
    ]


def test_the_tighter_of_the_two_clamps_wins():
    assert _grid(10, 5, max_remaining=12, max_extra=40).tolist() == [0, 10, 12, 12, 12]
    assert _grid(10, 5, max_remaining=40, max_extra=12).tolist() == [0, 10, 12, 12, 12]


def test_an_exhausted_table_leaves_every_slot_at_zero():
    # Nothing left to draw: every slot collapses onto "stop now", which is what makes
    # `post_optimization_check`'s argmin land on slot 0 and end the loop.
    assert _grid(10, 5, max_remaining=0).tolist() == [0, 0, 0, 0, 0]
    assert _grid(10, 5, max_remaining=-5).tolist() == [0, 0, 0, 0, 0]


def test_the_grid_is_budget_major_across_jobs():
    # One value per job, in blocks of `num_jobs // num_budgets` -- the layout
    # `loss.total.view(num_methods, num_budgets, num_initializations)` assumes.
    grid = _grid(10, 3, num_jobs=12)
    assert grid.tolist() == [0, 0, 0, 0, 10, 10, 10, 10, 30, 30, 30, 30]


def test_a_zero_batch_size_disables_the_whole_grid():
    # This is the default (non-adaptive) configuration: `config.batch_size` is None, so
    # `post_optimization_check` passes 0 and every slot proposes nothing.
    assert _grid(0, 5).tolist() == [0, 0, 0, 0, 0]
