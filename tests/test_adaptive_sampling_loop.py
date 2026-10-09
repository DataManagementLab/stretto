"""The iterative sampling loop in `GradientDescentOptimizer.tune_pipeline`.

Wiring, not numerics: the solve itself is stubbed out so these run on CPU in
milliseconds. What is pinned is the loop's contract with everything around it --
how many rounds it draws, what batch it asks the sampler for, when it stops, that it
forwards the pruning filter to the profiler, and that a non-adaptive run still makes
exactly one pass with no filter at all.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pandas as pd


from reasondb.optimizer.gd_optimizer import (
    GradientDescentOptimizer,
    OptimizationConfig,
)
from reasondb.optimizer.profiler import ProfilingOutput


class DummyLogger:
    def __truediv__(self, _other):
        return self

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _FakeSample:
    """A `ProfilingSampleSpecification` stand-in that accumulates like the real one.

    `prepend` really does concatenate, because the loop reads its total row count back
    off the accumulated frame to decide whether the profiling budget is spent -- a stub
    that dropped the accumulation would let the loop overrun its cap and the test would
    pass anyway.
    """

    def __init__(self, num_rows, first_index=0):
        self.index_column_values = pd.DataFrame(
            {"0": list(range(first_index, first_index + num_rows))}
        )
        self.sample_fraction = 0.01

    def prepend(self, other):
        if other is None:
            return
        self.index_column_values = pd.concat(
            [other.index_column_values, self.index_column_values], ignore_index=True
        )


def _run(optimizer, *, terminate_after_round=1, drawn_per_round=None):
    """Drive `tune_pipeline` with a stubbed sampler/profiler/solver.

    Returns `(sampler, profiler, solve_calls)`.
    """
    optimizer.set_database(MagicMock())

    drawn = drawn_per_round or {}
    sampler = MagicMock()
    sampler.batch_size = optimizer.optimizer_config.batch_size
    cursor = {"next": 0}

    def _do_sample(**kwargs):
        size = kwargs.get("sample_size")
        rows = drawn.get(size, size if size is not None else 160)
        spec = _FakeSample(rows, first_index=cursor["next"])
        cursor["next"] += rows
        return spec

    sampler.sample.side_effect = _do_sample
    optimizer.get_sampler = MagicMock(return_value=sampler)

    profiler = MagicMock()
    profiler.profile = AsyncMock(side_effect=lambda **_kw: ProfilingOutput())
    optimizer.get_profiler = MagicMock(return_value=profiler)

    solve_calls = []

    async def _solve(**kwargs):
        solve_calls.append(kwargs)
        met = len(solve_calls) >= terminate_after_round
        return (met, 0, 0, {}, MagicMock())

    optimizer.gd_optimize = AsyncMock(side_effect=_solve)
    optimizer.get_tuned_pipeline_from_config = AsyncMock(
        return_value=(MagicMock(), MagicMock(), {}, MagicMock())
    )
    optimizer.get_reorderer = MagicMock(
        return_value=MagicMock(reorder=MagicMock(return_value=MagicMock()))
    )

    pipeline = MagicMock()
    pipeline.get_virtual_input_columns.return_value = ["some_column"]
    intermediate_state = MagicMock()
    intermediate_state.materialization_points = []

    asyncio.run(
        optimizer.tune_pipeline(
            pipeline=pipeline,
            intermediate_state=intermediate_state,
            guarantees=[],
            logger=DummyLogger(),
        )
    )
    return sampler, profiler, solve_calls


# ── the default: unchanged ──────────────────────────────────────────────────────


def test_a_non_adaptive_run_draws_exactly_one_sample():
    optimizer = GradientDescentOptimizer(OptimizationConfig())
    sampler, profiler, _ = _run(optimizer)
    assert sampler.sample.call_count == 1
    assert profiler.profile.await_count == 1


def test_a_non_adaptive_run_lets_the_sampler_pick_its_own_size():
    optimizer = GradientDescentOptimizer(OptimizationConfig())
    sampler, _, _ = _run(optimizer)
    assert sampler.sample.call_args.kwargs["sample_size"] is None


def test_a_non_adaptive_run_profiles_every_candidate():
    """No filter at all -- `None` means "profile everything" in `profile_cascade`."""
    optimizer = GradientDescentOptimizer(OptimizationConfig())
    _, profiler, _ = _run(optimizer)
    assert profiler.profile.await_args.kwargs["operator_filter"] is None


def test_a_non_adaptive_run_reports_no_rounds_remaining():
    optimizer = GradientDescentOptimizer(OptimizationConfig())
    _, _, solves = _run(optimizer)
    assert solves[0]["rounds_remaining"] == 0


# ── the adaptive loop ───────────────────────────────────────────────────────────


def _adaptive(**overrides):
    """The adaptive config: two numbers, everything else derived."""
    kwargs = dict(adaptive_sampling=True, sample_size=160, first_round_rows=20)
    kwargs.update(overrides)
    return GradientDescentOptimizer(OptimizationConfig(**kwargs))


def test_it_stops_as_soon_as_the_solve_is_satisfied():
    optimizer = _adaptive()
    sampler, profiler, _ = _run(optimizer, terminate_after_round=1)
    assert sampler.sample.call_count == 1
    assert profiler.profile.await_count == 1


def test_it_keeps_sampling_while_the_solve_asks_for_more():
    optimizer = _adaptive()
    # Never satisfied: the loop runs until the row budget is spent.
    sampler, _, _ = _run(optimizer, terminate_after_round=99)
    sizes = [c.kwargs["sample_size"] for c in sampler.sample.call_args_list]
    assert sizes == [20, 20, 40, 80]


def test_the_schedule_it_walks_is_the_one_it_prices():
    """Each round draws the sample it already has, which is exactly what the what-if
    grid's `(2 ** k - 1) * current_sample` prices -- so the round the optimizer decides
    to buy is the round the sampler actually draws."""
    optimizer = _adaptive()
    sampler, _, _ = _run(optimizer, terminate_after_round=99)
    sizes = [c.kwargs["sample_size"] for c in sampler.sample.call_args_list]
    assert sum(sizes) == 160
    # The accumulated sample doubles every round: 20 / 40 / 80 / 160.
    totals, running = [], 0
    for n in sizes:
        running += n
        totals.append(running)
    assert totals == [20, 40, 80, 160]
    assert all(b == 2 * a for a, b in zip(totals, totals[1:]))


def test_the_row_budget_is_never_exceeded():
    optimizer = _adaptive(sample_size=50)
    sampler, _, _ = _run(optimizer, terminate_after_round=99)
    sizes = [c.kwargs["sample_size"] for c in sampler.sample.call_args_list]
    assert sum(sizes) <= 50
    assert sizes == [20, 20, 10]


def test_the_row_budget_is_what_bounds_the_loop():
    """There is no separate round count to disagree with the budget.

    40 rows from a 20-row first round is two rounds (20 + 20), and that is the only
    thing that stops the loop.
    """
    optimizer = _adaptive(sample_size=40)
    sampler, _, _ = _run(optimizer, terminate_after_round=99)
    assert sampler.sample.call_count == 2
    assert optimizer.optimizer_config.max_sampling_rounds == 2


def test_rounds_remaining_counts_down_to_zero():
    """The last round must know it is the last, or it defers to a sample it cannot draw."""
    optimizer = _adaptive(sample_size=80)  # 20 + 20 + 40 -> three rounds
    _, _, solves = _run(optimizer, terminate_after_round=99)
    remaining = [s["rounds_remaining"] for s in solves]
    assert remaining[0] == 2
    assert remaining[-1] == 0


def test_each_round_gets_its_own_seed():
    optimizer = _adaptive()
    sampler, _, _ = _run(optimizer, terminate_after_round=99)
    indices = [c.kwargs["round_index"] for c in sampler.sample.call_args_list]
    assert indices == list(range(len(indices)))


def test_each_round_accumulates_onto_the_previous_sample():
    optimizer = _adaptive()
    sampler, _, _ = _run(optimizer, terminate_after_round=99)
    previous = [c.kwargs["previous_sample"] for c in sampler.sample.call_args_list]
    assert previous[0] is None
    assert all(p is not None for p in previous[1:])


def test_an_empty_draw_ends_the_loop():
    optimizer = _adaptive()
    # Rounds 0 and 1 both draw 20; the table runs dry on the 40 that round 2 asks for.
    sampler, profiler, _ = _run(
        optimizer, terminate_after_round=99, drawn_per_round={40: 0}
    )
    assert sampler.sample.call_count == 3
    # The empty draw is not profiled, and the loop stops rather than trying round 3.
    assert profiler.profile.await_count == 2


def test_the_budget_axis_narrows_inside_the_real_loop():
    """`resize_budgets` is called from `tune_pipeline`, not merely callable.

    The width arithmetic is unit-tested against a standalone `DifferentiableConfig`;
    without the call in the loop every solve would keep paying for duplicate slots.
    This reads the width the solve actually saw.
    """
    optimizer = _adaptive()
    widths = []

    async def _solve(**kwargs):
        widths.append((kwargs["config"].num_budgets, kwargs["config"].num_jobs))
        return (False, 0, 0, {}, MagicMock())

    optimizer.set_database(MagicMock())
    sampler = MagicMock()
    sampler.batch_size = optimizer.optimizer_config.batch_size
    cursor = {"next": 0}

    def _do_sample(**kw):
        spec = _FakeSample(kw["sample_size"], first_index=cursor["next"])
        cursor["next"] += kw["sample_size"]
        return spec

    sampler.sample.side_effect = _do_sample
    optimizer.get_sampler = MagicMock(return_value=sampler)
    profiler = MagicMock()
    profiler.profile = AsyncMock(side_effect=lambda **_kw: ProfilingOutput())
    optimizer.get_profiler = MagicMock(return_value=profiler)
    optimizer.gd_optimize = AsyncMock(side_effect=_solve)
    optimizer.get_tuned_pipeline_from_config = AsyncMock(
        return_value=(MagicMock(), MagicMock(), {}, MagicMock())
    )
    optimizer.get_reorderer = MagicMock(
        return_value=MagicMock(reorder=MagicMock(return_value=MagicMock()))
    )
    pipeline = MagicMock()
    pipeline.get_virtual_input_columns.return_value = ["c"]
    state = MagicMock()
    state.materialization_points = []
    asyncio.run(
        optimizer.tune_pipeline(
            pipeline=pipeline,
            intermediate_state=state,
            guarantees=[],
            logger=DummyLogger(),
        )
    )

    # Three solve attempts per round, so take the first of each round.
    per_round = widths[::3]
    assert [w for w, _ in per_round] == [4, 3, 2, 1]
    # The job axis follows, which is where the saving actually lands.
    n_init = optimizer.optimizer_config.num_initializations
    assert [j for _, j in per_round] == [4 * n_init, 3 * n_init, 2 * n_init, n_init]


def test_an_empty_first_draw_falls_back_instead_of_crashing():
    """Nothing profiled at all -- the loop exits gracefully rather than crashing."""
    optimizer = _adaptive()
    sampler, profiler, solves = _run(
        optimizer, terminate_after_round=99, drawn_per_round={20: 0}
    )
    assert sampler.sample.call_count == 1
    assert profiler.profile.await_count == 0
    assert solves == [], "nothing was profiled, so nothing could be optimized"
    # The guarantee is reported as unreachable rather than silently met.
    assert optimizer.last_optimization_report is None


def test_the_config_learns_the_next_round_s_batch_and_remaining_budget():
    """Both feed the what-if grid, and both must be read *after* the round is drawn.

    Read before the draw they would each be one round stale, so the grid would price
    rows the budget forbids and the "keep sampling" decision would come out too eager.
    """
    seen = []
    optimizer = _adaptive()

    async def _capture(**kwargs):
        config = kwargs["config"]
        seen.append((config.batch_size, config.remaining_budget))
        return (False, 0, 0, {}, MagicMock())

    optimizer.set_database(MagicMock())
    sampler = MagicMock()
    sampler.batch_size = optimizer.optimizer_config.batch_size
    cursor = {"next": 0}

    def _do_sample(**kw):
        spec = _FakeSample(kw["sample_size"], first_index=cursor["next"])
        cursor["next"] += kw["sample_size"]
        return spec

    sampler.sample.side_effect = _do_sample
    optimizer.get_sampler = MagicMock(return_value=sampler)
    profiler = MagicMock()
    profiler.profile = AsyncMock(side_effect=lambda **_kw: ProfilingOutput())
    optimizer.get_profiler = MagicMock(return_value=profiler)
    optimizer.gd_optimize = AsyncMock(side_effect=_capture)
    optimizer.get_tuned_pipeline_from_config = AsyncMock(
        return_value=(MagicMock(), MagicMock(), {}, MagicMock())
    )
    optimizer.get_reorderer = MagicMock(
        return_value=MagicMock(reorder=MagicMock(return_value=MagicMock()))
    )
    pipeline = MagicMock()
    pipeline.get_virtual_input_columns.return_value = ["c"]
    state = MagicMock()
    state.materialization_points = []
    asyncio.run(
        optimizer.tune_pipeline(
            pipeline=pipeline,
            intermediate_state=state,
            guarantees=[],
            logger=DummyLogger(),
        )
    )
    # Three solve attempts per round, so take the first of each round.
    per_round = seen[::3]
    batches = [b for b, _ in per_round]
    remaining = [r for _, r in per_round]
    # Drawn 20 / 20 / 40 / 80. After each round the unit is the round still to come --
    # which is the sample accumulated so far -- and the budget left is 160 minus it.
    assert batches == [20, 40, 80, 0]
    assert remaining == [140, 120, 80, 0]
    # The property both of those exist to give: no hypothetical the grid can build ever
    # exceeds the rows still permitted. The top slot is
    # `batch * (2 ** (num_budgets - 1) - 1)`, clamped by `remaining_budget` -- so
    # checking the unit against the remainder is enough, and on the final round both
    # are zero: nothing more can be bought.
    assert all(b <= r for b, r in zip(batches, remaining))
    assert (batches[-1], remaining[-1]) == (0, 0)


# ── pruning ─────────────────────────────────────────────────────────────────────


def test_the_first_round_profiles_everything_even_with_pruning_on():
    """There is no evidence to prune on until a round has run."""
    optimizer = _adaptive(prune_unpicked_operators=True)
    _, profiler, _ = _run(optimizer, terminate_after_round=99)
    assert profiler.profile.await_args_list[0].kwargs["operator_filter"] is None


def test_pruning_stays_off_when_it_was_not_asked_for():
    optimizer = _adaptive(prune_unpicked_operators=False)
    optimizer._update_operator_filter = MagicMock()
    _run(optimizer, terminate_after_round=99)
    optimizer._update_operator_filter.assert_not_called()


def test_the_derived_filter_reaches_the_next_round_s_profiler():
    optimizer = _adaptive(prune_unpicked_operators=True)
    keep = {(0, 0): {1, 3}}
    optimizer._update_operator_filter = MagicMock(return_value=(keep, True))
    _, profiler, _ = _run(optimizer, terminate_after_round=99)
    filters = [c.kwargs["operator_filter"] for c in profiler.profile.await_args_list]
    assert filters[0] is None
    assert all(f == keep for f in filters[1:])


def test_the_valve_stops_further_pruning_for_the_rest_of_the_pipeline():
    """Prune on the first round, then report nothing feasible: the loop stops asking."""
    optimizer = _adaptive(prune_unpicked_operators=True)
    seen_filters = []

    def _update(**kwargs):
        previous = kwargs["previous_filter"]
        seen_filters.append(previous)
        if len(seen_filters) == 1:
            return {(0, 0): {3}}, True
        # What the real one does when nothing is feasible: hand the filter *back*
        # rather than dropping it. Pruning is irreversible -- `ProfilingOutput.prepend`
        # has already discarded the pruned operators' rows -- so widening again would
        # profile them from scratch, paying exactly what pruning saved.
        return previous, False

    optimizer._update_operator_filter = MagicMock(side_effect=_update)
    _, profiler, _ = _run(optimizer, terminate_after_round=99)

    # Round 1 prunes, round 2 reports nothing feasible, and round 3 never asks again.
    assert optimizer._update_operator_filter.call_count == 2
    assert seen_filters == [None, {(0, 0): {3}}]
    # The filter the second round profiled with is the one the first round derived, and
    # the rounds after the valve closes keep it rather than reverting to None.
    filters = [c.kwargs["operator_filter"] for c in profiler.profile.await_args_list]
    assert filters[0] is None
    assert all(f == {(0, 0): {3}} for f in filters[1:])


def test_pruning_is_not_attempted_on_the_last_round():
    """Nothing left to profile, so a filter derived there could only cost time."""
    optimizer = _adaptive(sample_size=40, prune_unpicked_operators=True)
    optimizer._update_operator_filter = MagicMock(return_value=({(0, 0): {3}}, True))
    _run(optimizer, terminate_after_round=99)
    # Two rounds, but only the first has a successor to narrow.
    assert optimizer._update_operator_filter.call_count == 1
