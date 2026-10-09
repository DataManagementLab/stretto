"""Tests for the differentiable cost model's order-awareness.

Once sequenced, only tuples surviving the earlier operators of a materialization
stage reach a later one, so costs must not assume every cascade sees the full input.
CHOOSE_OPERATORS has no concrete, reordered plan yet to derive a real order
from (that needs operator choice to have settled first), so it costs the
pipeline against GradientDescentOptimizer.compute_initial_step_order's cheap
heuristic instead: every (cascade_id, tier) operator across the pipeline,
sorted ascending by its own per-tuple cost. Once CHOOSE_OPERATORS settles on
a choice, compute_reordered_cascade_order takes over: it materializes that
choice and runs it through the same reorderer tune_pipeline uses at the end
(DPReorderer, via get_reorderer) -- not Abacus's own reordering search, which
only ever picks a single physical operator per logical operator and so can't
stand in for the multi-operator escalation chains the gd/reorderer combo
allows. The result is a full (cascade_id, tier) sequence at *individual
operator* granularity, not just a per-cascade order: a cascade's escalation
tiers can end up interleaved with other cascades' operators rather than
running as one atomic block. apply_step_order_discount uses whichever order
is current (heuristic or reorderer-derived) to discount each operator's cost
by the survival probability of every *other* cascade's operator scheduled
before it (an operator's own cascade's earlier tiers are excluded -- that's
already priced into its cost by compute_cost).

compute_initial_step_order is exercised directly against a fake
profiling_output. compute_reordered_cascade_order is exercised by
monkeypatching its three collaborators (post_optimization_check,
get_tuned_pipeline_from_config, get_reorderer) with lightweight fakes, rather
than building a real Profiler/TuningPipeline/Database/IntermediateState --
this pins the order-extraction logic (mapping the reordered concrete steps
back to (cascade_id, tier) pairs, deduping repeats, padding tiers the
near-discrete snapshot dropped, falling back safely) independent of that
heavier machinery. apply_step_order_discount is exercised directly against
fake SimulatedPipelinePass-like objects exposing per_tier_cost/
per_tier_alive_after/set_cost.

Not covered: gd_optimize's own wiring (seeding step_order from
compute_initial_step_order, then replacing it once via
compute_reordered_cascade_order after CHOOSE_OPERATORS, threading it through
CHOOSE_PARAMETERS and the final post_optimization_check) and compute_cost's
own per-tier breakdown -- both need a full
Profiler/TuningPipeline/ProfilingOutput/Database to exercise end-to-end.
"""

import asyncio
from types import SimpleNamespace

import pytest

try:
    import torch

    from reasondb.optimizer.gd_optimizer import (
        GradientDescentOptimizer,
        OptimizationConfig,
        SimulatedPipelinePass,
    )
    from reasondb.query_plan.physical_operator import CostType, ProfilingCost
except ImportError:
    pytest.skip("optimizer deps not installed", allow_module_level=True)


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def __truediv__(self, other):
        return self


def _optimizer(**overrides) -> GradientDescentOptimizer:
    config_kwargs = {
        "cost_type": CostType.RUNTIME,
        "device": torch.device("cpu"),
        **overrides,
    }
    return GradientDescentOptimizer(OptimizationConfig(**config_kwargs))


class _Identifier:
    """Stand-in for a logical plan step's identifier: hashable, and equal
    only to itself, matching how real identifiers behave."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Identifier({self.name})"


def _unoptimized_step(identifier, num_tiers=1):
    return SimpleNamespace(
        logical_plan_step=SimpleNamespace(identifier=identifier),
        operators=list(range(num_tiers)),
    )


def _tuned_step(identifier, tier=0):
    return SimpleNamespace(
        logical_plan_step=SimpleNamespace(identifier=identifier), operator_index=tier
    )


def _pipeline(identifiers, tier_counts=None):
    """identifiers: list of _Identifier, index = cascade_id. tier_counts:
    parallel list of tier counts per cascade (default 1 each)."""
    if tier_counts is None:
        tier_counts = [1] * len(identifiers)
    return SimpleNamespace(
        steps_in_order_with_ids=[
            (cascade_id, 0, _unoptimized_step(identifier, num_tiers))
            for cascade_id, (identifier, num_tiers) in enumerate(
                zip(identifiers, tier_counts)
            )
        ],
        dependencies=[set() for _ in identifiers],
        steps_in_parallel=[None] * len(identifiers),
    )


def _stub_collaborators(monkeypatch, optimizer, reordered_steps, calls=None):
    """Wire up post_optimization_check / get_tuned_pipeline_from_config /
    get_reorderer so compute_reordered_cascade_order runs end-to-end without
    real Profiler/Database/IntermediateState machinery. `reordered_steps`:
    list of (identifier, tier) -- the reorderer fake returns plan_steps for
    these, in order, which is the only thing the order-extraction logic
    actually depends on."""
    if calls is None:
        calls = {}

    def _fake_post_optimization_check(**kwargs):
        calls["post_optimization_check"] = kwargs
        return (False, 0, 0, {}, SimpleNamespace())

    async def _fake_get_tuned_pipeline_from_config(**kwargs):
        calls["get_tuned_pipeline_from_config"] = kwargs
        return (SimpleNamespace(), [], [], SimpleNamespace())

    class _FakeReorderer:
        def reorder(self, **kwargs):
            calls["reorder"] = kwargs
            return SimpleNamespace(
                plan_steps=[
                    _tuned_step(identifier, tier) for identifier, tier in reordered_steps
                ]
            )

    monkeypatch.setattr(optimizer, "post_optimization_check", _fake_post_optimization_check)
    monkeypatch.setattr(
        optimizer, "get_tuned_pipeline_from_config", _fake_get_tuned_pipeline_from_config
    )
    monkeypatch.setattr(GradientDescentOptimizer, "get_reorderer", lambda self: _FakeReorderer())
    return calls


def _compute_order(optimizer, pipeline):
    return asyncio.run(
        optimizer.compute_reordered_cascade_order(
            profiler=object(),
            pipeline=pipeline,
            guarantees=[],
            config=object(),
            sample=SimpleNamespace(sample_fraction=1.0, index_column_values=[]),
            profiling_output=SimpleNamespace(total_cost=object()),
            intermediate_state=SimpleNamespace(database=object()),
            duplication_factors={},
            input_sizes={},
            logger=_NullLogger(),
        )
    )


# --- compute_initial_step_order ------------------------------------------


def _profiling_output_with_costs(costs):
    """costs: {(cascade_id, tier): cost}."""
    return SimpleNamespace(
        per_operator_and_sample_cost=lambda cascade_id, level, op_id: SimpleNamespace(
            get_cost=lambda cost_type: costs[(cascade_id, op_id)]
        )
    )


def test_compute_initial_step_order_sorts_by_cost_ascending():
    a, b = _Identifier("a"), _Identifier("b")
    optimizer = _optimizer()
    pipeline = _pipeline([a, b], tier_counts=[2, 1])
    profiling_output = _profiling_output_with_costs(
        {(0, 0): 30.0, (0, 1): 5.0, (1, 0): 10.0}
    )

    order = optimizer.compute_initial_step_order(pipeline, profiling_output)

    assert order == [(0, 1), (1, 0), (0, 0)]


def test_compute_initial_step_order_respects_cost_type():
    """The heuristic must sort by whichever cost the optimizer is actually
    configured to minimize, not always runtime -- cascade 0 is cheap on
    runtime but expensive on monetary cost, and vice versa for cascade 1."""
    a, b = _Identifier("a"), _Identifier("b")
    pipeline = _pipeline([a, b])
    profiling_output = SimpleNamespace(
        per_operator_and_sample_cost=lambda cascade_id, level, op_id: {
            0: ProfilingCost(runtime=1.0, monetary_cost=100.0),
            1: ProfilingCost(runtime=100.0, monetary_cost=1.0),
        }[cascade_id]
    )

    runtime_optimizer = _optimizer(cost_type=CostType.RUNTIME)
    monetary_optimizer = _optimizer(cost_type=CostType.MONETARY)

    assert runtime_optimizer.compute_initial_step_order(pipeline, profiling_output) == [
        (0, 0),
        (1, 0),
    ]
    assert monetary_optimizer.compute_initial_step_order(pipeline, profiling_output) == [
        (1, 0),
        (0, 0),
    ]


# --- compute_reordered_cascade_order ------------------------------------


def test_compute_reordered_cascade_order_matches_reorderer_output(monkeypatch):
    a, b, c = _Identifier("a"), _Identifier("b"), _Identifier("c")
    optimizer = _optimizer()
    _stub_collaborators(
        monkeypatch, optimizer, reordered_steps=[(c, 0), (a, 0), (b, 0)]
    )

    order = _compute_order(optimizer, _pipeline([a, b, c]))

    assert order == [(2, 0), (0, 0), (1, 0)]


def test_compute_reordered_cascade_order_interleaves_individual_operators(monkeypatch):
    """Unlike Abacus (which picks exactly one physical operator per logical
    operator), the gd/reorderer combo can chain a cascade's escalation tiers
    into multiple TunedPipelineSteps, and the reorderer can interleave other
    cascades' operators *between* them -- this is exactly the fidelity a
    per-cascade order would lose."""
    a, b = _Identifier("a"), _Identifier("b")
    optimizer = _optimizer()
    # cascade a (2 tiers) sandwiches cascade b (1 tier): a-tier0, b-tier0, a-tier1.
    _stub_collaborators(
        monkeypatch, optimizer, reordered_steps=[(a, 0), (b, 0), (a, 1)]
    )

    order = _compute_order(optimizer, _pipeline([a, b], tier_counts=[2, 1]))

    assert order == [(0, 0), (1, 0), (0, 1)]


def test_compute_reordered_cascade_order_pads_tiers_the_snapshot_dropped(monkeypatch):
    """A tier GD has all but ruled out (near-zero pick score) at the
    near-discrete snapshot used for materialization may not survive into
    tuned_pipeline.plan_steps at all, so the reorderer never sees or places
    it. It must still show up in the result (appended, since its near-zero
    cost makes its exact position immaterial) -- dropping it entirely would
    break apply_step_order_discount's completeness check."""
    a, b = _Identifier("a"), _Identifier("b")
    optimizer = _optimizer()
    # cascade a has 2 tiers but only tier 1 (gold) survived materialization.
    _stub_collaborators(monkeypatch, optimizer, reordered_steps=[(b, 0), (a, 1)])

    order = _compute_order(optimizer, _pipeline([a, b], tier_counts=[2, 1]))

    assert order == [(1, 0), (0, 1), (0, 0)]


def test_compute_reordered_cascade_order_forwards_snapshot_into_materialization(monkeypatch):
    """The used_method/best_index/operator_received_data/selectivities
    post_optimization_check returns for the current (post-CHOOSE_OPERATORS)
    snapshot must be exactly what gets materialized -- not some other/stale
    state."""
    a = _Identifier("a")
    optimizer = _optimizer()
    calls = {}

    def _fake_post_optimization_check(**kwargs):
        return (False, 7, 3, {"sentinel": True}, "the-selectivities")

    async def _fake_get_tuned_pipeline_from_config(**kwargs):
        calls["get_tuned_pipeline_from_config"] = kwargs
        return (SimpleNamespace(), [], [], SimpleNamespace())

    class _FakeReorderer:
        def reorder(self, **kwargs):
            return SimpleNamespace(plan_steps=[_tuned_step(a, 0)])

    monkeypatch.setattr(optimizer, "post_optimization_check", _fake_post_optimization_check)
    monkeypatch.setattr(
        optimizer, "get_tuned_pipeline_from_config", _fake_get_tuned_pipeline_from_config
    )
    monkeypatch.setattr(GradientDescentOptimizer, "get_reorderer", lambda self: _FakeReorderer())

    _compute_order(optimizer, _pipeline([a]))

    forwarded = calls["get_tuned_pipeline_from_config"]
    assert forwarded["used_method"] == 7
    assert forwarded["best_index"] == 3
    assert forwarded["operator_received_data"] == {"sentinel": True}
    assert forwarded["selectivities"] == "the-selectivities"
    assert forwarded["termination_criterion_met"] is True


def test_compute_reordered_cascade_order_disabled_returns_none(monkeypatch):
    a = _Identifier("a")
    optimizer = _optimizer(order_aware_cost=False)
    calls = _stub_collaborators(monkeypatch, optimizer, reordered_steps=[(a, 0)])

    order = _compute_order(optimizer, _pipeline([a]))

    assert order is None
    assert calls == {}  # short-circuits before touching any collaborator


def test_compute_reordered_cascade_order_survives_exceptions(monkeypatch):
    async def _boom(**kwargs):
        raise RuntimeError("boom")

    a = _Identifier("a")
    optimizer = _optimizer()
    monkeypatch.setattr(
        optimizer, "post_optimization_check", lambda **kwargs: (False, 0, 0, {}, None)
    )
    monkeypatch.setattr(optimizer, "get_tuned_pipeline_from_config", _boom)

    order = _compute_order(optimizer, _pipeline([a]))

    assert order is None


# --- apply_step_order_discount -------------------------------------------


class _FakeSimulatedPass:
    def __init__(self, per_tier_cost, per_tier_alive_after):
        self.per_tier_cost = torch.tensor(per_tier_cost)
        self.per_tier_alive_after = torch.tensor(per_tier_alive_after)
        self.cost = None

    def set_cost(self, cost):
        self.cost = cost


def test_apply_step_order_discount_discounts_a_later_cascade():
    optimizer = _optimizer()
    # cascade 0: one tier, selective (alive_after 0.5). cascade 1: one tier.
    passes = [
        _FakeSimulatedPass([[10.0]], [[0.5]]),
        _FakeSimulatedPass([[10.0]], [[1.0]]),
    ]

    optimizer.apply_step_order_discount(passes, [(0, 0), (1, 0)])

    assert passes[0].cost.tolist() == pytest.approx([10.0])  # nothing runs before it
    assert passes[1].cost.tolist() == pytest.approx([5.0])  # discounted by cascade 0


def test_apply_step_order_discount_only_discounts_what_comes_after():
    """A selective cascade scheduled *last* can't discount anything -- it's
    whatever runs before a step that determines what it sees, not the other
    way around."""
    optimizer = _optimizer()
    passes = [
        _FakeSimulatedPass([[10.0]], [[0.5]]),
        _FakeSimulatedPass([[10.0]], [[1.0]]),
    ]

    optimizer.apply_step_order_discount(passes, [(1, 0), (0, 0)])

    assert passes[0].cost.tolist() == pytest.approx([10.0])
    assert passes[1].cost.tolist() == pytest.approx([10.0])


def test_apply_step_order_discount_excludes_same_cascade_precedence():
    """The whole point of going to operator granularity: cascade 0's own
    tier 0 must NOT discount cascade 0's tier 1 (that precedence is already
    priced into tier 1's cost by compute_cost's own input_probas), but it
    DOES discount an interleaved cascade 1 operator, which in turn discounts
    cascade 0's tier 1. Order: (0,0) cost=100 alive=0.4, (1,0) cost=10
    alive=0.9, (0,1) cost=20 alive=1.0."""
    optimizer = _optimizer()
    passes = [
        _FakeSimulatedPass([[100.0, 20.0]], [[0.4, 1.0]]),  # cascade 0, 2 tiers
        _FakeSimulatedPass([[10.0]], [[0.9]]),  # cascade 1, 1 tier
    ]

    optimizer.apply_step_order_discount(passes, [(0, 0), (1, 0), (0, 1)])

    # cascade 1's tier is discounted by cascade 0's tier 0 (0.4): 10*0.4=4.
    # cascade 0's tier 1 is discounted by cascade 1's tier (0.9), NOT by its
    # own tier 0: 20*0.9=18. cascade 0 total = 100 (undiscounted tier 0) + 18.
    assert passes[0].cost.tolist() == pytest.approx([118.0])
    assert passes[1].cost.tolist() == pytest.approx([4.0])


def test_apply_step_order_discount_is_job_dependent():
    """Different jobs (parallel restarts) can pick different operators/
    thresholds and thus have different selectivity -- the discount must
    track each job's own survival probabilities."""
    optimizer = _optimizer()
    passes = [
        _FakeSimulatedPass([[10.0], [10.0]], [[0.1], [0.9]]),  # 2 jobs
        _FakeSimulatedPass([[10.0], [10.0]], [[1.0], [1.0]]),
    ]

    optimizer.apply_step_order_discount(passes, [(0, 0), (1, 0)])

    assert passes[1].cost.tolist() == pytest.approx([1.0, 9.0])


def test_apply_step_order_discount_single_cascade_is_unaffected():
    """With only one cascade in the pipeline, every operator ordered before
    another belongs to that same cascade, so the cross-cascade mask is
    always empty -- costs must come out exactly as summed, regardless of
    how the (single cascade's own) tiers are ordered."""
    optimizer = _optimizer()
    passes = [_FakeSimulatedPass([[10.0, 20.0]], [[0.1, 1.0]])]  # 1 cascade, 2 tiers

    optimizer.apply_step_order_discount(passes, [(0, 0), (0, 1)])

    assert passes[0].cost.tolist() == pytest.approx([30.0])


def test_apply_step_order_discount_none_order_is_a_no_op():
    optimizer = _optimizer()
    passes = [_FakeSimulatedPass([[10.0]], [[0.5]])]

    optimizer.apply_step_order_discount(passes, None)

    assert passes[0].cost is None  # set_cost was never called


def test_apply_step_order_discount_incomplete_order_is_a_no_op():
    """An order missing a tier (e.g. stale, computed against a
    differently-shaped pipeline) must be ignored rather than misapplied."""
    optimizer = _optimizer()
    passes = [
        _FakeSimulatedPass([[10.0]], [[0.5]]),
        _FakeSimulatedPass([[10.0]], [[1.0]]),
    ]

    optimizer.apply_step_order_discount(passes, [(0, 0)])  # missing (1, 0)

    assert passes[0].cost is None
    assert passes[1].cost is None


class _RecordingLogger(_NullLogger):
    def __init__(self):
        self.warnings = []

    def warning(self, *args, **kwargs):
        self.warnings.append((args, kwargs))


def test_apply_step_order_discount_incomplete_order_warns_once_per_call():
    """apply_step_order_discount runs on every GD step -- up to thousands of
    times per gd_optimize call -- so a persistent mismatch must not spam the
    log. Only the first occurrence warns; self._step_order_mismatch_warned
    (reset at the top of gd_optimize, see gd_optimizer.py) tracks that across
    calls sharing one optimizer instance."""
    optimizer = _optimizer()
    logger = _RecordingLogger()
    passes = [
        _FakeSimulatedPass([[10.0]], [[0.5]]),
        _FakeSimulatedPass([[10.0]], [[1.0]]),
    ]

    optimizer.apply_step_order_discount(passes, [(0, 0)], logger=logger)
    optimizer.apply_step_order_discount(passes, [(0, 0)], logger=logger)

    assert len(logger.warnings) == 1

    # A fresh gd_optimize call resets the flag, so the next mismatch warns again.
    optimizer._step_order_mismatch_warned = False
    optimizer.apply_step_order_discount(passes, [(0, 0)], logger=logger)

    assert len(logger.warnings) == 2


def test_apply_step_order_discount_incomplete_order_without_logger_does_not_warn():
    optimizer = _optimizer()
    passes = [
        _FakeSimulatedPass([[10.0]], [[0.5]]),
        _FakeSimulatedPass([[10.0]], [[1.0]]),
    ]

    optimizer.apply_step_order_discount(passes, [(0, 0)])  # no logger -> no crash

    assert passes[0].cost is None


def test_apply_step_order_discount_is_differentiable():
    """The discount must stay inside the autograd graph GD backprops
    through -- per_tier_cost/per_tier_alive_after are themselves
    differentiable functions of the pick scores/thresholds being
    optimized."""
    optimizer = _optimizer()
    cost0 = torch.tensor([[10.0]], requires_grad=True)
    alive0 = torch.tensor([[0.5]], requires_grad=True)
    cost1 = torch.tensor([[10.0]], requires_grad=True)
    alive1 = torch.tensor([[1.0]], requires_grad=True)
    passes = [
        _FakeSimulatedPass.__new__(_FakeSimulatedPass),
        _FakeSimulatedPass.__new__(_FakeSimulatedPass),
    ]
    passes[0].per_tier_cost, passes[0].per_tier_alive_after = cost0, alive0
    passes[1].per_tier_cost, passes[1].per_tier_alive_after = cost1, alive1
    passes[0].set_cost = lambda cost: setattr(passes[0], "cost", cost)
    passes[1].set_cost = lambda cost: setattr(passes[1], "cost", cost)

    optimizer.apply_step_order_discount(passes, [(0, 0), (1, 0)])
    (passes[0].cost.sum() + passes[1].cost.sum()).backward()

    assert cost0.grad is not None
    assert alive0.grad is not None
    assert cost1.grad is not None


def test_apply_step_order_discount_clamp_kills_gradient_when_saturated():
    """alive_after gets clamp(min=1e-6, max=1.0) before the log/exp trick
    (see apply_step_order_discount in gd_optimizer.py) -- and CHOOSE_PARAMETERS deliberately anneals
    pick_temperature down to 1e-6 (see get_temperature), driving near-certain
    tiers' alive_after below that floor by design, not as a rare edge case.
    d(clamp(x))/dx is exactly 0 outside the clamped range, so once a tier's
    alive_after saturates, it stops contributing gradient to the discount it
    imposes on later, other-cascade tiers -- even though it's the value
    "responsible" for that discount. cost1's own gradient is unaffected (just
    scaled down by the tiny clamped discount factor), only alive0's is dead."""
    optimizer = _optimizer()
    # cascade 0's alive_after is already below the clamp floor (1e-6).
    cost0 = torch.tensor([[10.0]], requires_grad=True)
    alive0 = torch.tensor([[1e-9]], requires_grad=True)
    # cascade 1 runs after cascade 0, so its cost gets discounted by alive0.
    cost1 = torch.tensor([[10.0]], requires_grad=True)
    alive1 = torch.tensor([[1.0]], requires_grad=True)
    passes = [
        _FakeSimulatedPass.__new__(_FakeSimulatedPass),
        _FakeSimulatedPass.__new__(_FakeSimulatedPass),
    ]
    passes[0].per_tier_cost, passes[0].per_tier_alive_after = cost0, alive0
    passes[1].per_tier_cost, passes[1].per_tier_alive_after = cost1, alive1
    passes[0].set_cost = lambda cost: setattr(passes[0], "cost", cost)
    passes[1].set_cost = lambda cost: setattr(passes[1], "cost", cost)

    optimizer.apply_step_order_discount(passes, [(0, 0), (1, 0)])

    # cascade 1's cost is discounted by the clamped floor (1e-6), not by the
    # true (far smaller) alive0 -- clamping changes the *value*, not just the
    # gradient.
    assert passes[1].cost.item() == pytest.approx(10.0 * 1e-6, abs=1e-9)

    (passes[0].cost.sum() + passes[1].cost.sum()).backward()

    assert alive0.grad.item() == 0.0  # dead: clamp saturated at the floor
    assert cost0.grad.item() == pytest.approx(1.0)  # nothing discounts cascade 0
    assert cost1.grad.item() == pytest.approx(1e-6)  # scaled down, not dead


# --- SimulatedPipelinePass.per_tier_cost / per_tier_alive_after ---------


def test_simulated_pipeline_pass_per_tier_cost_unset_raises():
    simulated_pass = SimulatedPipelinePass(transition_probabilities={})

    with pytest.raises(AssertionError):
        _ = simulated_pass.per_tier_cost
    with pytest.raises(AssertionError):
        _ = simulated_pass.per_tier_alive_after


def test_simulated_pipeline_pass_per_tier_cost_roundtrip():
    simulated_pass = SimulatedPipelinePass(transition_probabilities={})
    cost = torch.tensor([[1.0, 2.0]])
    alive_after = torch.tensor([[0.5, 0.9]])

    simulated_pass.set_per_tier_cost(cost, alive_after)

    assert torch.equal(simulated_pass.per_tier_cost, cost)
    assert torch.equal(simulated_pass.per_tier_alive_after, alive_after)


# --- compute_cost: per-tier breakdown -------------------------------------


class _FakeCascadeConfig:
    """Minimal stand-in for DifferentiableConfig's slice of the interface
    compute_cost needs: per-operator pick scores and the not_allow_accept/
    discard gold-mixing penalty factors (both zeroed out here so the
    penalized states compute_cost works on are exactly the raw ones)."""

    def __init__(self, pick_scores):
        self._pick_scores = pick_scores  # {operator_id: Tensor([num_jobs])}

    def get_operator_pick_score(self, cascade_id, level, physical_operator_id):
        return self._pick_scores[physical_operator_id]

    def not_allow_accept_scores(self, cascade_id, temperature):
        return torch.zeros(1)

    def not_allow_discard_scores(self, cascade_id, temperature):
        return torch.zeros(1)


def test_compute_cost_per_tier_breakdown_matches_hand_computed_values():
    """compute_cost exposes, besides the cascade's total cost, a per-tier
    breakdown (per_tier_cost) and each tier's post-resolution survival
    probability (per_tier_alive_after), which apply_step_order_discount needs. Pins the exact values against a
    hand-computed two-tier escalation: operator 0 discards 30% of tuples and
    leaves the rest ("unsure"); operator 1 (gold) then keeps everything
    still unsure.
    """
    optimizer = _optimizer()
    simulated_pass = SimulatedPipelinePass(transition_probabilities={})
    # One tuple, one job. Decision order is (KEEP, DISCARD, UNSURE).
    initial_state = torch.tensor([[[0.0, 0.0, 1.0]]])
    after_op0 = torch.tensor([[[0.0, 0.3, 0.7]]])  # op0 discards 30%, rest unsure
    after_op1 = torch.tensor([[[0.7, 0.3, 0.0]]])  # op1 (gold) keeps the remaining 70%
    simulated_pass.add([initial_state, after_op0, after_op1])

    # Saturate the sigmoid so both operators are fully "picked".
    config = _FakeCascadeConfig(
        pick_scores={0: torch.tensor([1.0]), 1: torch.tensor([1.0])}
    )
    profiling_output = SimpleNamespace(
        per_operator_and_sample_cost=lambda cascade_id, level, op_id: SimpleNamespace(
            get_cost=lambda cost_type: {0: 10.0, 1: 100.0}[op_id]
        )
    )

    cost_per_value, max_cost, per_tier_cost, per_tier_alive_after = (
        optimizer.compute_cost(
            profiling_output=profiling_output,
            cascade_id=0,
            simulated_pass=simulated_pass,
            config=config,
            operators=[None, None],
            pick_temperature=0.000001,
            temperature_gold_mixing=0.000001,
            level=0,
            logger=_NullLogger(),
        )
    )

    # op0 always runs (P(unsure entering it) = 1.0): cost = 1.0 * 10 = 10.
    # op1 only runs on the 70% still unsure after op0: cost = 0.7 * 100 = 70.
    assert per_tier_cost[0].tolist() == pytest.approx([10.0, 70.0])
    # After op0, 70% haven't been discarded; after op1, still 70% (op1
    # resolved all remaining-unsure tuples to KEEP, not DISCARD).
    assert per_tier_alive_after[0].tolist() == pytest.approx([0.7, 0.7])
    # The aggregate values must still match what per-tier sums/inputs give.
    assert cost_per_value.tolist() == pytest.approx([80.0])
    assert per_tier_cost.sum(dim=1).tolist() == pytest.approx(cost_per_value.tolist())
    assert max_cost.item() == pytest.approx(110.0)
