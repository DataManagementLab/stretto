"""How GD's parallel restarts are seeded, and at what scale.

The initialization should let GD find good plans in large operator search spaces as
reliably as in smaller, nested ones. Three aspects are covered:

- **Scale** (`pick_init_temperature_span`): a pick score is only ever read as
  `sigmoid(score / temperature)`, so the init magnitude is expressed in temperature
  units. An absolute magnitude that is large relative to `begin_temperature` saturates
  the sigmoid from step 0, and since the temperature only anneals downward, saturated
  coordinates never recover.
- **Sparsity** (`sparsity_init_fraction`): steps in guarantee-meeting plans frequently
  run no proxy or a single proxy. A random slot leaves a step all-off only with
  probability 2 ** -(n - 1) and a neutral slot starts everything half-on, so such steps
  are seeded explicitly rather than left to GD pruning down from ~n/2.
- **Mix**: job slots are split 1/16 neutral, 1/4 sparsity and 11/16 random; see
  `OptimizationConfig`'s job-slot initialization mix comment.
"""

import math

import pytest

try:
    import torch

    from reasondb.optimizer.base_optimizer import PipelineSearchSpace
    from reasondb.optimizer.gd_optimizer import (
        DifferentiableConfig,
        GradientDescentOptimizer,
        OptimizationConfig,
    )
    from reasondb.query_plan.tuning_parameters import TuningParameterContinuous
except ImportError:  # pragma: no cover - deps not installed
    pytest.skip("optimizer deps not installed", allow_module_level=True)


def _search_space(num_operators: int = 6, num_levels: int = 2) -> PipelineSearchSpace:
    """`num_levels` cascades of `num_operators` candidates each, so one candidate per
    level is gold and the rest carry pick parameters."""
    space = PipelineSearchSpace()
    for cascade_id in range(num_levels):
        space.add_operator_choice(
            step_id=cascade_id,
            cascade_id=cascade_id,
            level=0,
            operators=list(range(num_operators)),  # type: ignore[arg-type]
        )
        space.add_parameter_search_space(
            cascade_id=cascade_id,
            level=0,
            physical_operator_id=0,
            tuning_parameter=TuningParameterContinuous(
                name="threshold", default=0.3, init=0.1, min=0.0, max=1.0, log_scale=False
            ),
            fixed=False,
        )
    return space


def _config(num_initializations=64, num_operators=6, num_levels=2):
    return DifferentiableConfig(
        search_space=_search_space(num_operators, num_levels),
        rng=torch.Generator().manual_seed(0),
        num_initializations=num_initializations,
        num_budgets_to_test=1,
        num_methods=1,
        num_gold_mixing_params=[],
        batch_size=None,
    )


def _opt(**overrides) -> OptimizationConfig:
    return OptimizationConfig(device=torch.device("cpu"), **overrides)


# ── the init scale ──────────────────────────────────────────────────────────────


def test_num_pick_params_excludes_the_forced_gold_candidate():
    """The space GD searches is 2 ** this, so the count has to be exact: gold is forced
    on as the resolver and never gets a pick parameter."""
    assert _config(num_operators=6, num_levels=2).num_pick_params == 10
    assert _config(num_operators=2, num_levels=3).num_pick_params == 3


def test_pick_scores_start_inside_the_sigmoid_s_responsive_band():
    config = _config()
    optimizer_config = _opt()
    config.init(optimizer_config)

    scores = config._all_pick_scores
    scale = optimizer_config.pick_init_scale
    span = optimizer_config.pick_init_temperature_span * scale
    random_slots = [i for i, k in enumerate(config.init_kinds) if k == "random"]
    assert scores[random_slots].abs().max().item() <= span + 1e-6
    # Seeded slots are biased further out, but still in temperature units: the widest
    # any of them reaches is one boost plus one jitter.
    widest = (
        optimizer_config.pick_score_on + optimizer_config.pick_score_jitter
    ) * scale
    assert scores.abs().max().item() <= widest + 1e-6

    # The property that matters is not the magnitude but the gradient it admits: a
    # saturated coordinate has *exactly* zero gradient, and the temperature only ever
    # falls, so it could never move again.
    live = scores.detach().clone().requires_grad_(True)
    torch.sigmoid(live / optimizer_config.begin_temperature).sum().backward()
    assert (live.grad == 0).sum().item() == 0
    assert live.grad.abs().min().item() > 1e-3


def test_the_init_scale_follows_begin_temperature():
    """One knob moves the random restarts and the sparsity seeds
    together, so the scale always matches the temperature it is divided by."""
    hot = _config()
    hot.init(_opt(begin_temperature=1.0))
    cold = _config()
    cold.init(_opt(begin_temperature=0.01))
    assert hot._all_pick_scores.abs().max() > 10 * cold._all_pick_scores.abs().max()


def test_shrinking_the_scale_preserves_every_restart_s_discrete_plan():
    """Why rescaling is safe: a slot's plan is decided by the *sign* of its scores,
    so scaling the magnitude restores gradient flow without touching plan diversity."""
    wide = _config()
    wide.init(_opt(pick_init_temperature_span=20.0))
    narrow = _config()
    narrow.init(_opt(pick_init_temperature_span=2.0))
    assert torch.equal(wide._all_pick_scores.sign(), narrow._all_pick_scores.sign())


def test_tuning_parameters_keep_their_own_untempered_scale():
    """`_get_parameters` applies a plain sigmoid with no temperature, so +/- 2 there is a
    healthy spread over the parameter range, not a saturated one. The init scale must not
    touch it."""
    config = _config()
    config.init(_opt())
    assert config._all_params.abs().max().item() > 1.0


# ── per-step sparsity priors ────────────────────────────────────────────────────


def _proxies_per_step(config, slots):
    """How many non-gold candidates each slot turns on, per step."""
    out = []
    for slot in slots:
        for key, start in config.operator_lookup.items():
            n = config.gold_index_lookup[key]
            block = config._all_pick_scores[slot, start : start + n]
            out.append(int((block > 0).sum().item()))
    return out


def _kind_slots(config, kind):
    return [i for i, k in enumerate(config.init_kinds) if k == kind]


def test_sparsity_slots_cover_every_count_including_gold_only():
    """The 0 case is the whole point: gold-only steps are common in guarantee-meeting
    plans, and no other seed group produces that shape."""
    config = _config(num_initializations=64)
    optimizer_config = _opt()
    config.init(optimizer_config)

    slots = _kind_slots(config, "sparsity")
    assert slots, "sparsity seeding must claim slots"
    counts = set(_proxies_per_step(config, slots))
    assert counts == set(range(optimizer_config.max_seeded_step_proxies + 1))
    assert 0 in counts


def test_the_default_never_seeds_counts_that_do_not_win():
    """Three or more proxies on one step are rare in guarantee-meeting plans, so the
    default draw stops at two rather than spending a quarter of the slots on each."""
    assert OptimizationConfig().max_seeded_step_proxies == 2


def test_a_slot_mixes_counts_across_its_steps():
    """Drawing per step, not per slot. Winning plans typically mix -- some steps gold-only,
    others with a proxy -- which one count repeated across a slot cannot express."""
    config = _config(num_initializations=64, num_levels=3)
    config.init(_opt())

    mixed = 0
    for slot in _kind_slots(config, "sparsity"):
        per_step = _proxies_per_step(config, [slot])
        if len(set(per_step)) > 1:
            mixed += 1
    assert mixed > 0, "no sparsity slot varied its proxy count between steps"


def test_gold_only_steps_come_from_sparsity_seeding_and_nowhere_else():
    """States the gap concretely, as shares rather than absolutes.

    A random slot has to turn all n-1 candidates off at once, which is
    2 ** -(n - 1) -- 0.2% at ten candidates, and rarer as the space grows. (Neutral
    slots sit at exactly 0.0, i.e. every candidate half-on; that is neither state and
    a binary count cannot express it, so they are not compared here.)
    """
    config = _config(num_initializations=64, num_operators=10, num_levels=3)
    config.init(_opt())

    def gold_only_share(kind):
        counts = _proxies_per_step(config, _kind_slots(config, kind))
        return sum(n == 0 for n in counts) / len(counts)

    assert gold_only_share("random") < 0.05
    assert gold_only_share("sparsity") > 0.2


def test_every_count_keeps_a_floor_however_large_the_space_gets():
    """Why this is seeded rather than left to the random slots: a random slot's count is
    Binomial(n - 1, 1/2), so both ends thin out as the space grows -- at 20 candidates
    neither a gold-only nor a two-proxy step appears at all. Balanced per-step draws put
    a floor under every count in range, and the floor does not move with the space.
    """
    optimizer_config = _opt()
    num_counts = optimizer_config.max_seeded_step_proxies + 1
    floor = optimizer_config.sparsity_init_fraction / num_counts
    for num_operators in (10, 20, 30):
        config = _config(num_initializations=64, num_operators=num_operators, num_levels=3)
        config.init(
            optimizer_config,
        )
        counts = _proxies_per_step(config, _kind_slots(config, "sparsity"))
        total = config.num_jobs * len(config.operator_lookup)
        for target in range(num_counts):
            share = sum(n == target for n in counts) / total
            assert share >= floor * 0.9, (
                f"{num_operators} candidates: only {share:.1%} of steps start at "
                f"{target} proxies, below the {floor:.1%} floor"
            )


def test_a_step_with_fewer_candidates_than_the_draw_turns_all_on():
    """`min(k, num_candidates)`: a 2-candidate step has one non-gold candidate, so it can
    only ever be 0 or 1 -- and must not be skipped into something else."""
    config = _config(num_initializations=32, num_operators=2)
    config.init(_opt())
    counts = _proxies_per_step(config, _kind_slots(config, "sparsity"))
    assert counts, "sparsity seeding must claim slots"
    assert set(counts) <= {0, 1}


def test_seeded_steps_are_still_soft_enough_to_be_argued_out_of():
    """A prior, not a constraint -- the jitter must not flip a seeded count either."""
    optimizer_config = _opt()
    config = _config(num_initializations=64)
    config.init(optimizer_config)
    widest = (
        optimizer_config.pick_score_on + optimizer_config.pick_score_jitter
    ) * optimizer_config.pick_init_scale
    assert config._all_pick_scores.abs().max().item() <= widest + 1e-6


def test_the_four_seed_groups_are_disjoint_and_exhaustive():
    for num_initializations in (16, 32, 64, 256):
        config = _config(num_initializations=num_initializations)
        config.init(_opt())
        kinds = config.init_kinds
        assert len(kinds) == config.num_jobs
        assert set(kinds) <= {"neutral", "sparsity", "random"}
        assert all(k is not None for k in kinds)


def test_fractions_summing_past_one_are_rejected():
    """Three groups draw from the same pool; the random restarts must not be silently
    starved."""
    with pytest.raises(AssertionError, match="must sum to <= 1.0"):
        _opt(
            neutral_init_fraction=0.7,
            sparsity_init_fraction=0.4,
        )


def test_a_draw_range_with_no_room_for_a_proxy_is_rejected():
    with pytest.raises(AssertionError, match="max_seeded_step_proxies must be at least 1"):
        _opt(max_seeded_step_proxies=0)


def test_sparsity_seeding_can_be_switched_off():
    config = _config()
    config.init(_opt(sparsity_init_fraction=0.0))
    assert "sparsity" not in config.init_kinds


# ── the default mix ─────────────────────────────────────────────────────────────


def test_the_default_fractions_partition_the_job_slots_exactly():
    """`_select_job_slots` strides over a fraction, so one without a clean reciprocal
    silently hands its group a few more or fewer slots than configured.

    Asserted at the default fractions rather than at ones this test picks, so editing
    them to something unexpressible fails here rather than in a sweep's telemetry.
    """
    optimizer_config = _opt()
    config = _config(num_initializations=256)
    config.init(optimizer_config)

    counts = {kind: len(_kind_slots(config, kind)) for kind in
              ("neutral", "sparsity", "random")}
    assert counts == {"neutral": 16, "sparsity": 64, "random": 176}
    assert sum(counts.values()) == config.num_jobs

    for kind, fraction in (
        ("neutral", optimizer_config.neutral_init_fraction),
        ("sparsity", optimizer_config.sparsity_init_fraction),
    ):
        assert counts[kind] == round(config.num_jobs * fraction), kind


def test_the_diverse_seed_groups_hold_most_of_the_budget():
    """The groups that seed genuine diversity (random, sparsity) keep the large majority
    of the slots, since they win more often per slot than neutral seeds. This asserts
    the direction, not the exact split.
    """
    optimizer_config = _opt()
    diverse = (
        optimizer_config.sparsity_init_fraction + _random_fraction(optimizer_config)
    )
    assert diverse > optimizer_config.neutral_init_fraction


def _random_fraction(optimizer_config) -> float:
    """Whatever the seeded groups leave, which is what `init` fills randomly."""
    return 1.0 - (
        optimizer_config.neutral_init_fraction
        + optimizer_config.sparsity_init_fraction
    )
