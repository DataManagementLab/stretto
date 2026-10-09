"""What a hypothetical larger profiling sample does to the confidence bounds.

Each budget slot asks "if I profiled `what_if` more rows and they behaved like the ones
I have, how tight would the bound get?". Scaling the counts and re-running the same Beta
posterior answers it: `sigma` shrinks as `1/sqrt(growth * n)`.

It is an *optimistic* estimate -- it credits rows nobody drew -- and the reason that is
safe is the invariant this file exists to pin:

    SLOT 0 IS EXACT, AND SLOT 0 IS THE ONLY THING THAT RENDERS A VERDICT.

Every test below is CPU-only and calls `_compute_metrics` directly; nothing here needs a
pipeline, a database or a GPU.
"""

import torch

from reasondb.optimizer.gd_optimizer import (
    GradientDescentOptimizer,
    OptimizationConfig,
)


class _StubConfig:
    """The four attributes `_compute_metrics` reads off a `DifferentiableConfig`."""

    def __init__(self, batch_size, num_budgets, num_initializations=1):
        self.batch_size = batch_size
        self.num_budgets = num_budgets
        self.num_initializations = num_initializations
        self.num_jobs_single_method = num_budgets * num_initializations
        self.remaining_budget = None


def _metrics(
    *,
    batch_size,
    num_budgets,
    num_initializations=1,
    tp=30.0,
    fp=10.0,
    fn=10.0,
    sample_frac=0.01,
    current_sample_size=100,
    **config_overrides,
):
    """Run `_compute_metrics` on flat per-job counts and return its 4-column stack."""
    optimizer = GradientDescentOptimizer(OptimizationConfig(**config_overrides))
    config = _StubConfig(batch_size, num_budgets, num_initializations)
    num_jobs = config.num_jobs_single_method
    # One profiled row per job column, holding the whole count -- `_compute_metrics`
    # only ever uses `.sum(dim=0)`, so this is the same as spreading it over n rows.
    shape = (1, num_jobs)
    return optimizer._compute_metrics(
        tp=torch.full(shape, tp),
        fp=torch.full(shape, fp),
        fn=torch.full(shape, fn),
        config=config,
        sample_frac=sample_frac,
        precision_confidence=0.95,
        recall_confidence=0.95,
        current_sample_size=current_sample_size,
    )


def _precision_lower(result):
    return result[:, 0]


def _recall_lower(result):
    return result[:, 2]


# ── the invariant ───────────────────────────────────────────────────────────────


def test_slot_zero_is_the_exact_bound_measured_on_drawn_rows():
    """The reported guarantee must never rest on rows nobody drew."""
    optimizer = GradientDescentOptimizer(OptimizationConfig())
    exact_precision = optimizer.beta_bounds(
        num_successes=torch.tensor([30.0]),
        num_failures=torch.tensor([10.0]),
        confidence=0.95,
    )
    exact_recall = optimizer.beta_bounds(
        num_successes=torch.tensor([30.0]),
        num_failures=torch.tensor([10.0]),
        confidence=0.95,
    )
    result = _metrics(batch_size=10, num_budgets=5)
    assert torch.allclose(_precision_lower(result)[0], exact_precision[0], atol=1e-6)
    # Recall additionally carries the finite-population blend, which at slot 0 mixes in
    # the empirical value over the sampled fraction. The bound it blends *from* is the
    # exact one, so it can only sit between the exact bound and the empirical value.
    empirical_recall = 30.0 / 40.0
    assert exact_recall[0] <= _recall_lower(result)[0] <= empirical_recall + 1e-6


def test_a_bigger_hypothetical_sample_tightens_both_bounds():
    result = _metrics(batch_size=10, num_budgets=5)
    precision = _precision_lower(result).tolist()
    recall = _recall_lower(result).tolist()
    # Strictly increasing: more (hypothetical) evidence, less slack in the bound.
    assert all(b > a for a, b in zip(precision, precision[1:])), precision
    assert all(b > a for a, b in zip(recall, recall[1:])), recall


def test_precision_and_recall_are_treated_alike():
    """Both precision and recall bounds respond to the what-if growth."""
    result = _metrics(batch_size=10, num_budgets=5, fp=10.0, fn=10.0)
    precision = _precision_lower(result)
    recall = _recall_lower(result)
    assert precision[-1] > precision[0]
    assert recall[-1] > recall[0]


def test_a_bound_never_exceeds_the_empirical_rate_it_bounds():
    result = _metrics(batch_size=10, num_budgets=5)
    assert torch.all(_precision_lower(result) <= result[:, 1] + 1e-6)
    assert torch.all(_recall_lower(result) <= result[:, 3] + 1e-6)


# ── the switches that make it inert ─────────────────────────────────────────────


def test_no_extrapolation_when_the_discount_is_zero():
    result = _metrics(batch_size=10, num_budgets=5, extrapolation_discount=0.0)
    precision = _precision_lower(result)
    assert torch.allclose(precision, precision[0].expand_as(precision), atol=1e-6)


def test_a_half_discount_lands_between_believing_none_and_all_of_it():
    none_of_it = _precision_lower(
        _metrics(batch_size=10, num_budgets=5, extrapolation_discount=0.0)
    )[-1]
    half = _precision_lower(
        _metrics(batch_size=10, num_budgets=5, extrapolation_discount=0.5)
    )[-1]
    all_of_it = _precision_lower(
        _metrics(batch_size=10, num_budgets=5, extrapolation_discount=1.0)
    )[-1]
    assert none_of_it < half < all_of_it


def test_the_default_non_adaptive_config_extrapolates_nothing():
    """The guard for every existing sweep: batch_size None -> one flat slot."""
    result = _metrics(batch_size=None, num_budgets=1, num_initializations=4)
    precision = _precision_lower(result)
    assert torch.allclose(precision, precision[0].expand_as(precision), atol=1e-9)


def test_the_finite_population_blend_is_recall_only():
    """Precision's slot-0 bound is the raw Beta bound, with no finite-population blend.

    The asymmetry is deliberate, not a missing feature: blending precision would raise
    slot 0 -- the bound reported as the achieved guarantee -- for every run ever made.
    Pinned here so that if someone adds the mirrored lines, they have to come and
    acknowledge that this is the number it moves.
    """
    optimizer = GradientDescentOptimizer(OptimizationConfig())
    exact = optimizer.beta_bounds(
        num_successes=torch.tensor([30.0]),
        num_failures=torch.tensor([10.0]),
        confidence=0.95,
    )[0]
    result = _metrics(batch_size=10, num_budgets=5)
    assert torch.allclose(_precision_lower(result)[0], exact, atol=1e-6)
    # Recall's slot 0 *is* blended, so it sits strictly above its own raw bound.
    raw_recall = optimizer.beta_bounds(
        num_successes=torch.tensor([30.0]),
        num_failures=torch.tensor([10.0]),
        confidence=0.95,
    )[0]
    assert _recall_lower(result)[0] > raw_recall


# ── layout ──────────────────────────────────────────────────────────────────────


def test_every_restart_in_a_budget_block_sees_the_same_hypothetical_sample():
    """Jobs are budget-major: `num_initializations` restarts share one budget slot."""
    result = _metrics(batch_size=10, num_budgets=3, num_initializations=4)
    precision = _precision_lower(result).tolist()
    blocks = [precision[0:4], precision[4:8], precision[8:12]]
    for block in blocks:
        assert len(set(round(v, 9) for v in block)) == 1, block
    assert blocks[0][0] < blocks[1][0] < blocks[2][0]
