"""An unreached guarantee must be reported, and the search for one must be bounded.

Two coupled behaviours:

**The resample loop is bounded.** `tune_pipeline` draws a profiling sample, optimizes, and
-- if the guarantees are not met -- may draw another and try again. Without a cap other
than the base table's size, a large table would mean hundreds of full GD solves.
`OptimizationConfig.max_sampling_rounds` bounds it, and is **one** by default: profile the
sample budget, optimize, done.

**A miss is reported rather than hidden.** When the rounds run out without meeting the
targets, the plan falls back to the highest-quality *executable* operator everywhere and
`guarantee_met=False` reaches the results table. With a model as the last tier, a plan
can always escalate tuples to a gold operator the loss scores as perfect; with a human
label source excluded from execution, a target can simply be out of reach.
"""

import pytest

try:
    import torch

    from reasondb.optimizer.gd_optimizer import (
        FEASIBILITY_TOLERANCE,
        VIOLATION_LOSS_MULTIPLIER,
        GradientDescentOptimizer,
        OptimizationConfig,
        OptimizationLoss,
        OptimizationReport,
        is_feasible,
        raw_violation,
    )
    from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
except ImportError:  # pragma: no cover - optional deps
    pytest.skip("optimizer deps not installed", allow_module_level=True)


#: Imported rather than restated: these tests build a loss the way
#: `post_optimization_check` does, so a second copy of the number would keep passing after
#: the real one changed, and `_build_report` would silently start reporting bounds off by
#: the ratio between them.
VIOLATION_MULTIPLIER = VIOLATION_LOSS_MULTIPLIER


# --- the feasibility test ------------------------------------------------------


def _scaled(raw):
    """A violation as it appears on the loss: raw, times the multiplier, in float32 --
    the dtype `compute_loss` produces, and the reason the tolerance matters."""
    return torch.tensor([raw], dtype=torch.float32) * VIOLATION_MULTIPLIER


def test_no_violation_is_feasible():
    assert is_feasible(_scaled(0.0)).all()


def test_a_violation_the_sample_could_not_have_resolved_is_feasible():
    """The bound is a normal approximation over a few hundred tuples, uncertain at
    O(1e-2). Anything this far inside it is noise, not a miss."""
    assert is_feasible(_scaled(1e-9)).all()
    assert is_feasible(_scaled(1e-5)).all()


def test_a_real_miss_is_not_feasible():
    """A guarantee that actually failed misses by a margin the tolerance never touches --
    a percentage point is four orders of magnitude above it."""
    assert not is_feasible(_scaled(0.01)).any()
    assert not is_feasible(_scaled(0.2)).any()


def test_the_tolerance_is_the_boundary():
    assert not is_feasible(_scaled(FEASIBILITY_TOLERANCE)).any()
    assert is_feasible(_scaled(FEASIBILITY_TOLERANCE / 2)).all()


def test_a_bound_a_few_float32_ulps_short_of_its_target_still_counts_as_met():
    """Float32 rounding noise must not count as a miss. A threshold of `< 1.0` on the
    *scaled* violation would mean raw < 1e-7; one ulp at a 0.9 target is 5.96e-8, so it
    would tolerate one representable step and fail at two -- and two ulps is nothing, given
    the rescale in `compute_loss` and the sums across cascades."""
    target = torch.tensor([0.9], dtype=torch.float32)
    zero = torch.zeros_like(target)
    one_ulp_short = torch.nextafter(target, zero)
    two_ulps_short = torch.nextafter(one_ulp_short, zero)

    def scaled_violation(bound):
        return torch.relu(target - bound) * VIOLATION_MULTIPLIER

    # A threshold at 1.0 splits these two: it tolerates one step and not the next.
    assert scaled_violation(one_ulp_short).item() < 1.0
    assert scaled_violation(two_ulps_short).item() > 1.0

    # Both are noise, and both are met.
    assert is_feasible(scaled_violation(one_ulp_short)).all()
    assert is_feasible(scaled_violation(two_ulps_short)).all()


def test_feasibility_is_elementwise():
    """`post_optimization_check` passes one entry per budget, not a scalar."""
    verdicts = is_feasible(
        torch.tensor([0.0, 1e-5, 0.01], dtype=torch.float32) * VIOLATION_MULTIPLIER
    )
    assert verdicts.tolist() == [True, True, False]


def test_raw_violation_inverts_the_multiplier():
    """The report's bounds and the feasibility test must read the same number, so both go
    through this one conversion."""
    assert raw_violation(_scaled(0.05)).item() == pytest.approx(0.05, rel=1e-6)
    assert raw_violation(0.05 * VIOLATION_MULTIPLIER) == pytest.approx(0.05, rel=1e-6)


def test_the_tolerance_sits_clear_of_float32_noise():
    """A guard on the constant itself: shrink it below ~1e-6 and the ulp failure above
    reappears, since the violation is computed and stored in float32."""
    assert FEASIBILITY_TOLERANCE > 100 * torch.finfo(torch.float32).eps
    assert FEASIBILITY_TOLERANCE < 1e-2, "must stay well inside the sampling noise"


# --- the round budget ----------------------------------------------------------


def test_one_sampling_round_by_default():
    """Profile the sample budget once, optimize, done."""
    assert OptimizationConfig().max_sampling_rounds == 1


def test_the_budget_is_what_makes_more_rounds():
    """Rounds are derived from the row budget, not configured beside it.

    150 rows from the default 20-row first round is four rounds; there is no separate
    count that could disagree with what the sampler will actually draw.
    """
    config = OptimizationConfig(adaptive_sampling=True, sample_size=150)
    assert config.max_sampling_rounds == 4
    # Halve the budget and the loop loses a round, with nothing else touched.
    assert OptimizationConfig(adaptive_sampling=True, sample_size=70).max_sampling_rounds == 3


# --- the report ----------------------------------------------------------------


def _report(precision_lower, recall_lower, precision, recall, meets_targets):
    """Drive `_build_report` with a loss shaped as `post_optimization_check` sees it:
    violations already scaled by the multiplier."""
    optimizer = GradientDescentOptimizer(OptimizationConfig(device=torch.device("cpu")))
    loss = OptimizationLoss(
        precision_violation=torch.tensor(
            [max(0.0, precision - precision_lower) * VIOLATION_MULTIPLIER]
        ),
        recall_violation=torch.tensor(
            [max(0.0, recall - recall_lower) * VIOLATION_MULTIPLIER]
        ),
        costs=torch.tensor([[1.0]]),
        max_costs=torch.tensor([[1.0]]),
    )
    return optimizer._build_report(
        loss=loss,
        guarantees=[PrecisionGuarantee(precision), RecallGuarantee(recall)],
        job_index=0,
        meets_targets=meets_targets,
    )


def test_a_missed_target_reports_how_far_short_it_fell():
    """`achieved_*_lower` is what the guarantee was actually judged on, reconstructed
    from the violation -- so a result row shows the gap, not just the failure."""
    report = _report(
        precision_lower=0.82, recall_lower=0.79, precision=0.99, recall=0.99,
        meets_targets=False,
    )
    assert report.meets_targets is False
    assert report.achieved_precision_lower == pytest.approx(0.82)
    assert report.achieved_recall_lower == pytest.approx(0.79)
    assert report.precision_target == 0.99


def test_a_met_target_reports_no_violation():
    report = _report(
        precision_lower=0.95, recall_lower=0.95, precision=0.9, recall=0.9,
        meets_targets=True,
    )
    assert report.meets_targets is True
    # No violation, so the reconstruction returns the target itself.
    assert report.achieved_precision_lower == pytest.approx(0.9)


def test_the_report_serializes_for_telemetry():
    payload = _report(0.82, 0.79, 0.99, 0.99, meets_targets=False).to_json()
    assert payload["guarantee_met"] is False
    assert payload["achieved_precision_lower"] == pytest.approx(0.82)
    assert set(payload) == {
        "guarantee_met",
        "achieved_precision_lower",
        "achieved_recall_lower",
        "precision_target",
        "recall_target",
    }


# --- what the executor reports for a whole query -------------------------------


class _Executor:
    """Just the method under test -- constructing a real `Executor` needs a database, a
    reasoner and a configurator."""

    from reasondb.executor import Executor

    _guarantee_telemetry = Executor._guarantee_telemetry

    def __init__(self, reports):
        self.last_optimization_reports = reports


def _stub(meets, precision_lower, recall_lower):
    return OptimizationReport(
        meets_targets=meets,
        achieved_precision_lower=precision_lower,
        achieved_recall_lower=recall_lower,
        precision_target=0.9,
        recall_target=0.9,
    )


def test_one_missed_pipeline_makes_the_whole_query_missed():
    """The guarantee is end-to-end, so it is an `all`, not an `any`."""
    telemetry = _Executor(
        [_stub(True, 0.95, 0.95), _stub(False, 0.4, 0.4)]
    )._guarantee_telemetry()
    assert telemetry["guarantee_met"] is False
    # The worst pipeline is what the query achieved, for the same reason.
    assert telemetry["achieved_precision_lower"] == pytest.approx(0.4)


def test_all_pipelines_meeting_the_target_means_the_query_did():
    telemetry = _Executor(
        [_stub(True, 0.95, 0.93), _stub(True, 0.97, 0.99)]
    )._guarantee_telemetry()
    assert telemetry["guarantee_met"] is True
    assert telemetry["achieved_recall_lower"] == pytest.approx(0.93)


def test_no_reports_emits_nothing():
    """The baselines and the label pass produce none; the monitor rejects only *missing
    required* keys, so omitting these is safe -- and "not measured" must not be recorded
    as "measured and met"."""
    assert _Executor([])._guarantee_telemetry() == {}
