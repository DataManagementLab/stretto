"""Annotation is counted, not priced -- and kept out of the optimizer's arithmetic.

Two separate things, easy to conflate:

**The exclusion is load-bearing.** Any cost a ``Perfect*`` label operator reports (e.g. a
large per-row sentinel) would, once it is profiled, reach three places:

- ``total_cost``, the tuning cost of every benchmark row;
- ``total_cost_per_sample``, which ``post_optimization_check`` weighs against the value of
  drawing a bigger sample -- a large figure there makes the optimizer stop sampling after
  the first batch, always;
- ``per_operator_and_sample_cost``, which builds the differentiable cost model's
  ``cost_vector``, whose sum is the denominator every job's cost is scaled by
  (``OptimizationLoss.scaled_cost``) -- a large entry makes the optimizer cost-blind.

**The price is not ours to invent.** A label operator therefore reports zero, and what
gets reported instead is ``n_labels_requested``: how many tuples a human actually had to
label. That is measurable. Seconds-per-label and dollars-per-label are assumptions for
whoever reads the results table, and a fabricated figure in a CSV column reads as data.
"""

import pytest

try:
    from reasondb.optimizer.profiler import ProfilingOutput
    from reasondb.query_plan.physical_operator import ProfilingCost
except ImportError:  # pragma: no cover - optional deps
    pytest.skip("optimizer deps not installed", allow_module_level=True)


MODEL_KEY = (0, 0, 1)
LABEL_KEY = (0, 0, 2)
SAMPLE_ROWS = 100


def _output(with_label_operator: bool = True) -> ProfilingOutput:
    out = ProfilingOutput()
    out.num_input_tuples[(0, 0)] = SAMPLE_ROWS
    out.per_operator_cost[(0, 0, 0)] = ProfilingCost(1.0, 0.1, 0.0)
    out.per_operator_cost[MODEL_KEY] = ProfilingCost(9.0, 0.9, 0.0)
    # A large per-row sentinel cost: the exclusion must hold regardless of what the
    # operator reports.
    out.per_operator_cost[LABEL_KEY] = ProfilingCost(1_000_000.0, 1_000_000.0, 0.0)
    if with_label_operator:
        out.label_only_keys.add(LABEL_KEY)
    return out


# --- the count -----------------------------------------------------------------


def test_labels_are_counted_not_priced():
    """One label per profiled tuple, per label-supplying step."""
    assert _output().n_labels_requested == SAMPLE_ROWS


def test_two_labelled_steps_need_two_passes_of_labels():
    out = _output()
    out.num_input_tuples[(1, 0)] = 40
    out.label_only_keys.add((1, 0, 3))
    assert out.n_labels_requested == SAMPLE_ROWS + 40


def test_a_model_labelled_run_needs_no_human():
    assert _output(with_label_operator=False).n_labels_requested == 0


# --- the exclusion --------------------------------------------------------------


def test_total_cost_excludes_the_label_operator():
    assert _output().total_cost.runtime == pytest.approx(10.0)
    assert _output().total_cost.monetary_cost == pytest.approx(1.0)


def test_without_a_label_operator_nothing_changes():
    """The exclusion is keyed on `label_only_keys`, so an ordinary run is unaffected."""
    out = _output(with_label_operator=False)
    assert out.total_cost.runtime == pytest.approx(1_000_010.0)
    assert out.n_labels_requested == 0


def test_the_resample_decision_does_not_see_the_label_operator():
    """`post_optimization_check` compares this against the value of a bigger sample. With
    the label operator included the comparison inverts: more samples would mean more
    labels, so the optimizer would refuse to sample."""
    assert _output().total_cost_per_sample.runtime == pytest.approx(0.1)


def test_the_cost_model_sees_zero_for_a_label_operator():
    """This is what keeps it out of `compute_cost`'s `max_cost`, which is
    `cost_vector.sum()` -- the denominator of every job's scaled cost."""
    out = _output()
    assert out.per_operator_and_sample_cost(0, 0, 2).runtime == 0.0
    assert out.per_operator_and_sample_cost(0, 0, 1).runtime == pytest.approx(0.09)


def test_prepend_keeps_the_label_keys_from_both_rounds():
    """Each sampling round profiles again; an operator absent from one round's output
    must not lose its label-source status and start being billed as query cost."""
    first = _output()
    second = ProfilingOutput()
    second.num_input_tuples[(0, 0)] = SAMPLE_ROWS
    second.per_operator_cost[LABEL_KEY] = ProfilingCost(1_000_000.0, 1_000_000.0, 0.0)
    second.merged_output_tuples = {}
    second.prepend(first, logger=_NullLogger())
    assert LABEL_KEY in second.label_only_keys
    assert second.total_cost.runtime == pytest.approx(0.0)


class _NullLogger:
    def __truediv__(self, _other):
        return self

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass
