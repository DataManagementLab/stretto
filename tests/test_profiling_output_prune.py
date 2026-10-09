"""Accumulating profiling output across rounds when the operator set shrinks.

Two things are pinned here, both of which only become observable once profiling happens
more than once:

- ``prepend`` must survive an operator that the *previous* round profiled and this one
  did not; indexing ``self.profiler_outputs[key]`` off ``other``'s keys would raise
  ``KeyError``, which would make operator pruning impossible.
- Cost must be divided by the tuples that *operator* ran on, not by the level's
  ever-growing total. Otherwise an operator that sat out a round looks cheaper by
  exactly the fraction of rounds it missed -- so a flaky or pruned operator becomes the
  optimizer's favourite.
"""

import pandas as pd
import torch

from reasondb.optimizer.profiler import ProfilingOutput
from reasondb.query_plan.physical_operator import ProfilingCost
from reasondb.utils.logging import FileLogger


def _output(operator_ids, *, index_start, num_rows, cost_per_operator):
    """A one-cascade, one-level `ProfilingOutput` over `num_rows` distinct rows."""
    out = ProfilingOutput()
    index = list(range(index_start, index_start + num_rows))
    frame = pd.DataFrame({"0": index}, index=index)
    out.merged_output_tuples[(0, 0)] = frame
    out.labels[(0, 0)] = torch.ones(num_rows, dtype=torch.bool)
    out.num_input_tuples[(0, 0)] = num_rows
    for operator_id in operator_ids:
        key = (0, 0, operator_id)
        out.profiler_outputs[key] = torch.zeros(num_rows, 1)
        out.output_masks[key] = torch.ones(num_rows, dtype=torch.bool)
        out.per_operator_output_tuples[key] = frame.copy()
        out.observations[key] = f"observation-{operator_id}"
        out.per_operator_cost[key] = ProfilingCost(cost_per_operator, 0.0, 0.0)
        out.num_input_tuples_per_operator[key] = num_rows
    out.compute_numeric_indexes()
    return out


def _accumulate(first, second):
    """`second` is this round, `first` the previous one -- the call order in the loop."""
    second.prepend(first, logger=FileLogger())
    return second


def test_an_operator_missing_this_round_does_not_raise():
    previous = _output([0, 1], index_start=0, num_rows=4, cost_per_operator=8.0)
    current = _output([0], index_start=4, num_rows=4, cost_per_operator=8.0)
    merged = _accumulate(previous, current)
    assert (0, 0, 1) not in merged.observations


def test_a_pruned_operator_is_forgotten_rather_than_half_carried():
    """Half an operator reads downstream as one that discarded every later row."""
    previous = _output([0, 1], index_start=0, num_rows=4, cost_per_operator=8.0)
    current = _output([0], index_start=4, num_rows=4, cost_per_operator=8.0)
    merged = _accumulate(previous, current)
    for store in (
        merged.profiler_outputs,
        merged.output_masks,
        merged.per_operator_output_tuples,
        merged.observations,
        merged.per_operator_cost,
        merged.num_input_tuples_per_operator,
    ):
        assert (0, 0, 1) not in store
    assert (0, 0, 1) not in merged.label_only_keys


def test_the_surviving_operator_keeps_both_rounds_of_rows():
    previous = _output([0, 1], index_start=0, num_rows=4, cost_per_operator=8.0)
    current = _output([0], index_start=4, num_rows=4, cost_per_operator=8.0)
    merged = _accumulate(previous, current)
    assert merged.profiler_outputs[(0, 0, 0)].shape[0] == 8
    assert merged.num_input_tuples[(0, 0)] == 8
    assert merged.num_input_tuples_per_operator[(0, 0, 0)] == 8


def test_cost_per_tuple_uses_the_rounds_the_operator_actually_ran():
    """Each operator's cost is divided by the tuples that operator itself ran on."""
    previous = _output([0, 1], index_start=0, num_rows=4, cost_per_operator=8.0)
    current = _output([0], index_start=4, num_rows=4, cost_per_operator=8.0)
    merged = _accumulate(previous, current)
    # Operator 0 ran on all 8 rows for 16.0 total.
    assert merged.per_operator_and_sample_cost(0, 0, 0).runtime == 2.0
    # Operator 1 is gone entirely, so it prices at nothing rather than at a discount.
    assert merged.per_operator_and_sample_cost(0, 0, 1).runtime == 0.0


def test_an_operator_profiled_in_one_round_of_three_is_priced_over_that_round():
    a = _output([0, 1], index_start=0, num_rows=4, cost_per_operator=8.0)
    b = _output([0, 1], index_start=4, num_rows=4, cost_per_operator=8.0)
    c = _output([0, 1], index_start=8, num_rows=4, cost_per_operator=8.0)
    merged = _accumulate(_accumulate(a, b), c)
    assert merged.num_input_tuples[(0, 0)] == 12
    assert merged.num_input_tuples_per_operator[(0, 0, 0)] == 12
    # 24.0 over 12 tuples, the same per-tuple price each round measured on its own.
    assert merged.per_operator_and_sample_cost(0, 0, 0).runtime == 2.0


def test_the_single_round_path_is_unchanged():
    """No previous round: the per-operator count and the level count agree anyway."""
    only = _output([0, 1], index_start=0, num_rows=4, cost_per_operator=8.0)
    only.prepend(None, logger=FileLogger())
    assert only.per_operator_and_sample_cost(0, 0, 0).runtime == 2.0
    assert only.total_cost.runtime == 16.0
    assert only.total_cost_per_sample.runtime == 4.0


def test_the_divisor_falls_back_when_per_operator_counts_are_absent():
    """A `ProfilingOutput` built without per-operator counts."""
    out = _output([0], index_start=0, num_rows=4, cost_per_operator=8.0)
    out.num_input_tuples_per_operator.clear()
    assert out.per_operator_and_sample_cost(0, 0, 0).runtime == 2.0
