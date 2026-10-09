"""Where a `ProfilingOutput`'s tensors live, across a multi-round optimization.

`GradientDescentOptimizer.gd_optimize` moves the profiling output onto the optimizer's
device, and the object is shared: the next sampling round prepends the previous round's
output to a freshly profiled one, which arrives on the CPU. Without explicit alignment
the merge would inherit whatever the previous consumer left behind, which is a no-op when
the optimizer runs on the CPU and a hard device mismatch when it does not. Adaptive
sampling makes the multi-round path routine rather than opt-in.

These tests stand `meta` in for the accelerator: it needs no GPU, and `torch.cat`
rejects a meta/CPU mix exactly the way it rejects a cuda/CPU one. The orientation is
flipped from the real case (there the *previous* round is the one on the accelerator)
because you cannot copy data *out* of a meta tensor -- what is under test is that
`prepend` puts both sides on one device before concatenating, which is the same code
either way.
"""

import pandas as pd
import pytest

try:
    import torch

    from reasondb.optimizer.profiler import ProfilingOutput
    from reasondb.query_plan.physical_operator import ProfilingCost
except ImportError:  # pragma: no cover - optional deps
    pytest.skip("optimizer deps not installed", allow_module_level=True)


CPU = torch.device("cpu")
OTHER_DEVICE = torch.device("meta")
KEY = (0, 0)
OP_KEY = (0, 0, 0)


class _NullLogger:
    def __truediv__(self, _other):
        return self

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def _round(indices: list) -> ProfilingOutput:
    """One sampling round's output over `indices`, one cascade, one operator."""
    out = ProfilingOutput()
    frame = pd.DataFrame({"value": list(range(len(indices)))}, index=indices)
    out.merged_output_tuples[KEY] = frame
    out.per_operator_output_tuples[OP_KEY] = frame
    out.labels[KEY] = torch.zeros(len(indices), dtype=torch.bool)
    out.profiler_outputs[OP_KEY] = torch.zeros(len(indices), 1)
    out.output_masks[OP_KEY] = torch.ones(len(indices), dtype=torch.bool)
    out.observations[OP_KEY] = object()
    out.per_operator_cost[OP_KEY] = ProfilingCost(1.0, 0.1, 0.0)
    out.num_input_tuples[KEY] = len(indices)
    return out


def test_device_reports_where_the_tensors_are():
    assert _round([0, 1]).device == CPU
    assert _round([0, 1]).to(OTHER_DEVICE).device == OTHER_DEVICE


def test_device_of_an_empty_output_is_unknown():
    assert ProfilingOutput().device is None


def test_prepending_a_round_from_another_device_aligns_first():
    """The second round's `torch.cat` must align devices, or it raises once round one
    has been moved."""
    previous = _round([0, 1])
    current = _round([2, 3]).to(OTHER_DEVICE)

    current.prepend(previous, logger=_NullLogger())

    assert current.labels[KEY].shape[0] == 4
    assert current.labels[KEY].device == OTHER_DEVICE
    assert current.profiler_outputs[OP_KEY].device == OTHER_DEVICE
    assert current.output_masks[OP_KEY].device == OTHER_DEVICE


def test_prepending_two_rounds_on_one_device_is_unchanged():
    previous = _round([0, 1])
    current = _round([2, 3])

    current.prepend(previous, logger=_NullLogger())

    assert current.labels[KEY].shape[0] == 4
    assert current.device == CPU
    assert current.num_input_tuples[KEY] == 4


def test_every_store_really_does_move_together():
    """The invariant `device` claims -- turned into an assertion rather than a docstring.

    `device` answers from the first tensor it finds, so a store that stays behind makes
    it report a device the object does not uniformly hold. This includes
    `_numeric_indexes`: `compute_numeric_indexes` must build its index tensors on the
    output's device, because they feed `scatter_reduce` against the optimizer's device
    tensors in `compute_merged_scores`.
    """
    output = _round([0, 1]).to(OTHER_DEVICE)
    # `prepend` recomputes the numeric indexes; they must land on the same device as the
    # rest of the output, not on the default device.
    output.prepend(None, logger=_NullLogger())

    stores = {
        "labels": output.labels,
        "profiler_outputs": output.profiler_outputs,
        "output_masks": output.output_masks,
        "_numeric_indexes": output._numeric_indexes,
    }
    for name, store in stores.items():
        for key, tensor in store.items():
            assert tensor.device == OTHER_DEVICE, f"{name}[{key}] stayed behind"
    assert output.device == OTHER_DEVICE
