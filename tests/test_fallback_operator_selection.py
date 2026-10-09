"""Which operator a step falls back to when the plan would otherwise emit none.

The terminal executable tier is the answer whenever it profiled. When it is the gold
operator it always has (`Profiler.profile_level` papers a failed gold profile over with
`fallback_gold_profile`). A label operator, however, takes the gold slot and demotes it to
an ordinary candidate, whose `Mistake` is a plain skip -- so both fallback paths (GD's
per-step rescue and `Optimizer.fallback_pipeline`) have to cope with it being absent.
"""

import pytest

try:
    from reasondb.optimizer.base_optimizer import Optimizer
    from reasondb.query_plan.physical_operator import (
        LABEL_OPERATOR_QUALITY,
        PhysicalOperatorsNoPseudos,
    )
    from reasondb.query_plan.unoptimized_physical_plan import UnoptimizedPhysicalPlanStep
except ImportError:  # pragma: no cover - optional deps
    pytest.skip("optimizer deps not installed", allow_module_level=True)


class _Interface:
    def __init__(self, name):
        self.name = name


class _FakeOperator:
    def __init__(self, name, quality, is_label_only=False):
        self.quality = quality
        self.is_label_only = is_label_only
        self._name = name

    def get_llm_parameters(self):
        return _Interface(self._name)

    def get_operation_identifier(self):
        return self._name

    def get_tuning_parameters(self):
        return []


class _FakeLogicalStep:
    validated = True
    inputs = ()
    output = None
    expression = "x"


class _RecordingLogger:
    def __init__(self):
        self.warnings = []

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, _module, message):
        self.warnings.append(message)

    def debug(self, *_args, **_kwargs):
        pass


class _FakeProfilingOutput:
    def __init__(self, profiled_operator_ids):
        self.observations = {(0, 0, op_id): object() for op_id in profiled_operator_ids}


def _step(with_label_operator: bool):
    operators = [
        _FakeOperator("cheap", 1.0),
        _FakeOperator("vanilla70B", 8.0),
    ]
    step = UnoptimizedPhysicalPlanStep(
        logical_plan_step=_FakeLogicalStep(),
        operators=PhysicalOperatorsNoPseudos(operators),
        llm_configurations={"cheap": {}, "vanilla70B": {}},
        estimated_best_operator_idx=0,
    )
    step._index = 0
    if with_label_operator:
        step.attach_label_operator(
            _FakeOperator("Labeler", LABEL_OPERATOR_QUALITY, is_label_only=True), {}
        )
    return step


def _pick(with_label_operator: bool, profiled_operator_ids, logger=None):
    return Optimizer.get_fallback_operator_index(
        None,
        unoptimized_step=_step(with_label_operator),
        profiling_output=_FakeProfilingOutput(profiled_operator_ids),
        cascade_id=0,
        level=0,
        logger=logger or _RecordingLogger(),
    )


@pytest.mark.parametrize("with_label_operator", [True, False])
def test_the_terminal_tier_wins_when_it_profiled(with_label_operator):
    """The default answer is unchanged: the highest-quality operator a plan may run."""
    assert _pick(with_label_operator, profiled_operator_ids=[0, 1]) == 1


def test_falls_back_to_the_best_surviving_candidate():
    """The case human labels make possible. Dropping the step instead would silently pass every
    input row on; adding the unprofiled operator would put a quality and selectivity
    nothing measured into the guarantee arithmetic."""
    logger = _RecordingLogger()
    assert _pick(True, profiled_operator_ids=[0, 2], logger=logger) == 0
    assert any("falling back" in message for message in logger.warnings), (
        "silently planning a weaker tier than the step was optimized around is exactly "
        "the kind of thing that has to show up in the log"
    )


def test_a_profiled_terminal_tier_is_not_announced():
    logger = _RecordingLogger()
    _pick(True, profiled_operator_ids=[0, 1, 2], logger=logger)
    assert logger.warnings == []


def test_the_label_operator_is_never_the_fallback():
    """It profiled -- it always does, every other candidate is scored against it -- and it
    is still not something a plan may reach for: selecting it means asking a human to
    label every tuple at query time."""
    with pytest.raises(RuntimeError, match="No candidate"):
        _pick(True, profiled_operator_ids=[2])


def test_a_step_with_nothing_profiled_raises():
    """Not implementable, and not droppable either -- fail where the cause is."""
    with pytest.raises(RuntimeError, match="No candidate"):
        _pick(False, profiled_operator_ids=[])
