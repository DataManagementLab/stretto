"""A label operator is profiled but never planned.

The whole point of human labels is that the label source is *not* an operator the plan may
fall back to -- selecting it would mean labelling the dataset by hand at query time. Every
place that selects `step.operators[-1]` is covered here, since each is independently
capable of putting it back in a plan.
"""

import pytest

try:
    import torch

    from reasondb.optimizer.base_optimizer import (
        OperatorChoiceKey,
        Optimizer,
        PipelineSearchSpace,
    )
    from reasondb.optimizer.gd_optimizer import DifferentiableConfig, OptimizationConfig
    from reasondb.query_plan.physical_operator import (
        LABEL_OPERATOR_QUALITY,
        PhysicalOperatorsNoPseudos,
    )
    from reasondb.query_plan.tuning_parameters import TuningParameterContinuous
    from reasondb.query_plan.unoptimized_physical_plan import UnoptimizedPhysicalPlanStep
except ImportError:  # pragma: no cover - optional deps
    pytest.skip("optimizer deps not installed", allow_module_level=True)


class _Interface:
    def __init__(self, name):
        self.name = name


class _FakeOperator:
    def __init__(self, name, quality, is_label_only=False, tuning_parameters=()):
        self.quality = quality
        self.is_label_only = is_label_only
        self._name = name
        self._tuning_parameters = list(tuning_parameters)

    def get_llm_parameters(self):
        return _Interface(self._name)

    def get_operation_identifier(self):
        return self._name

    def get_tuning_parameters(self):
        return self._tuning_parameters


class _FakeLogicalStep:
    validated = True
    inputs = ()
    output = None
    expression = "x"


def _thresholds():
    """Both bounds, as a real threshold filter has -- Lotus asserts on finding exactly
    one of each for its proxy operator."""
    return [
        TuningParameterContinuous(
            name=name, default=0.0, init=init, min=-10.0, max=10.0, log_scale=False
        )
        for name, init in (
            ("logodds_threshold_lower", -1.0),
            ("logodds_threshold_upper", 1.0),
        )
    ]


def _step(with_label_operator: bool, index: int = 0):
    operators = [
        _FakeOperator("cheap", 1.0, tuning_parameters=_thresholds()),
        _FakeOperator("vanilla70B", 8.0, tuning_parameters=_thresholds()),
    ]
    step = UnoptimizedPhysicalPlanStep(
        logical_plan_step=_FakeLogicalStep(),
        operators=PhysicalOperatorsNoPseudos(operators),
        llm_configurations={"cheap": {}, "vanilla70B": {}},
        estimated_best_operator_idx=0,
    )
    step._index = index
    if with_label_operator:
        step.attach_label_operator(
            _FakeOperator("Labeler", LABEL_OPERATOR_QUALITY, is_label_only=True), {}
        )
    return step


class _FakePipeline:
    def __init__(self, step):
        self.steps_in_parallel = [[step]]


def _search_space(with_label_operator: bool) -> PipelineSearchSpace:
    return Optimizer.get_search_space(
        None, _FakePipeline(_step(with_label_operator)), logger=None
    )


def _diff_config(with_label_operator: bool) -> DifferentiableConfig:
    config = DifferentiableConfig(
        search_space=_search_space(with_label_operator),
        rng=torch.Generator().manual_seed(0),
        num_initializations=4,
        num_budgets_to_test=1,
        num_methods=1,
        num_gold_mixing_params=[1],
        batch_size=None,
    )
    config.init(OptimizationConfig(device=torch.device("cpu")))
    return config


# --- the pick score ------------------------------------------------------------

KEY = OperatorChoiceKey(0, 0)


def test_label_operator_is_never_picked():
    config = _diff_config(with_label_operator=True)
    scores = config.get_operator_pick_score(
        cascade_id=0, level=0, physical_operator_id=2
    )
    assert (scores < 0).all(), "a label operator must be unpickable, not forced on"


def test_a_model_in_the_last_slot_is_still_forced_on():
    """Without a label source the highest-quality model keeps its role as the
    resolver for every tuple the cheaper tiers leave unsure."""
    config = _diff_config(with_label_operator=False)
    scores = config.get_operator_pick_score(
        cascade_id=0, level=0, physical_operator_id=1
    )
    assert (scores > 0).all()


def test_the_best_model_becomes_a_real_choice():
    """With a label source attached, the vanilla 70B moves to the second-to-last slot,
    where it gets a learnable pick score instead of a hardcoded constant."""
    config = _diff_config(with_label_operator=True)
    scores = config.get_operator_pick_score(
        cascade_id=0, level=0, physical_operator_id=1
    )
    assert scores.requires_grad or scores.grad_fn is not None or scores.is_leaf
    # A constant would be identical across restarts; a learnable score is not.
    assert len(torch.unique(scores)) > 1


def test_pick_parameter_allocation_rule_is_unchanged():
    """One learnable pick score per candidate *except the last*, either way -- the last
    one is a constant, only its sign differs. The absolute counts do differ (3 candidates
    vs 2), and that is the point: the vanilla 70B gains a score it did not have."""
    with_label = _diff_config(with_label_operator=True)
    without = _diff_config(with_label_operator=False)
    assert with_label._all_pick_scores.shape[1] == 3 - 1
    assert without._all_pick_scores.shape[1] == 2 - 1


# --- tuning parameters ---------------------------------------------------------


def test_the_best_model_gains_tunable_parameters():
    """`get_search_space` pins the *last* candidate's parameters so the labels cannot
    move as the optimizer tunes. When a label operator takes that slot, the vanilla 70B
    below it is freed to be tuned like any other candidate."""
    space = _search_space(with_label_operator=True)
    tunable = {
        (k.physical_operator_id, k.tuning_parameter)
        for k in space.parameter_search_spaces
        if not k.fixed
    }
    assert (1, "logodds_threshold_upper") in tunable

    space_without = _search_space(with_label_operator=False)
    tunable_without = {
        (k.physical_operator_id, k.tuning_parameter)
        for k in space_without.parameter_search_spaces
        if not k.fixed
    }
    assert (1, "logodds_threshold_upper") not in tunable_without


def test_search_space_records_the_label_operator_index():
    space = _search_space(with_label_operator=True)
    assert space.label_operator_index[KEY] == 2
    assert space.get_cascade_search_space(0).has_label_operator is True

    space_without = _search_space(with_label_operator=False)
    assert space_without.label_operator_index[KEY] is None
    assert space_without.get_cascade_search_space(0).has_label_operator is False


# --- the other selectors -------------------------------------------------------


def test_last_executable_index_skips_the_label_operator():
    assert _step(with_label_operator=True).get_last_executable_operator_index() == 1
    assert _step(with_label_operator=False).get_last_executable_operator_index() == 1


def test_a_step_whose_only_candidate_is_a_label_operator_executes_it():
    """The gold-label pass: `get_label_configurator`'s toolbox is the perfect operators,
    so every step it configures holds exactly one candidate and that candidate is
    label-only. Skipping it there would leave a well-formed step with nothing to run.

    Distinguishing it from an attached label operator by the operator count is sound
    because `attach_label_operator` appends to a step `parse` already refused to build
    empty; the assertion below is what keeps that true."""
    step = UnoptimizedPhysicalPlanStep(
        logical_plan_step=_FakeLogicalStep(),
        operators=PhysicalOperatorsNoPseudos(
            [_FakeOperator("PerfectFilter", LABEL_OPERATOR_QUALITY, is_label_only=True)]
        ),
        llm_configurations={"PerfectFilter": {}},
        estimated_best_operator_idx=0,
    )

    assert step.get_label_operator_index() == 0
    assert step.get_last_executable_operator_index() == 0


def test_attaching_a_label_operator_always_leaves_something_below_it():
    step = _step(with_label_operator=True)
    assert len(step.operators) > 1
    assert step.get_last_executable_operator_index() < step.get_label_operator_index()


def test_label_optimizer_picks_the_last_executable_operator():
    """Otherwise the silver label pass would run the label source and emit ground truth
    under the name 'silver', invalidating every row scored against it."""
    step = _step(with_label_operator=True)
    operator = step.operators[step.get_last_executable_operator_index()]
    assert operator.get_operation_identifier() == "vanilla70B"


def _lotus_pipeline(with_label_operator: bool):
    step = _step(with_label_operator)
    pipeline = _FakePipeline(step)
    pipeline.steps_in_order_with_ids = [(0, 0, step)]
    return step, pipeline


def test_lotus_refuses_a_plan_with_a_label_operator():
    """Lotus tunes its thresholds to match its *silver* operator, and both threshold
    searches assume escalation resolves to that reference. Against human labels the
    recall estimate is optimistic and the precision target is unreachable, in neither
    case visibly -- so the plan is rejected rather than tuned into a number that reads
    like a Lotus guarantee and is not one."""
    from reasondb.optimizer.baselines.lotus_optimizer import (
        LotusHumanLabelsUnsupported,
        LotusSearchSpace,
    )

    _, pipeline = _lotus_pipeline(with_label_operator=True)

    with pytest.raises(LotusHumanLabelsUnsupported, match="human label source"):
        LotusSearchSpace.from_pipeline_search_space(
            pipeline=pipeline,
            pipeline_search_space=_search_space(with_label_operator=True),
            proxy_operators={"cheap"},
            logger=_NullLogger(),
        )


def test_lotus_still_builds_its_cascade_without_a_label_operator():
    """The refusal must be scoped to the human-labels mode: Lotus's own silver/proxy
    selection is unchanged for every ordinary run."""
    from reasondb.optimizer.baselines.lotus_optimizer import LotusSearchSpace

    step, pipeline = _lotus_pipeline(with_label_operator=False)

    space = LotusSearchSpace.from_pipeline_search_space(
        pipeline=pipeline,
        pipeline_search_space=_search_space(with_label_operator=False),
        proxy_operators={"cheap"},
        logger=_NullLogger(),
    )
    cascade = space.search_space[0, 0]
    assert step.operators[cascade.silver_operator_id].get_operation_identifier() == (
        "vanilla70B"
    )
    assert step.operators[cascade.proxy_operator_id].get_operation_identifier() == (
        "cheap"
    )


class _NullLogger:
    def __truediv__(self, _other):
        return self

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def debug(self, *_args, **_kwargs):
        pass
