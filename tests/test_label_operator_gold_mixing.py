"""Gold mixing is closed for any plan whose labels come from a human.

"Gold mixing" lets a cheaper operator declare a fraction of tuples unsure and hand them to
the step's last tier; the loss then credits those tuples with precision and recall of
exactly 1.0 (`compute_split_loss`'s `(1-f)*p + f*1.0`, `compute_global_loss`'s
`p / (p + (1-f)(1-p))`). That is why *every* guarantee is satisfiable today: the optimizer
can always buy its way to the target by escalating more tuples.

It is only defensible when the last tier is a model the plan can run. When it is a human
label source, escalating means paying a person per tuple at query time -- the one thing
the human-labels mode exists to prevent -- so the channel is closed outright rather than
merely made expensive, and unreachable targets become genuinely unreachable.

The knob's *granularity* varies by optimization mode, which is why every mode is covered
here: GLOBAL has one knob for the whole plan, LOCAL/SHIFT_BUDGET one per cascade, COMBO
both.
"""

import pytest

try:
    import torch

    from reasondb.optimizer.base_optimizer import Optimizer, PipelineSearchSpace
    from reasondb.optimizer.gd_optimizer import DifferentiableConfig, OptimizationConfig
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


def _step(with_label_operator: bool, index: int):
    step = UnoptimizedPhysicalPlanStep(
        logical_plan_step=_FakeLogicalStep(),
        operators=PhysicalOperatorsNoPseudos(
            [_FakeOperator("cheap", 1.0), _FakeOperator("vanilla70B", 8.0)]
        ),
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
    def __init__(self, steps):
        # One cascade per step, mirroring a plan of independent filters.
        self.steps_in_parallel = [[s] for s in steps]


def _config(labelled_cascades, num_cascades, num_gold_mixing_params):
    steps = [_step(i in labelled_cascades, index=i) for i in range(num_cascades)]
    search_space = Optimizer.get_search_space(None, _FakePipeline(steps), logger=None)
    config = DifferentiableConfig(
        search_space=search_space,
        rng=torch.Generator().manual_seed(0),
        num_initializations=4,
        num_budgets_to_test=1,
        num_methods=2 if len(num_gold_mixing_params) > 1 else 1,
        num_gold_mixing_params=num_gold_mixing_params,
        batch_size=None,
    )
    config.init(OptimizationConfig(device=torch.device("cpu")))
    # Push the raw parameters high, so an unmasked sigmoid would read ~1.0 and a masked
    # one is unambiguously 0.
    with torch.no_grad():
        for param in list(config._not_allow_accept) + list(config._not_allow_discard):
            param.fill_(10.0)
    return config


def _scores(config, cascade_id):
    return (
        config.not_allow_accept_scores(cascade_id, temperature=1.0),
        config.not_allow_discard_scores(cascade_id, temperature=1.0),
    )


# --- the channel is open when there is no label operator -----------------------


def test_gold_mixing_stays_open_without_a_label_operator():
    config = _config(labelled_cascades=set(), num_cascades=2, num_gold_mixing_params=[1])
    accept, discard = _scores(config, cascade_id=0)
    assert (accept > 0.9).all()
    assert (discard > 0.9).all()


# --- and closed when there is ---------------------------------------------------


def test_global_mode_closes_the_single_shared_knob():
    config = _config(labelled_cascades={0}, num_cascades=1, num_gold_mixing_params=[1])
    accept, discard = _scores(config, cascade_id=0)
    assert (accept == 0).all()
    assert (discard == 0).all()


def test_per_cascade_mode_closes_only_the_labelled_cascade():
    """LOCAL/SHIFT_BUDGET give each cascade its own knob, so a mixed plan keeps deferral
    available for the steps that still have a model as their label source."""
    config = _config(
        labelled_cascades={1}, num_cascades=3, num_gold_mixing_params=[3]
    )
    assert (_scores(config, cascade_id=1)[0] == 0).all()
    assert (_scores(config, cascade_id=1)[1] == 0).all()
    for open_cascade in (0, 2):
        assert (_scores(config, cascade_id=open_cascade)[0] > 0.9).all()
        assert (_scores(config, cascade_id=open_cascade)[1] > 0.9).all()


def test_global_mode_is_conservative_for_mixed_plans():
    """One knob for the whole plan means one human-labelled step closes deferral
    everywhere. Conservative rather than optimistic: the alternative would credit
    accuracy nobody earned to steps sharing the knob."""
    config = _config(
        labelled_cascades={1}, num_cascades=3, num_gold_mixing_params=[1]
    )
    for cascade_id in range(3):
        assert (_scores(config, cascade_id)[0] == 0).all()


def test_combo_mode_masks_both_stacked_widths():
    """COMBO stacks a global knob and a per-cascade one, so the mask must be built by the
    same comprehension rather than broadcast against a single shape."""
    config = _config(
        labelled_cascades={0}, num_cascades=2, num_gold_mixing_params=[1, 2]
    )
    accept, discard = _scores(config, cascade_id=0)
    # Both halves of the hstack are covered.
    assert accept.shape == discard.shape
    assert (accept == 0).all()
    assert (discard == 0).all()


def test_the_aggregating_caller_gets_the_conservative_answer():
    """`compute_global_loss` passes `cascade_id=None` because it scores every cascade at
    once; it must not be handed cascade 0's knob when another cascade is labelled."""
    config = _config(
        labelled_cascades={1}, num_cascades=2, num_gold_mixing_params=[2]
    )
    assert (config.not_allow_accept_scores(None, temperature=1.0) == 0).all()
    assert (config.not_allow_discard_scores(None, temperature=1.0) == 0).all()
