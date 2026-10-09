"""Tests for attaching a human-label operator to a step's candidate list.

`PlanConfigurator` appends a label-only operator to any step whose logical operator
carries a `LabelsDefinition`, when `use_human_labels` is on. Because "gold" is positional
-- `Profiler.profile_level` reads `step.operators[-1]`, and
`UnoptimizedPhysicalPlanStep.__init__` sorts candidates ascending by quality -- giving it
`LABEL_OPERATOR_QUALITY` is what makes it the label source, and what pushes the best model
to the second-to-last slot where it gets a learnable pick score.
"""

import pytest

try:
    from reasondb.optimizer.configurator import _label_operator_for
    from reasondb.query_plan.logical_plan import (
        LogicalExtract,
        LogicalFilter,
        LogicalJoin,
        LogicalTransform,
    )
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
    """A stand-in candidate. Only `quality`, `is_label_only` and the interface name are
    read by the code under test."""

    def __init__(self, name, quality, is_label_only=False):
        self.quality = quality
        self.is_label_only = is_label_only
        self._name = name

    def get_llm_parameters(self):
        return _Interface(self._name)

    def get_operation_identifier(self):
        return self._name

    def __repr__(self):
        return f"_FakeOperator({self._name}, q={self.quality})"


class _FakeLogicalStep:
    validated = True
    inputs = ()
    output = None
    expression = "x"


def _step(qualities=(1.0, 5.0, 8.0)):
    operators = [_FakeOperator(f"op{i}", q) for i, q in enumerate(qualities)]
    return UnoptimizedPhysicalPlanStep(
        logical_plan_step=_FakeLogicalStep(),
        operators=PhysicalOperatorsNoPseudos(operators),
        llm_configurations={f"op{i}": {} for i in range(len(qualities))},
        estimated_best_operator_idx=0,
    )


def _label_operator():
    return _FakeOperator("Labeler", LABEL_OPERATOR_QUALITY, is_label_only=True)


# --- which logical operators have a label source -------------------------------


def test_filters_and_extracts_have_a_label_source():
    from reasondb.operators.perfect_operators.perfect_extract import PerfectExtract
    from reasondb.operators.perfect_operators.perfect_filter import PerfectFilter

    # LogicalFilter covers join predicates too -- a join predicate is a filter over two
    # input tables.
    assert _label_operator_for(LogicalFilter) is PerfectFilter
    assert _label_operator_for(LogicalExtract) is PerfectExtract


def test_other_logical_operators_have_none():
    """Steps with no label source keep model-derived labels, so a plan can mix the two."""
    assert _label_operator_for(LogicalJoin) is None
    assert _label_operator_for(LogicalTransform) is None


# --- attachment ----------------------------------------------------------------


def test_attached_operator_sorts_last_and_is_reported_as_the_label_source():
    step = _step()
    step.attach_label_operator(_label_operator(), {"__expression__": "x"})

    assert step.operators[len(step.operators) - 1].is_label_only
    assert step.get_label_operator_index() == len(step.operators) - 1
    assert len(step.operators) == 4


def test_best_model_moves_to_the_last_executable_slot():
    """The highest-quality *model* must not be the last candidate, so that it keeps a
    learnable pick score and tunable parameters."""
    step = _step(qualities=(1.0, 5.0, 8.0))
    step.attach_label_operator(_label_operator(), {"__expression__": "x"})

    last_executable = step.get_last_executable_operator_index()
    assert last_executable == len(step.operators) - 2
    assert step.operators[last_executable].quality == 8.0
    assert not step.operators[last_executable].is_label_only


def test_without_a_label_operator_the_last_slot_is_still_executable():
    step = _step()
    assert step.get_label_operator_index() is None
    assert step.get_last_executable_operator_index() == len(step.operators) - 1


def test_parallel_lists_stay_aligned():
    """`tuning_parameters` and `observations` are indexed by operator id throughout the
    profiler and optimizer."""
    step = _step()
    step.attach_label_operator(_label_operator(), {"__expression__": "x"})

    assert len(step.tuning_parameters) == len(step.operators)
    assert len(step.observations) == len(step.operators)
    assert step.llm_configurations["Labeler"] == {"__expression__": "x"}


def test_attaching_twice_is_rejected():
    """`rename_inputs` reconstructs the step from the same operator collection, so a
    second attach would mean the same object grew two label sources."""
    step = _step()
    step.attach_label_operator(_label_operator(), {"__expression__": "x"})
    with pytest.raises(AssertionError, match="already has a label operator"):
        step.attach_label_operator(_label_operator(), {"__expression__": "x"})


def test_rename_inputs_does_not_duplicate_the_label_operator():
    step = _step()
    step.attach_label_operator(_label_operator(), {"__expression__": "x"})
    renamed = step.rename_inputs({}, reset_validation=False)

    label_ops = [op for op in renamed.operators if op.is_label_only]
    assert len(label_ops) == 1
    assert renamed.get_label_operator_index() == len(renamed.operators) - 1


def test_a_non_label_operator_is_rejected():
    step = _step()
    with pytest.raises(AssertionError, match="not a label operator"):
        step.attach_label_operator(
            _FakeOperator("plain", LABEL_OPERATOR_QUALITY), {"__expression__": "x"}
        )


# --- the configurator decides when to attach ------------------------------------


class _FakeOptions:
    def __init__(self, step):
        self._step = step

    def parse(self, logical_step, response, database_state):
        return self._step


def _configure(logical_step, use_human_labels):
    """Drive `map_logical_to_physical` with `parse` and `validate_step` stubbed -- both
    need a live `IntermediateState`, and neither is what this asserts."""
    from reasondb.optimizer.configurator import PlanConfigurator

    step = _step()
    step.validate_step = lambda *_a, **_k: None
    configurator = PlanConfigurator(
        llm=None, physical_operators=None, use_human_labels=use_human_labels
    )
    configurator.map_logical_to_physical(
        logical_step=logical_step,
        options=_FakeOptions(step),
        database_state=None,
        response="[]",
    )
    return step


class _LabelledFilter(LogicalFilter):
    """A `LogicalFilter` carrying a `LabelsDefinition`, as the hand-written benchmark
    plans do."""

    def __init__(self, labels):
        self._labels = labels
        self.inputs = ()
        self.expression = "{t.x} is red"

    def get_labels(self):
        return self._labels


def test_configurator_attaches_only_when_the_flag_and_the_labels_are_both_present():
    labels = object()
    assert _configure(_LabelledFilter(labels), use_human_labels=True).get_label_operator_index() is not None
    assert _configure(_LabelledFilter(labels), use_human_labels=False).get_label_operator_index() is None
    # A step with no ground-truth file keeps model-derived labels even with the flag on,
    # so a plan can mix human-labelled and model-labelled steps.
    assert _configure(_LabelledFilter(None), use_human_labels=True).get_label_operator_index() is None


class _TraditionalJoin(LogicalJoin):
    """The equi-join every rotowire plan carries: `{a.game_id} equals {b.game_id}`."""

    def __init__(self):
        self.inputs = ()
        self.expression = "{joined_players_games.game_id} equals {reports.game_id}"

    def get_labels(self):
        return None


def test_a_step_with_no_label_source_is_not_reported_as_unlabelled(monkeypatch):
    """`_label_operator_for` is asked before the labels are.

    An exact SQL equi-join has nothing a `LabelsDefinition` could annotate, so the
    "keeps model-derived labels" warning would name a problem that does not exist, on a
    step that runs no model at all. A *semantic* step without ground truth is the case
    that warning is for, and still reports."""
    from reasondb.optimizer import configurator as configurator_module

    reported = []
    monkeypatch.setattr(
        configurator_module._monitor,
        "record_error",
        lambda kind, message: reported.append((kind, message)),
    )

    _configure(_TraditionalJoin(), use_human_labels=True)
    assert reported == []

    _configure(_LabelledFilter(None), use_human_labels=True)
    assert [kind for kind, _ in reported] == ["human_labels"]


def test_human_labels_is_rejected_for_a_benchmark_without_ground_truth():
    """A `RandomBenchmark` samples its queries from an operator pool, so its predicates
    are generated and nobody has labelled them. Without ground truth no step carries a
    `LabelsDefinition`, so no label operator is attached and the run behaves exactly like
    a normal one -- a silent no-op indistinguishable in the results from a real
    human-labels run. Fail at enumeration instead, before any GPU time is spent."""
    import argparse

    import pytest as _pytest

    from reasondb.coordinator.producers import run_benchmark as rbp

    class _NoGroundTruth:
        has_ground_truth = False

        @classmethod
        def name(cls):
            return "fake_bench"

        @classmethod
        def load(cls, _split):
            return cls()

        def query_count(self, _debug_query):
            return 1

    args = argparse.Namespace(
        benchmarks=["fake_bench"],
        split="dev",
        use_indexes=False,
        human_labels=True,
        precision_guarantees=[0.7],
        recall_guarantees=[0.7],
        all_guarantee_combinations=False,
        cost_type="runtime",
        debug_query=None,
        simulate=None,
        precompute=None,
        labels=["silver"],
        select_executors=[],
        # The two --human-labels guards are independent, and the lotus one is checked
        # first because it needs no benchmark loaded. Skip it so this test exercises the
        # ground-truth guard rather than whichever happens to come first.
        skip_executors=["lotus"],
    )
    original = dict(rbp.ALL_BENCHMARKS)
    rbp.ALL_BENCHMARKS.clear()
    rbp.ALL_BENCHMARKS["fake_bench"] = _NoGroundTruth
    try:
        with _pytest.raises(AssertionError, match="per-tuple ground truth"):
            rbp.enumerate_jobs("t1", __import__("pathlib").Path("/tmp"), args)
    finally:
        rbp.ALL_BENCHMARKS.clear()
        rbp.ALL_BENCHMARKS.update(original)


def test_human_labels_is_rejected_together_with_lotus():
    """Lotus's guarantee is defined relative to its own silver operator, so it has no
    meaning against a human reference -- and dropping it silently would leave a run that
    enumerates four approaches looking exactly like one where Lotus produced no rows."""
    import argparse

    import pytest as _pytest

    from reasondb.coordinator.producers import run_benchmark as rbp

    def _args(**overrides):
        base = dict(
            benchmarks=["artwork"],
            split="dev",
            use_indexes=False,
            human_labels=True,
            precision_guarantees=[0.7],
            recall_guarantees=[0.7],
            all_guarantee_combinations=False,
            cost_type="runtime",
            debug_query=None,
            simulate=None,
            precompute=None,
            labels=["silver"],
            select_executors=["optim_global", "lotus"],
            skip_executors=[],
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    with _pytest.raises(AssertionError, match="cannot be combined with the 'lotus'"):
        rbp.enumerate_jobs("t1", __import__("pathlib").Path("/tmp"), _args())

    # ... and only for that combination: neither flag alone is affected.
    for ok in (_args(human_labels=False), _args(select_executors=["optim_global"])):
        assert not (
            getattr(ok, "human_labels") and "lotus" in rbp._selected_executors(ok)
        )


def test_attached_operator_carries_the_expression_config_it_needs():
    """`PerfectFilter.get_observation` reads `llm_parameters['__expression__']`, and
    `Profiler.profile_level` looks the config up by the operator's interface name."""
    from reasondb.query_plan.llm_parameters import LlmParameterTemplate

    step = _configure(_LabelledFilter(object()), use_human_labels=True)
    label_op = step.operators[step.get_label_operator_index()]
    config = step.llm_configurations[label_op.get_llm_parameters().name]
    assert isinstance(config["__expression__"], LlmParameterTemplate)


def test_a_label_operator_that_does_not_outrank_the_candidates_is_rejected():
    """Otherwise the quality sort would not place it last and the profiler would silently
    derive labels from the wrong operator."""
    step = _step(qualities=(1.0, 5.0, 8.0))
    with pytest.raises(AssertionError, match="must outrank"):
        step.attach_label_operator(
            _FakeOperator("Labeler", 8.0, is_label_only=True), {"__expression__": "x"}
        )
