"""``ReorderOnlyOptimizer``: the gold plan, profiled, DP-ordered, and nothing else.

The pieces it is made of are already covered where they live -
``Optimizer.fallback_pipeline`` picks the operator, ``Profiler`` measures it, ``DPReorderer``
orders it. What is only true *here* is how they are wired together, and each of these tests
pins one property of that wiring that a plausible refactor would break:

- the plan is the gold one, not a search's pick;
- the profiling pass is paid for and *reported* (``tuning_runtime_s`` is read off the
  returned cost, so a zero there would hide the arm's whole price);
- only the operator that runs is profiled;
- it is not a ``LabelOptimizer``, because ``Executor`` derives ``role="label"`` from that
  type and the dashboard then hides the arm.

A real ``tune_pipeline`` needs a Database, a Profiler and a TuningPipeline, so the seams are
exercised against stubs; the end-to-end path is the ``reorder_only`` producer's own run.
"""

import asyncio

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from reasondb.optimizer.label_optimizer import LabelOptimizer  # noqa: E402
from reasondb.optimizer.reorder_only_optimizer import ReorderOnlyOptimizer  # noqa: E402
from reasondb.optimizer.reorderer import DPReorderer  # noqa: E402
from reasondb.query_plan.physical_operator import CostType, ProfilingCost  # noqa: E402


class _FakeTuningParameter:
    def __init__(self, name, default):
        self.name = name
        self.default = default


class _FakeOperator:
    """Enough of a ``PhysicalOperator`` for the selectivity helper."""

    def __init__(self, keeps, parameters=()):
        # (num_rows, num_jobs, 3) as `profile_get_decision_matrix` declares. KEEP is
        # Decision.KEEP == 0, so a kept row is one-hot on the first column.
        self._matrix = torch.zeros(len(keeps), 1, 3)
        for row, keep in enumerate(keeps):
            self._matrix[row, 0, 0 if keep else 1] = 1.0
        self._parameters = [_FakeTuningParameter(*p) for p in parameters]

    def get_tuning_parameters(self):
        return self._parameters

    def profile_get_decision_matrix(self, parameters, profile_output):
        # Read one parameter so a test can assert the defaults are what gets asked for.
        for parameter in self._parameters:
            parameters(parameter.name)
        return self._matrix


class _FakeStep:
    def __init__(self, operators, last_executable):
        self.operators = operators
        self._last_executable = last_executable

    def get_last_executable_operator_index(self):
        return self._last_executable


class _FakePipeline:
    def __init__(self, cascades):
        self.steps_in_parallel = cascades


class _FakeProfilingOutput:
    def __init__(self, keeps_by_key, index):
        import pandas as pd

        self.merged_output_tuples = {
            key: pd.DataFrame(index=list(index)) for key in {k[:2] for k in keeps_by_key}
        }
        self._profiles = {key: torch.zeros(1) for key in keeps_by_key}
        self._masks = {
            key: torch.ones(len(index), dtype=torch.bool) for key in keeps_by_key
        }

    def get(self, cascade_id, level, operator_id):
        return self._profiles[cascade_id, level, operator_id]

    def get_output_mask(self, cascade_id, level, operator_id):
        return self._masks[cascade_id, level, operator_id]


def _optimizer():
    return ReorderOnlyOptimizer(CostType.RUNTIME, sample_size=25)


# ── What it is, and is not ───────────────────────────────────────────────────


def test_it_is_not_a_label_optimizer():
    """``Executor`` derives ``role="label"`` from that type and the monitor hides label
    passes, so an arm built on it would be absent from every dashboard panel while still
    appearing in the merged CSV."""
    assert not isinstance(_optimizer(), LabelOptimizer)


def test_it_orders_with_the_dynamic_program():
    assert isinstance(_optimizer().get_reorderer(), DPReorderer)


def test_it_draws_its_budget_in_one_round():
    """No second round to size: the plan is fixed before profiling starts, so no
    measurement can change it."""
    sampler = _optimizer().get_sampler()
    assert sampler.sample_size == 25
    assert sampler.batch_size is None


# ── Only the operator that runs is profiled ──────────────────────────────────


def test_only_the_last_executable_operator_is_profiled():
    """The cheap candidates of a step are not measured, because nothing may pick them -
    profiling them would charge this arm for a search it does not do. The *last
    executable* one rather than the last: a label operator is a human."""
    pipeline = _FakePipeline(
        [
            [_FakeStep([object(), object(), object()], last_executable=1)],
            [_FakeStep([object(), object()], last_executable=0)],
        ]
    )

    assert ReorderOnlyOptimizer._gold_only_filter(pipeline) == {
        (0, 0): {1},
        (1, 0): {0},
    }


# ── Selectivity, measured rather than assumed ────────────────────────────────


def test_selectivity_is_the_kept_share_of_distinct_sampled_tuples():
    operator = _FakeOperator(keeps=[True, False, True, False])
    pipeline = _FakePipeline([[_FakeStep([operator], last_executable=0)]])
    profiling_output = _FakeProfilingOutput({(0, 0, 0): None}, index=[10, 11, 12, 13])

    selectivities = _optimizer()._selectivities(pipeline, profiling_output, logger=None)

    assert selectivities.get_inter_selectivity(0, 0, 0) == pytest.approx(0.5)
    # Nothing to defer to: one tier, so no share of the input falls through to a next one.
    assert selectivities.get_intra_selectivity(0, 0, 0) == 0.0


def test_duplicated_tuples_count_once():
    """A multi-modal cell is repeated across rows, so counting rows would report one cell
    several times - the same distinct-index denominator ``ParetoCascades`` uses."""
    operator = _FakeOperator(keeps=[True, True, False, False])
    pipeline = _FakePipeline([[_FakeStep([operator], last_executable=0)]])
    profiling_output = _FakeProfilingOutput({(0, 0, 0): None}, index=[7, 7, 8, 8])

    selectivities = _optimizer()._selectivities(pipeline, profiling_output, logger=None)

    assert selectivities.get_inter_selectivity(0, 0, 0) == pytest.approx(0.5)


def test_the_decision_matrix_is_read_at_the_declared_defaults():
    """Nothing here tunes, so the defaults are the only setting that may be asked for -
    a threshold read from anywhere else would be a tuned plan wearing this arm's name."""
    asked = []

    class _RecordingOperator(_FakeOperator):
        def profile_get_decision_matrix(self, parameters, profile_output):
            asked.append(float(parameters("threshold")[0]))
            return self._matrix

    operator = _RecordingOperator(keeps=[True], parameters=[("threshold", 0.42)])
    pipeline = _FakePipeline([[_FakeStep([operator], last_executable=0)]])
    profiling_output = _FakeProfilingOutput({(0, 0, 0): None}, index=[1])

    _optimizer()._selectivities(pipeline, profiling_output, logger=None)

    assert asked == pytest.approx([0.42])  # float32, via torch.Tensor


def test_an_unprofiled_step_is_ordered_as_if_it_filtered_nothing():
    """It must not look attractive: with no measurement behind it, the DP should push it
    last rather than early."""
    warnings = []

    class _Logger:
        def warning(self, _module, message):
            warnings.append(message)

    operator = _FakeOperator(keeps=[True])
    pipeline = _FakePipeline([[_FakeStep([operator], last_executable=0)]])
    profiling_output = _FakeProfilingOutput({}, index=[1])

    selectivities = _optimizer()._selectivities(pipeline, profiling_output, _Logger())

    assert selectivities.get_inter_selectivity(0, 0, 0) == pytest.approx(1.0)
    assert warnings, "a step nobody measured should say so"


# ── What it reports ──────────────────────────────────────────────────────────


def test_the_profiling_pass_is_reported_as_the_tuning_cost():
    """``tuning_runtime_s`` is read off this return value, not off the timing spans. A
    zero here - what ``LabelOptimizer`` returns - would hide the profiling this arm pays
    for in the one comparison the experiment exists to make.
    """
    optimizer = _optimizer()
    optimizer.set_database(object())
    recorded = ProfilingCost(runtime=12.5, monetary_cost=0.25)

    calls = {}

    class _StubProfilingOutput:
        total_cost = recorded
        n_labels_requested = 3

    async def _run():
        # Every collaborator is stubbed: this test is about what tune_pipeline returns,
        # and the pieces it calls are covered above and in their own modules.
        optimizer.get_sampler = lambda: _StubSampler()
        optimizer.get_profiler = lambda: _StubProfiler()
        optimizer._selectivities = lambda *a, **k: None
        optimizer.get_reorderer = lambda: _StubReorderer()

        async def _fallback(**kwargs):
            calls["fallback"] = kwargs
            return ("tuned", [set()], [1.0], "selectivities")

        optimizer.fallback_pipeline = _fallback
        return await optimizer.tune_pipeline(
            pipeline=_StubTuningPipeline(),
            intermediate_state=_StubIntermediateState(),
            guarantees=[],
            logger=None,
        )

    class _StubSampler:
        def sample(self, **kwargs):
            return "sample"

    class _StubProfiler:
        async def profile(self, **kwargs):
            calls["profiled"] = kwargs["operator_filter"]
            return _StubProfilingOutput()

    class _StubReorderer:
        def reorder(self, **kwargs):
            calls["reordered"] = kwargs
            return "ordered"

    class _StubTuningPipeline:
        steps_in_parallel = [[_FakeStep([object()], last_executable=0)]]
        dependencies = [set()]

        def get_virtual_input_columns(self):
            return ["column"]

    class _StubMaterializationPoint:
        identifier = "table"

        def get_duplication_factor(self):
            return {}

        def estimated_len(self):
            return 1000

    class _StubIntermediateState:
        materialization_points = [_StubMaterializationPoint()]
        database = object()

    pipeline, cost = asyncio.run(_run())

    assert pipeline == "ordered"
    assert cost is recorded, "the profiling pass must be visible in tuning_runtime_s"
    assert optimizer.last_n_labels_requested == 3
    # The DP is handed the costs and selectivities that came out of the profiled plan,
    # in the plan's own step order.
    assert calls["reordered"]["per_operator_and_sample_costs"] == [1.0]
    assert calls["reordered"]["selectivities"] == "selectivities"
    assert calls["profiled"] == {(0, 0): {0}}
