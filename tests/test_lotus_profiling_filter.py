"""Tests for restricting profiling to the operators an optimizer can actually
deploy at execution time.

LotusOptimizer only ever executes at most two physical operators per step (a
proxy operator plus the gold/silver operator), so the shared `Profiler` need not
profile every candidate operator in `step.operators`. These tests cover:
  - `Profiler.profile_level`/`profile_cascade` honoring an `operator_filter`,
    while always still profiling the gold operator (needed for labels, and to
    advance `intermediate_state` across cascade levels -- not
    `step.chosen_operator_idx`, which is an unrelated "estimated best
    operator" heuristic set by the reasoner/configurator).
  - `LotusCascadeSearchSpace.get_operator_ids()` returning the right operator
    ids with and without a proxy operator.
  - `LotusOptimizer.tune_pipeline` computing its search space before profiling
    and forwarding the resulting silver/proxy-only filter to the profiler.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import torch

from reasondb.optimizer.baselines.lotus_optimizer import (
    LotusCascadeSearchSpace,
    LotusOptimizer,
    LotusSearchSpace,
)
from reasondb.optimizer.profiler import Profiler, ProfilingOutput
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.physical_operator import CostType, ProfilingCost


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class FakeOperator:
    """Minimal duck-typed stand-in for a PhysicalOperator.

    profile_level/profile_cascade only ever call the methods implemented
    here, so there is no need to subclass the real (heavy) PhysicalOperator
    abstract base for these tests.
    """

    def __init__(self, name: str, keep: bool = True):
        self.name = name
        self.keep = keep
        self.profiled = False
        self.observed = False

    def get_operation_identifier(self) -> str:
        return self.name

    def get_llm_parameters(self):
        return SimpleNamespace(name="default")

    def get_default_tuning_parameters(self):
        return {}

    def get_tuning_parameters(self):
        return []

    def scale_cost(self, cost, sample_size, dataset_size):
        return cost

    def profile_get_decision_matrix(self, parameters, profile_output):
        return profile_output

    async def get_observation(self, **kwargs):
        self.observed = True
        return object()

    async def profile(self, **kwargs):
        self.profiled = True
        n = len(kwargs["data_sample"][0])
        df = pd.DataFrame(
            {"value": range(n)}, index=pd.RangeIndex(n, name="row_id")
        )
        m = torch.zeros(n, 1, 3)
        if self.keep:
            m[:, 0, 0] = 1000.0
            m[:, 0, 1] = -1000.0
        else:
            m[:, 0, 0] = -1000.0
            m[:, 0, 1] = 1000.0
        m[:, 0, 2] = -1000.0
        return df, m, ProfilingCost(1.0, 0.0, 0.0)


class FakeStep:
    def __init__(self, operators, inputs, output, chosen_operator_idx):
        self.operators = operators
        self.llm_configurations = {"default": {}}
        self.inputs = inputs
        self.output = output
        self.chosen_operator_idx = chosen_operator_idx

    def get_output_columns(self):
        return []

    @property
    def logical_plan_step(self):
        return None


def _sample() -> ProfilingSampleSpecification:
    return ProfilingSampleSpecification(
        index_column_values=pd.DataFrame(),
        virtual_input_columns=[],
        original_concrete_input_columns=[],
        materialization_point=[],
        index_columns=[],
        sample_fraction=1.0,
    )


def test_profile_level_filter_skips_disallowed_operators():
    """Only operators named in `allowed_operator_ids` (plus the always-on
    gold operator) should actually be profiled."""
    proxy = FakeOperator("proxy")
    other_candidate = FakeOperator("other_candidate")
    gold = FakeOperator("gold")
    step = FakeStep(
        operators=[proxy, other_candidate, gold],
        inputs=["t0"],
        output="t1",
        chosen_operator_idx=2,
    )
    input_data = {"t0": pd.DataFrame({"col": [0, 1, 2, 3]})}

    profiler = Profiler(database=None)
    result = asyncio.run(
        profiler.profile_level(
            input_data=input_data,
            cascade_id=0,
            cascade=[step],
            intermediate_state=None,
            sample=_sample(),
            previous_observations=None,
            level=0,
            logger=DummyLogger(),
            allowed_operator_ids={0},  # only the proxy (operator_id 0)
        )
    )

    assert proxy.profiled is True
    assert other_candidate.profiled is False  # correctly filtered out
    assert gold.profiled is True  # gold always profiled (needed for labels)
    assert set(result.profiler_outputs.keys()) == {0, 2}


def test_profile_level_no_filter_profiles_everything():
    proxy = FakeOperator("proxy")
    other_candidate = FakeOperator("other_candidate")
    gold = FakeOperator("gold")
    step = FakeStep(
        operators=[proxy, other_candidate, gold],
        inputs=["t0"],
        output="t1",
        chosen_operator_idx=2,
    )
    input_data = {"t0": pd.DataFrame({"col": [0, 1, 2, 3]})}

    profiler = Profiler(database=None)
    asyncio.run(
        profiler.profile_level(
            input_data=input_data,
            cascade_id=0,
            cascade=[step],
            intermediate_state=None,
            sample=_sample(),
            previous_observations=None,
            level=0,
            logger=DummyLogger(),
            allowed_operator_ids=None,
        )
    )

    assert proxy.profiled is True
    assert other_candidate.profiled is True
    assert gold.profiled is True


def test_profile_cascade_advances_state_with_gold_not_chosen_operator_idx():
    """profile_cascade advances intermediate_state across cascade levels using
    the gold operator's observation, not `step.chosen_operator_idx` (an
    unrelated heuristic "estimated best operator" index set by the reasoner/
    configurator). Gold is always profiled regardless of any operator_filter,
    so this works without needing to force-profile chosen_operator_idx too --
    and an excluded, non-gold operator that happens to match
    chosen_operator_idx should stay skipped."""
    unwanted = FakeOperator("unwanted")
    chosen_but_excluded = FakeOperator("chosen_but_excluded")
    gold = FakeOperator("gold")
    step1 = FakeStep(
        operators=[unwanted, chosen_but_excluded, gold],
        inputs=["t0"],
        output="t1",
        chosen_operator_idx=1,  # not gold (2), and not requested by the filter below
    )
    input_data = {"t0": pd.DataFrame({"col": [0, 1, 2, 3]})}
    profiling_output = ProfilingOutput()

    profiler = Profiler(database=None)
    asyncio.run(
        profiler.profile_cascade(
            cascade_id=0,
            cascade=[step1],
            input_data=input_data,
            intermediate_state=None,
            sample=_sample(),
            profiling_output=profiling_output,
            previous_observations=None,
            logger=DummyLogger(),
            operator_filter={(0, 0): set()},  # nothing beyond the always-on gold
        )
    )

    assert unwanted.profiled is False
    assert chosen_but_excluded.profiled is False  # not force-profiled
    assert gold.profiled is True


def test_lotus_cascade_search_space_operator_ids_with_proxy():
    css = LotusCascadeSearchSpace(
        silver_operator_id=2,
        silver_operator_name="gold",
        proxy_operator_id=0,
        proxy_operator_name="proxy",
        lower_tuning_param=None,
        upper_tuning_param=None,
    )
    assert css.get_operator_ids() == {0, 2}


def test_lotus_cascade_search_space_operator_ids_without_proxy():
    css = LotusCascadeSearchSpace(
        silver_operator_id=2,
        silver_operator_name="gold",
        proxy_operator_id=None,
        proxy_operator_name=None,
        lower_tuning_param=None,
        upper_tuning_param=None,
    )
    assert css.get_operator_ids() == {2}


def test_lotus_tune_pipeline_profiles_only_silver_and_proxy_operators():
    """End-to-end wiring test: tune_pipeline should compute the search space
    up front and pass a filter restricting profiling to just the silver
    (gold) and proxy operator ids per step -- not every candidate operator."""
    optimizer = LotusOptimizer(cost_type=CostType.FAKE_COST, proxy_operators=["proxy"])
    optimizer.set_database(MagicMock())

    fake_sample = object()
    fake_sampler = MagicMock()
    fake_sampler.sample.return_value = fake_sample
    optimizer.get_sampler = MagicMock(return_value=fake_sampler)

    fake_profiler = MagicMock()
    fake_profiling_output = ProfilingOutput()
    fake_profiler.profile = AsyncMock(return_value=fake_profiling_output)
    optimizer.get_profiler = MagicMock(return_value=fake_profiler)

    search_space = LotusSearchSpace(
        {
            (0, 0): LotusCascadeSearchSpace(
                silver_operator_id=2,
                silver_operator_name="gold",
                proxy_operator_id=0,
                proxy_operator_name="proxy",
                lower_tuning_param=None,
                upper_tuning_param=None,
            ),
            (1, 0): LotusCascadeSearchSpace(
                silver_operator_id=1,
                silver_operator_name="gold2",
                proxy_operator_id=None,
                proxy_operator_name=None,
                lower_tuning_param=None,
                upper_tuning_param=None,
            ),
        }
    )
    optimizer.get_lotus_search_space = MagicMock(return_value=search_space)
    optimizer.lotus_optimize = MagicMock(return_value={})
    optimizer.get_tuned_pipeline = MagicMock(return_value=MagicMock())

    pipeline = MagicMock()
    pipeline.get_virtual_input_columns.return_value = ["some_column"]
    intermediate_state = MagicMock()

    asyncio.run(
        optimizer.tune_pipeline(
            pipeline=pipeline,
            intermediate_state=intermediate_state,
            guarantees=[],
            logger=DummyLogger(),
        )
    )

    optimizer.get_lotus_search_space.assert_called_once()
    assert optimizer.get_lotus_search_space.call_args.args == (pipeline,)
    fake_profiler.profile.assert_awaited_once()
    call_kwargs = fake_profiler.profile.await_args.kwargs
    assert call_kwargs["operator_filter"] == {
        (0, 0): {0, 2},  # proxy + gold
        (1, 0): {1},  # gold only, no proxy for this step
    }
