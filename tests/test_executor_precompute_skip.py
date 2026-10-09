"""`REASONDB_PRECOMPUTE_SKIP_MODALITIES` support in `Executor._precompute_pipeline`.

Ecommerce-style benchmarks need every text AND image KV cache server up at once to
precompute, which doesn't always fit on one machine's GPUs. Setting
REASONDB_PRECOMPUTE_SKIP_MODALITIES lets a precompute pass run with only some
servers up: operators whose `get_modality()` is in the skip set must not have
`run_outside_db` called (the server isn't there to answer), but must also *not*
be marked precomputed, so a later pass with that server up still fills them in.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from reasondb.backends.simulate_store import SimulateStore
from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.executor import Executor
from reasondb.utils import precompute_modalities


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class FakeLogicalStep:
    def __init__(self, expression="expr"):
        self.expression = expression

    def get_labels(self):
        return None


class FakeOperator:
    """Minimal duck-typed stand-in for a PhysicalOperator, mirroring the one in
    test_lotus_profiling_filter.py: `_precompute_pipeline` only ever calls the
    methods implemented here.
    """

    def __init__(self, name: str, modality):
        self._name = name
        self._modality = modality
        self.prefers_run_outside_db = True
        self.run_outside_db = AsyncMock()

    def get_operation_identifier(self) -> str:
        return self._name

    def get_llm_parameters(self):
        return SimpleNamespace(name=self._name)

    def requires_data_sample(self) -> bool:
        return False

    def get_modality(self):
        return self._modality

    async def get_observation(self, **kwargs):
        return SimpleNamespace()


class FakeStep:
    def __init__(self, operators, inputs=(VirtualTableIdentifier("t0"),)):
        self.operators = operators
        self.inputs = list(inputs)
        self.observations = [None] * len(operators)
        self.llm_configurations = {op.get_operation_identifier(): {} for op in operators}
        self.output = "out"
        self.logical_plan_step = FakeLogicalStep()

    def get_output_columns(self):
        return []

    async def get_input_sample(self, **kwargs):
        return [pd.DataFrame({"col": [1, 2, 3]})]


class FakePipeline:
    def __init__(self, cascades):
        self.steps_in_parallel = cascades


@pytest.fixture(autouse=True)
def _reset_simulate_store():
    SimulateStore.set_precompute(None)
    yield
    SimulateStore.set_precompute(None)


def _run_precompute_pipeline(pipeline):
    executor = Executor.__new__(Executor)
    asyncio.run(
        executor._precompute_pipeline(
            pipeline=pipeline,
            intermediate_state=SimpleNamespace(),
            logger=DummyLogger(),
        )
    )


def test_operator_with_skipped_modality_is_not_run(monkeypatch):
    monkeypatch.setattr(
        precompute_modalities, "PRECOMPUTE_SKIP_MODALITIES", frozenset({"image"})
    )

    text_op = FakeOperator("text_op", "text")
    image_op = FakeOperator("image_op", "image")
    pipeline = FakePipeline([[FakeStep([text_op, image_op])]])

    _run_precompute_pipeline(pipeline)

    text_op.run_outside_db.assert_awaited_once()
    image_op.run_outside_db.assert_not_awaited()


def test_skipped_operator_is_not_marked_precomputed_so_it_can_resume(monkeypatch):
    """A skipped operator must stay eligible for a later precompute pass (with its
    server up) to fill in -- unlike a genuinely completed one, it must not be
    recorded in the SimulateStore's precomputed-ops markers.
    """
    monkeypatch.setattr(
        precompute_modalities, "PRECOMPUTE_SKIP_MODALITIES", frozenset({"image"})
    )
    store = SimulateStore()
    SimulateStore.set_precompute(store)

    text_op = FakeOperator("text_op", "text")
    image_op = FakeOperator("image_op", "image")
    pipeline = FakePipeline([[FakeStep([text_op, image_op])]])

    _run_precompute_pipeline(pipeline)

    assert not any("image_op" in key for key in store._precomputed_ops)
    assert any("text_op" in key for key in store._precomputed_ops)


def test_no_skip_modalities_runs_every_operator(monkeypatch):
    monkeypatch.setattr(precompute_modalities, "PRECOMPUTE_SKIP_MODALITIES", frozenset())

    text_op = FakeOperator("text_op", "text")
    image_op = FakeOperator("image_op", "image")
    audio_op = FakeOperator("audio_op", "audio")
    pipeline = FakePipeline([[FakeStep([text_op, image_op, audio_op])]])

    _run_precompute_pipeline(pipeline)

    text_op.run_outside_db.assert_awaited_once()
    image_op.run_outside_db.assert_awaited_once()
    audio_op.run_outside_db.assert_awaited_once()


def test_already_precomputed_operator_is_skipped_regardless_of_modality(monkeypatch):
    """Resume behavior composes with the modality gate: an operator already marked
    precomputed is skipped even if its modality
    isn't in the skip set.
    """
    monkeypatch.setattr(precompute_modalities, "PRECOMPUTE_SKIP_MODALITIES", frozenset())
    store = SimulateStore()
    SimulateStore.set_precompute(store)

    text_op = FakeOperator("text_op", "text")
    step = FakeStep([text_op])
    canonical_expression = step.logical_plan_step.expression
    precompute_key = f"{text_op.get_operation_identifier()}|{canonical_expression}|_T0"
    store.mark_op_precomputed(precompute_key)

    _run_precompute_pipeline(FakePipeline([[step]]))

    text_op.run_outside_db.assert_not_awaited()
