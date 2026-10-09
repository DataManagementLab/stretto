"""`REASONDB_PRECOMPUTE_SKIP_MODALITIES` support in `PlanConfigurator.prepare`.

Unlike `run_outside_db`, an operator's KV cache materialization in `prepare()`
(e.g. `TextQaFilter.prepare` posting to its server's `/prepare_caches`) isn't
gated by anything else - `KvTextQABackend.prepare` only skips its network call
under `--simulate`, not `--precompute`. So a mixed-modality benchmark (e.g.
ecommerce, which registers both text and image operators) would still crash
`PlanConfigurator.prepare` reaching for a server the modality skip was
supposed to let stay down, unless `prepare()` itself also honors the skip set.
This mirrors test_executor_precompute_skip.py's coverage of the equivalent gate
in `Executor._precompute_pipeline`.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from reasondb.backends.simulate_store import SimulateStore
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.utils import precompute_modalities


class FakeOperator:
    def __init__(self, name: str, modality):
        self._name = name
        self._modality = modality
        self.prepare = AsyncMock()

    def get_operation_identifier(self) -> str:
        return self._name

    def get_modality(self):
        return self._modality


class FakeToolbox:
    def __init__(self, operators):
        self._operators = operators

    def __iter__(self):
        return iter(self._operators)


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def _reset_simulate_store():
    SimulateStore.set_precompute(None)
    yield
    SimulateStore.set_precompute(None)


def _configurator(operators):
    return PlanConfigurator(llm=SimpleNamespace(prepare=AsyncMock()), physical_operators=FakeToolbox(operators))


def _run_prepare(configurator):
    asyncio.run(configurator.prepare(database=SimpleNamespace(), logger=DummyLogger()))


class TestPrepareSkipsConfiguredModalitiesDuringPrecompute:
    """Skipping only applies while a precompute store is active."""

    def test_skipped_modality_operator_is_not_prepared(self, monkeypatch):
        monkeypatch.setattr(
            precompute_modalities, "PRECOMPUTE_SKIP_MODALITIES", frozenset({"image"})
        )
        SimulateStore.set_precompute(SimulateStore())

        text_op = FakeOperator("text_op", "text")
        image_op = FakeOperator("image_op", "image")
        _run_prepare(_configurator([text_op, image_op]))

        text_op.prepare.assert_awaited_once()
        image_op.prepare.assert_not_awaited()

    def test_no_modality_operator_is_always_prepared(self, monkeypatch):
        monkeypatch.setattr(
            precompute_modalities, "PRECOMPUTE_SKIP_MODALITIES", frozenset({"text", "image", "audio"})
        )
        SimulateStore.set_precompute(SimulateStore())

        traditional_op = FakeOperator("traditional_op", None)
        _run_prepare(_configurator([traditional_op]))

        traditional_op.prepare.assert_awaited_once()

    def test_skip_set_ignored_outside_precompute(self, monkeypatch):
        """Regular query execution and --simulate must not be affected: the skip
        set is only meaningful while SimulateStore.get_precompute() is active.
        """
        monkeypatch.setattr(
            precompute_modalities, "PRECOMPUTE_SKIP_MODALITIES", frozenset({"image"})
        )
        SimulateStore.set_precompute(None)

        image_op = FakeOperator("image_op", "image")
        _run_prepare(_configurator([image_op]))

        image_op.prepare.assert_awaited_once()
