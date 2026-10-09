"""Operator time and phase time must be measured on the same clock.

Under ``--simulate`` a model call returns a stored response instead of running, so the
wall clock across an operator collapses while ``reasondb.utils.timing.measure`` credits
its spans the stored runtime (see ``SimulatedClock``). If ``run_outside_db`` recorded a
bare ``perf_counter`` delta, the dashboard's operator-level and phase-level time charts
would plot different clocks and disagree by however much runtime simulate skipped.

An ``operator_run`` event carries *two* seconds-valued fields and the dashboard plots
both: ``seconds`` is that elapsed clock, and ``runtime`` is ``ProfilingCost.runtime``,
the cost model's sum over the backend's reported per-call runtimes. They measure
different things - ``seconds`` also covers orchestration around the model calls - but
they describe the same work, so they must stay within a few percent of each other. The
second half of this file checks that agreement above a join; see also
``tests/test_backend_dedup_contract.py``.
"""

import asyncio
import time
from pathlib import Path

import pandas as pd
import pytest

from reasondb.backends.image_qa import VisionModelImageQABackend
from reasondb.backends.simulate_store import SimulateStore
from reasondb.backends.vision_model import LocalVisionModel
from reasondb.database.indentifier import VirtualColumnIdentifier
from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    ProfilingCost,
    RunOutsideResult,
)
from reasondb.utils.logging import FileLogger
from reasondb.utils.timing import SimulatedClock, measure, timing_session

SKIPPED_RUNTIME_S = 5.0


class _SimulatedOperator(PhysicalOperator):
    """Credits the simulated clock exactly as a KV backend does on a --simulate hit."""

    def get_operation_identifier(self):
        return "FakeOp-cr0.0"

    def get_llm_parameters(self):
        class _Interface:
            name = "FakeOp"

        return _Interface()

    @property
    def prefers_run_outside_db(self):
        return True

    async def _run_outside_db(self, **kwargs):
        SimulatedClock.add(SKIPPED_RUNTIME_S)
        return RunOutsideResult(
            output_data=[],
            cost=ProfilingCost(SKIPPED_RUNTIME_S, 0.0),
            input_data=kwargs["input_data"],
        )

    # Abstract surface this test never exercises.
    def implements_logical_operator(self): return None
    def is_tuned(self): return False
    def setup(self, database, logger): pass
    async def profile(self, **kwargs): raise NotImplementedError
    def profile_get_decision_matrix(self, *args): raise NotImplementedError
    async def get_observation(self, **kwargs): raise NotImplementedError
    def get_capabilities(self): return []
    def get_free_form_equivalence_prompt(self, *a, **k): return ""
    def get_hidden_column_type(self, *a, **k): return None
    def get_is_expensive(self): return False
    def get_is_multi_modal(self): return False
    def get_is_potentially_flawed(self): return False
    def is_pipeline_breaker(self): return False
    async def prepare(self, *a, **k): pass
    def shutdown(self, *a, **k): pass
    async def wind_down(self, *a, **k): pass


@pytest.fixture(autouse=True)
def clean_sink():
    yield
    if monitor.get_collector() is not None:
        monitor.get_collector().close()


def test_operator_seconds_are_simulate_corrected_like_phase_spans():
    collector = Collector(jsonl_path=None).install()
    try:
        operator = _SimulatedOperator(quality=1.0, fake_cost=0.0)
        frame = pd.DataFrame({"x": [1, 2, 3]})
        with timing_session() as session, measure("execution"):
            asyncio.run(
                operator.run_outside_db(
                    inputs=[], input_data=[frame], llm_parameters={},
                    database_state=None, observation=None, labels=None, logger=None,
                )
            )
        deadline = time.time() + 3
        while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
            time.sleep(0.01)

        (bucket,) = collector.snapshot_aggregates()["operator_buckets"]
        phase_span = session.durations["execution"]

        # Both clocks carry the runtime simulate skipped...
        assert bucket["seconds"] >= SKIPPED_RUNTIME_S
        assert phase_span >= SKIPPED_RUNTIME_S
        # ...and operator time is a subset of the phase that contains it, which is the
        # relationship the dashboard's "Outside operators" remainder depends on.
        assert bucket["seconds"] <= phase_span + 1e-6
    finally:
        collector.close()


# ── The other clock on the same event: `runtime` ────────────────────────────────

VISION_MODEL = "llava-hf/llama3-llava-next-8b-hf"
RUNTIME_PER_IMAGE = 1.94
IMAGES = ["/img/a.jpg", "/img/b.jpg", "/img/c.jpg"]
QUESTION = "Does this depict a landscape?"


class _StubState:
    """The two things `ImageQaFilter._run_outside_db` asks its database state for."""

    cache_dir = Path("/tmp")

    def get_concrete_column_from_virtual(self, column, avoid_materialization_points=False):
        return None


@pytest.fixture
def vision_store():
    store = SimulateStore()
    for image in IMAGES:
        store.record_vision(
            VISION_MODEL, QUESTION, image,
            response="yes", log_odds=1.0, runtime=RUNTIME_PER_IMAGE, cost=0.0,
            effective_compression_ratio=None, materialized_compression_ratio=None,
            vanilla=False,
        )
    SimulateStore.set_simulate(store)
    yield store
    SimulateStore.set_simulate(None)
    SimulateStore.set_precompute(None)


def test_operator_runtime_and_seconds_agree_above_a_join(vision_store):
    """Runtime is charged per model call above a join, from backend to monitor event.

    A semantic join is rewritten into a cartesian product plus a single-input filter, so
    this filter receives every image once per candidate pair - nine rows over three
    images. The model is called three times either way, and what the operator *reports*
    having spent must reflect that: it reaches `ProfilingCost.runtime`, the optimizer's
    cost-per-tuple and the `execution_cost_runtime` column.
    """
    collector = Collector(jsonl_path=None).install()
    try:
        column = VirtualColumnIdentifier("t.image")
        operator = ImageQaFilter(
            image_qa_backend=VisionModelImageQABackend(LocalVisionModel(VISION_MODEL)),
            quality=1.0,
            fake_cost=0.0,
        )
        cartesian = [left for left in IMAGES for _right in IMAGES]
        frame = pd.DataFrame({"image": cartesian})
        with timing_session() as session, measure("execution"):
            asyncio.run(
                operator.run_outside_db(
                    inputs=[],
                    input_data=[frame],
                    llm_parameters={
                        "question": QUESTION,
                        "image_column": column,
                        "keep_answer": "yes",
                    },
                    database_state=_StubState(),
                    observation=None,
                    labels=None,
                    logger=FileLogger(),
                )
            )
        deadline = time.time() + 3
        while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
            time.sleep(0.01)

        (bucket,) = collector.snapshot_aggregates()["operator_buckets"]
        assert bucket["input_rows"] == len(cartesian) == 9, "the fan-out under test"

        expected = len(IMAGES) * RUNTIME_PER_IMAGE
        assert bucket["runtime"] == pytest.approx(expected), "charged per call, not per row"
        # Both fields describe the same three calls, so they agree - and operator time
        # stays a subset of the phase span, which is what makes the dashboard's
        # "Outside operators" remainder a real quantity rather than a clamped-away
        # negative. `remainderOvershoot` in static/aggregate.js reports it when this
        # stops holding.
        assert bucket["seconds"] == pytest.approx(expected, rel=0.05)
        assert bucket["runtime"] <= session.durations["execution"] + 1e-6
    finally:
        collector.close()
