"""Compression-ratio metadata on `operator_run` events must come from duck-typed
backend attributes, not from parsing `get_operation_identifier()` strings.

That identifier is built ad hoc per operator/backend pair (see
`reasondb/backends/text_qa.py:KvTextQABackend.model_id`) and its `-mat{x}` suffix is
*conditional* (only present when materialized != effective), so regex-parsing it back
apart is fragile. `_extract_cr_info` instead reads the same instance attributes the
backends already expose (`effective_compression_ratio`, `materialized_compression_ratio`,
`vanilla`, `_model_id`), which is exact and doesn't care about string formatting.

The three CR-bearing backend classes (`KvTextQABackend`, `KvVisionModel`, `KvAudioModel`)
share this attribute shape by convention only - there's no common base class - and
non-KV backends either omit the attribute (`LLMTextQABackend`, `LocalAudioModel`, raising
on plain access) or declare it `None` at the class level (`LocalVisionModel`). Both cases
must resolve to "no CR info" here, not a crash and not a misleading `None`/`0.0` in the
output dict.
"""

import asyncio

import pandas as pd
import pytest

from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    ProfilingCost,
    RunOutsideResult,
    _extract_cr_info,
)


class _Backend:
    def __init__(self, model_id, effective, materialized=None, vanilla=False):
        self._model_id = model_id
        self.effective_compression_ratio = effective
        self.materialized_compression_ratio = (
            materialized if materialized is not None else effective
        )
        self.vanilla = vanilla


class _NoCrBackend:
    """Mimics LocalVisionModel: the attribute exists but is None at the class level."""

    effective_compression_ratio = None
    materialized_compression_ratio = None
    vanilla = False


class _Op:
    """A bare stand-in for a PhysicalOperator: only the backend attribute matters."""

    def __init__(self, **backends):
        for name, value in backends.items():
            setattr(self, name, value)


def test_kv_text_backend_yields_full_cr_info():
    op = _Op(text_qa_backend=_Backend("meta-llama/Llama-3.1-8B-Instruct", 0.5))
    info = _extract_cr_info(op)
    assert info == {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "effective_compression_ratio": 0.5,
        "materialized_compression_ratio": 0.5,
        "vanilla": False,
        "keep_in_memory": False,
    }


def test_kv_image_backend_is_checked_too():
    op = _Op(image_qa_backend=_Backend("llava-hf/llama3-llava-next-8b-hf", 0.9))
    info = _extract_cr_info(op)
    assert info["model_name"] == "llava-hf/llama3-llava-next-8b-hf"
    assert info["effective_compression_ratio"] == 0.9


def test_kv_audio_backend_is_checked_too():
    op = _Op(audio_qa_backend=_Backend("Qwen/Qwen2-Audio-7B-Instruct", 0.3))
    info = _extract_cr_info(op)
    assert info["model_name"] == "Qwen/Qwen2-Audio-7B-Instruct"


class _WrapperBackend:
    """Mimics `VisionModelImageQABackend`/`AudioModelAudioQABackend`: it delegates to the
    model underneath and carries no CR attributes of its own."""

    def __init__(self, attr_name, model):
        setattr(self, attr_name, model)


def test_wrapped_vision_model_cr_info_is_found_one_level_down():
    """The production shape: `ImageQaFilter.image_qa_backend` is the *wrapper*, and the
    ratio lives on the `KvVisionModel` it delegates to. Reading only the wrapper would
    make every image operator report no model and no ratio."""
    op = _Op(
        image_qa_backend=_WrapperBackend(
            "vision_model", _Backend("llava-hf/llava-next-72b-hf", 0.99)
        )
    )
    info = _extract_cr_info(op)
    assert info["model_name"] == "llava-hf/llava-next-72b-hf"
    assert info["effective_compression_ratio"] == 0.99


def test_wrapped_audio_model_cr_info_is_found_one_level_down():
    op = _Op(
        audio_qa_backend=_WrapperBackend(
            "audio_model", _Backend("Qwen/Qwen2-Audio-7B-Instruct", 0.3)
        )
    )
    assert _extract_cr_info(op)["model_name"] == "Qwen/Qwen2-Audio-7B-Instruct"


def test_wrapped_non_kv_vision_model_still_yields_empty_dict():
    """LocalVisionModel behind the same wrapper: the attribute exists but is None, so
    the unwrap must not turn "no compression info" into a misleading entry."""
    op = _Op(image_qa_backend=_WrapperBackend("vision_model", _NoCrBackend()))
    assert _extract_cr_info(op) == {}


def test_differing_materialized_ratio_is_preserved():
    op = _Op(text_qa_backend=_Backend("m", effective=0.9, materialized=0.3))
    info = _extract_cr_info(op)
    assert info["effective_compression_ratio"] == 0.9
    assert info["materialized_compression_ratio"] == 0.3


def test_vanilla_flag_propagates():
    op = _Op(text_qa_backend=_Backend("m", effective=0.0, vanilla=True))
    info = _extract_cr_info(op)
    assert info["vanilla"] is True


def test_operator_with_no_backend_attribute_yields_empty_dict():
    """TraditionalFilter: no text_qa_backend/image_qa_backend/audio_qa_backend at all."""
    op = _Op()
    assert _extract_cr_info(op) == {}


def test_non_kv_backend_with_none_class_level_attribute_yields_empty_dict():
    """LocalVisionModel: attribute exists (declared on VisionModel) but is None."""
    op = _Op(image_qa_backend=_NoCrBackend())
    assert _extract_cr_info(op) == {}


def test_backend_missing_the_attribute_entirely_does_not_raise():
    """LLMTextQABackend/LocalAudioModel: no effective_compression_ratio attribute."""

    class _PlainBackend:
        pass

    op = _Op(text_qa_backend=_PlainBackend())
    assert _extract_cr_info(op) == {}


def test_none_backend_attribute_is_skipped_not_treated_as_present():
    """An operator class declares the attribute name but hasn't set it (edge case)."""
    op = _Op(text_qa_backend=None, image_qa_backend=_Backend("m", 0.5))
    info = _extract_cr_info(op)
    assert info["model_name"] == "m"


# ── Wired through run_outside_db into the emitted event ──────────────────────


class _MinimalTextFilter(PhysicalOperator):
    """The smallest concrete PhysicalOperator that can flow through run_outside_db.

    Every abstract method is stubbed with a value that is never exercised by this
    test; only `text_qa_backend` (read by _extract_cr_info) and `_run_outside_db`
    (the thing being timed) matter here.
    """

    def get_operation_identifier(self):
        return "TextQaFilter-LLMTextQABackend-m-cr0.5"

    def get_llm_parameters(self):
        return {}

    def implements_logical_operator(self, logical_operator):
        return True

    def get_capabilities(self):
        return frozenset()

    def setup(self, database, logger):
        pass

    def shutdown(self):
        pass

    def get_is_multi_modal(self):
        return True

    def get_free_form_equivalence_prompt(self):
        return ""

    def get_is_expensive(self):
        return True

    def get_is_potentially_flawed(self):
        return False

    def get_hidden_column_type(self):
        return None

    def is_pipeline_breaker(self):
        return False

    def prefers_run_outside_db(self):
        return True

    def is_tuned(self):
        return True

    def get_observation(self):
        return None

    async def prepare(self, database, logger):
        pass

    async def wind_down(self):
        pass

    async def profile(self, *args, **kwargs):
        raise NotImplementedError

    async def _run_outside_db(self, inputs, input_data, llm_parameters, database_state, observation, labels, logger):
        return RunOutsideResult(
            output_data=[],
            cost=ProfilingCost(runtime=0.01, monetary_cost=0.0),
            input_data=input_data,
        )


@pytest.fixture(autouse=True)
def clean_sink():
    yield
    if monitor.get_collector() is not None:
        monitor.get_collector().close()


def test_run_outside_db_includes_cr_fields_on_the_emitted_event(tmp_path):
    op = _MinimalTextFilter(quality=1.0, fake_cost=0.0)
    op.text_qa_backend = _Backend("meta-llama/Llama-3.1-8B-Instruct", 0.5)

    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        df = pd.DataFrame({"x": [1, 2, 3]})
        asyncio.run(
            op.run_outside_db(
                inputs=[], input_data=[df], llm_parameters={}, database_state=None,
                observation=None, labels=None, logger=None,
            )
        )
        import time

        deadline = time.time() + 2
        while c.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
            time.sleep(0.01)
        agg = c.snapshot_aggregates()
        (bucket,) = [e for e in agg["operator_buckets"] if e["operation_class"] == "_MinimalTextFilter"]
        assert bucket["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert bucket["effective_compression_ratio"] == 0.5
        assert bucket["vanilla"] is False
    finally:
        c.close()


def test_run_outside_db_without_a_kv_backend_still_records_the_operator(tmp_path):
    """TraditionalFilter-shaped case: no CR fields, but the operator itself still shows up."""
    op = _MinimalTextFilter(quality=1.0, fake_cost=0.0)  # no text_qa_backend set at all

    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        df = pd.DataFrame({"x": [1, 2, 3]})
        asyncio.run(
            op.run_outside_db(
                inputs=[], input_data=[df], llm_parameters={}, database_state=None,
                observation=None, labels=None, logger=None,
            )
        )
        import time

        deadline = time.time() + 2
        while c.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
            time.sleep(0.01)
        agg = c.snapshot_aggregates()
        (bucket,) = [e for e in agg["operator_buckets"] if e["operation_class"] == "_MinimalTextFilter"]
        assert bucket["cr_label"] == "n/a"
        assert bucket["model_name"] is None
    finally:
        c.close()


def test_run_outside_db_tags_the_phase_it_ran_in(tmp_path):
    """`PhysicalOperator.profile` reaches `run_outside_db` too, so without this tag the
    tuples an operator processed during execution are pooled with the sample it
    processed while being profiled - and "rows per operator" means nothing."""
    from reasondb.utils.timing import measure, timing_session

    op = _MinimalTextFilter(quality=1.0, fake_cost=0.0)
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        df = pd.DataFrame({"x": [1, 2, 3]})
        for phase in ("profiling", "execution"):
            with timing_session(), measure(phase):
                asyncio.run(
                    op.run_outside_db(
                        inputs=[], input_data=[df], llm_parameters={}, database_state=None,
                        observation=None, labels=None, logger=None,
                    )
                )
        import time

        deadline = time.time() + 2
        while c.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
            time.sleep(0.01)
        buckets = {
            e["phase"]: e
            for e in c.snapshot_aggregates()["operator_buckets"]
            if e["operation_class"] == "_MinimalTextFilter"
        }
        assert set(buckets) == {"profiling", "execution"}
        assert buckets["execution"]["input_rows"] == 3
    finally:
        c.close()


def test_run_outside_db_tags_the_plan_step_it_ran_for(tmp_path):
    """One plan can run the *same* operator - same class, backend and ratio - at
    several positions with different prompts. `get_operation_identifier()` cannot tell
    those apart, so without a step tag the Query tab's per-operator chart would fold
    several plan steps into one bar and report their summed time as a single operator's.

    The tag must be exactly `str()` of the `__expression__` value, because that is what
    `TunedPipelineStep.to_json` writes into `operator_config` (it stringifies the same
    `llm_parameters` dict) and the dashboard joins a recorded call back to its plan step
    on plain string equality. If this assertion ever needs relaxing, the join breaks
    silently and the bars lose their order, not their data.
    """
    from reasondb.query_plan.llm_parameters import LlmParameterTemplate

    op = _MinimalTextFilter(quality=1.0, fake_cost=0.0)
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_query_start(query="q", query_index=0, n_queries=1)
        df = pd.DataFrame({"x": [1, 2, 3]})
        configs = [
            {"__expression__": LlmParameterTemplate("{t.text} mentions a director")},
            {"__expression__": LlmParameterTemplate("{t.text} praises an actor")},
        ]
        for llm_parameters in configs:
            asyncio.run(
                op.run_outside_db(
                    inputs=[], input_data=[df], llm_parameters=llm_parameters,
                    database_state=None, observation=None, labels=None, logger=None,
                )
            )
        monitor.record_query_end(query="q", cached=False, component_times={})
        import time

        deadline = time.time() + 2
        while c.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
            time.sleep(0.01)

        (run,) = c.snapshot_query_detail("q")["runs"]
        recorded = sorted(o["step_expression"] for o in run["operators"])
        # The exact strings TunedPipelineStep.to_json would put in operator_config.
        as_serialized_by_the_plan = sorted(
            {k: str(v) for k, v in cfg.items()}["__expression__"] for cfg in configs
        )
        assert recorded == as_serialized_by_the_plan
        assert len(run["operators"]) == 2
    finally:
        c.close()


def test_run_outside_db_without_an_expression_records_no_step(tmp_path):
    """Operators configured without an expression must carry None rather than a
    fabricated key."""
    op = _MinimalTextFilter(quality=1.0, fake_cost=0.0)
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_query_start(query="q", query_index=0, n_queries=1)
        df = pd.DataFrame({"x": [1, 2, 3]})
        asyncio.run(
            op.run_outside_db(
                inputs=[], input_data=[df], llm_parameters={}, database_state=None,
                observation=None, labels=None, logger=None,
            )
        )
        monitor.record_query_end(query="q", cached=False, component_times={})
        import time

        deadline = time.time() + 2
        while c.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
            time.sleep(0.01)

        (run,) = c.snapshot_query_detail("q")["runs"]
        (recorded,) = run["operators"]
        assert recorded["step_expression"] is None
    finally:
        c.close()
