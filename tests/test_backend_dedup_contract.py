"""Every backend entry point evaluates each distinct input once and fans the answer out.

The shape matters above a join: the operator receives the cartesian product of the two
sides, so each distinct (question, context) - or extracted-value pair, or image - recurs
once per candidate row. `tests/test_vision_invoke_dedup.py` covers `VisionModel.invoke`;
this file covers the remaining entry points:

- `LLMTextQABackend.run` - the gold/vanilla text backend. Its on-disk prompt cache spares
  the API the repeats, but a cache hit credits its stored runtime to the SimulatedClock,
  so the *timing* would still inflate per occurrence.
- `KvTextQABackend.run_direct` - the KV counterpart of the above.
- `KvVisionModel.invoke_text_direct` - reached only by `ExtractAndQaImageFilter`, which
  dedups upstream; held to the same contract regardless.
- `TextSimilarityBackend.run` - cheap per item, linear in join rows.

What every test here asserts is the same triple: the backend issues one unit of work per
distinct input, *and* the caller still gets one result per input row in input order, *and*
the ``(runtime, cost)`` totals it returns are summed over the calls rather than the rows.
The second clause is what makes deduplication safe to do at all; the third keeps the
reported cost (and from there the optimizer's cost-per-tuple) consistent with the work
actually done. `totals_over_distinct` in `reasondb/backends/backend.py` is the single
implementation of the rule.
"""

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from reasondb.backends.audio_model import (
    CHARACTERISTICS_DICT as AUDIO_CHARACTERISTICS,
    AudioModel,
    AudioModelOutputItem,
)
from reasondb.backends.audio_qa import AudioModelAudioQABackend
from reasondb.backends.image_qa import VisionModelImageQABackend
from reasondb.backends.simulate_store import SimulateStore
from reasondb.backends.text_embeddings import TextSimilarityBackend
from reasondb.backends.text_qa import KvTextQABackend, LLMTextQABackend
from reasondb.backends.vision_model import (
    KvVisionModel,
    LocalVisionModel,
    VisionModelOutputItem,
)
from reasondb.database.indentifier import DataType, VirtualColumnIdentifier
from reasondb.query_plan.llm_parameters import LlmParameterTemplate
from reasondb.utils.logging import FileLogger
from reasondb.utils.timing import SimulatedClock

VISION_MODEL = "llava-hf/llama3-llava-next-8b-hf"
AUDIO_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"


# ── LLMTextQABackend.run ────────────────────────────────────────────────────────


class _RecordingLLM:
    """Stands in for LargeLanguageModel: counts prompts instead of calling anything."""

    model_id = "meta-llama/Llama-3.1-70B-Instruct"

    def __init__(self):
        self.prompts = []

    async def invoke_with_runtime_and_cost(self, prompt, logger, stop=[]):
        self.prompts.append(str(prompt))
        return "yes", 1.94, 0.0


def _run_llm_backend(contexts):
    """Three rows over `contexts`, as a join fan-out would deliver them."""
    llm = _RecordingLLM()
    backend = LLMTextQABackend(llm)
    column = VirtualColumnIdentifier("t.description")
    data = pd.DataFrame({"description": contexts})
    result, runtime, _cost = asyncio.run(
        backend.run(
            question_template=LlmParameterTemplate("Is this a hat?"),
            columns=[column],
            context_column_virtual=column,
            context_column_concrete=None,
            data=data,
            data_type=DataType.STRING,
            cache_dir=Path("/tmp"),
            boolean_question=True,
            logger=FileLogger(),
        )
    )
    return llm, result, runtime


def test_llm_text_backend_calls_once_per_distinct_pair():
    llm, result, runtime = _run_llm_backend(["a", "a", "a", "b"])
    assert len(llm.prompts) == 2, "one call per distinct (question, context)"
    assert len(result) == 4, "still one result per input row"
    assert runtime == pytest.approx(2 * 1.94), "runtime charged per distinct call"


def test_llm_text_backend_preserves_row_order_and_ids():
    _llm, result, _runtime = _run_llm_backend(["b", "a", "b"])
    assert [row[0] for row in result] == [0, 1, 2]
    assert all(row[1] == "yes" for row in result)


def test_llm_text_backend_without_duplicates_is_unchanged():
    llm, result, runtime = _run_llm_backend(["a", "b", "c"])
    assert len(llm.prompts) == 3
    assert len(result) == 3
    assert runtime == pytest.approx(3 * 1.94)


def test_llm_run_direct_calls_once_per_distinct_pair():
    """`ExtractAndQaFilter`/`RawTextQaFilter` reach the same backend by this door."""
    llm = _RecordingLLM()
    results, runtime, _cost = asyncio.run(
        LLMTextQABackend(llm).run_direct(
            questions=["hat?", "hat?", "shoe?"],
            contexts=["ctx", "ctx", "ctx"],
            boolean_question=True,
            logger=FileLogger(),
        )
    )
    assert len(llm.prompts) == 2
    assert len(results) == 3, "still one result per input position"
    assert [r[0] for r in results] == ["yes", "yes", "yes"]
    assert runtime == pytest.approx(2 * 1.94)


# ── KvTextQABackend.run_direct ──────────────────────────────────────────────────
#
# The sibling of `LLMTextQABackend.run_direct` above, reached whenever the same operators
# are configured with a KV backend rather than OpenAI - which is the shipped default, so
# this is the live path.

TEXT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DIRECT_RUNTIME_PER_PAIR = 0.75


def _kv_backend():
    return KvTextQABackend(
        model_id=TEXT_MODEL,
        effective_compression_ratio=0.0,
        materialized_compression_ratio=0.0,
        vanilla=True,
    )


@pytest.fixture
def direct_store():
    store = SimulateStore()
    for question in ("same?", "other?"):
        store.record_text_qa(
            _kv_backend().model_id,
            question,
            "ctx",
            response="yes",
            log_odds=1.0,
            runtime=DIRECT_RUNTIME_PER_PAIR,
            effective_compression_ratio=None,
            materialized_compression_ratio=None,
            vanilla=False,
        )
    SimulateStore.set_simulate(store)
    return store


def test_kv_run_direct_looks_up_each_distinct_pair_once(direct_store, monkeypatch):
    """Under --simulate a lookup is not free: it advances the SimulatedClock."""
    lookups = []
    original = direct_store.lookup_text_qa

    def counting_lookup(model_id, question, context):
        lookups.append((question, context))
        return original(model_id, question, context)

    monkeypatch.setattr(direct_store, "lookup_text_qa", counting_lookup)

    before = SimulatedClock.now()
    results, runtime, _cost = asyncio.run(
        _kv_backend().run_direct(
            questions=["same?", "same?", "same?", "other?"],
            contexts=["ctx"] * 4,
            boolean_question=True,
            logger=FileLogger(),
        )
    )
    assert lookups == [("same?", "ctx"), ("other?", "ctx")]
    assert len(results) == 4, "still one result per input position"
    assert runtime == pytest.approx(2 * DIRECT_RUNTIME_PER_PAIR)
    # The clock and the reported total move together, which is the property that keeps
    # `operator_run.seconds` and `operator_run.runtime` comparable.
    assert SimulatedClock.now() - before == pytest.approx(runtime)


def test_kv_run_direct_fans_the_right_answer_to_each_position(direct_store):
    results, _runtime, _cost = asyncio.run(
        _kv_backend().run_direct(
            questions=["same?", "other?", "same?"],
            contexts=["ctx"] * 3,
            boolean_question=True,
            logger=FileLogger(),
        )
    )
    assert all(r is not None for r in results), "no position left unfilled"
    assert [r[0] for r in results] == ["yes", "yes", "yes"]


def test_kv_run_direct_sends_only_distinct_pairs_to_the_server(monkeypatch):
    """The live branch: the request body itself must not carry the fan-out."""
    captured = []

    class _Response:
        status_code = 200

        def json(self):
            n = len(captured[-1]["questions"])
            return {"answers": ["yes"] * n, "log_odds": [float(i) for i in range(n)]}

    monkeypatch.setattr(
        "reasondb.backends.text_qa.requests.post",
        lambda url, json=None, **kwargs: (captured.append(json), _Response())[1],
    )

    results, _runtime, _cost = asyncio.run(
        _kv_backend().run_direct(
            questions=["a?", "b?", "a?"],
            contexts=["ctx"] * 3,
            boolean_question=True,
            logger=FileLogger(),
        )
    )
    assert captured[-1]["questions"] == ["a?", "b?"]
    assert len(results) == 3, "still one result per input position"
    # Positions sharing a pair share its score; distinct pairs keep their own.
    assert [r[1] for r in results] == [0.0, 1.0, 0.0]


# ── KvVisionModel.invoke_text_direct ────────────────────────────────────────────


@pytest.fixture
def text_direct_posts(monkeypatch):
    """Capture /text_qa_direct payloads instead of hitting the LLaVA server."""
    captured = []

    class _Response:
        status_code = 200

        def json(self):
            n = len(captured[-1]["questions"])
            return {"log_odds": [float(i) for i in range(n)]}

    def fake_post(url, json=None, **kwargs):
        captured.append(json)
        return _Response()

    monkeypatch.setattr("reasondb.backends.vision_model.requests.post", fake_post)
    return captured


def _model():
    return KvVisionModel(
        model_id=VISION_MODEL,
        effective_compression_ratio=0.0,
        materialized_compression_ratio=0.0,
        vanilla=True,
    )


def test_text_direct_sends_only_distinct_pairs(text_direct_posts):
    questions = ["same?", "same?", "same?", "other?"]
    contexts = ["ctx", "ctx", "ctx", "ctx"]
    items = asyncio.run(
        _model().invoke_text_direct(
            questions=questions,
            contexts=contexts,
            boolean_question=True,
            logger=FileLogger(),
        )
    )
    sent = text_direct_posts[-1]
    assert sent["questions"] == ["same?", "other?"]
    assert sent["contexts"] == ["ctx", "ctx"]
    assert len(items) == 4, "still one item per input position"


def test_text_direct_fans_the_right_score_to_each_position(text_direct_posts):
    """Positions sharing a pair share its score; distinct pairs keep their own."""
    items = asyncio.run(
        _model().invoke_text_direct(
            questions=["a?", "b?", "a?"],
            contexts=["c", "c", "c"],
            boolean_question=True,
            logger=FileLogger(),
        )
    )
    # fake server returns log_odds [0.0, 1.0] for the two distinct pairs, in order
    assert [item.log_odds for item in items] == [0.0, 1.0, 0.0]


# ── TextSimilarityBackend.run ───────────────────────────────────────────────────


def _pair_score(text1: str, text2: str) -> float:
    """A score that depends on the pair alone, so assertions need no request order."""
    return float(len(text1) * 10 + len(text2))


def _sent_pairs(payload):
    texts = payload["texts"]
    return [(texts[a], texts[b]) for a, b in zip(payload["idx1"], payload["idx2"])]


@pytest.fixture
def sim_posts(monkeypatch):
    """Answers a request the way the server does: one score per index pair.

    The request carries the distinct texts plus index pairs rather than two lists of
    strings (a pair list is the join fan-out, so spelling it out is the expensive part),
    and deduplication does not preserve first-occurrence order - hence a fake that
    scores the pair rather than its position.
    """
    captured = []

    class _Response:
        status_code = 200

        def json(self):
            pairs = _sent_pairs(captured[-1])
            return {"similarities": [_pair_score(t1, t2) for t1, t2 in pairs]}

    def fake_post(url, json=None, **kwargs):
        captured.append(json)
        return _Response()

    monkeypatch.setattr("reasondb.backends.text_embeddings.requests.post", fake_post)
    return captured


def test_similarity_scores_each_distinct_pair_once(sim_posts):
    texts1 = ["red", "red", "blue", "red"]
    texts2 = ["crimson", "crimson", "azure", "crimson"]
    similarities, _runtime, _cost = asyncio.run(
        TextSimilarityBackend("model").run(texts1, texts2)
    )
    assert sorted(_sent_pairs(sim_posts[-1])) == [("blue", "azure"), ("red", "crimson")]
    assert sorted(sim_posts[-1]["texts"]) == ["azure", "blue", "crimson", "red"]
    assert len(similarities) == 4, "one score per input position"
    assert similarities == [_pair_score(t1, t2) for t1, t2 in zip(texts1, texts2)]


def test_similarity_without_duplicates_is_unchanged(sim_posts):
    similarities, _runtime, _cost = asyncio.run(
        TextSimilarityBackend("model").run(["a", "b"], ["x", "y"])
    )
    assert sorted(_sent_pairs(sim_posts[-1])) == [("a", "x"), ("b", "y")]
    assert similarities == [_pair_score("a", "x"), _pair_score("b", "y")]


# ── The reported totals: summed over calls, not rows ────────────────────────────
#
# The clause above deduplicates the work; these assert that what the backend *reports*
# having spent is on the same basis. `VisionModel.invoke` and `AudioModel.invoke` return
# one item per input row, each carrying the full runtime of the single call behind it, so
# summing that list is the mistake `totals_over_distinct` exists to prevent.

RUNTIME_PER_ITEM = 1.94
COST_PER_ITEM = 0.25
IMAGES = [Path("/img/a.jpg"), Path("/img/b.jpg"), Path("/img/c.jpg")]
QUESTION = "Does this depict a landscape?"


def _cartesian(items):
    """What a self-join hands the operator: every item once per candidate pair."""
    return [left for left in items for _right in items]


@pytest.fixture(autouse=True)
def clean_store():
    """The store is a process-global registry; don't leak it between tests."""
    yield
    SimulateStore.set_precompute(None)
    SimulateStore.set_simulate(None)


@pytest.fixture
def vision_store():
    store = SimulateStore()
    for image in IMAGES:
        store.record_vision(
            VISION_MODEL,
            QUESTION,
            str(image),
            response="yes",
            log_odds=1.0,
            runtime=RUNTIME_PER_ITEM,
            cost=COST_PER_ITEM,
            effective_compression_ratio=None,
            materialized_compression_ratio=None,
            vanilla=False,
        )
    SimulateStore.set_simulate(store)
    return store


def _run_image_backend(paths):
    backend = VisionModelImageQABackend(LocalVisionModel(VISION_MODEL))
    column = VirtualColumnIdentifier("t.image")
    return asyncio.run(
        backend.run(
            question=QUESTION,
            image_column_virtual=column,
            image_column_concrete=None,
            boolean_question=True,
            data=pd.DataFrame({"image": [str(p) for p in paths]}),
            data_type=DataType.STRING,
            cache_dir=Path("/tmp"),
            logger=FileLogger(),
        )
    )


def test_image_backend_charges_per_call_not_per_row(vision_store):
    """Nine join rows over three images cost three calls, not nine."""
    paths = _cartesian(IMAGES)
    assert len(paths) == 9, "the join fan-out this test is about"
    result, runtime, cost = _run_image_backend(paths)
    assert len(result) == 9, "still one result per input row"
    assert runtime == pytest.approx(len(IMAGES) * RUNTIME_PER_ITEM)
    assert cost == pytest.approx(len(IMAGES) * COST_PER_ITEM)


def test_image_backend_without_duplicates_is_unchanged(vision_store):
    """No path recurs, so calls and rows coincide and nothing moves."""
    result, runtime, cost = _run_image_backend(IMAGES)
    assert len(result) == 3
    assert runtime == pytest.approx(len(IMAGES) * RUNTIME_PER_ITEM)
    assert cost == pytest.approx(len(IMAGES) * COST_PER_ITEM)


@pytest.fixture
def text_direct_store():
    store = SimulateStore()
    # Keyed on the *compression-suffixed* id, as `invoke_text_direct` looks it up -- see
    # its docstring on why a text-only call still carries the ratio metadata.
    for question in ("same?", "other?"):
        store.record_text_qa(
            _model().model_id,
            question,
            "ctx",
            response="yes",
            log_odds=1.0,
            runtime=RUNTIME_PER_ITEM,
            effective_compression_ratio=None,
            materialized_compression_ratio=None,
            vanilla=False,
        )
    SimulateStore.set_simulate(store)
    return store


def test_text_direct_backend_charges_per_call_not_per_row(text_direct_store):
    """`run_text_direct`'s only caller dedups upstream; the contract holds anyway."""
    log_odds, runtime, _cost = asyncio.run(
        VisionModelImageQABackend(_model()).run_text_direct(
            questions=["same?", "same?", "same?", "other?"],
            contexts=["ctx", "ctx", "ctx", "ctx"],
            boolean_question=True,
            logger=FileLogger(),
        )
    )
    assert len(log_odds) == 4, "still one score per input position"
    assert runtime == pytest.approx(2 * RUNTIME_PER_ITEM)


def test_image_join_backend_charges_per_pair_not_per_row(monkeypatch):
    """`run_join` asks `invoke_join` for distinct pairs and is billed per pair, not per row."""
    backend = VisionModelImageQABackend(_model())  # invoke_join is the Kv path's

    async def fake_invoke_join(left_column, pairs, question, boolean_question, cache_dir, logger):
        return [
            VisionModelOutputItem(
                image_path=right,
                response="yes",
                log_odds=1.0,
                runtime=RUNTIME_PER_ITEM,
                cost=COST_PER_ITEM,
            )
            for _left, right in pairs
        ]

    monkeypatch.setattr(backend.vision_model, "invoke_join", fake_invoke_join)

    left = VirtualColumnIdentifier("t.left_image")
    right = VirtualColumnIdentifier("t.right_image")
    # Two distinct pairs, each arriving on three rows.
    data = pd.DataFrame(
        {
            "left_image": ["/img/a.jpg"] * 3 + ["/img/b.jpg"] * 3,
            "right_image": ["/img/x.jpg"] * 3 + ["/img/y.jpg"] * 3,
        }
    )
    result, runtime, cost = asyncio.run(
        backend.run_join(
            question=QUESTION,
            left_image_column_virtual=left,
            left_image_column_concrete=None,
            right_image_column_virtual=right,
            right_image_column_concrete=None,
            boolean_question=True,
            data=data,
            cache_dir=Path("/tmp"),
            logger=FileLogger(),
        )
    )
    assert len(result) == 6, "still one result per input row"
    assert runtime == pytest.approx(2 * RUNTIME_PER_ITEM)
    assert cost == pytest.approx(2 * COST_PER_ITEM)


class _RecordingAudioModel(AudioModel):
    """Answers from memory instead of a server, one item per input row like the real one."""

    def __init__(self):
        super().__init__(AUDIO_MODEL, AUDIO_CHARACTERISTICS[AUDIO_MODEL])

    @property
    def cache_enabled(self) -> bool:
        return False

    @property
    def returns_log_odds(self) -> bool:
        return False

    def setup(self, logger):
        pass

    async def prepare(self, column, cache_dir, audio_paths):
        pass

    async def wind_down(self):
        pass

    async def _invoke(
        self, column, question, audio_paths, cache_dir, boolean_question, logger
    ):
        return [
            AudioModelOutputItem(
                audio_path=path,
                response="yes",
                log_odds=1.0,
                runtime=RUNTIME_PER_ITEM,
                cost=COST_PER_ITEM,
            )
            for path in audio_paths
        ]


def test_audio_backend_charges_per_call_not_per_row():
    """No audio benchmark joins, but the audio backend is held to the same rule."""
    clips = [Path("/audio/a.wav"), Path("/audio/b.wav")]
    paths = _cartesian(clips)
    column = VirtualColumnIdentifier("t.clip")
    result, runtime, cost = asyncio.run(
        AudioModelAudioQABackend(_RecordingAudioModel()).run(
            question=QUESTION,
            audio_column_virtual=column,
            audio_column_concrete=None,
            boolean_question=True,
            data=pd.DataFrame({"clip": [str(p) for p in paths]}),
            cache_dir=Path("/tmp"),
            logger=FileLogger(),
        )
    )
    assert len(result) == 4, "still one result per input row"
    assert runtime == pytest.approx(len(clips) * RUNTIME_PER_ITEM)
    assert cost == pytest.approx(len(clips) * COST_PER_ITEM)
