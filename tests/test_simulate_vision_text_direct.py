"""`--simulate` must replay the vision model's text-only endpoint, not call it.

`ExtractAndQaImageFilter` runs three model phases: extract left images, extract
right images, then a text-only match over the extracted values via
`/text_qa_direct`. The two extraction phases go through `VisionModel.invoke`,
which is recorded/replayed. The match phase goes through
`KvVisionModel.invoke_text_direct`, which must be recorded/replayed as well -
otherwise a simulated run of any image-join query would still need a live vision
model server.
"""

import asyncio
from pathlib import Path

import pytest

from reasondb.backends.simulate_store import SimulateStore
from reasondb.backends.vision_model import KvVisionModel
from reasondb.utils.logging import FileLogger

VISION_MODEL = "llava-hf/llama3-llava-next-8b-hf"
QUESTIONS = ["Is 'halo' related to 'radiance'?", "Is 'halo' related to 'dog'?"]
CONTEXTS = ["halo", "halo"]
LOG_ODDS = [1.5, -2.5]


@pytest.fixture(autouse=True)
def clean_store():
    """The store is a process-global registry; don't leak it between tests."""
    yield
    SimulateStore.set_precompute(None)
    SimulateStore.set_simulate(None)


@pytest.fixture
def posts(monkeypatch):
    """Capture /text_qa_direct calls instead of hitting the LLaVA server."""
    captured = []

    class _Response:
        status_code = 200

        def json(self):
            return {"log_odds": LOG_ODDS}

    def fake_post(url, json=None, **kwargs):
        captured.append((url, json))
        return _Response()

    monkeypatch.setattr("reasondb.backends.vision_model.requests.post", fake_post)
    return captured


def _invoke(model):
    return asyncio.run(
        model.invoke_text_direct(
            questions=QUESTIONS,
            contexts=CONTEXTS,
            boolean_question=True,
            logger=FileLogger(),
        )
    )


def test_precompute_records_every_call(posts):
    store = SimulateStore()
    SimulateStore.set_precompute(store)
    model = KvVisionModel(VISION_MODEL, 0.9, 0.0)

    items = _invoke(model)

    assert len(posts) == 1, "precompute still runs the real call"
    assert [i.log_odds for i in items] == LOG_ODDS
    # Text channel, keyed on (question, context) - the endpoint takes no image.
    bucket = store._text_qa[model.model_id]
    assert len(bucket) == 2
    record = bucket[store._hash(f"{QUESTIONS[0]}-{CONTEXTS[0]}")]
    assert record["log_odds"] == LOG_ODDS[0]
    assert record["effective_compression_ratio"] == 0.9
    assert record["materialized_compression_ratio"] == 0.0


def test_simulate_replays_without_touching_the_server(posts):
    store = SimulateStore()
    SimulateStore.set_precompute(store)
    model = KvVisionModel(VISION_MODEL, 0.9, 0.0)
    _invoke(model)
    posts.clear()

    SimulateStore.set_precompute(None)
    SimulateStore.set_simulate(store)
    items = _invoke(model)

    assert posts == [], "simulate must not call the LLaVA server"
    assert [i.log_odds for i in items] == LOG_ODDS


def test_simulate_survives_save_and_load(posts, tmp_path):
    store = SimulateStore()
    SimulateStore.set_precompute(store)
    model = KvVisionModel(VISION_MODEL, 0.9, 0.0)
    _invoke(model)
    store.save(tmp_path / "precompute.json")
    posts.clear()

    SimulateStore.set_precompute(None)
    SimulateStore.set_simulate(SimulateStore.load(tmp_path / "precompute.json"))
    items = _invoke(model)

    assert posts == []
    assert [i.log_odds for i in items] == LOG_ODDS


def test_simulate_miss_is_an_error_not_a_network_call(posts):
    SimulateStore.set_simulate(SimulateStore())
    model = KvVisionModel(VISION_MODEL, 0.9, 0.0)

    with pytest.raises(RuntimeError, match="missing precomputed text-direct result"):
        _invoke(model)
    assert posts == [], "a miss must not silently fall back to the server"


def test_replayed_runtime_is_credited_to_the_simulated_clock(posts):
    """Phase timers must still see the runtime the replayed call stands in for."""
    from reasondb.utils.timing import SimulatedClock

    store = SimulateStore()
    store.record_text_qa(
        KvVisionModel(VISION_MODEL, 0.9, 0.0).model_id,
        QUESTIONS[0],
        CONTEXTS[0],
        "",
        LOG_ODDS[0],
        runtime=7.5,
        effective_compression_ratio=0.9,
        materialized_compression_ratio=0.0,
        vanilla=False,
    )
    SimulateStore.set_simulate(store)
    model = KvVisionModel(VISION_MODEL, 0.9, 0.0)

    before = SimulatedClock.now()
    items = asyncio.run(
        model.invoke_text_direct(
            questions=[QUESTIONS[0]],
            contexts=[CONTEXTS[0]],
            boolean_question=True,
            logger=FileLogger(),
        )
    )

    assert items[0].runtime == 7.5
    assert SimulatedClock.now() - before == pytest.approx(7.5)


def test_image_paths_are_absent_from_replayed_items(posts):
    """The endpoint is text-only; callers read log_odds, never an image path."""
    store = SimulateStore()
    SimulateStore.set_precompute(store)
    model = KvVisionModel(VISION_MODEL, 0.0, 0.0)
    _invoke(model)

    SimulateStore.set_precompute(None)
    SimulateStore.set_simulate(store)
    assert all(i.image_path == Path("") for i in _invoke(model))
