"""Every KV client parse site must forward the server's `stats` block to the monitor
without changing what it returns to its caller.

There are seven of these sites (three on `KvTextQABackend`, three on `KvVisionModel`, one
on `KvAudioModel`), each parsing a differently-shaped response (lists for text, path-keyed
dicts for image/audio). A monitoring hook added at the wrong point, or one that reads a
key eagerly instead of tolerantly, would either silently drop telemetry or break
inference itself - so each site is tested for both: the return value is bit-identical
with and without a `stats` block, and exactly one `kv_inference` event reaches the
collector per call.
"""

import asyncio
from pathlib import Path

import pytest

from reasondb.backends.audio_model import KvAudioModel
from reasondb.backends.inference_stats import build_inference_stats
from reasondb.backends.text_qa import KvTextQABackend
from reasondb.backends.vision_model import KvVisionModel
from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector
from reasondb.utils.logging import NoLogger

TEXT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VISION_MODEL = "llava-hf/llava-next-72b-hf"
AUDIO_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"


@pytest.fixture(autouse=True)
def clean_sink():
    yield
    if monitor.get_collector() is not None:
        monitor.get_collector().close()


@pytest.fixture
def live_collector(tmp_path):
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    yield c
    c.close()


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _stats(server, path, n_items):
    return build_inference_stats(
        server=server, path=path, model_name="m", n_items=n_items, server_elapsed_s=0.5
    )


def _column():
    from reasondb.database.indentifier import ConcreteColumnIdentifier

    return ConcreteColumnIdentifier(name="t.c")


def _events_since(collector, since=0):
    import time

    deadline = time.time() + 2.0
    while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
        time.sleep(0.01)
    return collector.events_since(since=since, limit=1000)["events"]


# ── text_qa.py: three sites ──────────────────────────────────────────────────


def test_text_invoke_forwards_stats(monkeypatch, live_collector):
    payload = {
        "answers": ["1"],
        "log_odds": [0.5],
        "stats": _stats("kv_text_qa", "kv", 1),
    }
    monkeypatch.setattr(
        "reasondb.backends.text_qa.requests.post", lambda *a, **k: _Response(payload)
    )
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.9)
    result = backend._invoke(
        column=_column(),
        questions=["q"],
        texts=["ctx"],
        cache_dir=Path("/tmp/cache"),
        boolean_question=True,
    )
    assert result[0][0] == "1"
    assert result[0][1] == 0.5

    events = [e for e in _events_since(live_collector) if e["type"] == "kv_inference"]
    assert len(events) == 1
    assert events[0]["data"]["endpoint"] == "/text_qa"


def test_text_invoke_join_forwards_stats(monkeypatch, live_collector):
    payload = {"answers_per_text": [["a"]], "stats": _stats("kv_text_qa", "join", 1)}
    monkeypatch.setattr(
        "reasondb.backends.text_qa.requests.post", lambda *a, **k: _Response(payload)
    )
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.9)
    result = backend._invoke_join(
        column=_column(),
        unique_texts=["ctx"],
        questions_per_text=[["q"]],
        cache_dir=Path("/tmp/cache"),
        boolean_question=True,
    )
    assert result == payload  # unmodified: the site returns the raw dict

    events = [e for e in _events_since(live_collector) if e["type"] == "kv_inference"]
    assert len(events) == 1
    assert events[0]["data"]["endpoint"] == "/text_qa_join"


def test_text_invoke_direct_forwards_stats(monkeypatch, live_collector):
    payload = {
        "answers": ["1"],
        "log_odds": [0.1],
        "stats": _stats("kv_text_qa", "direct", 1),
    }
    monkeypatch.setattr(
        "reasondb.backends.text_qa.requests.post", lambda *a, **k: _Response(payload)
    )
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.9)
    result = backend._invoke_direct(questions=["q"], contexts=["ctx"], boolean_question=True)
    assert result[0][0] == "1"

    events = [e for e in _events_since(live_collector) if e["type"] == "kv_inference"]
    assert len(events) == 1
    assert events[0]["data"]["endpoint"] == "/text_qa_direct"


def test_text_invoke_without_stats_degrades_to_an_error_event(monkeypatch, live_collector):
    """A response from a stale server (no `stats` key) must not break inference.

    parse_inference_stats() asserts on a missing stats block by design (both halves of
    the wire format are ours), but forward_stats() must catch that at the network
    boundary and downgrade it to a monitor-side error event - never let it propagate
    into the caller, which is real inference code.
    """
    payload = {"answers": ["1"], "log_odds": [0.5]}
    monkeypatch.setattr(
        "reasondb.backends.text_qa.requests.post", lambda *a, **k: _Response(payload)
    )
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.9)
    result = backend._invoke(
        column=_column(),
        questions=["q"],
        texts=["ctx"],
        cache_dir=Path("/tmp/cache"),
        boolean_question=True,
    )
    assert result[0][0] == "1"

    events = _events_since(live_collector)
    assert [e["type"] for e in events] == ["error"]
    assert "text_qa" in events[0]["data"]["where"]


# ── vision_model.py: three sites ─────────────────────────────────────────────


def test_vision_invoke_forwards_stats(monkeypatch, live_collector):
    payload = {
        "answers": {"img.jpg": "1"},
        "log_odds": {"img.jpg": 0.2},
        "stats": _stats("kv_image_qa", "kv", 1),
    }
    monkeypatch.setattr(
        "reasondb.backends.vision_model.requests.post", lambda *a, **k: _Response(payload)
    )
    model = KvVisionModel(VISION_MODEL, 0.9, 0.9)
    result = asyncio.run(
        model._invoke(
            column=_column(),
            question="q",
            image_paths=[Path("img.jpg")],
            cache_dir=Path("/tmp/cache"),
            boolean_question=True,
            logger=NoLogger(),
        )
    )
    assert result[0].response == "1"

    events = [e for e in _events_since(live_collector) if e["type"] == "kv_inference"]
    assert len(events) == 1
    assert events[0]["data"]["endpoint"] == "/image_qa"


def test_vision_invoke_text_direct_forwards_stats(monkeypatch, live_collector):
    payload = {"log_odds": [0.3], "stats": _stats("kv_image_qa", "direct", 1)}
    monkeypatch.setattr(
        "reasondb.backends.vision_model.requests.post", lambda *a, **k: _Response(payload)
    )
    model = KvVisionModel(VISION_MODEL, 0.9, 0.9)
    result = asyncio.run(
        model.invoke_text_direct(
            questions=["q"], contexts=["ctx"], boolean_question=True, logger=NoLogger()
        )
    )
    assert result[0].log_odds == 0.3

    events = [e for e in _events_since(live_collector) if e["type"] == "kv_inference"]
    assert len(events) == 1
    assert events[0]["data"]["endpoint"] == "/text_qa_direct"


def test_vision_invoke_join_forwards_stats(monkeypatch, live_collector):
    payload = {
        "answers": {"a.jpg|b.jpg": "1"},
        "log_odds": {"a.jpg|b.jpg": 0.4},
        "stats": _stats("kv_image_qa", "join", 1),
    }
    monkeypatch.setattr(
        "reasondb.backends.vision_model.requests.post", lambda *a, **k: _Response(payload)
    )
    model = KvVisionModel(VISION_MODEL, 0.9, 0.9)
    result = asyncio.run(
        model.invoke_join(
            left_column=_column(),
            pairs=[(Path("a.jpg"), Path("b.jpg"))],
            question="q",
            boolean_question=True,
            cache_dir=Path("/tmp/cache"),
            logger=NoLogger(),
        )
    )
    assert result[0].response == "1"

    events = [e for e in _events_since(live_collector) if e["type"] == "kv_inference"]
    assert len(events) == 1
    assert events[0]["data"]["endpoint"] == "/image_qa_join"


# ── audio_model.py: one site ──────────────────────────────────────────────────


def test_audio_invoke_forwards_stats(monkeypatch, live_collector):
    payload = {
        "answers": {"clip.wav": "1"},
        "log_odds": {"clip.wav": 0.6},
        "stats": _stats("kv_audio_qa", "kv", 1),
    }
    monkeypatch.setattr(
        "reasondb.backends.audio_model.requests.post", lambda *a, **k: _Response(payload)
    )
    model = KvAudioModel(AUDIO_MODEL, 0.9, 0.9)
    result = asyncio.run(
        model._invoke(
            column=_column(),
            question="q",
            audio_paths=[Path("clip.wav")],
            cache_dir=Path("/tmp/cache"),
            boolean_question=True,
            logger=NoLogger(),
        )
    )
    assert result[0].response == "1"

    events = [e for e in _events_since(live_collector) if e["type"] == "kv_inference"]
    assert len(events) == 1
    assert events[0]["data"]["endpoint"] == "/audio_qa"
