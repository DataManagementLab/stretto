"""Every KV cache server must attach a `stats` block to every response, on every route -
including the audio server, which has no per-batch instrumentation to draw from.

This is the server half of the contract `reasondb.backends.inference_stats` defines.
Model-loading paths need a GPU and are out of scope for a unit test; these tests instead
target the thin `compute_*_response` wrapper methods with the model call mocked out,
the same seam `tests/test_kv_join_materialized_cr.py` uses. Its `_bare()` helper is
reused here; it carries `compression_ratio_to_batch_size` and `model_name`, the two
attributes `_request_stats` needs.
"""

import asyncio
import sys
import types

import pytest

# Match test_kv_join_materialized_cr.py's kvpress stub: these servers import concrete
# press classes at module level, which the stats wiring under test doesn't exercise.
try:
    import kvpress  # noqa: F401

    kvpress.KeyRerotationPress
    kvpress.ExpectedAttentionPress
    kvpress.KVzipPress
    kvpress.FinchPress
except (ImportError, AttributeError):
    _stub = types.ModuleType("kvpress")
    for _name in ("KeyRerotationPress", "ExpectedAttentionPress", "KVzipPress", "FinchPress"):
        setattr(_stub, _name, object)
    sys.modules["kvpress"] = _stub

from reasondb.backends.inference_stats import STATS_KEYS
from reasondb.backends.kv_cache_audio_qa_server import KvAudioQaModelWrapper
from reasondb.backends.kv_cache_image_qa_server import KvImageQaModelWrapper


def _bare(cls, compression_ratios):
    obj = object.__new__(cls)
    obj.compression_ratios = compression_ratios
    obj.compression_ratio_to_batch_size = {cr: None for cr in compression_ratios}
    obj.model_name = "test-model"
    return obj


def test_image_kv_response_always_carries_stats(monkeypatch):
    obj = _bare(KvImageQaModelWrapper, (0.0, 0.9))

    async def fake_run(**kwargs):
        return {"img.jpg": "1"}, {"img.jpg": 0.5}, {}

    monkeypatch.setattr(obj, "_run_kv_cache_multimodal", fake_run)
    resp = obj.compute_image_qa_response(
        column_name="c",
        image_paths=["img.jpg"],
        question="q?",
        compression_ratio=0.9,
        boolean_question=True,
        cache_dir="/tmp/cache",
        materialized_compression_ratio=0.9,
        vanilla=False,
    )
    assert set(resp["stats"]) == STATS_KEYS
    assert resp["stats"]["server"] == "kv_image_qa"
    assert resp["stats"]["path"] == "kv"
    assert resp["answers"] == {"img.jpg": "1"}  # response body itself is unchanged


def test_image_join_response_always_carries_stats(monkeypatch):
    obj = _bare(KvImageQaModelWrapper, (0.0, 0.9))

    async def fake_join(**kwargs):
        return {"a.jpg|b.jpg": "1"}, {"a.jpg|b.jpg": 0.1}, {}

    monkeypatch.setattr(obj, "_run_kv_cache_multimodal_join", fake_join)
    resp = obj.compute_image_qa_join_response(
        left_column_name="c",
        pairs=[{"left": "a.jpg", "right": "b.jpg"}],
        question="q?",
        compression_ratio=0.9,
        boolean_question=True,
        cache_dir="/tmp/cache",
        materialized_compression_ratio=0.9,
        vanilla=False,
    )
    assert set(resp["stats"]) == STATS_KEYS
    assert resp["stats"]["path"] == "join"


def test_image_text_only_response_always_carries_stats(monkeypatch):
    obj = _bare(KvImageQaModelWrapper, (0.0, 0.9))

    async def fake_direct(**kwargs):
        return [0.2]

    monkeypatch.setattr(obj, "_run_text_only_qa", fake_direct)
    resp = obj.compute_text_only_qa_response(
        questions=["q"], contexts=["ctx"], boolean_question=True
    )
    assert set(resp["stats"]) == STATS_KEYS
    assert resp["stats"]["path"] == "direct"
    assert resp["log_odds"] == [0.2]


def test_audio_kv_response_always_carries_stats_despite_no_instrumentation(monkeypatch):
    """The audio server has no per-batch debug logs; stats must still be complete."""
    obj = _bare(KvAudioQaModelWrapper, (0.0, 0.9))

    async def fake_run(**kwargs):
        return {"clip.wav": "1"}, {"clip.wav": 0.3}

    monkeypatch.setattr(obj, "_run_kv_cache_multimodal", fake_run)
    resp = obj.compute_audio_qa_response(
        column_name="c",
        audio_paths=["clip.wav"],
        question="q?",
        compression_ratio=0.9,
        boolean_question=True,
        cache_dir="/tmp/cache",
        materialized_compression_ratio=0.9,
        vanilla=False,
    )
    assert set(resp["stats"]) == STATS_KEYS
    assert resp["stats"]["server"] == "kv_audio_qa"
    assert resp["stats"]["cache_load_s"] is None  # never measured on this server
