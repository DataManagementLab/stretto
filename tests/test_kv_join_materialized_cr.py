"""materialized_compression_ratio wire-forwarding across all three KV cache servers.

`_validate_client_crs` is validate-only (it returns `None`), so every
`compute_*_response` method (text join, text non-join, image non-join, image join) must
call it for its side effect only and forward the client's declared
`materialized_compression_ratio` unchanged. Assigning its return value would replace the
declared ratio with `None`, which then fails the comparison against the baseline CR
resolved from the on-disk relative-index metadata on every relative-indexed request.
These tests pin the forwarding on each wire-facing entry point.

The audio server has no relative-indices path at all — `_validate_client_crs` there
requires materialized == effective and rejects vanilla outright, so its equivalent
coverage is a validation test rather than a forwarding test.
"""

import sys
import types

# These servers import concrete press classes from kvpress at module level. The tests
# here only exercise pure control flow (which value gets forwarded/validated where) and
# need none of them, so fall back to a stub when kvpress isn't installed rather than
# requiring the full CUDA/kvpress stack just to run these tests.
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

import pytest

from reasondb.backends.kv_cache_audio_qa_server import KvAudioQaModelWrapper
from reasondb.backends.kv_cache_image_qa_server import KvImageQaModelWrapper
from reasondb.backends.kv_cache_text_qa_server import KvTextQaModelWrapper


def _bare(cls, compression_ratios):
    """A model wrapper with no model loaded — enough state for _validate_client_crs
    and for the inference-stats block every compute_*_response now always attaches."""
    obj = object.__new__(cls)
    obj.compression_ratios = compression_ratios
    obj.compression_ratio_to_batch_size = {cr: None for cr in compression_ratios}
    obj.model_name = "test-model"
    return obj


# ── Text ─────────────────────────────────────────────────────────────────────


def _bare_text():
    return _bare(KvTextQaModelWrapper, (0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 0.99))


def test_text_join_forwards_declared_materialized_cr(monkeypatch):
    obj = _bare_text()
    received = {}

    async def fake_run_join(**kwargs):
        received.update(kwargs)
        return {"answers_per_text": [], "log_odds_per_text": []}

    monkeypatch.setattr(obj, "_run_kv_cache_text_join", fake_run_join)

    obj.compute_text_qa_join_response(
        column_name="c",
        unique_texts=["context"],
        questions_per_text=[["q?"]],
        compression_ratio=0.6,
        boolean_question=True,
        cache_dir="/tmp/cache",
        materialized_compression_ratio=0.3,
        vanilla=False,
    )

    assert received["materialized_compression_ratio"] == 0.3


def test_text_join_vanilla_forwards_materialized_cr_unchanged(monkeypatch):
    """Vanilla requests must also pass materialized_compression_ratio through untouched."""
    obj = _bare_text()
    received = {}

    async def fake_run_join(**kwargs):
        received.update(kwargs)
        return {"answers_per_text": [], "log_odds_per_text": []}

    monkeypatch.setattr(obj, "_run_kv_cache_text_join", fake_run_join)

    obj.compute_text_qa_join_response(
        column_name="c",
        unique_texts=["context"],
        questions_per_text=[["q?"]],
        compression_ratio=0.0,
        boolean_question=True,
        cache_dir="/tmp/cache",
        materialized_compression_ratio=0.0,
        vanilla=True,
    )

    assert received["materialized_compression_ratio"] == 0.0


def test_text_non_join_forwards_declared_materialized_cr(monkeypatch):
    """The non-join text endpoint forwards the declared materialized ratio too."""
    obj = _bare_text()
    received = {}

    async def fake_run(**kwargs):
        received.update(kwargs)
        return {"answers": [], "log_odds": []}

    monkeypatch.setattr(obj, "_run_kv_cache_text", fake_run)

    obj.compute_text_qa_response(
        column_name="c",
        texts=["context"],
        questions=["q?"],
        compression_ratio=0.6,
        boolean_question=True,
        cache_dir="/tmp/cache",
        materialized_compression_ratio=0.3,
        vanilla=False,
    )

    assert received["materialized_compression_ratio"] == 0.3


# ── Image ────────────────────────────────────────────────────────────────────


def _bare_image():
    return _bare(KvImageQaModelWrapper, (0.0, 0.3, 0.5, 0.6, 0.8, 0.9, 0.99))


def test_image_non_join_forwards_declared_materialized_cr(monkeypatch):
    obj = _bare_image()
    received = {}

    async def fake_run(**kwargs):
        received.update(kwargs)
        return ({}, {}, {})

    monkeypatch.setattr(obj, "_run_kv_cache_multimodal", fake_run)

    obj.compute_image_qa_response(
        column_name="c",
        image_paths=["img.jpg"],
        question="q?",
        compression_ratio=0.6,
        boolean_question=True,
        cache_dir="/tmp/cache",
        materialized_compression_ratio=0.3,
        vanilla=False,
    )

    assert received["materialized_compression_ratio"] == 0.3


def test_image_non_join_vanilla_forwards_materialized_cr_unchanged(monkeypatch):
    obj = _bare_image()
    received = {}

    async def fake_run(**kwargs):
        received.update(kwargs)
        return ({}, {}, {})

    monkeypatch.setattr(obj, "_run_kv_cache_multimodal", fake_run)

    obj.compute_image_qa_response(
        column_name="c",
        image_paths=["img.jpg"],
        question="q?",
        compression_ratio=0.0,
        boolean_question=True,
        cache_dir="/tmp/cache",
        materialized_compression_ratio=0.0,
        vanilla=True,
    )

    assert received["materialized_compression_ratio"] == 0.0


def test_image_join_forwards_declared_materialized_cr(monkeypatch):
    obj = _bare_image()
    received = {}

    async def fake_run_join(**kwargs):
        received.update(kwargs)
        return ({}, {}, {})

    monkeypatch.setattr(obj, "_run_kv_cache_multimodal_join", fake_run_join)

    obj.compute_image_qa_join_response(
        left_column_name="c",
        pairs=[{"left": "a.jpg", "right": "b.jpg"}],
        question="q?",
        compression_ratio=0.6,
        boolean_question=True,
        cache_dir="/tmp/cache",
        materialized_compression_ratio=0.3,
        vanilla=False,
    )

    assert received["materialized_compression_ratio"] == 0.3


def test_image_join_rejects_vanilla():
    """The image-join path has no vanilla implementation — must reject, not silently load."""
    obj = _bare_image()
    with pytest.raises(AssertionError, match="vanilla"):
        obj.compute_image_qa_join_response(
            left_column_name="c",
            pairs=[{"left": "a.jpg", "right": "b.jpg"}],
            question="q?",
            compression_ratio=0.0,
            boolean_question=True,
            cache_dir="/tmp/cache",
            materialized_compression_ratio=0.0,
            vanilla=True,
        )


# ── Audio ────────────────────────────────────────────────────────────────────


def _bare_audio():
    return _bare(KvAudioQaModelWrapper, (0.0, 0.9))


def test_audio_rejects_materialized_cr_diverging_from_effective():
    """The audio server has no relative-indices path: materialized must equal effective."""
    obj = _bare_audio()
    with pytest.raises(AssertionError, match="materialized"):
        obj._validate_client_crs(
            effective_compression_ratio=0.9,
            materialized_compression_ratio=0.0,
            vanilla=False,
        )


def test_audio_rejects_vanilla():
    obj = _bare_audio()
    with pytest.raises(AssertionError, match="vanilla"):
        obj._validate_client_crs(
            effective_compression_ratio=0.0,
            materialized_compression_ratio=0.0,
            vanilla=True,
        )


def test_audio_accepts_matching_materialized_cr():
    obj = _bare_audio()
    obj._validate_client_crs(  # must not raise
        effective_compression_ratio=0.9,
        materialized_compression_ratio=0.9,
        vanilla=False,
    )
