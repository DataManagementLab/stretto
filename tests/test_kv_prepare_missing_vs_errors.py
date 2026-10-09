"""Client-side crash contract for KV cache setup (`prepare()`).

`prepare_caches` on the server reports two different kinds of gaps, and `prepare()` must
react to them differently:

  - `n_missing`: a cache/relative-index that was never generated at all (relative-indices
    mode, when the offline pregeneration script — `generate_kv_caches_indices.py` /
    `generate_kv_caches_image_indices.py` — was skipped). This is a setup mistake, likely
    affecting most/all of the data, and must crash immediately, before any query runs.
  - `n_generation_errors`: a handful of inputs where generation genuinely failed (e.g. a
    corrupted image or malformed text), in physical mode. The serve path already tolerates
    this per-item (skip the cache, answer "Not sure"/empty, keep going), so `prepare()`
    must NOT crash on it — otherwise a few bad rows would take down the whole operator.

These tests pin that contract on the client side (`KvTextQABackend.prepare` /
`KvVisionModel.prepare`) using a fake `requests.post`, so no server or model is needed.
"""

import asyncio
from pathlib import Path

import pytest

from reasondb.backends.text_qa import KvTextQABackend
from reasondb.backends.vision_model import KvVisionModel

TEXT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VISION_MODEL = "llava-hf/llava-next-72b-hf"


class _Response:
    """Stand-in for requests.Response covering what the backends read."""

    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _fake_post(payload):
    def post(url, json=None, **kwargs):
        return _Response(payload)

    return post


def _column():
    from reasondb.database.indentifier import ConcreteColumnIdentifier

    return ConcreteColumnIdentifier(name="t.c")


# ── Text backend ─────────────────────────────────────────────────────────────


def test_text_prepare_crashes_when_never_generated(monkeypatch):
    monkeypatch.setattr(
        "reasondb.backends.text_qa.requests.post",
        _fake_post(
            {
                "status": "cache_ready",
                "n_texts": 10,
                "n_missing": 3,
                "missing_hashes": ["abc123", "def456", "ghi789"],
            }
        ),
    )
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.3)
    with pytest.raises(RuntimeError, match=r"3/10"):
        asyncio.run(
            backend.prepare(
                column=_column(), cache_dir=Path("/tmp/cache"), texts=["a"] * 10
            )
        )


def test_text_prepare_tolerates_a_few_generation_errors(monkeypatch):
    """A couple of corrupted texts must not block the whole operator's setup."""
    monkeypatch.setattr(
        "reasondb.backends.text_qa.requests.post",
        _fake_post(
            {
                "status": "cache_ready",
                "n_texts": 10,
                "n_generation_errors": 2,
                "generation_error_hashes": ["bad1", "bad2"],
            }
        ),
    )
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.9)
    asyncio.run(  # must not raise
        backend.prepare(column=_column(), cache_dir=Path("/tmp/cache"), texts=["a"] * 10)
    )


def test_text_prepare_ok_when_nothing_missing(monkeypatch):
    """No `n_missing` key at all (e.g. an older server) must default to 0."""
    monkeypatch.setattr(
        "reasondb.backends.text_qa.requests.post",
        _fake_post({"status": "cache_ready"}),
    )
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.9)
    asyncio.run(  # must not raise
        backend.prepare(column=_column(), cache_dir=Path("/tmp/cache"), texts=["a"])
    )


# ── Vision backend ───────────────────────────────────────────────────────────


def test_vision_prepare_crashes_when_never_generated(monkeypatch):
    monkeypatch.setattr(
        "reasondb.backends.vision_model.requests.post",
        _fake_post(
            {
                "status": "cache_ready",
                "n_images": 5,
                "n_missing": 5,
                "missing_hashes": ["img1"],
            }
        ),
    )
    backend = KvVisionModel(VISION_MODEL, 0.9, 0.3)
    with pytest.raises(RuntimeError, match=r"5/5"):
        asyncio.run(
            backend.prepare(
                column=_column(),
                cache_dir=Path("/tmp/cache"),
                image_paths=[Path("a.jpg")] * 5,
            )
        )


def test_vision_prepare_tolerates_a_few_generation_errors(monkeypatch):
    """A couple of corrupted images must not block the whole operator's setup."""
    monkeypatch.setattr(
        "reasondb.backends.vision_model.requests.post",
        _fake_post(
            {
                "status": "cache_ready",
                "n_images": 5,
                "n_generation_errors": 1,
                "generation_error_hashes": ["bad_img"],
            }
        ),
    )
    backend = KvVisionModel(VISION_MODEL, 0.9, 0.9)
    asyncio.run(  # must not raise
        backend.prepare(
            column=_column(), cache_dir=Path("/tmp/cache"), image_paths=[Path("a.jpg")] * 5
        )
    )


def test_vision_prepare_ok_when_nothing_missing(monkeypatch):
    monkeypatch.setattr(
        "reasondb.backends.vision_model.requests.post",
        _fake_post({"status": "cache_ready"}),
    )
    backend = KvVisionModel(VISION_MODEL, 0.9, 0.9)
    asyncio.run(  # must not raise
        backend.prepare(
            column=_column(), cache_dir=Path("/tmp/cache"), image_paths=[Path("a.jpg")]
        )
    )
