"""The client KV backends drive cache selection; the servers only validate.

These tests pin the client half of that contract: construction-time validation, the
`model_id` encoding (which keys the precompute cache), and the exact fields put on the
wire. The servers read effective/materialized/vanilla/keep_in_memory as *required* JSON
keys, so a client that stops sending one of them breaks every request.
"""

import asyncio
from pathlib import Path

import pytest

from reasondb.backends.audio_model import KvAudioModel
from reasondb.backends.text_qa import KvTextQABackend
from reasondb.backends.vision_model import KvVisionModel, LlmVisionModel, VisionModel

TEXT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VISION_MODEL = "llava-hf/llava-next-72b-hf"
AUDIO_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

WIRE_FIELDS = {
    "effective_compression_ratio",
    "materialized_compression_ratio",
    "vanilla",
    "keep_in_memory",
}


class _Response:
    """Stand-in for requests.Response covering what the backends read."""

    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _fresh_prepare_memo():
    """prepare() is memoized process-wide, so two tests issuing the same request would
    otherwise have the second one silently skip its POST (see prepare_memo)."""
    from reasondb.backends.prepare_memo import reset_prepare_memo

    reset_prepare_memo()
    yield
    reset_prepare_memo()


@pytest.fixture
def captured_posts(monkeypatch):
    """Capture every requests.post the backends make instead of hitting a server."""
    posts = []

    def fake_post(url, json=None, **kwargs):
        posts.append((url, json))
        return _Response(
            {
                "status": "cache_ready",
                "answers": {},
                "log_odds": {},
                # What a server returns for a keep_in_memory prepare; the disk path
                # ignores them.
                "n_pin_targets": 1,
                "n_resident": 1,
                "pinned_gb": 0.5,
                "pin_budget_gb": 10.0,
            }
        )

    monkeypatch.setattr("reasondb.backends.text_qa.requests.post", fake_post)
    return posts


# ── Construction ────────────────────────────────────────────────────────────


def test_constructor_validates_ratios():
    with pytest.raises(AssertionError):
        KvTextQABackend(TEXT_MODEL, 0.3, 0.9)  # effective < materialized


def test_constructor_validates_vanilla_requires_zero_ratios():
    with pytest.raises(AssertionError):
        KvTextQABackend(TEXT_MODEL, 0.5, 0.5, vanilla=True)


def test_constructor_rejects_vanilla_plus_keep_in_memory():
    """Vanilla runs no pre-computed cache, so there is nothing to hold in RAM."""
    with pytest.raises(AssertionError, match="mutually exclusive"):
        KvTextQABackend(TEXT_MODEL, 0.0, 0.0, vanilla=True, keep_in_memory=True)


def test_constructor_rejects_keep_in_memory_on_an_indexed_cache():
    """Under relative indices the resident artifact would be the *baseline*, and every
    query would still pay a CPU gather plus a GPU rerotate — not what this mode claims."""
    with pytest.raises(AssertionError, match="directly materialized"):
        KvTextQABackend(TEXT_MODEL, 0.8, 0.0, keep_in_memory=True)
    for model, cls in ((VISION_MODEL, KvVisionModel), (AUDIO_MODEL, KvAudioModel)):
        with pytest.raises(AssertionError, match="directly materialized"):
            cls(model, 0.9, 0.5, keep_in_memory=True)


def test_ratios_are_exposed_as_given():
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.3)
    assert backend.effective_compression_ratio == 0.9
    assert backend.materialized_compression_ratio == 0.3
    assert backend.vanilla is False
    assert backend.keep_in_memory is False


def test_no_legacy_compression_ratio_alias():
    """There is no single-ratio alias; callers must name both ratios explicitly."""
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.3)
    assert not hasattr(backend, "compression_ratio")


# ── model_id encoding (keys the precompute cache) ────────────────────────────


@pytest.mark.parametrize(
    "kwargs, expected_suffix",
    [
        (dict(effective_compression_ratio=0.9, materialized_compression_ratio=0.9), "-cr0.9"),
        (dict(effective_compression_ratio=0.9, materialized_compression_ratio=0.3), "-cr0.9-mat0.3"),
        (
            dict(
                effective_compression_ratio=0.0,
                materialized_compression_ratio=0.0,
                vanilla=True,
            ),
            "-cr0.0-vanilla",
        ),
        (
            dict(
                effective_compression_ratio=0.8,
                materialized_compression_ratio=0.8,
                keep_in_memory=True,
            ),
            "-cr0.8-in-memory",
        ),
    ],
)
def test_text_model_id_encoding(kwargs, expected_suffix):
    assert KvTextQABackend(TEXT_MODEL, **kwargs).model_id == TEXT_MODEL + expected_suffix


def test_materialized_only_shows_when_it_differs():
    """effective == materialized is the common case and must not perturb the cache key."""
    assert "mat" not in KvTextQABackend(TEXT_MODEL, 0.9, 0.9).model_id
    assert "mat" in KvTextQABackend(TEXT_MODEL, 0.9, 0.3).model_id


def test_vision_and_audio_model_id_encoding():
    assert (
        KvVisionModel(VISION_MODEL, 0.9, 0.3).model_id == f"{VISION_MODEL}-cr0.9-mat0.3"
    )
    assert (
        KvAudioModel(AUDIO_MODEL, 0.5, 0.5).model_id == f"{AUDIO_MODEL}-cr0.5"
    )
    assert (
        KvVisionModel(VISION_MODEL, 0.9, 0.9, keep_in_memory=True).model_id
        == f"{VISION_MODEL}-cr0.9-in-memory"
    )


def test_in_memory_suffix_comes_last_and_never_with_mat():
    """The identifier grammar is {model}-cr{eff}[-mat{mat}][-vanilla][-in-memory], and
    scripts/analyze_precompute_runtime.py's suffix regex depends on that order. -mat can
    never co-occur, since keep_in_memory requires effective == materialized."""
    model_id = KvTextQABackend(TEXT_MODEL, 0.8, 0.8, keep_in_memory=True).model_id
    assert model_id.endswith("-in-memory")
    assert "-mat" not in model_id


# ── Wire protocol ────────────────────────────────────────────────────────────


def test_prepare_sends_all_three_fields(captured_posts):
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.3)
    asyncio.run(
        backend.prepare(column=_column(), cache_dir=Path("/tmp/cache"), texts=["a", "b"])
    )

    (url, payload), = captured_posts
    assert url.endswith("/prepare_caches")
    assert WIRE_FIELDS <= payload.keys()
    assert payload["effective_compression_ratio"] == 0.9
    assert payload["materialized_compression_ratio"] == 0.3
    assert payload["vanilla"] is False


def test_no_legacy_compression_ratio_key_on_the_wire(captured_posts):
    """The servers reject a bare `compression_ratio`; don't send one."""
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.3)
    asyncio.run(
        backend.prepare(column=_column(), cache_dir=Path("/tmp/cache"), texts=["a"])
    )

    (_, payload), = captured_posts
    assert "compression_ratio" not in payload


def test_disk_backend_sends_keep_in_memory_false(captured_posts):
    """Present and False, not absent: the servers read it as a required key."""
    backend = KvTextQABackend(TEXT_MODEL, 0.9, 0.3)
    asyncio.run(
        backend.prepare(column=_column(), cache_dir=Path("/tmp/cache"), texts=["a"])
    )

    (_, payload), = captured_posts
    assert payload["keep_in_memory"] is False


def test_in_memory_backend_sends_keep_in_memory_true(captured_posts):
    backend = KvTextQABackend(TEXT_MODEL, 0.8, 0.8, keep_in_memory=True)
    asyncio.run(
        backend.prepare(column=_column(), cache_dir=Path("/tmp/cache"), texts=["a"])
    )

    (_, payload), = captured_posts
    assert payload["keep_in_memory"] is True


def test_vanilla_backend_sends_vanilla_true(captured_posts):
    backend = KvTextQABackend(TEXT_MODEL, 0.0, 0.0, vanilla=True)
    asyncio.run(
        backend.prepare(column=_column(), cache_dir=Path("/tmp/cache"), texts=["a"])
    )

    (_, payload), = captured_posts
    assert payload["vanilla"] is True
    assert payload["effective_compression_ratio"] == 0.0
    assert payload["materialized_compression_ratio"] == 0.0


# ── Non-KV vision models ─────────────────────────────────────────────────────


def test_non_kv_vision_models_report_no_compression():
    """`VisionModel.invoke` records these for every subclass, KV-backed or not."""
    assert VisionModel.effective_compression_ratio is None
    assert VisionModel.materialized_compression_ratio is None
    assert VisionModel.vanilla is False
    assert VisionModel.keep_in_memory is False
    assert LlmVisionModel.effective_compression_ratio is None
    assert LlmVisionModel.vanilla is False
    assert LlmVisionModel.keep_in_memory is False


def _column():
    from reasondb.database.indentifier import ConcreteColumnIdentifier

    return ConcreteColumnIdentifier(name="t.c")
