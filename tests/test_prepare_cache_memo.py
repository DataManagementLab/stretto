"""A KV cache's existence is checked once per distinct request, not once per operator.

`PlanConfigurator.prepare` walks every operator in the toolbox and `Executor.prepare`
runs per query, while `build_toolbox` gives one backend to four operators (QA filter, QA
extract, and both join predicates). So the server's `/prepare_caches` -- which hashes
every text/image in the column and stats its cache file -- would otherwise be scanned
four times per spec per query over a byte-identical payload, inside the measured
`end_to_end` span.

`reasondb/backends/prepare_memo.py` fingerprints what the request is a function of and
issues it once. What these tests pin is the two halves of "once": identical requests
collapse, and anything that would check a *different* cache (other ratio, other column,
other cache dir, changed contents) still goes out. Plus the failure clause -- a request
that did not come back clean is not remembered, so the next operator retries it rather
than inheriting a cache that was never verified.
"""

import asyncio
from pathlib import Path

import pytest

from reasondb.backends import prepare_memo
from reasondb.backends.text_qa import KvTextQABackend
from reasondb.backends.vision_model import KvVisionModel
from reasondb.database.indentifier import ConcreteColumnIdentifier

TEXT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VISION_MODEL = "llava-hf/llama3-llava-next-8b-hf"


class _RecordingPost:
    """Stands in for requests.post: records payloads, answers "cache_ready"."""

    def __init__(self, response=None):
        self.calls = []
        self._response = response or {
            "status": "cache_ready",
            "n_missing": 0,
            # What a server returns for a keep_in_memory prepare; a disk prepare's
            # response has these absent and the client never reads them.
            "n_pin_targets": 1,
            "n_resident": 1,
        }

    def __call__(self, url, json=None, **kwargs):
        self.calls.append((url, json))
        return _Response(self._response)


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _fresh_memo():
    prepare_memo.reset_prepare_memo()
    yield
    prepare_memo.reset_prepare_memo()


@pytest.fixture
def post(monkeypatch):
    recorder = _RecordingPost()
    monkeypatch.setattr("reasondb.backends.text_qa.requests.post", recorder)
    monkeypatch.setattr("reasondb.backends.vision_model.requests.post", recorder)
    return recorder


def _text_backend(effective=0.5, materialized=0.5, keep_in_memory=False):
    return KvTextQABackend(
        TEXT_MODEL,
        effective_compression_ratio=effective,
        materialized_compression_ratio=materialized,
        keep_in_memory=keep_in_memory,
    )


def _prepare_text(backend, texts, column="t.review", cache_dir="/caches/movie"):
    asyncio.run(
        backend.prepare(
            column=ConcreteColumnIdentifier(column),
            cache_dir=Path(cache_dir),
            texts=texts,
        )
    )


def test_same_request_from_several_backends_checks_once(post):
    """The four operators of one spec may share a backend or use separate ones.
    Either way the request is the same, so it goes out once."""
    texts = ["a review", "another review"]
    for _ in range(4):
        _prepare_text(_text_backend(), texts)
    assert len(post.calls) == 1


def test_repeated_queries_do_not_re_check(post):
    """`Executor.prepare` runs per query over an unchanged column."""
    backend = _text_backend()
    for _ in range(3):
        _prepare_text(backend, ["a review"])
    assert len(post.calls) == 1


@pytest.mark.parametrize(
    "second",
    [
        pytest.param(dict(texts=["a review", "a different review"]), id="contents"),
        pytest.param(dict(column="t.summary"), id="column"),
        pytest.param(dict(cache_dir="/caches/email"), id="cache_dir"),
    ],
)
def test_a_different_cache_is_still_checked(post, second):
    """Only an identical request collapses: each of these names other cache files."""
    backend = _text_backend()
    first = dict(texts=["a review"], column="t.review", cache_dir="/caches/movie")
    _prepare_text(backend, **first)
    _prepare_text(backend, **{**first, **second})
    assert len(post.calls) == 2


def test_each_compression_ratio_is_checked_separately(post):
    """Two ratios are two different sets of files on disk, not a repeat -- and
    under indexing they share a materialized baseline but not an index dir."""
    texts = ["a review"]
    _prepare_text(_text_backend(effective=0.5, materialized=0.5), texts)
    _prepare_text(_text_backend(effective=0.8, materialized=0.8), texts)
    _prepare_text(_text_backend(effective=0.8, materialized=0.5), texts)
    assert len(post.calls) == 3


def test_in_memory_and_disk_operators_do_not_share_a_fingerprint(post):
    """The one collision that would fail *silently*.

    A disk backend and an -in-memory backend over the same column agree on every other
    field the fingerprint hashes — same server, same column, same cache dir, same ratios,
    same texts. If the flag were left out, whichever prepared first would suppress the
    other's request; and when the disk one wins, the in-memory operator's caches are
    never loaded into the server's RAM and it serves... nothing it can find. The run does
    not crash, it is just no longer the run anyone asked for.
    """
    texts = ["a review"]
    _prepare_text(_text_backend(effective=0.8, materialized=0.8), texts)
    _prepare_text(
        _text_backend(effective=0.8, materialized=0.8, keep_in_memory=True), texts
    )
    assert len(post.calls) == 2
    assert [call[1]["keep_in_memory"] for call in post.calls] == [False, True]


def test_repeated_in_memory_prepares_still_check_once(post):
    """Pinning is idempotent server-side, but the request should not repeat either."""
    backend = _text_backend(effective=0.8, materialized=0.8, keep_in_memory=True)
    for _ in range(4):
        _prepare_text(backend, ["a review"])
    assert len(post.calls) == 1


def test_two_modalities_do_not_share_a_fingerprint(post):
    """Same column name, same ratios, different server and cache tree."""
    column = ConcreteColumnIdentifier("t.content")
    asyncio.run(
        _text_backend().prepare(
            column=column, cache_dir=Path("/caches/ecomm"), texts=["x"]
        )
    )
    asyncio.run(
        KvVisionModel(
            VISION_MODEL,
            effective_compression_ratio=0.5,
            materialized_compression_ratio=0.5,
        ).prepare(
            column=column, cache_dir=Path("/caches/ecomm"), image_paths=[Path("x")]
        )
    )
    assert len(post.calls) == 2


def test_a_failed_check_is_retried(monkeypatch):
    """A prepare that reported missing caches raises -- and is not remembered, so
    the next operator asks again instead of assuming the column is ready."""
    recorder = _RecordingPost(
        {"status": "cache_ready", "n_missing": 1, "missing_hashes": ["deadbeef"]}
    )
    monkeypatch.setattr("reasondb.backends.text_qa.requests.post", recorder)
    backend = _text_backend()
    for _ in range(2):
        with pytest.raises(RuntimeError):
            _prepare_text(backend, ["a review"])
    assert len(recorder.calls) == 2
