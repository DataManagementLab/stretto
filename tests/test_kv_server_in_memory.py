"""Server-side handling of ``keep_in_memory``: what is pinned, and what is refused.

Three properties, in rough order of how expensive getting them wrong would be:

1. **Nothing is pinned that the serve path would not have loaded.** A cache with a
   recorded generation error has no file, and the servers already skip (image) or assert
   (text) on it. Pinning that path would turn a tolerated data issue into a load failure
   — the one way this mode could stop an already-running benchmark.
2. **A missing pin is never repaired by a disk read.** The whole claim of an -in-memory
   operator is where its bytes came from; a silent fallback makes a broken run merely a
   slow one.
3. **The pinned object is never mutated.** It is shared by every query over that row, so
   the per-query GPU copy has to build a new cache rather than write back into it.
"""

import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

# The server modules import kvpress at module scope for the compression presses; the
# control flow here touches none of it. Same stub as tests/test_kv_join_materialized_cr.py.
try:
    import kvpress  # noqa: F401

    kvpress.KeyRerotationPress
    kvpress.ExpectedAttentionPress
    kvpress.KVzipPress
    kvpress.FinchPress
except (ImportError, AttributeError):
    _stub = types.ModuleType("kvpress")
    for _name in (
        "KeyRerotationPress",
        "ExpectedAttentionPress",
        "KVzipPress",
        "FinchPress",
    ):
        setattr(_stub, _name, object)
    sys.modules["kvpress"] = _stub

from reasondb.backends import kv_cache_base  # noqa: E402
from reasondb.backends.kv_cache_audio_qa_server import (  # noqa: E402
    KvAudioQaModelWrapper,
)
from reasondb.backends.kv_cache_base import (  # noqa: E402
    PinnedKVStore,
    PinnedKVUnavailable,
)
from reasondb.backends.kv_cache_image_qa_server import (  # noqa: E402
    KvImageQaModelWrapper,
)
from reasondb.backends.kv_cache_text_qa_server import (  # noqa: E402
    KvTextQaModelWrapper,
)


def _bare(cls, compression_ratios=(0.0, 0.5, 0.8, 0.9, 0.99)):
    obj = object.__new__(cls)
    obj.compression_ratios = compression_ratios
    obj.compression_ratio_to_batch_size = {cr: None for cr in compression_ratios}
    obj.model_name = "test-model"
    return obj


@pytest.fixture
def store(monkeypatch):
    """A generously-sized pin store, swapped in for the process-wide singleton."""
    fresh = PinnedKVStore(1.0)
    monkeypatch.setattr(kv_cache_base, "PINNED_KV_STORE", fresh)
    for module in (
        "reasondb.backends.kv_cache_text_qa_server",
        "reasondb.backends.kv_cache_image_qa_server",
    ):
        monkeypatch.setattr(sys.modules[module], "PINNED_KV_STORE", fresh)
    return fresh


# ── Validation ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cls", [KvTextQaModelWrapper, KvImageQaModelWrapper]
)
def test_servers_reject_vanilla_plus_keep_in_memory(cls):
    with pytest.raises(AssertionError, match="mutually exclusive"):
        _bare(cls)._validate_client_crs(0.0, 0.0, True, True)


@pytest.mark.parametrize(
    "cls", [KvTextQaModelWrapper, KvImageQaModelWrapper]
)
def test_servers_reject_keep_in_memory_on_an_indexed_cache(cls):
    with pytest.raises(AssertionError, match="directly materialized"):
        _bare(cls)._validate_client_crs(0.9, 0.5, False, True)


def test_audio_server_rejects_keep_in_memory_outright():
    """No in-memory path there, as with vanilla and relative indices."""
    wrapper = _bare(KvAudioQaModelWrapper, (0.0, 0.5, 0.9))
    with pytest.raises(AssertionError, match="not supported by the audio server"):
        wrapper._validate_client_crs(0.5, 0.5, False, True)
    wrapper._validate_client_crs(0.5, 0.5, False, False)  # the disk path still works


# ── Pinning at prepare() ─────────────────────────────────────────────────────


def _write_cache(path, elements=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros(elements), path)
    return str(path)


def test_pin_prepared_caches_reports_targets_and_residents(store, tmp_path):
    paths = [_write_cache(tmp_path / f"cache_entry_{i}.pt") for i in range(3)]
    result = _bare(KvTextQaModelWrapper)._pin_prepared_caches(
        paths, column_name="t.review", compression_ratio=0.8
    )
    assert result["n_pin_targets"] == 3
    assert result["n_resident"] == 3
    assert result["pin_budget_gb"] == pytest.approx(1.0)
    assert store.n_pinned() == 3


def test_duplicate_items_pin_one_entry(store, tmp_path):
    """A column with a repeated text hashes to one cache file.

    This is why the client asserts on the server's own n_pin_targets rather than deriving
    an expected count from its row count — the two are not the same number.
    """
    path = _write_cache(tmp_path / "cache_entry_dup.pt")
    result = _bare(KvTextQaModelWrapper)._pin_prepared_caches(
        [path, path, path], column_name="t.review", compression_ratio=0.8
    )
    assert result["n_pin_targets"] == 1
    assert result["n_resident"] == 1
    assert store.n_pinned() == 1


def test_pinning_more_than_the_budget_raises(monkeypatch, tmp_path):
    tiny = PinnedKVStore(1e-9 * 4)  # 4 bytes: one element, not two
    monkeypatch.setattr(kv_cache_base, "PINNED_KV_STORE", tiny)
    paths = [_write_cache(tmp_path / f"cache_entry_{i}.pt", elements=4) for i in range(2)]
    with pytest.raises(PinnedKVUnavailable, match="KV_CACHE_PIN_GB"):
        _bare(KvTextQaModelWrapper)._pin_prepared_caches(
            paths, column_name="t.review", compression_ratio=0.8
        )


def test_an_unconfigured_server_refuses_before_touching_the_filesystem():
    unconfigured = PinnedKVStore(0)
    with pytest.raises(PinnedKVUnavailable, match="--kv-cache-pin-gb"):
        unconfigured.require_configured("column 't.review' on test-model (cr=0.8)")


# ── Generation errors: behaviour must not depend on the flag ─────────────────


def test_a_cache_with_a_recorded_generation_error_is_not_a_pin_target(store, tmp_path):
    """Errored items degrade per item exactly as they do for a disk-served operator.

    ``prepare_caches`` filters its pin targets with exactly this rule (the file exists and
    is not in ``errors``), so an errored item never reaches the store, and the serve path
    skips or asserts on it exactly as it does for a disk-served operator.
    """
    good = _write_cache(tmp_path / "cache_entry_good.pt")
    errored = str(tmp_path / "cache_entry_errored.pt")  # never written
    errors = {errored: "corrupt input"}

    usable = [p for p in (good, errored) if p not in errors and os.path.exists(p)]

    result = _bare(KvTextQaModelWrapper)._pin_prepared_caches(
        usable, column_name="t.review", compression_ratio=0.8
    )
    assert result["n_pin_targets"] == 1
    assert store.contains(good)
    assert not store.contains(errored)


# ── The pinned object must survive being served ──────────────────────────────


class _Layer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _Cache:
    """The transformers 5.x layout the server's accessors take their first branch on."""

    def __init__(self, n_layers=2):
        self.layers = [
            _Layer(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4))
            for _ in range(n_layers)
        ]


def test_serving_a_pinned_cache_does_not_consume_it():
    """The per-request copy must be built out of place.

    The serve loop releases each layer of the caches it batched (`_cache_set_kv(c, li,
    None, None)`) as soon as that layer is concatenated. If the routed cache were the
    pinned object itself (layers scattered in place), the *second* query over the same
    row would find its layers set to None, and the pin budget would be accounting for CPU
    bytes that had become GPU ones.
    """
    from reasondb.backends.kv_cache_text_qa_server import (
        _cache_kv,
        _cache_num_layers,
        _cache_set_kv,
    )
    from transformers import DynamicCache

    pinned = _Cache()
    devices = [torch.device("cpu")] * _cache_num_layers(pinned)

    # Exactly what the route's Path B does, then what the batching loop does after it.
    routed = DynamicCache()
    for li in range(_cache_num_layers(pinned)):
        k, v = _cache_kv(pinned, li)
        routed.update(k.to(devices[li]), v.to(devices[li]), li)
    for li in range(_cache_num_layers(routed)):
        _cache_set_kv(routed, li, None, None)

    for li in range(_cache_num_layers(pinned)):
        keys, values = _cache_kv(pinned, li)
        assert keys is not None and values is not None, (
            f"layer {li} of the pinned cache was released with the request's copy"
        )
        assert keys.device.type == "cpu"


# --- POST /release_pinned_kv ----------------------------------------------------------
# The pin store has no eviction, so a server outliving one benchmark would carry its
# column into the next dataset's budget. This endpoint is the only way anything leaves.


_SERVER_MODULES = [
    "reasondb.backends.kv_cache_text_qa_server",
    "reasondb.backends.kv_cache_image_qa_server",
]


def _client(module_name):
    return sys.modules[module_name].app.test_client()


@pytest.mark.parametrize("module_name", _SERVER_MODULES)
def test_release_endpoint_drops_every_pin_and_takes_no_body(module_name, store):
    """No body on purpose: `requests.post(url)` sends an empty one, which the
    `request.get_json(force=True)` every other handler in these files uses would raise on.
    """
    store.pin("a.pt", lambda: torch.zeros(1000))
    store.pin("b.pt", lambda: torch.zeros(1000))

    response = _client(module_name).post("/release_pinned_kv")

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "released"
    assert body["n_released"] == 2
    # After, not before: a client checks the server is actually empty rather than trusting
    # the count it was handed.
    assert body["n_pinned"] == 0
    assert body["pinned_gb"] == 0.0
    assert store.n_pinned() == 0


@pytest.mark.parametrize("module_name", _SERVER_MODULES)
def test_releasing_an_empty_server_is_a_200(module_name, store):
    """A worker releases on its first job too, when there is normally nothing held."""
    response = _client(module_name).post("/release_pinned_kv")

    assert response.status_code == 200
    assert response.get_json()["n_released"] == 0


def test_the_audio_server_has_no_release_endpoint():
    """It asserts `not keep_in_memory` and never touches the store, so the route would be
    permanently dead code. Pinned so nobody adds it for symmetry — and so the client's port
    filter, which never POSTs there, stays right."""
    from reasondb.backends import kv_cache_audio_qa_server

    assert kv_cache_audio_qa_server.app.test_client().post(
        "/release_pinned_kv"
    ).status_code == 404


def test_a_released_column_is_a_miss_on_the_lookup_the_serve_path_uses(store):
    """Ties the endpoint to the operator-visible consequence: the serve path's only
    accessor is `PINNED_KV_STORE.get`, and after a release it misses. That `get` never
    repairs a miss with a disk read is pinned by tests/test_pinned_kv_store.py
    (test_get_never_loads_and_reports_a_miss, test_a_released_path_is_a_miss_not_a_disk_read);
    the serve loops turn that miss into a raised PinnedKVUnavailable.
    """
    store.pin("a.pt", lambda: torch.zeros(1000))
    assert store.get("a.pt") is not None

    _client(_SERVER_MODULES[0]).post("/release_pinned_kv")

    assert store.get("a.pt") is None
