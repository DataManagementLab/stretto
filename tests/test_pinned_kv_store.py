"""The KV pin store: what an ``-in-memory`` operator's promise actually rests on.

Unlike an opportunistic LRU cache, the pin store retains nothing unless it was asked for
by name, never evicts, and never quietly repairs a miss with a disk read; every test here
pins one of these properties. Those are the
properties that make "this operator was served from RAM" a fact rather than a hope.
"""

import threading

import pytest

torch = pytest.importorskip("torch")

from reasondb.backends.kv_cache_base import (  # noqa: E402
    PinnedKVStore,
    PinnedKVUnavailable,
    _obj_nbytes,
)

# 1 MB of float32 = 262144 elements; a tensor of N elements is 4N bytes.
_MB = 1e-3  # in the GB units configure() takes


def _tensor(mb: float):
    return torch.zeros(int(mb * 1e6 / 4), dtype=torch.float32)


def _store(budget_gb: float) -> PinnedKVStore:
    return PinnedKVStore(budget_gb)


def test_pin_then_get_returns_the_same_object_and_counts_a_hit():
    store = _store(10 * _MB)
    obj = _tensor(1)
    store.pin("a.pt", lambda: obj)
    assert store.get("a.pt") is obj
    hits, misses, gb = store.stats()
    assert (hits, misses) == (1, 0)
    assert gb == pytest.approx(_obj_nbytes(obj) / 1e9)


def test_pin_is_idempotent():
    """One backend serves four operators and prepare() runs per query."""
    store = _store(10 * _MB)
    calls = []

    def loader():
        calls.append(1)
        return _tensor(1)

    store.pin("a.pt", loader)
    store.pin("a.pt", loader)
    assert len(calls) == 1
    assert store.n_pinned() == 1


def test_get_never_loads_and_reports_a_miss():
    """The serve path must not repair a missing pin by reading the file.

    ``get`` takes no loader at all — that is the design — so the only thing to assert is
    that a miss is reported rather than papered over.
    """
    store = _store(10 * _MB)
    assert store.get("absent.pt") is None
    assert store.stats()[1] == 1


def test_contains_does_not_move_the_hit_counters():
    """prepare()-time accounting must not pollute the serve-time measurement."""
    store = _store(10 * _MB)
    store.pin("a.pt", lambda: _tensor(1))
    assert store.contains("a.pt") is True
    assert store.contains("b.pt") is False
    assert store.stats()[:2] == (0, 0)


def test_overflow_raises_and_retains_nothing():
    """No eviction to fall back on: a budget too small is a loud configuration error."""
    store = _store(2 * _MB)
    store.pin("small.pt", lambda: _tensor(1))
    with pytest.raises(PinnedKVUnavailable) as excinfo:
        store.pin("big.pt", lambda: _tensor(5))
    assert "KV_CACHE_PIN_GB" in str(excinfo.value)
    assert store.n_pinned() == 1
    assert store.get("big.pt") is None
    # The one already held is untouched — an LRU would have evicted it to make room.
    assert store.get("small.pt") is not None


def test_a_pinned_entry_is_never_evicted_by_later_pressure():
    store = _store(50 * _MB)
    store.pin("first.pt", lambda: _tensor(1))
    for i in range(20):
        store.pin(f"later{i}.pt", lambda: _tensor(1))
    assert store.get("first.pt") is not None


# --- release_all ---------------------------------------------------------------------
# The one way an entry leaves the store. Still not eviction: nothing above changes, because
# nothing is ever dropped to make room — a client gives a whole dataset's pins up at once.


def test_release_all_empties_the_store_and_reports_what_it_freed():
    store = _store(10 * _MB)
    store.pin("a.pt", lambda: _tensor(1))
    store.pin("b.pt", lambda: _tensor(2))
    held_gb = store.stats()[2]

    n_released, released_gb = store.release_all()

    assert n_released == 2
    assert released_gb == pytest.approx(held_gb)
    assert store.n_pinned() == 0
    assert store.stats()[2] == 0.0


def test_release_all_is_idempotent():
    """A worker releases on its first job too, when there is usually nothing held."""
    store = _store(10 * _MB)
    store.pin("a.pt", lambda: _tensor(1))
    store.release_all()
    assert store.release_all() == (0, 0.0)


def test_release_all_does_not_reset_the_hit_and_miss_counters():
    """The serve paths bracket ONE request with two stats() reads and act on the delta —
    and the miss side is a truthiness check. Zeroing the counters here would make that
    delta negative (raising with a nonsense count) or falsely zero (silently disabling the
    residency guard that exists to catch exactly this bug). Do not "tidy" this away.
    """
    store = _store(10 * _MB)
    store.pin("a.pt", lambda: _tensor(1))
    store.get("a.pt")
    store.get("absent.pt")
    assert store.stats()[:2] == (1, 1)

    store.release_all()

    assert store.stats()[:2] == (1, 1)


def test_a_released_path_is_a_miss_not_a_disk_read():
    """The sibling of test_get_never_loads_and_reports_a_miss, and what makes a release
    that should not have happened loud rather than silent."""
    store = _store(10 * _MB)
    store.pin("a.pt", lambda: _tensor(1))
    store.release_all()

    assert store.get("a.pt") is None
    assert store.stats()[1] == 1


def test_an_object_handed_out_before_a_release_stays_alive():
    """`get` returns the object itself, so a request already holding one is unaffected by a
    concurrent release — it keeps it alive by refcount."""
    store = _store(10 * _MB)
    obj = _tensor(1)
    store.pin("a.pt", lambda: obj)
    in_flight = store.get("a.pt")

    store.release_all()

    assert in_flight is obj
    assert float(in_flight[0]) == 0.0  # still readable, not freed underneath us


def test_the_budget_survives_a_release_and_the_space_is_reusable():
    """The point of releasing: the next dataset's column fits where the last one's did."""
    store = _store(3 * _MB)
    store.pin("first.pt", lambda: _tensor(2))
    with pytest.raises(PinnedKVUnavailable):
        store.pin("second.pt", lambda: _tensor(2))

    store.release_all()

    assert store.budget_bytes == int(3 * _MB * 1e9)
    store.pin("second.pt", lambda: _tensor(2))  # would still overflow if nothing was freed
    assert store.n_pinned() == 1


def test_a_release_racing_a_pin_leaves_consistent_accounting():
    store = _store(50 * _MB)
    store.pin("held.pt", lambda: _tensor(1))
    errors = []

    def pin_many():
        try:
            for i in range(20):
                store.pin(f"racer{i}.pt", lambda: _tensor(1))
        except Exception as exc:  # noqa: BLE001 - the assertion is that none escapes
            errors.append(exc)

    racer = threading.Thread(target=pin_many)
    racer.start()
    store.release_all()
    racer.join(timeout=5)

    assert not errors
    # _pinned_bytes must still describe exactly what survived the race, whichever pins
    # landed on which side of it.
    assert store.stats()[2] == pytest.approx(store.n_pinned() * _obj_nbytes(_tensor(1)) / 1e9)


def test_loaders_run_outside_the_lock():
    """Two threads pinning different files must not serialize on I/O."""
    store = _store(10 * _MB)
    entered = threading.Event()
    release = threading.Event()

    def blocking_loader():
        entered.set()
        assert release.wait(timeout=5), "second pin never completed"
        return _tensor(1)

    slow = threading.Thread(target=lambda: store.pin("slow.pt", blocking_loader))
    slow.start()
    assert entered.wait(timeout=5)
    store.pin("fast.pt", lambda: _tensor(1))  # would deadlock if the lock were held
    release.set()
    slow.join(timeout=5)
    assert store.n_pinned() == 2


def test_require_configured_names_both_ways_to_set_the_budget():
    store = _store(0)
    assert store.configured is False
    with pytest.raises(PinnedKVUnavailable) as excinfo:
        store.require_configured("column 'text' on some-model")
    message = str(excinfo.value)
    assert "column 'text' on some-model" in message
    assert "KV_CACHE_PIN_GB" in message and "--kv-cache-pin-gb" in message
    _store(1 * _MB).require_configured("ok")  # configured → silent


def test_configure_rejects_a_non_empty_store():
    store = _store(10 * _MB)
    store.pin("a.pt", lambda: _tensor(1))
    with pytest.raises(AssertionError, match="already"):
        store.configure(20 * _MB)


def test_configure_rejects_a_negative_budget():
    with pytest.raises(AssertionError, match=">= 0"):
        PinnedKVStore().configure(-1)


def test_configure_rejects_the_removed_resident_env_var(monkeypatch):
    """A stale export in a job script must fail, not silently do nothing.

    KV_CACHE_RAM_RESIDENT_GB configured an opportunistic LRU cache that no longer exists;
    a server started with it set would otherwise cache nothing and give no clue why.
    """
    monkeypatch.setenv("KV_CACHE_RAM_RESIDENT_GB", "430")
    with pytest.raises(AssertionError) as excinfo:
        PinnedKVStore().configure(10 * _MB)
    assert "KV_CACHE_PIN_GB" in str(excinfo.value)


def test_configure_reads_the_env_var_when_no_value_is_given(monkeypatch):
    monkeypatch.delenv("KV_CACHE_RAM_RESIDENT_GB", raising=False)
    monkeypatch.setenv("KV_CACHE_PIN_GB", "2")
    store = PinnedKVStore()
    store.configure(None)
    assert store.budget_bytes == 2_000_000_000


def test_configure_rejects_a_non_numeric_env_var(monkeypatch):
    monkeypatch.delenv("KV_CACHE_RAM_RESIDENT_GB", raising=False)
    monkeypatch.setenv("KV_CACHE_PIN_GB", "lots")
    with pytest.raises(AssertionError, match="must be a number"):
        PinnedKVStore().configure(None)


def test_the_opportunistic_store_is_gone():
    """The pin store is the only KV residency mechanism; no LRU store may exist beside it."""
    import reasondb.backends.kv_cache_base as base

    for name in ("ResidentKVStore", "RESIDENT_KV_STORE"):
        assert not hasattr(base, name), f"{name} should have been removed"
    assert not hasattr(PinnedKVStore, "get_or_load")


class _Layer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _Layers:
    """transformers 5.x-style cache."""

    def __init__(self, layers):
        self.layers = layers


class _KeyCache:
    """transformers 4.x-style cache."""

    def __init__(self, key_cache, value_cache):
        self.key_cache = key_cache
        self.value_cache = value_cache


@pytest.mark.parametrize(
    "obj, expected_elements",
    [
        (torch.zeros(10), 10),
        ([torch.zeros(3), torch.zeros(4)], 7),
        ({"k": torch.zeros(2), "v": torch.zeros(5)}, 7),
        (_Layers([_Layer(torch.zeros(2), torch.zeros(3))]), 5),
        (_KeyCache([torch.zeros(4)], [torch.zeros(6)]), 10),
        (object(), 0),
    ],
)
def test_obj_nbytes_covers_every_layout_it_claims_to(obj, expected_elements):
    """The pin accounting is only as honest as this sizing."""
    assert _obj_nbytes(obj) == expected_elements * 4
