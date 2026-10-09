"""--precompute must persist which KV cache produced each recorded response.

Without the ratios, a precomputed parquet/json is only interpretable via the `model_id`
bucket, which encodes the effective ratio but nothing about the materialized cache it
was derived from.
"""

import pytest

from reasondb.backends.simulate_store import SimulateStore


def _record_text(store, **overrides):
    kwargs = dict(
        model_id="m-cr0.3-mat0.9",
        question="q",
        context="c",
        response="Yes",
        log_odds=0.5,
        runtime=1.0,
        effective_compression_ratio=0.3,
        materialized_compression_ratio=0.9,
        vanilla=False,
    )
    kwargs.update(overrides)
    store.record_text_qa(**kwargs)
    return kwargs


def test_text_qa_persists_ratios():
    store = SimulateStore()
    _record_text(store)

    record = store._text_qa["m-cr0.3-mat0.9"][store._hash("q-c")]
    assert record["effective_compression_ratio"] == 0.3
    assert record["materialized_compression_ratio"] == 0.9
    assert record["vanilla"] is False


def test_vision_persists_ratios():
    store = SimulateStore()
    store.record_vision(
        "m-cr0.0-vanilla",
        "q",
        "/img/a.jpg",
        "Yes",
        0.5,
        1.0,
        0.0,
        effective_compression_ratio=0.0,
        materialized_compression_ratio=0.0,
        vanilla=True,
    )

    record = store._vision["m-cr0.0-vanilla"][store._hash("q-/img/a.jpg")]
    assert record["effective_compression_ratio"] == 0.0
    assert record["vanilla"] is True


def test_non_kv_vision_records_none_ratios():
    """LLM/local vision models use no pre-computed cache; None is the honest value."""
    store = SimulateStore()
    store.record_vision(
        "gpt-4o",
        "q",
        "/img/a.jpg",
        "Yes",
        0.5,
        1.0,
        0.01,
        effective_compression_ratio=None,
        materialized_compression_ratio=None,
        vanilla=False,
    )

    record = store._vision["gpt-4o"][store._hash("q-/img/a.jpg")]
    assert record["effective_compression_ratio"] is None
    assert record["materialized_compression_ratio"] is None


def test_lookup_shape_is_unchanged_by_the_extra_fields():
    """Callers unpack these tuples positionally."""
    store = SimulateStore()
    _record_text(store)
    assert store.lookup_text_qa("m-cr0.3-mat0.9", "q", "c") == ("Yes", 0.5, 1.0)


def test_ratios_survive_save_and_load(tmp_path):
    store = SimulateStore()
    _record_text(store)
    store.save(tmp_path / "precompute")

    reloaded = SimulateStore.load(tmp_path / "precompute")

    record = reloaded._text_qa["m-cr0.3-mat0.9"][reloaded._hash("q-c")]
    assert record["effective_compression_ratio"] == 0.3
    assert record["materialized_compression_ratio"] == 0.9
    assert record["vanilla"] is False


@pytest.mark.parametrize(
    "method, args",
    [
        ("record_text_qa", ("m", "q", "c", "Yes", 0.5, 1.0)),
        ("record_vision", ("m", "q", "/img/a.jpg", "Yes", 0.5, 1.0, 0.0)),
    ],
)
def test_ratio_arguments_are_required(method, args):
    """Recording without them would silently produce an uninterpretable cache."""
    store = SimulateStore()
    with pytest.raises(TypeError):
        getattr(store, method)(*args)


def test_in_memory_and_disk_operators_get_distinct_buckets():
    """`-in-memory` is a different bucket, and that is the point.

    The two operators answer identically — same model, same cache, same tokens — so
    aliasing them would look harmless. It is not: the store records `runtime`, and
    `--simulate` replays it through the simulated clock. A shared bucket would replay the
    disk latency for the RAM-served operator and erase the only difference the mode
    exists to make.
    """
    store = SimulateStore()
    _record_text(
        store,
        model_id="m-cr0.8",
        effective_compression_ratio=0.8,
        materialized_compression_ratio=0.8,
        runtime=4.0,
    )
    _record_text(
        store,
        model_id="m-cr0.8-in-memory",
        effective_compression_ratio=0.8,
        materialized_compression_ratio=0.8,
        runtime=1.0,
    )

    assert set(store._text_qa) == {"m-cr0.8", "m-cr0.8-in-memory"}
    assert store.lookup_text_qa("m-cr0.8", "q", "c")[2] == 4.0
    assert store.lookup_text_qa("m-cr0.8-in-memory", "q", "c")[2] == 1.0


def test_a_store_without_the_in_memory_bucket_misses_rather_than_falling_back():
    """The loud half of that decision: no aliasing to the disk operator.

    A store recorded without this operator lacks the bucket, so a --simulate replay of
    the default suite fails on its first lookup instead of quietly replaying a
    different operator's numbers. One precompute top-up pass fills it in.
    """
    store = SimulateStore()
    _record_text(store, model_id="m-cr0.8")
    assert store.lookup_text_qa("m-cr0.8-in-memory", "q", "c") is None
