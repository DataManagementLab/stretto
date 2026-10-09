"""Every KV server response must carry a uniform, complete `stats` block.

`reasondb.backends.inference_stats` is the single definition both halves of the wire
protocol are built from: `build_inference_stats` on the server, `parse_inference_stats`
on the client. If either drifts from `STATS_KEYS` a client silently loses a metric (dict
`.get` never raises) or, worse, the two disagree about which keys are always present and
a consumer has to guess whether a missing key means "None" or "not implemented yet".
These tests pin the frozen key set across every server/path combination and the
assertions that catch a server or client falling out of sync.
"""

import pytest

from reasondb.backends.inference_stats import (
    PATHS,
    SERVERS,
    STATS_KEYS,
    build_inference_stats,
    gpu_entry,
    parse_inference_stats,
)


def _minimal(**overrides):
    kwargs = dict(
        server="kv_text_qa",
        path="kv",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        n_items=4,
        server_elapsed_s=1.2,
    )
    kwargs.update(overrides)
    return build_inference_stats(**kwargs)


@pytest.mark.parametrize("server", SERVERS)
@pytest.mark.parametrize("path", PATHS)
def test_every_server_path_combination_produces_the_full_key_set(server, path):
    stats = _minimal(server=server, path=path)
    assert set(stats) == STATS_KEYS


def test_inapplicable_fields_are_none_not_absent():
    """A field that doesn't apply on a path is null, never dropped."""
    stats = _minimal(path="direct")
    assert stats["cache_load_s"] is None
    assert stats["pinned_hits"] is None
    assert "cache_load_s" in stats  # present, just null


def test_no_cuda_yields_empty_gpu_list():
    stats = _minimal(gpu=None)
    assert stats["gpu"] == []


def test_gpu_entries_are_validated_against_gpu_keys():
    entry = gpu_entry(index=0, peak_allocated_gb=4.0, free_gb=10.0, total_gb=24.0)
    stats = _minimal(gpu=[entry])
    assert stats["gpu"] == [entry]

    with pytest.raises(AssertionError):
        _minimal(gpu=[{"index": 0}])  # not built with gpu_entry(); missing keys


@pytest.mark.parametrize("bad_server", ["kv_text", "", None])
def test_unknown_server_asserts(bad_server):
    with pytest.raises(AssertionError):
        _minimal(server=bad_server)


@pytest.mark.parametrize("bad_path", ["batch", "", None])
def test_unknown_path_asserts(bad_path):
    with pytest.raises(AssertionError):
        _minimal(path=bad_path)


@pytest.mark.parametrize("field", ["n_items", "batch_size", "n_batches", "n_errors"])
def test_negative_counts_assert(field):
    with pytest.raises(AssertionError):
        _minimal(**{field: -1})


def test_batch_size_zero_asserts():
    with pytest.raises(AssertionError):
        _minimal(batch_size=0)


@pytest.mark.parametrize(
    "field", ["effective_compression_ratio", "materialized_compression_ratio"]
)
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_compression_ratio_out_of_range_asserts(field, value):
    with pytest.raises(AssertionError):
        _minimal(**{field: value})


def test_empty_model_name_asserts():
    with pytest.raises(AssertionError):
        _minimal(model_name="")


# ── Client-side parsing ──────────────────────────────────────────────────────


def test_parse_round_trips_a_server_response():
    stats = _minimal()
    parsed = parse_inference_stats(
        {"answers": ["x"], "log_odds": [0.1], "stats": stats},
        client_elapsed_s=1.5,
        endpoint="/text_qa",
    )
    assert parsed["schema"] == stats["schema"]
    assert parsed["client_elapsed_s"] == 1.5
    assert parsed["endpoint"] == "/text_qa"


def test_parse_missing_stats_key_asserts():
    with pytest.raises(AssertionError, match="stats"):
        parse_inference_stats({"answers": []}, client_elapsed_s=0.0, endpoint="/text_qa")


def test_parse_wrong_schema_version_asserts():
    stats = _minimal()
    stats["schema"] = 999
    with pytest.raises(AssertionError, match="schema"):
        parse_inference_stats(
            {"stats": stats}, client_elapsed_s=0.0, endpoint="/text_qa"
        )


def test_parse_partial_stats_object_asserts():
    stats = _minimal()
    del stats["gpu"]
    with pytest.raises(AssertionError, match="gpu"):
        parse_inference_stats(
            {"stats": stats}, client_elapsed_s=0.0, endpoint="/text_qa"
        )


def test_parse_non_dict_stats_asserts():
    with pytest.raises(AssertionError):
        parse_inference_stats(
            {"stats": "not-a-dict"}, client_elapsed_s=0.0, endpoint="/text_qa"
        )


def test_parse_negative_client_elapsed_asserts():
    with pytest.raises(AssertionError):
        parse_inference_stats(
            {"stats": _minimal()}, client_elapsed_s=-1.0, endpoint="/text_qa"
        )
