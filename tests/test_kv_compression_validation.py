"""Validation of the client-side KV cache-serving arguments.

Compression ratio is the fraction of KV entries DROPPED (0.0 = no compression,
1.0 = maximum). The effective cache is derived by indexing into the materialized one, which
can only drop entries, hence effective >= materialized.

``keep_in_memory`` (the server holds this operator's caches in RAM) adds two rules: it
cannot combine with ``vanilla``, which has no cache to hold, and it requires a directly
materialized cache, since an indexed one is reconstructed per query.
"""

import pytest

from reasondb.backends.kv_cache_base import validate_kv_compression_ratios


@pytest.mark.parametrize(
    "effective, materialized, vanilla, keep_in_memory",
    [
        (0.9, 0.9, False, False),  # physical cache: effective == materialized
        (0.9, 0.3, False, False),  # relative indices: effective from materialized
        (0.99, 0.0, False, False),  # maximally compressed from an uncompressed baseline
        (0.0, 0.0, False, False),  # CR=0 physical cache (not the same as vanilla)
        (1.0, 1.0, False, False),  # maximum compression, physical cache
        (0.0, 0.0, True, False),  # vanilla: no pre-computed cache
        (0.8, 0.8, False, True),  # in-memory: a physical cache, held in RAM
        (0.0, 0.0, False, True),  # an uncompressed physical cache, held in RAM
    ],
)
def test_accepts_valid_combinations(effective, materialized, vanilla, keep_in_memory):
    validate_kv_compression_ratios(effective, materialized, vanilla, keep_in_memory)


@pytest.mark.parametrize(
    "effective, materialized, vanilla, keep_in_memory, reason",
    [
        (0.3, 0.9, False, False, "effective < materialized: indexing cannot add back"),
        (1.5, 1.5, False, False, "effective above the [0, 1] range"),
        (-0.1, 0.5, False, False, "effective below the [0, 1] range"),
        (0.5, 1.5, False, False, "materialized above the [0, 1] range"),
        (0.5, -0.1, False, False, "materialized below the [0, 1] range"),
        (0.5, 0.5, True, False, "vanilla uses no cache, so both ratios must be 0.0"),
        (0.0, 0.3, True, False, "vanilla with a non-zero materialized ratio"),
        (0.0, 0.0, True, True, "vanilla has no cache to hold in RAM"),
        (0.8, 0.0, False, True, "in-memory needs a directly materialized cache"),
    ],
)
def test_rejects_invalid_combinations(
    effective, materialized, vanilla, keep_in_memory, reason
):
    with pytest.raises(AssertionError):
        validate_kv_compression_ratios(
            effective, materialized, vanilla, keep_in_memory
        )
