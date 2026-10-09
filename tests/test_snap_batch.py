"""`KVCachingBackendBase._snap_batch` grid.

The memory-safe batch-size estimate is snapped down to a multiple of 8. Compared with
snapping to the largest power of two <= max_batch (which can discard up to ~50% of the
estimate, e.g. 115 -> 64), this keeps most of the allocator-reuse/tensor-core-alignment
benefit while discarding at most 7 of the available batch slots.
"""

import pytest

from reasondb.backends.kv_cache_base import KVCachingBackendBase


@pytest.mark.parametrize(
    "max_batch, expected",
    [
        (0, 1),  # non-positive input floors to a usable batch of 1
        (-5, 1),
        (1, 1),  # below the grid: use max_batch as-is, don't round down to 0
        (7, 7),
        (8, 8),  # exactly on the grid
        (9, 8),
        (15, 8),
        (45, 40),  # representative estimates from real workloads
        (48, 48),
        (90, 88),
        (115, 112),
        (182, 176),
        (999, 992),
        (1000, 1000),
        (1024, 1024),  # exactly at the cap
        (1027, 1024),  # above the cap: clamp down, not up to the next multiple of 8
        (5000, 1024),
    ],
)
def test_snap_batch(max_batch, expected):
    assert KVCachingBackendBase._snap_batch(max_batch) == expected


def test_snap_batch_never_exceeds_input():
    """The snapped batch must be memory-safe: never larger than the raw estimate."""
    for max_batch in range(0, 300):
        assert KVCachingBackendBase._snap_batch(max_batch) <= max(max_batch, 1)
