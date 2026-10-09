"""`KVCachingBackendBase._load_known_errors` key-format contract.

ERRORS.json is written by different generation paths with different key formats:
some write the bare content hash (e.g. `prepare_indices_relative`), others write the
full cache file path (e.g. the physical `prepare_caches`,
`"{save_dir}/cache_entry_{hash}.pt"`). `_load_known_errors` is used by the
relative-indices `prepare` check to tell "never generated" (n_missing, must crash)
apart from "generation genuinely failed for this item" (n_generation_errors, tolerated)
— see test_kv_prepare_missing_vs_errors.py for that client-side contract. That
distinction only works if entries recorded under either key format are found by a
bare-hash lookup; a mismatch silently reclassifies known, already-tolerated failures as
"missing" and crashes setup.
"""

import json

from reasondb.backends.kv_cache_base import KVCachingBackendBase

HASH_A = "5793bdc770049f66872e75d4152da6204830a1929633200f4068724e9349ecb4"
HASH_B = "0f5e0e6d35d2d6a47f2adcffba19f378ff38b71b294be01e46a38b8989c55fa1"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


# ── _normalize_error_key ─────────────────────────────────────────────────────


def test_normalize_error_key_passes_through_bare_hash():
    assert KVCachingBackendBase._normalize_error_key(HASH_A) == HASH_A


def test_normalize_error_key_strips_full_cache_path():
    path = f"/some/dir/comp0/cache_entry_{HASH_A}.pt"
    assert KVCachingBackendBase._normalize_error_key(path) == HASH_A


# ── _load_known_errors ───────────────────────────────────────────────────────


def test_load_known_errors_reads_full_path_keyed_errors(tmp_path):
    """Physical-mode ERRORS.json (keys are full cache file paths) must still be
    recognized by a bare-hash lookup."""
    press_dir = tmp_path / "expected_attention"
    _write_json(
        press_dir / "comp0" / "ERRORS.json",
        {
            f"{press_dir}/comp0/cache_entry_{HASH_A}.pt": "decompression bomb",
            f"{press_dir}/comp0/cache_entry_{HASH_B}.pt": "cannot identify image file",
        },
    )
    known_errors = KVCachingBackendBase._load_known_errors(str(press_dir), "comp0", "comp0")
    assert HASH_A in known_errors
    assert HASH_B in known_errors


def test_load_known_errors_reads_bare_hash_keyed_errors(tmp_path):
    """Relative-indices-mode ERRORS.json (keys are already bare hashes)."""
    press_dir = tmp_path / "expected_attention"
    _write_json(press_dir / "comp0" / "ERRORS.json", {HASH_A: "malformed text"})
    known_errors = KVCachingBackendBase._load_known_errors(str(press_dir), "comp0", "comp0")
    assert HASH_A in known_errors


def test_load_known_errors_merges_baseline_via_meta(tmp_path):
    """Target's {base}/indices/{tag}/_meta.json points at the baseline tag whose ERRORS.json
    (full-path keyed) must also be merged in, in addition to the target dir's own."""
    press_dir = tmp_path / "expected_attention"
    _write_json(
        press_dir / "comp0" / "indices" / "comp0_9" / "_meta.json", {"from": "comp0"}
    )
    _write_json(
        press_dir / "comp0" / "ERRORS.json",
        {f"{press_dir}/comp0/cache_entry_{HASH_A}.pt": "decompression bomb"},
    )
    _write_json(press_dir / "comp0_9" / "ERRORS.json", {HASH_B: "malformed text"})

    known_errors = KVCachingBackendBase._load_known_errors(
        str(press_dir), "comp0_9", "comp0"
    )
    assert HASH_A in known_errors  # from the baseline
    assert HASH_B in known_errors  # from the target itself


def test_load_known_errors_empty_when_nothing_on_disk(tmp_path):
    press_dir = tmp_path / "expected_attention"
    assert KVCachingBackendBase._load_known_errors(str(press_dir), "comp0", "comp0") == {}
