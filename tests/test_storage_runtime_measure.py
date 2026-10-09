"""On-disk accounting contract for ``reasondb.evaluation.parameter_sweep`` helpers.

``measure_materialized_storage_bytes`` reports the footprint of one materialized
level, scoped to one press (``press_name``) and to the effective ratios the current
run can actually reach (``allowed_effective_ratios``). Under the nested index layout
(``{press}/comp{base}/indices/comp{e}/``) the relative indices a baseline unlocks are
part of that baseline's cost, so they ARE counted when ``e`` is in
``allowed_effective_ratios``; every nested index has effective ``e >= base`` by
construction and a misplaced one trips an assert regardless of the filter. The legacy
flat layout (``{press}/indices/comp{e}/``) cannot be attributed by position and stays
excluded. ``has_effective_index`` finds an index dir for an effective ratio under
either layout, scoped to one press. These tests pin that behavior.
"""

import pytest

try:
    from reasondb.evaluation.parameter_sweep import (
        from_compression_tag,
        has_effective_index,
        measure_materialized_storage_bytes,
        to_compression_tag,
    )
except ImportError:
    pytest.skip("parameter_sweep deps not installed", allow_module_level=True)

HASH_A = "5793bdc770049f66872e75d4152da6204830a1929633200f4068724e9349ecb4"
HASH_B = "aa93bdc770049f66872e75d4152da6204830a1929633200f4068724e9349ecff"

PRESS = "expected_attention"


def _write(path, nbytes):
    """Write a file of exactly ``nbytes`` bytes, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * nbytes)


def _press_dir(tmp_path, model="Llama-3B", press=PRESS):
    return tmp_path / "artwork_random" / "kv-text-qa-cache" / model / press


def test_roundtrip_compression_tag():
    for cr in (0.0, 0.5, 0.7, 0.9):
        assert from_compression_tag(to_compression_tag(cr)) == cr
    # accepts the bare tag as well as the comp-prefixed form
    assert from_compression_tag("comp0_7") == 0.7
    assert from_compression_tag("0") == 0.0


def test_counts_baseline_plus_allowed_nested_indices(tmp_path):
    press = _press_dir(tmp_path)
    base = press / "comp0"  # materialized baseline at cr 0.0
    _write(base / f"cache_entry_{HASH_A}.pt", 1000)
    # two effective ratios served from this baseline via cheap relative indices
    _write(base / "indices" / "comp0_7" / f"idx_{HASH_A}.pt", 30)
    _write(base / "indices" / "comp0_7" / "_meta.json", 20)
    _write(base / "indices" / "comp0_9" / f"idx_{HASH_A}.pt", 10)

    total = measure_materialized_storage_bytes(
        tmp_path, 0.0, PRESS, allowed_effective_ratios=frozenset({0.7, 0.9})
    )
    assert total == 1000 + 30 + 20 + 10


def test_grid_scoping_excludes_indices_outside_the_current_run(tmp_path):
    """A ratio not in this run's grid must not inflate the footprint."""
    press = _press_dir(tmp_path)
    base = press / "comp0"
    _write(base / f"cache_entry_{HASH_A}.pt", 1000)
    _write(base / "indices" / "comp0_7" / f"idx_{HASH_A}.pt", 30)  # in this run's grid
    _write(base / "indices" / "comp0_9" / f"idx_{HASH_A}.pt", 999)  # NOT in this run's grid

    total = measure_materialized_storage_bytes(
        tmp_path, 0.0, PRESS, allowed_effective_ratios=frozenset({0.7})
    )
    assert total == 1000 + 30


def test_no_indexing_mode_excludes_all_indices_even_if_present_on_disk(tmp_path):
    """Direct (no --use-indexes) mode never reconstructs from indices, so none of
    their bytes belong to the footprint even if a prior indexed run left them
    behind on the same cache_root."""
    press = _press_dir(tmp_path)
    base = press / "comp0"
    _write(base / f"cache_entry_{HASH_A}.pt", 1000)
    _write(base / "indices" / "comp0_7" / f"idx_{HASH_A}.pt", 500)

    assert measure_materialized_storage_bytes(tmp_path, 0.0, PRESS) == 1000
    assert (
        measure_materialized_storage_bytes(
            tmp_path, 0.0, PRESS, allowed_effective_ratios=frozenset()
        )
        == 1000
    )


def test_press_scoping_ignores_a_different_press_at_the_same_ratio(tmp_path):
    """A cache left on disk by a different press must not be folded into this run's
    footprint: two presses sharing cache_root must not double-count."""
    _write(_press_dir(tmp_path, press=PRESS) / "comp0_5" / f"cache_entry_{HASH_A}.pt", 100)
    _write(_press_dir(tmp_path, press="finch") / "comp0_5" / f"cache_entry_{HASH_B}.pt", 700)

    assert measure_materialized_storage_bytes(tmp_path, 0.5, PRESS) == 100
    assert measure_materialized_storage_bytes(tmp_path, 0.5, "finch") == 700


def test_excludes_legacy_flat_indices(tmp_path):
    press = _press_dir(tmp_path)
    _write(press / "comp0" / f"cache_entry_{HASH_A}.pt", 1000)
    # flat layout: sibling of the baseline, not attributable by position -> excluded
    _write(press / "indices" / "comp0_9" / f"idx_{HASH_A}.pt", 500)

    assert (
        measure_materialized_storage_bytes(
            tmp_path, 0.0, PRESS, allowed_effective_ratios=frozenset({0.9})
        )
        == 1000
    )


def test_model_filter_attributes_bytes_per_model(tmp_path):
    a = _press_dir(tmp_path, "Llama-3B") / "comp0_5"
    b = _press_dir(tmp_path, "Llama-70B") / "comp0_5"
    _write(a / f"cache_entry_{HASH_A}.pt", 100)
    _write(b / f"cache_entry_{HASH_B}.pt", 700)

    assert measure_materialized_storage_bytes(tmp_path, 0.5, PRESS, "Llama-3B") == 100
    assert measure_materialized_storage_bytes(tmp_path, 0.5, PRESS, "Llama-70B") == 700
    assert measure_materialized_storage_bytes(tmp_path, 0.5, PRESS) == 800  # both


def test_returns_zero_when_level_absent(tmp_path):
    _write(_press_dir(tmp_path) / "comp0" / f"cache_entry_{HASH_A}.pt", 100)
    assert measure_materialized_storage_bytes(tmp_path, 0.9, PRESS) == 0
    assert measure_materialized_storage_bytes(tmp_path / "missing", 0.0, PRESS) == 0


def test_asserts_when_nested_index_undercuts_baseline(tmp_path):
    """A nested index with effective cr < baseline is an impossible layout, and the
    assert fires regardless of whether that ratio is in the allowed grid."""
    base = _press_dir(tmp_path) / "comp0_9"
    _write(base / f"cache_entry_{HASH_A}.pt", 100)
    _write(base / "indices" / "comp0_5" / f"idx_{HASH_A}.pt", 10)  # 0.5 < 0.9

    with pytest.raises(AssertionError):
        measure_materialized_storage_bytes(tmp_path, 0.9, PRESS)


def test_has_effective_index_nested_and_flat(tmp_path):
    press = _press_dir(tmp_path)
    _write(press / "comp0" / "indices" / "comp0_7" / f"idx_{HASH_A}.pt", 10)  # nested
    _write(press / "indices" / "comp0_9" / "_meta.json", 10)  # flat

    assert has_effective_index(tmp_path, 0.7, PRESS, "Llama-3B")  # nested
    assert has_effective_index(tmp_path, 0.9, PRESS, "Llama-3B")  # flat
    assert not has_effective_index(tmp_path, 0.5, PRESS, "Llama-3B")  # absent
    assert not has_effective_index(tmp_path, 0.7, PRESS, "OtherModel")  # model filter misses


def test_has_effective_index_ignores_a_different_press(tmp_path):
    """An index generated by a different press must not satisfy the check for the
    press this run actually serves: an unscoped check would let
    the sweep proceed while the real serving press still has nothing to reconstruct
    from, silently corrupting the storage/runtime measurement instead of failing
    fast in prepare_sweep."""
    press = _press_dir(tmp_path, press="finch")
    _write(press / "comp0" / "indices" / "comp0_7" / f"idx_{HASH_A}.pt", 10)  # nested
    _write(press / "indices" / "comp0_9" / "_meta.json", 10)  # flat

    assert not has_effective_index(tmp_path, 0.7, PRESS, "Llama-3B")
    assert not has_effective_index(tmp_path, 0.9, PRESS, "Llama-3B")
    assert has_effective_index(tmp_path, 0.7, "finch", "Llama-3B")
    assert has_effective_index(tmp_path, 0.9, "finch", "Llama-3B")


# ── measure_level: the same walk, read as items as well as bytes ─────────────
#
# A footprint in gigabytes says what a level costs here; divided by the items behind it
# it says what the operator costs per row, which is the form that carries to a dataset
# nobody has measured. These pin what counts as an item.


def test_measure_level_counts_cache_entries_beside_the_bytes(tmp_path):
    from reasondb.evaluation.parameter_sweep import measure_level

    press = _press_dir(tmp_path)
    base = press / "comp0_5"
    _write(base / f"cache_entry_{HASH_A}.pt", 1000)
    _write(base / f"cache_entry_{HASH_B}.pt", 1000)

    level = measure_level(tmp_path, 0.5, PRESS)

    assert level == (2000, 2)
    assert level.bytes == 2000 and level.entries == 2


def test_the_error_log_is_part_of_the_footprint_but_is_not_an_item(tmp_path):
    """``ERRORS.json`` sits beside the entries and is bytes on disk like anything else -
    but counting it as an entry would understate what one cached item costs."""
    from reasondb.evaluation.parameter_sweep import measure_level

    press = _press_dir(tmp_path)
    base = press / "comp0_5"
    _write(base / f"cache_entry_{HASH_A}.pt", 1000)
    _write(base / "ERRORS.json", 40)

    assert measure_level(tmp_path, 0.5, PRESS) == (1040, 1)


def test_relative_index_files_are_bytes_and_never_items(tmp_path):
    """No physical cache lives under ``indices/``, so an index is a cheaper way to serve
    an item that is already counted once at its baseline."""
    from reasondb.evaluation.parameter_sweep import measure_level

    press = _press_dir(tmp_path)
    base = press / "comp0"
    _write(base / f"cache_entry_{HASH_A}.pt", 1000)
    _write(base / "indices" / "comp0_7" / f"idx_{HASH_A}.pt", 30)

    level = measure_level(
        tmp_path, 0.0, PRESS, allowed_effective_ratios=frozenset({0.7})
    )

    assert level == (1030, 1)


def test_measure_level_reports_nothing_for_a_level_that_is_not_there(tmp_path):
    from reasondb.evaluation.parameter_sweep import measure_level

    assert measure_level(tmp_path, 0.9, PRESS) == (0, 0)
    assert measure_level(tmp_path / "missing", 0.0, PRESS) == (0, 0)


def test_measure_materialized_storage_bytes_is_measure_levels_byte_half(tmp_path):
    """The bytes-only function is a projection, not a second implementation - so the
    byte contracts above also describe the walk that counts items."""
    from reasondb.evaluation.parameter_sweep import measure_level

    press = _press_dir(tmp_path)
    _write(press / "comp0_5" / f"cache_entry_{HASH_A}.pt", 700)

    assert (
        measure_materialized_storage_bytes(tmp_path, 0.5, PRESS)
        == measure_level(tmp_path, 0.5, PRESS).bytes
    )
