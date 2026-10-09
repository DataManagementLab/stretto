"""`migrate_legacy_index_dirs` startup migration contract.

Older generators wrote relative indices to a flat ``{press}/indices/{target}`` layout keyed
only by the effective ratio, so a single target could only be indexed from one baseline. The
current layout nests indices under their materialized baseline
(``{press}/{base}/indices/{target}``). To keep every reader on a single code path, the servers
call ``migrate_legacy_index_dirs`` once at startup to relocate any flat dir into the nested
layout using each dir's own ``_meta.json`` (key ``"from"``). These tests pin that behavior.
"""

import json

import pytest

try:
    # kv_cache_reconstruct imports the kvpress submodule at module load.
    from reasondb.backends.kv_cache_reconstruct import (
        _relative_index_dir,
        migrate_legacy_index_dirs,
    )
except ImportError:
    pytest.skip("kvpress submodule not installed", allow_module_level=True)

HASH_A = "5793bdc770049f66872e75d4152da6204830a1929633200f4068724e9349ecb4"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_migrates_flat_dir_into_nested_layout(tmp_path):
    press_dir = tmp_path / "expected_attention"
    flat = press_dir / "indices" / "comp0_9"
    _write_json(flat / "_meta.json", {"from": "comp0"})
    (flat / f"idx_{HASH_A}.pt").write_bytes(b"payload")

    migrate_legacy_index_dirs(str(press_dir))

    nested = press_dir / "comp0" / "indices" / "comp0_9"
    assert (nested / "_meta.json").exists()
    assert (nested / f"idx_{HASH_A}.pt").read_bytes() == b"payload"
    assert not flat.exists()  # legacy dir moved, not copied
    # the reader now resolves the target through the nested layout only
    assert _relative_index_dir(str(press_dir), "comp0_9", "comp0") == str(nested)


def test_leaves_flat_dir_without_meta_untouched(tmp_path):
    """A flat dir with no _meta.json is an absolute-index dir, not a relative one."""
    press_dir = tmp_path / "expected_attention"
    flat = press_dir / "indices" / "comp0_9"
    (flat).mkdir(parents=True)
    (flat / f"idx_{HASH_A}.pt").write_bytes(b"payload")

    migrate_legacy_index_dirs(str(press_dir))

    assert flat.exists()
    assert not (press_dir / "comp0" / "indices" / "comp0_9").exists()


def test_skips_when_nested_target_already_exists(tmp_path):
    """A stale flat dup is left in place rather than clobbering the live nested dir."""
    press_dir = tmp_path / "expected_attention"
    _write_json(press_dir / "indices" / "comp0_9" / "_meta.json", {"from": "comp0"})
    nested_meta = press_dir / "comp0" / "indices" / "comp0_9" / "_meta.json"
    _write_json(nested_meta, {"from": "comp0"})

    migrate_legacy_index_dirs(str(press_dir))

    assert (press_dir / "indices" / "comp0_9" / "_meta.json").exists()  # dup untouched
    assert nested_meta.exists()


def test_noop_when_no_legacy_root(tmp_path):
    press_dir = tmp_path / "expected_attention"
    press_dir.mkdir()
    migrate_legacy_index_dirs(str(press_dir))  # must not raise
