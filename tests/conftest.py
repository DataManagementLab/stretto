"""Shared fixtures and fakes for the test suite."""

from typing import Dict, Iterable, List, Tuple

import pytest


@pytest.fixture(autouse=True)
def _isolated_kv_cache_sizes(tmp_path, monkeypatch):
    """Point the recorded KV cache sizes at a per-test file.

    ``prepare_sweep`` records every storage walk it makes, and the default recording is a
    file inside the package - the one a checkout ships and a replay reads. A test that
    reached a real walk would otherwise rewrite it, or read whatever the developer's own
    machine had recorded there and pass or fail on that.
    """
    monkeypatch.setenv("REASONDB_KV_CACHE_SIZES", str(tmp_path / "kv_cache_sizes.json"))


def pytest_addoption(parser):  # pragma: no cover - only used interactively
    """``--regenerate-golden``: rewrite the golden fixtures instead of asserting on them.

    Defined here because pytest only collects this hook from conftest files and plugins.
    """
    parser.addoption("--regenerate-golden", action="store_true")


def fake_slot_map(states: Iterable[Tuple[Dict[str, List[float]], int]]) -> Dict[str, object]:
    """A ``slot_by_key`` consistent with these states, for faking ``prepare_sweep``.

    ``prepare_sweep`` returns a state list *and* the ``ModelSlot`` map those states' keys
    index into, and the two are consistent by construction. This builds the map from the
    same states so that a faked ``prepare_sweep`` keeps that invariant.
    """
    from reasondb.evaluation.parameter_sweep import ModelSlot
    from reasondb.interface.default_operator_toolbox import (
        IMAGE_MODEL_8B,
        IMAGE_MODEL_70B,
        TEXT_MODEL_8B,
        TEXT_MODEL_70B,
    )

    models = {
        "text_small": ("text", TEXT_MODEL_8B, False),
        "text_large": ("text", TEXT_MODEL_70B, True),
        "image_small": ("image", IMAGE_MODEL_8B, False),
        "image_large": ("image", IMAGE_MODEL_70B, True),
    }
    keys = {key for state, _footprint in states for key in state}
    return {
        key: ModelSlot(key, models[key][0], models[key][1], large=models[key][2])
        for key in keys
        if key in models
    }


def cached_answer(base_dir, name, frame):
    """Write one answer where the executor's result cache would have put it.

    A shard's ``results`` is a manifest of paths into that cache rather than a second
    copy of the answers (see ``coordinator.producers.shards.write_shard``), so a test
    that builds a shard by hand has to put a file on disk for it to point at. *base_dir*
    is anywhere under the task directory - what matters is only that
    ``manifest_base`` can relativize it.
    """
    import pickle
    from pathlib import Path

    from reasondb.evaluation.row_signature import RowSignature

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{name}.sig.pkl"
    with open(path, "wb") as f:
        pickle.dump(RowSignature.of(frame), f)
    return path
