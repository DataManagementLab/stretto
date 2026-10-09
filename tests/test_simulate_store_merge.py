"""``--simulate`` takes several precompute files and merges them.

``--precompute`` records one file per dataset, so any sweep spanning datasets - which is
the normal case for a coordinator task - needs all of them live at once. These pin the
merge's two interesting properties: response caches union (their keys are content
hashes, so cross-dataset collisions cannot happen by accident), and a genuine
disagreement about a *pinned operator config* resolves deterministically instead of
depending on which file was listed first.
"""

import json

import pytest

from reasondb.backends.simulate_store import SimulateStore


def _write(path, **sections):
    payload = {
        "text_qa": sections.get("text_qa", {}),
        "vision": sections.get("vision", {}),
        "precomputed_ops": sections.get("precomputed_ops", []),
        "operator_configs": sections.get("operator_configs", {}),
    }
    path.write_text(json.dumps(payload))
    return path


def test_load_merges_several_files(tmp_path):
    movie = _write(
        tmp_path / "movie.json",
        text_qa={"llama-8b": {"k1": {"response": "yes"}}},
        precomputed_ops=["TextQaFilter|is it long|movies"],
        operator_configs={"TextQaFilter|long|movies": {"question": "long?"}},
    )
    artwork = _write(
        tmp_path / "artwork.json",
        text_qa={"llama-8b": {"k2": {"response": "no"}}, "llama-70b": {"k3": {"response": "x"}}},
        vision={"llava": {"k4": {"response": "img"}}},
        precomputed_ops=["ImageQaFilter|is it a portrait|art"],
        operator_configs={"ImageQaFilter|portrait|art": {"question": "portrait?"}},
    )

    store = SimulateStore.load([movie, artwork])
    counts = store.counts()
    # Both datasets' responses are live at once, including a model only one file had.
    assert counts["n_text_qa"] == 3
    assert counts["n_vision"] == 1
    assert counts["n_ops"] == 2
    assert counts["n_configs"] == 2


def test_load_still_accepts_a_single_path(tmp_path):
    """Passing a single Path must keep working."""
    one = _write(tmp_path / "one.json", text_qa={"m": {"k": {"response": "y"}}})
    assert SimulateStore.load(one).counts()["n_text_qa"] == 1
    assert SimulateStore.load(str(one)).counts()["n_text_qa"] == 1
    assert SimulateStore.load([one]).counts()["n_text_qa"] == 1


def test_merging_the_same_model_keeps_both_files_entries(tmp_path):
    """Two datasets recorded against one model must not shadow each other - the naive
    ``dict.update`` at the model level would drop a whole dataset's responses."""
    a = _write(tmp_path / "a.json", text_qa={"llama-8b": {"k1": {"response": "1"}}})
    b = _write(tmp_path / "b.json", text_qa={"llama-8b": {"k2": {"response": "2"}}})
    store = SimulateStore.load([a, b])
    assert set(store._text_qa["llama-8b"]) == {"k1", "k2"}


def test_conflicting_operator_configs_keep_the_first_and_warn(tmp_path, caplog):
    """A pinned config exists so every run configures an operator identically. Two files
    disagreeing is a real ambiguity, so it resolves to the first file - stable regardless
    of how the caller ordered them - and says so out loud."""
    a = _write(tmp_path / "a.json", operator_configs={"shared": {"question": "first"}})
    b = _write(tmp_path / "b.json", operator_configs={"shared": {"question": "second"}})

    with caplog.at_level("WARNING"):
        store = SimulateStore.load([a, b])
    assert store._operator_configs["shared"] == {"question": "first"}
    assert "already pinned differently" in caplog.text
    # Identical configs in both files are not a conflict and stay silent.
    caplog.clear()
    c = _write(tmp_path / "c.json", operator_configs={"shared": {"question": "first"}})
    with caplog.at_level("WARNING"):
        SimulateStore.load([a, c])
    assert "already pinned differently" not in caplog.text


def test_load_rejects_an_empty_path_list(tmp_path):
    with pytest.raises(AssertionError):
        SimulateStore.load([])
