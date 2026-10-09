"""`scripts/merge_precompute.py`: combining `--precompute` JSON files recorded by
separate per-modality processes (see that script's module docstring for the full
seed-then-parallel-then-merge workflow this supports).
"""

import json

import pytest

from reasondb.evaluation.precompute_merge import merge_precompute_data


def _write(path, data):
    path.write_text(json.dumps(data))
    return path


def test_disjoint_text_and_vision_buckets_are_unioned(tmp_path):
    text_file = _write(
        tmp_path / "text.json",
        {
            "text_qa": {"model-a": {"h1": {"question": "q1", "response": "yes"}}},
            "vision": {},
            "precomputed_ops": ["TextQaFilter|expr|_T0"],
            "operator_configs": {"TextQaFilter|expr|_T0": {"question_template": "q1"}},
        },
    )
    image_file = _write(
        tmp_path / "image.json",
        {
            "text_qa": {},
            "vision": {"model-b": {"h2": {"question": "q2", "response": "no"}}},
            "precomputed_ops": ["ImageQaFilter|expr2|_T0"],
            "operator_configs": {"ImageQaFilter|expr2|_T0": {"question_template": "q2"}},
        },
    )

    merged, conflicts = merge_precompute_data([text_file, image_file])

    assert conflicts == []
    assert merged["text_qa"] == {"model-a": {"h1": {"question": "q1", "response": "yes"}}}
    assert merged["vision"] == {"model-b": {"h2": {"question": "q2", "response": "no"}}}
    assert merged["precomputed_ops"] == ["ImageQaFilter|expr2|_T0", "TextQaFilter|expr|_T0"]
    assert merged["operator_configs"] == {
        "TextQaFilter|expr|_T0": {"question_template": "q1"},
        "ImageQaFilter|expr2|_T0": {"question_template": "q2"},
    }


def test_identical_overlapping_entries_are_not_a_conflict(tmp_path):
    """Seeding both files from the same starting file (the recommended workflow)
    means shared operator_configs keys should be byte-identical - merging that
    overlap must be silent, not a conflict.
    """
    shared_config = {"TextQaFilter|expr|_T0": {"question_template": "q1"}}
    file_a = _write(
        tmp_path / "a.json",
        {"text_qa": {}, "vision": {}, "precomputed_ops": [], "operator_configs": shared_config},
    )
    file_b = _write(
        tmp_path / "b.json",
        {"text_qa": {}, "vision": {}, "precomputed_ops": [], "operator_configs": shared_config},
    )

    merged, conflicts = merge_precompute_data([file_a, file_b])

    assert conflicts == []
    assert merged["operator_configs"] == shared_config


def test_diverging_operator_config_is_reported_as_conflict(tmp_path):
    """Two processes that independently derived phrasing for the same operator
    (i.e. weren't seeded from a shared file first) must be flagged, not silently
    merged - a silently-picked phrasing might not match how a response was
    actually recorded, breaking --simulate's cache lookup.
    """
    file_a = _write(
        tmp_path / "a.json",
        {
            "text_qa": {},
            "vision": {},
            "precomputed_ops": [],
            "operator_configs": {"TextQaFilter|expr|_T0": {"question_template": "phrasing A"}},
        },
    )
    file_b = _write(
        tmp_path / "b.json",
        {
            "text_qa": {},
            "vision": {},
            "precomputed_ops": [],
            "operator_configs": {"TextQaFilter|expr|_T0": {"question_template": "phrasing B"}},
        },
    )

    merged, conflicts = merge_precompute_data([file_a, file_b])

    assert len(conflicts) == 1
    assert "TextQaFilter|expr|_T0" in conflicts[0]
    # First file's value wins when the caller proceeds past the conflict (--force).
    assert merged["operator_configs"] == {"TextQaFilter|expr|_T0": {"question_template": "phrasing A"}}


def test_filter_stats_are_carried_through_the_merge(tmp_path):
    """The merged file is what --simulate loads. Dropping the matrix here would make
    every later run fall back to regenerating its queries from nothing - silently, since
    a missing matrix looks the same as a benchmark that never had one.
    """
    payload = {"keys": {"": {"overlap_matrix": [[1]]}}, "base_table_hashes": {}}
    text_file = _write(
        tmp_path / "text.json",
        {
            "text_qa": {},
            "vision": {},
            "precomputed_ops": [],
            "operator_configs": {},
            "filter_stats": {"movie_random": {"dev": payload}},
        },
    )
    image_file = _write(
        tmp_path / "image.json",
        {"text_qa": {}, "vision": {}, "precomputed_ops": [], "operator_configs": {}},
    )

    merged, conflicts = merge_precompute_data([text_file, image_file])

    assert conflicts == []
    assert merged["filter_stats"] == {"movie_random": {"dev": payload}}


def test_two_matrices_for_one_benchmark_split_are_a_conflict(tmp_path):
    """Unlike a phrasing conflict, this one changes which queries exist at all - so the
    caller has to be told rather than handed whichever file was listed first."""
    file_a = _write(
        tmp_path / "a.json",
        {
            "text_qa": {},
            "vision": {},
            "precomputed_ops": [],
            "operator_configs": {},
            "filter_stats": {"movie_random": {"dev": {"keys": {"": {"overlap_matrix": [[1]]}}}}},
        },
    )
    file_b = _write(
        tmp_path / "b.json",
        {
            "text_qa": {},
            "vision": {},
            "precomputed_ops": [],
            "operator_configs": {},
            "filter_stats": {"movie_random": {"dev": {"keys": {"": {"overlap_matrix": [[0]]}}}}},
        },
    )

    merged, conflicts = merge_precompute_data([file_a, file_b])

    assert len(conflicts) == 1
    assert "filter_stats" in conflicts[0] and "movie_random" in conflicts[0]
    assert merged["filter_stats"]["movie_random"]["dev"]["keys"][""]["overlap_matrix"] == [[1]]


def test_conflicting_qa_record_under_same_hash_raises(tmp_path):
    """A hash collision with differing content should be impossible (the hash is
    over the full question+context/image_path) - surfaced loudly rather than
    silently dropped, since either possibility (corruption or a real collision)
    needs a human to look at it.
    """
    file_a = _write(
        tmp_path / "a.json",
        {
            "text_qa": {"model-a": {"h1": {"question": "q1", "response": "yes"}}},
            "vision": {},
            "precomputed_ops": [],
            "operator_configs": {},
        },
    )
    file_b = _write(
        tmp_path / "b.json",
        {
            "text_qa": {"model-a": {"h1": {"question": "q1", "response": "no"}}},
            "vision": {},
            "precomputed_ops": [],
            "operator_configs": {},
        },
    )

    with pytest.raises(ValueError, match="conflicts with an already-merged record"):
        merge_precompute_data([file_a, file_b])
