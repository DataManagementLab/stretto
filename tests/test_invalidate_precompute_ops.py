"""`scripts/invalidate_precompute_ops.py`: dropping resume markers, and - only when
asked - the pinned LLM configuration alongside them.

The two live under differently shaped keys: a marker names one physical variant
(``ExtractAndQaImageFilter-ImageQABackend-llava-.../...``), a pinned config names the
interface they share (``ExtractAndQaImageFilter``). One ``--operators`` prefix has to
select both, or ``--drop-configs`` would unpin operators whose markers survive - the
worst of both states, since the next run re-derives a configuration for work it then
skips.
"""

import json

import pytest

from scripts.invalidate_precompute_ops import main, matches, operator_of

MARKERS = [
    "ExtractAndQaImageFilter-ImageQABackend-llava-hf/llava-next-72b-hf-cr0.5|e1|_T0",
    "ExtractAndQaImageFilter-ImageQABackend-llava-hf/llava-next-72b-hf-cr0.0|e1|_T0",
    "ImageQaFilter-ImageQABackend-llava-hf/llava-next-72b-hf-cr0.0|e2|_T0",
]
CONFIGS = {
    "ExtractAndQaImageFilter|e1|_T0": {"match_question_template": "{right_extracted}"},
    "ImageQaFilter|e2|_T0": {"question_template": "q"},
}


def _write(tmp_path, **overrides):
    data = {
        "text_qa": {"model-a": {"h1": {"question": "q1", "response": "yes"}}},
        "vision": {},
        "precomputed_ops": list(MARKERS),
        "operator_configs": dict(CONFIGS),
        **overrides,
    }
    path = tmp_path / "precompute.json"
    path.write_text(json.dumps(data))
    return path


def _run(monkeypatch, path, *argv):
    monkeypatch.setattr(
        "sys.argv", ["invalidate_precompute_ops.py", str(path), *argv]
    )
    main()
    return json.loads(path.read_text())


def test_a_config_key_and_its_markers_answer_to_the_same_prefix():
    prefixes = ["ExtractAndQaImageFilter"]
    assert [m for m in MARKERS if matches(m, prefixes)] == MARKERS[:2]
    assert [k for k in CONFIGS if matches(k, prefixes)] == [
        "ExtractAndQaImageFilter|e1|_T0"
    ]
    assert operator_of("ImageQaFilter|e2|_T0") == "ImageQaFilter"


def test_configs_survive_a_plain_marker_drop(monkeypatch, tmp_path):
    """The default: a stale pin is still the phrasing every recorded response is under."""
    path = _write(tmp_path)
    data = _run(monkeypatch, path, "--operators", "ExtractAndQaImageFilter", "--apply")

    assert data["precomputed_ops"] == MARKERS[2:]
    assert data["operator_configs"] == CONFIGS


def test_drop_configs_removes_the_pin_of_the_same_operators(monkeypatch, tmp_path):
    path = _write(tmp_path)
    data = _run(
        monkeypatch, path,
        "--operators", "ExtractAndQaImageFilter", "--drop-configs", "--apply",
    )

    assert data["precomputed_ops"] == MARKERS[2:]
    assert data["operator_configs"] == {"ImageQaFilter|e2|_T0": {"question_template": "q"}}
    # Responses are keyed by content, so they are never the thing being invalidated.
    assert data["text_qa"] == {"model-a": {"h1": {"question": "q1", "response": "yes"}}}


def test_buckets_the_script_does_not_know_about_survive(monkeypatch, tmp_path):
    """It rewrites the whole JSON, so anything it does not enumerate is at risk. The
    filter-stats matrix in particular must not be dropped by an operator invalidation -
    it is the only copy the store carries, and losing it silently changes the query set.
    """
    stats = {"movie_random": {"dev": {"keys": {"": {"overlap_matrix": [[1, 0]]}}}}}
    path = _write(tmp_path, filter_stats=stats)
    data = _run(
        monkeypatch, path,
        "--operators", "ExtractAndQaImageFilter", "--drop-configs", "--apply",
    )
    assert data["filter_stats"] == stats


def test_a_dry_run_writes_nothing(monkeypatch, tmp_path):
    path = _write(tmp_path)
    before = path.read_text()
    _run(monkeypatch, path, "--operators", "ExtractAndQaImageFilter", "--drop-configs")
    assert path.read_text() == before


def test_a_pin_can_be_dropped_after_its_markers_already_were(monkeypatch, tmp_path):
    """Marker-drop first, then the prompt shape turns out to have changed too.

    The second pass matches no markers, but must still reach the configs - otherwise
    the recovery from a half-done invalidation is to hand-edit the JSON.
    """
    path = _write(tmp_path, precomputed_ops=MARKERS[2:])
    data = _run(
        monkeypatch, path,
        "--operators", "ExtractAndQaImageFilter", "--drop-configs", "--apply",
    )
    assert "ExtractAndQaImageFilter|e1|_T0" not in data["operator_configs"]


def test_operators_is_required(monkeypatch, tmp_path):
    path = _write(tmp_path)
    with pytest.raises(SystemExit):
        _run(monkeypatch, path, "--drop-configs", "--apply")
