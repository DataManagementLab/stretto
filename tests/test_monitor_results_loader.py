"""The Results tab must read exactly what the `run_benchmark` producer writes, and refuse
malformed inputs by naming the problem rather than raising or rendering blank.

`reasondb.monitor.results` mirrors `scripts/plot_benchmark.py`'s `load_df` (the
guarantee back-fill for no-guarantee baselines) so the two never disagree about which
guarantee level a `gpt` or fixed-compression row belongs on. These tests build synthetic
metrics CSVs under `tmp_path` - the same shape `evaluation.evaluate()` writes - rather
than depending on a real benchmark run.
"""

import pandas as pd
import pytest

from reasondb.monitor import results


REQUIRED_COLS = [
    "approach_name",
    "query",
    "precision_guarantee",
    "recall_guarantee",
    "precision",
    "recall",
    "f1_score",
    "execution_cost_runtime",
    "tuning_cost_runtime",
    "total_cost_runtime",
    "time_end_to_end",
    "time_execution",
]


def _write_metrics_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(**overrides):
    row = {
        "approach_name": "optim_global",
        "query": "find artworks",
        "precision_guarantee": 0.8,
        "recall_guarantee": 0.8,
        "precision": 0.9,
        "recall": 0.7,
        "f1_score": 0.79,
        "execution_cost_runtime": 12.0,
        "tuning_cost_runtime": 3.0,
        "total_cost_runtime": 15.0,
        "time_end_to_end": 20.0,
        "time_execution": 12.0,
    }
    row.update(overrides)
    return row


# ── Discovery ─────────────────────────────────────────────────────────────────


def test_discovers_benchmark_split_dirs(tmp_path):
    d = tmp_path / "artwork_random" / "dev"
    d.mkdir(parents=True)
    _write_metrics_csv(d / "silver_metrics.csv", [_row()])

    found = results.discover_result_dirs([tmp_path])
    assert len(found) == 1
    assert found[0]["benchmark"] == "artwork_random"
    assert found[0]["split"] == "dev"
    assert found[0]["files"][0]["labels_type"] == "silver_metrics"


def test_discover_ignores_directories_without_metrics_csv(tmp_path):
    (tmp_path / "empty_dir").mkdir()
    assert results.discover_result_dirs([tmp_path]) == []


def test_discover_nonexistent_root_returns_empty(tmp_path):
    assert results.discover_result_dirs([tmp_path / "does_not_exist"]) == []


# ── Loading ───────────────────────────────────────────────────────────────────


def test_loads_all_required_columns(tmp_path):
    d = tmp_path / "b" / "dev"
    d.mkdir(parents=True)
    _write_metrics_csv(d / "silver_metrics.csv", [_row(), _row(query="find other art")])

    data = results.load_result_dir(d)
    assert "error" not in data
    assert len(data["rows"]) == 2
    assert data["approaches"] == ["optim_global"]


def test_missing_directory_reports_structured_error(tmp_path):
    data = results.load_result_dir(tmp_path / "nope")
    assert "error" in data


def test_directory_without_metrics_csv_reports_structured_error(tmp_path):
    (tmp_path / "empty").mkdir()
    data = results.load_result_dir(tmp_path / "empty")
    assert "error" in data


def test_missing_required_column_is_reported_not_raised(tmp_path):
    d = tmp_path / "b" / "dev"
    d.mkdir(parents=True)
    rows = [_row()]
    del rows[0]["recall"]
    _write_metrics_csv(d / "silver_metrics.csv", rows)

    data = results.load_result_dir(d)
    assert "error" in data
    assert "recall" in data["error"]


def test_gold_metrics_is_optional(tmp_path):
    d = tmp_path / "b" / "dev"
    d.mkdir(parents=True)
    _write_metrics_csv(d / "silver_metrics.csv", [_row()])

    data = results.load_result_dir(d)
    assert data["labels_types"] == ["silver_metrics"]


def test_both_labels_types_are_loaded_and_tagged(tmp_path):
    d = tmp_path / "b" / "dev"
    d.mkdir(parents=True)
    _write_metrics_csv(d / "silver_metrics.csv", [_row()])
    _write_metrics_csv(d / "gold_metrics.csv", [_row(precision=1.0)])

    data = results.load_result_dir(d)
    assert sorted(data["labels_types"]) == ["gold_metrics", "silver_metrics"]


# ── Derived columns (must match plot_benchmark.py's semantics) ───────────────


def test_none_guarantees_are_filled_with_the_smallest_observed_target(tmp_path):
    """Mirrors plot_benchmark.load_df: NaN guarantee rows adopt the lowest real one."""
    d = tmp_path / "b" / "dev"
    d.mkdir(parents=True)
    rows = [
        _row(approach_name="optim_global", precision_guarantee=0.9, recall_guarantee=0.9),
        _row(approach_name="gpt", precision_guarantee=None, recall_guarantee=None),
    ]
    _write_metrics_csv(d / "silver_metrics.csv", rows)

    data = results.load_result_dir(d)
    gpt_row = next(r for r in data["rows"] if r["approach_name"] == "gpt")
    assert gpt_row["precision_guarantee"] == 0.9
    assert gpt_row["recall_guarantee"] == 0.9


def test_target_met_ratio_matches_plot_benchmark_formula(tmp_path):
    d = tmp_path / "b" / "dev"
    d.mkdir(parents=True)
    _write_metrics_csv(
        d / "silver_metrics.csv",
        [_row(precision=0.6, precision_guarantee=0.8, recall=0.9, recall_guarantee=0.9)],
    )

    data = results.load_result_dir(d)
    row = data["rows"][0]
    assert row["precision_met"] == pytest.approx(0.75)
    assert row["recall_met"] == pytest.approx(1.0)


def test_guarantee_setting_label_is_stable_for_grouping(tmp_path):
    d = tmp_path / "b" / "dev"
    d.mkdir(parents=True)
    _write_metrics_csv(
        d / "silver_metrics.csv", [_row(precision_guarantee=0.7, recall_guarantee=0.9)]
    )
    data = results.load_result_dir(d)
    assert data["rows"][0]["guarantee_setting"] == "p:0.7_r:0.9"


def test_time_other_is_end_to_end_minus_named_components_clipped_at_zero(tmp_path):
    d = tmp_path / "b" / "dev"
    d.mkdir(parents=True)
    _write_metrics_csv(
        d / "silver_metrics.csv",
        [
            _row(
                time_end_to_end=10.0,
                **{
                    "time_execution": 3.0,
                    "time_profiling": 1.0,
                    "time_optimization": 1.0,
                    "time_configuring": 1.0,
                    "time_reasoning": 1.0,
                },
            )
        ],
    )
    data = results.load_result_dir(d)
    assert data["rows"][0]["time_other"] == pytest.approx(3.0)


def test_nan_values_serialize_as_none_not_nan_literal(tmp_path):
    d = tmp_path / "b" / "dev"
    d.mkdir(parents=True)
    rows = [_row()]
    rows[0]["f1_score"] = float("nan")
    _write_metrics_csv(d / "silver_metrics.csv", rows)

    data = results.load_result_dir(d)
    assert data["rows"][0]["f1_score"] is None


# ── operator_stats.csv ────────────────────────────────────────────────────────


def test_operator_stats_loaded_when_present(tmp_path):
    d = tmp_path / "b" / "dev"
    d.mkdir(parents=True)
    _write_metrics_csv(d / "silver_metrics.csv", [_row()])
    pd.DataFrame(
        [{"approach": "optim_global", "operator": "TextQaFilter", "count": 4}]
    ).to_csv(d / "operator_stats.csv", index=False)

    data = results.load_result_dir(d)
    assert data["operator_stats"]["available"] is True
    assert data["operator_stats"]["rows"][0]["count"] == 4


def test_operator_stats_missing_file_is_not_an_error(tmp_path):
    d = tmp_path / "b" / "dev"
    d.mkdir(parents=True)
    _write_metrics_csv(d / "silver_metrics.csv", [_row()])

    data = results.load_result_dir(d)
    assert data["operator_stats"] == {"rows": [], "available": False}


# ── Presentation constants (must never crash if seaborn/matplotlib is absent) ─


def test_presentation_falls_back_gracefully_without_plotting_module(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "reasondb.evaluation.plotting":
            raise ImportError("simulated missing seaborn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    payload = results.presentation()
    assert payload["label_map"] == {}
    assert payload["dataset_order"] == []
    assert len(payload["breakdown_components"]) == 6
