"""kvop01's arithmetic: one KV operator's speedup against the vanilla-only arm, and its cost.

The speedup itself is ``evaluation.ablation_gains``, tested on its own - it pairs on
``(benchmark, query, guarantee_setting)`` and refuses to divide two aggregates over
different query sets. What these pin is the loop over every
compressed arm against the one reference, the arm labels that make a state readable as an
operator, and the join back to what each operator costs to keep.
"""

import json

import pytest

try:
    from reasondb.evaluation import kv_operator_gains as kvgains
except ImportError:  # pragma: no cover
    pytest.skip("evaluation deps not installed", allow_module_level=True)

import pandas as pd

from reasondb.evaluation.sweep_frames import VANILLA_ONLY_ARM

ARMS = [
    # (step, retained slot, ratio, storage GB, cached entries, runtime)
    (0, None, None, 0.0, 0, 100.0),
    (1, "text_small", 0.5, 8.0, 1000, 40.0),
    (2, "text_large", 0.8, 4.0, 1000, 80.0),
]


def _frame(queries=("q1", "q2"), targets=(0.5, 0.9)) -> pd.DataFrame:
    rows = []
    for step, slot, cr, storage_gb, entries, runtime in ARMS:
        for target in targets:
            for query in queries:
                row = {
                    "benchmark": "movie_random",
                    "dataset": "movie_random",
                    "split": "dev",
                    "step": step,
                    "approach": "optim_global",
                    "query": query,
                    "precision_guarantee": target,
                    "recall_guarantee": target,
                    "guarantee_setting": f"p:{target}_r:{target}",
                    "execution_runtime_s": runtime,
                    "tuning_runtime_s": 5.0,
                    "total_runtime_s": runtime + 5.0,
                    "wall_clock_s": runtime + 6.0,
                    "storage_gb": storage_gb,
                    "storage_bytes": storage_gb * 1024**3,
                    "cache_entries": entries,
                    "storage_bytes_per_entry": (
                        storage_gb * 1024**3 / entries if entries else None
                    ),
                    "num_tuples": 1000,
                    "modality": "text",
                }
                for key in ("text_small", "text_large", "image_small", "image_large"):
                    row[f"{key}_crs"] = (
                        json.dumps([cr]) if slot == key else None
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def test_arms_are_named_for_the_operator_a_state_holds():
    """Not for the step index: the arms have to be comparable between two benchmarks
    whose materialized grids differ, and recognisable when one has no image slots."""
    armed = kvgains.with_arms(_frame())

    assert set(armed["arm"]) == {VANILLA_ONLY_ARM, "text-S cr0.5", "text-L cr0.8"}
    assert kvgains.compressed_arms(armed) == ["text-S cr0.5", "text-L cr0.8"]


def test_every_compressed_arm_is_paired_against_the_one_reference():
    pairs = kvgains.all_pairs(kvgains.with_arms(_frame()))

    assert set(pairs["arm"]) == {"text-S cr0.5", "text-L cr0.8"}
    # Two queries x two targets per arm, each paired with the reference's own run.
    assert len(pairs) == 8
    assert pairs.loc[pairs["arm"] == "text-S cr0.5", "execution_speedup"].eq(2.5).all()
    assert pairs.loc[pairs["arm"] == "text-L cr0.8", "execution_speedup"].eq(1.25).all()


def test_a_query_the_reference_never_ran_is_dropped_rather_than_filled():
    """A speedup needs both halves, and the plan enumerates them over the same grid - so
    a missing half is a failed job, not a shape of the experiment.

    Counted rather than named: ``paired_speedups`` pairs on (benchmark, query, target) and
    then drops those keys, so what survives is the count and the target column.
    """
    frame = kvgains.with_arms(_frame())
    frame = frame[~((frame["arm"] == VANILLA_ONLY_ARM) & (frame["query"] == "q2"))]

    pairs = kvgains.all_pairs(frame)

    # One query instead of two, still at both targets, for each of the two arms.
    assert len(pairs) == 4
    assert set(pairs["arm"]) == {"text-S cr0.5", "text-L cr0.8"}


def test_an_arm_with_no_reference_at_all_contributes_nothing():
    frame = kvgains.with_arms(_frame())
    frame = frame[frame["arm"] != VANILLA_ONLY_ARM]

    assert kvgains.all_pairs(frame).empty


def test_the_summary_carries_what_each_operator_costs_to_keep():
    """The whole point of the table: manageable on one axis, beneficial on the other."""
    summary = kvgains.summary(kvgains.with_arms(_frame()))

    assert set(summary["arm"]) == {"text-S cr0.5", "text-L cr0.8"}
    small = summary[summary["arm"] == "text-S cr0.5"].iloc[0]
    assert small["median"] == pytest.approx(2.5)
    assert small["kv_model_size"] == "small"
    assert small["kv_cr"] == 0.5
    assert small["storage_gb"] == 8.0
    # 8 GiB over 1000 items, reported in MB because a text item runs to a few of them.
    assert small["storage_mb_per_entry"] == pytest.approx(8 * 1024 / 1000)


def test_the_summary_is_empty_when_nothing_pairs():
    """The caller then says so, rather than drawing an empty plane."""
    frame = kvgains.with_arms(_frame())
    assert kvgains.summary(frame[frame["arm"] != VANILLA_ONLY_ARM]).empty


def test_a_state_holding_several_levels_describes_no_single_operator():
    """A greedy state has none, and inventing one would put a column on those rows that
    reads as a fact and is a choice."""
    from reasondb.evaluation.sweep_frames import add_kv_operator_columns

    frame = _frame()
    frame.loc[frame["step"] == 1, "text_small_crs"] = json.dumps([0.0, 0.5])

    described = add_kv_operator_columns(frame)
    multi = described[described["step"] == 1]

    assert multi["kv_model_size"].isna().all()
    assert multi["kv_cr"].isna().all()
    assert set(multi["kv_operator_label"]) == {"2 levels"}


def test_a_state_holding_one_level_per_modality_names_both():
    """A kv_operator_pairs state has an operator in each modality and the label says
    which; the single-operator columns stay empty, since there is no one operator."""
    from reasondb.evaluation.sweep_frames import add_kv_operator_columns

    frame = _frame()
    frame.loc[frame["step"] == 1, "text_small_crs"] = json.dumps([0.8])
    frame.loc[frame["step"] == 1, "image_small_crs"] = json.dumps([0.9])

    described = add_kv_operator_columns(frame)
    pair = described[described["step"] == 1]

    assert set(pair["kv_operator_label"]) == {"text-S cr0.8 + image-S cr0.9"}
    assert pair["kv_cr"].isna().all()
