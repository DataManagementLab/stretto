"""Manageable against beneficial: one KV operator's disk cost beside its speedup.

``scripts/plot_sweep.py --experiment kvop01`` draws the experiment's arms as totals. This
module computes, per operator, the speedup *ratio* against the vanilla-only arm and joins
it with what that operator costs to keep, i.e. a cost-benefit plane.

The arithmetic is :mod:`reasondb.evaluation.ablation_gains`:
:func:`~reasondb.evaluation.ablation_gains.paired_speedups` takes the two arm labels to
divide, pairs on ``(benchmark, query, guarantee_setting)`` and refuses to divide two
aggregates over different query sets. This module loops it over every compressed arm
against the one reference and joins the result back to what each arm costs.

Pandas only, no matplotlib, as its sibling is: the arithmetic is testable without a
display, and ``scripts/plot_kv_operator_gains.py`` is the drawing half.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from reasondb.evaluation import ablation_gains as gains
from reasondb.evaluation.sweep_frames import (
    VANILLA_ONLY_ARM,
    add_kv_operator_columns,
)

#: What each arm costs, carried from the sweep rows onto the summary. Constant within an
#: arm and a dataset by construction - a state has one footprint - so the first value is
#: the value, and a disagreement is a merged frame that pooled two tasks.
COST_COLUMNS = (
    "storage_gb",
    "storage_bytes",
    "cache_entries",
    "storage_bytes_per_entry",
    "num_tuples",
    "tokens_per_item",
    "modality",
    "kv_modality",
    "kv_model_size",
    "kv_cr",
)

#: Bytes to megabytes, for the per-item axis. A text item's cache runs to a few MB and an
#: image item's to tens; in bytes both are nine-digit numbers on a tick label.
BYTES_TO_MB = 1.0 / (1024**2)


def with_arms(df: pd.DataFrame) -> pd.DataFrame:
    """The sweep frame with its ``arm`` column set to the KV operator each state holds."""
    out = add_kv_operator_columns(df)
    out["arm"] = out["kv_operator_label"].astype(str)
    return out


def compressed_arms(df: pd.DataFrame) -> List[str]:
    """Every arm but the vanilla-only reference, in the frame's own order."""
    return [
        arm
        for arm in pd.unique(df["arm"].astype(str))
        if arm != VANILLA_ONLY_ARM
    ]


def paired_against_vanilla(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """One compressed arm's per-query speedups against the vanilla-only arm.

    ``fan_out_blind=False``: both arms here are ``optim_global`` and both read the
    guarantees, so there is no guarantee-blind arm to replicate across targets.
    """
    pairs = gains.paired_speedups(
        df, base_arm=VANILLA_ONLY_ARM, with_arm=arm, fan_out_blind=False
    )
    if not pairs.empty:
        pairs["arm"] = arm
    return pairs


def all_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Every compressed arm paired against the reference, in one frame.

    An arm the reference cannot be paired with is dropped rather than filled, which is
    :func:`~reasondb.evaluation.ablation_gains.paired_speedups`' own rule: a speedup needs
    both halves, and the plan enumerates them over the same grid, so a missing half is a
    failed job rather than a shape of the experiment.
    """
    frames = [
        pairs
        for arm in compressed_arms(df)
        if not (pairs := paired_against_vanilla(df, arm)).empty
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    """``(dataset, arm) -> what keeping that operator costs``.

    Read off the sweep rows rather than recomputed, so the cost a point is drawn at is the
    cost the run measured. ``storage_mb_per_entry`` is the column the cost axis actually
    wants: a footprint divided by the items behind it, which is the only form of this
    number that is comparable across datasets.
    """
    present = [column for column in COST_COLUMNS if column in df.columns]
    table = (
        df.groupby(["dataset", "arm"], dropna=False, sort=False)[present]
        .first()
        .reset_index()
    )
    if "storage_bytes_per_entry" in table.columns:
        table["storage_mb_per_entry"] = (
            pd.to_numeric(table["storage_bytes_per_entry"], errors="coerce") * BYTES_TO_MB
        )
    else:
        table["storage_mb_per_entry"] = np.nan
    return table


def summary(
    df: pd.DataFrame,
    metric: str = "execution",
    by: Sequence[str] = ("dataset", "arm", "guarantee_setting"),
) -> pd.DataFrame:
    """One row per (dataset, arm, target): the speedup distribution and the disk cost.

    ``median`` and ``share_faster_meaningful`` say whether the operator paid off;
    ``storage_mb_per_entry`` says what it costs to keep.

    ``execution`` rather than ``total`` by default, and the difference is not cosmetic:
    a total nets the execution saving against the profiling an extra operator costs, and
    profiling draws a fixed sample while execution scales with the table. See
    :func:`~reasondb.evaluation.ablation_gains.amortization`, which projects the one into
    the other.
    """
    pairs = all_pairs(df)
    if pairs.empty:
        return pd.DataFrame()
    grouped = gains.speedup_summary(pairs, metric=metric, by=by)
    if grouped.empty:
        return grouped
    costs = cost_table(df)
    join_on = [column for column in ("dataset", "arm") if column in grouped.columns]
    return grouped.merge(costs, on=join_on, how="left")
