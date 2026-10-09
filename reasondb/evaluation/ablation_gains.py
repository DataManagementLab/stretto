"""What the KV-compressed operators buy, read off the ablation's two optimizer arms.

The ablation measures two steps: arm 3 -> arm 2 removes the optimizer, arm 2 -> arm 1
removes the KV-compressed operators. This module breaks the second step down beyond the
pooled total, which hides two effects:

* **A total is execution plus profiling, and compression moves them in opposite
  directions.** More operators in the search space is more to profile, so the arm that
  executes faster also tunes slower. Profiling is a *fixed* per-query cost - it draws a
  sample, not a table - so it does not scale with the data while the execution saving
  does. :func:`phase_ledger` separates the two and :func:`amortization` projects them.
* **A pooled mean is one number over a distribution that is not centred.** Compression
  cannot help a query whose guarantee forces the gold operator, so roughly half the
  queries move not at all and the rest move a lot. The mean over that is small and says
  nothing about either half; :func:`paired_speedups` keeps the per-query pairs so the
  shape can be reported instead.

Everything here is *paired*: a row is one (benchmark, query, guarantee) run by both arms,
so a speedup is that query's own, never a ratio of two aggregates over possibly different
query sets. Pandas only - no matplotlib - so the arithmetic is testable without a display,
as in :mod:`reasondb.evaluation.sweep_frames`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from reasondb.evaluation.sweep_frames import fan_out_guarantee_blind_rows

#: The three arms, as :func:`sweep_frames._ablation_arm` labels them.
NO_OPTIM_ARM = "No optimization"
VANILLA_ARM = "Stretto, vanilla only"
FULL_ARM = "Stretto"

#: The two steps the ablation measures, each ``(slower arm, faster arm)`` so that
#: ``slower / faster`` is a speedup above 1.0 when the added component helped.
STEPS = {
    "kv": (VANILLA_ARM, FULL_ARM),
    "optimizer": (NO_OPTIM_ARM, VANILLA_ARM),
}

#: Default step: the KV step.
BASE_ARM = VANILLA_ARM
WITH_ARM = FULL_ARM

#: Columns a paired row carries through, with the name each takes per arm.
_PAIRED_METRICS = {
    "total": "total_runtime_s",
    "execution": "execution_runtime_s",
    "tuning": "tuning_runtime_s",
    "wall": "wall_clock_s",
}

#: How far from 1.0 a speedup has to be before it counts as a change rather than as the
#: same plan measured twice. One percent: the arms share a configurator and a replayed
#: clock, so an identical plan lands within a few parts in 10^5, and anything this side of
#: 1% is a query neither arm did differently.
UNCHANGED_BAND = 0.01

#: What a query is keyed on. The guarantee is part of the key rather than averaged over:
#: the same query at two targets is two different optimization problems.
PAIR_KEYS = ["benchmark", "query", "guarantee_setting"]


def paired_speedups(
    df: pd.DataFrame,
    arm_col: str = "arm",
    base_arm: str = BASE_ARM,
    with_arm: str = WITH_ARM,
    fan_out_blind: bool = True,
) -> pd.DataFrame:
    """One row per (query, guarantee) with both arms' costs and their ratios.

    The frame must already carry the ablation ``arm`` labels. Rows only one arm ran are
    dropped rather than filled: a speedup needs both halves, and the arms are enumerated
    over the same grid so a missing half is a failed job, not a shape of the experiment.

    Ratios are ``base / with``, i.e. **above 1.0 means the added component made it
    faster**. A zero or missing denominator yields NaN rather than an infinity, so a
    single degenerate row cannot dominate a mean.

    *fan_out_blind* replicates the guarantee-blind arm across the targets the optimizing
    arms ran at. ``LabelOptimizer`` never reads the guarantees, so its single measured run
    is the correct baseline at *every* target; without the fan-out the optimizer step would
    be confined to the one target that arm was enumerated at.
    """
    if fan_out_blind:
        df = fan_out_guarantee_blind_rows(df)
    base = df[df[arm_col] == base_arm].set_index(PAIR_KEYS)
    with_kv = df[df[arm_col] == with_arm].set_index(PAIR_KEYS)
    common = base.index.intersection(with_kv.index)
    if not len(common):
        return pd.DataFrame()

    out = pd.DataFrame(index=common)
    for carry in ("dataset", "num_semops", "num_sem_filter", "num_sem_join"):
        if carry in base.columns:
            out[carry] = base.loc[common, carry]
    out["guarantee_setting"] = [key[2] for key in common]

    for name, column in _PAIRED_METRICS.items():
        if column not in base.columns:
            continue
        lhs = pd.to_numeric(base.loc[common, column], errors="coerce")
        rhs = pd.to_numeric(with_kv.loc[common, column], errors="coerce")
        out[f"{name}_base"] = lhs
        out[f"{name}_with"] = rhs
        out[f"{name}_speedup"] = np.where(rhs > 0, lhs / rhs.where(rhs > 0), np.nan)

    for name in ("achieved_f1", "achieved_precision", "achieved_recall"):
        if name in base.columns:
            out[f"{name}_base"] = base.loc[common, name]
            out[f"{name}_with"] = with_kv.loc[common, name]
    for name in ("precision_guarantee", "recall_guarantee"):
        if name in base.columns:
            out[name] = base.loc[common, name]
    return out.reset_index(drop=True)


def speedup_summary(
    pairs: pd.DataFrame, metric: str = "execution", by: Sequence[str] = ()
) -> pd.DataFrame:
    """Median / mean / tail percentiles of one metric's speedup, optionally grouped.

    Percentiles are reported because the population is typically bimodal: queries the
    added component could help and queries it could not.

    "Faster" uses a threshold: a query resolved to the same plan on both arms has a ratio
    of 1.0 up to clock noise, so the split is three-way against :data:`UNCHANGED_BAND`
    (slower, unchanged, faster). ``share_faster`` is the plain ``> 1.0`` share;
    ``share_faster_meaningful`` applies the band.
    """
    column = f"{metric}_speedup"
    if column not in pairs.columns:
        return pd.DataFrame()

    def block(group: pd.DataFrame) -> pd.Series:
        values = group[column].dropna()
        if values.empty:
            return pd.Series(dtype=float)
        return pd.Series(
            {
                "queries": float(len(values)),
                "median": values.median(),
                "mean": values.mean(),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.90),
                "p95": values.quantile(0.95),
                "max": values.max(),
                "share_faster": float((values > 1.0).mean()),
                "share_slower": float((values < 1.0 - UNCHANGED_BAND).mean()),
                "share_unchanged": float(
                    ((values >= 1.0 - UNCHANGED_BAND) & (values <= 1.0 + UNCHANGED_BAND)).mean()
                ),
                "share_faster_meaningful": float((values > 1.0 + UNCHANGED_BAND).mean()),
                "share_1_5x": float((values > 1.5).mean()),
                "share_2x": float((values > 2.0).mean()),
            }
        )

    if not by:
        return block(pairs).to_frame(metric).T
    return (
        pairs.groupby(list(by), dropna=False, sort=False)
        .apply(block, include_groups=False)
        .reset_index()
    )


def phase_ledger(
    df: pd.DataFrame,
    arm_col: str = "arm",
    base_arm: str = BASE_ARM,
    with_arm: str = WITH_ARM,
) -> Dict[str, float]:
    """Hours of execution and profiling per arm, over the rows **both** arms ran.

    Reports how much execution the added component saved, and how much of that saving its
    extra profiling cost.

    Restricted to the paired rows: the unoptimized arm is enumerated once per benchmark
    (``COLLAPSE_GUARANTEE_AXIS``) while the optimizer arms run at several guarantee
    targets, so summing each arm over its own rows would compare different run sets.
    """
    pairs = paired_speedups(df, arm_col, base_arm, with_arm)
    if pairs.empty:
        return {}
    hours = pd.DataFrame(
        {
            base_arm: {
                "execution_runtime_s": pairs["execution_base"].sum(),
                "tuning_runtime_s": pairs["tuning_base"].sum(),
                "total_runtime_s": pairs["total_base"].sum(),
            },
            with_arm: {
                "execution_runtime_s": pairs["execution_with"].sum(),
                "tuning_runtime_s": pairs["tuning_with"].sum(),
                "total_runtime_s": pairs["total_with"].sum(),
            },
        }
    ).T / 3600.0
    BASE_ARM_, WITH_ARM_ = base_arm, with_arm
    saved = hours.loc[BASE_ARM_, "execution_runtime_s"] - hours.loc[WITH_ARM_, "execution_runtime_s"]
    tax = hours.loc[WITH_ARM_, "tuning_runtime_s"] - hours.loc[BASE_ARM_, "tuning_runtime_s"]
    return {
        "execution_base_h": hours.loc[BASE_ARM_, "execution_runtime_s"],
        "execution_with_h": hours.loc[WITH_ARM_, "execution_runtime_s"],
        "tuning_base_h": hours.loc[BASE_ARM_, "tuning_runtime_s"],
        "tuning_with_h": hours.loc[WITH_ARM_, "tuning_runtime_s"],
        "total_base_h": hours.loc[BASE_ARM_, "total_runtime_s"],
        "total_with_h": hours.loc[WITH_ARM_, "total_runtime_s"],
        "execution_saved_h": saved,
        "profiling_tax_h": tax,
        "tax_share_of_saving": tax / saved if saved else np.nan,
        "execution_ratio": (
            hours.loc[BASE_ARM_, "execution_runtime_s"] / hours.loc[WITH_ARM_, "execution_runtime_s"]
        ),
        "total_ratio": (
            hours.loc[BASE_ARM_, "total_runtime_s"] / hours.loc[WITH_ARM_, "total_runtime_s"]
        ),
    }


def amortization(ledger: Dict[str, float], scales: Sequence[float]) -> pd.DataFrame:
    """Total-runtime ratio if the tables were *scale* times larger, profiling unchanged.

    Profiling draws a fixed sample whatever the table holds, so it is the one phase that
    does **not** grow with the data; execution is the one that does. The pooled total
    therefore understates compression by an amount that depends on data size, and the
    ratio approaches ``execution_ratio`` from below.

    A projection, not a measurement: it assumes execution scales linearly, which holds for
    per-row operators but not for joins.
    """
    if not ledger:
        return pd.DataFrame()
    rows: List[Dict[str, float]] = []
    for scale in scales:
        base = ledger["execution_base_h"] * scale + ledger["tuning_base_h"]
        with_kv = ledger["execution_with_h"] * scale + ledger["tuning_with_h"]
        rows.append({"scale": scale, "total_ratio": base / with_kv})
    return pd.DataFrame(rows)


def guarantee_costs(df: pd.DataFrame, arm_col: str = "arm") -> pd.DataFrame:
    """Violation rate and mean F1 per arm and guarantee target.

    Reported beside the speedups because a faster arm may trade execution time for a
    higher risk of missing the guarantee.
    """
    frame = df.copy()
    frame["violated"] = (
        (frame["achieved_precision"] < frame["precision_guarantee"])
        | (frame["achieved_recall"] < frame["recall_guarantee"])
    ).astype(float)
    return (
        frame.groupby([arm_col, "guarantee_setting"], dropna=False)
        .agg(
            queries=("violated", "size"),
            violation_rate=("violated", "mean"),
            mean_f1=("achieved_f1", "mean"),
        )
        .reset_index()
    )
