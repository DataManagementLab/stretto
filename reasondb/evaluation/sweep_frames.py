"""Reading and aggregating the coordinator's merged sweep CSVs.

Every sweep producer - ``baselines``, ``sample_size``, ``operator_count``,
``adaptive_sampling``, ``ablation`` and the bare ``parameter_sweep`` engine - lands on
one schema at ``merged/<benchmark>/<split>/<producer>.csv``
(``coordinator.producers.parameter_sweep.merge``). This module is the reader for it,
and deliberately holds **no matplotlib or seaborn import**: the aggregation below is the
part that decides what a headline number means, so it has to be testable without a
display, a font cache or ``scienceplots``. ``sweep_figures`` is the drawing half.

It provides:

* the ``<benchmark>/<split>/<file>`` discovery and the ``guarantee_setting`` /
  ``dataset`` derivations;
* the explosion of the JSON ``component_times`` column into per-phase columns;
* the cross-dataset aggregation, which is drawn both ways - a *geometric* mean
  (:func:`geomean_shares`), which is scale-free and so reports the system rather than
  whichever benchmark is slowest, and a plain sum (:func:`sum_overall`), which is the
  fleet's actual bill. :data:`POOLINGS` names them; ``plot_sweep.py --pooling`` picks.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from reasondb.monitor.dimensions import DIMENSION_LABELS
from reasondb.monitor.phases import (
    RUNTIME_BREAKDOWN_COMPONENTS,
    derive_phase_components,
)

logger = logging.getLogger(__name__)

#: The six non-overlapping phase columns, bottom-to-top as they stack. This is what a CSV
#: is *parsed* into, and it stays complete: dropping a phase here would silently change
#: what ``time_end_to_end`` is checked against.
PHASE_COLUMNS: List[str] = [column for column, _ in RUNTIME_BREAKDOWN_COMPONENTS]
PHASE_LABELS: Dict[str, str] = dict(RUNTIME_BREAKDOWN_COMPONENTS)

#: The phases the runtime breakdown *draws*, in stacking order. ``Configuring``,
#: ``Reasoning`` and ``Other`` are omitted: they are plan bookkeeping rather than work the
#: optimizer trades off. A bar is therefore the sum of these three phases, **not**
#: end-to-end runtime; the filter is applied at the totals so segments, bar and pooled
#: geometric mean all cover the same phases.
BREAKDOWN_PHASE_COLUMNS: List[str] = [
    "time_execution",
    "time_profiling",
    "time_optimization",
]

#: The four model slots a sweep state can retain compression levels for. Mirrors
#: ``evaluation.parameter_sweep.ALL_SLOT_KEYS``; duplicated rather than imported because
#: that module pulls in the whole execution stack and this one is a CSV reader.
SLOT_KEYS = ["text_small", "text_large", "image_small", "image_large"]

#: The label used for the synthetic cross-dataset facet. ``DATASET_ORDER`` in
#: ``evaluation.plotting`` already lists it first, so it lands leftmost for free.
OVERALL = "Overall"

#: How a pooled panel combines the datasets it stands for. The geometric mean is
#: scale-free (an arm that costs *k* times another on every benchmark shows a ratio of
#: *k*); the sum is the actual total cost, dominated by the largest benchmark.
POOLINGS: Tuple[str, ...] = ("geomean", "sum")

#: What each pooling is called on the pooled panel's own title, so a figure cannot be
#: read as the other one.
POOLING_NOTES: Dict[str, str] = {"geomean": "geo. mean", "sum": "sum", "mean": "mean"}


# ---------------------------------------------------------------------------
# Discovery and derived columns
# ---------------------------------------------------------------------------


def find_sweep_csvs(
    output_dirs: Sequence[Path],
    csv_name: str,
    split: str = "dev",
    benchmarks: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Locate ``<output-dir>/<benchmark>/<split>/<csv_name>`` under each output dir.

    With no ``benchmarks`` this globs, which is what makes "plot whatever the task
    produced" the default; naming them turns it into an exact lookup so a typo is a
    missing file rather than a silently smaller figure.
    """
    paths: List[Path] = []
    for output_dir in output_dirs:
        if benchmarks:
            for benchmark in benchmarks:
                candidate = Path(output_dir) / benchmark / split / csv_name
                if candidate.is_file():
                    paths.append(candidate)
                else:
                    logger.warning("No %s", candidate)
        else:
            paths.extend(sorted(Path(output_dir).glob(f"*/{split}/{csv_name}")))
    return paths


def parse_crs(value: Any) -> List[float]:
    """Parse one ``<slot>_crs`` cell into a list of retained compression ratios.

    The cell is ``json.dumps``'d by the sweep writer and ``None`` for a slot the dataset
    has no modality for, but the same frame read back from the ``.parquet`` twin carries
    a real list - so both spellings have to work.
    """
    if value is None or isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        logger.warning("Unparseable compression-ratio cell %r; treating as empty", value)
        return []
    if not isinstance(parsed, (list, tuple)):
        return [float(parsed)]
    return [float(v) for v in parsed]


#: How one retained slot is named on the axis of a ``kv_operator`` figure. The slot keys
#: say small and large rather than 8B and 70B, and so does this: which model fills a slot
#: is a flag on the run (``--text-large-model``), not something a merged CSV records, and a
#: figure that printed "70B" from a column that never said so would be wrong the first time
#: someone swept the model.
SLOT_ARM_LABELS: Dict[str, str] = {
    "text_small": "text-S",
    "text_large": "text-L",
    "image_small": "image-S",
    "image_large": "image-L",
}

#: The arm of a state that retains nothing: the vanilla operators alone.
VANILLA_ONLY_ARM = "vanilla only"


def retained_levels(row: pd.Series) -> List[Tuple[str, float]]:
    """``(slot key, ratio)`` for everything a row's state keeps, in ``SLOT_KEYS`` order."""
    return [
        (key, cr)
        for key in SLOT_KEYS
        if f"{key}_crs" in row.index
        for cr in parse_crs(row[f"{key}_crs"])
    ]


def add_kv_operator_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Describe the single KV operator a ``kv_operator`` state holds, as columns.

    ``kv_modality``, ``kv_model_size`` and ``kv_cr`` - the three axes the experiment sweeps
    - read off the one retained ``(slot, ratio)`` pair. A state retaining nothing is the
    vanilla-only reference and gets ``None``/NaN rather than a zero, which would sort as a
    compression ratio and plot as one.

    Empty for any state retaining more than one level, deliberately: a greedy state has no
    single operator to describe, and inventing one (the first, the largest) would put a
    column on those rows that reads as a fact and is a choice. The columns are a
    ``kv_operator`` reading of a frame, not a general one, and the arm builder degrades the
    same way.
    """
    out = df.copy()
    if not any(f"{key}_crs" in out.columns for key in SLOT_KEYS):
        logger.warning("No *_crs columns; the KV operator of a state is unavailable.")
        out["kv_modality"] = None
        out["kv_model_size"] = None
        out["kv_cr"] = np.nan
        # The step index is what such a frame still knows, so the arms stay distinct and
        # the figure draws - unlabelled rather than mislabelled.
        out["kv_operator_label"] = (
            out["step"].map(lambda v: f"s{v}") if "step" in out.columns else "?"
        )
        return out

    retained = out.apply(retained_levels, axis=1)
    single = retained.map(lambda pairs: pairs[0] if len(pairs) == 1 else None)
    out["kv_modality"] = single.map(
        lambda pair: pair[0].split("_")[0] if pair else None
    )
    out["kv_model_size"] = single.map(
        lambda pair: pair[0].split("_")[1] if pair else None
    )
    out["kv_cr"] = single.map(lambda pair: pair[1] if pair else np.nan)
    out["kv_operator_label"] = [_kv_operator_label(pairs) for pairs in retained]
    return out


def _kv_operator_label(pairs: List[Tuple[str, float]]) -> str:
    """``text-S cr0.8``, or ``text-S cr0.8 + image-S cr0.9`` for a ``kv_operator_pairs`` state.

    Joined only when every level sits in a different modality - one operator per modality
    is a state that plan names, and the label can then say what it is. Anything else
    (two levels of one slot, a greedy state) keeps the count, for the reason
    :func:`add_kv_operator_columns` gives for leaving its columns empty.
    """
    if not pairs:
        return VANILLA_ONLY_ARM
    modalities = [key.split("_")[0] for key, _cr in pairs]
    if len(pairs) > 1 and len(set(modalities)) < len(pairs):
        return f"{len(pairs)} levels"
    return " + ".join(f"{SLOT_ARM_LABELS.get(key, key)} cr{cr:g}" for key, cr in pairs)


def explode_component_times(
    df: pd.DataFrame, column: str = "component_times"
) -> pd.DataFrame:
    """Add the six phase columns plus ``time_end_to_end`` from the JSON ``column``.

    The split itself is :func:`reasondb.monitor.phases.derive_phase_components`, which is
    also what the live dashboard and the metrics CSV use, so the breakdowns agree.

    Never sum ``component_times.values()``: the keys nest (``end_to_end`` spans
    everything, ``tuning`` spans ``profiling``, and ``optimizer_solve`` sits inside
    ``tuning`` too), so a naive sum charges the same seconds up to three times.
    """
    out = df.copy()
    if column not in out.columns:
        logger.warning(
            "No %s column; phase breakdown columns will be zero. (A sweep CSV written "
            "before component timing existed?)",
            column,
        )
        for name in PHASE_COLUMNS + ["time_end_to_end"]:
            out[name] = 0.0
        return out

    unparseable = 0

    def _components(cell: Any) -> Dict[str, float]:
        nonlocal unparseable
        if isinstance(cell, dict):
            return derive_phase_components(cell)
        if cell is None or (isinstance(cell, float) and math.isnan(cell)):
            return derive_phase_components(None)
        try:
            parsed = json.loads(str(cell))
        except (TypeError, ValueError):
            unparseable += 1
            return derive_phase_components(None)
        if not isinstance(parsed, dict):
            # Valid JSON of the wrong shape: count it as unparseable rather than letting
            # `derive_phase_components` raise.
            unparseable += 1
            return derive_phase_components(None)
        return derive_phase_components(parsed)

    parts = out[column].map(_components)
    for name in PHASE_COLUMNS + ["time_end_to_end"]:
        out[name] = parts.map(lambda p, key=name: p[key]).astype(float)

    if unparseable:
        logger.warning(
            "%d of %d %s cells were unparseable and counted as zero.",
            unparseable,
            len(out),
            column,
        )
    return out


def add_operator_count(df: pd.DataFrame, include_vanilla: bool = True) -> pd.DataFrame:
    """Add ``n_cached_levels`` and ``n_operators`` from the per-slot ``*_crs`` columns.

    ``n_cached_levels`` counts the KV caches a state keeps materialized, which is the
    quantity the greedy walk prunes one of per step. ``n_operators`` adds the vanilla
    operator each present modality keeps regardless, so the gold-only terminal state
    counts one LLM operator per modality.

    A modality counts as present when *any* row of that dataset retains a level in
    either of its two slots, so a state that has pruned a modality down to vanilla still
    reports it.

    **The stamped column wins where there is one.** The sweep records
    ``n_llm_operators`` (``evaluation.parameter_sweep.count_llm_operators``), so the writer
    knows something this cannot: whether the state took the *small* model's vanilla
    operator. Two states with identical ``*_crs`` cells differ by one operator over that,
    which is the whole difference between the ``gold`` plan and the ablation's second arm,
    and between the two halves of a ``kv_operator`` sweep. The derivation below is the
    fallback for CSVs without the column, and assumes one vanilla operator per modality.
    """
    out = df.copy()
    present = [key for key in SLOT_KEYS if f"{key}_crs" in out.columns]
    if "n_llm_operators" in out.columns and out["n_llm_operators"].notna().any():
        out["n_cached_levels"] = (
            sum(out[f"{key}_crs"].map(lambda v: len(parse_crs(v))) for key in present)
            if present
            else pd.NA
        )
        out["n_operators"] = (
            out["n_llm_operators"] if include_vanilla else out["n_cached_levels"]
        )
        return out

    if not present:
        logger.warning("No *_crs columns; operator counts unavailable.")
        out["n_cached_levels"] = pd.NA
        out["n_operators"] = pd.NA
        return out

    counts = {key: out[f"{key}_crs"].map(lambda v: len(parse_crs(v))) for key in present}
    out["n_cached_levels"] = sum(counts.values())

    if not include_vanilla:
        out["n_operators"] = out["n_cached_levels"]
        return out

    dataset_col = "benchmark" if "benchmark" in out.columns else None
    modalities = {"text": ["text_small", "text_large"], "image": ["image_small", "image_large"]}

    def _vanilla_for(frame: pd.DataFrame) -> int:
        return sum(
            1
            for slots in modalities.values()
            if any(counts[key].loc[frame.index].sum() > 0 for key in slots if key in counts)
        )

    if dataset_col is None:
        out["n_operators"] = out["n_cached_levels"] + _vanilla_for(out)
        return out

    vanilla = out.groupby(dataset_col, sort=False).apply(
        _vanilla_for, include_groups=False
    )
    out["n_operators"] = out["n_cached_levels"] + out[dataset_col].map(vanilla).fillna(0).astype(int)
    return out


def guarantee_short_label(setting: str) -> str:
    """``"p:0.5_r:0.5"`` -> ``"0.5"``, ``"p:0.5_r:0.7"`` -> ``"0.5/0.7"``.

    A compact tick label; ``LABEL_MAP`` holds the long form ("Prec=0.5/Rec=0.5") for
    legends and facet titles. The halves collapse to one number when they are equal.
    """
    text = str(setting)
    if not text.startswith("p:") or "_r:" not in text:
        return text
    precision, _, recall = text[2:].partition("_r:")
    return precision if precision == recall else f"{precision}/{recall}"


def add_guarantee_setting(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``p:0.5_r:0.5`` key ``LABEL_MAP`` already knows how to relabel."""
    out = df.copy()
    if "guarantee_setting" in out.columns:
        return out
    out["guarantee_setting"] = [
        f"p:{p}_r:{r}"
        for p, r in zip(out["precision_guarantee"], out["recall_guarantee"])
    ]
    return out


def load_sweep(csv_paths: Sequence[Path]) -> pd.DataFrame:
    """Concatenate merged sweep CSVs and add every derived column.

    Tolerant of missing columns: a CSV may lack an axis column such as ``approach`` or
    ``sample_size``, and a task whose scoring pass never ran has ``achieved_*`` entirely
    NaN. Missing axis columns are filled with the engine's default, so such a comparison
    renders as a single bar rather than raising.
    """
    if not csv_paths:
        raise SystemExit("No sweep CSVs to load.")
    df = pd.concat([pd.read_csv(p) for p in csv_paths], axis=0, ignore_index=True)

    for column, default in (
        ("approach", "optim_global"),
        ("step", 0),
        ("tune_parameters", True),
        ("sample_size", pd.NA),
        ("adaptive_sampling", False),
        # Query shape, stamped by the merge's scoring pass; absent values draw as one "?"
        # bucket. Only `num_semops` is defaulted: `facet_panels` reports a clear error
        # for the other statistics.
        ("num_semops", pd.NA),
        # Measured beside the footprint, but absent from CSVs that do not record it, so
        # `storage_bytes_per_entry` is a column a figure has to tolerate being absent
        # rather than one it can assume.
        ("cache_entries", pd.NA),
        ("storage_bytes_per_entry", pd.NA),
        ("num_tuples", pd.NA),
        ("modality", pd.NA),
    ):
        if column not in df.columns:
            logger.info("Column %r absent from these CSVs; defaulting to %r.", column, default)
            df[column] = default

    if "dataset" not in df.columns:
        df["dataset"] = df["benchmark"]
    df = add_guarantee_setting(df)
    df = explode_component_times(df)
    df = add_operator_count(df)
    df = add_kv_operator_columns(df)
    df = add_dataset_token_stats(df)
    return df


def add_dataset_token_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Join the pinned per-benchmark token statistics on ``benchmark``.

    How long an item is decides what a KV cache of it costs, and it is a property of the
    dataset rather than of the run - so it is pinned once by
    ``scripts/dataset_token_stats.py`` and joined here rather than stamped on a row, which
    makes it available to every merged CSV on re-plot, with no re-run.

    Absent statistics leave the columns empty rather than raising, exactly as a missing
    axis column does above.
    """
    from reasondb.evaluation.dataset_stats import load_token_stats, tokens_per_item

    out = df.copy()
    stats = load_token_stats()
    per_benchmark = {
        name: tokens_per_item(stats, name) for name in out["benchmark"].unique()
    }
    for column in ("tokens_per_item", "tokens_per_item_p95"):
        out[column] = out["benchmark"].map(
            lambda name: per_benchmark.get(name, {}).get(column, np.nan)
        )
    # num_tuples is stamped on rows by the sweep; for CSVs without it the pinned
    # statistics are the only place it exists, and they agree by construction (both
    # count the same table files).
    if "num_tuples" in out.columns and out["num_tuples"].isna().all():
        out["num_tuples"] = out["benchmark"].map(
            lambda name: (stats.get(name) or {}).get("num_tuples", np.nan)
        )
    if "modality" in out.columns and out["modality"].isna().all():
        out["modality"] = out["benchmark"].map(
            lambda name: (stats.get(name) or {}).get("modality")
        )
    return out


def _panel_key(value: Any) -> Tuple[float, str]:
    """Sort a panel name by its leading number, if it has one.

    Plain ``sorted`` is lexicographic, which would put "10 sem. ops" before "2 sem. ops".
    """
    text = str(value)
    try:
        return (float(text.split(" ", 1)[0]), text)
    except ValueError:
        return (math.inf, text)


def dataset_col_order(df: pd.DataFrame, column: str = "dataset") -> List[str]:
    """Facet order: the curated ``DATASET_ORDER`` first, then anything else sorted."""
    from reasondb.evaluation.plotting import DATASET_ORDER

    present = list(pd.unique(df[column]))
    ordered = [d for d in DATASET_ORDER if d in present]
    ordered += [d for d in sorted(present, key=_panel_key) if d not in ordered]
    return ordered


#: Panel columns other than the dataset, and how one bucket is titled.
#:
#: The human label goes in the *value*, because ``sweep_figures._relabel_facet_titles``
#: strips the ``dataset = `` prefix off any title ``LABEL_MAP`` does not recognise.
FACET_LABELS: Dict[str, Callable[[Any], str]] = {
    "num_semops": lambda v: f"{int(float(v))} sem. ops",
    "num_sem_filter": lambda v: f"{int(float(v))} filters",
    "num_sem_extract": lambda v: f"{int(float(v))} extracts",
    "num_sem_join": lambda v: f"{int(float(v))} joins",
    "num_tradops": lambda v: f"{int(float(v))} trad. ops",
}


def facet_panels(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Panel on *column* instead of the dataset, keeping ``benchmark`` for the pooling.

    The figures panel on a column named ``dataset``, so this moves the bucket into that
    slot. The benchmark is kept because a panel spans every dataset and the aggregation
    pools them by geometric mean (as :func:`geomean_shares` does for the Overall panel).
    """
    if column not in df.columns or df[column].isna().all():
        raise SystemExit(
            f"No usable {column!r} column in these CSVs. Query shape statistics reach a "
            "sweep CSV through the merge's scoring pass, so re-run merge for this task - "
            "it reads the shards already on disk and re-runs no query."
        )
    missing = df[column].isna()
    if missing.any():
        logger.warning(
            "%d of %d rows carry no %s and are dropped from the panels; a query set whose "
            "shapes declare no additional_info scores without it.",
            int(missing.sum()),
            len(df),
            column,
        )
    out = df[~missing].copy()
    out["dataset"] = out[column].map(FACET_LABELS.get(column, str))
    return out


# ---------------------------------------------------------------------------
# The ablation's arms
# ---------------------------------------------------------------------------

#: (step, approach) -> arm label, in ablation order. The pair rather than either half:
#: ``optim_global`` names two arms and step 1 names two arms, only the pair is unique.
#: Kept in step with ``coordinator/producers/ablation.py``, which is what decides that
#: step 1 is the vanilla-only state.
ARMS: Dict[Tuple[int, str], str] = {
    (0, "optim_global"): "Stretto",
    (1, "optim_global"): "Stretto, vanilla only",
    (1, "no_optim"): "No optimization",
}
ARM_ORDER: List[str] = list(ARMS.values())

#: Short forms for axes that put all three arms under one group. "Stretto, vanilla only"
#: is 20 characters and simply does not fit three-to-a-group; the full names stay on the
#: per-metric bar charts, which give each arm its own tick.
ARM_SHORT_LABELS: Dict[str, str] = {
    "Stretto": "Stretto",
    "Stretto, vanilla only": "Vanilla only",
    "No optimization": "No optim.",
}

#: The approaches that ignore their guarantees, hence the arms enumerated at a single
#: guarantee pair and fanned back out here. See ``ablation.COLLAPSE_GUARANTEE_AXIS`` and
#: ``reorder_only.COLLAPSE_GUARANTEE_AXIS``.
#:
#: ``reorder_only`` collapses *both* of its arms; a task whose every arm is blind has
#: nothing to fan out to, and keeps its single guarantee setting.
GUARANTEE_BLIND_APPROACHES = frozenset({"no_optim", "no_optim_reorder"})


#: ``reorder`` -> arm label for ``coordinator.producers.reordering``: the full system
#: against itself with operator reordering disabled.
REORDER_ARMS: Dict[bool, str] = {
    True: "Stretto",
    False: "Stretto, no reordering",
}
REORDER_ARM_ORDER: List[str] = list(REORDER_ARMS.values())

REORDER_ARM_SHORT_LABELS: Dict[str, str] = {
    "Stretto": "Stretto",
    "Stretto, no reordering": "No reorder",
}

#: ``(approach, reorder)`` -> arm label for ``coordinator.producers.reorder_only``. The
#: two arms differ in both columns at once. ``tune_parameters`` is not in the key because
#: it is inert on both arms.
REORDER_ONLY_ARMS: Dict[Tuple[str, bool], str] = {
    ("no_optim", False): "No optimization",
    ("no_optim_reorder", True): "Reordering only",
}
REORDER_ONLY_ARM_ORDER: List[str] = list(REORDER_ONLY_ARMS.values())

REORDER_ONLY_ARM_SHORT_LABELS: Dict[str, str] = {
    "No optimization": "No optim.",
    "Reordering only": "Reorder only",
}


@dataclass(frozen=True)
class ReferenceArm:
    """One arm borrowed from another task and appended to the compared axis.

    E.g. the ablation's unoptimized arm shown beside a sample-size sweep, as an extra
    category at the right-hand end of the axis. Both tasks must replay the same query
    sets over the same benchmarks.

    The arm keeps its *approach*, so it takes that approach's hue in ``sweep_figures``
    (``ARM_HUE_ALIASES`` maps this label onto ``no_optim``'s hue).
    """

    #: Which ``approach`` of the reference CSVs to take.
    approach: str = "no_optim"
    #: The arm value the figures carry, and the name ``arm_colors`` looks the hue up by.
    label: str = "No optimization"

    @property
    def tick(self) -> str:
        """The label as an x tick: one word per line.

        Breaking at spaces keeps a multi-word name within one bar's width without
        shrinking the tick type.
        """
        return "\n".join(str(self.label).split())


def scores_by_construction(rows: pd.DataFrame, tolerance: float = 1e-9) -> bool:
    """Whether *rows* report a perfect score on every query, hence measure nothing.

    Every random benchmark is scored against **silver** - a labelling pass by the best
    operators - and a labelling pass is exactly what the unoptimized arm executes. So its
    ``achieved_*`` columns are 1.0 by construction, and it is kept off accuracy figures.

    Checked on the data rather than declared on the arm, since other borrowed arms are
    scored normally.
    """
    columns = [
        c for c in ("achieved_precision", "achieved_recall", "achieved_f1")
        if c in rows.columns
    ]
    values = rows[columns].to_numpy(dtype=float) if columns else np.empty((0, 0))
    values = values[~np.isnan(values)]
    return values.size > 0 and bool(np.all(np.abs(values - 1.0) <= tolerance))


def reference_arm_rows(
    target: pd.DataFrame, reference: pd.DataFrame, arm: ReferenceArm
) -> pd.DataFrame:
    """The *reference* rows that may be read beside *target*, as one extra arm.

    Two checks stand between the two tasks, and both drop a benchmark rather than
    reporting a comparison that is not one:

    * **the query sets must be identical.** The bars are per-benchmark totals over a query
      set, so two tasks that drew different queries produce two numbers with no ratio
      between them. Identity is asked of the query strings rather than of their count.
    * **the guarantee targets must line up.** A guarantee-blind arm carries whichever
      single pair it was enumerated at (``ablation.COLLAPSE_GUARANTEE_AXIS``) and is
      replicated across the targets the target frame holds, which is sound because it
      never read them. An arm run per target contributes only where both tasks have one.
    """
    rows = reference[reference["approach"].astype(str) == arm.approach]
    if rows.empty:
        raise SystemExit(
            f"No {arm.approach!r} rows in the reference CSVs. Available approaches: "
            f"{sorted(reference['approach'].astype(str).unique())}."
        )

    kept: List[pd.DataFrame] = []
    for (benchmark, split), group in rows.groupby(["benchmark", "split"], sort=False):
        wanted = target[
            (target["benchmark"] == benchmark) & (target["split"] == split)
        ]
        if wanted.empty:
            logger.info(
                "%s is not in this sweep; leaving its %s rows out.", benchmark, arm.approach
            )
            continue
        theirs, ours = set(group["query"]), set(wanted["query"])
        if theirs != ours:
            logger.warning(
                "%s: the reference ran %d quer(ies) and this sweep ran %d, %d of them "
                "shared - the per-benchmark totals are not comparable, so the %s arm is "
                "left off this benchmark.",
                benchmark, len(theirs), len(ours), len(theirs & ours), arm.label,
            )
            continue

        settings = list(pd.unique(wanted["guarantee_setting"]))
        present = list(pd.unique(group["guarantee_setting"]))
        if len(present) == 1:
            # Guarantee-blind: one measurement, read at every target.
            for setting in settings:
                copy = group.copy()
                copy["guarantee_setting"] = setting
                kept.append(copy)
            continue
        for setting in settings:
            if setting not in present:
                logger.warning(
                    "%s: the reference has no %s rows, so the %s arm is missing from that "
                    "target's bars.", benchmark, setting, arm.label,
                )
                continue
            kept.append(group[group["guarantee_setting"] == setting])

    if not kept:
        raise SystemExit(
            f"No benchmark of this sweep can be read against the {arm.approach!r} "
            "reference - see the warnings above for which check each one failed."
        )
    out = pd.concat(kept, axis=0, ignore_index=True)
    out["arm"] = arm.label
    logger.info(
        "Attached %d %s row(s) as the %r arm, over %d benchmark(s).",
        len(out), arm.approach, arm.label, out["benchmark"].nunique(),
    )
    return out


def fan_out_guarantee_blind_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the unoptimized arm across every guarantee setting in *df*.

    ``LabelOptimizer`` never reads the guarantees, so the producer enumerates that arm
    once per benchmark instead of once per guarantee pair (the copies would be
    identical). Its rows carry whichever pair it was enumerated at, so this copies them
    to every setting the optimizing arms ran at.

    A no-op when the producer's ``COLLAPSE_GUARANTEE_AXIS`` is off, and when *every* arm
    is guarantee-blind (``reorder_only``).
    """
    if "approach" not in df.columns:
        return df
    blind = df[df["approach"].isin(GUARANTEE_BLIND_APPROACHES)]
    if blind.empty:
        return df

    optimized = df[~df["approach"].isin(GUARANTEE_BLIND_APPROACHES)]
    copies = []
    for (benchmark, split), group in blind.groupby(["benchmark", "split"]):
        # The settings this benchmark's *optimizing* arms ran at, per benchmark so
        # differing guarantee grids are respected.
        settings = optimized[
            (optimized["benchmark"] == benchmark) & (optimized["split"] == split)
        ]["guarantee_setting"].unique()
        present = set(group["guarantee_setting"].unique())
        for setting in settings:
            if setting in present:
                continue
            copy = group.copy()
            copy["guarantee_setting"] = setting
            copies.append(copy)

    if not copies:
        return df
    logger.info(
        "Fanned the %s arm(s) out to %d further guarantee setting(s).",
        sorted(set(blind["approach"])),
        sum(len(c) for c in copies),
    )
    return pd.concat([df, *copies], axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def per_dataset_totals(
    df: pd.DataFrame,
    arm_col: str,
    value_cols: Sequence[str] = tuple(PHASE_COLUMNS),
    group_cols: Sequence[str] = ("dataset", "guarantee_setting"),
    scale: float = 1.0,
    agg: str = "sum",
) -> pd.DataFrame:
    """Combine *value_cols* over each group's queries by *agg*, then multiply by *scale*.

    A dataset panel shows total work over that benchmark's whole query set, which is
    what a sum is for. The geometric mean enters one level up, across datasets
    (:func:`geomean_overall`), where a sum would be dominated by whichever benchmark is
    slowest.

    ``agg="mean"`` divides by the query count instead, for axes whose buckets hold
    different numbers of queries (e.g. faceting on query complexity). It is the same
    choice ``METRICS`` makes between ``total_runtime`` and ``mean_total_runtime``.
    """
    keys = [c for c in (*group_cols, arm_col)]
    grouped = df.groupby(keys, dropna=False, sort=False)[list(value_cols)]
    totals = (grouped.mean() if agg == "mean" else grouped.sum()).reset_index()
    for column in value_cols:
        totals[column] = totals[column] * scale
    return totals


def geomean_shares(per_dataset: np.ndarray) -> np.ndarray:
    """Combine one arm's per-dataset phase totals into a single stacked bar.

    ``per_dataset`` is ``(n_datasets, n_phases)`` and non-negative. The bar's *height* is
    the geometric mean of the per-dataset totals, and its *composition* is each phase's
    mean share of its own dataset's total, renormalized::

        total   = exp(mean(log(row_sums)))
        share_p = mean_d(per_dataset[d, p] / row_sums[d]);  share /= share.sum()
        return    total * share

    Unlike a per-phase geometric mean, this guarantees:

    * the segments sum to the bar, so the stack is still a runtime;
    * the bar is exactly ``geomean`` of the totals, so if arm A costs ``k`` times arm B
      on *every* dataset the two bars are in ratio ``k`` - where a sum would have
      reported the ratio on the largest dataset alone.

    Scaling commutes (``geomean_shares(c * X) == c * geomean_shares(X)``), so seconds may
    be converted to hours before or after.

    A dataset whose total is zero is dropped from both the mean and the shares rather
    than zeroing the result, since ``log(0)`` is ``-inf``.
    """
    values = np.asarray(per_dataset, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D (datasets, phases) array, got {values.shape}")
    if values.size == 0:
        return np.zeros(values.shape[1] if values.ndim == 2 else 0)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)

    row_sums = values.sum(axis=1)
    live = row_sums > 0
    if not live.any():
        return np.zeros(values.shape[1])
    if not live.all():
        logger.warning(
            "%d of %d datasets contributed a zero total and were excluded from the "
            "geometric mean.",
            int((~live).sum()),
            len(row_sums),
        )

    total = float(np.exp(np.mean(np.log(row_sums[live]))))
    shares = (values[live] / row_sums[live, None]).mean(axis=0)
    share_sum = shares.sum()
    if share_sum <= 0:
        return np.zeros(values.shape[1])
    return total * (shares / share_sum)


def geomean_overall(
    totals: pd.DataFrame,
    arm_col: str,
    value_cols: Sequence[str] = tuple(PHASE_COLUMNS),
    group_cols: Sequence[str] = ("guarantee_setting",),
    dataset_col: str = "dataset",
    label: str = OVERALL,
) -> pd.DataFrame:
    """One *label* row per (``group_cols`` x arm), combined by :func:`geomean_shares`."""
    keys = [c for c in (*group_cols, arm_col)]
    rows = []
    for key_values, group in totals.groupby(keys, dropna=False, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        combined = geomean_shares(group[list(value_cols)].to_numpy())
        row = dict(zip(keys, key_values))
        row[dataset_col] = label
        row.update(dict(zip(value_cols, combined)))
        rows.append(row)
    return pd.DataFrame(rows, columns=[dataset_col, *keys, *value_cols])


def sum_overall(
    totals: pd.DataFrame,
    arm_col: str,
    value_cols: Sequence[str] = tuple(PHASE_COLUMNS),
    group_cols: Sequence[str] = ("guarantee_setting",),
    dataset_col: str = "dataset",
    label: str = OVERALL,
) -> pd.DataFrame:
    """One *label* row per (``group_cols`` x arm), holding the datasets' summed totals.

    The counterpart of :func:`geomean_overall`: the actual total cost, dominated by the
    largest benchmark. Segments still sum to the bar, so no share renormalization is
    needed.
    """
    keys = [c for c in (*group_cols, arm_col)]
    pooled = (
        totals.groupby(keys, dropna=False, sort=False)[list(value_cols)]
        .sum()
        .reset_index()
    )
    pooled[dataset_col] = label
    return pooled[[dataset_col, *keys, *value_cols]]


def pooled_overall(
    totals: pd.DataFrame,
    arm_col: str,
    value_cols: Sequence[str] = tuple(PHASE_COLUMNS),
    group_cols: Sequence[str] = ("guarantee_setting",),
    dataset_col: str = "dataset",
    label: str = OVERALL,
    how: str = "geomean",
) -> pd.DataFrame:
    """The pooled rows alone, by *how* (:data:`POOLINGS`)."""
    if how not in POOLINGS:
        raise ValueError(f"unknown pooling {how!r}; expected one of {POOLINGS}")
    combine = geomean_overall if how == "geomean" else sum_overall
    return combine(totals, arm_col, value_cols, group_cols, dataset_col, label)


def with_overall(
    totals: pd.DataFrame,
    arm_col: str,
    value_cols: Sequence[str] = tuple(PHASE_COLUMNS),
    group_cols: Sequence[str] = ("guarantee_setting",),
    dataset_col: str = "dataset",
    label: str = OVERALL,
    how: str = "geomean",
) -> pd.DataFrame:
    """*totals* with the cross-dataset row appended, or unchanged if there is one dataset."""
    n_datasets = totals[dataset_col].nunique()
    if n_datasets < 2:
        logger.info(
            "Only %d dataset present; pooling is the identity, so no %r panel is added.",
            n_datasets,
            label,
        )
        return totals
    overall = pooled_overall(
        totals, arm_col, value_cols, group_cols, dataset_col, label, how
    )
    return pd.concat([totals, overall], axis=0, ignore_index=True)


def drop_all_zero_phases(
    totals: pd.DataFrame, value_cols: Sequence[str] = tuple(BREAKDOWN_PHASE_COLUMNS)
) -> List[str]:
    """The subset of *value_cols* that is non-zero somewhere in *totals*.

    E.g. ``time_reasoning`` is structurally zero on every sweep row (the benchmark path
    ``execute_logical_plan`` has no ``measure("reasoning")`` span).
    """
    return [c for c in value_cols if c in totals.columns and totals[c].abs().sum() > 0]


# ---------------------------------------------------------------------------
# Single-column metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """One scalar column, with the two aggregations that make it comparable.

    ``agg`` combines a benchmark's queries; ``pool`` combines the benchmarks. They differ
    per metric:

    * a runtime or a cost is **summed** over queries (total work) and pooled over datasets
      by whichever of :data:`POOLINGS` the caller asked for - geometric mean, which is
      scale-free (artwork's totals run an order of magnitude above movie's, so a sum
      reports artwork), or the sum itself, which is the bill;
    * an accuracy is **averaged** over queries and pooled by **arithmetic mean** under
      either, because it is already a ratio bounded by 1. See :meth:`pool_rule`.
    """

    key: str
    column: str
    label: str
    agg: str = "sum"
    pool: str = "geomean"
    scale: float = 1.0
    #: Whether this measures how well the plan answered rather than what it cost. Arms
    #: that score by construction (:func:`scores_by_construction`) are kept off these.
    accuracy: bool = False

    def pool_rule(self, how: str = "geomean") -> str:
        """Which rule pools this metric's datasets when the caller asked for *how*.

        ``--pooling`` is a choice between two ways of combining an *extensive* quantity,
        so it applies to runtimes and costs only; an accuracy is pooled by arithmetic mean
        under either.
        """
        return how if self.pool == "geomean" else self.pool

    @property
    def varies_with_pooling(self) -> bool:
        """Whether ``--pooling`` changes this metric's pooled panel at all."""
        return self.pool == "geomean"


_HOURS = 1.0 / 3600.0

METRICS: Dict[str, Metric] = {
    "execution_runtime": Metric(
        "execution_runtime", "execution_runtime_s", "Execution time [h]", scale=_HOURS
    ),
    "tuning_runtime": Metric(
        "tuning_runtime", "tuning_runtime_s", "Tuning time [h]", scale=_HOURS
    ),
    "total_runtime": Metric(
        "total_runtime", "total_runtime_s", "Total runtime [h]", scale=_HOURS
    ),
    "end_to_end": Metric(
        "end_to_end", "wall_clock_s", "End-to-end runtime [h]", scale=_HOURS
    ),
    "monetary": Metric("monetary", "monetary_cost", "Monetary cost [$]"),
    "f1": Metric("f1", "achieved_f1", "F1", agg="mean", pool="mean", accuracy=True),
    # `mean_` in the key distinguishes these raw averages from the per-query
    # achieved/target ratios drawn by `plot_target_met`.
    "mean_precision": Metric(
        "mean_precision", "achieved_precision", "Mean precision",
        agg="mean", pool="mean", accuracy=True,
    ),
    "mean_recall": Metric(
        "mean_recall", "achieved_recall", "Mean recall",
        agg="mean", pool="mean", accuracy=True,
    ),
    # Per-query means of the runtime columns, for axes whose buckets hold different
    # numbers of queries (e.g. `--facet-by num_semops`). In seconds, since a single
    # query is far below an hour; datasets are still pooled by geometric mean.
    "mean_total_runtime": Metric(
        "mean_total_runtime", "total_runtime_s", "Total runtime per query [s]", agg="mean"
    ),
    "mean_execution_runtime": Metric(
        "mean_execution_runtime",
        "execution_runtime_s",
        "Execution time per query [s]",
        agg="mean",
    ),
    "mean_tuning_runtime": Metric(
        "mean_tuning_runtime",
        "tuning_runtime_s",
        "Tuning time per query [s]",
        agg="mean",
    ),
    "mean_end_to_end": Metric(
        "mean_end_to_end", "wall_clock_s", "End-to-end runtime per query [s]", agg="mean"
    ),
    "mean_monetary": Metric(
        "mean_monetary", "monetary_cost", "Monetary cost per query [$]", agg="mean"
    ),
}

#: What ``--metrics`` draws when not told otherwise.
DEFAULT_METRICS = (
    "tuning_runtime", "execution_runtime", "total_runtime", "monetary",
    "f1", "mean_precision", "mean_recall",
)


def metric_totals(
    df: pd.DataFrame,
    arm_col: str,
    metric: Metric,
    group_cols: Sequence[str] = ("dataset", "guarantee_setting"),
    x_column: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate one metric per (group x arm), keeping *x_column* if the axis is numeric."""
    if metric.column not in df.columns:
        logger.warning("Column %s absent; skipping %s.", metric.column, metric.key)
        return pd.DataFrame()
    keys = [*group_cols, arm_col] + ([x_column] if x_column else [])
    # The operator counts are a function of (dataset, arm), so grouping by them adds no
    # rows and carries them through to the figure for annotating numeric x-axes.
    keys += [
        column
        for column in ("n_operators", "n_cached_levels")
        if column in df.columns and column not in keys
    ]
    frame = df.dropna(subset=[metric.column])
    if frame.empty:
        logger.warning("Every %s value is empty; skipping %s.", metric.column, metric.key)
        return pd.DataFrame()
    out = frame.groupby(keys, dropna=False, sort=False)[metric.column].agg(metric.agg).reset_index()
    out[metric.column] = out[metric.column] * metric.scale
    return out


def pool_metric(
    totals: pd.DataFrame,
    arm_col: str,
    metric: Metric,
    group_cols: Sequence[str] = ("guarantee_setting",),
    x_column: Optional[str] = None,
    dataset_col: str = "dataset",
    label: str = OVERALL,
    how: str = "geomean",
) -> pd.DataFrame:
    """The pooled rows *alone*, by whichever rule :meth:`Metric.pool_rule` names.

    Separate from :func:`with_overall_metric` because ``--facet-by`` applies the same
    rule *inside* each panel rather than across panels.
    """
    # A numeric x has to join the grouping or the pooled point has no position on it.
    keys = list(group_cols) + ([x_column] if x_column else [])
    rule = metric.pool_rule(how)
    if rule in POOLINGS:
        return pooled_overall(
            totals, arm_col, [metric.column], keys, dataset_col, label, rule
        )
    pooled = (
        totals.groupby([*keys, arm_col], dropna=False, sort=False)[metric.column]
        .mean()
        .reset_index()
    )
    pooled[dataset_col] = label
    return pooled


def with_overall_metric(
    totals: pd.DataFrame,
    arm_col: str,
    metric: Metric,
    group_cols: Sequence[str] = ("guarantee_setting",),
    x_column: Optional[str] = None,
    dataset_col: str = "dataset",
    how: str = "geomean",
) -> pd.DataFrame:
    """Append the pooled row, by whichever rule this metric's ``pool`` names."""
    if totals.empty or totals[dataset_col].nunique() < 2:
        return totals
    pooled = pool_metric(
        totals, arm_col, metric, group_cols, x_column, dataset_col, OVERALL, how
    )
    return pd.concat([totals, pooled], axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# Comparisons: how a sweep's rows become an x-axis
# ---------------------------------------------------------------------------


def _label_map() -> Dict[str, str]:
    from reasondb.evaluation.plotting import LABEL_MAP

    return LABEL_MAP


@dataclass(frozen=True)
class Comparison:
    """One sweep axis, as an arm column plus an order and tick labels for it."""

    name: str
    build: Callable[[pd.DataFrame], pd.Series]
    order: Callable[[pd.DataFrame], List[str]]
    ticks: Callable[[pd.DataFrame], Dict[str, str]]
    #: Order and ticks are a function of the dataset, not of the whole frame
    #: (``operator_count`` only).
    per_dataset: bool = False
    axis_label: str = ""
    #: Whether the tick labels name the axis themselves (e.g. approach names), making
    #: ``axis_label`` redundant. False where ticks are bare values (a sample size).
    ticks_name_the_axis: bool = False
    #: A numeric column to put on the x-axis of the per-metric figures, which turns them
    #: from bars into curves. ``None`` keeps the arm itself as a categorical axis.
    x_column: Optional[str] = None
    x_label: str = ""
    #: Whether that x column means the same thing in every panel, hence whether the
    #: panels may share an axis. False for ``operator_count`` alone: its x is each
    #: benchmark's own cache footprint in GB, which is also why it is the only
    #: ``per_dataset`` comparison.
    x_shared: bool = True


APPROACH_ORDER = [
    "abacus",
    "lotus",
    "optim_local",
    "optim_shift_budget",
    "optim_global",
    "no_optim",
    "no_optim_reorder",
]


def _approach_order(df: pd.DataFrame) -> List[str]:
    present = set(df["arm"].astype(str))
    # Unknown approaches are appended rather than dropped, so they still render.
    ordered = [a for a in APPROACH_ORDER if a in present]
    return ordered + sorted(present - set(ordered))


def _approach_ticks(df: pd.DataFrame) -> Dict[str, str]:
    labels = _label_map()
    return {arm: labels.get(arm, arm) for arm in set(df["arm"].astype(str))}


def _numeric_order(df: pd.DataFrame) -> List[str]:
    def key(value: str) -> Tuple[int, float, str]:
        try:
            return (0, float(value), "")
        except ValueError:
            return (1, 0.0, value)

    return sorted(set(df["arm"].astype(str)), key=key)


def _sample_size_arm(df: pd.DataFrame) -> pd.Series:
    return df["sample_size"].map(lambda v: "?" if pd.isna(v) else str(int(float(v))))


def _query_stat_arm(column: str) -> Callable[[pd.DataFrame], pd.Series]:
    """A ``build`` over one query-shape statistic: the integer, or ``"?"``.

    Float-tolerant because a CSV column with NaNs reads integers back as floats; also
    tolerant of the column being absent.
    """

    def build(df: pd.DataFrame) -> pd.Series:
        values = (
            df[column] if column in df.columns else pd.Series(pd.NA, index=df.index)
        )
        return values.map(lambda v: "?" if pd.isna(v) else str(int(float(v))))

    return build


def _sampling_protocol_arm(df: pd.DataFrame) -> pd.Series:
    def one(size: Any, adaptive: Any) -> str:
        n = "?" if pd.isna(size) else str(int(float(size)))
        return f"{'adaptive' if bool(adaptive) else 'fixed'}_n{n}"

    return pd.Series(
        [one(s, a) for s, a in zip(df["sample_size"], df["adaptive_sampling"])],
        index=df.index,
    )


def _sampling_protocol_order(df: pd.DataFrame) -> List[str]:
    # Fixed first: it is the default configuration.
    def key(arm: str) -> Tuple[int, float]:
        fixed = arm.startswith("fixed")
        try:
            n = float(arm.split("_n")[-1])
        except ValueError:
            n = 0.0
        return (0 if fixed else 1, n)

    return sorted(set(df["arm"].astype(str)), key=key)


def _sampling_protocol_ticks(df: pd.DataFrame) -> Dict[str, str]:
    """Just "Fixed" / "Adaptive".

    The row budget defines each arm rather than being compared, so it belongs in the
    caption rather than on the tick.
    """
    return {
        arm: ("Adaptive" if arm.startswith("adaptive") else "Fixed")
        for arm in set(df["arm"].astype(str))
    }


def _ablation_arm(df: pd.DataFrame) -> pd.Series:
    arms = [
        ARMS.get((int(step), str(approach)))
        for step, approach in zip(df["step"], df["approach"])
    ]
    unknown = sorted(
        {
            (int(s), str(a))
            for s, a, arm in zip(df["step"], df["approach"], arms)
            if arm is None
        }
    )
    if unknown:
        raise SystemExit(
            f"Rows with (step, approach) {unknown} are not one of this experiment's arms "
            f"{sorted(ARMS)}. That CSV was produced by a different producer, or the "
            "ablation's state plan changed and ARMS in sweep_frames did not."
        )
    return pd.Series(arms, index=df.index)


def _reorder_column(df: pd.DataFrame) -> pd.Series:
    """The ``reorder`` column as bools, defaulting to True where a CSV predates the axis.

    A CSV without the column describes runs that all reordered (the default).
    """
    if "reorder" not in df.columns:
        return pd.Series(True, index=df.index)
    column = df["reorder"].fillna(True)
    # `astype(bool)` alone would read the string "False" as True.
    return column.map(
        lambda v: str(v).strip().lower() not in ("false", "0", "no")
        if isinstance(v, str)
        else bool(v)
    )


def _reorder_arm(df: pd.DataFrame) -> pd.Series:
    return _reorder_column(df).map(REORDER_ARMS)


def _reorder_only_arm(df: pd.DataFrame) -> pd.Series:
    reorder = _reorder_column(df)
    arms = [
        REORDER_ONLY_ARMS.get((str(approach), bool(flag)))
        for approach, flag in zip(df["approach"], reorder)
    ]
    unknown = sorted(
        {
            (str(a), bool(f))
            for a, f, arm in zip(df["approach"], reorder, arms)
            if arm is None
        }
    )
    if unknown:
        raise SystemExit(
            f"Rows with (approach, reorder) {unknown} are not one of this experiment's "
            f"arms {sorted(REORDER_ONLY_ARMS)}. That CSV was produced by a different "
            "producer, or reorder_only's ARMS changed and REORDER_ONLY_ARMS in "
            "sweep_frames did not."
        )
    return pd.Series(arms, index=df.index)


def operator_count_ticks(
    df: pd.DataFrame, dataset: Optional[str] = None
) -> Dict[str, str]:
    """``step`` -> ``"6 ops\\n32.9 GB"``, read off that dataset's own storage.

    The operator count is what the optimizer searches over, the footprint is what it
    costs to keep on disk; the mapping between them is per benchmark.
    """
    frame = df if dataset is None else df[df["dataset"] == dataset]
    ticks: Dict[str, str] = {}
    for step, group in frame.groupby(frame["arm"].astype(str), sort=False):
        n_ops = group["n_operators"].dropna().unique()
        storage = group["storage_gb"].dropna().unique() if "storage_gb" in group else []
        if len(storage) > 1:
            logger.warning(
                "Step %s of %s reports %d different storage footprints (%s); labelling "
                "with the first.",
                step,
                dataset or "the pooled frame",
                len(storage),
                storage,
            )
        left = f"{int(n_ops[0])} ops" if len(n_ops) else f"s{step}"
        ticks[str(step)] = f"{left}\n{storage[0]:.1f} GB" if len(storage) else left
    return ticks


def _kv_operator_arm(df: pd.DataFrame) -> pd.Series:
    """The one KV operator a ``kv_operator`` state holds, or the vanilla-only reference.

    Built from the retained levels rather than from ``step``, so an arm says what it is
    instead of where it sat in the plan - which is what lets two benchmarks with different
    materialized grids be read side by side, and what keeps the reference arm recognisable
    when a dataset has no image slots to shift the numbering.
    """
    if "kv_operator_label" not in df.columns:
        df = add_kv_operator_columns(df)
    return df["kv_operator_label"].astype(str)


def _kv_operator_order(df: pd.DataFrame) -> List[str]:
    """Vanilla-only first, then by modality, model size and compression ratio.

    The reference arm leads because every other arm is read against it. The rest sort as
    the plan enumerates them, which is also how a reader scans them: one modality's
    operators together, cheapest cache last.
    """
    arms = set(df["arm"].astype(str))
    ordered = [VANILLA_ONLY_ARM] if VANILLA_ONLY_ARM in arms else []
    slot_rank = {label: i for i, label in enumerate(SLOT_ARM_LABELS.values())}

    def key(arm: str) -> Tuple[int, float, str]:
        # A kv_operator_pairs arm sorts by its first operator, which is its text level:
        # the pairs then read small before large and cheapest cache last, like the singles.
        label, _, ratio = str(arm).split(" + ")[0].partition(" cr")
        try:
            return (slot_rank.get(label, len(slot_rank)), float(ratio), arm)
        except ValueError:
            return (len(slot_rank), math.inf, arm)

    return ordered + sorted((a for a in arms if a != VANILLA_ONLY_ARM), key=key)


def kv_operator_ticks(
    df: pd.DataFrame, dataset: Optional[str] = None
) -> Dict[str, str]:
    """``arm -> "text-L cr0.8\\n12.4 GB"``, read off that dataset's own storage.

    The same two-line shape ``operator_count_ticks`` uses and for the same reason: the arm
    says which operator, the footprint says what keeping it costs, and the mapping between
    them is per benchmark. Here the footprint is that operator's own rather than a whole
    state's, which is the point of the experiment.
    """
    frame = df if dataset is None else df[df["dataset"] == dataset]
    ticks: Dict[str, str] = {}
    for arm, group in frame.groupby(frame["arm"].astype(str), sort=False):
        storage = group["storage_gb"].dropna().unique() if "storage_gb" in group else []
        ticks[str(arm)] = f"{arm}\n{storage[0]:.1f} GB" if len(storage) else str(arm)
    return ticks


COMPARISONS: Dict[str, Comparison] = {
    "approach": Comparison(
        name="approach",
        build=lambda df: df["approach"].astype(str),
        order=_approach_order,
        ticks=_approach_ticks,
        # Named because under --facet-by the panel titles are complexity buckets and
        # nothing else says what the x-axis is.
        axis_label="Approach",
        ticks_name_the_axis=True,
    ),
    "sample_size": Comparison(
        name="sample_size",
        build=_sample_size_arm,
        order=_numeric_order,
        ticks=lambda df: {a: a for a in set(df["arm"].astype(str))},
        axis_label="Profiling sample size [rows]",
        x_column="sample_size",
        x_label="Profiling sample size [rows]",
    ),
    "operator_count": Comparison(
        name="operator_count",
        build=lambda df: df["step"].map(lambda v: str(int(v))),
        order=_numeric_order,
        ticks=operator_count_ticks,
        per_dataset=True,
        axis_label="Search space / materialized cache",
        # Storage rather than the step index: in the default no-index serving mode each
        # greedy step drops one materialized baseline, so footprint tracks search space.
        x_column="storage_gb",
        x_label="Storage [GB]",
        # Each panel's GB axis is that benchmark's own footprint, so sharing it would put
        # movie's 33 GB and ecommerce's 4 TB on one scale.
        x_shared=False,
    ),
    "kv_operator": Comparison(
        name="kv_operator",
        build=_kv_operator_arm,
        order=_kv_operator_order,
        ticks=kv_operator_ticks,
        # Which arms exist is per benchmark: a text-only dataset has no image operators to
        # hold, and every arm's footprint is that dataset's own.
        per_dataset=True,
        axis_label="KV operator / its cache",
        ticks_name_the_axis=True,
        # Footprint rather than the arm index, as ops01 does: what the experiment asks is
        # what an operator costs against what it saves, and this is the axis with a unit.
        x_column="storage_gb",
        x_label="Cache footprint [GB]",
        x_shared=False,
    ),
    "num_semops": Comparison(
        name="num_semops",
        build=_query_stat_arm("num_semops"),
        order=_numeric_order,
        ticks=lambda df: {a: a for a in set(df["arm"].astype(str))},
        # The dashboard's label for this dimension, for consistent naming.
        axis_label=DIMENSION_LABELS["num_semops"],
        x_column="num_semops",
        x_label=DIMENSION_LABELS["num_semops"],
    ),
    "sampling_protocol": Comparison(
        name="sampling_protocol",
        build=_sampling_protocol_arm,
        order=_sampling_protocol_order,
        ticks=_sampling_protocol_ticks,
        axis_label="",
    ),
    "reorder": Comparison(
        name="reorder",
        build=_reorder_arm,
        order=lambda df: [a for a in REORDER_ARM_ORDER if a in set(df["arm"])],
        ticks=lambda df: {
            a: REORDER_ARM_SHORT_LABELS.get(a, a) for a in set(df["arm"].astype(str))
        },
        axis_label="",
    ),
    "reorder_only_arm": Comparison(
        name="reorder_only_arm",
        build=_reorder_only_arm,
        order=lambda df: [a for a in REORDER_ONLY_ARM_ORDER if a in set(df["arm"])],
        ticks=lambda df: {
            a: REORDER_ONLY_ARM_SHORT_LABELS.get(a, a)
            for a in set(df["arm"].astype(str))
        },
        axis_label="",
    ),
    "ablation_arm": Comparison(
        name="ablation_arm",
        build=_ablation_arm,
        order=lambda df: [a for a in ARM_ORDER if a in set(df["arm"])],
        ticks=lambda df: {a: ARM_SHORT_LABELS.get(a, a) for a in set(df["arm"].astype(str))},
        axis_label="",
    ),
}


@dataclass(frozen=True)
class SweepPreset:
    """One shipped experiment: which CSV, which axis, what to call the figures."""

    name: str
    csv_name: str
    comparison: str
    prefix: str
    include_overall: bool = True
    fan_out_guarantee_blind: bool = False
    #: "facets" - one panel per dataset; "grouped" - one axes, x = guarantee setting.
    layout: str = "facets"
    figures: Tuple[str, ...] = ("target-met", "breakdown")
    #: Sum the phase totals over the guarantee targets before drawing, collapsing each
    #: arm to one bar. For an experiment whose own axis is already long, three bars per
    #: point is denser than the comparison needs.
    collapse_targets: bool = False
    #: Fraction of ``--width`` this experiment's figure gets. The ablation is a single
    #: axes and belongs in one column of a two-column page, not across both.
    width_scale: float = 1.0
    #: Multiplier on ``--overall-width`` for the pooled figure. The quarter-page default
    #: suits three or more arms; with two it leaves a panel narrower than its own legend.
    overall_width_scale: float = 1.0
    #: The ``task_id``\ s in ``scripts/cluster.yaml`` this preset draws, accepted as
    #: aliases (a task id names the directory the CSVs are read from).
    task_ids: Tuple[str, ...] = ()
    note: str = ""


PRESETS: Dict[str, SweepPreset] = {
    "baselines": SweepPreset(
        name="baselines",
        task_ids=("base01",),
        csv_name="baselines.csv",
        comparison="approach",
        prefix="baselines",
        note="base01: the optimizers against each other at the default operator suite.",
    ),
    "modes": SweepPreset(
        name="modes",
        task_ids=("mode01",),
        csv_name="baselines.csv",
        comparison="approach",
        prefix="modes",
        note=(
            "mode01: one optimizer's GlobalOptimizationModes against each other, over "
            "every materialized operator."
        ),
    ),
    "sample_size": SweepPreset(
        name="sample_size",
        task_ids=("samp01",),
        csv_name="sample_size.csv",
        comparison="sample_size",
        prefix="sample_size",
        # The target is not what this experiment varies, so targets are summed away.
        collapse_targets=True,
        note="samp01: what a larger profiling sample buys.",
    ),
    "operator_count": SweepPreset(
        name="operator_count",
        task_ids=("ops01",),
        csv_name="operator_count.csv",
        comparison="operator_count",
        prefix="operator_count",
        # Storage is per dataset, so there is no meaningful pooled GB figure.
        include_overall=False,
        note="ops01: runtime against search-space size and cache footprint.",
    ),
    "kv_operator": SweepPreset(
        name="kv_operator",
        # The other kv_operator state plans use the same preset; their arms label
        # themselves.
        task_ids=("kvop01",),
        csv_name="kv_operator.csv",
        comparison="kv_operator",
        prefix="kv_operator",
        # Which arms a dataset has depends on what it materialized, so the arms are
        # disjoint across panels and a pooled one would average over different sets of
        # them. The same reason `operator_count` has no pooled panel, arriving by a
        # different route: there the x axis is per dataset, here the arms are.
        include_overall=False,
        note=(
            "kvop01: what adding one KV operator (one per modality) to the uncompressed "
            "reference suite costs on disk and buys in runtime, against that suite in "
            "the same task."
        ),
    ),
    "adaptive_sampling": SweepPreset(
        name="adaptive_sampling",
        task_ids=("adapt01",),
        csv_name="adaptive_sampling.csv",
        comparison="sampling_protocol",
        prefix="adaptive_sampling",
        overall_width_scale=1.5,
        note="adapt01: the shipped single-shot sample against the adaptive schedule.",
    ),
    "ablation": SweepPreset(
        name="ablation",
        task_ids=("abl01",),
        csv_name="ablation.csv",
        comparison="ablation_arm",
        prefix="ablation",
        fan_out_guarantee_blind=True,
        layout="grouped",
        width_scale=0.5,
        # No target-met figure: every sweep benchmark is scored against silver, and a
        # silver pass *is* the no_optim arm's plan, so its accuracy is 1.0 by
        # construction. This experiment is a cost comparison.
        figures=("breakdown",),
        note="abl01: Stretto, minus KV compression, minus the optimizer.",
    ),
    "reordering": SweepPreset(
        name="reordering",
        task_ids=("abl02",),
        csv_name="reordering.csv",
        comparison="reorder",
        prefix="reordering",
        layout="grouped",
        width_scale=0.5,
        # The full figure set: reordering changes cost, not answers, so the target-met
        # figure checks that both arms hold the same guarantees.
        note="abl02: Stretto against itself with operator reordering taken off.",
    ),
    "reorder_only": SweepPreset(
        name="reorder_only",
        task_ids=("abl03",),
        csv_name="reorder_only.csv",
        comparison="reorder_only_arm",
        prefix="reorder_only",
        fan_out_guarantee_blind=True,
        layout="grouped",
        width_scale=0.5,
        # No target-met figure, as for `ablation`: the `no_optim` arm's accuracy is 1.0
        # by construction against silver labels. This is a cost comparison.
        figures=("breakdown",),
        note="abl03: no optimization against reordering alone, at the gold state.",
    ),
}


#: Where the coordinator puts a task by default - ``scripts/run_coordinator.py`` sets
#: ``--output-dir`` to ``benchmark_results/<task-id>`` and ``merge`` writes ``merged/``
#: under it. Mirrored rather than imported, since the package does not import scripts.
DEFAULT_RESULTS_ROOT = Path("benchmark_results")


def default_output_dir(
    preset: "SweepPreset", requested: str = "", root: Optional[Path] = None
) -> Optional[Path]:
    """Where *preset*'s merged CSVs live if the task ran with the coordinator's defaults.

    ``benchmark_results/<task-id>/merged``.

    *requested* is what the caller typed: if that was already a task id it wins, so a
    preset with several tasks resolves to the one actually named. Returns ``None`` when
    the preset has no task id to derive from, leaving the caller to insist on an explicit
    directory.
    """
    if not preset.task_ids:
        return None
    task_id = requested if requested in preset.task_ids else preset.task_ids[0]
    return (root or DEFAULT_RESULTS_ROOT) / task_id / "merged"


#: Producers whose results this script does not draw, so the error can say where to go.
OTHER_PRODUCERS = {
    "ref01": "scripts/plot_label_reference.py",
    "label_reference": "scripts/plot_label_reference.py",
}


def resolve_preset(name: str) -> SweepPreset:
    """Look a preset up by its own name or by a ``cluster.yaml`` task id.

    The preset name says what the figure *is* (``baselines``); the task id says which run
    it came from (``base01``). A table rather than a producer lookup, because several
    presets share a producer (e.g. ``modes`` runs the ``baselines`` producer).
    """
    key = name.strip()
    if key in PRESETS:
        return PRESETS[key]
    for preset in PRESETS.values():
        if key in preset.task_ids:
            logger.info("Task %r is drawn by the %r preset.", key, preset.name)
            return preset
    if key in OTHER_PRODUCERS:
        raise SystemExit(
            f"{key!r} is not a sweep experiment - its results are drawn by "
            f"{OTHER_PRODUCERS[key]}."
        )
    known = sorted(
        f"{p.name} ({', '.join(p.task_ids)})" if p.task_ids else p.name
        for p in PRESETS.values()
    )
    raise SystemExit(f"Unknown experiment {key!r}. Available: " + "; ".join(known))


def prepare_sweep_frame(
    df: pd.DataFrame, comparison: Comparison, fan_out_guarantee_blind: bool = False
) -> pd.DataFrame:
    """Add the ``arm`` column, fanning the guarantee-blind arm out first if asked.

    The fan-out must happen before any aggregation, or the blind arm would be missing
    from every guarantee group but one.
    """
    out = df.copy()
    if fan_out_guarantee_blind:
        out = fan_out_guarantee_blind_rows(out)
    out["arm"] = comparison.build(out)
    return out
