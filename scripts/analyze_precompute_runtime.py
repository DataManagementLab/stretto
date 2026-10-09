"""Summarise per-call model runtimes recorded in ``--precompute`` JSON files.

``scripts/run_coordinator.py --producer run_benchmark --precompute <json>`` records every text-QA and
vision call together with the KV-cache configuration it ran under (effective /
materialized compression ratio, vanilla flag) and the measured runtime.

Each input JSON is analysed independently and yields one pair of outputs, named
after the JSON's stem:

  * ``<stem>_runtime.csv`` — one row per (modality, model, effective CR,
    materialized CR, vanilla) with count/mean/median/std/min/max/p95 runtime.
  * ``<stem>_runtime.pdf`` — one facet per model, x-axis the effective
    compression ratio, y-axis runtime (mean solid, median dashed). One colored
    line per *materialized* ratio: that is one cache on disk, and every
    effective ratio on its line is served from it. Vanilla runs and models
    without a KV cache get their own line. Facets are grouped into one row
    block per modality (``text_qa``, ``vision``, ...), wrapped to a fixed
    number of columns within each block, and stacked vertically — so adding a
    modality adds rows below rather than more columns.

Examples
--------
    python scripts/analyze_precompute_runtime.py benchmark_results/precompute.json

    python scripts/analyze_precompute_runtime.py \
        benchmark_results/artwork.json benchmark_results/movie.json \
        --output-dir benchmark_results/precompute_analysis
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from reasondb.evaluation.plotting import apply_default_style

logger = logging.getLogger(__name__)

# Suffix appended to the base model name by the KV backends' ``model_id``, e.g.
# "meta-llama/Llama-3.1-70B-cr0.9-mat0.3", "...-cr0.0-vanilla", "...-cr0.8-in-memory".
MODEL_ID_SUFFIX = re.compile(
    r"-cr[\d.]+(?:-mat[\d.]+)?(?:-vanilla)?(?:-in-memory)?$"
)

GROUP_KEYS = ["modality", "model", "effective_cr", "materialized_cr", "vanilla"]


def base_model_name(model_id: str) -> str:
    """Strip the compression-ratio suffix the backends encode into ``model_id``."""
    return MODEL_ID_SUFFIX.sub("", model_id)


def load_records(path: Path) -> pd.DataFrame:
    """Flatten a precompute JSON into one row per recorded model call."""
    with open(path, "r") as f:
        data = json.load(f)

    rows: List[Dict] = []
    for modality in ("text_qa", "vision"):
        for model_id, bucket in data.get(modality, {}).items():
            for record in bucket.values():
                runtime = record.get("runtime")
                if runtime is None:
                    continue
                rows.append(
                    {
                        "modality": modality,
                        "model_id": model_id,
                        "model": base_model_name(model_id),
                        "effective_cr": record.get("effective_compression_ratio"),
                        "materialized_cr": record.get("materialized_compression_ratio"),
                        "vanilla": bool(record.get("vanilla", False)),
                        "runtime": float(runtime),
                    }
                )

    if not rows:
        raise ValueError(f"No runtime records found in {path}.")
    df = pd.DataFrame(rows)
    # Models that use no pre-computed KV cache (e.g. GPT) record None ratios.
    df[["effective_cr", "materialized_cr"]] = df[["effective_cr", "materialized_cr"]].fillna(-1.0)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Runtime statistics per model and KV-cache configuration."""
    agg = (
        df.groupby(GROUP_KEYS)["runtime"]
        .agg(
            calls="count",
            mean_runtime_s="mean",
            median_runtime_s="median",
            std_runtime_s="std",
            min_runtime_s="min",
            max_runtime_s="max",
            p95_runtime_s=lambda s: s.quantile(0.95),
        )
        .reset_index()
        .sort_values(GROUP_KEYS)
    )
    agg["total_runtime_s"] = agg["calls"] * agg["mean_runtime_s"]
    return agg


def _series_label(materialized_cr: float, vanilla: bool) -> str:
    if vanilla:
        return "vanilla"
    if materialized_cr < 0:
        return "no KV cache"
    return f"mat={materialized_cr:g}"


def series_colors(agg: pd.DataFrame) -> Dict[str, tuple]:
    """One color per materialized compression ratio, shared by every figure.

    A materialized ratio is one cache on disk; the effective ratios drawn along
    its line are all served from it. The ratio is ordered, so the colors come
    from a sequential map (dark = little compression, bright = heavily
    compressed) and are keyed on the value itself — ``mat=0.3`` is the same
    color in every model's figure.
    """
    kv = agg[(~agg["vanilla"]) & (agg["materialized_cr"] >= 0)]
    ratios = sorted(kv["materialized_cr"].unique())
    span = plt.cm.viridis(
        [0.05 + 0.85 * i / max(len(ratios) - 1, 1) for i in range(len(ratios))]
    )
    colors = {f"mat={r:g}": c for r, c in zip(ratios, span)}
    colors["vanilla"] = (0.4, 0.4, 0.4, 1.0)
    colors["no KV cache"] = (0.0, 0.0, 0.0, 1.0)
    return colors


def _facet_title(model: str) -> str:
    """Drop the HF org prefix; the full id stays in the CSV."""
    return model.split("/")[-1]


DIAGONAL_COLOR = (0.6, 0.6, 0.6, 1.0)


def _draw_facet(data: pd.DataFrame, colors: Dict[str, tuple], **kwargs) -> None:
    """Draw one model's lines into the current facet axis."""
    ax = plt.gca()

    # Guide through the configurations that materialize exactly what they use.
    # Each of those points belongs to a different cache, so it is the only line
    # a diagonal-only precompute can show — drawn behind the real series.
    diagonal = data[
        (~data["vanilla"])
        & (data["materialized_cr"] >= 0)
        & (data["materialized_cr"] == data["effective_cr"])
    ].sort_values("effective_cr")
    if len(diagonal) > 1:
        ax.plot(
            diagonal["effective_cr"], diagonal["mean_runtime_s"],
            color=DIAGONAL_COLOR, linewidth=1, zorder=1,
        )
        ax.plot(
            diagonal["effective_cr"], diagonal["median_runtime_s"],
            color=DIAGONAL_COLOR, linewidth=1, linestyle="--", zorder=1,
        )

    for series, group in data.groupby("series"):
        group = group.sort_values("effective_cr")
        color = colors[series]
        ax.plot(
            group["effective_cr"], group["mean_runtime_s"],
            marker="o", color=color, label=series,
        )
        ax.plot(
            group["effective_cr"], group["median_runtime_s"],
            marker="s", linestyle="--", color=color,
        )


def plot_models(agg: pd.DataFrame, colors: Dict[str, tuple], output_path: Path) -> None:
    """One facet per model, grouped into a row block per modality.

    Facets keep their own y-scale — a 70B model and an embedding model differ by
    orders of magnitude, and a shared axis would flatten the smaller ones.
    Each modality (``text_qa``, ``vision``, ...) gets its own wrapped block of
    facet rows, stacked below the previous modality's block, so a modality
    with more models simply grows taller instead of widening every row.
    """
    plot_df = agg.copy()
    plot_df["series"] = plot_df.apply(
        lambda r: _series_label(r["materialized_cr"], r["vanilla"]), axis=1
    )
    # Models without a KV cache carry the -1 sentinel; draw them at "no compression".
    plot_df["effective_cr"] = plot_df["effective_cr"].clip(lower=0.0)
    plot_df["model_label"] = plot_df["model"].map(_facet_title)

    modalities = sorted(plot_df["modality"].unique())
    modality_models = {
        m: sorted(plot_df.loc[plot_df["modality"] == m, "model_label"].unique())
        for m in modalities
    }

    n_cols = min(3, max(len(models) for models in modality_models.values()))
    # One entry per row of the figure: (modality, is_first_row_of_block, models_in_row).
    row_blocks = []
    for m in modalities:
        models = modality_models[m]
        n_block_rows = -(-len(models) // n_cols)  # ceil division
        for r in range(n_block_rows):
            row_blocks.append((m, r == 0, models[r * n_cols : (r + 1) * n_cols]))

    fig, axes = plt.subplots(
        len(row_blocks), n_cols,
        figsize=(n_cols * 4.2, len(row_blocks) * 3.5),
        squeeze=False,
    )

    for row_idx, (modality, is_first_row, row_models) in enumerate(row_blocks):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(row_models):
                ax.axis("off")
                continue
            model_label = row_models[col_idx]
            data = plot_df[
                (plot_df["modality"] == modality) & (plot_df["model_label"] == model_label)
            ]
            plt.sca(ax)
            _draw_facet(data, colors)
            ax.set_title(model_label)
            ax.set_xlabel("Effective compression ratio")
            ax.set_ylabel("Runtime per call [s]")
            ax.set_ylim((0, None))
        if is_first_row:
            axes[row_idx][0].annotate(
                modality, xy=(0, 1.3), xycoords="axes fraction",
                fontsize=13, fontweight="bold", ha="left",
            )

    # Color carries the materialized ratio, line style carries the statistic;
    # listing them separately keeps the legend at (ratios + 2) instead of
    # (ratios x 2) entries. Both groups live in one legend so they cannot
    # overlap each other whatever the facet layout.
    present = [s for s in colors if s in set(plot_df["series"])]
    handles = [Line2D([], [], color=colors[s], marker="o", label=s) for s in present]
    handles.append(Line2D([], [], color=DIAGONAL_COLOR, linewidth=1, label="mat = eff"))
    handles += [
        Line2D([], [], linestyle="none", label=""),
        Line2D([], [], color="black", marker="o", label="Mean"),
        Line2D([], [], color="black", marker="s", linestyle="--", label="Median"),
    ]
    fig.legend(
        handles=handles, title="Materialized",
        loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def main():
    logging.basicConfig(level=logging.INFO)
    apply_default_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "precompute_json", type=Path, nargs="+",
        help="Precompute JSON file(s) to analyse; each yields its own CSV and PDF.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write the CSVs and figures "
        "(default: next to each input JSON).",
    )
    args = parser.parse_args()

    aggs: Dict[Path, pd.DataFrame] = {}
    for path in args.precompute_json:
        try:
            df = load_records(path)
        except ValueError as e:
            logger.error("%s — skipping.", e)
            continue
        logger.info(
            "%s: %d calls across %d models", path.name, len(df), df["model"].nunique()
        )
        aggs[path] = aggregate(df)

    if not aggs:
        raise SystemExit("No runtime records found in any input.")

    # Colors come from the union so a ratio keeps its color across all figures.
    colors = series_colors(pd.concat(aggs.values(), ignore_index=True))

    for path, agg in aggs.items():
        output_dir = args.output_dir or path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"{path.stem}_runtime.csv"
        agg.to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)

        plot_models(agg, colors, output_dir / f"{path.stem}_runtime.pdf")


if __name__ == "__main__":
    main()
