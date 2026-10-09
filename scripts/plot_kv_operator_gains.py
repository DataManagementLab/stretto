"""The kvop01 figure the preset cannot draw: cost against benefit, one point per operator.

``scripts/plot_sweep.py --experiment kvop01`` draws the arms as totals, which says how
much each configuration cost to run. The experiment asks something a bar chart has no
shape for: whether a KV operator is **manageable** - what it costs to keep per cached item
- and **beneficial** - what it saves against the vanilla-only arm in the same task. Those
are two numbers per operator and they belong on two axes.

    manageable  ->  x: the operator's own cache, per cached item
    beneficial  ->  y: the paired speedup against the vanilla-only arm, at each target

A point below the rule at 1.0 is an operator that cost disk and bought nothing. The
interesting region is upper-left; both axes carry units, since where that region starts
depends on the deployment.

Arithmetic in ``reasondb/evaluation/kv_operator_gains.py`` (pandas only, so it is testable
without a display), which is itself a loop over ``evaluation/ablation_gains.py`` - the same
paired, per-query speedup the ablation reports, against a different reference arm. Page
fitting, palette and legend placement come from ``evaluation/sweep_figures.py``, so these
figures match the preset ones.

    python scripts/plot_kv_operator_gains.py --output-dirs benchmark_results/kvop01/merged
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reasondb.evaluation import kv_operator_gains as kvgains
from reasondb.evaluation import sweep_frames as frames
from reasondb.evaluation.plotting import apply_default_style
from reasondb.evaluation.sweep_figures import (
    A4_TEXT_WIDTH_IN,
    page_geometry,
    plain_log_axis,
)

logger = logging.getLogger("plot_kv_operator_gains")

#: Where a speedup of exactly 1.0 sits. On every figure here, because "did this operator
#: pay at all" is the question each of them refines. Above the gridlines, which are dashed
#: in the same grey and would otherwise be mistaken for it.
UNITY = {"color": "#333333", "linestyle": (0, (4, 3)), "linewidth": 0.9, "zorder": 2.5}

#: Marker and line style per proxy model size, so the two data axes stay free for cost and
#: benefit. Line style too, since colour encodes the dataset and connected small- and
#: large-model lines would otherwise look identical.
SIZE_MARKERS = {"small": "o", "large": "^"}
SIZE_STYLES = {"small": "-", "large": (0, (3, 2))}


#: Cache sizes worth a tick, in MB. A ladder rather than whatever a locator volunteers,
#: so two datasets' panels can be read against each other and a log axis narrower than a
#: decade does not fall back to scientific notation between two crowded labels.
_SIZE_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def size_ticks(values) -> List[float]:
    """The subset of :data:`_SIZE_TICKS` that spans *values*, with one step either side.

    Read off the data rather than hardcoded, since per-item cache sizes span two orders
    of magnitude between compressed text and image caches.
    """
    finite = np.asarray([v for v in np.ravel(values) if np.isfinite(v) and v > 0])
    if not finite.size:
        return [1]
    lo, hi = float(finite.min()), float(finite.max())
    inside = [t for t in _SIZE_TICKS if lo / 1.5 <= t <= hi * 1.5]
    if len(inside) >= 2:
        return inside
    # Fewer than two ticks in range (all operators cost about the same per item): add
    # the data's own ends so the axis still shows its span.
    ends = sorted({round(lo, 2), round(hi, 2)})
    return sorted(set(inside) | set(ends))


def _finish(fig, axes, fonts, out_path: Path, handles=None, labels=None) -> Path:
    for ax in np.ravel(axes):
        ax.tick_params(labelsize=fonts["tick"])
        ax.xaxis.label.set_size(fonts["label"])
        ax.yaxis.label.set_size(fonts["label"])
        ax.title.set_size(fonts["title"])
    if handles is not None:
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
            ncol=4, frameon=False, fontsize=fonts["legend"],
            handlelength=1.4, columnspacing=1.0,
        )
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path


def plot_cost_benefit(
    table: pd.DataFrame, out_dir: Path, prefix: str, width_in: float
) -> Path:
    """Per-item cache size against paired speedup, one panel per guarantee target.

    Faceted by target rather than pooled, because whether the optimizer can use an
    operator depends on the target (e.g. P/R 0.5 vs 0.9).

    A log x axis, since the per-item costs span two orders of magnitude between a
    compressed text cache and an uncompressed image one, and a linear axis would put every
    text operator on the spine. Its labels come from :func:`size_ticks`, which avoids
    crowded scientific-notation labels over ranges narrower than a decade.
    """
    targets = sorted(set(table["guarantee_setting"]))
    # Only the font sizes are taken from `page_geometry`, so the hand-built panels match
    # the preset figures at the same width.
    fonts = page_geometry(len(targets), width_in, n_x=5)["fonts"]
    fig, axes = plt.subplots(
        1, len(targets), figsize=(width_in, width_in / max(len(targets), 1) * 0.95),
        sharey=True, sharex=True,
    )
    axes = np.atleast_1d(axes)
    datasets = frames.dataset_col_order(table)
    palette = dict(zip(datasets, sns.color_palette("colorblind", len(datasets))))
    ticks = size_ticks(table["storage_mb_per_entry"])

    for ax, target in zip(axes, targets):
        panel = table[table["guarantee_setting"] == target]
        for (dataset, size), group in panel.groupby(
            ["dataset", "kv_model_size"], dropna=False
        ):
            ax.scatter(
                group["storage_mb_per_entry"],
                group["median"],
                s=26,
                marker=SIZE_MARKERS.get(str(size), "s"),
                color=palette.get(dataset, "#777777"),
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        ax.axhline(1.0, **UNITY)
        ax.set_xscale("log")
        plain_log_axis(ax.xaxis, ticks)
        ax.set_title(f"P/R={frames.guarantee_short_label(target)}")
    axes[0].set_ylabel("Execution speedup vs vanilla only")
    # One label under the row rather than one per panel: the axis is shared, and three
    # copies of it collide with the legend beneath them.
    fig.supxlabel("Cache per item [MB]", fontsize=fonts["label"])

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=palette[dataset], label=dataset)
        for dataset in datasets
    ] + [
        plt.Line2D([], [], marker=marker, linestyle="", color="#555555", label=f"{size} model")
        for size, marker in SIZE_MARKERS.items()
    ]
    return _finish(
        fig, axes, fonts, out_dir / f"{prefix}cost_benefit.pdf",
        handles=handles, labels=[h.get_label() for h in handles],
    )


def plot_speedup_by_ratio(
    table: pd.DataFrame, out_dir: Path, prefix: str, width_in: float
) -> Path:
    """Speedup against the compression ratio itself, one line per dataset and model size.

    The same data read along the axis a reader tunes: given that an operator is affordable,
    how hard may it be compressed before it stops paying. Ratios are the *materialized*
    ones, which in direct serving are also the effective ones.
    """
    fonts = page_geometry(1, width_in, n_x=4)["fonts"]
    fig, ax = plt.subplots(figsize=(width_in, width_in * 0.62))
    pooled = (
        table.groupby(["dataset", "kv_model_size", "kv_cr"], dropna=False)["median"]
        .median()
        .reset_index()
    )
    datasets = frames.dataset_col_order(pooled)
    palette = dict(zip(datasets, sns.color_palette("colorblind", len(datasets))))
    for (dataset, size), group in pooled.groupby(["dataset", "kv_model_size"], dropna=False):
        group = group.sort_values("kv_cr")
        ax.plot(
            group["kv_cr"], group["median"],
            marker=SIZE_MARKERS.get(str(size), "s"),
            linestyle=SIZE_STYLES.get(str(size), ":"),
            color=palette.get(dataset, "#777777"),
            linewidth=1.0, markersize=4,
        )
    ax.axhline(1.0, **UNITY)
    ax.set_xlabel("Materialized compression ratio (KV entries dropped)")
    ax.set_ylabel("Execution speedup")
    handles = [
        plt.Line2D([], [], color=palette[dataset], label=dataset) for dataset in datasets
    ] + [
        plt.Line2D(
            [], [], marker=SIZE_MARKERS[size], linestyle=style, color="#555555",
            label=f"{size} model",
        )
        for size, style in SIZE_STYLES.items()
    ]
    return _finish(
        fig, ax, fonts, out_dir / f"{prefix}speedup_by_ratio.pdf",
        handles=handles, labels=[h.get_label() for h in handles],
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dirs", type=Path, nargs="+", required=True,
        help="Merged directories holding kv_operator.csv, e.g. benchmark_results/kvop01/merged.",
    )
    parser.add_argument("--csv-name", type=str, default="kv_operator.csv")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument(
        "--figure-dir", type=Path, default=None,
        help="Where to write. Defaults to the first --output-dirs entry.",
    )
    parser.add_argument("--prefix", type=str, default="kv_operator_")
    parser.add_argument(
        "--metric", type=str, default="execution",
        choices=["execution", "total", "tuning", "wall"],
        help=(
            "Which clock the speedup divides. 'execution' by default: a total nets the "
            "saving against the profiling an extra operator costs, and profiling draws a "
            "fixed sample while execution scales with the table."
        ),
    )
    parser.add_argument("--width", type=float, default=A4_TEXT_WIDTH_IN)
    parser.add_argument(
        "--table", action="store_true",
        help="Also write the summary as CSV, which is what the numbers in prose come from.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    apply_default_style()
    sns.set_context("paper")

    paths = frames.find_sweep_csvs(args.output_dirs, args.csv_name, args.split)
    if not paths:
        raise SystemExit(f"No {args.csv_name} under {[str(d) for d in args.output_dirs]}.")
    df = kvgains.with_arms(frames.load_sweep(paths))

    table = kvgains.summary(df, metric=args.metric)
    if table.empty:
        raise SystemExit(
            "No (query, guarantee) was run by both a compressed arm and the "
            f"{frames.VANILLA_ONLY_ARM!r} arm, so there is no speedup to report."
        )
    logger.info(
        "Summarized %d (dataset, operator, target) point(s) over %d operator(s).",
        len(table), table["arm"].nunique(),
    )

    out_dir = args.figure_dir or args.output_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.table:
        csv_path = out_dir / f"{args.prefix}summary.csv"
        table.to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)

    plot_cost_benefit(table, out_dir, args.prefix, args.width)
    plot_speedup_by_ratio(table, out_dir, args.prefix, args.width)


if __name__ == "__main__":
    main()
