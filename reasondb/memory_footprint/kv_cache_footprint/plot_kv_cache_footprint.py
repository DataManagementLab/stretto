"""Heatmaps of the KV cache footprint by model, corpus size and compression ratio.

Reads the whole CSV family (kv_cache_footprint_{text,image}_{1k,10k,100k,1M}.csv),
normalising every column to its nominal item count -- Rotowire is measured on 728
items, so its "1k" column is scaled up by 1000/728 like the projected ones.

Produces, in ./figures:
  kv_footprint_<modality>_<dataset>.{png,pdf}  one figure per dataset, a panel per model
  kv_footprint_overview_<modality>.{png,pdf}  every dataset of a modality in one grid

Run:  python plot_kv_cache_footprint.py
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb
from matplotlib.patches import FancyBboxPatch

SRC_DIR = Path(__file__).resolve().parent
HEADER_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<items>[\d\s,]+)\)$")

# Nominal corpus sizes, in file-suffix order. Missing files are simply skipped.
COLUMNS: list[tuple[str, int]] = [
    ("1k", 1_000),
    ("10k", 10_000),
    ("100k", 100_000),
    ("1M", 1_000_000),
]

# --- palette -----------------------------------------------------------------
# Green (small) -> red (large): ColorBrewer RdYlGn, reversed, without its darkest ends.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GREEN_RED_RAMP = [
    "#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#fee08b", "#fdae61", "#f46d43", "#d73027",
]
CMAP = LinearSegmentedColormap.from_list("kv_green_red", GREEN_RED_RAMP)
# Fixed 1 GB -> 10 PB scale, so colours mean the same thing in every figure.
NORM = LogNorm(vmin=1.0, vmax=1e7)

# Serif, matching the repo's other figures (reasondb.evaluation.plotting,
# scripts/plot_kvop.py). STIXGeneral is a Times face that ships with matplotlib, so it
# renders the same on every machine; Times New Roman is the fallback.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
})

#: Type sizes. The cells are fixed-size boxes, so the in-cell numbers are the ones to
#: watch: at these sizes the widest ("580 TB") still leaves a third of the cell free.
FONT = {
    "cell": 15,        # the size inside a cell, and the ratio and column labels beside it
    "cell_overview": 14.5,
    "axis": 16,        # "number of items", "compression ratio"
    "head": 17,        # the model (or dataset) above a panel
    "side": 16,        # the model down the side of an overview row
    "bar": 13,         # the colour bar's caption and tick labels
}

CELL_W, CELL_H = 1.04, 0.48  # inches

#: Room left of the leftmost label, and between the side labels and the ratio ticks.
EDGE_IN, LABEL_GAP_IN = 0.08, 0.12
#: How far left of a panel its ratio ticks end, in inches.
RATIO_TICK_IN = 0.16 * CELL_W


def text_width_in(text: str, size: float, weight: str = "normal") -> float:
    """Width of *text* in inches as the current font sets it -- measured, not guessed."""
    probe = plt.figure()
    t = probe.text(0, 0, text, fontsize=size, fontweight=weight)
    width = t.get_window_extent(probe.canvas.get_renderer()).width / probe.dpi
    plt.close(probe)
    return width


def ratio_axis_label(block_h: float) -> tuple[str, float]:
    """The rotated "compression ratio" beside a block *block_h* inches tall.

    One line when it fits along the block, two stacked lines when it would overrun it --
    at the size it is set in, the one-line label is longer than a three-row panel.
    Returns the label and how thick it is across, in inches.
    """
    one_line = "compression ratio"
    line_h = 1.2 * FONT["axis"] / 72
    if text_width_in(one_line, FONT["axis"]) <= block_h - 0.06:
        return one_line, line_h
    return "compression\nratio", 2 * line_h


def ratio_ticks_width_in(table: "Table", size: float) -> float:
    return max(text_width_in(r, size) for m in table.models for r in table.ratios[m])
CELL_GAP = 0.022             # surface gap between cells, in cell units


def human_gb(gb: float) -> str:
    """Format a GB figure with the unit that keeps it in [1, 1000)."""
    for scale, unit in ((1e6, "PB"), (1e3, "TB"), (1.0, "GB")):
        if gb >= scale:
            value = gb / scale
            return f"{value:.0f} {unit}" if value >= 10 else f"{value:.1f} {unit}"
    return f"{gb:.2g} GB"


def label_ink(rgba) -> str:
    """Ink that stays legible on top of a filled cell.

    Crossover at L = 0.179, where white-on-fill and ink-on-fill contrast are equal
    (4.58:1); either side of it, the pick is the higher-contrast one.
    """
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return INK if luminance > 0.179 else "#ffffff"


# --- data --------------------------------------------------------------------
@dataclass
class Table:
    modality: str
    datasets: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    ratios: dict[str, list[str]] = field(default_factory=dict)
    columns: list[tuple[str, int]] = field(default_factory=list)
    source_items: dict[str, int] = field(default_factory=dict)
    cells: dict[tuple[str, str, str, int], float] = field(default_factory=dict)

    def value(self, dataset: str, model: str, ratio: str, items: int) -> float | None:
        return self.cells.get((dataset, model, ratio, items))


def load(modality: str) -> Table:
    table = Table(modality=modality)
    for tag, nominal in COLUMNS:
        path = SRC_DIR / f"kv_cache_footprint_{modality}_{tag}.csv"
        if not path.exists():
            continue
        rows = list(csv.reader(path.open()))
        header, body = rows[0], [r for r in rows[1:] if r]

        columns: dict[int, tuple[str, int]] = {}
        for i, col in enumerate(header):
            m = HEADER_RE.match(col)
            if m:
                items = int(m.group("items").replace(",", "").replace(" ", ""))
                columns[i] = (m.group("name"), items)

        table.columns.append((tag, nominal))
        for row in body:
            model, ratio = row[0], row[1]
            if model not in table.models:
                table.models.append(model)
                table.ratios[model] = []
            if ratio not in table.ratios[model]:
                table.ratios[model].append(ratio)
            for i, (dataset, items) in columns.items():
                if dataset not in table.datasets:
                    table.datasets.append(dataset)
                    table.source_items[dataset] = items
                # Normalise to the nominal corpus size; footprint is linear in items.
                table.cells[(dataset, model, ratio, nominal)] = float(row[i]) * nominal / items
        print(f"read {path.name}")
    return table


# --- drawing -----------------------------------------------------------------
def draw_panel(fig, rect, table, dataset, model, *, show_ratios, show_columns, fontsize,
               show_items_title=False):
    """One model's grid: ratios down, corpus sizes across. `rect` is in inches."""
    x, y, w, h = rect
    ax = fig.add_axes([x / fig.get_figwidth(), y / fig.get_figheight(),
                       w / fig.get_figwidth(), h / fig.get_figheight()])
    ratios = table.ratios[model]
    ax.set_xlim(0, len(table.columns))
    ax.set_ylim(len(ratios), 0)
    ax.set_axis_off()

    for i, ratio in enumerate(ratios):
        for j, (tag, items) in enumerate(table.columns):
            gb = table.value(dataset, model, ratio, items)
            if gb is None:
                continue
            rgba = CMAP(NORM(gb))
            ax.add_patch(FancyBboxPatch(
                (j + CELL_GAP, i + CELL_GAP), 1 - 2 * CELL_GAP, 1 - 2 * CELL_GAP,
                boxstyle="round,pad=0,rounding_size=0.045",
                facecolor=rgba, edgecolor="none", mutation_aspect=CELL_W / CELL_H,
            ))
            ax.text(j + 0.5, i + 0.5, human_gb(gb), ha="center", va="center",
                    fontsize=fontsize, color=label_ink(rgba))
        if show_ratios:
            ax.text(-0.16, i + 0.5, ratio, ha="right", va="center",
                    fontsize=fontsize, color=INK_2, clip_on=False)

    if show_columns:
        for j, (tag, _) in enumerate(table.columns):
            ax.text(j + 0.5, -0.42, tag, ha="center", va="center",
                    fontsize=fontsize, color=INK_2, clip_on=False)
    if show_items_title:
        ax.text(len(table.columns) / 2, -1.05, "number of items", ha="center", va="center",
                fontsize=FONT["axis"], color=INK_2, clip_on=False)
    return ax


CAPTION_W = 1.25  # inches reserved left of the colorbar for its caption
#: Colorbar length, inches. Its last two ticks (1 PB, 10 PB) are one decade apart out of
#: seven, so the bar has to be long enough for their two labels to fit side by side.
BAR_W = 3.80
BAR_H = 0.16      # colorbar thickness, inches
#: Heights above the top row of cells, in inches: the model or dataset name, and the
#: bottom of the colour bar, whose tick labels hang below it and must clear that name.
HEAD_LINE = 0.82
BAR_LINE = 1.26


def add_colorbar(fig, x, y, width, *, height=0.13, caption="KV cache size"):
    fig.text(x / fig.get_figwidth(), (y + height / 2) / fig.get_figheight(), caption,
             ha="left", va="center", fontsize=FONT["bar"], color=INK_2)
    x += CAPTION_W
    cax = fig.add_axes([x / fig.get_figwidth(), y / fig.get_figheight(),
                        width / fig.get_figwidth(), height / fig.get_figheight()])
    bar = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    ticks = [1, 1e2, 1e4, 1e6, 1e7]
    bar.set_ticks(ticks)
    bar.set_ticklabels([human_gb(t).replace(".0", "") for t in ticks])
    bar.outline.set_visible(False)
    cax.minorticks_off()
    cax.tick_params(axis="x", length=0, pad=3, labelsize=FONT["bar"], labelcolor=INK_2)


def figure_for_dataset(table: Table, dataset: str, outdir: Path, dpi: int, formats):
    models = table.models
    n_cols = len(table.columns)
    panel_w = n_cols * CELL_W
    # Top strip, panel upwards: column tags, "number of items", the model name, and
    # the colour bar on the top line (there is no title).
    top, bottom = BAR_LINE + BAR_H + 0.06, 0.10
    max_rows = max(len(table.ratios[m]) for m in models)
    # Left margin, outward from the panel: the ratio ticks, a gap, the rotated axis label.
    axis_label, axis_t = ratio_axis_label(max_rows * CELL_H)
    axis_x = EDGE_IN + axis_t / 2
    left = (axis_x + axis_t / 2 + LABEL_GAP_IN
            + ratio_ticks_width_in(table, FONT["cell"]) + RATIO_TICK_IN)
    gap, right = 0.48, 0.16

    fig_w = left + len(models) * panel_w + (len(models) - 1) * gap + right
    fig_h = top + max_rows * CELL_H + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))

    def fx(inches):
        return inches / fig_w

    def fy(inches):
        return inches / fig_h

    for k, model in enumerate(models):
        rows = len(table.ratios[model])
        x = left + k * (panel_w + gap)
        y = fig_h - top - rows * CELL_H
        draw_panel(fig, (x, y, panel_w, rows * CELL_H), table, dataset, model,
                   show_ratios=True, show_columns=True, fontsize=FONT["cell"],
                   show_items_title=True)
        fig.text(fx(x), fy(fig_h - top + HEAD_LINE), model, ha="left", va="center",
                 fontsize=FONT["head"], color=INK, fontweight="semibold")

    fig.text(fx(axis_x), fy(fig_h - top - max_rows * CELL_H / 2), axis_label,
             rotation=90, ha="center", va="center", multialignment="center",
             fontsize=FONT["axis"], color=INK_2)

    # No title -- the paper caption is one. The colour bar keeps the top line to itself.
    add_colorbar(fig, fig_w - right - CAPTION_W - BAR_W, fig_h - top + BAR_LINE, BAR_W,
                 height=BAR_H)

    save(fig, outdir / f"kv_footprint_{table.modality}_{slug(dataset)}", dpi, formats)


def figure_overview(table: Table, outdir: Path, dpi: int, formats):
    datasets, models = table.datasets, table.models
    n_cols = len(table.columns)
    panel_w = n_cols * CELL_W
    top, bottom, row_gap = BAR_LINE + BAR_H + 0.06, 0.10, 0.80
    # Left margin, outward from the panels: ratio ticks, a gap, the rotated axis label, a
    # gap, the model name. One label style for the whole figure, chosen by the shortest
    # panel, so the rows do not mix one-line and two-line labels.
    axis_label, axis_t = ratio_axis_label(
        min(len(table.ratios[m]) for m in models) * CELL_H)
    side_t = 1.2 * FONT["side"] / 72
    side_x = EDGE_IN + side_t / 2
    axis_x = side_x + side_t / 2 + LABEL_GAP_IN + axis_t / 2
    left = (axis_x + axis_t / 2 + LABEL_GAP_IN
            + ratio_ticks_width_in(table, FONT["cell_overview"]) + RATIO_TICK_IN)
    gap, right = 0.46, 0.16

    fig_w = left + len(datasets) * panel_w + (len(datasets) - 1) * gap + right
    rows_h = sum(len(table.ratios[m]) * CELL_H for m in models) + (len(models) - 1) * row_gap
    fig_h = top + rows_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))

    def fx(inches):
        return inches / fig_w

    def fy(inches):
        return inches / fig_h

    y = fig_h - top
    for model in models:
        rows = len(table.ratios[model])
        y -= rows * CELL_H
        for k, dataset in enumerate(datasets):
            x = left + k * (panel_w + gap)
            draw_panel(fig, (x, y, panel_w, rows * CELL_H), table, dataset, model,
                       show_ratios=(k == 0), show_columns=True,
                       fontsize=FONT["cell_overview"], show_items_title=True)
            if model == models[0]:
                fig.text(fx(x), fy(fig_h - top + HEAD_LINE), dataset, ha="left",
                         va="center", fontsize=FONT["head"], color=INK,
                         fontweight="semibold")
        fig.text(fx(side_x), fy(y + rows * CELL_H / 2), model, rotation=90,
                 ha="center", va="center", fontsize=FONT["side"], color=INK,
                 fontweight="semibold")
        fig.text(fx(axis_x), fy(y + rows * CELL_H / 2), axis_label, rotation=90,
                 ha="center", va="center", multialignment="center",
                 fontsize=FONT["axis"], color=INK_2)
        y -= row_gap

    # No title -- the paper caption is one. The colour bar keeps the top line to itself.
    add_colorbar(fig, fig_w - right - CAPTION_W - BAR_W, fig_h - top + BAR_LINE, BAR_W,
                 height=BAR_H)

    save(fig, outdir / f"kv_footprint_overview_{table.modality}", dpi, formats)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save(fig, stem: Path, dpi: int, formats):
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        # Crop to the drawn content: whatever margin the layout left over comes off here.
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=SRC_DIR / "figures")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    for modality in ("image", "text"):
        table = load(modality)
        if not table.columns:
            continue
        for dataset in table.datasets:
            figure_for_dataset(table, dataset, args.outdir, args.dpi, formats)
        figure_overview(table, args.outdir, args.dpi, formats)


if __name__ == "__main__":
    main()
