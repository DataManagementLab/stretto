"""Figures over the merged sweep CSVs: guarantee satisfaction and phase breakdowns.

The drawing half of :mod:`reasondb.evaluation.sweep_frames`, which holds the aggregation.
Split so the aggregation can be unit-tested without a display or a font cache.

Two figure families, both parameterized by the sweep axis being compared rather than
hardcoding one:

* :func:`plot_target_met` - per-query ``achieved / target`` ratios as boxes, one facet
  per dataset plus a pooled one, a rule at 1.0.
* :func:`plot_phase_breakdown_facets` / :func:`plot_phase_breakdown_grouped` - stacked
  wall-clock, summed over a dataset's queries and combined across datasets by the
  geometric mean. Three axes in one panel: the arm on x, the guarantee target on the bar
  colour, the phase on its hatch, with a legend for the latter two.

Adapted from ``scripts/plot_benchmark.py``, which draws the same two things for
``run_benchmark``'s differently-shaped ``*metrics.csv``.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

from reasondb.evaluation.plotting import LABEL_MAP, fix_labels
from reasondb.evaluation.sweep_frames import (
    APPROACH_ORDER,
    OVERALL,
    PHASE_COLUMNS,
    PHASE_LABELS,
    POOLING_NOTES,
    dataset_col_order,
    drop_all_zero_phases,
    guarantee_short_label,
)

logger = logging.getLogger(__name__)

# Hatch lines are drawn in the patch's (white) edge colour. Thin, because the pattern is
# dense: thicker strokes merge into a solid wash. Must stay in step with the repeat count
# in `HATCH_CYCLE`. Assigned rather than ``setdefault`` because the key always exists.
plt.rcParams["hatch.linewidth"] = 0.25
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

#: Resolution the hatched bars are rasterized into the PDF at (see ``_stack_bars``). Only
#: that layer - the type, axes and legend on the same page stay vector - so this is not the
#: figure's resolution. 1200 dpi is the usual floor for line art, and the flat-colour layer
#: compresses well, so the higher resolution costs little file size.
RASTER_DPI = 1200

#: Paul Tol bright, matching the live dashboard's ``BREAKDOWN_PALETTE`` and
#: ``scripts/plot_benchmark.py`` so the same phase is the same colour everywhere.
BREAKDOWN_PALETTE: Dict[str, str] = {
    "Execution": "#004488",
    "Profiling": "#DDAA33",
    "Optimization": "#BB5566",
    "Configuring": "#66CCEE",
    "Reasoning": "#228833",
    "Other": "#BBBBBB",
}

#: The pooled accuracy panel is a distribution, not a total, so it cannot be a geometric
#: mean and must not claim to be one.
POOLED = "All datasets"

#: Precision/recall in the guarantee figure. Passed explicitly because ``map_dataframe``
#: hands each facet a single ``color=``, which seaborn then expands into a gradient over
#: the hue levels - rendering one of the two boxes near-black and warning about it.
TARGET_TYPE_PALETTE = {"Precision": "#DDAA33", "Recall": "#004488"}

#: The bands above and below the rule at 1.0, as ``((low, high), (colour, label))`` with
#: ``None`` meaning "to the axis limit". Defined once so the shading and its legend entry
#: stay consistent.
TARGET_BANDS = [
    ((1.0, None), ("green", "Target met")),
    ((None, 1.0), ("red", "Target missed")),
]
BAND_ALPHA = 0.12

#: The colour scheme for every figure in this module: the *arm* (the approach, or whatever
#: the experiment compares) takes the **hue**, and the guarantee target takes the
#: **brightness** of that hue. Hues are keyed by arm *name*, so an approach keeps its
#: colour across figures.
ARM_HUES = [
    "#1F5C99",
    "#A63A3A",
    "#2E7D4F",
    "#7A4E9E",
    "#B26A00",
    "#3C7A7A",
    # One per entry of APPROACH_ORDER; beyond that, colours wrap around and repeat.
    "#94427A",
]

#: How far the darkest and brightest target move their arm's hue: positive mixes toward
#: black, negative toward white (:func:`_mix`). Dark is the *lowest* target, so a group
#: reads left to right as one gradient. The bright end stops short of pale because the
#: white hatch would otherwise become invisible.
SHADE_RANGE = (0.45, -0.15)

#: The target legend cannot use a hue - the hue is the *arm*, and the legend describes the
#: brightness axis alone - so its swatches are this grey run through the same ladder.
LEGEND_GREY = "#707070"

#: Hatches distinguishing the phases stacked *within* one bar, plain first. Colour is
#: spent on the arm and target, so the stack is told apart by texture.
#:
#: Repeating the character requests a finer pattern from matplotlib, which reads better
#: at column width. Capped at ten repeats because the PDF backend emits hatch strokes per
#: patch, so finer hatches increase file size.
HATCH_CYCLE = [
    "",
    "/" * 10,
    "." * 10,
    "\\" * 10,
    "x" * 10,
    "+" * 10,
    "o" * 10,
]

#: Proportional scale applied to the width-derived panel height of every faceted figure
#: (see :func:`_facet_geometry`).
HEIGHT_SCALE = 0.88

#: Dash patterns for the arm on a curve whose x is not the arm itself. Pinned rather than
#: left to seaborn's own cycle because the legend below is drawn by hand and has to
#: reproduce them exactly. Solid first, so a single-arm figure is unchanged.
ARM_DASHES: List[Tuple[float, ...]] = [
    (1, 0),
    (4, 1.5),
    (1, 1),
    (3, 1, 1, 1),
    (5, 1, 1, 1, 1, 1),
]

#: Total figure width in inches. A4 is 8.27in wide, so this is the full text width at a
#: typical 0.6in margin. Every figure is laid out *to* this width rather than accumulating
#: it per facet.
A4_TEXT_WIDTH_IN = 7.0

#: Fraction of each group's unit width the bars occupy. The remainder is the gutter that
#: separates one arm's group from the next, and is what makes a group legible as a group.
GROUP_SPAN = 0.68

#: Fraction of its slot a single bar fills, so adjacent bars in a group stay distinct.
BAR_FILL = 0.86

#: Where the x-axis label starts when it shares its row with the legend
#: (:func:`_place_inline_legend`). Left of centre, as far as the y-axis label and the
#: leftmost tick allow, so the legend has the rest of the row.
INLINE_LEGEND_LABEL_X = 0.12

#: Rough width of one character as a fraction of the font size, for the legend, whose
#: entries are sized against a nominal word rather than against their own text.
#: Tick labels are measured instead - see :func:`text_width_em`.
_CHAR_WIDTH_EM = 0.60

#: Clear space each tick label keeps to its neighbour, as a fraction of its own width.
#: Shared by the two rules that measure a tick: the one that sizes the type against the
#: label (:func:`_shrink_fonts`) and the one that blanks the labels the type could not fit
#: (:func:`stride_labels`). Both must use the same margin, or a label sized to just fit
#: would then be blanked.
_TICK_WIDTH_SAFETY = 1.10

#: Angle every categorical x tick is rotated by - ``plot_benchmark``'s. Shallow on purpose:
#: a 45 degree label is *taller* than a 25 degree one, so the horizontal overlap a steeper
#: angle avoids is paid for in figure height, which is the scarcer resource here.
TICK_ROTATION_DEG = 25.0

#: How much of the room left of a panel's first tick a rotated label may claim.
#:
#: Labels are anchored ``ha="right"``, so the binding case is the leftmost tick, which
#: seaborn pads by half a slot. Slightly under 1.0 because the neighbouring panel's y tick
#: labels sit in the same strip.
_LABEL_LEFT_ROOM = 0.9

#: Smallest tick type this will produce. A label that cannot fit even here is reported
#: rather than shrunk into illegibility - see :func:`_shrink_fonts`.
_TICK_FLOOR_PT = 4.4

#: The band every figure's tick type is held inside, whatever its panel count.
#:
#: The available room picks a size *inside* this band, keeping type consistent across the
#: figures of one experiment. Only a label that will not fit may go below it, down to
#: :data:`_TICK_FLOOR_PT`.
_TICK_BAND_PT = (5.8, 7.4)

#: Gutter between panels, as a fraction of panel width - one per figure family, because
#: what has to fit in it differs.
#:
#: Every runtime family draws with ``sharey=False``, so every panel carries its own y tick
#: labels in the gutter to its left. Values are fixed empirically for the tightest layout
#: (six panels, longest y labels), since the final layout cannot be measured while drawing.
Y_GUTTER_BOXES = 0.12    # target-met: shared y, so only the first panel is labelled
Y_GUTTER_BARS = 0.20     # phase breakdown: per-panel hour totals, up to 3 digits
Y_GUTTER_CURVES = 0.24

#: How far the legend hangs below the lowest ink it sits under, as a fraction of one
#: legend row. Must stay non-negative: a lift would overlap the other panels' tick labels.
LEGEND_CLEARANCE_ROWS = 0.10

#: How a numeric-x figure reports the size of the search space at each x position:
#: ``"top-axis"``, ``"point"``, ``"tick"``, or ``"none"``. See
#: :func:`annotate_operator_counts` for what each one costs.
OPERATOR_AXIS_MODE = "top-axis"

#: Whether a numeric-x metric figure rings and labels the cheapest point of each curve.
#: Off by default; enabled with ``--annotate-minimum``, which writes a separate file.
ANNOTATE_MINIMA = False

#: ``"linear"``, ``"sqrt"`` or ``"symlog"`` for a numeric x-axis. :func:`apply_x_scale`.
X_SCALE = "linear"

#: Per-axes symlog threshold, so a twin axis can reproduce its host's transform.
SYMLOG_LINTHRESH: Dict[Any, float] = {}


# ---------------------------------------------------------------------------
# The colour scheme: hue is the arm, brightness is the guarantee target
# ---------------------------------------------------------------------------


def _mix(color: str, amount: float) -> str:
    """*color* moved toward black (*amount* > 0) or white (*amount* < 0), 0 leaves it."""
    from matplotlib.colors import to_hex, to_rgb

    red, green, blue = to_rgb(color)
    toward = 0.0 if amount >= 0 else 1.0
    weight = min(abs(float(amount)), 1.0)
    return to_hex(tuple(c * (1 - weight) + toward * weight for c in (red, green, blue)))


def shade_amounts(n_levels: int) -> List[float]:
    """The :data:`SHADE_RANGE` ladder in *n_levels* steps, darkest first.

    A single level is left unshaded: with one guarantee target there is no brightness axis.
    """
    if n_levels <= 1:
        return [0.0]
    dark, bright = SHADE_RANGE
    step = (bright - dark) / (n_levels - 1)
    return [dark + step * i for i in range(n_levels)]


#: Arms that *are* a named approach under another label, so that they take its hue and the
#: same system is the same colour across experiments. The ablation's "Stretto, vanilla
#: only" is deliberately absent so it does not share ``optim_global``'s hue.
ARM_HUE_ALIASES: Dict[str, str] = {
    "Stretto": "optim_global",
    "No optimization": "no_optim",
    "Reordering only": "no_optim_reorder",
}


def arm_colors(arm_order: Sequence[str]) -> Dict[str, str]:
    """Hue per arm: by approach identity where there is one, by position for the rest.

    Keying an approach on its *name* keeps its colour stable across figures. Arms without
    a canonical identity (a sample size, a sweep state) fall back to position, skipping
    hues the named arms already claimed so no two arms of one figure share a colour.
    """
    order = [str(a) for a in arm_order]
    colors: Dict[str, str] = {}
    for name in order:
        approach = ARM_HUE_ALIASES.get(name, name)
        if approach in APPROACH_ORDER:
            colors[name] = ARM_HUES[APPROACH_ORDER.index(approach) % len(ARM_HUES)]
    taken = set(colors.values())
    free = [h for h in ARM_HUES if h not in taken] or list(ARM_HUES)
    unnamed = [n for n in order if n not in colors]
    for position, name in enumerate(unnamed):
        colors[name] = free[position % len(free)]
    return colors


def arm_hue(arm: str, arm_order: Sequence[str]) -> str:
    """The hue for *arm* - see :func:`arm_colors`."""
    return arm_colors(arm_order).get(str(arm), ARM_HUES[0])


def arm_shade(
    arm: str, arm_order: Sequence[str], level: int, n_levels: int
) -> str:
    """:func:`arm_hue` for *arm*, at brightness *level* of *n_levels* (0 = darkest)."""
    amounts = shade_amounts(n_levels)
    return _mix(arm_hue(arm, arm_order), amounts[min(level, len(amounts) - 1)])


#: What a caller may hand a drawer as the hue: nothing (derive it from the arm order), one
#: colour for the whole figure, or one per arm.
HueSpec = Union[str, Mapping[str, str], None]


def resolve_hue(hue: HueSpec, arm: str, default: Optional[str] = None) -> Optional[str]:
    """The hue *arm* takes under *hue*, or *default* where *hue* does not name one.

    The mapping form lets a figure whose x-axis is *not* the approach still colour by
    approach when more than one is present (e.g. a sample-size sweep with a reference arm).
    A plain string is the single-approach case. Arms the mapping does not cover fall back
    to *default*.
    """
    if hue is None:
        return default
    if isinstance(hue, str):
        return hue
    return hue.get(str(arm), default)


def grey_shades(n_levels: int) -> List[str]:
    """:data:`LEGEND_GREY` on the same ladder - the legend's stand-in for the hue axis."""
    return [_mix(LEGEND_GREY, a) for a in shade_amounts(n_levels)]


# ---------------------------------------------------------------------------
# Page fitting, shared with scripts/plot_label_reference.py
# ---------------------------------------------------------------------------


def page_geometry(
    n_facets: int, width_in: float, n_x: int, height_ratio: float = 1.65
) -> Dict[str, object]:
    """Facet height/aspect and type sizes for a row laid out to *width_in* inches.

    The public seam over :func:`_facet_geometry` and :func:`_shrink_fonts`, so a figure
    outside this module gets the same treatment rather than its own constants. ``n_x`` is
    how many ticks a panel carries - the room one of them gets is what the type sizes are
    derived from.
    """
    height, aspect, panel_width, col_wrap = _facet_geometry(
        n_facets, width_in, height_ratio=height_ratio
    )
    return {
        "height": height,
        "aspect": aspect,
        "col_wrap": col_wrap,
        "fonts": _shrink_fonts(panel_width / max(n_x, 1)),
    }


def finish_page_figure(
    grid: sns.FacetGrid,
    *,
    fonts: Dict[str, float],
    width_in: float,
    out_path: Path,
    handles: Optional[Sequence] = None,
    labels: Optional[Sequence[str]] = None,
    rotation: int = 25,
    wspace: float = 0.14,
    legend_inline: bool = False,
) -> Path:
    """Retitle, shrink, de-duplicate the axis labels, park the legend, and save.

    Everything a faceted figure needs between "the data is drawn" and "this fits a page":
    facet titles through ``LABEL_MAP``, one axis label per row rather than one per panel,
    type scaled to the panel, and a legend measured against the axes' own lowest ink.

    ``legend_inline`` puts a short legend on the *same row* as the x-axis label instead of
    under it, saving a row of height (off by default).
    """
    _relabel_facet_titles(grid, fonts["title"])
    _rotate_ticks(grid, rotation=rotation, fontsize=fonts["tick"])
    _apply_label_sizes(grid, fonts)
    # One x and one y label for the whole row rather than one per panel.
    axes = list(grid.axes.flat)
    for index, ax in enumerate(axes):
        if index:
            ax.set_ylabel("")
        if index != len(axes) // 2:
            ax.set_xlabel("")
    _relayout(grid.figure)
    grid.figure.subplots_adjust(wspace=wspace)
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    if grid.legend is not None:
        grid.legend.remove()
    if handles is not None and labels is not None:
        if legend_inline:
            _place_inline_legend(grid, axes, handles, labels, fonts)
        else:
            grid.figure.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, _content_bottom(grid.figure)),
                ncol=_legend_columns(width_in, labels, fonts["legend"]),
                frameon=False,
                fontsize=fonts["legend"],
                handlelength=1.4,
                columnspacing=1.0,
            )
    return _save_to_width(grid.figure, out_path, width_in)


def _place_inline_legend(
    grid: sns.FacetGrid,
    axes: Sequence,
    handles: Sequence,
    labels: Sequence[str],
    fonts: Dict[str, float],
) -> None:
    """Draw the x-axis label and the legend side by side on one row.

    The axis label is redrawn as figure text so it can sit beside the legend rather than
    centred on its own panel.
    """
    figure = grid.figure
    label = next((ax.get_xlabel() for ax in axes if ax.get_xlabel()), "")
    for ax in axes:
        ax.set_xlabel("")
    top = _content_bottom(figure)

    # Both positions are set at construction: `set_bbox_to_anchor` on an existing figure
    # legend is not reliably honoured at render time. Label and legend spread across the
    # row, so neither needs the other's width.
    figure.text(
        INLINE_LEGEND_LABEL_X, top, label,
        ha="left", va="top", fontsize=fonts["label"],
    )
    # Anchor to the right edge of the last panel, not of the figure, so the legend does
    # not overhang the panel grid.
    right = max((ax.get_position().x1 for ax in axes), default=1.0)
    figure.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(right, top),
        ncol=len(handles), frameon=False, fontsize=fonts["legend"],
        handlelength=1.4, columnspacing=1.0, borderpad=0.0, handletextpad=0.5,
    )


# ---------------------------------------------------------------------------
# Guarantee satisfaction
# ---------------------------------------------------------------------------


def target_met_frame(df: pd.DataFrame, arm_col: str = "arm") -> pd.DataFrame:
    """Long-form ``target_met = achieved / target``, one half per metric.

    The sweep schema carries no ``precision``/``recall`` columns - ``achieved_precision``
    and ``achieved_recall`` *are* the measured values, filled by the merge's scoring
    pass against the labeller named in the ``labels`` column (``silver`` on every random
    benchmark, since none of them has ground truth).
    """
    missing = [
        c
        for c in ("achieved_precision", "achieved_recall", "precision_guarantee", "recall_guarantee")
        if c not in df.columns
    ]
    if missing:
        raise SystemExit(
            f"Columns {missing} are absent, so guarantee satisfaction cannot be drawn. "
            "A sweep whose scoring pass never ran has achieved_* entirely empty - merge "
            "the task first, or pass --figures breakdown."
        )

    halves = []
    for metric, target, name in (
        ("achieved_precision", "precision_guarantee", "Precision"),
        ("achieved_recall", "recall_guarantee", "Recall"),
    ):
        half = df.copy()
        half["target_met"] = df[metric] / df[target]
        half["target_type"] = name
        halves.append(half)
    out = pd.concat(halves, axis=0, ignore_index=True)
    kept = out.dropna(subset=["target_met"])
    if len(kept) < len(out):
        logger.warning(
            "%d of %d rows have no achieved_* value and were dropped; was the task "
            "scored?",
            len(out) - len(kept),
            len(out),
        )
    if kept.empty:
        raise SystemExit(
            "Every achieved_* value is empty - this task was never scored, so there is "
            "no guarantee satisfaction to plot. Use --figures breakdown."
        )
    return kept


def plot_target_met(
    df: pd.DataFrame,
    *,
    arm_col: str = "arm",
    arm_order: Sequence[str],
    tick_text: Optional[Dict[str, str]] = None,
    out_path: Path,
    panels: str = "all",
    ylim: Tuple[float, float] = (0.2, 2.0),
    xlabel: str = "",
    width_in: float = A4_TEXT_WIDTH_IN,
    pooled_label: str = POOLED,
    include_pooled: bool = True,
    match_panels: Optional[int] = None,
    match_width: float = A4_TEXT_WIDTH_IN,
) -> Path:
    """Boxes of ``achieved / target`` per arm, one facet per dataset.

    ``1.0`` is exactly hitting the target; the green band above it and the red band
    below make "met" readable without counting. Whiskers are the 5th/95th percentiles.

    ``panels="all"`` draws the pooled facet alongside the datasets, ``panels="overall"``
    draws only the pooled one.

    Unlike the runtime figures, this family keeps its pooled facet under ``--facet-by``:
    pooling here concatenates per-query ratios, which stays meaningful over buckets that
    partition one query set. ``pooled_label`` names it; ``include_pooled=False``
    (``--no-overall``) drops it.
    """
    plot_df = target_met_frame(df, arm_col)
    pooled = plot_df.assign(dataset=pooled_label)
    if panels == "overall":
        plot_df = pooled
    elif include_pooled:
        plot_df = pd.concat([pooled, plot_df], axis=0, ignore_index=True)

    col_order = [pooled_label] + [
        d for d in dataset_col_order(plot_df) if d != pooled_label
    ]
    col_order = [d for d in col_order if d in set(plot_df["dataset"].astype(str))]
    order = [a for a in arm_order if a in set(plot_df[arm_col].astype(str))]

    # Shorter than the metrics figures: the fixed 0.2-2.0 axis reads well at any height,
    # and this family needs room for rotated ticks and a legend under the axes.
    matched = (
        _facet_geometry(match_panels, match_width, height_ratio=1.40)
        if match_panels
        else None
    )
    # One panel of the faceted figure, at its size and in its type - see
    # `plot_phase_breakdown_facets` and :func:`_fit_panel`.
    cut_out = bool(matched) and len(col_order) == 1
    height, aspect, panel_width, col_wrap = (
        _cut_out_geometry(matched, width_in, height_ratio=1.40) if cut_out
        else _facet_geometry(
            len(col_order), width_in, height_ratio=1.40,
            height_in=matched[0] if matched else None,
        )
    )
    # The same rule the runtime families use, for consistent type across figures.
    label_em, label_upright = _tick_geometry(
        [""], lambda _: order, lambda _: tick_text or {}
    )
    fonts = _shrink_fonts(panel_width / max(len(order), 1), label_em, label_upright)
    # Scale stroke widths with the type size so lines stay thin at column width.
    line_w = max(0.18, fonts["tick"] / 15.0)
    grid = sns.FacetGrid(
        plot_df,
        col="dataset",
        col_order=col_order,
        col_wrap=col_wrap,
        margin_titles=col_wrap is None,
        sharey=True,
        height=height,
        aspect=aspect,
    )
    for ax in grid.axes.flat:
        ax.axhline(1.0, color="black", linewidth=line_w * 1.5, linestyle="--", zorder=1)
        for (low, high), (colour, _) in TARGET_BANDS:
            ax.axhspan(
                ylim[0] if low is None else low,
                ylim[1] if high is None else high,
                color=colour,
                alpha=BAND_ALPHA,
                zorder=0,
            )

    grid.map_dataframe(
        sns.boxplot,
        x=arm_col,
        y="target_met",
        order=order,
        hue="target_type",
        hue_order=["Precision", "Recall"],
        palette=TARGET_TYPE_PALETTE,
        whis=(5, 95),
        showfliers=False,
        linewidth=line_w,
        # Not every seaborn version propagates `linewidth` to caps and whiskers.
        medianprops={"linewidth": line_w * 1.4},
        whiskerprops={"linewidth": line_w},
        capprops={"linewidth": line_w},
    )

    ticks = dict(tick_text or {})
    for ax in grid.axes.flat:
        ax.set_ylim(*ylim)
        # Explicit ticks: the automatic locator places too few on a 0.2-2.0 axis.
        ax.set_yticks([t for t in (0.5, 1.0, 1.5, 2.0) if ylim[0] <= t <= ylim[1]])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            stride_labels(
                [ticks.get(a, LABEL_MAP.get(a, a)) for a in order],
                panel_width, fonts["tick"],
            )
        )
    grid.set_axis_labels(xlabel, "Achieved / target")
    _relabel_facet_titles(grid, fonts["title"])
    _rotate_ticks(grid, fontsize=fonts["tick"])
    _apply_label_sizes(grid, fonts)
    # One y label for the row: the panels share this scale.
    for index, ax in enumerate(grid.axes.flat):
        if index:
            ax.set_ylabel("")
            ax.set_xlabel("")
    _relayout(grid.figure)
    grid.figure.subplots_adjust(wspace=Y_GUTTER_BOXES)
    if cut_out:
        width_in = _fit_panel(
            grid.figure, grid.axes.flat[0], panel_width / (1.0 + Y_GUTTER_BOXES)
        )

    from matplotlib.patches import Patch

    handles, labels = grid.axes.flat[0].get_legend_handles_labels()
    handles, labels = list(handles[:2]), list(labels[:2])
    # Name the met/missed bands in the legend. The swatch is drawn stronger than the
    # faint axes wash so it stays visible at swatch size.
    for _, (colour, label) in TARGET_BANDS:
        handles.append(Patch(facecolor=colour, alpha=0.30, edgecolor="#888888", linewidth=0.3))
        labels.append(label)

    # Matplotlib fills legends column-major, so two columns give an aligned 2x2 grid
    # (box types | bands) that also stays narrower than a single panel.
    ncol = 2 if cut_out else _legend_columns(width_in, labels, fonts["legend"])
    # Placed by the same rule as the other families' legends (:func:`_legend_band`).
    band = _legend_band(grid.figure, grid.axes.flat[0], fonts["legend"])
    grid.figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(band.centre, band.top),
        ncol=ncol,
        frameon=False,
        fontsize=fonts["legend"],
        handlelength=PATCH_HANDLE_LENGTH,
        handleheight=0.9,
        columnspacing=1.0,
        # The anchor is already measured off the lowest ink; drop matplotlib's default pad.
        borderaxespad=0.0,
        borderpad=0.1,
    )
    for ax in grid.axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    _save_to_width(grid.figure, out_path, width_in)
    plt.close(grid.figure)
    return out_path


# ---------------------------------------------------------------------------
# Phase breakdown
# ---------------------------------------------------------------------------


def _stack_bars(
    ax,
    positions: np.ndarray,
    values: pd.DataFrame,
    columns: Sequence[str],
    width: float,
    label: bool = True,
    color: Optional[Union[str, Sequence[str]]] = None,
) -> None:
    """One stacked bar per position, bottom-to-top in ``RUNTIME_BREAKDOWN_COMPONENTS``.

    With *color* given (one colour, or one per position), every segment of a bar takes it -
    the colour encodes the arm and its target (:func:`arm_shade`) and the phases are told
    apart by hatch. With *color* ``None`` the phases are coloured with
    :data:`BREAKDOWN_PALETTE` and not hatched.

    ``values`` is reindexed to ``positions``' order by the caller, so a missing value
    leaves a gap at a fixed x rather than shifting its neighbours.
    """
    bottom = np.zeros(len(positions))
    for index, column in enumerate(columns):
        heights = values[column].to_numpy(dtype=float)
        name = PHASE_LABELS[column]
        ax.bar(
            positions,
            heights,
            width=width,
            bottom=bottom,
            color=(
                list(color)
                if color is not None and not isinstance(color, str)
                else (color or BREAKDOWN_PALETTE[name])
            ),
            label=name if label else None,
            edgecolor="white",
            linewidth=0.4,
            hatch=(HATCH_CYCLE[index % len(HATCH_CYCLE)] or None) if color else None,
            # Hatched bars are rasterized at RASTER_DPI to keep the PDF small (the backend
            # emits hatch strokes per patch); everything else stays vector, as do
            # unhatched bars.
            rasterized=bool(color),
        )
        bottom += heights


def _draw_grouped_stacks(
    ax,
    frame: pd.DataFrame,
    *,
    outer_col: str,
    outer_order: Sequence[str],
    inner_col: str,
    inner_order: Sequence[str],
    columns: Sequence[str],
    outer_ticks: Dict[str, str],
    inner_ticks: Dict[str, str],
    arm_axis: str = "outer",
    hue_override: HueSpec = None,
    panel_width_in: float = 0.0,
    tick_fontsize: float = 0.0,
    overflowing_ticks: Sequence[str] = (),
) -> None:
    """Groups along x by *outer_col*, one stacked bar per *inner_col* value in each group.

    Colour carries both categorical axes at once: the **hue** is the approach and the
    **brightness** is the guarantee target, per :func:`arm_shade`. The phases stacked
    inside a bar take the hatch. The x axis therefore carries exactly one tick per group.

    *arm_axis* says which nesting this caller uses: the faceted breakdown groups by arm
    with targets inside, while the ablation nests arms *within* a target. Either way the
    hue stays on the arm.

    *hue_override* covers figures whose compared axis is **not** the approach (a sample
    size, a sweep state): every bar takes the fixed approach's hue, and single-bar groups
    stay hatched. As a mapping, a reference arm from another task keeps its own hue
    (:func:`resolve_hue`).

    Bars sit at fixed offsets from the group centre and a missing combination keeps its
    slot, padded with NaN rather than zero so no bar is drawn (a 0.0 would look like a
    measured zero).
    """
    lookup = {
        (str(o), str(i)): row
        for (o, i), row in frame.set_index([outer_col, inner_col])[list(columns)].iterrows()
    }
    n_inner = max(len(inner_order), 1)
    single = n_inner == 1
    width = GROUP_SPAN / n_inner

    for j, inner in enumerate(inner_order):
        positions, rows = [], []
        for i, outer in enumerate(outer_order):
            positions.append(i + (j - (n_inner - 1) / 2) * width)
            row = lookup.get((str(outer), str(inner)))
            rows.append(
                dict(row) if row is not None else {c: np.nan for c in columns}
            )
        _stack_bars(
            ax,
            np.asarray(positions),
            pd.DataFrame(rows, columns=list(columns)),
            columns,
            width=width * BAR_FILL,
            label=False,
            # One colour per position, so the hue follows the arm along x. With one bar per
            # group and no hue override, the phases are coloured instead (`_stack_bars`).
            color=(
                None
                if single and hue_override is None
                else [
                    _mix(
                        resolve_hue(hue_override, str(o), ARM_HUES[0]),
                        shade_amounts(n_inner)[j],
                    )
                    if hue_override is not None
                    else arm_shade(str(o), outer_order, j, n_inner)
                    if arm_axis == "outer"
                    else arm_shade(str(inner), inner_order, i, len(outer_order))
                    for i, o in enumerate(outer_order)
                ]
            ),
        )

    ax.set_xticks(np.arange(len(outer_order)))
    labels = [outer_ticks.get(str(o), str(o)) for o in outer_order]
    # Blank labels that do not fit (e.g. long greedy walks with many states).
    if panel_width_in and tick_fontsize:
        labels = stride_labels(
            labels,
            panel_width_in,
            tick_fontsize,
            overflowing=[
                i for i, o in enumerate(outer_order)
                if str(o) in {str(a) for a in overflowing_ticks}
            ],
        )
    # 25 degrees, as in ``plot_benchmark``. Two-line labels stay upright: rotating them
    # costs more height than the overlap they avoid.
    multiline = any("\n" in t for t in labels)
    ax.set_xticklabels(
        labels,
        rotation=0 if multiline else 25,
        ha="center" if multiline else "right",
        rotation_mode=None if multiline else "anchor",
    )



#: Swatch, handle and column gap of one legend entry, in inches (everything but the label).
#: Sized for the longer of the two handles below, so it is conservative for patches.
_LEGEND_ENTRY_PAD_IN = 0.28

#: Legend handle length in ems: a line needs length to show its dash pattern, while a
#: patch (colour and hatch) is drawn as a square swatch.
PATCH_HANDLE_LENGTH = 1.0
LINE_HANDLE_LENGTH = 1.4

#: Legend columns, where the author pins them (``--legend-columns``). ``None`` measures.
LEGEND_COLUMNS: Optional[int] = None


def _legend_row_width_in(labels: Sequence[str], fontsize: float) -> float:
    """Width one row of *labels* needs, swatches and column gaps included."""
    return sum(_LEGEND_ENTRY_PAD_IN + fontsize * text_width_em(str(l)) / 72.0 for l in labels)


def _legend_columns(
    width_in: float, entries: Union[int, Sequence[str]], fontsize: float
) -> int:
    """How many legend entries fit across *width_in* at *fontsize*.

    *entries* is either the labels, in which case the widest is measured
    (:func:`text_width_em`), or a bare count, which assumes a nominal eight-character label.
    """
    n_entries = entries if isinstance(entries, int) else len(entries)
    if n_entries <= 1:
        return max(n_entries, 1)
    if LEGEND_COLUMNS:
        return max(1, min(n_entries, LEGEND_COLUMNS))
    label_em = (
        8.0 * _CHAR_WIDTH_EM if isinstance(entries, int)
        else max((text_width_em(str(e)) for e in entries), default=8.0 * _CHAR_WIDTH_EM)
    )
    # Swatch and gap, then the label itself, in inches.
    entry_in = _LEGEND_ENTRY_PAD_IN + fontsize * label_em / 72.0
    return max(1, min(n_entries, int(width_in // entry_in) or 1))


def _content_bottom(figure, gap_pt: float = 0.8) -> float:
    """Figure fraction *gap_pt* below the lowest ink the axes draw, tick labels included.

    Measured rather than a fixed offset, since the tick label extent depends on rotation.
    The gap is in points so it does not grow with figure height.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    height_px = figure.get_window_extent(renderer).height
    bottoms = [
        ax.get_tightbbox(renderer).ymin
        for ax in figure.axes
        if ax.get_tightbbox(renderer) is not None
    ]
    if not bottoms or not height_px:
        return 0.0
    height_in = figure.get_size_inches()[1]
    gap = (gap_pt / 72.0) / height_in if height_in else 0.0
    return min(bottoms) / height_px - gap


def _content_bottom_beside(figure, labelled_ax) -> float:
    """:func:`_content_bottom`, ignoring the one panel that still carries an axis label.

    With the x label drawn only once, the band under the other panels is free; measuring
    without the labelled panel lets the legend sit there, beside the label.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    height_px = figure.get_window_extent(renderer).height
    bottoms = [
        ax.get_tightbbox(renderer).ymin
        for ax in figure.axes
        if ax is not labelled_ax and ax.get_tightbbox(renderer) is not None
    ]
    return (min(bottoms) / height_px) if bottoms and height_px else 0.0


def _legend_row_fraction(figure, fontsize: float) -> float:
    """One legend row as a fraction of figure height - the unit legend offsets move in."""
    return (fontsize * 1.5 / 72.0) / max(figure.get_size_inches()[1], 0.1)


class LegendBand(NamedTuple):
    """The strip a legend may occupy, in figure fractions.

    ``left`` is where the strip starts and ``top`` its baseline; ``centre`` is the x a
    legend in it is centred on, which on a one-panel figure is not the middle of the strip.
    """

    left: float
    centre: float
    top: float


def _legend_band(figure, beside, fontsize: float) -> LegendBand:
    """Where a legend goes, in figure fractions.

    The legend hangs just under the lowest ink it sits beneath:

    * **A row of panels.** The x label is drawn once, under the leftmost panel, so the
      legend shares that row in the strip to its right (measured with
      :func:`_content_bottom_beside`) and is centred on that strip.
    * **One panel.** The legend hangs below the label, centred on the panel rather than the
      page (the y label shifts the page midpoint left), with the whole page as its room.
    """
    row = _legend_row_fraction(figure, fontsize)
    others = [ax for ax in figure.axes if ax is not beside] if beside is not None else []
    beside_a_label = beside is not None and bool(others)
    ink = (
        _content_bottom_beside(figure, beside)
        if beside_a_label
        else _content_bottom(figure)
    )
    left = _label_right_edge(figure, beside) if beside_a_label else 0.0
    centre = (
        left + (1.0 - left) / 2.0
        if beside_a_label or beside is None
        else float(np.mean(beside.get_position().intervalx))
    )
    return LegendBand(left, centre, ink - row * LEGEND_CLEARANCE_ROWS)


def _label_right_edge(figure, ax) -> float:
    """Figure fraction just past the right edge of *ax*'s x-axis label, 0 if it has none.

    Legends placed beside the single x label start here so they do not overlap it.
    """
    text = getattr(getattr(ax, "xaxis", None), "label", None)
    if text is None or not text.get_text():
        return 0.0
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width_px = figure.get_window_extent(renderer).width
    if not width_px:
        return 0.0
    return min(0.5, float(text.get_window_extent(renderer).xmax) / width_px + 0.015)


def _phase_legend(
    figure,
    columns: Sequence[str],
    width_in: float,
    fontsize: Optional[float] = None,
    inner_labels: Optional[Sequence[str]] = None,
    inner_is_arm: bool = False,
    inner_keys: Optional[Sequence[str]] = None,
    hue_override: HueSpec = None,
    beside: Optional[Any] = None,
) -> None:
    """One legend for the phase colours, and - when bars are hatched - one for those.

    Two legends rather than one combined, since colour and texture are independent axes.
    Both are placed by :func:`_legend_band`.
    """
    from matplotlib.patches import Patch

    size = fontsize or float(plt.rcParams.get("font.size", 10.0))
    # A legend row is about 1.5 line heights, as a figure fraction.
    row_fraction = _legend_row_fraction(figure, size)
    # *beside* is the one panel still carrying an axis label (:func:`_legend_band`).
    band = _legend_band(figure, beside, size)
    left, top = band.left, band.top

    # Colour legend first, but only with several bars per group; with one, the phases
    # take the colour and there is no inner axis to name.
    hatched = bool(inner_labels) and len(inner_labels) >= 2
    # If the inner value is the target (brightness), swatches are grey on the shade
    # ladder; if it is the arm (the ablation), swatches use the arm's hue.
    labels_in = list(inner_labels or [])
    # Resolve colours from the raw arm values, not the shortened display text.
    keys_in = [str(k) for k in (inner_keys if inner_keys is not None else labels_in)]
    swatch = (
        [arm_hue(k, keys_in) for k in keys_in]
        if inner_is_arm
        else grey_shades(len(labels_in))
    )
    handles = (
        [
            Patch(facecolor=swatch[i], label=label)
            for i, label in enumerate(labels_in)
        ]
        if hatched
        else []
    )
    hatch_handles = [
        Patch(
            facecolor="#E8E8E8",
            edgecolor="#333333",
            linewidth=0.5,
            hatch=HATCH_CYCLE[i % len(HATCH_CYCLE)] or None,
            label=PHASE_LABELS[c],
        )
        for i, c in enumerate(columns)
    ]
    if not handles:
        if hue_override is None:
            # Single bar per group and no approach hue: one legend of phase colours.
            handles = [
                Patch(facecolor=BREAKDOWN_PALETTE[PHASE_LABELS[c]], label=PHASE_LABELS[c])
                for c in columns
            ]
        else:
            # The approach owns the colour, so the phases stay on the hatch and their
            # legend is the only one.
            handles = hatch_handles
        hatch_handles = []
    n_hatch = len(hatch_handles)
    share = len(handles) / (len(handles) + n_hatch) if n_hatch else 1.0
    # Both legends share the part of the row right of the axis label, so widths are
    # measured against that remainder rather than the page.
    span = 1.0 - left
    band_in = max(width_in * span, 0.1)
    labels_a = [h.get_label() for h in handles]
    labels_b = [h.get_label() for h in hatch_handles]
    row_a = _legend_row_width_in(labels_a, size)
    row_b = _legend_row_width_in(labels_b, size)

    def at(fraction: float) -> float:
        """The x a legend *fraction* of the way along the band is centred on.

        Relative to the band's *centre* (see :class:`LegendBand`).
        """
        return band.centre + span * (fraction - 0.5)

    # Three placements, cheapest in height first:
    #
    #   1. both legends as one row each, side by side  - one row of height
    #   2. one row each, stacked                       - two rows
    #   3. wrapped into columns, side by side          - as many rows as the widest wraps
    #
    # Pinning `--legend-columns` always uses columns.
    side_by_side = not LEGEND_COLUMNS and row_a + row_b <= band_in
    stacked = not LEGEND_COLUMNS and not side_by_side and max(row_a, row_b) <= band_in

    if side_by_side or not n_hatch:
        first_x = at(share / 2 if n_hatch else 0.5)
        second_x = at(share + (1.0 - share) / 2)
        first_top = second_top = top
        ncol_a = len(labels_a) if side_by_side else _legend_columns(
            band_in * share, labels_a, size
        )
        ncol_b = len(labels_b) if side_by_side else _legend_columns(
            band_in * (1 - share), labels_b, size
        )
    elif stacked:
        # Two rows drawn as one legend so the columns align: matplotlib fills
        # column-major, so interleaving puts targets on the top row and phases below.
        from matplotlib.patches import Patch as _Patch

        width = max(len(handles), len(hatch_handles))
        def _padded(items):
            return list(items) + [
                _Patch(facecolor="none", edgecolor="none", label=" ")
                for _ in range(width - len(items))
            ]

        merged = [h for pair in zip(_padded(handles), _padded(hatch_handles)) for h in pair]
        figure.legend(
            handles=merged, bbox_to_anchor=(at(0.5), top), ncol=width,
            loc="upper center", frameon=False, fontsize=size,
            handlelength=PATCH_HANDLE_LENGTH,
            handleheight=0.9, columnspacing=1.0, borderpad=0.1, borderaxespad=0.0,
        )
        return
    else:
        first_x = at(share / 2)
        second_x = at(share + (1.0 - share) / 2)
        first_top = second_top = top
        ncol_a = _legend_columns(band_in * share, labels_a, size)
        ncol_b = _legend_columns(band_in * (1 - share), labels_b, size)

    common = dict(
        loc="upper center", frameon=False, fontsize=size,
        handlelength=PATCH_HANDLE_LENGTH,
        handleheight=0.9, columnspacing=1.0, borderpad=0.1, borderaxespad=0.0,
    )
    phase_legend = figure.legend(
        handles=handles, bbox_to_anchor=(first_x, first_top), ncol=ncol_a, **common
    )
    if not n_hatch:
        return
    # `figure.legend` *replaces* the figure's legend rather than appending one, so the
    # first has to be re-attached as a plain artist before the second is added.
    figure.add_artist(phase_legend)
    figure.legend(
        handles=hatch_handles, bbox_to_anchor=(second_x, second_top), ncol=ncol_b, **common
    )


def _fit_panel(figure, ax, target_axes_in: float, tries: int = 3, tol: float = 0.01) -> float:
    """Widen *figure* until *ax* is *target_axes_in* across. Returns the figure's width.

    A pooled single-panel figure is a **cut-out** of the faceted one it accompanies - the
    same panel at the same size, in the same type - so the panel width is held fixed and
    the page grows by what its decorations (sized in points) need. Converges in a step or
    two, since that overhead does not change with the page width.
    """
    height_in = figure.get_size_inches()[1]
    for _ in range(max(tries, 1)):
        figure.canvas.draw()
        axes_in = ax.get_window_extent().width / figure.dpi
        shortfall = target_axes_in - axes_in
        if abs(shortfall) <= tol:
            break
        width_in = figure.get_size_inches()[0] + shortfall
        if width_in <= 0:
            break
        figure.set_size_inches(width_in, height_in)
        # Re-solve the margins at the new page size: the decorations keep their point size,
        # so the panel takes the whole of what was added.
        figure.tight_layout()
    return float(figure.get_size_inches()[0])


def _relayout(figure) -> None:
    """Solve the figure's margins again, now that the ticks carry their final text.

    Seaborn lays a ``FacetGrid`` out while its ticks still hold the raw category values at
    the default type size, which can make ``tight_layout`` give up. Re-solving once the
    labels are final fixes the layout. Must run *before* ``subplots_adjust`` (which sets
    the gutter) and before :func:`_content_bottom`.
    """
    figure.tight_layout()


def _save_to_width(figure, out_path: Path, width_in: float, tries: int = 1) -> Path:
    """Save *figure* as a tight-cropped PDF.

    Not a rescale loop: type size is in points, so resizing would only shrink the axes.
    The width is hit by choosing the geometry up front (:func:`_facet_geometry`) and by
    wrapping the legend to fit (:func:`_legend_columns`).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `dpi` applies only to rasterized artists (the hatched bars); the rest stays vector.
    figure.savefig(out_path, bbox_inches="tight", format="pdf", dpi=RASTER_DPI)
    logger.info("Wrote %s", out_path)
    return out_path


def _relabel_facet_titles(grid: sns.FacetGrid, fontsize: Optional[float] = None) -> None:
    for ax in grid.axes.flat:
        title = ax.get_title()
        new = LABEL_MAP.get(title, title)
        if new == title and "=" in title:
            lhs, _, rhs = title.partition("=")
            new = LABEL_MAP.get(rhs.strip(), rhs.strip())
        ax.set_title(new, fontsize=fontsize)



def _rotate_ticks(grid: sns.FacetGrid, rotation: int = 25, fontsize: Optional[float] = None) -> None:
    """Lay the x tick labels back at *rotation*, matching ``plot_benchmark``'s bar charts.

    Only the guarantee figure needs this; the breakdown's drawer sets its own rotation as
    it places the ticks.

    A two-line label stays upright, as in :func:`_draw_grouped_stacks`: rotated multi-line
    ticks overlap their neighbours and cost more height.
    """
    for ax in grid.axes.flat:
        multiline = any("\n" in t.get_text() for t in ax.get_xticklabels())
        for text in ax.get_xticklabels():
            if fontsize is not None:
                text.set_fontsize(fontsize)
            text.set_rotation(0 if multiline else rotation)
            text.set_ha("center" if multiline else "right")
            text.set_rotation_mode(None if multiline else "anchor")


def _size_ticks(grid: sns.FacetGrid, fontsize: float) -> None:
    """Set x tick type size, leaving rotation alone.

    The grouped drawer already chose the rotation; `_rotate_ticks` would overwrite it.
    """
    for ax in grid.axes.flat:
        for text in ax.get_xticklabels():
            text.set_fontsize(fontsize)


def _apply_label_sizes(grid: sns.FacetGrid, fonts: Dict[str, float]) -> None:
    """Axis labels, y ticks and any group-label text, at page-width type sizes."""
    for ax in grid.axes.flat:
        ax.xaxis.label.set_size(fonts["label"])
        ax.yaxis.label.set_size(fonts["label"])
        ax.tick_params(axis="y", labelsize=fonts["tick"])
        for text in ax.texts:
            text.set_fontsize(fonts["group"])


@lru_cache(maxsize=1024)
def text_width_em(text: str) -> float:
    """Width of *text* set at 1pt, in points - i.e. its width in em.

    Measured with a ``TextPath`` rather than estimated from a per-character average, which
    is inaccurate for proportional fonts. A multi-line label is its widest line. Cached
    because the same arm names are measured repeatedly.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    widest = 0.0
    for line in str(text).split("\n"):
        if not line.strip():
            continue
        path = TextPath((0, 0), line, size=1, prop=FontProperties(family=plt.rcParams["font.family"]))
        widest = max(widest, float(path.get_extents().width))
    return widest


def stride_labels(
    labels: Sequence[str],
    panel_width_in: float,
    fontsize: float,
    overflowing: Sequence[int] = (),
) -> List[str]:
    """*labels* with the ones that cannot fit blanked, keeping the first and the last.

    Complements ``_shrink_fonts``, which holds a legibility floor: when many short labels
    still do not fit at that floor, every *stride*-th label is kept and the rest blanked.
    Both ends are kept because they carry the range.

    *overflowing* names positions whose label may be wider than its slot
    (:func:`_tick_geometry`); they neither set the stride nor are blanked.
    """
    labels = list(labels)
    if len(labels) < 3 or panel_width_in <= 0:
        return labels
    spill = {i % len(labels) for i in overflowing}
    measured = [text for index, text in enumerate(labels) if index not in spill]
    widest = max((text_width_em(text) for text in measured), default=0.0)
    if widest <= 0:
        return labels
    needed_in = widest * _TICK_WIDTH_SAFETY * fontsize / 72.0
    slot_in = panel_width_in / len(labels)
    # Rounded: `_shrink_fonts` may size a label to fill its slot exactly, and float noise
    # (e.g. 1.0000000001) must not produce a stride of 2.
    stride = int(np.ceil(round(needed_in / slot_in, 6))) if slot_in > 0 else 1
    if stride <= 1:
        return labels
    keep = set(range(0, len(labels), stride)) | {0, len(labels) - 1} | spill
    return [text if index in keep else "" for index, text in enumerate(labels)]


def _thin_to_fit(
    positions: Sequence[Tuple[float, str]],
    axis,
    fontsize: float,
    pad_pt: float = 4.5,
) -> List[Tuple[float, str]]:
    """The subset of ``(data_x, text)`` whose labels do not collide, keeping the ends.

    The greedy walk's states cluster near zero storage (it prunes the biggest slot first),
    so labelling every one would overlap. Keeps both ends and walks inward, dropping any
    label that would touch the previous one.

    Widths are estimated (0.55 em per character, adequate for digits) since this runs
    before a draw.
    """
    if not positions:
        return []
    ordered = sorted(positions, key=lambda item: item[0])
    if len(ordered) < 3:
        return ordered
    width_pt = lambda text: max(len(text) * fontsize * 0.55, fontsize) + pad_pt
    to_display = lambda value: axis.transData.transform((value, 0))[0]
    dpi_scale = axis.figure.dpi / 72.0

    first, last = ordered[0], ordered[-1]
    kept = [first]
    for value, text in ordered[1:-1]:
        gap = abs(to_display(value) - to_display(kept[-1][0])) / dpi_scale
        if gap < (width_pt(text) + width_pt(kept[-1][1])) / 2:
            continue
        kept.append((value, text))
    # The right-hand end always wins a collision: it is the widest state of the walk.
    while len(kept) > 1:
        gap = abs(to_display(last[0]) - to_display(kept[-1][0])) / dpi_scale
        if gap >= (width_pt(last[1]) + width_pt(kept[-1][1])) / 2:
            break
        kept.pop()
    kept.append(last)
    return kept


def _operator_positions(
    frame: pd.DataFrame, dataset: str, x_column: str, count_column: str
) -> List[Tuple[float, str]]:
    """``(storage, "N ops")`` for one panel, one entry per state.

    Empty when the count does not vary along the axis (e.g. a sample-size axis), where the
    annotation would add nothing.
    """
    panel = frame[frame["dataset"].astype(str) == str(dataset)]
    if panel.empty or count_column not in panel.columns:
        return []
    pairs = panel[[x_column, count_column]].dropna().drop_duplicates()
    if pairs[count_column].nunique() < 2:
        return []
    return [
        (float(x), f"{int(n)}")
        for x, n in sorted(pairs.itertuples(index=False, name=None))
    ]


def annotate_operator_counts(
    grid: sns.FacetGrid,
    frame: pd.DataFrame,
    x_column: str,
    fonts: Dict[str, float],
    value_column: str,
    col_order: Sequence[str],
    mode: str = "top-axis",
    count_column: str = "n_operators",
) -> None:
    """Put the search-space size back on a figure whose x-axis is a footprint in GB.

    The greedy walk prunes operators, but the mapping from storage to operator count is
    per benchmark and non-linear, so it is shown explicitly. Three encodings:

    ``top-axis``  a second x-axis above the panel, ticked with operator counts at the
                  storage positions they sit at. Keeps the curve clean and states the
                  mapping explicitly, at the cost of a second axis to read.
    ``point``     the count above each marker of the topmost curve. Most direct, but it
                  competes with the curve for room.
    ``tick``      the count folded into the existing bottom ticks as a second line.
                  Cheapest in space, but ties the two quantities to the same ticks.

    All three thin their labels to fit; see :func:`_thin_to_fit`.
    """
    if mode not in {"top-axis", "point", "tick"} or count_column not in frame.columns:
        return
    size = max(3.4, fonts["tick"] * 0.92)
    # Match panels by position (FacetGrid follows `col_order`); titles are already relabelled.
    for ax, dataset in zip(grid.axes.flat, col_order):
        positions = _operator_positions(frame, dataset, x_column, count_column)
        if len(positions) < 2:
            continue
        kept = _thin_to_fit(positions, ax, size)
        if mode == "top-axis":
            twin = ax.twiny()
            linthresh = SYMLOG_LINTHRESH.get(ax)
            if linthresh is not None:
                twin.set_xscale("symlog", linthresh=linthresh, linscale=0.45)
            twin.set_xlim(ax.get_xlim())
            twin.set_xticks([value for value, _ in kept])
            twin.set_xticklabels([text for _, text in kept], fontsize=size)
            twin.tick_params(axis="x", length=2, pad=1.5)
            for side in ("top", "right", "bottom", "left"):
                twin.spines[side].set_visible(side == "top")
            twin.spines["top"].set_linewidth(0.6)
            # Named once, in the empty margin left of the first panel, on the tick row.
            if ax is grid.axes.flat[0]:
                twin.annotate(
                    "#ops", xy=(0, 1), xycoords="axes fraction",
                    xytext=(-3, 1), textcoords="offset points",
                    ha="right", va="bottom", fontsize=size, annotation_clip=False,
                )
            # Raise the facet title above the new top axis by one tick row, with the same
            # pad on every panel so titles share a baseline.
            ax.set_title(ax.get_title(), fontsize=fonts["title"], pad=size * 1.35 + 3)
        elif mode == "tick":
            # Whole GB above 10, one decimal below, to keep tick labels short.
            short = lambda v: f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"
            ax.set_xticks([value for value, _ in kept])
            ax.set_xticklabels(
                [f"{short(value)}\n{text} ops" for value, text in kept], fontsize=size
            )
        else:  # point
            panel = frame[frame["dataset"].astype(str) == str(dataset)]
            # Headroom so labels above the top curve stay inside the axes.
            low, high = ax.get_ylim()
            ax.set_ylim(low, high + (high - low) * 0.09)
            for value, text in kept:
                row = panel[panel[x_column] == value]
                if row.empty or value_column not in row:
                    continue
                # Above the highest curve at that x, so the label never lands on a line.
                y = float(row[value_column].max())
                ax.annotate(
                    text, (value, y), textcoords="offset points", xytext=(0, 3.5),
                    ha="center", va="bottom", fontsize=size, color=_mix(ARM_HUES[0], 0.25),
                )


def plain_log_axis(axis, ticks: Optional[Sequence[float]] = None) -> None:
    """Label a log axis only where the caller put ticks, in plain numbers.

    A log axis keeps its own minor locator and formatter, so matplotlib volunteers a
    "4 x 10^0" between hand-written labels - two notations for one quantity on one axis,
    in two type sizes - and, over a range narrower than a decade, crowds the whole minor
    ladder into the space of a few characters.

    *ticks* pins the major ones and labels them as integers where they are whole, which is
    what a footprint in GB or MB wants; omit it to keep matplotlib's majors and only
    silence the minors.
    """
    axis.set_minor_locator(mticker.NullLocator())
    axis.set_minor_formatter(mticker.NullFormatter())
    if ticks is None:
        return
    axis.set_major_locator(mticker.FixedLocator(list(ticks)))
    axis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, _pos: f"{value:g}" if value >= 1 else f"{value:.2g}"
        )
    )


def apply_x_scale(
    grid: sns.FacetGrid,
    frame: pd.DataFrame,
    x_column: str,
    col_order: Sequence[str],
    scale: str,
) -> None:
    """Spread out the crowded low end of a storage axis, without lying about the values.

    The walk prunes the biggest slot first, so many states land near zero on a linear
    axis. Three scales, in increasing order of distortion:

    ``linear``  the footprint as it is; crowded at the low end.
    ``sqrt``    a square-root axis. Still quantitative and passes through zero, so the
                gold-only state needs no special case.
    ``symlog``  linear inside a threshold and logarithmic outside. Spreads the low end
                hardest, at the cost of an axis whose reading changes partway along.
                The threshold is each panel's own smallest non-zero footprint, so no real
                state is compressed into the linear stretch.

    A plain ``log`` is not offered: the terminal state is at exactly 0 GB.
    """
    if x_column not in frame.columns or scale == "linear":
        return
    for ax, dataset in zip(grid.axes.flat, col_order):
        values = frame.loc[
            frame["dataset"].astype(str) == str(dataset), x_column
        ].dropna()
        positive = values[values > 0]
        if positive.empty:
            continue
        top = float(values.max())
        if scale == "sqrt":
            ax.set_xscale("function", functions=(np.sqrt, np.square))
            ax.set_xlim(0, top * 1.05)
            # Evenly spaced in the transformed space, then rounded; equal steps in GB would
            # crowd the right end.
            root = np.sqrt(top)
            raw = [(fraction * root) ** 2 for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)]
            ticks, seen = [], set()
            for value in raw:
                if value <= 0:
                    nice = 0.0
                else:
                    magnitude = 10.0 ** np.floor(np.log10(value))
                    nice = float(round(value / magnitude) * magnitude)
                if nice not in seen:
                    seen.add(nice)
                    ticks.append(nice)
            ax.set_xticks(ticks)
            continue
        linthresh = float(positive.min())
        ax.set_xscale("symlog", linthresh=linthresh, linscale=0.45)
        ax.set_xlim(0, top * 1.15)
        SYMLOG_LINTHRESH[ax] = linthresh
        # Explicit decade ticks plus zero. matplotlib's symlog locator fills the linear
        # stretch with minor ticks that overprint the "0" at this type size.
        decade, ticks = 10.0 ** np.ceil(np.log10(linthresh)), [0.0]
        while decade <= top:
            ticks.append(decade)
            decade *= 10.0
        ax.set_xticks(ticks)
        ax.xaxis.set_minor_locator(mticker.NullLocator())


def annotate_panel_minimum(
    grid: sns.FacetGrid,
    frame: pd.DataFrame,
    x_column: str,
    value_column: str,
    col_order: Sequence[str],
    group_col: str,
    fonts: Dict[str, float],
    unit: str,
    count_column: str = "n_operators",
) -> None:
    """Mark each panel's cheapest *state* once, with what it costs, buys and occupies.

    One mark per panel rather than one per curve: targets may bottom out at different
    states, so the minimum is taken over the runtime **summed across targets** and drawn
    as a vertical rule at that storage position. The label gives the operator count and
    the footprint.
    """
    if x_column not in frame.columns or value_column not in frame.columns:
        return
    size = max(3.8, fonts["tick"] * 1.02)
    ink = _mix(ARM_HUES[0], 0.45)
    for ax, dataset in zip(grid.axes.flat, col_order):
        panel = frame[frame["dataset"].astype(str) == str(dataset)].dropna(
            subset=[value_column, x_column]
        )
        if panel.empty:
            continue
        totals = panel.groupby(x_column)[value_column].sum()
        if totals.empty:
            continue
        x = float(totals.idxmin())
        at_x = panel[panel[x_column] == x]
        ops = at_x[count_column].dropna().max() if count_column in at_x else pd.NA
        # The runtime is on the y-axis already; the label adds the search-space size.
        label = "cheapest"
        detail = f"{int(ops)} ops · {x:,.0f} GB" if pd.notna(ops) else f"{x:,.0f} GB"
        # A wide translucent stripe, drawn below the curves so markers stay visible.
        ax.axvline(
            x, color=ink, linewidth=size * 0.95, alpha=0.20, zorder=1.2,
            solid_capstyle="butt",
        )
        # A hairline down the middle marks the exact position.
        ax.axvline(x, color=ink, linewidth=0.5, alpha=0.55, zorder=1.3)
        # Label at the top of the panel, away from the curves near the minimum.
        left, right = ax.get_xlim()
        position = (x - left) / (right - left) if right > left else 0.5
        align = "left" if position < 0.5 else "right"
        ax.annotate(
            f"{label}\n{detail}",
            xy=(x, 1.0), xycoords=("data", "axes fraction"),
            textcoords="offset points",
            xytext=(size * 0.6 if align == "left" else -size * 0.6, -size * 0.8),
            ha=align, va="top", fontsize=size, color=ink, linespacing=1.1, zorder=6,
            # A tinted box so the rule does not strike through the label.
            bbox=dict(
                boxstyle="round,pad=0.34",
                facecolor=_mix(ARM_HUES[0], -0.90),
                edgecolor=_mix(ARM_HUES[0], -0.50),
                linewidth=0.45,
            ),
        )
    # Headroom so the opaque label clears the topmost curve.
    for ax in grid.axes.flat:
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + (high - low) * 0.40)


def _facet_geometry(
    n_facets: int,
    width_in: float,
    panel_height_in: float = 1.9,
    col_wrap: Optional[int] = None,
    height_ratio: float = 1.30,
    height_in: Optional[float] = None,
) -> Tuple[float, float, float, Optional[int]]:
    """``(height, aspect, panel_width_in, col_wrap)`` for a grid totalling *width_in*.

    Seaborn sizes a ``FacetGrid`` bottom-up (panel height x aspect x panels); this inverts
    that so a fixed total width is divided among the panels and the figure fits the page.

    The panel height tracks the panel width (bounded below and above), then is scaled by
    ``HEIGHT_SCALE``.
    """
    n_facets = max(n_facets, 1)
    # One row by default.
    if col_wrap is None:
        col_wrap = n_facets
    columns = max(1, min(col_wrap, n_facets))
    panel_width = width_in / columns
    # *height_in* imposes a panel height, used by pooled single-panel figures to match the
    # height of the faceted figure they accompany.
    height = (
        height_in
        if height_in is not None
        else min(panel_height_in, max(1.05, panel_width * height_ratio)) * HEIGHT_SCALE
    )
    # A single row needs no col_wrap, and passing one makes seaborn drop `margin_titles`.
    return height, panel_width / height, panel_width, (columns if columns < n_facets else None)


def _tick_geometry(
    col_order: Sequence[str],
    order_for: Callable[[str], List[str]],
    ticks_for: Callable[[str], Dict[str, str]],
    overflowing: Sequence[str] = (),
) -> Tuple[float, bool]:
    """``(width of the widest label in em, drawn upright)`` over every panel of a grid.

    Measured over every panel, because the type size is shared by the whole figure.
    Multi-line labels are drawn upright by :func:`_draw_grouped_stacks` and measured by
    their longest line.

    *overflowing* names arms left out of the measurement: an arm appended to the end of
    the axis (a reference from another task) can spill into the gutter on its right.
    """
    widest = 0.0
    upright = False
    spill = {str(a) for a in overflowing}
    for dataset in col_order:
        ticks = ticks_for(dataset)
        for arm in order_for(dataset):
            label = str(ticks.get(str(arm), LABEL_MAP.get(str(arm), str(arm))))
            upright = upright or "\n" in label
            if str(arm) in spill:
                continue
            widest = max(widest, text_width_em(label))
    return widest, upright


def _cut_out_geometry(
    matched: Tuple[float, float, float, Optional[int]],
    page_width_in: float,
    height_ratio: float = 1.30,
) -> Tuple[float, float, float, Optional[int]]:
    """Geometry for a **one-panel** figure drawn beside a faceted one.

    The panel is sized to *page_width_in* - ``--overall-width``, the page this figure is
    asked to fill - by the same rule a one-panel row would get, and the type is then sized
    from that panel like any other figure's. ``--height-scale`` adjusts its height.
    """
    height, _, _, col_wrap = _facet_geometry(1, page_width_in, height_ratio=height_ratio)
    return height, page_width_in / height, page_width_in, col_wrap


def _shrink_fonts(
    x_room_in: float = 0.7,
    x_label_em: float = 0.0,
    x_labels_upright: bool = False,
) -> Dict[str, float]:
    """Type sizes for a figure that has to fit a page rather than a screen.

    ``apply_default_style`` sets a 16pt screen size, too large for page-width panels.

    The binding constraint is ``x_room_in`` - the width one x tick gets, i.e. the panel
    divided by its groups. The tick size is chosen within :data:`_TICK_BAND_PT`, and the
    other sizes (legend included) are derived from it.

    *x_label_em* is the width of the widest tick label, measured at 1pt by
    :func:`text_width_em`; the type is also sized so that label fits its slot. Zero
    ignores label width (for numeric ticks).
    """
    low, high = _TICK_BAND_PT
    tick = min(high, max(low, x_room_in * 13.0))
    if x_label_em > 0:
        if x_labels_upright:
            # Upright labels are centred on their tick and may use the whole slot, with
            # the same :data:`_TICK_WIDTH_SAFETY` margin `stride_labels` applies.
            room_in = x_room_in
            span_em = x_label_em * _TICK_WIDTH_SAFETY
        else:
            # A label rotated by TICK_ROTATION_DEG covers `cos(angle)` of its length
            # horizontally, and has `_LABEL_LEFT_ROOM` of half a slot to do it in.
            room_in = x_room_in * 0.5 * _LABEL_LEFT_ROOM
            span_em = x_label_em * np.cos(np.radians(TICK_ROTATION_DEG))
        fits = room_in * 72.0 / span_em
        if fits < _TICK_FLOOR_PT:
            # The label does not fit at any legible size: hold the floor and log it.
            logger.info(
                "A %.1fem x tick needs %.1fpt to fit %.2fin of panel; holding the "
                "%.1fpt floor, so it will reach into the panel beside it. Shorten the "
                "label if that matters.", x_label_em, fits, x_room_in, _TICK_FLOOR_PT,
            )
        tick = max(_TICK_FLOOR_PT, min(tick, fits))
    # The legend uses the tick's type size (capped); `_legend_columns` wraps it to fit.
    legend = min(7.5, tick)
    return {
        "tick": tick,
        "group": tick,
        "title": min(8.5, tick * 1.15),
        "label": tick * 0.95,
        "legend": legend,
    }




def plot_phase_breakdown_facets(
    totals: pd.DataFrame,
    *,
    arm_col: str = "arm",
    order_for: Callable[[str], List[str]],
    ticks_for: Callable[[str], Dict[str, str]],
    out_path: Path,
    panels: str = "all",
    ylabel: str = "Runtime [h]",
    xlabel: str = "",
    value_cols: Optional[Sequence[str]] = None,
    group_col: str = "guarantee_setting",
    width_in: float = A4_TEXT_WIDTH_IN,
    share_y: bool = False,
    pooling: str = "geomean",
    hue_override: HueSpec = None,
    overflowing_ticks: Sequence[str] = (),
    match_panels: Optional[int] = None,
    match_width: float = A4_TEXT_WIDTH_IN,
) -> Optional[Path]:
    """One panel per dataset; within a panel, one bar group per arm and one bar per target.

    Every guarantee target is in one figure, so an arm's behaviour across targets is read
    along a group (the layout ``plot_benchmark.plot_runtime_per_target`` uses).

    ``order_for``/``ticks_for`` are per dataset because, on the operator-count sweep,
    which states exist and their footprints are properties of the benchmark.
    """
    columns = list(value_cols) if value_cols else drop_all_zero_phases(totals)
    if not columns or totals[columns].to_numpy().sum() == 0:
        logger.warning("Nothing non-zero to draw for %s; skipping.", out_path.name)
        return None

    frame = totals if panels != "overall" else totals[totals["dataset"] == OVERALL]
    if frame.empty:
        logger.warning("No %s rows for %s; skipping.", OVERALL, out_path.name)
        return None

    group_order = sorted(frame[group_col].astype(str).unique())
    group_ticks = {g: guarantee_short_label(g) for g in group_order}
    # Legend labels spell out what the target number means.
    group_legend_labels = [f"P/R={group_ticks[g]}" for g in group_order]

    col_order = dataset_col_order(frame)
    matched = _facet_geometry(match_panels, match_width) if match_panels else None
    cut_out = bool(matched) and len(col_order) == 1
    # A pooled figure drawn beside a faceted one takes that figure's panel geometry and
    # type sizes; its page is then sized to the panel (:func:`_fit_panel`).
    height, aspect, panel_width, col_wrap = (
        _cut_out_geometry(matched, width_in) if cut_out
        else _facet_geometry(
            len(col_order), width_in, height_in=matched[0] if matched else None
        )
    )
    widest = max((len(order_for(d)) for d in col_order), default=1)
    label_em, label_upright = _tick_geometry(
        col_order, order_for, ticks_for, overflowing_ticks
    )
    fonts = _shrink_fonts(panel_width / max(widest, 1), label_em, label_upright)
    grid = sns.FacetGrid(
        frame,
        col="dataset",
        col_order=col_order,
        margin_titles=col_wrap is None,
        col_wrap=col_wrap,
        # Per-dataset x axes, since states and footprints differ per benchmark. Under
        # ``--facet-by`` the caller shares y so panels can be compared.
        sharey=share_y,
        sharex=False,
        height=height,
        aspect=aspect,
    )

    for dataset, ax in zip(col_order, grid.axes.flat):
        sub = frame[frame["dataset"] == dataset]
        order = [a for a in order_for(dataset) if a in set(sub[arm_col].astype(str))]
        if not order:
            continue
        _draw_grouped_stacks(
            ax,
            sub,
            outer_col=arm_col,
            outer_order=order,
            inner_col=group_col,
            inner_order=group_order,
            columns=columns,
            outer_ticks=ticks_for(dataset),
            inner_ticks=group_ticks,
            panel_width_in=panel_width,
            tick_fontsize=fonts["tick"],
            hue_override=hue_override,
            overflowing_ticks=overflowing_ticks,
        )
        if not share_y:
            ax.set_ylim(0, None)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    if share_y:
        # One limit for every panel from the tallest stack anywhere: on shared axes each
        # `set_ylim` overwrites the shared view, so per-panel limits would clip bars.
        tallest = float(np.nanmax(frame[columns].sum(axis=1).to_numpy(dtype=float)))
        for ax in grid.axes.flat:
            ax.set_ylim(0, tallest * 1.05 if tallest > 0 else None)

    _relabel_facet_titles(grid, fonts["title"])
    for ax in grid.axes.flat:
        # Name the pooling rule on the pooled panel's title.
        if ax.get_title() == OVERALL:
            note = POOLING_NOTES.get(pooling, pooling)
            ax.set_title(f"{OVERALL} ({note})", fontsize=fonts["title"])
    _size_ticks(grid, fonts["tick"])
    _apply_label_sizes(grid, fonts)
    # One y and one x label for the row, on the leftmost panel; the legends use the freed
    # band beside the x label.
    for index, ax in enumerate(grid.axes.flat):
        if index:
            ax.set_ylabel("")
            ax.set_xlabel("")
    _relayout(grid.figure)
    grid.figure.subplots_adjust(wspace=Y_GUTTER_BARS)
    if cut_out:
        # Match the axes width of a faceted panel (its slot less the gutter).
        width_in = _fit_panel(
            grid.figure, grid.axes.flat[0], panel_width / (1.0 + Y_GUTTER_BARS)
        )
    _phase_legend(
        grid.figure,
        columns,
        width_in,
        fontsize=fonts["legend"],
        inner_labels=group_legend_labels,
        hue_override=hue_override,
        beside=grid.axes.flat[0],
    )

    _save_to_width(grid.figure, out_path, width_in)
    plt.close(grid.figure)
    return out_path


def plot_metric(
    totals: pd.DataFrame,
    *,
    metric,
    arm_col: str = "arm",
    arm_order: Sequence[str],
    tick_text: Optional[Dict[str, str]] = None,
    out_path: Path,
    panels: str = "all",
    x_column: Optional[str] = None,
    xlabel: str = "",
    width_in: float = A4_TEXT_WIDTH_IN,
    group_col: str = "guarantee_setting",
    share_x: bool = True,
    share_y: bool = False,
    pooling: str = "geomean",
    hue_override: HueSpec = None,
    overflowing_ticks: Sequence[str] = (),
    match_panels: Optional[int] = None,
    match_width: float = A4_TEXT_WIDTH_IN,
) -> Optional[Path]:
    """One scalar metric per dataset panel: curves on a numeric axis, bars otherwise.

    Colour is :func:`arm_shade` throughout - hue for the arm, brightness for the guarantee
    target - matching the phase breakdown.

    On a numeric axis where several arms share an x position, the arm also takes a
    **dash pattern** so overlapping curves stay distinguishable (otherwise
    ``sns.lineplot`` would average them).
    """
    if totals.empty:
        logger.warning("Nothing to draw for %s; skipping.", out_path.name)
        return None
    frame = totals if panels != "overall" else totals[totals["dataset"] == OVERALL]
    if frame.empty:
        logger.warning("No %s rows for %s; skipping.", OVERALL, out_path.name)
        return None

    col_order = dataset_col_order(frame)
    groups = sorted(frame[group_col].astype(str).unique())
    n_x = frame[x_column].nunique() if x_column else len(arm_order)
    # Curves need vertical room, so these panels are taller than the breakdown's.
    matched = (
        _facet_geometry(match_panels, match_width, height_ratio=1.65)
        if match_panels
        else None
    )
    # One panel of the faceted figure, at its size and in its type - see
    # `plot_phase_breakdown_facets` and :func:`_fit_panel`.
    cut_out = bool(matched) and len(col_order) == 1
    height, aspect, panel_width, col_wrap = (
        _cut_out_geometry(matched, width_in, height_ratio=1.65) if cut_out
        else _facet_geometry(
            len(col_order), width_in, height_ratio=1.65,
            height_in=matched[0] if matched else None,
        )
    )
    # Numeric ticks are short and upright; only categorical arm names are measured.
    label_em, label_upright = (
        (0.0, False) if x_column
        else _tick_geometry(
            [""], lambda _: arm_order, lambda _: tick_text or {}, overflowing_ticks
        )
    )
    fonts = _shrink_fonts(panel_width / max(n_x, 1), label_em, label_upright)

    grid = sns.FacetGrid(
        frame,
        col="dataset",
        col_order=col_order,
        col_wrap=col_wrap,
        margin_titles=col_wrap is None,
        # No `hue` on the grid: FacetGrid's hue passes `label=` into the plot function,
        # which collides with seaborn's hue in `barplot`. The branches below set it.
        sharey=share_y,
        sharex=(not bool(x_column)) or share_x,
        height=height,
        aspect=aspect,
    )
    # Whether several arms land on the same x *within one panel*, which is what the dash
    # and per-arm hue separate. Where arm == x (e.g. sample size, storage footprint) there
    # is no second categorical axis, so only the target varies the brightness. Checked per
    # panel because each panel may own its x axis.
    arms_share_x = (
        bool(x_column)
        and x_column in frame.columns
        and bool(
            (frame.groupby(["dataset", x_column])[arm_col].nunique() > 1).any()
        )
    )
    # Whether the arm is a categorical axis of its own (x on bars, dash on curves). Where
    # arm == x, one curve spans the arms and cannot take an arm's colour.
    hue_by_arm = (not x_column) or arms_share_x
    # Colour is keyed by the (arm, target) pair, since it encodes both.
    target_index = {g: i for i, g in enumerate(groups)}

    def combo_color(arm: str, group: str) -> str:
        level, n_levels = target_index[group], len(groups)
        # A caller-supplied approach hue takes precedence, so e.g. every sample size of
        # one approach shares its hue.
        hue = resolve_hue(hue_override, arm)
        if hue is not None:
            return _mix(hue, shade_amounts(n_levels)[level])
        if not hue_by_arm:
            # One curve spans the arms, so nothing about it can be the arm's colour.
            return _mix(ARM_HUES[0], shade_amounts(n_levels)[level])
        return arm_shade(arm, arm_order, level, n_levels)

    if x_column:
        style_kwargs: Dict[str, Any] = {}
        hue_col, hue_levels = group_col, list(groups)
        if arms_share_x:
            style_kwargs = dict(
                style=arm_col,
                style_order=list(arm_order),
                dashes={
                    arm: ARM_DASHES[i % len(ARM_DASHES)]
                    for i, arm in enumerate(arm_order)
                },
            )
            # A synthetic (arm, target) hue level, so each line gets the arm's hue at the
            # target's brightness.
            hue_col = "_arm_target"
            frame = frame.assign(
                _arm_target=frame[arm_col].astype(str) + "|" + frame[group_col].astype(str)
            )
            grid.data = frame
            hue_levels = [f"{a}|{g}" for a in arm_order for g in groups]
            hue_levels = [h for h in hue_levels if h in set(frame["_arm_target"])]
        palette = {
            level: combo_color(*level.split("|", 1)) if "|" in level else combo_color("", level)
            for level in hue_levels
        }
        grid.map_dataframe(
            sns.lineplot, x=x_column, y=metric.column, marker="o",
            markersize=max(2.2, fonts["tick"] * 0.45),
            linewidth=max(0.7, fonts["tick"] / 6.0),
            # No confidence band around a mean of several arms per cell.
            errorbar=None,
            hue=hue_col, hue_order=hue_levels, palette=palette,
            **style_kwargs,
        )
    else:
        grid.map_dataframe(
            sns.barplot, x=arm_col, y=metric.column, order=arm_order,
            errorbar=None, linewidth=0,
            hue=group_col, hue_order=groups,
            palette={g: combo_color(arm_order[0], g) for g in groups},
        )
        # Seaborn colours bars per hue level (the target); recolour each patch so the arm
        # on x also carries its hue. One container per hue level, patches in `order`.
        for ax in grid.axes.flat:
            for j, container in enumerate(ax.containers[: len(groups)]):
                for i, patch in enumerate(container.patches):
                    if i < len(arm_order):
                        patch.set_facecolor(combo_color(arm_order[i], groups[j]))
        ticks = dict(tick_text or {})
        spill = {str(a) for a in overflowing_ticks}
        labels = stride_labels(
            [ticks.get(a, LABEL_MAP.get(a, a)) for a in arm_order],
            panel_width,
            fonts["tick"],
            overflowing=[i for i, a in enumerate(arm_order) if str(a) in spill],
        )
        # Rotation is applied by `_rotate_ticks` below.
        for ax in grid.axes.flat:
            ax.set_xticks(range(len(arm_order)))
            ax.set_xticklabels(labels)

    for ax in grid.axes.flat:
        ax.set_ylim(0, None)
    grid.set_axis_labels(xlabel, metric.label)
    _relabel_facet_titles(grid, fonts["title"])
    for ax in grid.axes.flat:
        if ax.get_title() == OVERALL:
            # The metric decides whether `--pooling` applies (accuracies are always means).
            rule = metric.pool_rule(pooling)
            ax.set_title(
                f"{OVERALL} ({POOLING_NOTES.get(rule, rule)})",
                fontsize=fonts["title"],
            )
    if not x_column:
        _rotate_ticks(grid, fontsize=fonts["tick"])
    else:
        # Integer ticks for count-valued axes; continuous axes keep the default locator.
        if pd.api.types.is_integer_dtype(frame[x_column].dropna().infer_objects()):
            for ax in grid.axes.flat:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _size_ticks(grid, fonts["tick"])
    _apply_label_sizes(grid, fonts)
    # One x and one y label for the whole row, on the leftmost panel.
    for index, ax in enumerate(grid.axes.flat):
        if index:
            ax.set_ylabel("")
            ax.set_xlabel("")
    # On a storage axis, also show the operator count where the frame carries it.
    if x_column:
        # Scale first: the operator twin axis copies these limits and thins ticks in
        # display space.
        apply_x_scale(grid, frame, x_column, col_order, X_SCALE)
        annotate_operator_counts(
            grid, frame, x_column, fonts, value_column=metric.column,
            col_order=col_order, mode=OPERATOR_AXIS_MODE,
        )
        if ANNOTATE_MINIMA:
            # Take the unit from the metric's axis label.
            match = re.search(r"\[([^\]]+)\]", metric.label)
            annotate_panel_minimum(
                grid, frame, x_column, metric.column, col_order, group_col, fonts,
                unit=match.group(1) if match else "",
            )
    _relayout(grid.figure)
    grid.figure.subplots_adjust(wspace=Y_GUTTER_CURVES)
    if cut_out:
        width_in = _fit_panel(
            grid.figure, grid.axes.flat[0], panel_width / (1.0 + Y_GUTTER_CURVES)
        )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Target swatches are grey when the hue encodes the arm; otherwise (arm == x) they use
    # the curves' single hue.
    swatch = grey_shades(len(groups)) if hue_by_arm else [
        combo_color("", g) for g in groups
    ]
    handles = [
        (Line2D([], [], color=swatch[i], marker="o", markersize=2.6, linewidth=1.0)
         if x_column else Patch(facecolor=swatch[i]))
        for i, g in enumerate(groups)
    ]
    band = _legend_band(grid.figure, grid.axes.flat[0], fonts["legend"])
    labels = [f"P/R={guarantee_short_label(g)}" for g in groups]
    if arms_share_x:
        # A second group naming the arms, each in its hue (middle target's brightness)
        # and dash pattern.
        ticks = dict(tick_text or {})
        handles += [
            Line2D(
                [], [],
                color=arm_shade(a, arm_order, len(groups) // 2, len(groups)),
                linewidth=1.0,
                dashes=ARM_DASHES[i % len(ARM_DASHES)],
            )
            for i, a in enumerate(arm_order)
        ]
        labels += [ticks.get(a, LABEL_MAP.get(a, a)) for a in arm_order]
    grid.figure.legend(
        handles,
        labels,
        loc="upper center",
        # Beside the x label on a row of panels, below it on a single panel
        # (`_legend_band`).
        bbox_to_anchor=(band.centre, band.top),
        ncol=_legend_columns(width_in, labels, fonts["legend"]),
        frameon=False,
        fontsize=fonts["legend"],
        # Use the line length whenever any entry is a line, so handles stay uniform.
        handlelength=(
            LINE_HANDLE_LENGTH if x_column or arms_share_x else PATCH_HANDLE_LENGTH
        ),
        columnspacing=1.0,
    )
    for ax in grid.axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    _save_to_width(grid.figure, out_path, width_in)
    plt.close(grid.figure)
    return out_path


def plot_phase_breakdown_grouped(
    totals: pd.DataFrame,
    *,
    arm_col: str = "arm",
    arm_order: Sequence[str],
    group_col: str = "guarantee_setting",
    out_path: Path,
    ylabel: str = "Runtime [h]",
    value_cols: Optional[Sequence[str]] = None,
    tick_text: Optional[Dict[str, str]] = None,
    width_in: float = A4_TEXT_WIDTH_IN,
    pooling: Optional[str] = None,
) -> Optional[Path]:
    """One axes: x grouped by *group_col*, one stacked bar per arm within each group.

    The ablation's layout: unlike :func:`plot_phase_breakdown_facets`, arms are nested
    *within* a target. Flip ``outer_col``/``inner_col`` below to match the other figures.

    Drawn from the pooled rows alone, so *pooling* is named on the y label.
    """
    columns = list(value_cols) if value_cols else drop_all_zero_phases(totals)
    if not columns or totals[columns].to_numpy().sum() == 0:
        logger.warning("Nothing non-zero to draw for %s; skipping.", out_path.name)
        return None

    groups = sorted(totals[group_col].astype(str).unique())
    order = [a for a in arm_order if a in set(totals[arm_col].astype(str))]
    if not order:
        logger.warning("No arms to draw for %s; skipping.", out_path.name)
        return None

    fonts = _shrink_fonts(width_in / max(len(groups) or 1, 1))
    # A single axes spanning the page width; type is sized from the per-group room.
    fig, ax = plt.subplots(figsize=(width_in, 2.2))
    _draw_grouped_stacks(
        ax,
        totals,
        outer_col=group_col,
        outer_order=groups,
        inner_col=arm_col,
        inner_order=order,
        columns=columns,
        outer_ticks={g: LABEL_MAP.get(g, g) for g in groups},
        inner_ticks=dict(tick_text or {}),
        # Arms within a target: the hue follows the inner value, brightness the x group.
        arm_axis="inner",
    )
    for text in ax.get_xticklabels():
        text.set_fontsize(fonts["tick"])
        if text.get_rotation() == 0:
            text.set_rotation(30)
            text.set_ha("right")
            text.set_rotation_mode("anchor")
    for text in ax.texts:
        text.set_fontsize(fonts["group"])
    ax.set_ylim(0, None)
    if pooling:
        ylabel = f"{ylabel}, {POOLING_NOTES.get(pooling, pooling)}"
    ax.set_ylabel(ylabel, fontsize=fonts["label"])
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=fonts["tick"])
    _phase_legend(
        fig,
        columns,
        width_in,
        fontsize=fonts["legend"],
        inner_labels=[dict(tick_text or {}).get(a, a) for a in order],
        # These entries name the arms and are drawn in their hues.
        inner_is_arm=True,
        inner_keys=list(order),
    )

    _save_to_width(fig, out_path, width_in)
    plt.close(fig)
    return out_path
