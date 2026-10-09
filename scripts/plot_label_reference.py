"""Plot the label_reference experiment: ground truth vs the best model as the optimizer's reference.

Reads the CSVs ``reasondb/coordinator/producers/label_reference.py`` merges, one per
label set, each carrying both arms in an ``optimized_against`` column.

The figures answer one question in the order the argument is made:

1. **The reference cross, per query** (``*_sections_precision.pdf`` /
   ``*_sections_recall.pdf``) - the headline. One panel per (optimized against, evaluated
   against) pair, then the human-tuned arm again keeping only the runs it *claimed*. One
   point per query, coloured by dataset, and the **marker is the optimizer's own
   ``guarantee_met``**: a circle below its rule is a query it believed it had satisfied
   and had not, a cross below the rule is a miss it reported. Height is measured, shape is
   claimed.
2. **The reference cross, as a distribution** (``*_sections_box.pdf``,
   ``*_sections_box_low_targets.pdf``) - four of those six panels as boxes, both metrics
   on one axes, no individual queries at all. This is the figure to read for "where does
   each arm sit"; figure 1 is the one for "which query". Two target variants, because the
   0.9 target is where both arms collapse and swamps the comparison.
3. **Per query, per target** (``*_precision.pdf`` / ``*_recall.pdf``) - one panel per
   guarantee target, one bar per query, the two arms as hue. Readable only at small
   query counts.
4. **Distribution within one label set** (``*_strip.pdf``, ``*_strip_by_dataset.pdf``) -
   the per-dataset breakdown.
5. **Violation rate** (``*_violations.pdf`` and a printed table) - the same thing counted,
   per (arm, target). The number the paper sentence cites. The table's
   ``claimed_met_rate`` column is the cross-shaped points above, counted.

Two columns say different things and both are needed. ``precision``/``recall`` are what
``evaluate()`` measured against the labels named in the filename. ``guarantee_met`` and
``achieved_*_lower`` are what the optimizer *believed* while tuning, against whichever
reference its arm used. The model arm reporting ``guarantee_met=True`` on a query whose
measured gold precision is below target is not an inconsistency - it is the finding.

    python scripts/plot_label_reference.py --output-dirs benchmark_results/ref01/merged

``--labels gold`` is the measurement; ``--labels silver`` is the control that shows the
model-optimized arm looking fine against its own reference. Default plots both.

Note on ``ecommerce_curated``: its labels are derived from the SemBench product catalog
rather than annotated per predicate (``LABEL_PROVENANCE`` in ``benchmarks/curated.py``).
This experiment deliberately treats them as human labels - they are non-model ground
truth, which is what the axis turns on - and the distinction belongs in the paper text
rather than in a facet here.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from reasondb.coordinator.producers.label_reference import (
    ARM_COLUMN,
    DEFAULT_BENCHMARKS,
    PRODUCER_NAME,
)
from reasondb.evaluation.plotting import apply_default_style, fix_labels, relabel_legend
from reasondb.evaluation.sweep_figures import (
    A4_TEXT_WIDTH_IN,
    TARGET_TYPE_PALETTE,
    finish_page_figure,
    page_geometry,
)

#: One column of a two-column page, for the two figures that are a single axes. Both are
#: headline figures rather than supplements, and a single axes stretched across a full
#: text width is mostly whitespace.
COLUMN_WIDTH_IN = A4_TEXT_WIDTH_IN / 2

#: Smallest type the axis labels and the legend entries take, whatever the ticks got.
#:
#: :func:`page_geometry` pins both to the tick size, which at column width is small. The
#: labels are short and are the first thing a reader reads, so they get a floor of their
#: own; it sits below what the page-width variant derives, so only the column-width
#: figure is affected.
AXIS_TYPE_FLOOR_PT = 6.6

#: Panel height per unit of panel *width* for the box figures.
#:
#: High enough that **both** the column-width and page-width variants reach
#: ``_facet_geometry``'s 1.9in panel cap, so the two read as one figure at two widths and
#: the narrow panels keep enough axes height (below titles, ticks and legend) for a 0.5 tick.
BOX_HEIGHT_RATIO = 2.4

logger = logging.getLogger(__name__)

#: Arm order and colours. The model arm first: it is the status quo the human arm is
#: being compared against, so it reads as the baseline in every legend.
ARM_ORDER = ["model", "human"]
ARM_PALETTE = {"model": "#BB5566", "human": "#004488"}
ARM_LABELS = {
    "model": "Optimized vs. best model",
    "human": "Optimized vs. human labels",
    ARM_COLUMN: "Optimizer reference",
    "target": "Target",
}

METRICS = {
    "precision": "precision_guarantee",
    "recall": "recall_guarantee",
}

#: Marker, size scale and legend entry per value of the optimizer's own ``guarantee_met``.
#:
#: The verdict is what the optimizer *believed* while tuning, against its arm's reference;
#: the point's height is what ``evaluate()`` measured against the label set in the
#: filename. Reading the two together is the whole experiment: a filled circle sitting
#: below its target rule is a query the optimizer thought it had satisfied and had not,
#: and a cross below the rule is a miss it reported honestly.
#:
#: Shape rather than colour, because colour is already the arm and this has to be legible
#: *within* each arm's cloud - the comparison is how the met points sit against the
#: unreachable ones for the same arm, which a second colour axis would turn into four
#: clouds nobody can pair up.
CLAIM_MARKERS = {
    "met": ("o", 1.0, "Optimizer: target met"),
    "missed": ("X", 1.35, "Optimizer: unreachable"),
    "unknown": ("s", 0.85, "Optimizer: not recorded"),
}
#: Draw order, back to front: the unreachable points are the rare class and the one being
#: looked for, so they go on top of the cloud rather than under it.
CLAIM_ORDER = ("unknown", "met", "missed")

#: Facet titles for the curated benchmarks. ``fix_labels`` splits "<lhs> = <rhs>" and maps
#: each half, and ``LABEL_MAP`` only knows the *random* datasets - so without these every
#: panel is titled "dataset = artwork_curated", which is both the widest thing in the
#: figure and mostly punctuation.
DATASET_LABELS = {
    "dataset": "Dataset",
    "artwork_curated": "Artwork",
    "email_curated": "Enron Email",
    "rotowire_curated": "Rotowire",
    "ecommerce_curated": "Ecommerce",
    # Its 10 000 reviews are an order of magnitude past the others, but the tick has no
    # room to say so and the caption does. Named like its sibling rather than
    # "movie_huge_curated", which is a filename rather than a label.
    "movie_huge_curated": "Movie",
}


def load_df(paths: List[Path], label_set: str) -> pd.DataFrame:
    """Concatenate the merged CSVs and add the columns the plots facet on."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        # <output-dir>/<benchmark>/<split>/<file>, mirroring how every plot_*.py locates
        # the benchmark a merged CSV came from.
        frame["dataset"] = path.parent.parent.name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["label_set"] = label_set

    missing = {"precision", "recall", ARM_COLUMN} - set(df.columns)
    assert not missing, (
        f"{sorted(missing)} absent from {[p.name for p in paths]}. These CSVs come from "
        f"the {PRODUCER_NAME} producer's merge; a run_benchmark merge writes "
        "gold_metrics.csv/silver_metrics.csv with no arm column and cannot be plotted here."
    )

    # One tick per query, short enough to read: queries are full sentences. Numbered
    # within a dataset so the x axis stays legible and the mapping is printed alongside.
    df = df.sort_values(["dataset", "query"])
    df["query_label"] = (
        df.groupby("dataset")["query"].transform(
            lambda s: pd.Series(pd.factorize(s)[0] + 1, index=s.index).astype(str)
        )
    )
    df["query_label"] = df["dataset"].str.replace("_curated", "", regex=False) + " Q" + df["query_label"]
    # No "=" in the value: `fix_labels` relabels a facet title of the form
    # "<lhs> = <rhs>" component-wise by splitting on the first "=", so a target spelled
    # "p=0.5, r=0.5" comes out as the title "target = p" with the rest discarded.
    df["target"] = (
        "P " + df["precision_guarantee"].round(2).astype(str)
        + " / R " + df["recall_guarantee"].round(2).astype(str)
    )
    return df


def plot_per_query(df: pd.DataFrame, metric: str, label_set: str, out_dir: Path) -> Path:
    """One panel per target, one bar per query, the arms side by side, a rule at the target.

    The rule is per panel rather than a single global line: each panel is a different
    target, and the whole point is whether that panel's bars clear that panel's target.
    """
    target_column = METRICS[metric]
    order = sorted(df["target"].unique())
    geometry = page_geometry(len(order), A4_TEXT_WIDTH_IN, df["query_label"].nunique())
    fonts = geometry["fonts"]
    grid = sns.catplot(
        data=df,
        x="query_label",
        y=metric,
        hue=ARM_COLUMN,
        hue_order=ARM_ORDER,
        palette=ARM_PALETTE,
        col="target",
        col_order=order,
        kind="bar",
        errorbar=None,
        height=geometry["height"],
        aspect=geometry["aspect"],
        col_wrap=geometry["col_wrap"],
        legend_out=True,
    )
    for ax, target in zip(grid.axes.flat, order):
        level = df.loc[df["target"] == target, target_column].iloc[0]
        ax.axhline(level, color="black", linestyle="--", linewidth=0.6, zorder=5)
        ax.set_ylim(0, 1.05)
    grid.set_axis_labels("", metric.capitalize())
    fix_labels(grid, {**ARM_LABELS, **DATASET_LABELS})

    handles = [Patch(facecolor=ARM_PALETTE[a], label=ARM_LABELS[a]) for a in ARM_ORDER]
    return finish_page_figure(
        grid,
        fonts=fonts,
        width_in=A4_TEXT_WIDTH_IN,
        out_path=out_dir / f"{PRODUCER_NAME}_{label_set}_{metric}.pdf",
        handles=handles,
        labels=[ARM_LABELS[a] for a in ARM_ORDER],
        rotation=90,
    )


#: Which label set is whose verdict. The filenames say gold/silver; the experiment is
#: about human against model, and the section titles have no room to explain the mapping.
LABEL_SET_REFERENCE = {"gold": "human", "silver": "model"}

#: The sections of the main figure, in reading order: ``(optimized_against, label_set,
#: claimed_only)``.
#:
#: The first four are the whole 2x2 - each arm scored against each reference - which is
#: what makes the diagonal readable. An arm scored against its *own* reference is the
#: control (the model arm against model labels, the human arm against human labels); the
#: off-diagonal is the measurement. The model arm against human labels is the finding the
#: experiment exists for.
#:
#: The last two repeat the human-optimized arm keeping only the queries the optimizer
#: *claimed*. Optimizing against human labels and then reporting every query mixes two
#: populations: the queries the optimizer believed it had solved, and the ones it reported
#: as unreachable and ran anyway. A guarantee is a claim about the first, so the second
#: pair is the arm read on its own terms. Both are shown because the filter is not free -
#: it is exactly the queries a deployment would have had to drop.
SECTION_SPECS = [
    ("model", "silver", False),
    ("model", "gold", False),
    ("human", "silver", False),
    ("human", "gold", False),
    ("human", "silver", True),
    ("human", "gold", True),
]


def section_label(arm: str, label_set: str, claimed_only: bool) -> str:
    """Two lines: what the optimizer tuned against, what the score was measured against."""
    evaluated = LABEL_SET_REFERENCE[label_set]
    suffix = "\n(optimizer claims met)" if claimed_only else ""
    return f"Opt: {arm}\nEval: {evaluated}{suffix}"


#: One colour per curated dataset. Colour is free in the section figure - the arm and the
#: reference are both on the facet - so it carries the dataset instead, which is what turns
#: a low point into an identifiable query rather than an anonymous one.
DATASET_PALETTE = {
    "artwork_curated": "#004488",
    "ecommerce_curated": "#DDAA33",
    "email_curated": "#BB5566",
    "movie_huge_curated": "#228833",
    "rotowire_curated": "#AA3377",
}


def build_sections(df: pd.DataFrame) -> pd.DataFrame:
    """The long frame the section figures draw: one row per (measurement, section).

    Rows are *duplicated* into the claimed-only sections rather than moved, because those
    sections are a filtered view of the same measurements and both views are on the
    figure. A section whose filter empties it is dropped, so a run in which the optimizer
    claimed nothing does not leave a blank panel with a rule floating in it.
    """
    frames = []
    for arm, label_set, claimed_only in SECTION_SPECS:
        part = df[(df[ARM_COLUMN] == arm) & (df["label_set"] == label_set)]
        if claimed_only:
            part = part[part.get("guarantee_met", pd.Series(dtype="object")).eq(True)]
        if part.empty:
            logger.info(
                "section %r is empty; dropping it",
                section_label(arm, label_set, claimed_only).replace("\n", " "),
            )
            continue
        part = part.copy()
        part["section"] = section_label(arm, label_set, claimed_only)
        part["claimed_only"] = claimed_only
        frames.append(part)
    assert frames, (
        "No section survived. Both label sets have to be loaded for this figure - pass "
        "--labels gold silver (the default) - and the CSVs must carry both arms."
    )
    return pd.concat(frames, ignore_index=True)


def claim_layers(frame: pd.DataFrame) -> List[tuple]:
    """Split a frame by the optimizer's own verdict, in draw order, dropping empty layers.

    ``guarantee_met`` may be absent from a CSV, or all-null when the costs behind it were
    replayed from a results cache without ``CostSummary.guarantee`` (see ``executor.py``'s
    ``from_json``) - so "not recorded" is a real third case, and it must not be drawn as if
    the optimizer had said something.

    ``.eq(True)`` rather than a truth test: read back from CSV the column is object dtype
    holding Python bools and ``NaN``, and ``NaN`` is neither true nor false here.
    """
    if "guarantee_met" not in frame.columns:
        return [("unknown", frame)]
    claim = frame["guarantee_met"]
    by_key = {
        "met": frame[claim.eq(True)],
        "missed": frame[claim.eq(False)],
        "unknown": frame[~claim.eq(True) & ~claim.eq(False)],
    }
    return [(key, by_key[key]) for key in CLAIM_ORDER if not by_key[key].empty]


def strip_handles(df: pd.DataFrame) -> Tuple[List[Line2D], List[str]]:
    """Legend handles for both encodings: arm on the colour, optimizer verdict on the shape.

    The verdict entries are drawn in a neutral grey, so the legend cannot be read as a
    third arm - the marker is the only thing those entries are saying.

    Present verdicts only, but *always* when the column is usable, even if a single one
    survives: with one layer the marker carries no contrast, and without an entry naming
    it a figure where every query was reported unreachable looks exactly like one where
    every query was met.
    """
    handles = [
        Line2D(
            [], [], marker="o", linestyle="none", markersize=5,
            color=ARM_PALETTE[arm], markeredgecolor="white", markeredgewidth=0.3,
        )
        for arm in ARM_ORDER
    ]
    labels = [ARM_LABELS[arm] for arm in ARM_ORDER]
    keys = [key for key, _ in claim_layers(df)]
    if keys != ["unknown"]:
        for key in keys:
            marker, scale, label = CLAIM_MARKERS[key]
            handles.append(
                Line2D(
                    [], [], marker=marker, linestyle="none", markersize=5 * scale,
                    color="0.35",
                )
            )
            labels.append(label)
    return handles, labels


def plot_metric_strip(
    df: pd.DataFrame, metric: str, label_set: str, out_dir: Path, by_dataset: bool = False
) -> Path:
    """One point per query, the two arms' clouds side by side within each target.

    The per-query bars above are unreadable past a couple of dozen queries, and the
    curated benchmarks are deliberately small - so this is the same data as a
    distribution: x segmented by target, ``dodge`` splitting each segment into the two
    arms, and a rule at that segment's own target.

    **The marker is the optimizer's own verdict** (``CLAIM_MARKERS``): one stripplot layer
    per value of ``guarantee_met``, over the same categorical positions. That is what puts
    the two halves of the experiment in one figure - the height is the measured quality,
    the shape is what the optimizer claimed while tuning, and the rule is the target both
    are read against.

    Layering rather than one call is forced: a marker is per-artist in matplotlib, and
    seaborn draws one artist per hue level. Every layer is therefore given the *same*
    ``order`` and ``hue_order`` explicitly, which is what keeps the dodged positions
    aligned when a layer holds only one arm - without it seaborn dodges by the levels
    present in that layer's frame and the crosses land off-centre.

    The rule is drawn with :meth:`~matplotlib.axes.Axes.hlines` over the segment's span
    rather than ``axhline``. Categorical x positions are ``0..n-1``, and an ``axhline``
    spans the whole axes - so the 0.5 rule would be drawn straight through the 0.9
    segment. ``plot_per_query`` gets away with ``axhline`` only because each target has a
    facet to itself, which is exactly what this figure trades away.
    """
    target_column = METRICS[metric]
    order = sorted(df["target"].unique())

    def draw_rules(ax, frame: pd.DataFrame) -> None:
        for i, target in enumerate(order):
            levels = frame.loc[frame["target"] == target, target_column]
            if levels.empty:
                continue
            ax.hlines(
                levels.iloc[0], i - 0.42, i + 0.42,
                color="black", linestyle="--", linewidth=1.2, zorder=5,
            )

    strip_kwargs = dict(
        x="target", y=metric, order=order,
        hue=ARM_COLUMN, hue_order=ARM_ORDER, palette=ARM_PALETTE,
        dodge=True, jitter=0.18, alpha=0.8, linewidth=0.3, edgecolor="white",
        # Every layer builds its own legend entries for the arms, so all of them are
        # suppressed and one legend is assembled from `strip_handles` instead.
        legend=False,
    )

    def draw_layers(ax, frame: pd.DataFrame, size: float) -> None:
        for key, layer in claim_layers(frame):
            marker, scale, _ = CLAIM_MARKERS[key]
            sns.stripplot(
                data=layer, ax=ax,
                **{**strip_kwargs, "marker": marker, "size": size * scale},
            )

    handles, labels = strip_handles(df)

    if by_dataset:
        datasets = sorted(df["dataset"].unique())
        geometry = page_geometry(len(datasets), A4_TEXT_WIDTH_IN, len(order))
        # A plain FacetGrid rather than `catplot`: catplot owns the drawing call, and this
        # figure needs several of them per panel. `set_titles` restores the "<lhs> = <rhs>"
        # titles catplot would have written, which is the form `fix_labels` relabels.
        grid = sns.FacetGrid(
            data=df, col="dataset", col_order=datasets,
            height=geometry["height"], aspect=geometry["aspect"],
            col_wrap=geometry["col_wrap"], sharey=True,
        )
        grid.set_titles(col_template="dataset = {col_name}")
        for ax, dataset in zip(grid.axes.flat, datasets):
            draw_layers(ax, df[df["dataset"] == dataset], size=4.5)
            draw_rules(ax, df[df["dataset"] == dataset])
            ax.set_ylim(0, 1.05)
        grid.set_axis_labels("Guarantee target", metric.capitalize())
        fix_labels(grid, {**ARM_LABELS, **DATASET_LABELS})
        return finish_page_figure(
            grid,
            fonts=geometry["fonts"],
            width_in=A4_TEXT_WIDTH_IN,
            out_path=out_dir / f"{PRODUCER_NAME}_{label_set}_{metric}_strip_by_dataset.pdf",
            handles=handles,
            labels=labels,
        )
    else:
        fonts = page_geometry(1, COLUMN_WIDTH_IN, len(order))["fonts"]
        figure, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.68))
        draw_layers(ax, df, size=3.0)
        draw_rules(ax, df)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Guarantee target", fontsize=fonts["label"])
        ax.set_ylabel(metric.capitalize(), fontsize=fonts["label"])
        ax.tick_params(labelsize=fonts["tick"])
        # Two columns rather than one: the verdict entries double the legend, and four
        # stacked rows under a half-column axes is taller than the axes.
        # No legend title: the entries already read "Optimized vs. ..." and "Optimizer:
        # ...", so a column name only adds a line that lands on the x-axis label.
        #
        # The offset clears the tick labels *and* the axis label below them - anchored to
        # the axes, which does not know about either, so anything smaller lands the legend
        # on top of "Guarantee target".
        figure.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.26),
            bbox_transform=ax.transAxes, ncol=2, frameon=False,
            fontsize=fonts["legend"], handlelength=1.2, borderpad=0.1,
            labelspacing=0.3, columnspacing=1.0,
        )
        suffix = "_strip"

    out_path = out_dir / f"{PRODUCER_NAME}_{label_set}_{metric}{suffix}.pdf"
    figure.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(figure)
    return out_path


def section_handles(df: pd.DataFrame) -> Tuple[List[Line2D], List[str]]:
    """Legend for the section figures: dataset on the colour, optimizer verdict on the shape.

    Same split of duties as :func:`strip_handles`, one axis over: there the colour was the
    arm, here the arm is the facet and the colour is free for the dataset. The verdict
    entries stay neutral grey so the legend cannot be read as a sixth dataset.
    """
    datasets = [d for d in DATASET_PALETTE if d in set(df["dataset"])]
    handles = [
        Line2D(
            [], [], marker="o", linestyle="none", markersize=5,
            color=DATASET_PALETTE[dataset], markeredgecolor="white", markeredgewidth=0.3,
        )
        for dataset in datasets
    ]
    labels = [DATASET_LABELS.get(d, d) for d in datasets]
    keys = [key for key, _ in claim_layers(df)]
    if keys != ["unknown"]:
        for key in keys:
            marker, scale, label = CLAIM_MARKERS[key]
            handles.append(
                Line2D([], [], marker=marker, linestyle="none", markersize=5 * scale, color="0.35")
            )
            labels.append(label)
    return handles, labels


def plot_sections(df: pd.DataFrame, metric: str, out_dir: Path) -> Path:
    """The main figure: every (optimized against, evaluated against) pair as its own panel.

    One panel per :data:`SECTION_SPECS` entry, x segmented by guarantee target, a rule at
    that segment's own target. What sits below the rule is a guarantee that did not hold
    *as measured against that panel's reference*, which is the comparison the whole
    experiment is: the same optimizer runs move down the page as the reference changes
    from the model's own verdicts to the human labels.

    Two encodings ride along, and neither is decoration:

    * **Shape is the optimizer's own ``guarantee_met``** (:data:`CLAIM_MARKERS`), so a
      cross below a rule is a miss the optimizer reported and a circle below a rule is one
      it did not. That distinction is the difference between a system that is wrong and a
      system that does not know it is wrong.
    * **Colour is the dataset**, which the facet no longer carries. A point below the rule
      is then traceable to the benchmark it came from without a second figure.

    The distribution view is :func:`plot_sections_box`, which drops the points entirely
    and puts both metrics on one axes; this figure is the one to read when the question is
    about an individual query.
    """
    target_column = METRICS[metric]
    order = sorted(df["target"].unique())
    sections = [
        section_label(*spec) for spec in SECTION_SPECS
        if section_label(*spec) in set(df["section"])
    ]
    geometry = page_geometry(len(sections), A4_TEXT_WIDTH_IN, len(order))

    grid = sns.FacetGrid(
        data=df, col="section", col_order=sections,
        height=geometry["height"], aspect=geometry["aspect"],
        col_wrap=geometry["col_wrap"], sharey=True,
    )
    grid.set_titles(col_template="{col_name}")

    datasets = [d for d in DATASET_PALETTE if d in set(df["dataset"])]
    common = dict(
        x="target", y=metric, order=order,
        hue="dataset", hue_order=datasets, palette=DATASET_PALETTE,
    )
    for ax, section in zip(grid.axes.flat, sections):
        frame = df[df["section"] == section]
        for key, layer in claim_layers(frame):
            marker, scale, _ = CLAIM_MARKERS[key]
            sns.stripplot(
                data=layer, ax=ax, marker=marker, size=4.0 * scale,
                dodge=False, jitter=0.22, alpha=0.85,
                linewidth=0.3, edgecolor="white", legend=False, **common,
            )
        for i, target in enumerate(order):
            levels = frame.loc[frame["target"] == target, target_column]
            if levels.empty:
                continue
            # hlines over the segment rather than axhline: the panel holds three targets
            # side by side, and an axhline would draw the 0.5 rule through the 0.9 one.
            ax.hlines(
                levels.iloc[0], i - 0.45, i + 0.45,
                color="black", linestyle="--", linewidth=1.0, zorder=5,
            )
        ax.set_ylim(0, 1.05)

    grid.set_axis_labels("Guarantee target", metric.capitalize())
    handles, labels = section_handles(df)
    return finish_page_figure(
        grid,
        fonts=geometry["fonts"],
        width_in=A4_TEXT_WIDTH_IN,
        out_path=out_dir / f"{PRODUCER_NAME}_sections_{metric}.pdf",
        handles=handles,
        labels=labels,
    )


#: The four sections the distribution figure keeps, out of :data:`SECTION_SPECS`' six.
#:
#: Named by their coordinates rather than sliced by index, so reordering the specs cannot
#: silently change which panels are drawn. The four are the argument end to end: the model
#: arm against its own reference (the control, where it looks perfect), the same runs
#: against human labels (the finding), the human arm against human labels (what tuning on
#: ground truth buys), and that arm again keeping only what it claimed (the guarantee read
#: on its own terms). The two dropped panels both score the human arm against the *model*
#: reference, which is neither what it tuned on nor what it is judged by.
BOX_SECTION_SPECS = [
    ("model", "silver", False),
    ("model", "gold", False),
    ("human", "gold", False),
    ("human", "gold", True),
]

#: Precision and recall on one axes, in the colours every other figure in the repo uses
#: for them (``sweep_figures.TARGET_TYPE_PALETTE``).
METRIC_ORDER = ["Precision", "Recall"]
METRIC_PALETTE = dict(TARGET_TYPE_PALETTE)

#: What the claimed-only panels are titled in the box figure. ``section_label``'s
#: "(optimizer claims met)" is 22 characters, and a panel of a four-panel figure drawn to
#: one column is about 0.75 inches: the long form runs off both sides of its panel and
#: collides with its neighbour's title.
#:
#: "claimed met" matches the violation table's ``claimed_met_rate`` column and the
#: per-query figure's legend ("Optimizer: target met").
BOX_CLAIMED_SUFFIX = "claimed met"


def box_section_title(arm: str, label_set: str, claimed_only: bool) -> str:
    """:func:`section_label`, with the claimed-only line shortened and set bold.

    Bold because that line is the only thing separating this panel from its neighbour:
    the two share "Opt: human" and "Eval: human" and differ in the filter alone, so a
    reader scanning the titles has one word of contrast to find. Through mathtext rather
    than a font weight, because a title is one artist and matplotlib gives it one weight -
    ``$\\bf{...}$`` is how a single line of it gets its own.
    """
    title = section_label(arm, label_set, claimed_only)
    bold = r"$\bf{" + BOX_CLAIMED_SUFFIX.replace(" ", r"\ ") + "}$"
    return title.replace("(optimizer claims met)", bold)


def melt_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, metric), so both metrics can share an axes as a hue.

    The guarantee level rides along in ``level`` because precision and recall are targeted
    separately - ``ref01`` happens to zip them to the same number, so the two rules
    coincide, and a figure that assumed that would be wrong the first time they differ.
    """
    long = df.melt(
        id_vars=[c for c in df.columns if c not in METRICS],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    )
    long["level"] = [
        row[METRICS[row["metric"]]] for _, row in long[["metric", *METRICS.values()]].iterrows()
    ]
    long["metric"] = long["metric"].str.capitalize()
    return long


def plot_sections_box(
    df: pd.DataFrame, out_dir: Path, targets: Optional[Sequence[str]] = None,
    suffix: str = "", width_in: float = A4_TEXT_WIDTH_IN,
) -> Path:
    """The distribution figure: four sections, both metrics, no individual queries.

    Deliberately without the per-query points the sibling figure carries. A box and a
    fifteen-point cloud on the same axes ask the reader to do two things at once, and the
    question here is only where each arm sits: the per-query view answers "which query"
    and this one answers "how far". Dropping the points also frees the colour, which is
    why both metrics fit on one axes - precision and recall in
    :data:`METRIC_PALETTE`, the pairing every other figure in the repo uses.

    ``targets`` restricts the guarantee axis. The 0.9 target is where both arms collapse
    (12 and 14 of 15 queries below it), so a figure holding all three is dominated by a
    column nobody disputes, and the human arm's claimed-only panel there rests on a single
    query. The restricted variant is the one to read for the comparison; the full one is
    what stops the restriction from looking like a choice of convenient targets.

    ``width_in`` is the slot the figure is laid out for, and the restricted variant is
    drawn at :data:`COLUMN_WIDTH_IN` so it fits one column of a two-column page. Four
    panels in half a text width is about 0.8 inches each, which only works because of the
    three things that come with it: the panel height is taken all the way to the module's
    cap (:data:`BOX_HEIGHT_RATIO`, because most of a panel here is title, ticks and legend
    rather than axes), the ticks are stacked rather than rotated (a ``\\n`` in the label is
    what ``_rotate_ticks`` reads to leave it upright, so "P 0.5 / R 0.5" becomes two short
    centred lines instead of one wide slanted one), and the panels are packed tighter than
    the page-width default. Everything else scales itself -
    ``page_geometry`` derives the type sizes from the room one tick gets, except that the
    axis labels and the legend are held to :data:`AXIS_TYPE_FLOOR_PT`, which is what that
    rule sizes too small at column width.
    """
    frame = df[df["section"].isin([section_label(*s) for s in BOX_SECTION_SPECS])]
    order = sorted(frame["target"].unique())
    if targets is not None:
        order = [t for t in order if t in set(targets)]
        frame = frame[frame["target"].isin(order)]
        assert not frame.empty, f"no rows left after restricting to targets {targets}"
    sections = [
        section_label(*spec) for spec in BOX_SECTION_SPECS
        if section_label(*spec) in set(frame["section"])
    ]
    long = melt_metrics(frame)
    # Two short lines rather than one wide slanted one. `_rotate_ticks` leaves any label
    # containing a newline upright, so this is also what un-rotates them.
    tick_labels = [t.replace(" / ", "\n") for t in order]
    # `_facet_geometry` reads a panel's height off its *width*, so four panels in half a
    # text width come out as slivers whose axes are shorter than the titles above them.
    # The ratio lifts them back to the same panel height the page-width variant gets -
    # both land on the 1.9in cap - so the two read as one figure at two widths.
    geometry = page_geometry(
        len(sections), width_in, len(order), height_ratio=BOX_HEIGHT_RATIO,
    )
    fonts = dict(geometry["fonts"])
    fonts["label"] = max(fonts["label"], AXIS_TYPE_FLOOR_PT)
    fonts["legend"] = max(fonts["legend"], AXIS_TYPE_FLOOR_PT)

    grid = sns.FacetGrid(
        data=long, col="section", col_order=sections,
        height=geometry["height"], aspect=geometry["aspect"],
        col_wrap=geometry["col_wrap"], sharey=True,
    )
    grid.set_titles(col_template="{col_name}")

    for ax, section in zip(grid.axes.flat, sections):
        part = long[long["section"] == section]
        sns.boxplot(
            data=part, x="target", y="value", order=order,
            hue="metric", hue_order=METRIC_ORDER, palette=METRIC_PALETTE,
            ax=ax, linewidth=0.7, fliersize=1.5, width=0.7, legend=False,
            # Whiskers to the extremes rather than 1.5 IQR: with fifteen queries a
            # "outlier" is one query, and the tail is what the experiment is about.
            whis=(0, 100),
        )
        for i, target in enumerate(order):
            levels = part.loc[part["target"] == target, "level"]
            if levels.empty:
                continue
            for level in sorted(levels.unique()):
                ax.hlines(
                    level, i - 0.45, i + 0.45,
                    color="black", linestyle="--", linewidth=1.0, zorder=5,
                )
        ax.set_ylim(0, 1.05)
        # Pinned rather than left to the locator, which reads the room off the axes: these
        # panels are short enough that it answers 0 and 1 alone, and it answers differently
        # for the two width variants, whose furniture leaves them different room. The rules
        # sit at 0.5 and 0.7, so the midpoint is what a box is read against.
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(tick_labels)
        ax.set_title(
            box_section_title(*next(
                s for s in BOX_SECTION_SPECS if section_label(*s) == section
            ))
        )

    grid.set_axis_labels("Guarantee target", "Achieved")
    handles = [Patch(facecolor=METRIC_PALETTE[m], label=m) for m in METRIC_ORDER]
    return finish_page_figure(
        grid,
        fonts=fonts,
        width_in=width_in,
        out_path=out_dir / f"{PRODUCER_NAME}_sections_box{suffix}.pdf",
        handles=handles,
        labels=METRIC_ORDER,
        # Tighter than the 0.14 a page-width figure can afford: at four panels in half a
        # text width the gutter is the only thing left to spend, and the y ticks are
        # already drawn once for the row.
        wspace=0.08,
        # Height is the scarce dimension in a column, and two entries fit beside a
        # two-word axis label - so the legend shares its row rather than taking one.
        legend_inline=width_in <= COLUMN_WIDTH_IN,
    )


def violation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, target): how often the measured metric fell short of the target it promised.

    ``met`` here is measured, not claimed - it compares the scored precision/recall
    against the guarantee, which is a different question from the optimizer's own
    ``guarantee_met``. Both are reported so the gap between them is visible.
    """
    rows = []
    for (arm, target), group in df.groupby([ARM_COLUMN, "target"]):
        short = (
            (group["precision"] < group["precision_guarantee"])
            | (group["recall"] < group["recall_guarantee"])
        )
        # Absent or all-null when the costs carry no `CostSummary.guarantee`. The rate
        # is then not measurable, and NaN says that where a 0.0 would claim the
        # optimizer never met a target.
        claimed = group.get("guarantee_met", pd.Series(dtype="object"))
        rows.append(
            {
                ARM_COLUMN: arm,
                "target": target,
                "n_queries": len(group),
                "violations": int(short.sum()),
                "violation_rate": short.mean(),
                # How often the optimizer said the target held. Where this is high and
                # violation_rate is also high, the self-report was optimistic.
                "claimed_met_rate": claimed.mean() if claimed.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values([ARM_COLUMN, "target"])


def plot_violations(table: pd.DataFrame, label_set: str, out_dir: Path) -> Path:
    fonts = page_geometry(1, COLUMN_WIDTH_IN, table["target"].nunique())["fonts"]
    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.68))
    sns.barplot(
        data=table, x="target", y="violation_rate", hue=ARM_COLUMN,
        hue_order=ARM_ORDER, palette=ARM_PALETTE, ax=ax, linewidth=0,
    )
    ax.set_ylabel("Fraction below target", fontsize=fonts["label"])
    ax.set_xlabel("Guarantee target", fontsize=fonts["label"])
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=fonts["tick"])
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_text(ARM_LABELS.get(text.get_text(), text.get_text()))
        sns.move_legend(
            ax, "upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
            frameon=False, title=None, fontsize=fonts["legend"],
            handlelength=1.2, borderpad=0.1, labelspacing=0.3,
        )

    out_path = out_dir / f"{PRODUCER_NAME}_{label_set}_violations.pdf"
    figure.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(figure)
    return out_path


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", type=str, nargs="+", default=list(DEFAULT_BENCHMARKS),
        help="Benchmark directories to read under each --output-dirs entry.",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=["gold", "silver"],
        choices=["gold", "silver"],
        help="Which scoring pass to plot. gold is the measurement, silver the control.",
    )
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument(
        "--output-dirs", type=Path, nargs="+", default=[Path("benchmark_results")],
        help="One or more <task>/merged directories.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write the PDFs (default: the first --output-dirs entry).",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=["sections", "bars", "strip", "violations"],
        default=["sections", "bars", "strip", "violations"],
        help="'sections' is the main figure: every (optimized against, evaluated "
        "against) pair as its own panel, plus the human-optimized arm filtered to what "
        "the optimizer claimed, drawn once per query and once as boxes. 'strip' is the "
        "distribution view within one label set; 'bars' is one bar per query, which only "
        "reads at small query counts.",
    )
    args = parser.parse_args()

    apply_default_style()
    out_dir = args.out_dir or args.output_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Kept across the label-set loop: the section figure is the one that spans both, so
    # it cannot be drawn inside it.
    loaded: List[pd.DataFrame] = []

    for label_set in args.labels:
        paths = [
            path
            for output_dir in args.output_dirs
            for benchmark in args.benchmarks
            for path in [
                output_dir / benchmark / args.split
                / f"{PRODUCER_NAME}_{label_set}_metrics.csv"
            ]
            if path.is_file()
        ]
        if not paths:
            logger.warning(
                "No %s_%s_metrics.csv under %s for %s; skipping. Has the task merged yet?",
                PRODUCER_NAME, label_set, [str(d) for d in args.output_dirs], args.benchmarks,
            )
            continue

        df = load_df(paths, label_set)
        loaded.append(df)
        logger.info(
            "%s: %d rows, %d queries, arms %s",
            label_set, len(df), df["query"].nunique(),
            sorted(df[ARM_COLUMN].unique()),
        )

        for metric in METRICS:
            if "bars" in args.figures:
                logger.info("wrote %s", plot_per_query(df, metric, label_set, out_dir))
            if "strip" in args.figures:
                logger.info("wrote %s", plot_metric_strip(df, metric, label_set, out_dir))
                # Only worth the extra file when there is more than one dataset to split
                # by; with one it is the pooled figure again.
                if df["dataset"].nunique() > 1:
                    logger.info(
                        "wrote %s",
                        plot_metric_strip(df, metric, label_set, out_dir, by_dataset=True),
                    )

        table = violation_table(df)
        table_path = out_dir / f"{PRODUCER_NAME}_{label_set}_violations.csv"
        table.to_csv(table_path, index=False)
        logger.info("wrote %s", table_path)
        if "violations" in args.figures:
            logger.info("wrote %s", plot_violations(table, label_set, out_dir))
        print(f"\n=== scored against {label_set} ===")
        print(table.to_string(index=False))

    if "sections" in args.figures and loaded:
        sections = build_sections(pd.concat(loaded, ignore_index=True))
        for metric in METRICS:
            logger.info("wrote %s", plot_sections(sections, metric, out_dir))
        # Two variants of the distribution figure: every target, and the two the arms do
        # not both collapse at. See `plot_sections_box`.
        logger.info("wrote %s", plot_sections_box(sections, out_dir))
        low = [t for t in sorted(sections["target"].unique()) if "0.9" not in t]
        if low and len(low) < sections["target"].nunique():
            logger.info(
                "wrote %s",
                plot_sections_box(
                    sections, out_dir, targets=low, suffix="_low_targets",
                    width_in=COLUMN_WIDTH_IN,
                ),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
