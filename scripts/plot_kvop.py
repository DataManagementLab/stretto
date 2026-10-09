"""What one KV-compressed operator costs on disk, and what it buys in runtime.

One row per (dataset, model), the two model sizes of a dataset adjacent. Every row is a
ladder of compression ratios laid out on a log storage axis and labelled with them, most
aggressive on the left. Each dot is shaded by the speedup that ratio delivered in the
chosen sweep, with everything that failed to beat break-even pooled into the one palest
band, so the figure reads as "how far left can I get before the colour washes out".

The first argument picks the sweep under ``artifacts/kvops``. ``kvop01`` *adds* one
compressed operator per state to the uncompressed reference suite (one per modality on
ecommerce, drawn as two paired rows), so the vanilla operators stay available to the
optimizer and the speedup is what adding that operator buys.

Ratio 0.0 is the full cache stored losslessly, not the absence of one -- it is the most
expensive point on the ladder in bytes and the cheapest in quality. Where the sweep ran
it (the 8B slots) it is a filled dot carrying its speedup, which is what caching buys
with compression taken out of the picture. The 70B/72B slots start at 0.3/0.5, so their
uncompressed footprint is drawn as a hollow ring: a size with no measurement behind it.

The condition with no cache at all is step 0, which the sweep records with
``storage_gb = 0`` and which every speedup here is measured against. It has no position
on a log storage axis and so is not a point on the figure.

Two halves, deliberately from different sources:
  storage   reasondb/memory_footprint/kv_cache_footprint/kv_cache_footprint_<mod>_<n>.csv
  speedup   artifacts/kvops/<sweep>/merged/<benchmark>/dev/kv_operator.{parquet,csv}

Corpus sizes are pinned per dataset (Movie 10k, Rotowire 728, the rest 1k) rather than
normalised to a nominal count, so a bar is the footprint that dataset actually has.

Several target ratios give one image with a panel each, side by side over a shared x
scale and one shared pair of keys, so a row is at the same height and a ratio at the same
x in every panel and only the colour moves between them. ``--compact`` folds those
panels into one: each row opens into a lane per target, so the comparison sits on
adjacent lines instead of across the page.

Run:  python scripts/plot_kvop.py kvop01 --target-ratio 0.5
      python scripts/plot_kvop.py kvop01 --target-ratio 0.7 0.9 --compact
"""

from __future__ import annotations

import argparse
import ast
import csv
import functools
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
KVOPS_ROOT = REPO_ROOT / "artifacts" / "kvops"
DEFAULT_FOOTPRINT = REPO_ROOT / "reasondb" / "memory_footprint" / "kv_cache_footprint"

#: The sweeps this plots, as ``name -> the sweep its ecommerce rows are read from``, None
#: meaning the sweep itself.
EXPERIMENTS: dict[str, str | None] = {
    "kvop01": None,
}
ECOMMERCE_BENCHMARK = "ecommerce_random_large"

#: ``Email (1000)`` -> name ``Email``, items ``1000``. Same shape the footprint plotter
#: parses, kept identical so one CSV family feeds both.
HEADER_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<items>[\d\s,]+)\)$")

# --- palette -----------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#d7d5d0"

#: Yellow (slow) -> dark green (fast). Sequential and monotone in lightness, so the order
#: survives greyscale; the band labels carry the sign, and :func:`add_speedup_bar` marks
#: where break-even falls. The yellow end is deeper than ColorBrewer's YlGn, whose
#: ``#ffffcc`` all but vanishes against this figure's near-white surface.
SPEEDUP_RAMP = ["#e9e06a", "#aed36f", "#6cbf6a", "#2f9e52", "#12662f"]

#: Storage sizes to mark with a rule: illustrative round budgets, not values the sweep
#: measured, which is what the key calls them.
DEFAULT_BUDGET_GB = (100.0, 1000.0)
BUDGET_RULE = "#5f8a2a"
BUDGET_LABEL = "example budget sizes"

#: What a speedup may be measured on: ``name -> (column, how the caption says it)``.
#:
#: ``total_runtime_s`` is ``execution_runtime_s + tuning_runtime_s``: execution plus the
#: tuning that chose the plan, excluding configuring. ``wall_clock_s`` is the run's own
#: ``component_times["end_to_end"]``, the only column that is a true elapsed time. The
#: top-level columns and the ``component_times`` entries are measured by separate clocks,
#: so the two differ by more than the configuring time alone.
METRICS: dict[str, tuple[str, str]] = {
    "execution": ("execution_runtime_s", "execution"),
    "total": ("total_runtime_s", "end-to-end runtime"),
    "wall-clock": ("wall_clock_s", "end-to-end wall-clock"),
}
DEFAULT_METRIC = "execution"

#: How a configuration's queries are reduced to one number, as
#: ``name -> (title prefix, title note, caption suffix)``.
#:
#: ``total`` is a ratio of sums over every query, so it states how much less compute the
#: whole workload cost. On these benchmarks that is dominated by whether a query is
#: cacheable at all rather than by how well caching works: 43-73% of the compute sits in
#: queries a configuration never touches, and they are the longest ones, so they
#: contribute an exact 1.0 and pull the total toward break-even.
#:
#: ``movers`` is the same ratio of sums over only the queries whose runtime the
#: configuration changed by more than :data:`MOVED_THRESHOLD`, in either direction.
#:
#: This is not the same as "the cache was used": the sweep does not record which operator
#: ran on which model, so runtime movement is the only observable. A query the cache
#: served for less than the threshold is dropped, and a query whose plan escalated to the
#: large model without touching the cache is kept.
#:
#: ``median`` is the middle per-query ratio, immune both to that weighting and to the
#: occasional escalation that triples one query's runtime.
#:
#: ``mean`` is the arithmetic mean of those ratios. It runs well above the median --
#: often by half a turn -- and that is a property of ratios, not of the caching: a query
#: that halves gives 2.0 while a query that doubles gives 0.5, so the upside is unbounded
#: and the downside is floored at zero. Read it as the most generous of the five.
#:
#: ``geomean`` is the mean that ratios actually want: ``exp(mean(log r))``, under which
#: 2x and 0.5x are equal and opposite, so a query that doubles cancels a query that
#: halves. It lands between the median and the arithmetic mean, and it is the one to
#: quote when a single "average speedup" has to stand for a set of ratios.
AGGREGATES: dict[str, tuple[str, str, str]] = {
    "total": ("", "", ""),
    "movers": ("", ", queries this configuration changed",
               " · only queries whose runtime it changed, faster or slower"),
    "median": ("median per query ", "", ""),
    "mean": ("mean per query ", "", ""),
    "geomean": ("geometric-mean per query ", "", ""),
}
DEFAULT_AGGREGATE = "total"

#: How far a query's runtime must move to count as touched by the cache. The simulator's
#: own run-to-run spread is about +-0.5%, so this is comfortably outside it.
MOVED_THRESHOLD = 0.02

#: Interior edges of the speedup bands. The first edge is break-even, which puts every
#: run that did not get faster in the one palest band and spends the rest of the ramp on
#: the speedups that are real. No band straddles "got faster" and "did not".
DEFAULT_BINS = (1.0, 1.1, 1.2, 1.4)

plt.rcParams.update({
    # Serif, matching the repo's other figures (reasondb.evaluation.plotting). STIXGeneral
    # is a Times face that ships with matplotlib, so it renders the same on every machine;
    # Times New Roman is the fallback.
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
})


# --- the two catalogues ------------------------------------------------------
@dataclass(frozen=True)
class Slot:
    """A sweep slot: which model it stands for, and how to find it in the CSVs."""

    modality: str
    model: str       # exactly as the footprint CSV's Model column spells it
    size: str        # shown in the row label


SLOTS: dict[str, Slot] = {
    "text_small": Slot("text", "Llama-3.1 8B", "8B"),
    "text_large": Slot("text", "Llama-3.1 70B", "70B"),
    "image_small": Slot("image", "Llava-Next 8B", "8B"),
    "image_large": Slot("image", "Llava-Next 72B", "72B"),
}

#: Modalities, for reading the footprint CSV family.
MODALITY_ORDER = ["image", "text"]

#: A step that moves one slot per modality at the same model size is one row, and that
#: row needs a slot of its own to be grouped and labelled by. ecommerce is swept this way
#: under both ``kv_operator_pairs`` and ``kv_operator_marginal`` (kvop01).
PAIR_SLOTS: dict[str, Slot] = {
    "pair_small": Slot("multimodal", "", "8B + 8B"),
    "pair_large": Slot("multimodal", "", "70B + 72B"),
}
SLOTS.update(PAIR_SLOTS)

#: The slots a sweep step can configure, in the order a joined ratio label spells them
#: (text before image, so ``0.8/0.99`` reads text-then-image).
REAL_SLOTS = ("text_small", "text_large", "image_small", "image_large")

#: Rows top to bottom: single-slot modalities first, then the paired rows, biggest model
#: first within each. Datasets keep :data:`DATASETS` order inside a group.
GROUP_ORDER = [
    ("image", ["image_large", "image_small"]),
    ("text", ["text_large", "text_small"]),
    ("multimodal", ["pair_large", "pair_small"]),
]


@dataclass(frozen=True)
class Dataset:
    """A benchmark, and the footprint column that measures the corpus it ran on."""

    benchmark: str   # directory under <sweep>/merged
    label: str       # shown in the row label
    items_tag: str   # kv_cache_footprint_<modality>_<items_tag>.csv
    column: str      # column name, before the "(n)" suffix


DATASETS: list[Dataset] = [
    Dataset("artwork_random_medium", "Artwork", "1k", "Artwork"),
    Dataset("ecommerce_random_large", "E-Comm.", "1k", "E-Comm."),
    Dataset("email_random", "Email", "1k", "Email"),
    Dataset("movie_random_huge", "Movie", "10k", "Movie"),
    Dataset("rotowire_random", "Rotowire", "728", "Rotowire"),
]


# --- formatting --------------------------------------------------------------
def human_gb(gb: float) -> str:
    """Format a GB figure with the unit that keeps it in [1, 1000)."""
    for scale, unit in ((1e6, "PB"), (1e3, "TB"), (1.0, "GB"), (1e-3, "MB")):
        if gb >= scale:
            value = gb / scale
            return f"{value:.0f} {unit}" if value >= 10 else f"{value:.1f} {unit}"
    return f"{gb * 1e6:.0f} KB"


def ratio_label(cr: float | str) -> str:
    """``0.99`` -> ``"0.99"``, ``0.5`` -> ``"0.5"`` -- no trailing zero padding.

    Zero keeps one decimal, so an uncompressed dot is labelled the way the legend and
    the footprint tables both spell that ratio. A sweep that moves several slots at once
    hands this an already-joined label such as ``"0.8/0.99"``, which passes straight
    through -- kvop01's ecommerce steps do exactly that.
    """
    if isinstance(cr, str):
        return cr
    return "0.0" if cr == 0 else f"{cr:g}"


def is_uncompressed(cr: float | str) -> bool:
    """Whether a dot is the ratio-0.0 point, and so drawn hollow and off the colour scale.

    A joined label counts only when every slot it names is uncompressed; a step that
    caches one modality in full and compresses the other is still a compressed step.
    """
    if isinstance(cr, str):
        return all(float(part) == 0 for part in cr.split("/"))
    return cr == 0


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def output_formats(spec: str) -> list[str]:
    """The formats to write, PDF among them whatever was asked for.

    The PDF is the paper figure, so it is always written, even with ``--formats png``.
    """
    formats = [f.strip() for f in spec.split(",") if f.strip()]
    if "pdf" not in formats:
        formats.append("pdf")
    return formats


# --- speedups ----------------------------------------------------------------
def parse_crs(value) -> float | None:
    """The compression ratio in a ``*_crs`` cell, or None when the slot is unused.

    The column arrives as a one-element list, as its ``"[0.5]"`` repr, or as NaN,
    depending on whether the frame came back from parquet or from CSV. All three mean
    the same thing and all three turn up in one sweep's output.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return None
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return values[0] if values else None


#: ``(slot, ratio)`` pairs of one step, which is what a row's dots are keyed by.
Config = tuple[tuple[str, float], ...]


def step_config(row) -> Config:
    """Every slot this step configured, not merely the first one.

    The ``kv_operator`` plan moves one slot at a time; ``kv_operator_pairs`` and
    ``kv_operator_marginal`` (kvop01) sweep ecommerce two at a time, one per modality. Stopping at the first populated column would drop the other modality's
    cache from the disk figure, which on the large ecommerce pair hides 2.9 TB of image
    cache behind 47 GB of text. A row that populates none of them is the step-0 baseline.
    """
    found = []
    for name in REAL_SLOTS:
        cr = parse_crs(row.get(f"{name}_crs"))
        if cr is not None:
            found.append((name, cr))
    return tuple(found)


def group_of(config: Config) -> str | None:
    """The row a step belongs to, or None if its slots are not a shape we draw.

    One slot is its own row. Two slots of the same model size are a paired row. Anything
    else -- a pair straddling sizes, or three at once -- has no honest place in a figure
    whose rows are "one dataset on one model", so it is reported rather than folded in.
    """
    slots = [slot for slot, _ in config]
    if not slots:
        return None
    if len(slots) == 1:
        return slots[0]
    sizes = {slot.rsplit("_", 1)[1] for slot in slots}
    if len(sizes) == 1 and len(slots) == 2:
        return f"pair_{sizes.pop()}"
    return None


def config_label(config: Config) -> float | str:
    """What the dots of this step are labelled with.

    A single slot keeps its float, so it formats as a plain ratio. A pair whose halves
    agree collapses to one number; a pair that disagrees names both, as ``0.8/0.99``.
    """
    ratios = [cr for _, cr in config]
    if len(ratios) == 1 or len(set(ratios)) == 1:
        return ratios[0]
    return "/".join(ratio_label(cr) for cr in ratios)


def read_sweep(kvops_dir: Path, dataset: Dataset) -> pd.DataFrame | None:
    """One benchmark's merged sweep, from parquet if it is there and CSV if it is not."""
    base = kvops_dir / "merged" / dataset.benchmark / "dev"
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        path = base / f"kv_operator{suffix}"
        if path.exists():
            return reader(path)
    return None


def aggregate_speedup(group, column: str, how: str) -> float | None:
    """One number for one configuration's queries, by the rule *how*.

    A ``movers`` group in which nothing moved is 1.0 rather than nothing: the
    configuration was available and changed no query, which is a result, not a gap.
    """
    ratio = group["baseline_s"] / group[column]
    if how in ("median", "mean", "geomean"):
        if not len(ratio):
            return None
        if how == "median":
            return float(ratio.median())
        if how == "mean":
            return float(ratio.mean())
        # exp(mean(log r)): a log is only defined on the positive side, and a ratio of
        # two runtimes is positive by construction -- the guard is for a malformed row.
        positive = ratio[ratio > 0]
        return float(np.exp(np.log(positive).mean())) if len(positive) else None
    if how == "movers":
        moved = (ratio - 1).abs() > MOVED_THRESHOLD
        if not moved.any():
            return 1.0
        group, ratio = group[moved], ratio[moved]
    total = group[column].sum()
    return group["baseline_s"].sum() / total if total > 0 else None


def speedups(
    kvops_dir: Path, target_ratio: float, metric: str = DEFAULT_METRIC,
    *, sources: dict[str, Path] | None = None, aggregate: str = DEFAULT_AGGREGATE,
) -> tuple[dict[tuple[str, Config], float], list[str]]:
    """Aggregate speedup per (benchmark, step config) at *target_ratio*, on *metric*.

    The ratio is a total over the queries, ``sum(baseline) / sum(with cache)``, not a mean
    of per-query ratios: one query whose runtime collapses to near zero would otherwise
    set the colour of a whole row. Queries are matched to their own step-0 baseline, so a
    row only ever compares like with like.

    *sources* redirects individual benchmarks to another sweep directory, e.g. to read
    ecommerce from a sweep that moved both modalities together rather than one slot at a
    time (see :data:`EXPERIMENTS`).
    """
    column = METRICS[metric][0]
    sources = sources or {}
    out: dict[tuple[str, Config], float] = {}
    notes: list[str] = []
    key = ["query", "precision_guarantee", "recall_guarantee"]

    for dataset in DATASETS:
        source = sources.get(dataset.benchmark, kvops_dir)
        frame = read_sweep(source, dataset)
        if frame is None:
            notes.append(
                f"{dataset.benchmark}: no kv_operator.parquet/.csv under {source}, skipped"
            )
            continue

        frame = frame[
            np.isclose(frame["precision_guarantee"], target_ratio)
            & np.isclose(frame["recall_guarantee"], target_ratio)
        ].copy()
        if frame.empty:
            notes.append(f"{dataset.benchmark}: nothing at target ratio {target_ratio:g}")
            continue

        frame["config"] = frame.apply(step_config, axis=1)

        bad = ~(frame[column] > 0)
        if bad.any():
            notes.append(
                f"{dataset.benchmark}: dropped {int(bad.sum())} row(s) with a "
                f"non-positive {column}"
            )
            frame = frame[~bad]

        baseline = frame[frame["config"].map(len) == 0].set_index(key)[column]
        if baseline.empty:
            notes.append(f"{dataset.benchmark}: no step-0 baseline, skipped")
            continue
        baseline = baseline[~baseline.index.duplicated()]

        swept = frame[frame["config"].map(len) > 0].copy()
        swept["baseline_s"] = swept.set_index(key).index.map(baseline)
        unmatched = swept["baseline_s"].isna()
        if unmatched.any():
            notes.append(
                f"{dataset.benchmark}: {int(unmatched.sum())} row(s) had no matching "
                f"baseline query"
            )
            swept = swept[~unmatched]

        for config, group in swept.groupby("config"):
            if group_of(config) is None:
                notes.append(
                    f"{dataset.benchmark}: skipped a step configuring "
                    f"{', '.join(f'{s}@{cr:g}' for s, cr in config)} -- not one slot, "
                    f"nor one pair at a single model size"
                )
                continue
            value = aggregate_speedup(group, column, aggregate)
            if value is not None:
                out[(dataset.benchmark, config)] = value

    return out, notes


# --- storage -----------------------------------------------------------------
def footprint(footprint_dir: Path) -> tuple[dict[tuple[str, str, str, float], float], list[str]]:
    """``(items_tag, column, model, ratio) -> GB``, straight from the CSVs.

    Nothing is rescaled. Rotowire is read from its own 728-item table rather than
    projected from the 1k one.
    """
    out: dict[tuple[str, str, str, float], float] = {}
    notes: list[str] = []
    tags = sorted({d.items_tag for d in DATASETS})

    for modality in MODALITY_ORDER:
        for tag in tags:
            path = footprint_dir / f"kv_cache_footprint_{modality}_{tag}.csv"
            if not path.exists():
                continue
            with path.open() as handle:
                rows = list(csv.reader(handle))
            if not rows:
                notes.append(f"{path.name}: empty, skipped")
                continue

            header, body = rows[0], [r for r in rows[1:] if r]
            columns: dict[int, str] = {}
            for i, cell in enumerate(header):
                match = HEADER_RE.match(cell)
                if match:
                    columns[i] = match.group("name")

            for row in body:
                model, ratio = row[0].strip(), float(row[1])
                for i, name in columns.items():
                    if i < len(row) and row[i].strip():
                        out[(tag, name, model, ratio)] = float(row[i])

    return out, notes


# --- assembling the rows -----------------------------------------------------
@dataclass
class Row:
    dataset: Dataset
    slot: str
    label: str
    dots: list[tuple[float, float | str, float]]  # (gb, ratio, speedup), smallest first
    uncompressed_gb: float | None


def footprint_of(sizes: dict, dataset: Dataset, config: Config) -> float | None:
    """Disk for one step: every slot it materialises, added up.

    The sum is what a paired sweep itself records in ``storage_gb``, so the two sources
    agree to the footprint tables' rounding.
    """
    total = 0.0
    for slot, cr in config:
        gb = sizes.get((dataset.items_tag, dataset.column, SLOTS[slot].model, cr))
        if gb is None:
            return None
        total += gb
    return total


def build_rows(sizes: dict, speed: dict, *, notes: list[str]) -> list[Row]:
    """Every (dataset, row-group) that has both a footprint and a speedup, in order.

    Dataset is the outer loop inside a group: the two sizes of one corpus belong
    together, and reading down the figure should step corpus by corpus.
    """
    by_group: dict[tuple[str, str], list[tuple[Config, float]]] = {}
    for (benchmark, config), value in speed.items():
        group = group_of(config)
        if group is not None:
            by_group.setdefault((benchmark, group), []).append((config, value))

    rows: list[Row] = []
    for _, group_slots in GROUP_ORDER:
        for dataset in DATASETS:
            for group in group_slots:
                entries = by_group.get((dataset.benchmark, group))
                if not entries:
                    continue

                dots: list[tuple[float, float | str, float]] = []
                missing: list[str] = []
                for config, value in entries:
                    gb = footprint_of(sizes, dataset, config)
                    if gb is None:
                        missing.append(ratio_label(config_label(config)))
                        continue
                    dots.append((gb, config_label(config), value))
                if missing:
                    notes.append(
                        f"{dataset.label} / {SLOTS[group].size}: no footprint for "
                        f"ratio(s) {', '.join(missing)}"
                    )
                if not dots:
                    continue

                dots.sort(key=lambda dot: dot[0])
                # The uncompressed point is every slot of this row at ratio 0.0 -- for a
                # pair, both caches in full.
                slots = [slot for slot, _ in entries[0][0]]
                uncompressed = footprint_of(
                    sizes, dataset, tuple((slot, 0.0) for slot in slots)
                )
                if uncompressed is None:
                    notes.append(
                        f"{dataset.label} / {SLOTS[group].size}: no uncompressed "
                        f"(0.0) footprint"
                    )

                rows.append(
                    Row(
                        dataset=dataset,
                        slot=group,
                        label=f"{dataset.label} · {SLOTS[group].size}",
                        dots=dots,
                        uncompressed_gb=uncompressed,
                    )
                )
    return rows


# --- drawing -----------------------------------------------------------------
def band_labels(edges: tuple[float, ...], *, break_even: float = 1.0) -> list[str]:
    """One explicit range per colour, ends left open the way the data is.

    When the palest band ends exactly at break-even it is named for what it means
    rather than for its bounds: everything in it failed to get faster.
    """
    first = (
        "no speedup"
        if math.isclose(edges[0], break_even)
        else f"x < {edges[0]:g}"
    )
    labels = [first]
    labels += [f"{lo:g} ≤ x < {hi:g}" for lo, hi in zip(edges, edges[1:])]
    labels.append(f"x ≥ {edges[-1]:g}")
    return labels


def fit_top_bin(edges: tuple[float, ...], top: float) -> tuple[float, ...]:
    """Lower the last band edge so the darkest colour has something in it.

    Snaps down to a 0.1 grid, so a figure topping out at 1.37 gets a ``x >= 1.3`` band
    rather than an empty ``x >= 1.4``, and any interior edge the new top has overtaken
    is dropped. The first edge survives whatever happens: it is break-even, and a figure
    that stops naming it stops saying which dots got faster at all.
    """
    if not edges or top >= edges[-1]:
        return edges
    snap = math.floor(top * 10 + 1e-9) / 10
    kept = [e for e in edges if e < snap] or [edges[0]]
    if snap > kept[-1]:
        kept.append(snap)
    return tuple(kept)


def make_norm(edges: tuple[float, ...]) -> tuple[ListedColormap, BoundaryNorm, list[float]]:
    """A discrete ramp over *edges*, with finite outer bands so a colorbar can draw them."""
    n_bands = len(edges) + 1
    colors = [SPEEDUP_RAMP[round(i * (len(SPEEDUP_RAMP) - 1) / max(n_bands - 1, 1))]
              for i in range(n_bands)]
    step = min(np.diff(edges)) if len(edges) > 1 else 0.2
    drawn = [edges[0] - step, *edges, edges[-1] + step]
    cmap = ListedColormap(colors)
    return cmap, BoundaryNorm(drawn, cmap.N), drawn


def add_speedup_bar(fig, cax, cmap, norm, drawn, edges, *, break_even=1.0,
                    metric: str = DEFAULT_METRIC, aggregate: str = DEFAULT_AGGREGATE,
                    fs: dict[str, float] | None = None):
    """Horizontal colorbar whose ticks name the band they sit in.

    Drawn into a figure-level axes rather than stolen from a plot, because with several
    panels there is one scale for all of them and it belongs to the figure.
    """
    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        spacing="uniform",
    )
    fs = fs or FONT
    centres = [(lo + hi) / 2 for lo, hi in zip(drawn, drawn[1:])]
    bar.set_ticks(centres)
    bar.set_ticklabels(band_labels(edges, break_even=break_even))
    bar.outline.set_visible(False)
    bar.ax.tick_params(axis="x", length=0, pad=5, labelsize=fs["band"], labelcolor=INK_2)
    # Break-even is the one boundary a reader must not have to infer from the numbers,
    # so when it falls inside the bar rather than on its edge it gets a rule. No caption
    # under the bar: what the speedup is measured on is the figure caption's job.
    if edges[0] < break_even < edges[-1]:
        bar.ax.axvline(break_even, color=INK, linewidth=1.4, zorder=5)
    return bar


#: Panel geometry, in inches. ``ROW_IN`` is the least pitch a row gets; draw() raises it
#: when the ratio labels are set large enough to need more to clear the row above.
ROW_IN = 0.43
TOP_MARGIN_IN = 0.10  # above everything; there is no title, the paper caption is one
PROVENANCE_IN = 0.26  # the extra line under it, when a row comes from another sweep
#: The strips around the plot -- the panel headers, the tick labels and caption, the
#: keys beside the colour bar -- are not fixed heights: draw() sizes each from the type
#: it holds, so larger sizes here or a --font-scale move the layout with them.
FOOT_IN = 0.10

#: The x-axis tick marks and the gap from them to their labels, in points.
TICK_LEN, TICK_PAD = 5.0, 3.0

#: Vertical rhythm inside a panel, in row units. A modality header needs clearing only
#: from the row labels beside it in the gutter -- the ratio labels above a row sit in
#: the plot, to its right -- so it can sit much closer than a full row pitch.
TOP_PAD = 0.55        # panel top to the first header
HEADER_TO_ROW = 0.60  # a header to the first row under it
MODALITY_GAP = 0.15   # extra space before a header, on top of the row pitch
BOTTOM_PAD = 0.32     # last row to the axis

#: The colour bar is never wider than this. How narrow it may get is measured instead:
#: every band name has to fit under its band, and when the keys beside the bar would
#: squeeze it below that, the keys go on a row of their own above it.
MAX_BAR_IN = 14.0

#: The two keys. The handles say "hollow point" and "dashed line", so the text only has
#: to say what each one means.
KEY_UNCOMPRESSED = "uncompressed cache, size only"

#: Type sizes, all scaled by ``--font-scale``. Kept in one place because "make it
#: readable" is a decision about the set, not about any one label.
FONT: dict[str, float] = {
    "title": 15,
    "provenance": 14,
    "panel": 19,      # "target p/r = 0.5"
    "row": 18.5,      # "Artwork · 72B"
    "group": 17.5,    # "image" / "text" / "multimodal"
    "ratio": 14.5,    # the number above each dot
    "lane": 16,       # the target tag on each lane of a --compact figure
    "tick": 17,       # "100 GB"
    "caption": 16.5,  # the axis caption
    "band": 17.5,     # "1.2 <= x < 1.4"
    "key": 18,        # the two key lines
    "budget": 11,     # "100 GB" on the budget rule
}


def fonts(scale: float) -> dict[str, float]:
    return {name: size * scale for name, size in FONT.items()}

#: Between two panels. The tick labels of one ("10 TB") and of the next ("1.0 GB") both
#: overhang their axes, so this is what keeps them apart rather than merely tidy.
GAP_X_IN = 0.50

#: How :option:`--width` splits between the plot and the right margin. The row-label
#: gutter to the left is not a share of it: it is measured from the labels themselves
#: (see :func:`draw`), so it fits whatever font and type size they are set in.
PLOT_FRAC, RIGHT_FRAC = 0.785, 0.015

#: The row labels and the group headers share one left edge, this far in from the
#: figure's; the longest of them then clears the plot -- or, in a compact figure, the
#: lane tags -- by ``LABEL_CLEAR_IN``, and a lane tag clears the plot by ``TAG_GAP_IN``.
LEFT_MARGIN_IN = 0.06
LABEL_CLEAR_IN = 0.25
TAG_GAP_IN = 0.06


def layout(rows: list[Row]) -> tuple[dict[int, float], list[tuple[float, str]], float]:
    """Row positions, modality header positions, and the y extent a panel needs."""
    y_of: dict[int, float] = {}
    headers: list[tuple[float, str]] = []
    y = 0.0
    last_modality = None
    for i, row in enumerate(rows):
        modality = SLOTS[row.slot].modality
        if modality != last_modality:
            if last_modality is not None:
                y += MODALITY_GAP
            headers.append((y, modality))
            y += HEADER_TO_ROW
            last_modality = modality
        y_of[i] = y
        y += 1.0
    return y_of, headers, y


def prepare_axes(ax, extent, lo, hi, args) -> None:
    """Log x axis, bare frame and the budget rules: what every panel starts from."""
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(extent - 1.0 + BOTTOM_PAD, -TOP_PAD)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)

    # Budget rules only, unlabelled: the x axis already names the sizes they stand on,
    # and what they mean is said once in the key rather than twice per panel.
    for budget in args.budget_gb:
        ax.axvline(budget, color=BUDGET_RULE, linestyle=(0, (5, 3)), linewidth=1.4,
                   zorder=1)


def finish_axes(ax, fs: dict[str, float]) -> None:
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: human_gb(v)))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    # A log axis volunteers a minor tick per 2..9; none of them are labelled, so none of
    # them should leave a mark on the spine either.
    ax.tick_params(axis="x", which="minor", length=0)
    # Panels sit side by side, so each carries its own tick labels; the caption they
    # share is written once, under the middle of the row. A short tick marks where each
    # label stands -- the budget rules already do that for their sizes, but 10 GB has
    # nothing else pointing at it.
    ax.tick_params(axis="x", direction="out", length=TICK_LEN, width=1.0, color=MUTED,
                   pad=TICK_PAD, labelsize=fs["tick"], labelcolor=INK_2)


def draw_ladder(ax, row: Row, yy: float, cmap, norm) -> list[tuple[float, float | str]]:
    """One row's connector, dots and hollow reference at height *yy*.

    Returns the points that want a ratio label, so the caller decides where -- and
    whether -- to write them.
    """
    # Ratio 0.0 is a materialised cache like any other -- the full one, losslessly --
    # so where the sweep measured it (the 8B slots) it is a filled dot carrying its
    # speedup. The hollow marker is only for a ratio the sweep did not run: the
    # 70B/72B slots start at 0.3/0.5, so their uncompressed footprint is a size with
    # no measurement behind it.
    measured = list(row.dots)
    reference = (
        None
        if any(is_uncompressed(cr) for _, cr, _ in measured)
        else row.uncompressed_gb
    )

    marks = [gb for gb, _, _ in measured] + ([reference] if reference else [])
    ax.plot([min(marks), max(marks)], [yy, yy], color=HAIRLINE, linewidth=1.2, zorder=2)

    if reference:
        ax.scatter([reference], [yy], s=175, facecolors="none",
                   edgecolors="#123f6b", linewidths=2.0, zorder=4)

    ax.scatter([gb for gb, _, _ in measured], [yy] * len(measured),
               s=125, c=[sp for _, _, sp in measured],
               cmap=cmap, norm=norm, edgecolors="none", zorder=4)

    # Every point carries its own ratio, the hollow one included.
    labelled: list[tuple[float, float | str]] = [(gb, cr) for gb, cr, _ in measured]
    if reference:
        labelled.append((reference, 0.0))
    return labelled


def _nondecreasing(values: list[float]) -> list[float]:
    """Least-squares non-decreasing fit to *values* (pool-adjacent-violators)."""
    blocks: list[list[float]] = []            # [sum, count]
    for v in values:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    return [total / count for total, count in blocks for _ in range(int(count))]


def spread_labels(centres: list[float], widths: list[float], gap: float) -> list[float]:
    """Label centres as close to *centres* as they can be without overlapping.

    Neighbours must sit at least half of each width plus *gap* apart. Subtracting the
    cumulative spacing each label needs turns that into "the shifted positions must not
    decrease", whose least-squares answer is an isotonic fit: labels in a crowded run
    move apart symmetrically around where their dots are, and uncrowded ones stay put.
    """
    offsets = [0.0]
    for left, right in zip(widths, widths[1:]):
        offsets.append(offsets[-1] + (left + right) / 2 + gap)
    fitted = _nondecreasing([c - o for c, o in zip(centres, offsets)])
    return [f + o for f, o in zip(fitted, offsets)]


def draw_ratio_labels(ax, labelled, yy: float, fs: dict[str, float]) -> None:
    """The ratio over each dot of a row, nudged sideways only where two would collide.

    Narrow panels put a ladder's top rungs (0.3, 0.0) close together on the log axis;
    rather than widening the figure to make room, the labels share it.
    """
    labelled = sorted(labelled, key=lambda item: item[0])
    if not labelled:
        return
    lo, hi = ax.get_xlim()
    fig = ax.get_figure()
    axis_in = ax.get_position().width * fig.get_figwidth()
    per_decade = axis_in / math.log10(hi / lo)
    texts = [ratio_label(cr) for _, cr in labelled]
    widths = [text_width_in(text, fs["ratio"]) for text in texts]
    at = [math.log10(gb / lo) * per_decade for gb, _ in labelled]
    for text, gb, x in zip(texts, (gb for gb, _ in labelled),
                           spread_labels(at, widths, gap=0.11)):
        ax.annotate(
            text, xy=(gb, yy), xytext=((x - math.log10(gb / lo) * per_decade) * 72, 10),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=fs["ratio"], color=INK_2,
        )


def draw_panel(ax, rows, placed, extent, args, cmap, norm, lo, hi, *, show_y: bool,
               fs: dict[str, float], label_x: float):
    """One target ratio's rows, into an axes someone else has already positioned.

    *extent* is the y extent every panel of the figure shares, so a dataset's row is at
    the same height in each; only the leftmost panel names the rows.
    """
    y_of, headers, _ = placed
    prepare_axes(ax, extent, lo, hi, args)

    if show_y:
        for y_head, modality in headers:
            # In the label gutter, so the header never lands on the budget band or a dot.
            # The gap above each group separates it; no rule is drawn across the plot.
            ax.text(label_x, y_head, modality, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=fs["group"], color=INK_2,
                    fontweight="semibold", clip_on=False)

    for i, row in enumerate(rows):
        yy = y_of[i]
        labelled = draw_ladder(ax, row, yy, cmap, norm)
        if show_y:
            ax.text(label_x, yy, row.label, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=fs["row"], color=INK,
                    clip_on=False)
        draw_ratio_labels(ax, labelled, yy, fs)

    finish_axes(ax, fs)


# --- compact layout ----------------------------------------------------------
#: ``--compact`` folds the per-target panels into one: every (dataset, model) keeps its
#: place, and under it sits one lane per target. The footprints do not depend on the
#: guarantee, so the lanes of a group have their dots at exactly the same x and differ
#: only in colour -- which is the comparison the separate panels made the eye do across
#: a whole page. That is also why a group's ratio labels are written once, above its
#: first lane: repeated per lane they would say the same thing three times.
#:
#: ``LANE_STEP`` is the pitch between two lanes of one group, in row units. With no text
#: between them the lanes only have to clear each other's dots.
LANE_STEP = 0.50



@dataclass
class Group:
    """One (dataset, model) of a compact figure: its label, and one lane per target."""

    label: str
    slot: str
    lanes: list[tuple[float, Row | None]]


def row_rank(row: Row) -> tuple[int, int, int]:
    """Where a row sits in :data:`GROUP_ORDER` and :data:`DATASETS` -- build_rows' order."""
    benchmarks = [d.benchmark for d in DATASETS]
    for gi, (_, slots) in enumerate(GROUP_ORDER):
        if row.slot in slots:
            return gi, benchmarks.index(row.dataset.benchmark), slots.index(row.slot)
    return len(GROUP_ORDER), 0, 0


def merge_lanes(panels: list[tuple[float, list[Row]]]) -> list[Group]:
    """Regroup per-target rows into per-(dataset, model) groups of lanes.

    A (dataset, model) that one target lacks still gets its lane, empty, so every group
    has the same lanes in the same order and a missing measurement shows as a gap
    rather than as the lanes shifting up.
    """
    targets = [target for target, _ in panels]
    first: dict[tuple[str, str], Row] = {}
    by_target: dict[tuple[str, str], dict[float, Row]] = {}
    for target, rows in panels:
        for row in rows:
            key = (row.dataset.benchmark, row.slot)
            first.setdefault(key, row)
            by_target.setdefault(key, {})[target] = row
    ordered = sorted(first.items(), key=lambda item: row_rank(item[1]))
    return [
        Group(label=row.label, slot=row.slot,
              lanes=[(t, by_target[key].get(t)) for t in targets])
        for key, row in ordered
    ]


def layout_compact(groups: list[Group], n_lanes: int):
    """Lane positions, group centres, modality headers and the y extent needed.

    The same grammar as :func:`layout` -- a header per modality, the same gaps -- with
    each row opened out into *n_lanes* lanes. A group's last lane is followed by a full
    row pitch, which is the room the next group's ratio labels need above it.
    """
    lane_y: dict[tuple[int, int], float] = {}
    centres: dict[int, float] = {}
    headers: list[tuple[float, str]] = []
    y = 0.0
    last_modality = None
    for gi, group in enumerate(groups):
        modality = SLOTS[group.slot].modality
        if modality != last_modality:
            if last_modality is not None:
                y += MODALITY_GAP
            headers.append((y, modality))
            y += HEADER_TO_ROW
            last_modality = modality
        for li in range(n_lanes):
            lane_y[(gi, li)] = y + li * LANE_STEP
        centres[gi] = y + (n_lanes - 1) * LANE_STEP / 2
        y += (n_lanes - 1) * LANE_STEP + 1.0
    return lane_y, centres, headers, y


def draw_compact_panel(ax, groups, placed, extent, args, cmap, norm, lo, hi, *,
                       fs: dict[str, float], label_x: float, tag_x: float):
    """Every target in one panel: a group per (dataset, model), a lane per target."""
    lane_y, centres, headers, _ = placed
    prepare_axes(ax, extent, lo, hi, args)
    yaxis = ax.get_yaxis_transform()

    for y_head, modality in headers:
        ax.text(label_x, y_head, modality, transform=yaxis, ha="left", va="center",
                fontsize=fs["group"], color=INK_2, fontweight="semibold", clip_on=False)
    # The lane tags are bare numbers; this heads their column once, like a table.
    if headers:
        ax.text(tag_x, headers[0][0], "target p/r", transform=yaxis, ha="right",
                va="center", fontsize=fs["lane"], color=INK_2, fontweight="semibold",
                clip_on=False)

    for gi, group in enumerate(groups):
        # Keyed by footprint: the same configuration sits at the same x in every lane,
        # so one label per x covers the whole group.
        to_label: dict[float, float | str] = {}
        for li, (target, row) in enumerate(group.lanes):
            yy = lane_y[(gi, li)]
            ax.text(tag_x, yy, f"{target:g}", transform=yaxis, ha="right", va="center",
                    fontsize=fs["lane"], color=INK_2, clip_on=False)
            if row is None:
                continue
            for gb, cr in draw_ladder(ax, row, yy, cmap, norm):
                to_label.setdefault(gb, cr)
        ax.text(label_x, centres[gi], group.label, transform=yaxis, ha="left",
                va="center", fontsize=fs["row"], color=INK, clip_on=False)
        draw_ratio_labels(ax, sorted(to_label.items()), lane_y[(gi, 0)], fs)

    finish_axes(ax, fs)


@functools.lru_cache(maxsize=None)
def text_width_in(text: str, size: float, weight: str = "normal") -> float:
    """Width of one *text*, in inches, as the current font sets it; memoised."""
    return measure_in([text], size, weight=weight)


def measure_in(labels, size: float, *, weight: str = "normal") -> float:
    """The widest of *labels*, in inches, as the current font sets them.

    The layout sizes its gutter and its bottom strip from real text widths, which it
    needs before the figure exists -- hence a throwaway figure to measure on.
    """
    labels = list(labels)
    if not labels:
        return 0.0
    probe = plt.figure()
    renderer = probe.canvas.get_renderer()
    widest = max(
        probe.text(0, 0, label, fontsize=size, fontweight=weight)
        .get_window_extent(renderer).width
        for label in labels
    )
    inches = widest / probe.dpi
    plt.close(probe)
    return inches


def keys_width_in(fs: dict[str, float]) -> float:
    """How wide the two keys come out: the longer text, its handle and the gap after."""
    text = measure_in((KEY_UNCOMPRESSED, BUDGET_LABEL), fs["key"])
    # The handle and the gap after it, in font-size units, as the legend draws them.
    return text + (2.0 + 0.6) * fs["key"] / 72


def draw(panels: list[tuple[float, list[Row]]], args, notes: list[str],
         *, experiment: str, metric: str = DEFAULT_METRIC,
         aggregate: str = DEFAULT_AGGREGATE, provenance: str = ""):
    """One panel per target ratio, side by side, under one shared pair of keys.

    *experiment* names the sweep in the file stem.
    """
    edges = tuple(args.speedup_bins)
    if getattr(args, "fit_top_bin", False):
        reached = [sp for _, rows in panels for row in rows for _, _, sp in row.dots]
        if reached:
            edges = fit_top_bin(edges, max(reached))
    cmap, norm, drawn = make_norm(edges)
    fs = fonts(getattr(args, "font_scale", 1.0))
    targets = [target for target, _ in panels]
    # --compact folds several targets into one panel; with one target there is nothing
    # to fold, and the figure is the ordinary single-panel one.
    compact = bool(getattr(args, "compact", False)) and len(panels) > 1
    if compact:
        groups = merge_lanes(panels)
        compact_placed = layout_compact(groups, len(targets))
        placements = []
    else:
        placements = [layout(rows) for _, rows in panels]
    n = 1 if compact else len(panels)
    multi = n > 1

    # One x scale and one y extent for every panel: the footprints do not depend on the
    # guarantee, so a dataset's ladder lands at the same place in each panel and only
    # the colours move between them. That is the whole point of putting them side by side.
    values = [gb for _, rows in panels for row in rows for gb, _, _ in row.dots]
    values += [row.uncompressed_gb for _, rows in panels for row in rows
               if row.uncompressed_gb]
    # Just past the outermost dots, with room for a ratio label over each, rather than
    # rounded out to whole decades, which would leave most of a decade empty at each end.
    lo = min(values) / 1.45
    hi = max(values) * 1.4
    extent = compact_placed[3] if compact else max(e for _, _, e in placements)

    # The gutter is as wide as its text needs: the longest row label or group header,
    # the clearance after it, and in a compact figure the lane tags -- whose column
    # header shares a line with the first group header, so that pair has to fit too.
    all_rows = [row for _, rows in panels for row in rows]
    group_names = list(dict.fromkeys(SLOTS[row.slot].modality for row in all_rows))
    names_w = max(measure_in((row.label for row in all_rows), fs["row"]),
                  measure_in(group_names, fs["group"], weight="semibold"))
    if compact:
        tags_w = measure_in((f"{t:g}" for t in targets), fs["lane"])
        tag_head_w = measure_in(["target p/r"], fs["lane"], weight="semibold")
        first_head_w = measure_in(group_names[:1], fs["group"], weight="semibold")
        text_w = max(names_w + LABEL_CLEAR_IN + tags_w,
                     first_head_w + LABEL_CLEAR_IN + tag_head_w) + TAG_GAP_IN
    else:
        text_w = names_w + LABEL_CLEAR_IN
    gutter_in = LEFT_MARGIN_IN + text_w
    panel_w = PLOT_FRAC * args.width
    label_x = -text_w / panel_w
    gap_x = GAP_X_IN if multi else 0.0
    width = gutter_in + n * panel_w + (n - 1) * gap_x + RIGHT_FRAC * args.width

    span = n * panel_w + (n - 1) * gap_x
    # One strip under the plot: the keys flush left, under the row labels, and the
    # colour bar after them, starting no further left than the plot does. Only if that
    # would squeeze the bar below legibility do the keys take a row of their own.
    # Heights of the text-bearing strips, from the sizes actually in use.
    pt = 1 / 72
    tick_h = (TICK_LEN + TICK_PAD + 1.2 * fs["tick"]) * pt   # ticks and their labels
    cap_h = 1.2 * fs["caption"] * pt
    xaxis_in = tick_h + 0.05 + cap_h + 0.05
    caption_y = tick_h + 0.05 + cap_h / 2           # its centre, below the axis
    bar_h = 0.22
    band_h = (5 + 1.2 * fs["band"]) * pt            # band names hanging under the bar
    bar_block = bar_h + band_h
    keys_h = (2 + 0.45) * 1.2 * fs["key"] * pt      # two lines and the space between
    keys_row_h = 1.2 * fs["key"] * pt               # the same two, side by side
    # A ratio label lifted over a row must clear the dots of the row above.
    row_in = max(ROW_IN, (10 + 1.2 * fs["ratio"]) * pt + 0.09)

    # The strip under the plot spans the whole figure, row-label gutter included: the
    # keys start at its left edge, and the colour bar runs from after them to the right.
    fig_right = gutter_in + span
    # The two keys go on one row whenever the figure is wide enough for it, stacked
    # only when it is not. The colour bar then sits beside them if it still has room
    # for its band names, and takes a full-width row of its own under them if not.
    keys_x = LEFT_MARGIN_IN
    handle_in = (2.0 + 0.6) * fs["key"] * pt
    keys_side_w = (measure_in([KEY_UNCOMPRESSED], fs["key"])
                   + measure_in([BUDGET_LABEL], fs["key"])
                   + 2 * handle_in + 1.8 * fs["key"] * pt)
    keys_one_row = keys_side_w <= fig_right - LEFT_MARGIN_IN
    keys_block_h = keys_row_h if keys_one_row else keys_h
    bar_x = keys_x + (keys_side_w if keys_one_row else keys_width_in(fs)) + 0.40
    bar_w = min(fig_right - bar_x, MAX_BAR_IN)
    # Legible only if every band name fits under its band, with a little air between.
    min_bar = (len(edges) + 1) * (measure_in(band_labels(edges), fs["band"]) + 0.14)
    abreast = bar_w >= min_bar

    # Every panel of an ordinary figure is headed with its target, a lone panel too,
    # since the figure has no title. A compact figure has its targets in the lane-tag
    # column instead.
    show_heads = not compact
    head_in = (0.12 + 1.2 * fs["panel"] * pt + 0.04) if show_heads else 0.0
    top_in = TOP_MARGIN_IN + (PROVENANCE_IN if provenance else 0.0)
    if abreast:
        keys_in = max(keys_block_h, bar_block) + 0.06
    else:
        keys_in = bar_block + 0.12 + keys_block_h + 0.04
    panel_h = row_in * (extent - 1.0 + BOTTOM_PAD + TOP_PAD)
    height = top_in + head_in + panel_h + xaxis_in + keys_in + FOOT_IN

    fig = plt.figure(figsize=(width, height))

    def fx(inches: float) -> float:
        return inches / width

    def fy(inches: float) -> float:
        return inches / height

    panel_top = height - top_in - head_in
    panel_bottom = panel_top - panel_h

    if compact:
        ax = fig.add_axes([fx(gutter_in), fy(panel_bottom), fx(panel_w), fy(panel_h)])
        # Row labels flush left at the gutter edge, as in the ordinary layout; the tags
        # right-aligned just clear of the plot.
        draw_compact_panel(ax, groups, compact_placed, extent, args, cmap, norm, lo, hi,
                           fs=fs,
                           label_x=label_x, tag_x=-TAG_GAP_IN / panel_w)

    for i, ((target, rows), placed) in enumerate(zip(panels, placements)):
        x = gutter_in + i * (panel_w + gap_x)
        if show_heads:
            fig.text(fx(x), fy(panel_top + 0.12), f"target p/r = {target:g}",
                     ha="left", va="bottom", fontsize=fs["panel"], color=INK,
                     fontweight="semibold")
        ax = fig.add_axes([fx(x), fy(panel_bottom), fx(panel_w), fy(panel_h)])
        # Rows are named once, down the left-hand gutter of the first panel.
        draw_panel(ax, rows, placed, extent, args, cmap, norm, lo, hi,
                   show_y=(i == 0), fs=fs, label_x=label_x)

    span_x0 = gutter_in
    span_x1 = span_x0 + span
    centre = (span_x0 + span_x1) / 2

    caption = "size on disk (log scale) · dot labels: compression ratio"
    half = measure_in([caption], fs["caption"]) / 2
    caption_x = min(max(centre, LEFT_MARGIN_IN + half), fig_right - half)
    fig.text(fx(caption_x), fy(panel_bottom - caption_y), caption,
             ha="center", va="center", fontsize=fs["caption"], color=MUTED)

    # The keys belong to the figure, not to a panel: they say the same thing about every
    # panel above them. Centred on the bar together with its band names, so the strip
    # reads as one line.
    if abreast:
        mid = FOOT_IN + keys_in / 2
        bar_y = mid - bar_block / 2 + band_h
        key_loc, key_x, key_y = "center left", keys_x, mid
    else:
        # Too narrow for keys and bar side by side: the bar takes the full width, and
        # the keys a row of their own above it.
        bar_x = LEFT_MARGIN_IN
        bar_w = min(fig_right - LEFT_MARGIN_IN, MAX_BAR_IN)
        bar_y = FOOT_IN + band_h
        key_loc, key_x = "center", (LEFT_MARGIN_IN + fig_right) / 2
        key_y = bar_y + bar_h + 0.12 + keys_block_h / 2

    fig.legend(
        handles=[Line2D([], [], marker="o", linestyle="none",
                        markersize=0.8 * fs["key"], markerfacecolor="none",
                        markeredgecolor="#123f6b", markeredgewidth=1.9,
                        label=KEY_UNCOMPRESSED),
                 Line2D([], [], color=BUDGET_RULE, linestyle=(0, (5, 3)),
                        linewidth=2.0, label=BUDGET_LABEL)],
        loc=key_loc, bbox_to_anchor=(fx(key_x), fy(key_y)),
        frameon=False, fontsize=fs["key"], handletextpad=0.6, labelcolor=INK_2,
        borderpad=0, borderaxespad=0, labelspacing=0.45,
        ncol=2 if keys_one_row else 1, columnspacing=1.8,
    )

    cax = fig.add_axes([fx(bar_x), fy(bar_y), fx(bar_w), fy(bar_h)])
    add_speedup_bar(fig, cax, cmap, norm, drawn, edges,
                    metric=metric, aggregate=aggregate, fs=fs)

    # No title: the figure goes into a paper, whose caption says what it shows. Which
    # metric and aggregate a file holds is in its name. A row drawn from another sweep
    # is still said so on the figure, though, not only in the shell.
    if provenance:
        fig.text(fx(LEFT_MARGIN_IN), 1 - fy(TOP_MARGIN_IN), provenance, ha="left", va="top",
                 fontsize=fs["provenance"], color=MUTED)

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(f"p{slug(f'{target:g}')}" for target, _ in panels)
    # The default metric and aggregate keep the bare stem; the others are marked in the
    # name so figures for different settings do not overwrite each other.
    mark = "" if metric == DEFAULT_METRIC else f"{slug(metric)}_"
    mark += "" if aggregate == DEFAULT_AGGREGATE else f"{slug(aggregate)}_"
    mark += "compact_" if compact else ""
    stem = args.outdir / f"{experiment}_footprint_speedup_{mark}{tag}"
    for fmt in output_formats(args.formats):
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)

    for note in notes:
        print(f"note: {note}", file=sys.stderr)


# --- entry point -------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "experiment", choices=list(EXPERIMENTS),
        help="which sweep under artifacts/kvops to plot",
    )
    parser.add_argument(
        "--target-ratio", type=float, nargs="+", default=[0.5], choices=[0.5, 0.7, 0.9],
        metavar="RATIO",
        help="the precision/recall guarantee(s) to plot (default: 0.5). Several give one "
             "image with a panel each, side by side over a shared x scale and shared keys",
    )
    parser.add_argument(
        "--metric", choices=list(METRICS), default=DEFAULT_METRIC,
        help="which runtime the speedup is measured on (default: execution). "
             "'total' is execution + tuning; 'wall-clock' is the run's end-to-end "
             "elapsed time, the only one that includes configuring",
    )
    parser.add_argument(
        "--aggregate", choices=list(AGGREGATES), default=DEFAULT_AGGREGATE,
        help="how a configuration's queries become one number (default: total). "
             "'total' is a ratio of sums over every query, so untouched queries pull it "
             "toward 1.0; 'movers' restricts that to queries whose runtime the "
             "configuration changed either way, which is not the same as the cache "
             "having been used; "
             "'median' is the middle per-query ratio, 'mean' their arithmetic mean, "
             "which ratios make the most generous, and 'geomean' exp(mean(log r)), "
             "the unbiased average of a set of ratios",
    )
    parser.add_argument(
        "--kvops-dir", type=Path, default=None,
        help="the sweep directory (default: artifacts/kvops/<experiment>)",
    )
    parser.add_argument(
        "--ecommerce-dir", type=Path, default=None,
        help="sweep to read ecommerce from (default: the sweep itself)",
    )
    parser.add_argument("--footprint-dir", type=Path, default=DEFAULT_FOOTPRINT)
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="default: a figures/ beside the sweep, as the footprint plotter does",
    )
    parser.add_argument(
        "--budget-gb", type=float, nargs="+", default=list(DEFAULT_BUDGET_GB),
        metavar="GB",
        help="storage sizes to mark with a dashed rule (default: 100 1000)",
    )
    parser.add_argument(
        "--speedup-bins", type=float, nargs="+", default=list(DEFAULT_BINS),
        help="interior edges of the speedup bands (default: 1.0 1.1 1.2 1.4)",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="with several targets, draw one panel instead of one per target: each "
             "(dataset, model) row opens into one lane per target, in the order "
             "given, with the ratio labels written once per row. No effect with a "
             "single target",
    )
    parser.add_argument(
        "--fit-top-bin", action="store_true",
        help="lower the top speedup band so the darkest colour is not empty "
             "(a figure reaching only 1.37 gets 'x >= 1.3'). Off by default, "
             "because two figures fitted separately no longer share a colour scale",
    )
    parser.add_argument("--width", type=float, default=5.9,
                        help="width of one panel's figure, inches")
    parser.add_argument("--font-scale", type=float, default=1.0,
                        help="multiply every type size by this (default: 1.0)")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()

    if len(args.speedup_bins) < 1 or sorted(args.speedup_bins) != list(args.speedup_bins):
        parser.error("--speedup-bins needs at least one edge, in increasing order")
    if args.kvops_dir is None:
        args.kvops_dir = KVOPS_ROOT / args.experiment
    if args.ecommerce_dir is None:
        ecommerce_sweep = EXPERIMENTS[args.experiment]
        args.ecommerce_dir = KVOPS_ROOT / ecommerce_sweep if ecommerce_sweep else args.kvops_dir
    if not args.kvops_dir.is_dir():
        parser.error(f"no {args.experiment} directory at {args.kvops_dir}")
    if args.outdir is None:
        args.outdir = args.kvops_dir / "figures"
    if not args.footprint_dir.is_dir():
        parser.error(f"no footprint directory at {args.footprint_dir}")

    sizes, notes = footprint(args.footprint_dir)
    if not sizes:
        raise SystemExit(
            f"no kv_cache_footprint_*.csv could be read under {args.footprint_dir}"
        )

    # Keep the order asked for, but a ratio named twice is one panel.
    targets = list(dict.fromkeys(args.target_ratio))

    sources = {}
    provenance = ""
    if args.ecommerce_dir.resolve() != args.kvops_dir.resolve():
        if not args.ecommerce_dir.is_dir():
            parser.error(f"no ecommerce sweep directory at {args.ecommerce_dir}")
        sources[ECOMMERCE_BENCHMARK] = args.ecommerce_dir
        provenance = (
            f"ecommerce read from {args.ecommerce_dir.name}, which sweeps the same "
            f"replacement one slot per modality at a time; every other row from "
            f"{args.kvops_dir.name}"
        )

    panels: list[tuple[float, list[Row]]] = []
    missing: list[str] = []
    for target in targets:
        speed, target_notes = speedups(args.kvops_dir, target, args.metric, sources=sources, aggregate=args.aggregate)
        rows = build_rows(sizes, speed, notes=target_notes) if speed else []
        notes.extend(f"p/r={target:g}: {note}" for note in target_notes)
        if rows:
            panels.append((target, rows))
        else:
            missing.append(f"{target:g}")

    # A ratio that was asked for and cannot be drawn is an error, not a quietly smaller
    # figure: the caller named it, and a panel silently absent is easy to miss.
    if missing:
        for note in notes:
            print(f"note: {note}", file=sys.stderr)
        raise SystemExit(
            f"nothing to plot at target ratio {', '.join(missing)}.\n"
            f"  looked under: {args.kvops_dir / 'merged'}\n"
            f"  a target needs sweep rows whose precision_guarantee and recall_guarantee "
            f"both equal it, and a footprint row for the same (dataset, model, ratio)."
        )

    draw(panels, args, notes, experiment=args.experiment, metric=args.metric,
         aggregate=args.aggregate, provenance=provenance)


if __name__ == "__main__":
    main()
