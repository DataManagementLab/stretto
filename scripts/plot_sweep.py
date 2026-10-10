"""Plot any of the coordinator's sweep experiments: guarantee satisfaction and phase runtime.

One driver for nine shipped experiments, because they differ in exactly one thing - which
column is the x-axis - and agree on everything else. They all read the same merged
schema (``merged/<benchmark>/<split>/<producer>.csv``), all want the same two figures,
and all want them aggregated the same way.

``--experiment`` takes either column - the preset name says what the figure is, the task
id says which run it came from, and the task id is already on the command line inside
``--output-dirs``:

======================  ==================  ==========================================
``--experiment``        or its task id      the axis being compared
======================  ==================  ==========================================
``baselines``           ``base01``          ``approach`` - the optimizers
``modes``               ``mode01``          ``approach`` - one optimizer's search modes
``sample_size``         ``samp01``          the profiling sample size
``operator_count``      ``ops01``           the sweep state, labelled ops + storage
``kv_operator``         ``kvop01``          one KV operator ADDED to the vanilla suite
``adaptive_sampling``   ``adapt01``         single-shot against the adaptive schedule
``ablation``            ``abl01``           Stretto, minus compression, minus optimizer
``reordering``          ``abl02``           Stretto with operator reordering, without
``reorder_only``        ``abl03``           no optimization against reordering alone
======================  ==================  ==========================================

Three figure families:

* ``<prefix>_target_met.pdf`` - per-query ``achieved / target`` as boxes, one facet per
  dataset plus a pooled one, a rule at 1.0. **Scored against silver** on every random
  benchmark, since none of them carries ground truth: the question is whether the
  guarantee held against the system's own best-model reference, which is precisely what
  ``--producer label_reference`` exists to challenge.
* ``<prefix>_breakdown.pdf`` - wall clock split into non-overlapping phases, summed over
  a dataset's queries and stacked. Every guarantee target is in this one figure, as
  adjacent bars within each arm's group, so reading an arm across targets is a glance
  along a group rather than a diff between files. Arm on x, target on the bar colour,
  phase on its hatch - two legends, side by side under the axes.

* ``<prefix>_<metric>.pdf`` (``--figures metrics``) - one scalar per figure over the
  same axis: curves where that axis is numeric (sample size, storage), bars where it is
  not. Which column is x and which metrics are drawn is part of the preset.

Each is written twice, once with every dataset panel and once with the pooled panel
alone (``--panels``), because a paper wants the headline on its own.

**Figures are laid out to a page width** (``--width``, A4 full text width by default;
``--overall-width`` for the single-panel pooled figures, sized so the page lands at half a
column of a two-column one) rather than accumulating one: panels and type shrink as datasets are
added, instead of the figure growing to the 45 inches six comfortable panels would want.

**Cross-dataset aggregation is drawn both ways** (``--pooling``, default ``both``), as two
files distinguished by a ``_geomean``/``_sum`` suffix, because the two answer different
questions and the choice is the reader's:

* ``geomean`` - the bar's height is the geometric mean of the per-dataset totals and its
  composition is each phase's mean share, so the segments still sum to the bar
  (``sweep_frames.geomean_shares``). Scale-free: an arm that costs *k* times another on
  every benchmark shows a ratio of *k*, whatever the benchmarks weigh.
* ``sum`` - the fleet's actual bill (``sweep_frames.sum_overall``), and therefore
  dominated by whichever benchmark is slowest: artwork's totals are an order of magnitude
  above movie's, so a summed bar largely reports artwork.

Only figures that actually hold a pooled panel are written twice; ``operator_count``
(``include_overall=False``), a single-benchmark run and ``--no-overall`` write one
unsuffixed file each. Two things ``--pooling`` deliberately does not reach: the
accuracy metrics (``f1``, ``mean_precision``, ``mean_recall``), which pool by arithmetic
mean under either rule since a sum of F1 scores is not a quantity, and the target-met
figure, whose pooled panel concatenates per-query ``achieved / target`` ratios rather than
combining totals at all.

``--output-dirs`` defaults to ``benchmark_results/<task-id>/merged``, which is where
``run_coordinator.py`` puts a task - so a run with default paths needs only
``--experiment``.

**One arm may be borrowed from another task** (``--reference-from``) and drawn as an extra
category at the right-hand end of the compared axis, in its own colour and under its own
filenames. ``--experiment samp01 --reference-from abl01`` puts the ablation's unoptimized
arm beside every sample size, which is the cost of not optimizing at all. The two tasks
are compared only
where they ran the same query set: that is checked per benchmark, and one that fails drops
out with a warning rather than being drawn.

Examples
--------
    python scripts/plot_sweep.py --experiment base01

    python scripts/plot_sweep.py --experiment operator_count \\
        --output-dirs benchmark_results/ops01/merged --figures breakdown

    # An axis no preset covers: override the pieces rather than adding code.
    python scripts/plot_sweep.py --experiment baselines --compare-by sample_size \\
        --csv-name sample_size.csv --output-dirs benchmark_results/samp01/merged
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from reasondb.evaluation.plotting import apply_default_style
from reasondb.monitor.dimensions import DIMENSION_LABELS
from reasondb.evaluation.sweep_frames import (
    APPROACH_ORDER,
    COMPARISONS,
    DEFAULT_METRICS,
    DEFAULT_RESULTS_ROOT,
    FACET_LABELS,
    METRICS,
    OVERALL,
    BREAKDOWN_PHASE_COLUMNS,
    POOLINGS,
    PRESETS,
    ReferenceArm,
    default_output_dir,
    facet_panels,
    metric_totals,
    pool_metric,
    pooled_overall,
    resolve_preset,
    with_overall_metric,
    dataset_col_order,
    find_sweep_csvs,
    load_sweep,
    per_dataset_totals,
    prepare_sweep_frame,
    reference_arm_rows,
    scores_by_construction,
    with_overall,
)
from reasondb.evaluation import sweep_figures
from reasondb.evaluation.sweep_figures import (
    A4_TEXT_WIDTH_IN,
    POOLED,
    arm_hue,
    plot_metric,
    plot_phase_breakdown_facets,
    plot_phase_breakdown_grouped,
    plot_target_met,
)

logger = logging.getLogger(__name__)

SECONDS_TO_HOURS = 1.0 / 3600.0


def derived_prefix(preset, args, comparison, figures) -> str:
    """The preset's prefix, plus whatever the flags changed about the figure.

    An override that changes what the figure *is* has to change what it is called, or
    ``--experiment base01 --x-column num_semops`` silently replaces the baselines figures
    with a different question's answer - the same collision ``resolve_preset`` keeps
    ``mode01`` and ``base01`` apart to avoid. ``--prefix`` still wins outright.

    Only overrides that reach *these* figures count: ``--layout`` is a breakdown-only
    flag, so naming a metrics figure after it would split one set of files across two
    prefixes for a difference none of them carries.
    """
    parts = [preset.prefix]
    if args.layout and args.layout != preset.layout and "breakdown" in figures:
        parts.append(args.layout)
    if args.compare_by and args.compare_by != preset.comparison:
        parts.append(f"by_{args.compare_by}")
    if args.facet_by:
        parts.append(f"per_{args.facet_by}")
    if args.x_column and args.x_column != (comparison.x_column or ""):
        parts.append(f"vs_{args.x_column}")
    # Narrowing the compared axis changes what the figure *is* as surely as changing the
    # x-column does - "the baselines" and "the baselines without abacus" are two claims -
    # so it has to change the name too, or the second silently overwrites the first.
    if args.exclude_arms:
        parts.append("without_" + "_".join(sorted(args.exclude_arms)))
    elif args.arms:
        parts.append("arms_" + "_".join(args.arms))
    # Widening the axis is as much a different figure as narrowing it, and this one is
    # meant to be read *beside* the plain sweep rather than instead of it - so both have
    # to survive in one directory.
    if getattr(args, "reference_from", None):
        parts.append(f"with_{args.reference_approach}")
    # A second reading of the same data, so it is a second file rather than a redraw of
    # the first: keeping both is the point, and one name cannot hold two figures.
    if getattr(args, "x_scale", "linear") != "linear":
        parts.append(f"x{args.x_scale}")
    if getattr(args, "annotate_minimum", False):
        parts.append("minima")
    return "_".join(parts)


def approach_hues(df, comparison, reference=None):
    """The hue the bars take: one colour, one colour per arm, or ``None``.

    The scheme is one rule (``sweep_figures.ARM_HUES``): the hue is the **approach**.
    Where the compared axis is the approach the drawers derive it themselves; where it is
    anything else - a sample size, a sweep state - the approach is fixed for the whole
    figure and every bar takes its colour, which is what keeps "Stretto is gold" true in
    the figures whose x-axis is not the optimizer.

    A reference arm borrowed from another task is the one thing that puts a *second*
    approach on such an axis, and it is the whole point of drawing it - so the answer
    becomes a mapping: the swept arms keep the sweep's colour and the reference takes its
    own. Deliberately not generalized past that. Two arms of the ablation are both
    ``optim_global`` and differ in their search space, so keying their colour on the
    approach alone would paint the pair whose gap *is* that experiment in one colour;
    those figures still colour by position, exactly as ``ARM_HUE_ALIASES`` says.

    ``None`` where the sweep itself spans several approaches, which is the case the
    drawers' own per-arm colouring already handles.
    """
    if comparison.name == "approach" or "approach" not in df.columns:
        return None
    label = reference.label if reference else None
    swept = df[df["arm"].astype(str) != label] if label else df
    approaches = sorted(swept["approach"].astype(str).unique())
    if len(approaches) != 1:
        return None
    hue = arm_hue(approaches[0], approaches)
    if not reference:
        # A curve on a numeric axis *spans* the arms, so it has no arm to look a mapping
        # up by; the single colour is the only form that reaches it.
        return hue
    reference_hue = arm_hue(reference.approach, APPROACH_ORDER)
    if reference_hue == hue:
        reference_hue = next(
            (h for h in sweep_figures.ARM_HUES if h != hue), reference_hue
        )
        logger.warning(
            "The %r reference shares its approach with the sweep, so it takes the next "
            "free hue rather than the sweep's own.", reference.label,
        )
    return {
        **{arm: hue for arm in swept["arm"].astype(str).unique()},
        label: reference_hue,
    }


def attach_reference(df, arm_order, args, split, benchmarks):
    """Append another task's arm to *df* as an extra category of the compared axis.

    Returns ``(df, arm_order, reference)`` unchanged when no reference was asked for. The
    reference is attached *after* ``--arms``/``--guarantees`` have narrowed the sweep, so
    it is read against exactly the rows that will be drawn and never filtered out by a
    flag that describes the sweep's own axis.
    """
    if not args.reference_from:
        return df, arm_order, None

    reference = ReferenceArm(
        approach=args.reference_approach, label=args.reference_label
    )
    preset = resolve_preset(args.reference_from)
    dirs = args.reference_dirs
    if not dirs:
        derived = default_output_dir(preset, args.reference_from.strip(), args.results_root)
        if derived is None or not derived.is_dir():
            raise SystemExit(
                f"No merged directory for the {args.reference_from!r} reference"
                + (f" at {derived}" if derived else "")
                + "; pass --reference-dirs."
            )
        dirs = [derived]
    paths = find_sweep_csvs(dirs, preset.csv_name, split, benchmarks)
    if not paths:
        raise SystemExit(
            f"No {preset.csv_name} under {[str(d) for d in dirs]} to read the "
            f"{reference.approach!r} arm from."
        )
    logger.info(
        "Reading the %r arm of %s from %d file(s).",
        reference.approach, args.reference_from, len(paths),
    )
    rows = reference_arm_rows(df, load_sweep(paths), reference)
    return (
        pd.concat([df, rows], axis=0, ignore_index=True),
        [*arm_order, reference.label],
        reference,
    )


def select_arms(df, arm_order, keep, drop):
    """Narrow the compared axis to *keep* (in that order) minus *drop*.

    ``--arms`` doubles as the ordering, because "drop abacus and put Stretto first" is one
    intention and splitting it over two flags invites them to disagree. An unknown name is
    an error rather than a silently smaller figure: the arm values are raw CSV strings
    (``fixed_n100``, not "Adaptive"), so a typo is likely and a figure quietly missing a
    bar is the worst way to find out.
    """
    present = set(df["arm"].astype(str))
    if keep:
        unknown = [a for a in keep if a not in present]
        if unknown:
            raise SystemExit(
                f"--arms {unknown} not in this sweep. Available: {sorted(present)}"
            )
        arm_order = list(dict.fromkeys(keep))
    if drop:
        unknown = [a for a in drop if a not in present]
        if unknown:
            logger.warning(
                "--exclude-arms %s not in this sweep (available: %s); ignoring.",
                unknown, sorted(present),
            )
        arm_order = [a for a in arm_order if a not in set(drop)]
    if keep or drop:
        df = df[df["arm"].astype(str).isin(set(arm_order))]
        if df.empty:
            raise SystemExit("No rows left after --arms/--exclude-arms.")
        logger.info("Kept %d arm(s): %s", len(arm_order), arm_order)
    return df, arm_order


def select_guarantees(df, keep):
    """Narrow to the named guarantee targets, by precision value or by full key."""
    if not keep:
        return df
    wanted = set(keep)
    settings = df["guarantee_setting"].astype(str)
    # "0.5" is what a reader would type; "p:0.5_r:0.5" is what the column holds.
    mask = settings.isin(wanted) | df["precision_guarantee"].astype(str).isin(wanted)
    if not mask.any():
        raise SystemExit(
            f"--guarantees {sorted(wanted)} matched nothing. "
            f"Available: {sorted(settings.unique())}"
        )
    logger.info("Kept guarantee target(s): %s", sorted(settings[mask].unique()))
    return df[mask]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Which shipped experiment to draw, by preset name or by its cluster.yaml "
        "task id: baselines/base01, modes/mode01, sample_size/samp01, "
        "operator_count/ops01, adaptive_sampling/adapt01, ablation/abl01, "
        "reordering/abl02, reorder_only/abl03. Sets the CSV "
        "name, the compared axis and the figure prefix; each can be overridden below.",
    )
    parser.add_argument(
        "--output-dirs", type=Path, nargs="+", default=None,
        help="Directories holding <benchmark>/<split>/<csv-name>, i.e. a task's merged/ "
        "directory. Defaults to <results-root>/<task-id>/merged, which is where the "
        "coordinator puts it - so a task that ran with default paths needs only "
        "--experiment. Pass this to read a task from somewhere else, or to pool several.",
    )
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help=f"Root the default --output-dirs is built under (default: "
        f"{DEFAULT_RESULTS_ROOT}), mirroring run_coordinator.py's own --output-dir default.",
    )
    parser.add_argument(
        "--benchmarks", type=str, nargs="+", default=None,
        help="Benchmarks to include (default: every one found under the output dirs).",
    )
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument(
        "--arms", type=str, nargs="+", default=None,
        help="Keep only these values of the compared axis, in this order — e.g. "
        "'--arms optim_global lotus' to drop abacus and put Stretto first. The names are "
        "the raw values the CSV carries (approaches: optim_global, lotus, abacus; sample "
        "sizes: 10, 25, ...; protocols: fixed_n100, adaptive_n160); an unknown one is an "
        "error listing what is available.",
    )
    parser.add_argument(
        "--exclude-arms", type=str, nargs="+", default=None,
        help="Drop these values of the compared axis, keeping the rest in their usual "
        "order. Easier than --arms when you want everything but one.",
    )
    parser.add_argument(
        "--reference-from", type=str, default=None,
        help="Borrow one arm from another experiment and draw it beside this one, as an "
        "extra category at the right-hand end of the compared axis - e.g. "
        "'--experiment samp01 --reference-from abl01', which reads the ablation's "
        "unoptimized arm in as a sixth bar of the sample-size sweep. Named by preset or "
        "task id, exactly as --experiment is. The two tasks must have run the same query "
        "set on a benchmark for it to appear there; one that did not is dropped with a "
        "warning rather than compared. The reference keeps its own approach, hence its own "
        "colour, and the figures are written under their own filenames.",
    )
    parser.add_argument(
        "--reference-approach", type=str, default="no_optim",
        help="Which approach of --reference-from to borrow (default: no_optim, the "
        "ablation's unoptimized arm).",
    )
    parser.add_argument(
        "--reference-label", type=str, default="No optimization",
        help="What that arm is called on the figures. Its x tick breaks at the spaces, "
        "one word per line, so a two-word name costs a line of panel height rather than "
        "shrinking every tick in the figure.",
    )
    parser.add_argument(
        "--reference-dirs", type=Path, nargs="+", default=None,
        help="Where --reference-from's merged CSVs live (default: its own task directory "
        "under --results-root, the same rule --output-dirs follows).",
    )
    parser.add_argument(
        "--guarantees", type=str, nargs="+", default=None,
        help="Keep only these guarantee targets, as the precision value ('0.5 0.9') or "
        "the full key ('p:0.5_r:0.5').",
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=None,
        help="Where to write the PDFs (default: the first --output-dirs entry).",
    )
    parser.add_argument(
        "--csv-name", type=str, default=None,
        help="Override the preset's CSV name, e.g. to draw an older sweep.",
    )
    parser.add_argument(
        "--compare-by", type=str, choices=sorted(COMPARISONS), default=None,
        help="Override which axis becomes the x-axis.",
    )
    parser.add_argument(
        "--facet-by", type=str, default=None, choices=sorted(FACET_LABELS),
        help="Panel on this column instead of the dataset - e.g. num_semops, to read a "
        "comparison against how hard the query is. The benchmarks are then pooled away "
        "inside each panel by the same geometric mean that pools them into the Overall "
        "panel today, and the pooled panel itself is dropped (a geometric mean over "
        "buckets that partition one query set is not a quantity).",
    )
    parser.add_argument(
        "--no-joins", action="store_true",
        help="Drop queries that contain a semantic join.",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=["target-met", "breakdown", "metrics"], default=None,
        help="Which figure families to draw (default: the preset's target-met and "
        "breakdown). 'metrics' adds one figure per --metrics entry - curves where the "
        "compared axis is numeric (sample size, storage), bars where it is not.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        help="Which scalar metrics '--figures metrics' draws. Choices: "
        + ", ".join(sorted(METRICS))
        + f". Default: {' '.join(DEFAULT_METRICS)}.",
    )
    parser.add_argument(
        "--x-column", type=str, default=None,
        help="Numeric column for the metric curves' x-axis, overriding the comparison's "
        "own (sample_size -> sample_size, operator_count -> storage_gb). Pass '' to force "
        "categorical bars.",
    )
    parser.add_argument(
        "--panels", nargs="+", choices=["all", "overall"], default=["all", "overall"],
        help="'all' draws every dataset panel, 'overall' the pooled panel alone. "
        "Both by default, as two files.",
    )
    parser.add_argument(
        "--layout", choices=["facets", "grouped"], default=None,
        help="Override the preset's breakdown layout: 'facets' is one panel per dataset "
        "plus the pooled one, 'grouped' a single axes with the arms nested inside each "
        "guarantee target (the ablation's shape). The two are different questions of the "
        "same totals - 'is the pooled bar one benchmark's story' against 'how does the "
        "gap move with the target' - so both are worth drawing, and the filename records "
        "which is which. Note the phases are measured on the component-times clock and "
        "sum to wall_clock_s, not to total_runtime_s.",
    )
    parser.add_argument(
        "--pooling", choices=[*POOLINGS, "mean", "both"], default="both",
        help="How a panel that stands for several datasets combines them: 'geomean' is "
        "scale-free and reports the system rather than whichever benchmark is slowest, "
        "'sum' is the fleet's actual bill. Both by default, as two files distinguished by "
        "a _geomean/_sum suffix - they answer different questions and the pooled panel "
        "names the rule it used. Figures with no pooled panel are written once, unsuffixed. "
        "'mean' is the third and simplest reading, and only combines with --facet-by: one "
        "plain arithmetic mean over every query the panel holds, with the benchmarks not "
        "combined at all but simply not a grouping key - so a dataset weighs what its "
        "query count is, where geomean weighs every dataset alike.",
    )
    parser.add_argument(
        "--x-scale",
        choices=("linear", "sqrt", "symlog"),
        default="linear",
        help="Spread out a crowded numeric x-axis. 'sqrt' keeps a real quantitative "
        "scale and passes through zero; 'symlog' spreads the low end harder. Anything "
        "but 'linear' writes to its own filename.",
    )
    parser.add_argument(
        "--annotate-minimum",
        action="store_true",
        help="Ring and label the cheapest point of every curve with its runtime, "
        "operator count and footprint. Numeric-x metric figures only; writes to a "
        "'_minima' filename so the plain figure is kept alongside it.",
    )
    parser.add_argument(
        "--no-overall", action="store_true",
        help="Never compute the pooled panel. Pinned on for operator_count, whose "
        "x-axis carries a per-dataset storage footprint that cannot be pooled.",
    )
    parser.add_argument(
        "--height-scale", type=float, default=1.0, metavar="F",
        help="Draw the panels F times their standard height. Type sizes are in points "
        "and do not scale with the figure, so the whole change lands on the axes: 2.0 is "
        "a panel with twice the room for its bars and the same labels around it. Width is "
        "unaffected (--width owns that).",
    )
    parser.add_argument(
        "--legend-columns", type=int, default=None, metavar="N",
        help="Pin every legend to N columns instead of fitting as many as the band "
        "holds. '1' makes each legend a vertical list, which is what a tall panel with "
        "two legends beside each other wants: two short columns read as two legends "
        "where two wide rows read as one band.",
    )
    parser.add_argument(
        "--x-label", type=str, default=None, metavar="TEXT",
        help="Override the x-axis label, or pass an empty string to drop it. The default "
        "is the comparison's own: none where the ticks already name the axis (approach "
        "names do, a sample size in rows does not), and the comparison's name under "
        "--facet-by, where the panel titles are buckets rather than datasets.",
    )
    parser.add_argument(
        "--match-panels", type=int, default=None, metavar="N",
        help="Draw a single-panel figure as one panel of an N-panel row: the same panel "
        "size and the same type as a subplot of the faceted figure it accompanies, "
        "rather than a panel sized from a page of its own. This is what the pooled "
        "'_overall' figures already do against their own row; pass it when a run draws "
        "one dataset (--benchmarks ecommerce_random_large) and the figure has to sit "
        "beside a six-panel one. Defaults to this run's own panel count, which leaves "
        "the figure exactly as it is drawn today.",
    )
    parser.add_argument("--prefix", type=str, default=None, help="Override the filename prefix.")
    parser.add_argument(
        "--width", type=float, default=A4_TEXT_WIDTH_IN,
        help="Total figure width in inches. The default is A4 full text width; every "
        "figure is laid out to fit it, so panels and type shrink as datasets are added "
        "rather than the figure growing.",
    )
    parser.add_argument(
        "--overall-width", type=float, default=None,
        help="Panel slot in inches for the pooled-panel-only figures (default: --width "
        "/ 5). It sizes the panel, not the page: a single panel carries a whole "
        "figure's y label and tick numbers where a faceted panel shares them, so the "
        "page comes out wider than this. The default is set so the page lands at about "
        "half a column of a two-column A4.",
    )
    parser.add_argument(
        "--collapse-targets", dest="collapse_targets", action="store_true", default=None,
        help="Sum the phase totals over the guarantee targets, so each arm is one bar. "
        "On by default for sample_size, whose own axis is already five points long.",
    )
    parser.add_argument(
        "--no-collapse-targets", dest="collapse_targets", action="store_false",
        help="Keep one bar per target even where the preset would collapse them.",
    )
    args = parser.parse_args()

    preset = resolve_preset(args.experiment)
    output_dirs = args.output_dirs
    if not output_dirs:
        derived = default_output_dir(preset, args.experiment.strip(), args.results_root)
        if derived is None:
            raise SystemExit(
                f"The {preset.name!r} preset has no task id to derive a path from; "
                "pass --output-dirs."
            )
        if not derived.is_dir():
            raise SystemExit(
                f"No {derived} to read. That is the default location for this "
                "experiment - pass --output-dirs if the task ran somewhere else, or "
                "extract/merge it first."
            )
        logger.info("Reading %s (default for %s).", derived, args.experiment)
        output_dirs = [derived]
    csv_name = args.csv_name or preset.csv_name
    comparison = COMPARISONS[args.compare_by or preset.comparison]
    figures = args.figures or list(preset.figures)
    layout = args.layout or preset.layout
    # `width_scale` is a property of the *grouped* layout - the ablation's breakdown is a
    # single axes and belongs in one column of a two-column page. Everything else it draws
    # is a facet grid of six panels, which needs the whole text width: at half of it the
    # panels come out 45pt tall with the y ticks of one overlapping the axis of the next.
    width = args.width
    grouped_width = args.width * preset.width_scale
    # This sizes the *panel*, not the page: `_fit_panel` holds the panel and lets the page
    # take whatever the decorations need, and a single panel pays for a y label and a column
    # of tick numbers that six panels in a row share. The divisor is set so the resulting
    # *figure* lands under half a column, which is the quantity a page is laid out in.
    overall_width = (args.overall_width or args.width / 5.0) * preset.overall_width_scale
    sweep_figures.ANNOTATE_MINIMA = bool(args.annotate_minimum)
    sweep_figures.X_SCALE = args.x_scale
    sweep_figures.LEGEND_COLUMNS = args.legend_columns
    # Height is one knob for the whole figure system (`HEIGHT_SCALE`), so asking one run
    # for taller panels is that knob turned for that run rather than a second rule.
    # Decorations keep their point size, so the whole change lands on the axes.
    if args.height_scale != 1.0:
        sweep_figures.HEIGHT_SCALE = sweep_figures.HEIGHT_SCALE * args.height_scale
        logger.info(
            "Panels drawn at %.2f x the standard height for this run.", args.height_scale
        )
    prefix = args.prefix or derived_prefix(preset, args, comparison, figures)
    include_overall = preset.include_overall and not args.no_overall
    collapse_targets = (
        preset.collapse_targets if args.collapse_targets is None else args.collapse_targets
    )
    figure_dir = args.figure_dir or output_dirs[0]

    logger.info("%s", preset.note or preset.name)
    paths = find_sweep_csvs(output_dirs, csv_name, args.split, args.benchmarks)
    if not paths:
        raise SystemExit(
            f"No {csv_name} found under {[str(d) for d in output_dirs]}. "
            "Has the task been merged? A coordinator run writes merged/ only once every "
            "job is terminal - re-run it with --merge-now if it was interrupted."
        )
    logger.info("Loading %d file(s) for %s", len(paths), csv_name)

    df = prepare_sweep_frame(
        load_sweep(paths), comparison, preset.fan_out_guarantee_blind
    )
    df, arm_order = select_arms(df, comparison.order(df), args.arms, args.exclude_arms)
    df = select_guarantees(df, args.guarantees)
    if args.no_joins:
        df = df[df["num_sem_join"].fillna(0) == 0]
    df, arm_order, reference = attach_reference(
        df, arm_order, args, args.split, args.benchmarks
    )
    logger.info("Comparing %s over %s", comparison.name, arm_order)

    # The reference arm is a category, not a value of the swept axis: it has no sample
    # size and no footprint, so there is no numeric x to put it at.
    reference_ticks = {reference.label: reference.tick} if reference else {}

    def comparison_ticks(scope) -> dict:
        return {**comparison.ticks(scope), **reference_ticks}

    if reference:
        if args.x_column is None and comparison.x_column:
            logger.info(
                "Drawing the metric figures as bars: the %r arm has no %s, so it has no "
                "position on a numeric axis. Run without --reference-from for the curves.",
                reference.label, comparison.x_column,
            )
        # "" is the existing spelling of "keep the arms categorical"; the reference is one
        # more arm, so it needs no second mechanism.
        args.x_column = "" if args.x_column is None else args.x_column
        if scores_by_construction(df[df["arm"].astype(str) == reference.label]):
            chosen = list(args.metrics or DEFAULT_METRICS)
            accuracy = [k for k in chosen if k in METRICS and METRICS[k].accuracy]
            dropped = accuracy
            args.metrics = [k for k in chosen if k not in set(accuracy)]
            figures = [f for f in figures if f != "target-met"]
            logger.info(
                "The %r arm scores 1.0 on every query - these benchmarks are scored "
                "against silver, and a silver pass is that arm's own plan - so this run "
                "draws the cost figures alone. Dropped: %s. The accuracy figures are the "
                "same run without --reference-from.",
                reference.label, ", ".join(["target-met", *dropped]),
            )

    # Every arm pools whatever the compared axis does not distinguish. Asked of the arms
    # themselves rather than of the comparison's name: `ablation_arm` is *built* from
    # (step, approach), so it separates the approaches even though it is not called
    # "approach", while `sample_size` on a multi-approach CSV really would average
    # optim_global with lotus and abacus into every bar.
    per_arm = df.groupby(df["arm"].astype(str))["approach"].nunique()
    mixed = sorted(per_arm[per_arm > 1].index)
    if mixed:
        logger.warning(
            "%d arm(s) of the %s axis span several approaches and pool them into one bar: "
            "%s. Compare by approach and put %s on --x-column or --facet-by to keep them "
            "apart.",
            len(mixed),
            comparison.name,
            mixed,
            comparison.x_column or comparison.name,
        )

    panels: List[str] = list(dict.fromkeys(args.panels))
    facet_by = args.facet_by
    if facet_by:
        if comparison.per_dataset:
            raise SystemExit(
                f"--facet-by cannot be combined with the {comparison.name!r} axis: its "
                "ticks are read off each dataset's own cache footprint, which a "
                f"{facet_by} bucket spanning several benchmarks does not have."
            )
        df = facet_panels(df, facet_by)
        if include_overall:
            logger.info(
                "Dropping the pooled panel: a geometric mean over %s buckets is not a "
                "quantity - they partition one query set rather than being independent "
                "measurements of it. Run without --facet-by for that number.",
                facet_by,
            )
        panels, include_overall = ["all"], False
    if not include_overall:
        panels = [p for p in panels if p != "overall"]

    # Whether any panel of these figures stands for more than one dataset - the pooled
    # facet, or (under --facet-by) every facet, since there the benchmarks are pooled away
    # inside each bucket. Where nothing is pooled the two rules cannot differ, so the
    # figure is written once and unsuffixed rather than twice and identically. One
    # benchmark counts as nothing pooled: both rules are then the identity.
    dataset_axis = "benchmark" if facet_by else "dataset"
    n_datasets = df[dataset_axis].nunique() if dataset_axis in df.columns else 1
    pools_datasets = (bool(facet_by) or include_overall) and n_datasets > 1
    poolings = list(POOLINGS) if args.pooling == "both" else [args.pooling]
    if not pools_datasets and len(poolings) > 1:
        logger.info(
            "No panel here pools datasets (one benchmark, --no-overall, or a per-dataset "
            "axis), so --pooling changes nothing; drawing each figure once."
        )
        poolings = poolings[:1]

    def tagged(stem: str, pooling: str, varies: bool = True) -> Path:
        """``figure_dir/<stem>.pdf``, with the pooling suffix where it means something."""
        suffix = f"_{pooling}" if (pools_datasets and varies) else ""
        return figure_dir / f"{stem}{suffix}.pdf"

    written = []

    # One rule for all three families: the axis label is drawn only where the panels
    # cannot supply it. Where the ticks name the axis outright - the approaches do, a
    # sample size in rows does not - a label repeated under every panel says nothing new
    # and costs each figure a row of height. Under --facet-by the panel titles are
    # complexity buckets and the label is the only statement of what x is, so it stays.
    arm_axis_label = (
        "" if comparison.ticks_name_the_axis and not facet_by else comparison.axis_label
    )
    if args.x_label is not None:
        arm_axis_label = args.x_label

    # Hue is the *approach*. Where the compared axis is the approach it varies bar to bar
    # and the drawers derive it themselves; where it is anything else - a sample size, a
    # sweep state, a sampling protocol - the approach is fixed for the whole figure, so
    # every bar takes its hue and only the target varies the brightness. Without it a figure
    # whose x-axis is not the optimizer falls back to colouring the *phases*, so "Stretto is
    # gold" stops holding exactly where the reader has least else to go on.
    # The pooled single-panel figures are drawn narrow, and `_facet_geometry` reads a panel's
    # height off its *width* - so one narrow panel comes out ~40% taller than the six-panel
    # row it accompanies and the pair does not line up on a page. Telling the pooled variant
    # how many panels the full figure has makes it
    # borrow that height instead of deriving its own.
    full_panels = n_datasets + (1 if include_overall and not facet_by else 0)
    # A figure holding one panel is a *cut-out* of the faceted one - drawn at that panel's
    # size and in its type (`_fit_panel`), so the two line up on a page. The pooled panel
    # is the case that always was one; a run narrowed to a single dataset is the same
    # figure with a different panel in it, and --match-panels is how it borrows the row it
    # has to match, since a narrowed run cannot know how many panels the wide one had.
    matched_panels = args.match_panels or full_panels
    single_panel = full_panels == 1 and not facet_by

    def panel_geometry(panel: str) -> Dict[str, Any]:
        """``width_in``/``match_panels`` for a figure drawing *panel*."""
        cut_out = panel == "overall" or single_panel
        return {
            "width_in": overall_width if cut_out else width,
            "match_panels": matched_panels if cut_out else None,
            "match_width": width,
        }

    fixed_hue = approach_hues(df, comparison, reference)
    if fixed_hue:
        logger.info("Hue is the approach that produced the bar: %s.", fixed_hue)

    # The borrowed arm is the last tick of the axis, so it has the panel's gutter to its
    # right where every other tick has a neighbour - it may spill over its slot, and must
    # not drag the whole figure's type down to the size that would contain it.
    overflowing_ticks = [reference.label] if reference else []

    if "target-met" in figures:
        for panel in panels:
            suffix = "_overall" if panel == "overall" else ""
            written.append(
                plot_target_met(
                    df,
                    arm_order=arm_order,
                    tick_text=comparison_ticks(df) if not comparison.per_dataset else None,
                    out_path=figure_dir / f"{prefix}_target_met{suffix}.pdf",
                    panels=panel,
                    xlabel=arm_axis_label,
                    **panel_geometry(panel),
                    # Pooling here concatenates per-query ratios, which stays meaningful
                    # over complexity buckets; only its name has to change.
                    pooled_label="All queries" if facet_by else POOLED,
                    # --no-overall asks for no pooled panel anywhere, this family
                    # included; --facet-by keeps it, since a concatenation of ratios over
                    # buckets that partition one query set is still a distribution.
                    include_pooled=not args.no_overall,
                )
            )

    if "breakdown" in figures:
        group_cols = ("dataset",) if collapse_targets else ("dataset", "guarantee_setting")
        if facet_by:
            group_cols = ("benchmark", *group_cols)
        # A panel that stands for a *bucket* rather than for a dataset holds however many
        # queries the generator drew at that shape - 120 at two semantic operators against
        # 90 at four - so a summed bar reports the bucket's size as much as its cost, and
        # the trend across panels is partly the query counts. Per query is the same choice
        # `METRICS` offers as `mean_total_runtime`, applied to the stacked phases.
        per_query = bool(facet_by)
        if per_query:
            logger.info(
                "Faceting on %s, so the breakdown is per query in seconds rather than "
                "summed in hours: the buckets hold different numbers of queries.",
                facet_by,
            )
        # Two lines, because a y label is set in points and rotated: "Runtime per query
        # [s]" is longer than the axes of a column-width panel are tall, and what overflows
        # a short panel is the label rather than the bars.
        breakdown_ylabel = "Runtime per\nquery [s]" if per_query else "Runtime [h]"
        per_dataset = per_dataset_totals(
            df, "arm", value_cols=BREAKDOWN_PHASE_COLUMNS, group_cols=group_cols,
            scale=1.0 if per_query else SECONDS_TO_HOURS,
            agg="mean" if per_query else "sum",
        )
        if collapse_targets:
            # One bar per arm, holding every target's work. The guarantee column has to
            # survive as a single value: it is what the drawers group the bars by, and the
            # pooling is done within it.
            per_dataset["guarantee_setting"] = "all targets"
            logger.info("Summed the phase totals over %d guarantee target(s).",
                        df["guarantee_setting"].nunique())

        if comparison.per_dataset:
            def order_for(dataset: str) -> List[str]:
                return comparison.order(df[df["dataset"] == dataset])

            def ticks_for(dataset: str) -> dict:
                # The pooled panel has no dataset of its own to read a footprint
                # from; fall back to the pooled labelling.
                scope = df if dataset == OVERALL else df[df["dataset"] == dataset]
                return comparison_ticks(scope)
        else:
            shared_order = arm_order
            shared_ticks = comparison_ticks(df)

            def order_for(dataset: str) -> List[str]:
                return shared_order

            def ticks_for(dataset: str) -> dict:
                return shared_ticks

        # The per-dataset totals are the same frame under either rule; only the pooled
        # rows built from them differ, so the sum is another pass over this loop rather
        # than a second read of the CSVs.
        for pooling in poolings:
            totals = per_dataset
            if facet_by and pooling == "mean":
                # The simplest reading of a bucket: every query in it, averaged once. The
                # benchmarks are not pooled - they are not a key - so a benchmark weighs
                # what it drew, and there is no second aggregation to explain.
                totals = per_dataset_totals(
                    df, "arm", value_cols=BREAKDOWN_PHASE_COLUMNS,
                    group_cols=tuple(c for c in group_cols if c != "benchmark"),
                    scale=1.0 if per_query else SECONDS_TO_HOURS,
                    agg="mean" if per_query else "sum",
                )
                if collapse_targets:
                    totals["guarantee_setting"] = "all targets"
            elif facet_by:
                # Pool the benchmarks away *inside* each panel, by the rule the pooled
                # facet would use across panels - the two must not drift apart.
                #
                # The keys come off the frame rather than from `group_cols`, because
                # `collapse_targets` *adds* `guarantee_setting` back above after grouping
                # without it - and a key dropped here is a column the drawer then cannot
                # find.
                value_cols = [
                    c for c in BREAKDOWN_PHASE_COLUMNS if c in totals.columns
                ]
                pool_keys = [
                    c for c in totals.columns
                    if c not in {"benchmark", "arm", *value_cols}
                ]
                totals = pooled_overall(
                    totals,
                    "arm",
                    value_cols,
                    group_cols=pool_keys,
                    dataset_col="benchmark",
                    label="pooled",
                    how=pooling,
                ).drop(columns=["benchmark"])
            elif include_overall:
                totals = with_overall(
                    totals, "arm", value_cols=BREAKDOWN_PHASE_COLUMNS, how=pooling
                )

            if layout == "grouped":
                # The ablation reads its three arms against each other at every target at
                # once, so the guarantee is the x-axis rather than one figure per value.
                pooled = totals[totals["dataset"] == OVERALL]
                if pooled.empty:
                    logger.info(
                        "No pooled rows (a single benchmark, or --no-overall); drawing "
                        "the grouped figure per dataset instead."
                    )
                    for dataset in dataset_col_order(totals):
                        written.append(
                            plot_phase_breakdown_grouped(
                                totals[totals["dataset"] == dataset],
                                arm_order=arm_order,
                                tick_text=comparison_ticks(df),
                                out_path=tagged(
                                    f"{prefix}_breakdown_{dataset}", pooling
                                ),
                                width_in=grouped_width,
                                ylabel=breakdown_ylabel,
                            )
                        )
                else:
                    written.append(
                        plot_phase_breakdown_grouped(
                            pooled,
                            arm_order=arm_order,
                            tick_text=comparison_ticks(df),
                            out_path=tagged(f"{prefix}_breakdown", pooling),
                            width_in=grouped_width,
                            ylabel=breakdown_ylabel,
                            # This whole axes *is* the pooled panel, so the rule goes on
                            # the y label - there are no facet titles to carry it.
                            pooling=pooling,
                        )
                    )
            else:
                # Every guarantee target in one figure, as adjacent bars within each arm's
                # group - not a file per target. Reading an arm across targets is the
                # point, and that is a glance along a group rather than a diff between
                # PDFs.
                for panel in panels:
                    suffix = "_overall" if panel == "overall" else ""
                    written.append(
                        plot_phase_breakdown_facets(
                            totals,
                            hue_override=fixed_hue,
                            overflowing_ticks=overflowing_ticks,
                            order_for=order_for,
                            ticks_for=ticks_for,
                            out_path=tagged(f"{prefix}_breakdown{suffix}", pooling),
                            panels=panel,
                            xlabel=arm_axis_label,
                            ylabel=breakdown_ylabel,
                            **panel_geometry(panel),
                            share_y=bool(facet_by),
                            pooling=pooling,
                        )
                    )

    if "metrics" in figures:
        chosen = args.metrics or list(DEFAULT_METRICS)
        unknown = [m for m in chosen if m not in METRICS]
        if unknown:
            raise SystemExit(
                f"--metrics {unknown} unknown. Available: {sorted(METRICS)}"
            )
        x_column = comparison.x_column if args.x_column is None else (args.x_column or None)
        # `comparison.x_label` describes the comparison's *own* x, so an override needs its
        # own name - otherwise `--x-column num_semops` on an axis that ships no numeric x
        # (the ablation) draws an unlabelled one.
        x_label = comparison.x_label
        if x_column and x_column != comparison.x_column:
            x_label = DIMENSION_LABELS.get(x_column, x_column)
        # All-NA as well as absent: `load_sweep` creates num_semops on a CSV that lacks it,
        # so presence alone does not mean there is anything to plot.
        if x_column and (x_column not in df.columns or df[x_column].isna().all()):
            logger.warning("No usable %s column; drawing %s as bars.", x_column, chosen)
            x_column = None

        metric_group_cols = (
            ("benchmark", "dataset", "guarantee_setting")
            if facet_by
            else ("dataset", "guarantee_setting")
        )
        for key in chosen:
            metric = METRICS[key]
            per_dataset = metric_totals(
                df, "arm", metric, group_cols=metric_group_cols, x_column=x_column
            )
            if per_dataset.empty:
                continue
            # An accuracy pools by arithmetic mean under either rule (`Metric.pool_rule`),
            # so drawing it twice would write the same figure under two names, one of them
            # claiming a sum of F1 scores.
            metric_poolings = poolings if metric.varies_with_pooling else poolings[:1]
            if len(metric_poolings) < len(poolings):
                logger.info(
                    "%s pools by arithmetic mean whichever --pooling is asked for; "
                    "writing it once.", key,
                )
            for pooling in metric_poolings:
                totals = per_dataset
                if facet_by and pooling == "mean":
                    # The plain reading, as in the breakdown: one arithmetic mean over
                    # every query the panel holds. The benchmarks are not combined - they
                    # are simply not a grouping key - so a benchmark weighs what it drew.
                    totals = metric_totals(
                        df, "arm", metric,
                        group_cols=("dataset", "guarantee_setting"),
                        x_column=x_column,
                    )
                elif facet_by:
                    totals = pool_metric(
                        totals,
                        "arm",
                        metric,
                        group_cols=("dataset", "guarantee_setting"),
                        x_column=x_column,
                        dataset_col="benchmark",
                        label="pooled",
                        how=pooling,
                    ).drop(columns=["benchmark"])
                elif include_overall:
                    totals = with_overall_metric(
                        totals, "arm", metric, x_column=x_column, how=pooling
                    )
                for panel in panels:
                    suffix = "_overall" if panel == "overall" else ""
                    written.append(
                        plot_metric(
                            totals,
                            metric=metric,
                            hue_override=fixed_hue,
                            overflowing_ticks=overflowing_ticks,
                            arm_order=arm_order,
                            tick_text=(
                                comparison_ticks(df) if not comparison.per_dataset else None
                            ),
                            out_path=tagged(
                                f"{prefix}_{key}{suffix}",
                                pooling,
                                varies=metric.varies_with_pooling,
                            ),
                            panels=panel,
                            x_column=x_column,
                            xlabel=x_label if x_column else arm_axis_label,
                            **panel_geometry(panel),
                            # An explicit --x-column is a global quantity by construction;
                            # otherwise the comparison says whether its own x is comparable
                            # across panels.
                            share_x=True if args.x_column else comparison.x_shared,
                            share_y=bool(facet_by),
                            pooling=pooling,
                        )
                    )

    written = [p for p in written if p is not None]
    logger.info("Wrote %d figure(s) to %s", len(written), figure_dir)


if __name__ == "__main__":
    apply_default_style()
    main()
