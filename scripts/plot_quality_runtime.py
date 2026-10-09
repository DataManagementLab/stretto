import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import argparse

# ==========================================
# Configuration and Styling for IEEE Paper
# ==========================================
DATASETS = {
    "movie": "Movie (text)",
    "email": "Email (text)",
    "rotowire": "Rotowire (text)",
    "artwork": "Artwork (image)",
}

# Color/marker encode SIZE (large = gold/square, small = blue/circle). Each family
# gets its own subplot (row), so families are separated by layout rather than by
# color; within a row the large/small models of that family are gold/blue.
MODELS = {
    "70B": {"label": "Large Model", "color": "#D49A25", "marker": "s"},
    "8B": {"label": "Small Model", "color": "#1E486D", "marker": "o"},
}

# registry size_class -> visual (color/marker) slot key above
SIZE_TO_SLOT = {"large": "large", "small": "small"}

# --compact-plots collapses the family rows into one panel per dataset, so colour
# has to carry family identity and the line style carries model size. Assigned in
# FAMILY_ORDER and never cycled. Validated as a set against a white surface on the
# *all-pairs* list — lines from any two families can cross anywhere in a shared
# panel — giving worst CVD dE 9.2 (>= 8 target) and worst normal-vision dE 16.3
# (>= 15 floor). Aqua sits at 2.82:1 contrast, just under 3:1, so the legend
# carries the identification.
FAMILY_COLORS = {
    "qwen": "#2a78d6",     # blue
    "mistral": "#eb6834",  # orange
    "llama": "#1baf7a",    # aqua
    "llava": "#4a3aa7",    # violet
}
UNKNOWN_FAMILY_COLOR = "#52514e"

# Model size is the second encoding in a compact panel: solid vs dashed line.
SIZE_LINESTYLES = {"large": "-", "small": "--"}

# Row order for the family grid; unknown families are appended alphabetically.
FAMILY_ORDER = ["qwen", "mistral", "llama", "llava"]

# Which minor ticks --show-minor-ticks keeps (and labels) on the log X axis,
# as multiples of each decade. Widen to (2, 3, 5, 7) for a denser axis.
MINOR_TICK_SUBS = (2, 5)

# Point label for the uncompressed full-prefill runs (no stored KV cache).
VANILLA_LABEL = "vanilla"

# Results are laid out as:
#   {results_dir}/{extract,filter}_stats/{dataset}_random/{press}/{split}/{method}/
#       {method}_vs_{reference}_silver_metrics.csv
# The two op-type roots map onto the --skip-extract / --skip-filter flags.
OP_ROOTS = {"extract": "extract_stats", "filter": "filter_stats"}

# IEEE Publication Styling Settings. FIG_WIDTH is for a 3-column row; per-panel
# width/height are derived from it so a family x dataset grid keeps panel size.
FIG_WIDTH = 9.55
FIG_HEIGHT = 2.5
PANEL_WIDTH = FIG_WIDTH / 3
PANEL_HEIGHT = FIG_HEIGHT

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
    }
)


# ==========================================
# Method / ratio / family parsing
# ==========================================
def _registry_maps():
    """Return (specs_by_longest_prefix, tag_to_compression_ratio).

    Both come straight from reasondb.config.model_registry so the plot stays
    consistent with how methods are named when the results are generated.
    """
    from reasondb.config.model_registry import (
        ModelRegistry,
        COMPRESSION_RATIOS,
        _cr_to_tag,
    )

    registry = ModelRegistry.get()
    tag_to_cr = {_cr_to_tag(cr): cr for cr in COMPRESSION_RATIOS}
    specs = [registry.spec_by_key(k) for k in registry.all_keys(modality=None)]
    # Longest prefix first so e.g. 'kvQwen72B' is preferred over any shorter one.
    specs.sort(key=lambda s: len(s.method_prefix), reverse=True)
    return specs, tag_to_cr


def _parse_method(method, specs, tag_to_cr):
    """Map a method dir name to (ModelSpec, ratio_label).

    Handles both namings produced by run_benchmark_single_operator.py:
      - KV-cache methods:  'kvQwen72B08'            -> (spec, '0.80')
      - vanilla methods:   'vanillaMistralSmall24B' -> (spec, 'vanilla')
        (full prefill, no stored caches: 'vanilla' + method_prefix without 'kv')

    Returns None for methods that are neither (e.g. 'gpt', 'silver').
    """
    for spec in specs:
        prefix = spec.method_prefix
        if method.startswith(prefix):
            tag = method[len(prefix):]
            if tag in tag_to_cr:
                return spec, f"{tag_to_cr[tag]:.2f}"
        if method == "vanilla" + prefix.removeprefix("kv"):
            return spec, VANILLA_LABEL
    return None


def _series_order(point):
    """Sort key for the points a line connects: vanilla first, then ascending
    compression ratio. Runtime is not monotonic in the ratio, so ordering by
    runtime would connect the dots out of sweep order."""
    if point["ratio_label"] == VANILLA_LABEL:
        return (0, 0.0)
    return (1, float(point["ratio_label"]))


def _registry_families(modality=None):
    """Model families the registry knows, in FAMILY_ORDER. modality=None is the
    full set (the --model choices); "text" is the set used to decide which
    vision models need a 'VL' suffix to stay distinguishable."""
    from reasondb.config.model_registry import ModelRegistry

    registry = ModelRegistry.get()
    families = {registry.spec_by_key(k).family for k in registry.all_keys(modality=modality)}
    return _ordered_families(families)


def _model_display_name(family, modality, text_families):
    """Name the models a panel actually plots.

    A family with both text and vision models needs the two told apart, so its
    vision panels read 'QwenVL' / 'MistralVL' against 'Qwen' / 'Mistral'. A
    vision-only family is already unambiguous and keeps its bare name ('Llava').
    """
    if modality == "vision" and family in text_families:
        return f"{family.title()}VL"
    return family.title()


def _panel_modality(panel):
    """Modality of the models in a panel ('text'/'vision'); text if it is empty."""
    for slot in ("large", "small"):
        for point in panel.get(slot, []):
            return point["modality"]
    return "text"


def _pack_family_rows(plot_data, ds_cols):
    """Lay the families out in rows, sharing a row where their columns do not clash.

    One row per family leaves whole panels empty whenever a family has no results
    for a dataset — llama is text-only so its Artwork panel is blank, llava is
    vision-only so its Movie/Email/Rotowire panels are. Those two are exactly
    complementary, so packing them into one row drops a row of empty panels.

    Returns a list of {dataset_key: family}; a dataset missing from a row means
    that panel is unused and gets hidden.
    """
    rows = []
    families = _ordered_families({fam for ds in plot_data for fam in plot_data[ds]})
    for family in families:
        columns = {
            ds for ds in ds_cols if any(plot_data[ds].get(family, {}).values())
        }
        if not columns:
            continue
        # First row whose columns this family does not collide with, else a new one.
        row = next((r for r in rows if not columns & r.keys()), None)
        if row is None:
            row = {}
            rows.append(row)
        row.update(dict.fromkeys(columns, family))
    return rows


def _ordered_families(families):
    """Known families first (FAMILY_ORDER), then any others alphabetically."""
    ordered = [f for f in FAMILY_ORDER if f in families]
    ordered += sorted(f for f in families if f not in FAMILY_ORDER)
    return ordered


# ==========================================
# Data Processing
# ==========================================
def load_and_process_data(args):
    specs, tag_to_cr = _registry_maps()

    # Columns to keep (--dataset) and rows to keep (--model); None means all.
    datasets = [ds for ds in DATASETS if args.dataset is None or ds in args.dataset]

    # plot_data[dataset][family][slot] = [ {ratio_label, runtime, f1}, ... ]
    plot_data = {ds: {} for ds in datasets}

    # Which op-type roots to scan, honoring --skip-extract / --skip-filter.
    op_names = []
    if not args.skip_extract:
        op_names.append("extract")
    if not args.skip_filter:
        op_names.append("filter")

    # dataset_key -> method -> [DataFrame, ...]  (extract and/or filter CSVs)
    frames = {ds: {} for ds in datasets}

    for op in op_names:
        stats_root = os.path.join(args.results_dir, OP_ROOTS[op])
        if not os.path.isdir(stats_root):
            continue

        for dataset_dir in sorted(os.listdir(stats_root)):
            # 'email_random' -> 'email'
            ds_key = (
                dataset_dir[: -len("_random")]
                if dataset_dir.endswith("_random")
                else dataset_dir
            )
            if ds_key not in plot_data:
                continue

            ds_root = os.path.join(stats_root, dataset_dir)
            # Walk down through the {press}/{split}[/{method}] levels and pick up
            # every per-method metrics CSV. The method is the filename prefix
            # before '_vs_' (works whether or not there is a per-method subdir).
            for root, _, files in os.walk(ds_root):
                for fn in files:
                    if not (fn.endswith("_metrics.csv") and "_vs_" in fn):
                        continue
                    method = fn.split("_vs_")[0]
                    frames[ds_key].setdefault(method, []).append(
                        pd.read_csv(os.path.join(root, fn))
                    )

    for ds_key, methods in frames.items():
        for method, dfs in methods.items():
            parsed = _parse_method(method, specs, tag_to_cr)
            if parsed is None:
                continue
            spec, ratio_label = parsed
            if args.model is not None and spec.family not in args.model:
                continue
            slot = SIZE_TO_SLOT.get(spec.size_class)
            if slot is None:
                continue

            combined_df = pd.concat(dfs, ignore_index=True)

            if args.average == "median":
                calc_runtime = combined_df["execution_cost_runtime"].median()
                calc_f1 = combined_df["f1_score"].median()
            else:
                calc_runtime = combined_df["execution_cost_runtime"].mean()
                calc_f1 = combined_df["f1_score"].mean()

            fam = plot_data[ds_key].setdefault(spec.family, {"large": [], "small": []})
            fam[slot].append(
                {
                    "ratio_label": ratio_label,
                    "runtime": calc_runtime,
                    "f1": calc_f1,
                    # Kept per point so a panel can name the model it actually
                    # plots: the 'qwen' row is Qwen on text, QwenVL on images.
                    "modality": spec.modality,
                }
            )

    # Order each (family, slot) series the way its line is drawn: vanilla first,
    # then by increasing compression ratio.
    for ds in plot_data:
        for fam in plot_data[ds]:
            for slot in plot_data[ds][fam]:
                plot_data[ds][fam][slot].sort(key=_series_order)

    # Short summary so it is easy to check against what is on disk.
    found_any = False
    for ds in plot_data:
        parts = []
        for fam, slots in plot_data[ds].items():
            for slot in ("large", "small"):
                if slots[slot]:
                    parts.append(f"{fam.title()} {slot}={len(slots[slot])}")
                    found_any = True
        if parts:
            print(f"{ds}: " + ", ".join(parts))
    if not found_any:
        # Distinguish "nothing on disk" from "the filters excluded everything".
        active = [
            f"--{name} {' '.join(getattr(args, name))}"
            for name in ("dataset", "model")
            if getattr(args, name) is not None
        ]
        suffix = f" matching {' '.join(active)}" if active else ""
        print(f"Warning: no metrics CSVs found under '{args.results_dir}'{suffix}.")

    return plot_data


# ==========================================
# Plotting
# ==========================================
def gold_point(panel):
    """The gold operator of a (family, dataset) panel: its large model at full
    prefill. Falls back to the least-compressed large config, then to the small
    model, so a family whose vanilla run is missing still has a reference (points
    arrive vanilla-first, then by increasing compression)."""
    for slot in ("large", "small"):
        points = panel.get(slot, [])
        for point in points:
            if point["ratio_label"] == VANILLA_LABEL:
                return point
        if points:
            return points[0]
    return None


def _x_values(points, gold):
    """Panel x values: absolute runtimes, or speedups over the gold operator."""
    if gold is None or gold["runtime"] <= 0:
        return [p["runtime"] for p in points]
    return [gold["runtime"] / p["runtime"] if p["runtime"] > 0 else 0.0
            for p in points]


def _x_axis_label(args, label_prefix):
    if args.speedup:
        return "Speedup over gold operator"
    return f"{label_prefix} Exec. Runtime\n(sec., log scale)"


def _draw_panel(ax, panel, adjustments, dataset_key, speedup=False):
    """Draw one family x dataset panel (large + small series)."""
    # --- Model points, one solid line per size slot ---
    gold = gold_point(panel) if speedup else None
    slot_to_style = {"large": MODELS["70B"], "small": MODELS["8B"]}
    for slot in ("large", "small"):
        pts = panel.get(slot, [])
        if not pts:
            continue
        style = slot_to_style[slot]

        x_vals = _x_values(pts, gold)
        y_vals = [p["f1"] for p in pts]
        labels = [p["ratio_label"] for p in pts]

        if slot == "large":
            base_x_off, base_y_off, h_align, v_align = -4, 4, "right", "bottom"
        else:
            base_x_off, base_y_off, h_align, v_align = 4, -4, "left", "top"

        ax.plot(x_vals, y_vals, color=style["color"], marker=style["marker"], zorder=2)

        for x, y, lbl in zip(x_vals, y_vals, labels):
            lookup_key = (str(dataset_key).strip(), str(slot).strip(), str(lbl).strip())
            adj_x, adj_y = adjustments.get(lookup_key, (0.0, 0.0))
            ax.annotate(
                lbl,
                xy=(x, y),
                xytext=(base_x_off + adj_x, base_y_off + adj_y),
                textcoords="offset points",
                fontsize=7,
                color=style["color"],
                fontweight="semibold",
                ha=h_align,
                va=v_align,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=4,
            )


def _size_style(slot):
    """Marker/label styling for a size slot ('large' -> the 70B visual slot)."""
    return MODELS["70B"] if slot == "large" else MODELS["8B"]


def _draw_compact_panel(ax, families, speedup=False):
    """Draw every family of one dataset into a single panel (--compact-plots).

    Colour identifies the family and the line style the model size, so the two
    encodings stay readable on top of each other. The per-point ratio labels the
    per-family panels carry are dropped here: eight lines' worth of them in one
    panel is unreadable.
    """
    for family in _ordered_families(families):
        color = FAMILY_COLORS.get(family, UNKNOWN_FAMILY_COLOR)
        # Each family is measured against its own gold operator.
        gold = gold_point(families[family]) if speedup else None
        for slot in ("large", "small"):
            points = families[family].get(slot, [])
            if not points:
                continue
            ax.plot(
                _x_values(points, gold),
                [p["f1"] for p in points],
                color=color,
                linestyle=SIZE_LINESTYLES[slot],
                marker=_size_style(slot)["marker"],
                zorder=2,
            )


def _compact_legend_handles(plot_data, text_families):
    """Legend for a compact figure: one entry per family, one per model size.

    A family plotted as both text and vision models is named for both ('Qwen /
    QwenVL'), since one colour covers them across panels.
    """
    names, sizes = {}, set()
    for ds in plot_data:
        for family, slots in plot_data[ds].items():
            if not any(slots.values()):
                continue
            display = _model_display_name(family, _panel_modality(slots), text_families)
            if display not in names.setdefault(family, []):
                names[family].append(display)
            sizes.update(slot for slot in ("large", "small") if slots[slot])

    handles = [
        Line2D([], [], color=FAMILY_COLORS.get(family, UNKNOWN_FAMILY_COLOR),
               linestyle="-", label=" / ".join(names[family]))
        for family in _ordered_families(names)
    ]
    handles += [
        Line2D([], [], color="0.35", linestyle=SIZE_LINESTYLES[slot],
               marker=_size_style(slot)["marker"], label=_size_style(slot)["label"])
        for slot in ("large", "small")
        if slot in sizes
    ]
    return handles


def _style_panel_axes(ax, dataset_key, args):
    """X axis, tick labelling, spines and grid — shared by both layouts.

    Runtime spans orders of magnitude and gets a log axis; speedup is a ratio
    anchored at 1x over a much narrower range, so it stays linear and the gaps
    between compression ratios read at their true size.
    """
    suffix = "x" if args.speedup else ""
    tick_label = ticker.FuncFormatter(lambda y, _: f"{y:g}{suffix}")

    if args.speedup:
        ax.xaxis.set_major_formatter(tick_label)
        if dataset_key in args.show_minor_ticks:
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_minor_formatter(tick_label)
    else:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(tick_label)
        if dataset_key in args.show_minor_ticks:
            # Thin the minor ticks to the 2x/5x of each decade, then label every
            # one of them. Labelling matplotlib's default minor ticks (2..9 per
            # decade) smears them together on a panel this narrow.
            ax.xaxis.set_minor_locator(
                ticker.LogLocator(base=10, subs=MINOR_TICK_SUBS)
            )
            ax.xaxis.set_minor_formatter(tick_label)
        else:
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6, zorder=1)


def _draw_compact_figure(plot_data, args, ds_cols, text_families, label_prefix):
    """One panel per dataset, every family drawn together (--compact-plots)."""
    fig, axes = plt.subplots(
        1,
        len(ds_cols),
        figsize=(PANEL_WIDTH * len(ds_cols), PANEL_HEIGHT),
        squeeze=False,
    )

    for c, dataset_key in enumerate(ds_cols):
        ax = axes[0][c]
        _draw_compact_panel(ax, plot_data[dataset_key], speedup=args.speedup)

        ax.set_title(DATASETS[dataset_key])
        ax.set_xlabel(_x_axis_label(args, label_prefix))
        if c == 0:
            ax.set_ylabel(f"{label_prefix} F1 Score")
        _style_panel_axes(ax, dataset_key, args)

    # One figure-level legend under the row: six-odd entries would cover the data
    # if they sat inside a panel.
    handles = _compact_legend_handles(plot_data, text_families)
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(handles), 6),
            frameon=True,
            edgecolor="0.8",
            fancybox=False,
        )


def _save_figure():
    plt.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.0)
    output_filename = "quality_runtime_results.pdf"
    plt.savefig(output_filename, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_filename}")


def generate_plot(plot_data, args, adjustments_file="offset_adjustments.csv"):
    # Load manual offset adjustments if the file exists
    adjustments = {}
    if os.path.exists(adjustments_file):
        try:
            adj_df = pd.read_csv(adjustments_file, dtype=str, encoding="utf-8-sig")
            for _, row in adj_df.iterrows():
                ds = str(row["dataset"]).strip()
                mod = str(row["model"]).strip()
                lbl = str(row["ratio_label"]).strip()
                x_adj = float(row["x_adjust"])
                y_adj = float(row["y_adjust"])

                adjustments[(ds, mod, lbl)] = (x_adj, y_adj)

            print(
                f"\nLoaded {len(adjustments)} manual offsets from '{adjustments_file}'"
            )
        except Exception as e:
            print(f"\nWARNING: Failed to parse '{adjustments_file}': {e}\n")

    label_prefix = "Median" if args.average == "median" else "Avg"

    # Grid: one column per dataset (with data), families packed into rows so
    # complementary ones (text-only llama, vision-only llava) share a row.
    # plot_data only holds what --dataset / --model kept, so the grid follows.
    ds_cols = [ds for ds in DATASETS if plot_data.get(ds)]
    fam_rows = _pack_family_rows(plot_data, ds_cols)

    if not fam_rows or not ds_cols:
        print("Nothing to plot (no families/datasets with data).")
        return

    # Titles/x-labels hang off the outermost *drawn* panel of each column, which is
    # not simply row 0 / the last row once packing leaves a hole in the grid.
    first_row_of_col = {ds: r for r in reversed(range(len(fam_rows))) for ds in fam_rows[r]}
    last_row_of_col = {ds: r for r in range(len(fam_rows)) for ds in fam_rows[r]}

    text_families = set(_registry_families(modality="text"))

    if args.compact_plots:
        _draw_compact_figure(plot_data, args, ds_cols, text_families, label_prefix)
        _save_figure()
        return

    nrows, ncols = len(fam_rows), len(ds_cols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(PANEL_WIDTH * ncols, PANEL_HEIGHT * nrows),
        squeeze=False,
    )

    # Which sizes appear anywhere (drives the shared legend content).
    any_large = any(plot_data[ds][fam]["large"] for ds in plot_data for fam in plot_data[ds])
    any_small = any(plot_data[ds][fam]["small"] for ds in plot_data for fam in plot_data[ds])

    legend_drawn = False
    for r, row in enumerate(fam_rows):
        # Model named by the panel to the left, so a label is only drawn where
        # the row switches models.
        prev_name = None

        for c, dataset_key in enumerate(ds_cols):
            ax = axes[r][c]
            family = row.get(dataset_key)
            if family is None:
                # Packing left this slot unused — hide it rather than drawing an
                # empty frame.
                ax.set_axis_off()
                continue

            panel = plot_data[dataset_key].get(family, {})
            _draw_panel(ax, panel, adjustments, dataset_key, speedup=args.speedup)

            model_name = _model_display_name(
                family, _panel_modality(panel), text_families
            )

            # Column header (dataset) on the topmost drawn panel. Each run of
            # panels sharing a model is headed by a label on its left, so a row
            # reads: Qwen | text panels | QwenVL | artwork panel.
            if r == first_row_of_col[dataset_key]:
                ax.set_title(DATASETS[dataset_key])
            if model_name != prev_name:
                ax.set_ylabel(f"{model_name}\n{label_prefix} F1 Score")
            prev_name = model_name
            if r == last_row_of_col[dataset_key]:
                ax.set_xlabel(_x_axis_label(args, label_prefix))

            # Legend once, on the first drawn panel; content reflects sizes present anywhere.
            if not legend_drawn:
                legend_drawn = True
                handles = []
                if any_large:
                    handles.append(
                        Line2D([], [], color=MODELS["70B"]["color"],
                               marker=MODELS["70B"]["marker"], linestyle="-",
                               label=MODELS["70B"]["label"])
                    )
                if any_small:
                    handles.append(
                        Line2D([], [], color=MODELS["8B"]["color"],
                               marker=MODELS["8B"]["marker"], linestyle="-",
                               label=MODELS["8B"]["label"])
                    )
                if handles:
                    ax.legend(handles=handles, loc="upper left", frameon=True,
                              edgecolor="0.8", fancybox=False).set_zorder(5)

            _style_panel_axes(ax, dataset_key, args)

    _save_figure()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process KV Cache quality metrics and plot results."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmark_results",
        help="Root directory holding {extract,filter}_stats/... metrics CSVs.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        choices=list(DATASETS.keys()),
        help="Only plot these datasets (one grid column each). Defaults to all.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        choices=_registry_families(),
        help="Only plot these model families (one grid row each), e.g. 'qwen'. "
        "Both the large and small model of the family are kept. Defaults to all.",
    )
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="Put speedup over the gold operator on the X axis instead of "
        "absolute runtime. Each family is measured against its own gold — its "
        "large model at full prefill — so every family starts at 1x and the "
        "curves become comparable rather than merely overlapping. Absolute "
        "seconds are not shown in this mode.",
    )
    parser.add_argument(
        "--compact-plots",
        action="store_true",
        help="Draw one panel per dataset with every family together instead of a "
        "family x dataset grid: colour identifies the family, a solid/dashed line "
        "the large/small model. Per-point compression labels are dropped (see "
        "csv_quality_runtime.py for the numbers).",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip data processing inside the 'extract_stats' folders.",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip data processing inside the 'filter_stats' folders.",
    )
    parser.add_argument(
        "--average",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation method used for averaging data points (default: mean).",
    )
    parser.add_argument(
        "--show-minor-ticks",
        nargs="*",
        default=None,
        choices=list(DATASETS.keys()),
        help="Datasets (e.g., 'movie', 'artwork') whose X-axis minor ticks should "
        f"be labelled, at {' and '.join(f'{s}x' for s in MINOR_TICK_SUBS)} of each "
        "decade. Pass the flag with no datasets to label all of them. "
        "Defaults to none.",
    )

    args = parser.parse_args()

    # nargs='*' turns a bare --show-minor-ticks into [], which is indistinguishable
    # from not passing it at all; read it as "every dataset" instead of a no-op.
    if args.show_minor_ticks is None:
        args.show_minor_ticks = []
    elif not args.show_minor_ticks:
        args.show_minor_ticks = list(DATASETS)

    data = load_and_process_data(args)
    generate_plot(data, args)
