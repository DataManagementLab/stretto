from collections import defaultdict
import logging
import argparse
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.path import Path as PlotPath
from reasondb.query_plan.physical_operator import CostType
from reasondb.evaluation.plotting import (
    DATASET_ORDER as dataset_order,
    LABEL_MAP as APPROACH_TO_LABEL,
    apply_default_style,
    fix_labels,
)
from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS as BENCHMARKS
from reasondb.evaluation.sweep_frames import geomean_overall
from reasondb.evaluation.sweep_figures import BREAKDOWN_PALETTE
from reasondb.monitor import phases
from typing import List, Optional


# Create a custom half-circle path (upper half)
theta = np.linspace(-np.pi / 2, np.pi / 2, 30)
verts = np.column_stack([np.cos(theta), np.sin(theta)])
verts = np.vstack([[0, 0], verts, [0, 0]])  # Close the shape
codes = [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(verts) - 2) + [PlotPath.CLOSEPOLY]
right_half_circle = PlotPath(verts, codes)

bucketized_facets = ["true_output_cardinality", "predicted_output_cardinality"]

# Wall-clock runtime breakdown (component times emitted by run_benchmark as
# `time_*`, in seconds). Stacked bottom-to-top; "Other" is the remainder of
# end-to-end not attributed to a named leaf phase (db prepare, dependency
# graph, wind-down). Reasoning/configuring are the query-planning phases;
# profiling + optimization + execution make up the engine runtime.
#
# The component list and its colours are shared with the live dashboard and the
# sweep figures so that every view names and colours each phase identically.
RUNTIME_BREAKDOWN_COMPONENTS = phases.RUNTIME_BREAKDOWN_COMPONENTS

FLIERS = False
STRIPPLOT = False
BAR_LABEL = False

apply_default_style()


def load_df(csv_paths):
    dfs = []
    for p in csv_paths:
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, axis=0)
    df.index.name = "id"
    df.reset_index(drop=False, inplace=True)

    query_id_dict = df.groupby("query").min()["id"].to_dict()
    df["query_id"] = df["query"].map(query_id_dict)

    precision_range = df["precision_guarantee"].dropna().unique()
    recall_range = df["recall_guarantee"].dropna().unique()
    precision_range.sort()
    recall_range.sort()
    df["precision_guarantee"] = df["precision_guarantee"].fillna(precision_range[0])
    df["recall_guarantee"] = df["recall_guarantee"].fillna(recall_range[0])

    for bucketized_facet in bucketized_facets:
        N = 4  # number of buckets

        cats, _ = pd.qcut(
            df[bucketized_facet],
            q=N,
            retbins=True,
            duplicates="drop",
        )

        df[bucketized_facet] = cats

    return df


def add_pooled_facet(df: pd.DataFrame, main_facet: Optional[str]) -> tuple:
    """Append a pooled ``"Overall"`` copy of *df* and return ``(frame, facet_column)``.

    ``main_facet=None`` means "no faceting": the frame gets a synthetic single-valued
    column so the same code path draws a one-panel figure.
    """
    if main_facet is None:
        out = df.copy()
        out["no_facet"] = "Overall"
        return out, "no_facet"
    pooled = df.copy()
    pooled[main_facet] = "Overall"
    return pd.concat((pooled, df), ignore_index=True), main_facet


def overall_by_geomean(
    per_facet: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    facet: str,
) -> pd.DataFrame:
    """Replace the pooled row of *per_facet* with a geometric mean over the facets.

    Summing runtimes across datasets reports whichever benchmark is slowest - artwork's
    totals run an order of magnitude above movie's - rather than reporting the system.
    The geometric mean is scale-free: if one approach costs k times another on *every*
    dataset, their pooled bars are in ratio k.

    Only applied when the facet is the dataset. Over ``num_semops`` or a cardinality
    bucket a sum is the honest total (the buckets partition one query set), and the
    argument above does not apply.
    """
    real = per_facet[per_facet[facet] != "Overall"]
    if facet not in ("dataset", "no_facet") or real[facet].nunique() < 2:
        return per_facet
    pooled = geomean_overall(
        real,
        arm_col="approach_name",
        value_cols=[value_col],
        group_cols=group_cols,
        dataset_col=facet,
        label="Overall",
    )
    return pd.concat([real, pooled], ignore_index=True)


def plot_meets_target(
    df: pd.DataFrame,
    main_facet: Optional[str],
    labels_type: str,
    approaches: List[str],
    output_path: Path,
):
    # A pooled *distribution*, not a pooled total - there is no sum here to take a
    # geometric mean of, so this facet stays a pooling of every query's ratio.
    df, main_facet = add_pooled_facet(df, main_facet)
    precision_df = df.copy()
    precision_df["target_met"] = df["precision"] / df["precision_guarantee"]
    precision_df["target_type"] = "Precision"
    recall_df = df.copy()
    recall_df["target_met"] = df["recall"] / df["recall_guarantee"]
    recall_df["target_type"] = "Recall"
    plot_df = pd.concat((precision_df, recall_df), axis=0).reset_index()

    grid_kwargs = {}
    if main_facet == "dataset":
        grid_kwargs["col_order"] = dataset_order

    g = sns.FacetGrid(plot_df, col=main_facet, margin_titles=True, **grid_kwargs)
    for _, ax in g.axes_dict.items():
        ax.axhline(
            y=1.0,
            color="black",
            linewidth=2,
            linestyle="--",
            label="Meets Target",
            zorder=1,
        )
        ax.axhspan(1, 2, color="green", alpha=0.2)
        ax.axhspan(0, 1, color="red", alpha=0.2)

    g.map_dataframe(
        sns.boxplot,
        x="approach_name",
        y="target_met",
        whis=[5, 95],  # type: ignore
        order=approaches,
        showfliers=FLIERS,
        fliersize=1,
        hue="target_type",
    )
    if main_facet == "no_facet":
        plt.legend(bbox_to_anchor=(-0.35, 1), loc="upper right", borderaxespad=0.0)  # type: ignore
    else:
        plt.legend(
            bbox_to_anchor=(1, -0.4), loc="upper right", borderaxespad=0.0, ncol=3
        )  # type: ignore
    if STRIPPLOT:
        g.map_dataframe(
            sns.stripplot,
            x="approach_name",
            y="target_met",
            jitter=True,
            size=3,
            alpha=0.6,
        )
    fix_labels(g)
    g.set(xlabel="")

    plt.ylim(0.2, 2)

    plt.savefig(
        output_path / f"guarantees_{labels_type}_{main_facet}_plot.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close()


def plot_runtime_per_target(
    df: pd.DataFrame,
    cost_type: CostType,
    main_facet: Optional[str],
    labels_type: str,
    approaches: List[str],
    output_path: Path,
):
    df, main_facet = add_pooled_facet(df, main_facet)

    for cost_part in ["total_cost", "execution_cost"]:
        value_col = f"{cost_part}_{cost_type.value}"
        group_cols = ["precision_guarantee", "recall_guarantee"]
        plot_df = (
            df.groupby(["approach_name", *group_cols, main_facet])[[value_col]]
            .sum()
            .reset_index()
        )
        # The pooled row is a geometric mean of the per-dataset totals rather than their
        # sum; see overall_by_geomean.
        plot_df = overall_by_geomean(plot_df, value_col, group_cols, main_facet)
        plot_df = plot_df[plot_df[value_col] > 0]
        if plot_df.empty:
            continue

        if cost_type == CostType.RUNTIME:
            plot_df[f"{cost_part}_{cost_type.value}"] /= 3600

        plot_df["guarantee_setting"] = plot_df.apply(
            lambda x: f"p:{x['precision_guarantee']}_r:{x['recall_guarantee']}",
            axis=1,
        )
        hue_order = sorted(plot_df["guarantee_setting"].unique())

        palette = ["#004488", "#DDAA33", "#BB5566"]
        palette_dict = dict(zip(hue_order, palette))

        grid_kwargs = {}
        if main_facet == "dataset":
            grid_kwargs["col_order"] = dataset_order

        if (main_facet == "dataset"):
            print(main_facet)
            print_df = plot_df[(plot_df["approach_name"] == "optim_global") | (plot_df["approach_name"] == "optim_global_no_compr")]
            print(print_df[["approach_name", main_facet, "guarantee_setting", f"{cost_part}_{cost_type.value}"]])
        g = sns.FacetGrid(
            plot_df, col=main_facet, margin_titles=True, sharey=False, **grid_kwargs
        )
        g.map_dataframe(
            sns.barplot,
            x="approach_name",
            y=f"{cost_part}_{cost_type.value}",
            hue="guarantee_setting",
            order=approaches,
            hue_order=hue_order,
            palette=palette_dict,
        )
        if BAR_LABEL:
            for ax in g.axes.flat:
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.0f", padding=3)
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(25)
                label.set_ha("right")
                label.set_rotation_mode("anchor")
            ax.set_ylim((0, None))

        if main_facet == "no_facet":
            plt.legend(bbox_to_anchor=(1.1, 1), loc="upper left", borderaxespad=0.0)  # type: ignore
        else:
            plt.legend(
                bbox_to_anchor=(1, -0.4),  # type: ignore
                loc="upper right",  # type: ignore
                borderaxespad=0.0,  # type: ignore
                ncol=3,  # type: ignore
            )  # type: ignore

        fix_labels(g)
        g.set(xlabel="")
        plt.savefig(
            output_path
            / f"runtime_per_target_{labels_type}_{cost_part}_{cost_type.value}_{main_facet}plot.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.close()


def plot_runtime_breakdown(
    df: pd.DataFrame,
    main_facet: Optional[str],
    labels_type: str,
    approaches: List[str],
    output_path: Path,
):
    """Stacked-bar breakdown of the wall-clock runtime components (`time_*`).

    One figure per guarantee setting; faceted by `main_facet` (e.g. dataset);
    each approach's bar is stacked into execution / profiling / optimization /
    configuring / reasoning / other. Component times are summed over queries and
    converted to hours (matching the runtime cost plots).
    """
    df, main_facet = add_pooled_facet(df, main_facet)

    named_cols = [c for c, _ in RUNTIME_BREAKDOWN_COMPONENTS if c != "time_other"]
    needed = named_cols + ["time_end_to_end"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logging.warning(
            "Runtime component columns %s missing (metrics predate component "
            "timing?); skipping runtime breakdown plot.",
            missing,
        )
        return

    guarantee_cols = ["precision_guarantee", "recall_guarantee"]
    group_cols = ["approach_name", *guarantee_cols, main_facet]
    agg = df.groupby(group_cols)[needed].sum().reset_index()
    # Remainder of end-to-end not covered by a named leaf phase.
    agg["time_other"] = (
        agg["time_end_to_end"] - agg[named_cols].sum(axis=1)
    ).clip(lower=0)
    all_component_cols = [c for c, _ in RUNTIME_BREAKDOWN_COMPONENTS]
    for col in all_component_cols:
        agg[col] = agg[col] / 3600  # seconds -> hours

    # Rebuild the pooled bar as a geometric mean of the per-dataset totals, split by
    # each phase's mean share so the segments still sum to the bar. Summing instead
    # would make the pooled bar a portrait of the slowest dataset.
    real = agg[agg[main_facet] != "Overall"]
    if main_facet in ("dataset", "no_facet") and real[main_facet].nunique() >= 2:
        pooled = geomean_overall(
            real,
            arm_col="approach_name",
            value_cols=all_component_cols,
            group_cols=guarantee_cols,
            dataset_col=main_facet,
            label="Overall",
        )
        agg = pd.concat([real, pooled], ignore_index=True)

    agg["guarantee_setting"] = agg.apply(
        lambda x: f"p:{x['precision_guarantee']}_r:{x['recall_guarantee']}",
        axis=1,
    )

    grid_kwargs = {}
    if main_facet == "dataset":
        grid_kwargs["col_order"] = dataset_order

    def draw_stack(data, color, **kwargs):
        ax = plt.gca()
        # Keep the full approach order (zeros for approaches absent from this
        # facet) so x positions line up across facets, like the bar plots.
        per_approach = data.set_index("approach_name")
        x = np.arange(len(approaches))
        bottom = np.zeros(len(approaches))
        for col, label in RUNTIME_BREAKDOWN_COMPONENTS:
            vals = np.array(
                [
                    float(per_approach.loc[a, col]) if a in per_approach.index else 0.0
                    for a in approaches
                ]
            )
            ax.bar(
                x, vals, bottom=bottom, label=label, color=BREAKDOWN_PALETTE[label]
            )
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(list(approaches))

    for guarantee in sorted(agg["guarantee_setting"].unique()):
        sub = agg[agg["guarantee_setting"] == guarantee]
        if sub[all_component_cols].to_numpy().sum() == 0:
            continue

        g = sns.FacetGrid(
            sub, col=main_facet, margin_titles=True, sharey=False, **grid_kwargs
        )
        g.map_dataframe(draw_stack)

        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(25)
                label.set_ha("right")
                label.set_rotation_mode("anchor")
            ax.set_ylim((0, None))

        fix_labels(g)
        g.set(xlabel="")
        for ax in g.axes.flat:
            ax.set_ylabel("Runtime [h]")

        # Single component legend (dedup across facets), ordered bottom->top.
        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        seen = {}
        for h, lab in zip(handles, labels):
            seen.setdefault(lab, h)
        ordered = [lab for _, lab in RUNTIME_BREAKDOWN_COMPONENTS if lab in seen]
        if main_facet == "no_facet":
            plt.legend(
                [seen[lab] for lab in ordered],
                ordered,
                bbox_to_anchor=(1.1, 1),
                loc="upper left",
                borderaxespad=0.0,
            )
        else:
            plt.legend(
                [seen[lab] for lab in ordered],
                ordered,
                bbox_to_anchor=(1, -0.4),
                loc="upper right",
                borderaxespad=0.0,
                ncol=3,
            )

        plt.savefig(
            output_path
            / f"runtime_breakdown_{labels_type}_{guarantee}_{main_facet}plot.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        plt.close()


def plot_operator_stats(csv_path: Path):
    df = (
        pd.read_csv(csv_path)
        .groupby(["approach", "operator", "precision_target", "recall_target"])
        .sum()
    ).reset_index()
    g = sns.FacetGrid(
        df,
        row="precision_target",
        col="recall_target",
        margin_titles=True,
    )
    g.map_dataframe(
        lambda color, data, **kwargs: sns.heatmap(
            data=pd.pivot_table(
                data, values="count", index="approach", columns="operator"
            ).fillna(0),
            **kwargs,
        )
    )
    fix_labels(g)
    output_path = csv_path.parent / "operator_stats.pdf"
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        type=str,
        choices=BENCHMARKS.keys(),
        default="artwork",
        help="The benchmark to plot.",
        nargs="+",
    )
    parser.add_argument(
        "--facets",
        type=str,
        choices=["num_semops", "dataset", "true_output_cardinality"],
        default=["dataset"],
        nargs="+",
        help="The split of the benchmark to run.",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        choices=[
            "abacus",
            "lotus",
            "optim_global",
            "optim_shift_budget",
            "optim_local",
            # "optim_combo",
            # "optim_no_guarantee"
            "optim_global_no_compr",
        ],
        default=[
            "abacus",
            "lotus",
            "optim_global",
        ],
        nargs="+",
        help="The split of the benchmark to run.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "test"],
        default="dev",
        help="The split of the benchmark to run.",
    )
    parser.add_argument(
        "--output-dirs", type=Path, default=[Path("benchmark_results")], nargs="+"
    )
    args = parser.parse_args()

    collected_data = defaultdict(lambda: defaultdict(pd.DataFrame))
    for benchmark in args.benchmarks:
        metric_file_dict = defaultdict(list)
        for output_dir in args.output_dirs:
            benchmark_dir = output_dir / benchmark / args.split
            for metrics_file in benchmark_dir.glob("*metrics.csv"):
                labels_type = metrics_file.stem
                metric_file_dict[labels_type].append(metrics_file)

        for labels_type, metric_files in metric_file_dict.items():
            out_dir = metric_files[0].parent
            df = load_df(metric_files)
            df["dataset"] = benchmark
            collected_data[labels_type][benchmark] = df

            for cost_type in CostType:
                for facet in args.facets:
                    plot_runtime_per_target(
                        df.copy(),
                        cost_type,
                        facet,
                        labels_type,
                        args.approaches,
                        out_dir,
                    )

            for facet in args.facets:
                plot_runtime_breakdown(
                    df.copy(), facet, labels_type, args.approaches, out_dir
                )

            for facet in args.facets:
                plot_meets_target(
                    df.copy(), facet, labels_type, args.approaches, out_dir
                )

    for labels_type, benchmarks_data in collected_data.items():
        df = pd.concat(benchmarks_data.values(), axis=0)
        for cost_type in CostType:
            for facet in args.facets:
                plot_runtime_per_target(
                    df.copy(),
                    cost_type,
                    facet,
                    labels_type,
                    args.approaches,
                    args.output_dirs[0],
                )

        for facet in args.facets:
            plot_runtime_breakdown(
                df.copy(),
                facet,
                labels_type,
                args.approaches,
                args.output_dirs[0],
            )

        for facet in args.facets:
            plot_meets_target(
                df.copy(),
                facet,
                labels_type,
                args.approaches,
                args.output_dirs[0],
            )

        for cost_type in CostType:
            plot_runtime_per_target(
                df.copy(),
                cost_type,
                None,
                labels_type,
                args.approaches,
                args.output_dirs[0],
            )
        plot_runtime_breakdown(
            df.copy(),
            None,
            labels_type,
            args.approaches,
            args.output_dirs[0],
        )
        plot_meets_target(
            df.copy(),
            None,
            labels_type,
            args.approaches,
            args.output_dirs[0],
        )


if __name__ == "__main__":
    main()
