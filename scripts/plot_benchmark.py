from collections import defaultdict
import logging
import argparse
from pathlib import Path
import tracemalloc
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.path import Path as PlotPath
from reasondb.query_plan.physical_operator import CostType
from run_benchmark import BENCHMARKS
import scienceplots
from typing import List, Optional


tracemalloc.start()
# warnings.filterwarnings("error", message="Using a slow.*")

# Create a custom half-circle path (upper half)
theta = np.linspace(-np.pi / 2, np.pi / 2, 30)
verts = np.column_stack([np.cos(theta), np.sin(theta)])
verts = np.vstack([[0, 0], verts, [0, 0]])  # Close the shape
codes = [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(verts) - 2) + [PlotPath.CLOSEPOLY]
right_half_circle = PlotPath(verts, codes)

dataset_order = [
    "Overall",
    "artwork_random_medium",
    "rotowire_random",
    "email_random",
    "movie_random",
    "ecommerce_random_large",
]
APPROACH_TO_LABEL = {
    "true_output_cardinality": "Cardinality",
    "kv09": "KV-0.9",
    "kv00": "KV-0.0",
    "gpt": "GPT",
    "abacus": "Abacus-style",
    "lotus": "Lotus-style",
    "optim_local": "Local",
    "optim_shift_budget": "Independent",
    "optim_global": "Stretto",
    "optim_combo": "Stretto Full",
    "optim_no_guarantee": "Stretto (no guarantees)",
    "approach_name": "Approach",
    "total_cost_fake_cost": "Total Cost (Fake)",
    "execution_cost_fake_cost": "Execution Cost (Fake)",
    "total_cost_runtime": "Total Cost (Runtime)",
    "execution_cost_runtime": "Runtime [h]",
    "total_cost_monetary": "Total Cost (Monetary)",
    "execution_cost_monetary": "Execution Cost (Monetary)",
    "precision_met": "Precision Met",
    "recall_met": "Recall Met",
    "dataset = Overall": "Overall",
    "no_facet = Overall": "Overall",
    "dataset = ecommerce_random_large": "Ecommerce",
    "dataset = artwork_random": "Artwork (small)",
    "dataset = artwork_random_medium": "Artwork",
    "dataset = rotowire_random": "Rotowire",
    "dataset = movie_random": "Movie",
    "dataset = email_random": "Enron Email",
    "dataset = ecommerce": "Ecommerce",
    "speedup_execution_cost_fake_cost": "Speedup Execution Cost (Fake)",
    "speedup_total_cost_fake_cost": "Speedup Total Cost (Fake)",
    "speedup_execution_cost_runtime": "Speedup Execution Cost (Runtime)",
    "speedup_total_cost_runtime": "Speedup Total Cost (Runtime)",
    "speedup_execution_cost_monetary": "Speedup Execution Cost (Monetary)",
    "speedup_total_cost_monetary": "Speedup Total Cost (Monetary)",
    "p:0.5_r:0.5": "Prec=0.5/\nRec=0.5",
    "p:0.7_r:0.7": "Prec=0.7/\nRec=0.7",
    "p:0.9_r:0.9": "Prec=0.9/\nRec=0.9",
    "num_semops = 2": "2 Semantic Ops",
    "num_semops = 3": "3 Semantic Ops",
    "num_semops = 4": "4 Semantic Ops",
    "target_met": "Target Met",
}

bucketized_facets = ["true_output_cardinality", "predicted_output_cardinality"]

FLIERS = False
STRIPPLOT = False
BAR_LABEL = False

plt.rcParams.update({"font.size": 16})
plt.style.use(["science", "no-latex", "grid", "high-contrast"])


def load_df(csv_path):
    df = pd.read_csv(csv_path)
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


def fix_labels(g: sns.FacetGrid):
    for ax in g.axes.flat:
        name_map = APPROACH_TO_LABEL

        # Axis labels
        ax.set_xlabel(name_map.get(ax.get_xlabel(), ax.get_xlabel()))
        ax.set_ylabel(name_map.get(ax.get_ylabel(), ax.get_ylabel()))

        # Legend labels
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_text(name_map.get(text.get_text(), text.get_text()))

        # Tick labels
        labels = [label.get_text() for label in ax.get_xticklabels()]
        labels = [name_map.get(lab, lab) for lab in labels]
        ax.set_xticklabels(labels, rotation=25, ha="right", rotation_mode="anchor")

        # Title labels
        old = ax.get_title()
        new = name_map.get(old, old)
        if new == old and "=" in old:
            # Handle facet titles
            lhs = old.split("=")[0].strip()
            rhs = old.split("=")[1].strip()
            new = f"{name_map.get(lhs, lhs)} = {name_map.get(rhs, rhs)}"
        ax.set_title(new)


def plot_meets_target(
    df: pd.DataFrame,
    main_facet: Optional[str],
    labels_type: str,
    approaches: List[str],
    output_path: Path,
):
    if main_facet is not None:
        overall = df.copy()
        overall[main_facet] = "Overall"
        df = pd.concat((overall, df))
    else:
        main_facet = "no_facet"
        df = df.copy()
        df[main_facet] = "Overall"

    overall = df.copy()
    overall[main_facet] = "Overall"
    df = pd.concat((overall, df))
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
    if main_facet is not None:
        overall = df.copy()
        overall[main_facet] = "Overall"
        df = pd.concat((overall, df))
    else:
        main_facet = "no_facet"
        df = df.copy()
        df[main_facet] = "Overall"

    for cost_part in ["total_cost", "execution_cost"]:
        plot_df = df
        print(
            plot_df[
                (plot_df["approach_name"] == "optim_global")
                & (plot_df["precision_guarantee"] == 0.9)
            ]
            .groupby(main_facet)
            .size()
        )

        plot_df = (
            plot_df.groupby(
                [
                    "approach_name",
                    "precision_guarantee",
                    "recall_guarantee",
                    main_facet,
                ]
            )[[f"{cost_part}_{cost_type.value}"]]
            .sum()
            .reset_index()
        )
        plot_df = plot_df[plot_df[f"{cost_part}_{cost_type.value}"] > 0]
        if plot_df.empty:
            continue

        if cost_type == CostType.RUNTIME:
            plot_df[f"{cost_part}_{cost_type.value}"] /= 3600

        plot_df["guarantee_setting"] = plot_df.apply(
            lambda x: f"p:{x['precision_guarantee']}_r:{x['recall_guarantee']}",
            axis=1,
        )
        # max_costs = plot_df.groupby(["guarantee_setting", main_facet])[
        #     f"{cost_part}_{cost_type.value}"
        # ].max()
        # max_costs.name = "max_cost"
        # plot_df = plot_df.merge(
        #     max_costs,
        #     left_on=["guarantee_setting", main_facet],
        #     right_index=True,
        # )
        # speedup = plot_df["max_cost"] / plot_df[f"{cost_part}_{cost_type.value}"]
        # plot_df[f"speedup_{cost_part}_{cost_type.value}"] = speedup
        hue_order = sorted(plot_df["guarantee_setting"].unique())
        # palette_pastel = sns.color_palette("tab10", n_colors=len(hue_order))

        palette = ["#004488", "#DDAA33", "#BB5566"]
        palette_dict = dict(zip(hue_order, palette))

        grid_kwargs = {}
        if main_facet == "dataset":
            grid_kwargs["col_order"] = dataset_order
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
            ax.set_ylim((1, None))

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
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    collected_data = defaultdict(lambda: defaultdict(pd.DataFrame))
    for benchmark in args.benchmarks:
        benchmark_dir = args.output_dir / benchmark / args.split

        for metrics_file in benchmark_dir.glob("*metrics.csv"):
            csv_path = metrics_file
            df = load_df(csv_path)
            df["dataset"] = benchmark
            labels_type = metrics_file.stem
            collected_data[labels_type][benchmark] = df

            for cost_type in CostType:
                for facet in args.facets:
                    plot_runtime_per_target(
                        df.copy(),
                        cost_type,
                        facet,
                        labels_type,
                        args.approaches,
                        csv_path.parent,
                    )

            for facet in args.facets:
                plot_meets_target(
                    df.copy(), facet, labels_type, args.approaches, csv_path.parent
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
                    args.output_dir,
                )

        for facet in args.facets:
            plot_meets_target(
                df.copy(), facet, labels_type, args.approaches, args.output_dir
            )

        for cost_type in CostType:
            plot_runtime_per_target(
                df.copy(),
                cost_type,
                None,
                labels_type,
                args.approaches,
                args.output_dir,
            )
        plot_meets_target(
            df.copy(), None, labels_type, args.approaches, args.output_dir
        )


if __name__ == "__main__":
    main()
