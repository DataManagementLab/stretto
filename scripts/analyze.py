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


APPROACHES = [
    # "kv09",
    # "kv00",
    # "gpt",
    "abacus",
    "lotus",
    "optim_local",
    "optim_shift_budget",
    "optim_global",
    "optim_combo",
]


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
    return df


def analyze(df: pd.DataFrame, compare: str, to: str):
    df_compare = df[df["approach_name"] == compare]
    df_to = df[df["approach_name"] == to]
    merged = pd.merge(
        df_compare,
        df_to,
        on=["query_id", "query", "precision_guarantee", "recall_guarantee"],
        suffixes=(f"_{compare}", f"_{to}"),
    )
    execution_time_diff = (
        merged[f"execution_cost_runtime_{compare}"]
        - merged[f"execution_cost_runtime_{to}"]
    )
    max_diff_query = merged.loc[execution_time_diff.idxmax()]
    print(f"Max execution time difference: {max_diff_query}")
    print(f"Query: {max_diff_query['query']}")


def zero_metric_queries(df: pd.DataFrame, compare: str):
    df_compare = df[df["approach_name"] == compare]
    recall_zero = df_compare[df_compare["recall"] == 0.0]
    precision_zero = df_compare[df_compare["precision"] == 0.0]
    print(f"Queries with zero recall for {compare}:")
    for _, row in recall_zero.iterrows():
        print(f"- Query ID {row['query_id']}: {row['query']}")

    print(f"Queries with zero precision for {compare}:")
    for _, row in precision_zero.iterrows():
        print(f"- Query ID {row['query_id']}: {row['query']}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=BENCHMARKS.keys(),
        default="artwork_random",
        help="The benchmark to analyze.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "test"],
        default="dev",
        help="The split of the benchmark to run.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default="optim_global",
    )
    parser.add_argument(
        "--to",
        type=str,
        default="lotus",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    benchmark_dir = args.output_dir / args.benchmark / args.split

    for metrics_file in benchmark_dir.glob("*metrics.csv"):
        csv_path = metrics_file
        df = load_df(csv_path)
        analyze(df, args.compare, args.to)
        zero_metric_queries(df, args.compare)


if __name__ == "__main__":
    main()
