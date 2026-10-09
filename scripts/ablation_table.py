"""The ablation table: per-query execution speedup of each ablation step, by guarantee target.

Two steps, each a paired comparison of the same (benchmark, query, target) run by two arms
(see :mod:`reasondb.evaluation.ablation_gains`):

    optimizer    no optimization -> Stretto without KV-compressed operators
    kv           Stretto without KV-compressed operators -> full Stretto

For each target and for all targets pooled, it reports the mean, 90th percentile and
maximum speedup, and the share of queries sped up by more than 1.01x, 1.5x and 2x.

    python scripts/ablation_table.py --output-dirs artifacts/abl01/merged
    python scripts/ablation_table.py --output-dirs artifacts/abl01/merged --latex
"""

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from reasondb.evaluation import ablation_gains as gains
from reasondb.evaluation import sweep_frames as frames

STEP_TITLES = {
    "optimizer": "Optimizer: no optimization -> Stretto, without KV cache-enabled operators",
    "kv": "Compression: Stretto, without KV cache-enabled operators -> full Stretto",
}

COLUMNS = {
    "mean": "Mean",
    "p90": "p90",
    "max": "Max",
    "share_faster_meaningful": ">1.01x",
    "share_1_5x": ">1.5x",
    "share_2x": ">2x",
}

SHARES = ("share_faster_meaningful", "share_1_5x", "share_2x")


def target_label(setting: str) -> str:
    """``p:0.5_r:0.5`` -> ``P/R 0.5`` when both targets agree, else the raw setting."""
    parts = dict(part.split(":") for part in setting.split("_"))
    if parts.get("p") == parts.get("r"):
        return f"P/R {parts['p']}"
    return setting


def step_table(df: pd.DataFrame, step: str) -> pd.DataFrame:
    """One row per guarantee target plus the pooled row, in the paper's columns."""
    base_arm, with_arm = gains.STEPS[step]
    pairs = gains.paired_speedups(df, base_arm=base_arm, with_arm=with_arm)
    if pairs.empty:
        return pd.DataFrame()
    by_target = gains.speedup_summary(pairs, "execution", by=["guarantee_setting"])
    by_target = by_target.sort_values("guarantee_setting")
    by_target["Target"] = by_target["guarantee_setting"].map(target_label)
    pooled = gains.speedup_summary(pairs, "execution").reset_index(drop=True)
    pooled["Target"] = "All targets"
    table = pd.concat([by_target, pooled], ignore_index=True)
    return table[["Target", *COLUMNS]].rename(columns=COLUMNS)


def format_rows(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for column in ("Mean", "p90", "Max"):
        out[column] = out[column].map(lambda v: f"{v:.2f}")
    for key in SHARES:
        out[COLUMNS[key]] = out[COLUMNS[key]].map(lambda v: f"{100 * v:.0f}%")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dirs", type=Path, nargs="+",
                        default=[Path("benchmark_results/abl01/merged")])
    parser.add_argument("--csv-name", type=str, default="ablation.csv")
    parser.add_argument("--latex", action="store_true",
                        help="Print the table body as LaTeX rows instead of plain text.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    paths = frames.find_sweep_csvs(args.output_dirs, args.csv_name)
    if not paths:
        raise SystemExit(f"No {args.csv_name} under {args.output_dirs}.")
    df = frames.load_sweep(paths)
    df["arm"] = frames._ablation_arm(df)

    for step in ("optimizer", "kv"):
        table = step_table(df, step)
        if table.empty:
            print(f"{step}: no query was run by both arms")
            continue
        rows = format_rows(table)
        if args.latex:
            print(f"\\multicolumn{{{len(rows.columns)}}}{{l}}{{\\textit{{{STEP_TITLES[step]}}}}} \\\\")
            for _, row in rows.iterrows():
                cells = [str(v).replace("%", "\\%") for v in row]
                print("    " + " & ".join(cells) + " \\\\")
        else:
            print(STEP_TITLES[step])
            print(rows.to_string(index=False))
            print()


if __name__ == "__main__":
    main()
