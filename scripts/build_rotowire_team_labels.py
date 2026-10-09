"""Re-key the rotowire *team* ground truth so the perfect operators can read it.

The raw ``rotowire_teams_ground_truth.csv`` is keyed by ``game_id`` + ``_index_team``,
whereas :func:`reasondb.operators.perfect_operators.label_lookup.lookup_labels` keys on
``index_column_names(base_tables)`` -- ``["_index_" + t for t in sorted(base_tables)]``
-- so a step over ``["reports", "teams"]`` asks for ``_index_reports`` and ``_index_teams``.
This script converts the team file to that layout, matching its player sibling
(``rotowire_players_ground_truth.csv``, keyed by ``_index_reports`` + ``_index_players``).

Two column renames, no relabelling:

- ``game_id`` -> ``_index_reports``, via the row ordinal of ``reports.csv``. That table's
  ``game_id`` happens to equal its row number today, but the mapping is built and checked
  rather than assumed: if ``reports.csv`` is ever regenerated in another order, an
  identity shortcut would silently attach every label to the wrong report.
- ``_index_team`` -> ``_index_teams``, matching the plural table name. Values already
  index ``teams.csv`` correctly.

Idempotent: a file that already carries ``_index_reports`` is left alone, so a re-run
after the conversion is a no-op rather than a second, wrong conversion.

    python scripts/build_rotowire_team_labels.py [--check]

``--check`` verifies without writing (what the test suite does).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = REPO_ROOT / "reasondb/evaluation/ground_truth/rotowire/rotowire_teams_ground_truth.csv"
FILES = REPO_ROOT / "reasondb/evaluation/benchmarks/files"

#: Emitted in this order: the two keys, then the stat columns a curated query may name.
STAT_COLUMNS = ["Wins", "Losses", "Total points", "Points in 4th quarter"]


def convert(labels: pd.DataFrame, reports: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Return *labels* re-keyed to ``_index_reports`` + ``_index_teams``."""
    report_ordinal = {game_id: i for i, game_id in enumerate(reports["game_id"])}
    assert len(report_ordinal) == len(reports), (
        f"reports.csv has duplicate game_id values; a label row cannot name one report."
    )

    unresolved = sorted(set(labels["game_id"]) - set(report_ordinal))
    assert not unresolved, (
        f"{len(unresolved)} game_id(s) in the labels have no row in reports.csv, so those "
        f"labels key nothing. First: {unresolved[:5]}."
    )

    out = pd.DataFrame(
        {
            "_index_reports": labels["game_id"].map(report_ordinal).astype(int),
            "_index_teams": labels["_index_team"].astype(int),
        }
    )
    for column in STAT_COLUMNS:
        out[column] = labels[column]

    out_of_range = out.loc[
        (out["_index_teams"] < 0) | (out["_index_teams"] >= len(teams)), "_index_teams"
    ]
    assert out_of_range.empty, (
        f"_index_teams must index teams.csv ({len(teams)} rows); out of range: "
        f"{sorted(set(out_of_range))[:5]}."
    )

    duplicated = out.duplicated(subset=["_index_reports", "_index_teams"])
    assert not duplicated.any(), (
        f"{int(duplicated.sum())} duplicate (report, team) key(s); the lookup takes one "
        "label per tuple, so a duplicate makes the ground truth ambiguous."
    )
    return out.sort_values(["_index_reports", "_index_teams"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the conversion without writing (non-zero exit if it would change).",
    )
    args = parser.parse_args()

    labels = pd.read_csv(LABELS)
    if "_index_reports" in labels.columns:
        print(f"{LABELS.name} is already keyed by _index_reports/_index_teams; nothing to do.")
        return 0

    reports = pd.read_csv(FILES / "reports.csv")
    teams = pd.read_csv(FILES / "teams.csv")
    team_games = pd.read_csv(FILES / "teams_to_games.csv")

    out = convert(labels, reports, teams)

    # One label per (team, game) pair the join produces. Short of this the lookup raises
    # MissingLabelsError on the first uncovered tuple during a run rather than here.
    assert len(out) == len(team_games), (
        f"{len(out)} label rows against {len(team_games)} rows in teams_to_games.csv: the "
        "ground truth must cover every tuple the join can produce."
    )

    if args.check:
        print(f"OK: {len(out)} rows would be re-keyed to {list(out.columns)}")
        return 0

    out.to_csv(LABELS, index=False)
    print(f"Wrote {len(out)} rows to {LABELS.relative_to(REPO_ROOT)}")
    print(f"  columns: {list(out.columns)}")
    for column in STAT_COLUMNS:
        filled = int(out[column].notna().sum())
        print(f"  {column:24s} {filled:5d}/{len(out)} labelled ({filled / len(out):.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
