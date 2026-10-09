"""Read finished benchmark results off disk for the monitor's Results tab.

This is the on-disk counterpart to the live collector: it loads what
the ``run_benchmark`` producer leaves behind in
``<output-dir>/<benchmark>/<split>/`` and reshapes it into the records the UI charts.

Deliberately mirrors ``scripts/plot_benchmark.py`` rather than reimplementing it:

* the same ``**/*metrics.csv`` discovery,
* the same back-fill of missing guarantee levels with the smallest observed target
  (``plot_benchmark.load_df``), so no-guarantee baselines line up with the guaranteed
  approaches on a shared axis,
* the same ``target_met`` ratios that ``plot_meets_target`` charts,
* the same runtime-breakdown components and palette as ``plot_runtime_breakdown``.

Labels come from ``reasondb.evaluation.plotting`` so the UI and the paper figures never
disagree about what ``optim_global`` is called.

Every loader returns a plain JSON-able dict. Failures are returned as ``{"error": ...}``
payloads rather than raised: a malformed CSV in one directory must not blank the page.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from reasondb.monitor import phases as phases_mod
from reasondb.monitor.dimensions import QUERY_STAT_COLUMNS

logger = logging.getLogger(__name__)

#: Columns ``evaluate()`` always writes. Their absence means the file is not a metrics
#: CSV (or predates the current schema), which is worth saying out loud.
REQUIRED_COLUMNS = (
    "approach_name",
    "query",
    "precision_guarantee",
    "recall_guarantee",
    "precision",
    "recall",
)

#: Bottom-to-top, matching ``plot_benchmark.RUNTIME_BREAKDOWN_COMPONENTS``. Defined in
#: :mod:`reasondb.monitor.phases` (stdlib-only, so the collector can share it) and
#: re-exported here, where every existing caller expects to find it.
RUNTIME_BREAKDOWN_COMPONENTS = phases_mod.RUNTIME_BREAKDOWN_COMPONENTS
derive_phase_components = phases_mod.derive_phase_components

#: Paul Tol's bright qualitative palette, as used by the benchmark figures.
BREAKDOWN_PALETTE = [
    "#004488",
    "#DDAA33",
    "#BB5566",
    "#66CCEE",
    "#228833",
    "#BBBBBB",
]

COST_PARTS = ("execution_cost", "tuning_cost", "total_cost")
COST_TYPES = ("runtime", "monetary", "fake_cost")


def _label_map() -> Dict[str, str]:
    """``plotting.LABEL_MAP`` if importable.

    ``reasondb.evaluation.plotting`` imports seaborn/matplotlib at module scope; the
    monitor must still work in an environment where those are absent.
    """
    try:
        from reasondb.evaluation.plotting import LABEL_MAP

        return dict(LABEL_MAP)
    except Exception as exc:  # pragma: no cover - depends on the environment
        logger.debug("Monitor: plotting labels unavailable (%s); using raw names.", exc)
        return {}


def _dataset_order() -> List[str]:
    try:
        from reasondb.evaluation.plotting import DATASET_ORDER

        return list(DATASET_ORDER)
    except Exception:  # pragma: no cover - depends on the environment
        return []


def presentation() -> Dict[str, Any]:
    """Labels, ordering and palettes, served to the UI so it never redefines them."""
    return {
        "label_map": _label_map(),
        "dataset_order": _dataset_order(),
        "breakdown_components": [
            {"column": col, "label": label, "color": color}
            for (col, label), color in zip(
                RUNTIME_BREAKDOWN_COMPONENTS, BREAKDOWN_PALETTE
            )
        ],
        "palette": BREAKDOWN_PALETTE,
        "cost_parts": list(COST_PARTS),
        "cost_types": list(COST_TYPES),
    }


def discover_result_dirs(roots: List[Path], max_depth: int = 5) -> List[Dict[str, Any]]:
    """Find every directory under ``roots`` holding at least one ``*metrics.csv``.

    Returns one record per directory with the benchmark/split inferred from the path
    tail (``<output-dir>/<benchmark>/<split>``), newest first. ``max_depth`` bounds the
    walk so pointing this at a large tree cannot take unbounded time.
    """
    assert max_depth >= 1, f"max_depth must be at least 1; got {max_depth}."
    seen: Dict[str, Dict[str, Any]] = {}
    patterns = ["*metrics.csv"] + [
        "/".join(["*"] * depth + ["*metrics.csv"]) for depth in range(1, max_depth + 1)
    ]
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for csv_path in sorted(
            {p for pattern in patterns for p in root.glob(pattern)}
        ):
            directory = csv_path.parent
            key = str(directory.resolve())
            record = seen.get(key)
            if record is None:
                parts = directory.resolve().parts
                record = {
                    "dir": key,
                    "display": str(directory),
                    "benchmark": parts[-2] if len(parts) >= 2 else None,
                    "split": parts[-1] if parts else None,
                    "files": [],
                    "mtime": 0.0,
                }
                seen[key] = record
            try:
                stat = csv_path.stat()
            except OSError:  # pragma: no cover - racing with a running benchmark
                continue
            record["files"].append(
                {
                    "name": csv_path.name,
                    "labels_type": csv_path.stem,
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
            record["mtime"] = max(record["mtime"], stat.st_mtime)

    out = list(seen.values())
    for record in out:
        record["files"].sort(key=lambda f: f["name"])
        record["has_operator_stats"] = (Path(record["dir"]) / "operator_stats.csv").is_file()
        record["has_pipeline_tracks"] = (
            Path(record["dir"]) / "pipeline_tracks.yaml"
        ).is_file()
    out.sort(key=lambda r: -r["mtime"])
    return out


def load_result_dir(directory: Path) -> Dict[str, Any]:
    """Load every ``*metrics.csv`` in ``directory`` plus its operator stats.

    Never raises: a problem with one file is reported in the payload.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return {"error": f"Not a directory: {directory}", "dir": str(directory)}

    csv_paths = sorted(directory.glob("*metrics.csv"))
    if not csv_paths:
        return {
            "error": f"No *metrics.csv found in {directory}. Has the run finished?",
            "dir": str(directory),
        }

    frames: List[pd.DataFrame] = []
    problems: List[str] = []
    for path in csv_paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            problems.append(f"{path.name}: {exc}")
            continue
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            problems.append(f"{path.name}: missing column(s) {', '.join(missing)}")
            continue
        if frame.empty:
            problems.append(f"{path.name}: no rows")
            continue
        frame["labels_type"] = path.stem
        frames.append(frame)

    if not frames:
        return {
            "error": "No usable metrics CSV in "
            f"{directory}: {'; '.join(problems) if problems else 'unknown reason'}",
            "dir": str(directory),
            "problems": problems,
        }

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = _backfill_guarantees(df)
    df = _add_derived(df)

    return {
        "dir": str(directory),
        "problems": problems,
        "rows": _records(df),
        "approaches": sorted(df["approach_name"].dropna().unique().tolist()),
        "labels_types": sorted(df["labels_type"].dropna().unique().tolist()),
        "guarantee_settings": sorted(df["guarantee_setting"].dropna().unique().tolist()),
        "cost_columns": [c for c in df.columns if c.endswith(tuple(COST_TYPES))],
        "time_columns": [c for c in df.columns if c.startswith("time_")],
        # Query shape statistics, present once evaluate() was given query_stats. Read
        # off the CSV rather than recomputed: the Results tab has no benchmark to ask.
        "facet_columns": [c for c in QUERY_STAT_COLUMNS if c in df.columns],
        "operator_stats": _load_operator_stats(directory / "operator_stats.csv"),
        "pipeline_tracks_available": (directory / "pipeline_tracks.yaml").is_file(),
    }


def load_pipeline_track(directory: Path, key: str) -> Dict[str, Any]:
    """One entry out of ``pipeline_tracks.yaml``, addressed by ``pipeline_track_key``."""
    path = Path(directory) / "pipeline_tracks.yaml"
    if not path.is_file():
        return {"error": f"No pipeline_tracks.yaml in {directory}"}
    try:
        import yaml

        with open(path, "r") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:
        return {"error": f"Could not read {path}: {exc}"}
    if key not in data:
        return {"error": f"No pipeline track for key {key!r}", "keys": sorted(data)[:50]}
    return {"key": key, "track": _jsonable(data[key])}


def _load_operator_stats(path: Path) -> Dict[str, Any]:
    """``operator_stats.csv`` as heatmap-ready records (approach x operator counts)."""
    if not path.is_file():
        return {"rows": [], "available": False}
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return {"rows": [], "available": False, "error": f"{path.name}: {exc}"}
    if frame.empty:
        return {"rows": [], "available": True}
    expected = {"approach", "operator", "count"}
    missing = expected - set(frame.columns)
    if missing:
        return {
            "rows": [],
            "available": True,
            "error": f"{path.name}: missing column(s) {', '.join(sorted(missing))}",
        }
    return {"rows": _records(frame), "available": True}


def _backfill_guarantees(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror ``plot_benchmark.load_df``: no-guarantee rows adopt the lowest target.

    Baselines like ``gpt`` and the fixed-compression executors run without guarantees and
    write NaN there. Charting them against the guaranteed approaches needs them on some
    level; the plotting script picks the lowest observed one and so do we.
    """
    for column in ("precision_guarantee", "recall_guarantee"):
        observed = df[column].dropna().unique()
        if len(observed):
            df[column] = df[column].fillna(sorted(observed)[0])
    return df


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the charts need but ``evaluate()`` does not write."""
    df["guarantee_setting"] = (
        "p:"
        + df["precision_guarantee"].astype(str)
        + "_r:"
        + df["recall_guarantee"].astype(str)
    )
    # The ratio ``plot_meets_target`` charts: >= 1.0 means the guarantee was met.
    df["precision_met"] = _safe_ratio(df["precision"], df["precision_guarantee"])
    df["recall_met"] = _safe_ratio(df["recall"], df["recall_guarantee"])

    named = [col for col, _ in RUNTIME_BREAKDOWN_COMPONENTS if col != "time_other"]
    if "time_end_to_end" in df.columns and any(c in df.columns for c in named):
        summed = sum(
            df[col].fillna(0.0) for col in named if col in df.columns
        )
        df["time_other"] = (df["time_end_to_end"].fillna(0.0) - summed).clip(lower=0.0)
    return df


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator.replace(0.0, pd.NA)
    return ratio.astype(float)


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame to JSON-safe records: NaN/Inf become ``None``, everything else str-able."""
    return [
        {key: _jsonable(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, (int, str, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if pd.isna(value):  # numpy NaT / pandas NA
        return None
    try:
        return value.item()  # numpy scalars
    except AttributeError:
        pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def load_telemetry_sidecar(path: Path, limit: int = 5000) -> Dict[str, Any]:
    """Read back a ``telemetry-*.jsonl`` written by a previous run.

    Only the last ``limit`` lines are kept, so opening a multi-gigabyte sidecar from a
    long run cannot exhaust memory.
    """
    path = Path(path)
    if not path.is_file():
        return {"error": f"No telemetry file at {path}"}
    events: List[Dict[str, Any]] = []
    skipped = 0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # A run killed mid-write leaves a truncated final line.
                    skipped += 1
                    continue
                if len(events) > limit:
                    del events[0]
    except OSError as exc:
        return {"error": f"Could not read {path}: {exc}"}
    return {"path": str(path), "events": events, "malformed_lines": skipped}


def discover_telemetry_sidecars(roots: List[Path]) -> List[Dict[str, Any]]:
    """Every ``<root>/**/_monitor/telemetry-*.jsonl``, newest first."""
    out: List[Dict[str, Any]] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in root.glob("**/_monitor/telemetry-*.jsonl"):
            try:
                stat = path.stat()
            except OSError:  # pragma: no cover
                continue
            out.append(
                {
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
    out.sort(key=lambda r: -r["mtime"])
    return out
