"""Read telemetry sidecars back into a live :class:`~reasondb.monitor.collector.Collector`.

Two callers, one mechanism:

* ``python -m reasondb.monitor --replay <file>`` feeds one past run into an empty
  collector and serves the dashboard against it, optionally paced to the original
  timings. This is how the UI is exercised without occupying a GPU.
* :func:`seed_from_sidecars`, called by :mod:`reasondb.monitor.session` at startup, feeds
  *every earlier run of the same output directory* into the collector a new run is about
  to use, so a restarted coordinator shows the work that came before it instead of an
  empty dashboard.

Three properties the seeding path depends on:

**Exactly once.** Only ``<output_dir>/_monitor/`` is read, never ``**/_monitor/``. A
worker's own sidecar lives under ``workers/<id>/_monitor/`` and its events are *already*
in the coordinator's file, forwarded there by :mod:`reasondb.coordinator.ingest`; a
recursive glob would count every worker event twice. Nothing here is idempotent -
``optimizer_solves`` and ``query_metrics`` are append-only by design - so the source set
has to be right by construction rather than deduplicated afterwards.

**Attribution.** Every seeded event is stamped with the run it came from before it is
handed over, so it can never be mistaken for the live one: the collector keys worker
state by ``(run, worker)`` and the dashboard gets a ``run_id`` dimension to group and
filter by. Sidecars whose events carry no ``run_id`` are stamped from their filename.

**Original timestamps.** Events keep their own ``t``, the same way a worker's forwarded
events do, so time-axis charts show when the work actually happened rather than when it
was read back.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from reasondb.monitor.collector import Collector
from reasondb.monitor.events import EVENT_TYPES

logger = logging.getLogger(__name__)

#: Sidecar filenames, as written by ``collector.default_sidecar_path``.
SIDECAR_PREFIX = "telemetry-"
SIDECAR_SUFFIX = ".jsonl"
SIDECAR_GLOB = f"{SIDECAR_PREFIX}*{SIDECAR_SUFFIX}"

#: Longest a bulk seed waits for the drain thread before giving up on backpressure and
#: pushing anyway (which costs dropped events, counted and surfaced, but never a hang).
_BACKPRESSURE_TIMEOUT_S = 30.0


def run_id_for_sidecar(path: Path) -> str:
    """The run id a sidecar belongs to, from its filename.

    The fallback for sidecars whose events do not carry a ``run_id`` of their own.
    ``telemetry-2026-07-30--08-46-15-10413.jsonl`` -> ``2026-07-30--08-46-15-10413``.
    """
    name = Path(path).name
    if name.startswith(SIDECAR_PREFIX):
        name = name[len(SIDECAR_PREFIX) :]
    if name.endswith(SIDECAR_SUFFIX):
        name = name[: -len(SIDECAR_SUFFIX)]
    return name or str(path)


def sidecar_dir(output_dir: Path) -> Path:
    """``<output_dir>/_monitor`` - the one directory a seed may read. See module docs."""
    return Path(output_dir) / "_monitor"


def sidecars_in(output_dir: Path, exclude: Optional[Path] = None) -> List[Path]:
    """Every sidecar directly under ``<output_dir>/_monitor``, oldest first.

    Deliberately not recursive: see the "exactly once" note in the module docstring.
    Ordered by modification time so runs are seeded in the order they happened; events
    carry their own timestamps regardless, so this only decides which run's ``run_start``
    the collector sees last.
    """
    directory = sidecar_dir(output_dir)
    if not directory.is_dir():
        return []
    excluded = Path(exclude).resolve() if exclude is not None else None
    found: List[Tuple[float, str, Path]] = []
    for path in directory.glob(SIDECAR_GLOB):
        if not path.is_file():
            continue
        if excluded is not None and path.resolve() == excluded:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:  # pragma: no cover - raced with a delete
            continue
        found.append((mtime, path.name, path))
    found.sort()
    return [path for _, _, path in found]


def iter_sidecar(path: Path) -> Iterator[Tuple[str, Dict[str, Any], float]]:
    """Yield ``(event_type, data, t)`` for each well-formed line of a sidecar.

    Skips anything malformed rather than raising: a run killed mid-write leaves a
    truncated final line, and one bad line must not cost the rest of the file.
    """
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind, data, t = event.get("type"), event.get("data"), event.get("t")
            if kind not in EVENT_TYPES or not isinstance(data, dict) or t is None:
                continue
            yield kind, data, t


def replay_into(
    collector: Collector,
    path: Path,
    speed: float = 0.0,
    stop: Optional[threading.Event] = None,
    run_id: Optional[str] = None,
) -> int:
    """Feed a sidecar's events back into ``collector``.

    ``speed`` is a multiplier on the original inter-event gaps: 0 replays everything at
    once (the whole run, instantly), 1 replays in real time, 60 replays an hour a minute.
    Gaps are capped at two seconds so a run that idled overnight between two events does
    not make the replay idle too.

    ``run_id`` stamps events that do not carry one, so they are attributed to their own
    run rather than merging into the live one.
    """
    assert speed >= 0, f"--speed must be non-negative; got {speed}."
    previous: Optional[float] = None
    replayed = 0
    for kind, data, t in iter_sidecar(path):
        if stop is not None and stop.is_set():
            break
        if run_id is not None and data.get("run_id") is None:
            data["run_id"] = run_id
        if speed and previous is not None:
            delay = (t - previous) / speed
            if delay > 0:
                if stop is not None:
                    stop.wait(min(delay, 2.0))
                else:  # pragma: no cover - the paced path always has a stop event
                    time.sleep(min(delay, 2.0))
        previous = t
        if not speed:
            _await_queue_room(collector, stop)
        collector.put(kind, data, at=t)
        replayed += 1
    return replayed


def _await_queue_room(collector: Collector, stop: Optional[threading.Event]) -> None:
    """Block while the drain thread is saturated.

    ``Collector.put`` drops events once ``max_queue`` is outstanding - correct for a
    benchmark, which must never stall on telemetry, and wrong here: a multi-million-event
    sidecar pushed at memory speed would lose most of itself to backpressure and the
    dashboard would show a run with holes in it. A finished file can simply wait. Bounded
    so a wedged drain thread degrades to the drop behaviour instead of hanging startup.
    """
    if collector.queue_depth < collector.max_queue // 2:
        return
    deadline = time.monotonic() + _BACKPRESSURE_TIMEOUT_S
    while collector.queue_depth >= collector.max_queue // 2:
        if time.monotonic() >= deadline:
            logger.warning(
                "Monitor: telemetry drain still saturated after %.0fs; seeding the "
                "remainder without backpressure (events may be dropped).",
                _BACKPRESSURE_TIMEOUT_S,
            )
            return
        if stop is not None and stop.is_set():
            return
        time.sleep(0.01)


def seed_from_sidecars(
    collector: Collector,
    output_dir: Path,
    exclude: Optional[Path] = None,
) -> Dict[str, Any]:
    """Prime ``collector`` with every earlier run recorded under ``output_dir``.

    Call once, before the live run announces itself, so the live ``run_start`` is the
    last one the collector folds and the header describes the run actually happening.

    Returns a summary ``{"runs": [...], "events": n, "files": n, "errors": [...]}`` for
    the run header - a merge the reader cannot see is a merge they will misread.
    """
    summary: Dict[str, Any] = {"runs": [], "events": 0, "files": 0, "errors": []}
    for path in sidecars_in(output_dir, exclude=exclude):
        run_id = run_id_for_sidecar(path)
        try:
            count = replay_into(collector, path, speed=0.0, run_id=run_id)
        except OSError as exc:
            # An unreadable sidecar is history lost, not a run that cannot start.
            logger.warning("Monitor: could not seed from %s (%s); skipping.", path, exc)
            summary["errors"].append({"path": str(path), "error": str(exc)})
            continue
        if not count:
            continue
        summary["files"] += 1
        summary["events"] += count
        summary["runs"].append({"run_id": run_id, "path": str(path), "events": count})
    return summary
