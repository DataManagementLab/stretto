"""Standalone dashboard: browse finished runs without starting a benchmark.

    python -m reasondb.monitor --output-dir benchmark_results

Serves the same UI as a live run, with the Run tab reporting "standalone" and the
Results tab reading whatever ``*metrics.csv`` and ``telemetry-*.jsonl`` files it finds
under the given roots.

    python -m reasondb.monitor --replay benchmark_results/.../_monitor/telemetry-*.jsonl

Replays a telemetry sidecar into a live ``Collector`` and serves the dashboard against
it. This is how the UI is exercised without occupying a GPU for hours: a sidecar from a
real multi-iteration (or multi-worker) run drives every tab with real data, at whatever
speed you ask for.
"""

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from reasondb.monitor.collector import Collector
from reasondb.monitor.replay import replay_into
from reasondb.monitor.server import DEFAULT_PORT, start_server

logger = logging.getLogger(__name__)


def replay(collector: Collector, path: Path, speed: float, stop: threading.Event) -> int:
    """Feed one sidecar's events back into ``collector``.

    A thin wrapper on :func:`reasondb.monitor.replay.replay_into`, which is the single
    implementation shared with the startup seeding in ``session.py``. Deliberately does
    *not* stamp a ``run_id``: a viewer opening one file is looking at one run, and
    leaving it unstamped keeps the collector in its single-run mode, where every worker
    and every record counts as current.
    """
    return replay_into(collector, path, speed=speed, stop=stop)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        nargs="+",
        default=[Path("benchmark_results")],
        help="Result roots to browse. Repeatable.",
    )
    parser.add_argument(
        "--monitor-port",
        type=int,
        default=None,
        help=f"Port to serve on (default {DEFAULT_PORT}, or $REASONDB_MONITOR_PORT).",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="A telemetry-*.jsonl sidecar to replay into a live collector.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help=(
            "Replay speed multiplier: 0 (default) loads everything at once, 1 is real "
            "time, 60 replays an hour a minute."
        ),
    )
    args = parser.parse_args()

    missing = [str(p) for p in args.output_dir if not Path(p).is_dir()]
    if missing:
        logger.warning("Monitor: result root(s) do not exist yet: %s", ", ".join(missing))

    collector: Optional[Collector] = None
    stop = threading.Event()
    if args.replay is not None:
        if not args.replay.is_file():
            logger.error("Monitor: no telemetry file at %s", args.replay)
            return 1
        # Not install()ed: nothing in this process produces events, and leaving the
        # global sink unset keeps the replay from being confused with a real run.
        collector = Collector(jsonl_path=None).start()

    server = start_server(
        collector=collector,
        result_roots=list(args.output_dir),
        run_info={
            "run_id": "replay" if collector else "standalone",
            "script": "reasondb.monitor",
            "replay_of": str(args.replay) if args.replay else None,
        },
        preferred_port=args.monitor_port,
    )
    if server is None:
        logger.error("Monitor: could not bind a port; nothing to serve.")
        if collector is not None:
            collector.close()
        return 1

    print(f"[monitor] dashboard: {server.url}  (Ctrl-C to stop)", flush=True)

    replay_thread = None
    if collector is not None:
        replay_thread = threading.Thread(
            target=lambda: logger.info(
                "Monitor: replayed %d event(s) from %s",
                replay(collector, args.replay, args.speed, stop),
                args.replay,
            ),
            name="reasondb-monitor-replay",
            daemon=True,
        )
        replay_thread.start()

    try:
        while server.failed is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        stop.set()
        if collector is not None:
            collector.close()
    logger.error("Monitor: server stopped: %s", server.failed)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
