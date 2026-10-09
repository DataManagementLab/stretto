"""One context manager that turns monitoring on for a benchmark run.

Every ``run_benchmark*`` script wraps its work in :func:`monitor_session`. Inside it, the
``record_*`` functions in :mod:`reasondb.monitor.collector` become live; outside it they
are a single ``is None`` check.

The session owns three things and guarantees none of them can end the run:

* the :class:`~reasondb.monitor.collector.Collector` and its JSONL sidecar,
* the HTTP server thread,
* ``SIGINT``/``SIGTERM`` handlers that flush telemetry before the previous handler runs.

Signal handling is additive: the previous handler (by default Python's own) is chained
to rather than replaced. A ``Ctrl-C`` during a long
run therefore still raises ``KeyboardInterrupt``, but the sidecar is complete.
"""

import argparse
import logging
import signal
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence

from reasondb.monitor.collector import (
    Collector,
    default_sidecar_path,
    make_run_id,
    record_run_end,
    record_run_start,
)
if TYPE_CHECKING:  # pragma: no cover - typing only
    from reasondb.monitor.server import MonitorServer

logger = logging.getLogger(__name__)


def _start_server(**kwargs):
    """Import flask lazily: a missing dashboard must not cost the JSONL telemetry."""
    try:
        from reasondb.monitor.server import start_server
    except ImportError as exc:
        logger.warning(
            "Monitor: flask is unavailable (%s); collecting telemetry to disk only.", exc
        )
        return None
    return start_server(**kwargs)


def _seed_history(collector: Collector, out_dir: Path, sidecar_path: Path) -> Dict[str, Any]:
    """Replay this output directory's earlier runs into a freshly built collector.

    The dashboard's live state is per *process*, while the work it watches is per
    *task*: the coordinator's job database survives a restart, so a restarted
    coordinator replays the earlier sidecars to show the work done so far.

    Never fatal: any error here degrades to an empty dashboard and a warning.
    """
    from reasondb.monitor.replay import seed_from_sidecars, sidecar_dir

    try:
        seeded = seed_from_sidecars(collector, out_dir, exclude=sidecar_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Monitor: could not seed earlier runs (%s); starting empty.", exc)
        return {"runs": [], "events": 0, "files": 0, "errors": [{"error": str(exc)}]}
    if seeded["events"]:
        logger.info(
            "Monitor: seeded %d event(s) from %d earlier run(s) in %s",
            seeded["events"],
            seeded["files"],
            sidecar_dir(out_dir),
        )
    return seeded


class MonitorHandle:
    """What a live session hands back. Inert when monitoring is off."""

    def __init__(
        self,
        enabled: bool,
        run_id: Optional[str] = None,
        collector: Optional[Collector] = None,
        server: Optional["MonitorServer"] = None,
    ) -> None:
        self.enabled = enabled
        self.run_id = run_id
        self.collector = collector
        self.server = server

    @property
    def url(self) -> Optional[str]:
        return self.server.url if self.server is not None else None


@contextmanager
def monitor_session(
    args: argparse.Namespace,
    output_dir: Optional[Path] = None,
    script: Optional[str] = None,
    extra_result_roots: Sequence[Path] = (),
) -> Iterator[MonitorHandle]:
    """Run the enclosed block with the monitor live.

    ``args`` is the parsed namespace of a ``run_benchmark*`` script; the flags added by
    ``reasondb.utils.benchmark_args.add_monitor_arguments`` are read off it, and every
    other attribute is echoed into the dashboard's run header so it is obvious which
    invocation you are looking at.
    """
    if getattr(args, "no_monitor", False):
        yield MonitorHandle(enabled=False)
        return

    out_dir = Path(output_dir or getattr(args, "output_dir", None) or "benchmark_results")
    run_id = make_run_id()
    roots = _result_roots(out_dir, extra_result_roots)

    sidecar_path = default_sidecar_path(out_dir, run_id)
    collector: Optional[Collector] = None
    server: Optional["MonitorServer"] = None
    restore_signals = None
    seeded: Dict[str, Any] = {"runs": [], "events": 0, "files": 0, "errors": []}
    try:
        collector = Collector(jsonl_path=sidecar_path, run_id=run_id).install()
        seeded = _seed_history(collector, out_dir, sidecar_path)
        server = _start_server(
            collector=collector,
            result_roots=roots,
            run_info={
                "run_id": run_id,
                "script": script,
                "output_dir": str(out_dir),
                "seeded": seeded,
            },
            preferred_port=getattr(args, "monitor_port", None),
        )
        restore_signals = _install_signal_flush(collector)
    except Exception as exc:
        # Monitoring is never worth failing a benchmark over.
        logger.warning("Monitor: could not start (%s); running unmonitored.", exc)
        if collector is not None:
            collector.close()
        yield MonitorHandle(enabled=False)
        return

    handle = MonitorHandle(
        enabled=True, run_id=run_id, collector=collector, server=server
    )
    if server is not None:
        logger.info("Monitor: dashboard at %s", server.url)
        print(f"[monitor] dashboard: {server.url}", flush=True)
    else:
        logger.info("Monitor: collecting telemetry to %s (no HTTP server)", collector.jsonl_path)

    # After seeding, so the live run_start is the last one the collector folds and the
    # header describes the run that is actually happening.
    record_run_start(
        run_id=run_id,
        script=script or Path(sys.argv[0]).name,
        argv=list(sys.argv[1:]),
        output_dir=str(out_dir),
        pid=_pid(),
        args=_describe_args(args),
        mode=_mode(args),
        seeded=seeded,
    )

    status, error = "ok", None
    try:
        yield handle
    except KeyboardInterrupt:
        status, error = "interrupted", "KeyboardInterrupt"
        raise
    except BaseException as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record_run_end(status=status, error=error, run_id=run_id)
        if restore_signals is not None:
            restore_signals()
        collector.close()
        if server is not None and server.failed:
            logger.warning("Monitor: HTTP server had failed: %s", server.failed)


def _result_roots(out_dir: Path, extra: Sequence[Path]) -> List[Path]:
    """Directories the Results tab may read from.

    The run's own output dir plus the repo-conventional siblings, so a dashboard started
    by one benchmark can still browse another's results.
    """
    roots: List[Path] = [Path(out_dir)]
    for candidate in (
        "benchmark_results",
        "benchmark_results_custom",
        *(str(p) for p in extra),
    ):
        path = Path(candidate)
        if path.is_dir() and path.resolve() not in {r.resolve() for r in roots if r.exists()}:
            roots.append(path)
    return roots


def _install_signal_flush(collector: Collector):
    """Flush telemetry on SIGINT/SIGTERM, then chain to whatever was installed before.

    Only possible on the main thread; a no-op elsewhere (e.g. under pytest-xdist).
    """
    if threading.current_thread() is not threading.main_thread():
        return None

    previous: Dict[int, Any] = {}

    def handler(signum, frame):
        try:
            collector.close(timeout=2.0)
        except Exception:  # pragma: no cover - best effort during teardown
            pass
        prior = previous.get(signum)
        if callable(prior):
            prior(signum, frame)
        elif prior == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)

    installed = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
            installed.append(signum)
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            continue

    def restore() -> None:
        for signum in installed:
            try:
                signal.signal(signum, previous[signum])
            except (ValueError, OSError):  # pragma: no cover
                pass

    return restore


def _describe_args(args: argparse.Namespace) -> Dict[str, str]:
    """Every parsed flag, stringified, for the dashboard's run header."""
    return {key: str(value) for key, value in sorted(vars(args).items())}


def _mode(args: argparse.Namespace) -> str:
    if getattr(args, "precompute", None) is not None:
        return "precompute"
    if getattr(args, "simulate", None) is not None:
        return "simulate"
    return "normal"


def _pid() -> int:
    import os

    return os.getpid()
