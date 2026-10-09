"""The monitor's HTTP surface: a Flask app on a daemon thread.

Built with ``Flask`` + ``flask_restful`` and exposing ``/status``, matching the six
inference servers in :mod:`reasondb.backends` — same shape, same conventions, no new
dependency.

Endpoints split by cost, not by topic. ``/api/run`` is small and polled every second;
``/api/aggregates`` is larger and polled every few; ``/api/queries/detail`` holds the
per-query tuned pipelines, which are far too large to poll at all and are fetched one
query at a time on demand.

Two properties this module must never violate:

* **It cannot slow the run down.** Every request reads a snapshot under a short-lived
  lock; no request touches the producer's path.
* **It cannot end the run.** The serving thread swallows everything, and a failure to
  bind simply means no monitor.
"""

import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from reasondb.monitor import dimensions as monitor_dimensions
from reasondb.monitor import results as results_mod
from reasondb.monitor.collector import Collector
from reasondb.monitor.events import DEFAULT_LOG_TYPES, EVENT_TYPES

logger = logging.getLogger(__name__)

#: The monitor's home port. Deliberately outside the 5005-5324 block the model servers
#: occupy (see ``reasondb/backends``), and not an IANA-registered service.
DEFAULT_PORT = 5099

#: How many consecutive ports to try when the default is taken, so a second concurrent
#: run gets its own dashboard instead of silently losing telemetry.
PORT_SCAN = 10

HOST = "127.0.0.1"
STATIC_DIR = Path(__file__).parent / "static"

MAX_EVENT_LIMIT = 5000


def resolve_port(preferred: Optional[int] = None) -> int:
    """CLI value, else ``REASONDB_MONITOR_PORT``, else :data:`DEFAULT_PORT`."""
    if preferred is not None:
        port = int(preferred)
    else:
        env = os.environ.get("REASONDB_MONITOR_PORT")
        port = int(env) if env else DEFAULT_PORT
    assert 1024 <= port <= 65535, (
        f"Monitor port must be an unprivileged port in [1024, 65535]; got {port}."
    )
    return port


def find_free_port(preferred: int, scan: int = PORT_SCAN, host: str = HOST) -> Optional[int]:
    """First bindable port in ``[preferred, preferred + scan)``, or ``None``.

    Probing beats catching a failure inside the serving thread: by the time Werkzeug
    raises there, we are on a thread whose exception nobody sees. ``host`` defaults to
    loopback; the coordinator passes a wider bind address (e.g. ``"0.0.0.0"``) so
    workers on other machines can reach it.
    """
    assert scan >= 1, f"scan must be at least 1; got {scan}."
    for candidate in range(preferred, min(preferred + scan, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
            except OSError:
                continue
            return candidate
    return None


def create_app(
    collector: Optional[Collector],
    result_roots: List[Path],
    run_info: Optional[Dict[str, Any]] = None,
) -> Flask:
    """Build the Flask app.

    ``collector`` is ``None`` in standalone mode (``python -m reasondb.monitor``), where
    only the results-browsing endpoints have anything to serve.
    """
    assert STATIC_DIR.is_dir(), (
        f"Monitor static assets missing at {STATIC_DIR}; the package is incomplete."
    )
    roots = [Path(r) for r in result_roots]
    info = dict(run_info or {})
    app = Flask(__name__, static_folder=None)
    started_at = time.time()

    # ── UI assets ──────────────────────────────────────────────────────────────

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:filename>")
    def static_file(filename: str):
        return send_from_directory(STATIC_DIR, filename)

    # ── House-convention health check ──────────────────────────────────────────

    @app.get("/status")
    def status():
        return jsonify(
            {
                "status": "alive",
                "service": "reasondb-monitor",
                "run_id": info.get("run_id"),
                "started_at": started_at,
                "live": collector is not None,
                "pid": os.getpid(),
            }
        )

    # ── Live run ───────────────────────────────────────────────────────────────

    @app.get("/api/run")
    def api_run():
        if collector is None:
            return jsonify(
                {"live": False, "status": "standalone", "run": info, "counts": {}}
            )
        payload = collector.snapshot_run()
        payload["live"] = True
        payload["run"] = {**info, **payload.get("run", {})}
        return jsonify(payload)

    @app.get("/api/events")
    def api_events():
        if collector is None:
            return jsonify({"events": [], "next_seq": 0, "latest_seq": 0, "live": False})
        try:
            since = int(request.args.get("since", 0))
            limit = int(request.args.get("limit", 500))
        except ValueError:
            return jsonify({"error": "'since' and 'limit' must be integers"}), 400
        if since < 0:
            return jsonify({"error": "'since' must be non-negative"}), 400
        if not 1 <= limit <= MAX_EVENT_LIMIT:
            return jsonify({"error": f"'limit' must be in [1, {MAX_EVENT_LIMIT}]"}), 400

        raw_types = request.args.getlist("type")
        if len(raw_types) == 1 and "," in raw_types[0]:
            raw_types = raw_types[0].split(",")
        types = [t for t in (s.strip() for s in raw_types) if t]
        unknown = [t for t in types if t not in EVENT_TYPES]
        if unknown:
            return jsonify({"error": f"unknown event type(s): {', '.join(unknown)}"}), 400

        payload = collector.events_since(since=since, limit=limit, types=types or None)
        payload["live"] = True
        return jsonify(payload)

    @app.get("/api/aggregates")
    def api_aggregates():
        if collector is None:
            return jsonify(
                {
                    "live": False,
                    "phases": {},
                    "operators": [],
                    "operator_buckets": [],
                    "query_times": [],
                    "kv": [],
                }
            )
        payload = collector.snapshot_aggregates()
        payload["live"] = True
        return jsonify(payload)

    @app.get("/api/queries")
    def api_queries():
        """Index for the Query tab's picker. Deliberately light: the per-configuration
        plans behind each row are an order of magnitude larger and are fetched one
        query at a time by ``/api/queries/detail``."""
        if collector is None:
            return jsonify({"live": False, "queries": []})
        payload = collector.snapshot_queries()
        payload["live"] = True
        return jsonify(payload)

    @app.get("/api/search-space")
    def api_search_space():
        """Every logical step's candidate operators, per configuration."""
        if collector is None:
            return jsonify({"live": False, "steps": []})
        payload = collector.snapshot_search_space()
        payload["live"] = True
        return jsonify(payload)

    @app.get("/api/optimizer")
    def api_optimizer():
        """Every GD solve outcome: which restart won, and how many could have."""
        if collector is None:
            return jsonify({"live": False, "solves": []})
        payload = collector.snapshot_optimizer_solves()
        payload["live"] = True
        return jsonify(payload)

    @app.get("/api/queries/detail")
    def api_query_detail():
        if collector is None:
            return jsonify({"live": False, "runs": []})
        key = request.args.get("query")
        if not key:
            return jsonify({"error": "missing required 'query' parameter"}), 400
        payload = collector.snapshot_query_detail(key)
        payload["live"] = True
        return jsonify(payload)

    # ── Results on disk ────────────────────────────────────────────────────────

    @app.get("/api/results/dirs")
    def api_result_dirs():
        return jsonify(
            {
                "roots": [str(r) for r in roots],
                "dirs": results_mod.discover_result_dirs(roots),
                "telemetry": results_mod.discover_telemetry_sidecars(roots),
            }
        )

    @app.get("/api/results")
    def api_results():
        directory = request.args.get("dir")
        if not directory:
            return jsonify({"error": "missing required 'dir' parameter"}), 400
        resolved = _guard_path(directory, roots)
        if resolved is None:
            return jsonify(
                {"error": f"'{directory}' is outside the configured result roots"}
            ), 403
        return jsonify(results_mod.load_result_dir(resolved))

    @app.get("/api/results/pipeline")
    def api_pipeline_track():
        directory = request.args.get("dir")
        key = request.args.get("key")
        if not directory or not key:
            return jsonify({"error": "'dir' and 'key' are both required"}), 400
        resolved = _guard_path(directory, roots)
        if resolved is None:
            return jsonify(
                {"error": f"'{directory}' is outside the configured result roots"}
            ), 403
        return jsonify(results_mod.load_pipeline_track(resolved, key))

    @app.get("/api/results/telemetry")
    def api_telemetry():
        path = request.args.get("path")
        if not path:
            return jsonify({"error": "missing required 'path' parameter"}), 400
        resolved = _guard_path(path, roots)
        if resolved is None:
            return jsonify({"error": f"'{path}' is outside the configured result roots"}), 403
        return jsonify(results_mod.load_telemetry_sidecar(resolved))

    # ── Presentation constants ─────────────────────────────────────────────────

    @app.get("/api/presentation")
    def api_presentation():
        payload = results_mod.presentation()
        payload["event_types"] = sorted(EVENT_TYPES)
        payload["default_log_types"] = list(DEFAULT_LOG_TYPES)
        # The dimension vocabulary lives in Python so one table serves the dashboard and
        # can be asserted against what the producers actually emit; see
        # reasondb/monitor/dimensions.py.
        payload.update(monitor_dimensions.presentation_payload())
        return jsonify(payload)

    @app.errorhandler(HTTPException)
    def on_http_error(exc: HTTPException):
        # Let Flask's own 404/405/etc. through unchanged - only unexpected handler bugs
        # get rewritten into the JSON shape below.
        return exc

    @app.errorhandler(Exception)
    def on_error(exc: Exception):
        # A handler bug must render as a message in the UI, never as a dead dashboard.
        logger.warning("Monitor: request %s failed: %s", request.path, exc)
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return app


def _guard_path(candidate: str, roots: List[Path]) -> Optional[Path]:
    """Confine filesystem parameters to the configured roots.

    The server binds to loopback, but the browser is an untrusted-ish client: any page
    the user has open can reach 127.0.0.1. Refusing paths outside the result roots keeps
    a stray request from reading arbitrary files.
    """
    try:
        resolved = Path(candidate).resolve()
    except OSError:
        return None
    for root in roots:
        try:
            resolved.relative_to(Path(root).resolve())
            return resolved
        except (ValueError, OSError):
            continue
    return None


class MonitorServer:
    """Owns the daemon thread the Flask app runs on."""

    def __init__(self, app: Flask, port: int, host: str = HOST) -> None:
        self.app = app
        self.port = port
        self.host = host
        # The advertised URL always uses loopback for a loopback bind; a wider bind
        # (coordinator) still prints something dereferenceable rather than "0.0.0.0".
        self.url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
        self.failed: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "MonitorServer":
        assert self._thread is None, "MonitorServer.start() called twice."
        self._thread = threading.Thread(
            target=self._serve, name="reasondb-monitor-http", daemon=True
        )
        self._thread.start()
        return self

    def _serve(self) -> None:
        try:
            # ``use_reloader=False`` is mandatory off the main thread; ``threaded=True``
            # keeps a slow results load from blocking the live polling.
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                threaded=True,
                use_reloader=False,
            )
        except Exception as exc:  # never propagate: this thread's death is not the run's
            self.failed = f"{type(exc).__name__}: {exc}"
            logger.warning("Monitor: HTTP server stopped (%s).", self.failed)


def start_server(
    collector: Optional[Collector],
    result_roots: List[Path],
    run_info: Optional[Dict[str, Any]] = None,
    preferred_port: Optional[int] = None,
    host: str = HOST,
) -> Optional[MonitorServer]:
    """Bind and serve, or return ``None`` and let the run continue unmonitored.

    ``host`` defaults to loopback, as used by the ``run_benchmark*`` scripts.
    The coordinator (``scripts/run_coordinator.py``) passes a wider bind address so
    workers on other machines can reach ``/api/jobs/*`` etc.
    """
    port = resolve_port(preferred_port)
    chosen = find_free_port(port, host=host)
    if chosen is None:
        logger.warning(
            "Monitor: ports %d-%d are all in use; running without the dashboard.",
            port,
            port + PORT_SCAN - 1,
        )
        return None
    if chosen != port:
        logger.warning(
            "Monitor: port %d is in use, serving on %d instead.", port, chosen
        )
    try:
        app = create_app(collector, result_roots, run_info)
    except Exception as exc:
        logger.warning("Monitor: could not build the dashboard app (%s); skipping.", exc)
        return None
    return MonitorServer(app, chosen, host=host).start()
