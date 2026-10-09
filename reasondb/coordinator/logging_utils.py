"""Make a coordinator/worker process a well-behaved background daemon in one call:
create its log directory, print a ``tail -f`` hint on the *original* terminal, ignore
``SIGHUP`` so it survives an SSH disconnect even backgrounded with a plain ``&``, and
redirect stdout/stderr (prints, tracebacks, Flask/Werkzeug's own banner - anything
that doesn't go through ``logging``, not just ``logging`` calls) to a file.

Two files, in fact: the per-request HTTP log is split off into a ``.http`` sidecar (see
:data:`HTTP_LOGGERS`), because it is far more voluminous than the main log.

Deliberately not built on ``reasondb.utils.logging.FileLogger`` for this: that class
requires its ``log_root_path`` nested under a cwd-relative ``"logging"`` directory (its
logger-name derivation does ``path.relative_to(Path("logging"))``, which raises for any
other path shape), the wrong shape for a log the operator wants on
the shared filesystem next to the coordinator's SQLite DB, or under a worker's own
``--worker-dir``. Per-job/per-backend logging (inside ``Executor``) *does* fit
FileLogger's contract and uses it directly - see each producer's ``run_job``.

No ``nohup ... > file 2>&1 &`` or ``mkdir -p`` is needed on the command line:
``python scripts/run_coordinator.py --task-id X ... &`` is enough - the script makes
itself immune to the hangup and finds its own log file.
"""

import logging
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

#: Loggers whose records describe the HTTP transport rather than what the process is
#: doing, and which are therefore split into their own file. The coordinator's Flask app
#: (``werkzeug``) logs one line per request and every worker polls it, which would drown
#: out everything else. The client-side two are quiet at INFO and would flood the same
#: way under ``--log-level debug``.
HTTP_LOGGERS = ("werkzeug", "urllib3", "requests")

#: The handler currently receiving them, so a *move* (``redirect_log`` called again when
#: a worker switches task) takes the sidecar with it instead of leaving it writing to the
#: previous task's directory.
_http_handler: Optional[logging.Handler] = None


def http_log_path(log_path: Path) -> Path:
    """``.../coordinator.log`` -> ``.../coordinator.http.log``.

    Beside the main log rather than under it, and sharing its stem, so the two are
    adjacent in a directory listing and a ``tail -f`` on either is an obvious guess.
    """
    log_path = Path(log_path)
    return log_path.with_name(f"{log_path.stem}.http{log_path.suffix}")


def _split_http_logs(log_path: Path) -> Path:
    """Route :data:`HTTP_LOGGERS` to their own file instead of the main log.

    ``propagate = False`` is what does the splitting: the root handler writes to
    ``sys.stderr``, which :func:`_redirect_stdio` has pointed at the main log, so a
    record that reaches the root ends up in both files. These are moved, not copied.

    ``delay=True`` so a run that serves no HTTP at all (``--local``) leaves no empty
    file behind.
    """
    global _http_handler
    target = http_log_path(log_path)
    handler = logging.FileHandler(target, mode="a", delay=True)
    handler.setFormatter(logging.Formatter(_FORMAT))
    for name in HTTP_LOGGERS:
        one = logging.getLogger(name)
        if _http_handler is not None:
            one.removeHandler(_http_handler)
        one.addHandler(handler)
        one.propagate = False
    if _http_handler is not None:
        _http_handler.close()
    _http_handler = handler
    return target


def _ignore_sighup() -> None:
    """Best-effort: SIGHUP doesn't exist on every platform (e.g. Windows), and this
    must never be the reason a background daemon fails to start."""
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except (AttributeError, ValueError, OSError):
        pass


def _redirect_stdio(log_path: Path) -> None:
    """Point fd 1 and fd 2 at ``log_path`` (append mode, line-buffered) so everything
    written to stdout/stderr from here on - by this process and anything it forks -
    lands in the file, not a terminal that may no longer exist by the time it's read.
    """
    fh = open(log_path, "a", buffering=1)
    os.dup2(fh.fileno(), sys.stdout.fileno())
    os.dup2(fh.fileno(), sys.stderr.fileno())
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)


def redirect_log(log_path: Path, prefix: str = "") -> None:
    """Create ``log_path``'s directory and point this process's stdout/stderr at it.

    Safe to call again later to *move* the log - a worker does this when it switches to
    the next task, so each task's log lands under that task's own ``--worker-dir``. The
    hint line prints before the switch, so the old file ends with a pointer to the new
    one. No second ``logging.basicConfig`` is needed after a move: the root handler holds
    ``sys.stderr`` and ``dup2`` re-points fd 2 underneath it, so existing loggers follow.
    The HTTP sidecar has no such indirection, so :func:`_split_http_logs` re-points it
    explicitly - the move must take both files or a worker's request log would keep
    growing in the previous task's directory.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[reasondb] {prefix}logging to {log_path} - watch it with: tail -f {log_path}", flush=True)
    # Before the redirect, like the line above and for the same reason: on the first call
    # it reaches the operator's terminal, and on a move it ends the *old* file with a
    # pointer to where both logs continue.
    print(f"[reasondb] per-request HTTP logs go to {http_log_path(log_path)} instead", flush=True)
    _redirect_stdio(log_path)
    _split_http_logs(log_path)


def log_job_exception(logger: logging.Logger, job_id: str, exc: BaseException) -> None:
    """Write the full traceback of a job-killing exception to this worker's log.

    ``JobResult.from_exception`` keeps the DB's ``error`` column to one line, which is
    right for the dashboard and wrong for debugging: an ``assert`` deep inside an
    operator says *which* predicate failed but not how execution got there. The
    traceback is only useful next to the job's other log lines, so it goes here - to
    ``<worker-dir>/logging/worker.log`` - rather than over the wire.
    """
    logger.error("Job %s raised %s:\n%s", job_id, type(exc).__name__, traceback.format_exc())


def daemonize(log_path: Path, level: int = logging.INFO) -> None:
    """Call once, near the top of ``main()``, after parsing args (so ``log_path`` can
    depend on them) but before doing anything worth logging.

    Order matters: the ``tail -f`` hint prints to the *real* terminal first (the one
    line the operator sees before the process goes quiet from their point of view),
    then SIGHUP is ignored, then stdio is redirected, then ``logging`` is configured
    to match - so a log line emitted the instant after this call already lands in
    the file with a timestamp.
    """
    _ignore_sighup()
    redirect_log(log_path, prefix="backgrounding; ")
    logging.basicConfig(level=level, format=_FORMAT, force=True)
