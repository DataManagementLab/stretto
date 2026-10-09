"""``redirect_log`` splits the HTTP request log off, and moves both files together.

A worker does this when it switches to the next ``--tasks`` entry, so each task's
worker.log lands under that task's own directory. Run in a subprocess: the function
dup2's onto fd 1/2, which would take pytest's own stdout with it.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_SCRIPT = """
import logging, sys
from reasondb.coordinator.logging_utils import daemonize, redirect_log

first, second = sys.argv[1], sys.argv[2]
daemonize(first)
logging.getLogger("t").info("in-first")
print("printed-first")
redirect_log(second)
logging.getLogger("t").info("in-second")
print("printed-second")
logging.getLogger("werkzeug").info("GET /claim 200")
"""


_HTTP_SCRIPT = """
import logging, sys
from reasondb.coordinator.logging_utils import daemonize

daemonize(sys.argv[1])
logging.getLogger("t").info("real-work")
logging.getLogger("werkzeug").info('POST /events HTTP/1.1" 200 -')
logging.getLogger("urllib3").info("outbound")
"""


def test_redirect_log_moves_the_log(tmp_path):
    first, second = tmp_path / "a" / "worker.log", tmp_path / "b" / "worker.log"
    subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(first), str(second)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )

    a, b = first.read_text(), second.read_text()
    # Everything before the move is in the first file, including the pointer to where
    # the log continues - the one line that makes a moved log followable.
    assert "in-first" in a and "printed-first" in a
    assert str(second) in a
    assert "in-second" not in a and "printed-second" not in a
    # Both `logging` (which holds the old sys.stderr object) and plain prints follow the
    # move, with no second basicConfig.
    assert "in-second" in b and "printed-second" in b


def test_request_logs_are_split_off_from_the_rest(tmp_path):
    """HTTP request lines go to their own file: in a coordinator log they vastly
    outnumber everything else and bury it."""
    from reasondb.coordinator.logging_utils import http_log_path

    log = tmp_path / "logging" / "coordinator.log"
    done = subprocess.run(
        [sys.executable, "-c", _HTTP_SCRIPT, str(log)],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )

    main, http = log.read_text(), http_log_path(log).read_text()
    assert "real-work" in main
    # Moved, not copied: the root handler writes to the redirected stderr, so a record
    # that still propagated would land in both files.
    assert "POST /events" not in main and "outbound" not in main
    assert "POST /events" in http and "outbound" in http
    assert "real-work" not in http
    # And the launching terminal is told where they went, beside the hint for the main
    # log, so nobody goes looking for them.
    assert str(http_log_path(log)) in done.stdout


def test_a_run_that_serves_no_http_leaves_no_sidecar(tmp_path):
    """`--local` runs no Flask app; an empty file beside the log would only mislead."""
    from reasondb.coordinator.logging_utils import http_log_path

    log = tmp_path / "logging" / "coordinator.log"
    subprocess.run(
        [sys.executable, "-c",
         "import sys\nfrom reasondb.coordinator.logging_utils import daemonize\n"
         "daemonize(sys.argv[1])\nprint('done')",
         str(log)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    assert log.is_file()
    assert not http_log_path(log).exists()


def test_moving_the_log_moves_the_http_sidecar_too(tmp_path):
    """A worker switching task must not keep writing requests into the old task's dir."""
    from reasondb.coordinator.logging_utils import http_log_path

    first, second = tmp_path / "a" / "worker.log", tmp_path / "b" / "worker.log"
    subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(first), str(second)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    assert not http_log_path(first).exists(), "nothing was logged there before the move"
    assert "GET /claim 200" in http_log_path(second).read_text()
    # The old log ends pointing at both of the new files, as it already did for the main
    # one - a moved log is only followable if it says where it went.
    assert str(http_log_path(second)) in first.read_text()
