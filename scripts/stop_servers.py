"""Report which reasondb backend servers are running on this machine, and stop them.

    python scripts/stop_servers.py --dry-run     # just say what is up
    python scripts/stop_servers.py               # ...and interrupt it
    python scripts/stop_servers.py --capability text
    python scripts/stop_servers.py --port 5010 5012

Rather than relying on recorded PIDs (which miss servers started by hand or by another
shell), this asks the known ports what is listening (via ``/status``, so it can also
report *which* model and which ``USE_INDICES`` mode each server uses) and resolves the
owner from the socket.

Signals escalate SIGINT -> SIGTERM -> SIGKILL, waiting between each. SIGINT first
because that is what these servers see when you Ctrl-C them in a terminal; SIGKILL last
because a hard-killed
KV server can leave a half-written cache file behind that the next run then has to
notice and redo.
"""

import argparse
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional

import psutil
import requests

from reasondb.coordinator.capabilities import (
    CAPABILITY_SCRIPTS,
    all_known_server_ports,
)
from reasondb.coordinator.models import WORKER_CAPABILITY_CHOICES

logger = logging.getLogger("stop_servers")

#: Seconds to wait for a process to go away after each signal before escalating. A KV
#: server can be mid-``prepare_caches`` on a large dataset when the signal lands, so the
#: polite signals get real time rather than a token half-second.
ESCALATION_WAIT_S = 15.0


class LiveServer:
    def __init__(self, port: int, label: str, status: Optional[dict], pids: List[int]):
        self.port = port
        self.label = label
        self.status = status or {}
        self.pids = pids

    @property
    def index_mode(self) -> str:
        if "use_relative_indices" not in self.status:
            return "n/a"
        return "indices" if self.status["use_relative_indices"] else "physical"

    def describe(self) -> str:
        model = self.status.get("model_name", "-")
        owner = ", ".join(str(p) for p in self.pids) or "unknown (not ours?)"
        return (
            f"  :{self.port:<5} {self.label:<44} model={model:<40} "
            f"caches={self.index_mode:<8} pid={owner}"
        )


def _pids_listening_on(port: int) -> List[int]:
    """PIDs with a LISTEN socket on `port`.

    Comes back empty for a server owned by another user (``net_connections`` hides the
    PID unless you are root) - the caller reports that rather than pretending nothing is
    there, since the port is demonstrably answering.
    """
    pids = []
    try:
        conns = psutil.net_connections(kind="inet")
    except psutil.AccessDenied:
        return []
    for c in conns:
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port and c.pid:
            pids.append(c.pid)
    return sorted(set(pids))


def discover(ports: Dict[int, str], timeout_s: float = 2.0) -> List[LiveServer]:
    """Everything currently answering, in port order.

    A port that accepts a connection but does not answer ``/status`` still counts: it is
    occupied, which is the thing that would make the next ``start_servers_*.sh`` fail to
    bind, whether or not it is a healthy reasondb server.
    """
    live = []
    for port in sorted(ports):
        status: Optional[dict] = None
        try:
            r = requests.get(f"http://localhost:{port}/status", timeout=timeout_s)
            if r.status_code == 200:
                try:
                    body = r.json()
                except ValueError:
                    body = None
                status = body if isinstance(body, dict) else {}
        except requests.RequestException:
            pass
        pids = _pids_listening_on(port)
        if status is None and not pids:
            continue
        live.append(LiveServer(port, ports[port], status, pids))
    return live


def _alive(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def stop(servers: List[LiveServer], wait_s: float = ESCALATION_WAIT_S) -> int:
    """SIGINT -> SIGTERM -> SIGKILL each server's PIDs. Returns the number still alive.

    Signals go to the whole process group where the server owns one: a KV server spawns
    dataloader/compression workers, and signalling only the parent leaves those holding
    GPU memory - which looks exactly like "the server did not shut down".
    """
    targets = sorted({pid for s in servers for pid in s.pids})
    if not targets:
        return 0
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        remaining = [pid for pid in targets if _alive(pid)]
        if not remaining:
            break
        logger.info("Sending %s to pid(s) %s.", sig.name, remaining)
        for pid in remaining:
            try:
                pgid = os.getpgid(pid)
                # Never signal our own group - that would kill this script before it
                # can escalate (and, run from a worker's shell, the worker too).
                if pgid != os.getpgrp():
                    os.killpg(pgid, sig)
                else:
                    os.kill(pid, sig)
            except (ProcessLookupError, PermissionError) as exc:
                logger.warning("  pid %d: %s", pid, exc)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and any(_alive(pid) for pid in remaining):
            time.sleep(0.5)
    still = [pid for pid in targets if _alive(pid)]
    if still:
        logger.error(
            "pid(s) %s survived SIGKILL - most likely stuck in an uninterruptible CUDA "
            "call. Check `nvidia-smi` before starting anything that needs those GPUs.",
            still,
        )
    return len(still)


def _selected_ports(args) -> Dict[int, str]:
    known = all_known_server_ports()
    if args.port:
        unknown = [p for p in args.port if p not in known]
        for p in unknown:
            known[p] = "unknown (not a reasondb default port)"
        return {p: known[p] for p in args.port}
    if args.capability:
        wanted = set(CAPABILITY_SCRIPTS[args.capability][1])
        return {p: label for p, label in known.items() if p in wanted}
    return known


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what is running and stop nothing. Everything else is unchanged, so "
        "this is the safe way to see whether a machine is free before claiming it.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--capability", choices=sorted(WORKER_CAPABILITY_CHOICES), default=None,
        help="Restrict to the ports this worker capability uses. Note the embedding "
        "pair belongs to every capability, so stopping one capability's servers stops "
        "the embedding servers the others also rely on.",
    )
    group.add_argument(
        "--port", type=int, nargs="+", default=None,
        help="Restrict to these ports (they need not be reasondb defaults).",
    )
    parser.add_argument(
        "--wait-s", type=float, default=ESCALATION_WAIT_S,
        help="Seconds to wait after each signal before escalating to the next.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()

    ports = _selected_ports(args)
    live = discover(ports)
    if not live:
        logger.info("No reasondb backend servers running on %d checked port(s).", len(ports))
        return

    logger.info("Running server(s):")
    for server in live:
        logger.info("%s", server.describe())

    modes = {s.index_mode for s in live} - {"n/a"}
    if len(modes) > 1:
        logger.warning(
            "Mixed cache modes across live servers (%s). A run can only match one of "
            "them; the other half will fail on 'no usable cache or relative index'.",
            ", ".join(sorted(modes)),
        )

    if args.dry_run:
        logger.info("--dry-run: stopping nothing.")
        return

    unowned = [s for s in live if not s.pids]
    if unowned:
        logger.warning(
            "No owning pid visible for port(s) %s - another user's process, or one this "
            "user may not inspect. Those will be left running.",
            [s.port for s in unowned],
        )

    survivors = stop(live, wait_s=args.wait_s)
    still_serving = [s.port for s in discover(ports)]
    if still_serving or survivors:
        logger.error(
            "Not clean: port(s) %s still answering, %d process(es) still alive. Do not "
            "start new servers until this is resolved - they would fail to bind.",
            still_serving, survivors,
        )
        sys.exit(1)
    logger.info("All targeted servers stopped.")


if __name__ == "__main__":
    main()
