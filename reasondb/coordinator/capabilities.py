"""Maps a worker's ``--capability`` flag to the backend servers it brings up locally.

Reuses ``scripts/start_servers_*.sh`` (which encode the model/GPU selection) via
``subprocess.Popen`` rather than reimplementing server startup in Python. The scripts'
default ``CUDA_VISIBLE_DEVICES`` assume one GPU layout; :func:`start_capability_servers`
sources a per-host override file first if one exists, letting a worker machine
override the GPU assignment without editing the shared scripts. Every capability must
appear here (see the CAPABILITY_SCRIPTS/WORKER_CAPABILITY_CHOICES assert below).
"""

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from reasondb.backends.audio_model import PORT_AUDIO, PORT_KV_AUDIO
from reasondb.backends.image_similarity import PORT_IMAGE_SIM
from reasondb.backends.kv_cache_base import describe_error_response
from reasondb.backends.prepare_memo import reset_prepare_memo
from reasondb.backends.text_embeddings import PORT_TEXT_SIM
from reasondb.backends.text_qa import PORT_KV_TEXT_QA
from reasondb.backends.vision_model import PORT_KV_VISION, PORT_VISION
from reasondb.coordinator.models import WORKER_CAPABILITY_CHOICES

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The embedding pair, needed by every mode including ``simulate`` (see
#: ``ImageSimilarityBackend.assert_ready``).
_EMBEDDING_PORTS = [PORT_IMAGE_SIM, PORT_TEXT_SIM]

#: The KV ports each start script actually launches, by model. These take *minutes* to
#: answer - a KV server loads its model and pre-compresses a cache for every row before
#: serving - whereas the embedding servers are up in seconds.
_TEXT_KV_PORTS = [
    PORT_KV_TEXT_QA["meta-llama/Llama-3.1-8B-Instruct"],
    PORT_KV_TEXT_QA["meta-llama/Llama-3.1-70B-Instruct"],
]
_IMAGE_KV_PORTS = [
    PORT_KV_VISION["llava-hf/llama3-llava-next-8b-hf"],
    PORT_KV_VISION["llava-hf/llava-next-72b-hf"],
]

#: (script relative to repo root, [ports to poll /status on before declaring "ready"])
#:
#: The KV ports are included, not just the embedding pair, so that ``wait_until_ready``
#: does not let a worker claim jobs while its KV servers are still loading.
CAPABILITY_SCRIPTS: Dict[str, "tuple[str, List[int]]"] = {
    "text": ("scripts/start_servers_text.sh", _TEXT_KV_PORTS + _EMBEDDING_PORTS),
    "image": ("scripts/start_servers_images.sh", _IMAGE_KV_PORTS + _EMBEDDING_PORTS),
    "both": (
        "scripts/start_servers_all.sh",
        _TEXT_KV_PORTS + _IMAGE_KV_PORTS + _EMBEDDING_PORTS,
    ),
    "embedding-only": ("scripts/start_servers_embedding_only.sh", list(_EMBEDDING_PORTS)),
    "simulate": ("scripts/start_servers_embedding_only.sh", list(_EMBEDDING_PORTS)),
    "audio": ("scripts/start_servers_audio.sh", [PORT_KV_AUDIO] + _EMBEDDING_PORTS),
}

# Catch drift at import time rather than at worker-startup time, deep in a launched
# process - same reasoning as scheduler.py's CAPABILITY_PROVIDES check.
assert set(CAPABILITY_SCRIPTS) == set(WORKER_CAPABILITY_CHOICES), (
    f"CAPABILITY_SCRIPTS {sorted(CAPABILITY_SCRIPTS)} and models.WORKER_CAPABILITY_"
    f"CHOICES {sorted(WORKER_CAPABILITY_CHOICES)} have drifted apart."
)

#: The only env vars scripts/start_servers_*.sh read as GPU overrides. A per-host config
#: file with any other key is almost certainly a typo (e.g. "TEXT_70_GPUS" for
#: "TEXT_70B_GPUS") that would otherwise silently fall back to the script's default.
KNOWN_GPU_OVERRIDE_KEYS = frozenset(
    {"TEXT_8B_GPUS", "TEXT_70B_GPUS", "IMAGE_8B_GPUS", "IMAGE_70B_GPUS", "EMBED_GPUS", "AUDIO_DEVICE_ID"}
)


def all_known_server_ports() -> "Dict[int, str]":
    """Every port a reasondb backend server can listen on, mapped to a human label.

    Wider than the union of :data:`CAPABILITY_SCRIPTS`' port lists on purpose: those
    name only what the start scripts launch, whereas anything sweeping the machine
    (``scripts/stop_servers.py``) has to also find servers started by hand for one
    model - the text KV registry alone has more models than any script starts.
    """
    labels = {
        PORT_IMAGE_SIM: "image_similarity",
        PORT_TEXT_SIM: "text_embed",
        PORT_KV_AUDIO: "kv_audio_qa",
        PORT_AUDIO: "audio_qa",
        PORT_VISION: "image_qa",
    }
    for model, port in PORT_KV_TEXT_QA.items():
        labels[port] = f"kv_text_qa[{model}]"
    for model, port in PORT_KV_VISION.items():
        labels[port] = f"kv_image_qa[{model}]"
    return labels


def per_host_env_path(hostname: Optional[str] = None) -> Path:
    """``configs/gpu_layout/<hostname>.env`` - sourced before the start script if present.

    Format: plain ``KEY=value`` lines (e.g. ``TEXT_70B_GPUS=4,5,6``), matching the
    overridable env vars the start_servers_*.sh scripts read
    (``TEXT_8B_GPUS``/``TEXT_70B_GPUS``/``IMAGE_8B_GPUS``/``IMAGE_70B_GPUS``/
    ``EMBED_GPUS``). Absent by default - every worker falls back to each script's
    default layout unless a host needs an override.
    """
    host = hostname or os.uname().nodename
    return REPO_ROOT / "configs" / "gpu_layout" / f"{host}.env"


def _load_env_file(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    overrides = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        assert key in KNOWN_GPU_OVERRIDE_KEYS, (
            f"{path} sets {key!r}, not one of {sorted(KNOWN_GPU_OVERRIDE_KEYS)} - "
            "likely a typo (it would otherwise be silently ignored, since the shell "
            "scripts only read the known override names)."
        )
        overrides[key] = value.strip()
    return overrides


class ServerIndexModeMismatch(RuntimeError):
    """A live server was started in the other ``USE_INDICES`` mode than this worker wants.

    Raised instead of reusing it, because both directions of the mismatch are silent for
    a long time and then wrong: indices on with ``--use-indexes`` off fails every job on
    "no usable cache or relative index", and indices off with ``--use-indexes`` on
    re-prefills each ratio, quietly costing the run the thing indexes exist to save.
    """


def probe_capability_servers(
    capability: str, timeout_s: float = 2.0
) -> "tuple[Dict[int, dict], List[int]]":
    """One non-blocking pass over this capability's ports: ``({up port: status}, not up)``.

    Unlike :func:`wait_until_ready` this never sleeps or retries - it answers "what is
    serving *right now*", which is what :func:`start_capability_servers` needs in order
    to decide whether launching anything is warranted at all. The ``/status`` payload
    comes back with it so the caller can also ask *what* is serving; a port that answers
    200 with a body that is not a JSON object maps to ``{}`` rather than failing the
    probe, since liveness is the part that must not depend on the payload's shape.
    """
    assert capability in CAPABILITY_SCRIPTS, (
        f"Unknown capability {capability!r}; expected one of {WORKER_CAPABILITY_CHOICES}."
    )
    _, ports = CAPABILITY_SCRIPTS[capability]
    up: Dict[int, dict] = {}
    down: List[int] = []
    for port in ports:
        try:
            r = requests.get(f"http://localhost:{port}/status", timeout=timeout_s)
            if r.status_code != 200:
                down.append(port)
                continue
            try:
                body = r.json()
            except ValueError:
                body = None
            up[port] = body if isinstance(body, dict) else {}
        except requests.RequestException:
            down.append(port)
    return up, down


def index_mode_mismatches(
    up: "Dict[int, dict]", use_indexes: bool
) -> "Dict[int, bool]":
    """``{port: its use_relative_indices}`` for live servers that disagree with us.

    Ports whose ``/status`` omits ``use_relative_indices`` are skipped rather than
    treated as ``False``: the embedding servers and the audio server have no
    relative-indices path at all (see ``kv_cache_audio_qa_server``).
    """
    return {
        port: bool(status["use_relative_indices"])
        for port, status in up.items()
        if "use_relative_indices" in status
        and bool(status["use_relative_indices"]) != use_indexes
    }


#: The ports whose server has a ``PINNED_KV_STORE``. Text and image only: the audio server
#: rejects ``keep_in_memory`` outright (see ``kv_cache_audio_qa_server``) and the embedding
#: pair has no KV path at all, so neither has the endpoint and neither is ever POSTed to.
_PIN_CAPABLE_KV_PORTS = frozenset(_TEXT_KV_PORTS + _IMAGE_KV_PORTS)

assert PORT_KV_AUDIO not in _PIN_CAPABLE_KV_PORTS, (
    "The audio KV server is listed as pin-capable, but it asserts `not keep_in_memory` and "
    "never touches PINNED_KV_STORE - a release would 404 there. Drift in the port constants?"
)

#: How long to give one ``/release_pinned_kv``. Two orders of magnitude above the ``/status``
#: probe's 2s, and deliberately: the handler frees tens of GB of page-locked host memory
#: (``cudaFreeHost`` synchronizes the device) and runs a ``gc.collect()``, and the image
#: server binds ``threaded=False`` so the POST queues behind any in-flight inference. A
#: probe-sized timeout here would turn a *successful* release into a failed job.
RELEASE_TIMEOUT_S = 120.0

#: The dataset the pins currently in the servers were taken for, as ``(benchmark, split)``.
#: Module-global rather than per-loop state because the servers - and therefore their pins -
#: outlive both job loops: ``claim_run_loop`` is called once per ``--tasks`` entry, and
#: ``run_coordinator.run_local`` has no worker state object at all. ``None`` initially, which
#: is what makes the *first* job of a process release too: ``start_capability_servers``
#: reuses warm servers, which may still hold a previous process's pins.
_last_released_dataset: "Optional[tuple]" = None
_release_lock = threading.Lock()


class PinnedKVReleaseFailed(RuntimeError):
    """A live KV server would not release its pinned caches.

    Fails the job rather than warning. The pins are still held and the client's prepare memo
    has already been cleared, so the next ``-in-memory`` ``prepare()`` re-pins on top of them
    and overflows the budget - a crash further from its cause than this one.
    """


def pin_capable_ports(capability: str) -> "List[int]":
    """This capability's ports that can hold pinned KV caches, in CAPABILITY_SCRIPTS order.

    Derived from the capability's own port list rather than written down again, so a
    capability gaining a KV server needs no edit here: ``text``/``image`` give 2 each,
    ``both`` 4, and ``embedding-only``/``simulate``/``audio`` give none - which is what makes
    a ``--simulate`` worker skip this whole path without a special case.
    """
    assert capability in CAPABILITY_SCRIPTS, (
        f"Unknown capability {capability!r}; expected one of {WORKER_CAPABILITY_CHOICES}."
    )
    _, ports = CAPABILITY_SCRIPTS[capability]
    return [port for port in ports if port in _PIN_CAPABLE_KV_PORTS]


def release_pinned_kv_caches(
    capability: str, timeout_s: float = RELEASE_TIMEOUT_S
) -> "Dict[int, int]":
    """POST ``/release_pinned_kv`` to every *live* pin-capable port: ``{port: n_released}``.

    Which ports are live is answered by :func:`probe_capability_servers` rather than inferred
    from the POST's own exception type - ``requests.ConnectionError`` covers both "nothing is
    listening" and "a live server dropped the connection", so an exception taxonomy would get
    the interesting case backwards. So: a port the probe calls down is skipped silently (a
    ``simulate`` worker starts no KV server), and *any* failure against a port the probe just
    called up raises.
    """
    ports = pin_capable_ports(capability)
    if not ports:
        return {}
    up, _down = probe_capability_servers(capability)
    released: Dict[int, int] = {}
    for port in ports:
        if port not in up:
            logger.debug(
                "Worker: nothing serving on port %d; no pinned KV caches to release there.",
                port,
            )
            continue
        try:
            response = requests.post(
                f"http://localhost:{port}/release_pinned_kv", timeout=timeout_s
            )
        except requests.RequestException as exc:
            raise PinnedKVReleaseFailed(
                f"Port {port} answered /status but its /release_pinned_kv did not: {exc}. "
                "Its KV caches are still pinned, so the next -in-memory prepare() would "
                "overflow the budget."
            ) from exc
        if response.status_code == 404:
            raise PinnedKVReleaseFailed(
                f"Port {port} has no /release_pinned_kv endpoint. That server runs an "
                "outdated version and was reused rather than restarted (see "
                "start_capability_servers). Stop it with scripts/stop_servers.py and let "
                "the worker start its own."
            )
        if response.status_code != 200:
            raise PinnedKVReleaseFailed(
                f"/release_pinned_kv failed on port {port} (HTTP "
                f"{response.status_code}): {describe_error_response(response)}"
            )
        body = response.json()
        if body.get("n_pinned"):
            raise PinnedKVReleaseFailed(
                f"Port {port} reported {body['n_pinned']} cache(s) still pinned after a "
                f"release ({body.get('pinned_gb')} GB). Its budget is not actually free."
            )
        released[port] = body.get("n_released", 0)
    return released


def release_pinned_kv_if_dataset_changed(
    capability: str, benchmark: str, split: str
) -> bool:
    """Release this capability's pinned KV caches iff the dataset changed. Did it release?

    Called at job start. Not per job: re-pinning a column is a ``torch.load`` per cache file
    and lands inside the first query's measured ``end_to_end`` span, so paying it between two
    jobs of the same benchmark would distort the very measurement ``-in-memory`` exists to
    make. Between datasets it is not a cost at all - those caches will never be asked for
    again, and holding them is what eventually overflows the budget.

    The prepare memo is cleared with it, and *before* the POSTs. It is client-side and
    process-global, so a memo entry outliving the pins it describes makes the next
    ``prepare()`` short-circuit, nothing re-pins, and the serve path raises rather than
    reading from disk. That bites on a dataset *revisit* (A -> B -> A) - jobs are claimed
    from a shared queue, so a benchmark's jobs are not contiguous. Clearing first means even
    a partial failure leaves the client's belief consistent with the servers; the cost of
    being early is at worst one redundant re-scan.
    """
    global _last_released_dataset
    dataset = (benchmark, split)
    with _release_lock:
        previous = _last_released_dataset
        if previous == dataset:
            return False
    # Before the POSTs: see the docstring. Whole-memo granularity is the only one available
    # (`_done` holds opaque digests), and it costs nothing here - a genuinely new dataset's
    # fingerprints would have missed anyway, and a revisit's *must* miss.
    reset_prepare_memo()
    released = release_pinned_kv_caches(capability)
    with _release_lock:
        # Only on success, so a failure is retried by the next job rather than assumed done.
        _last_released_dataset = dataset
    if released:
        logger.info(
            "Worker: released pinned KV caches moving from dataset %s to %s: %s.",
            previous, dataset,
            ", ".join(f"port {port}: {n}" for port, n in sorted(released.items())),
        )
    return True


def reset_released_dataset() -> None:
    """Forget which dataset the pins were taken for. For tests, mirroring
    ``prepare_memo.reset_prepare_memo``."""
    global _last_released_dataset
    with _release_lock:
        _last_released_dataset = None


def start_capability_servers(
    capability: str,
    worker_dir: Path,
    use_indexes: bool,
    hostname: Optional[str] = None,
    kv_cache_pin_gb: Optional[float] = None,
) -> "Optional[subprocess.Popen]":
    """Launch the given capability's backend servers as one subprocess group.

    ``kv_cache_pin_gb`` is exported as ``KV_CACHE_PIN_GB`` and is a budget *per server
    process*, not per node: the start scripts export it once and every KV server they
    launch reads the same variable, so a ``both`` worker's four servers each get this
    much. ``None`` leaves the variable alone, so a hand-launched worker keeps whatever
    its shell exported; the cluster config always passes a value
    (``workers.kv_cache_pin_gb``), since the scripts' own fallback of 0 means no
    ``-in-memory`` operator can be served at all.

    Returns ``None`` when every port this capability needs is already serving, having
    launched nothing: a second set of KV servers on a machine that already has one
    re-loads the models and re-compresses a cache per row (tens of minutes) only to
    fight the live set for the same GPUs. A reused server may still
    hold KV caches a *previous* process pinned, which is why
    :func:`release_pinned_kv_if_dataset_changed` releases on a worker's first job too rather
    than only on a change it has itself observed.

    Reuse is refused with :class:`ServerIndexModeMismatch` when a live server reports a
    ``use_relative_indices`` other than ``use_indexes`` - a set of servers in the wrong
    half of the ``USE_INDICES``/``--use-indexes`` pairing is worse than none, and the
    fix (restart them the other way, e.g. via ``scripts/stop_servers.py``) is left to
    the user, since the running set may belong to another run.

    Otherwise returns the ``Popen`` handle for the shell script itself (which
    backgrounds its own children with ``&`` - see the scripts - so this one handle's
    lifetime roughly tracks the whole group, good enough for "is my capability's servers
    process still around" liveness at the worker level; per-port ``/status`` polling in
    :func:`wait_until_ready` is the real readiness signal).
    """
    assert capability in WORKER_CAPABILITY_CHOICES, (
        f"Unknown capability {capability!r}; expected one of {WORKER_CAPABILITY_CHOICES}."
    )
    already_up, not_up = probe_capability_servers(capability)
    mismatched = index_mode_mismatches(already_up, use_indexes)
    if mismatched:
        raise ServerIndexModeMismatch(
            f"Capability {capability!r} wants use_relative_indices={use_indexes}, but "
            f"port(s) already serving report otherwise: "
            + ", ".join(f"{port} -> {mode}" for port, mode in sorted(mismatched.items()))
            + ". Not reusing them and not starting a second set. Either stop them "
            "(scripts/stop_servers.py) and let this worker start its own, or run the "
            f"worker with --use-indexes {'off' if use_indexes else 'on'} to match."
        )
    if not not_up:
        logger.info(
            "Worker: capability %r servers already answering on port(s) %s; starting "
            "nothing and reusing them.",
            capability, sorted(already_up),
        )
        return None
    if already_up:
        # The scripts start their whole set unconditionally, so the duplicates for the
        # live ports will fail to bind and exit while the missing ones come up; the
        # warning explains the resulting "Address already in use" tracebacks.
        logger.warning(
            "Worker: capability %r is only partly up (serving %s, missing %s). Running "
            "%s to bring up the rest; the already-bound ports will log a bind failure "
            "for their duplicate and that is expected.",
            capability, sorted(already_up), sorted(not_up), CAPABILITY_SCRIPTS[capability][0],
        )
    script, _ = CAPABILITY_SCRIPTS[capability]
    script_path = REPO_ROOT / script
    assert script_path.is_file(), (
        f"{script_path} does not exist - CAPABILITY_SCRIPTS points {capability!r} at "
        "a script that isn't there (moved/renamed?)."
    )
    env = dict(os.environ)
    env.update(_load_env_file(per_host_env_path(hostname)))
    if use_indexes:
        env["USE_INDICES"] = "1"
    if kv_cache_pin_gb is not None:
        assert kv_cache_pin_gb >= 0, f"kv_cache_pin_gb must be >= 0; got {kv_cache_pin_gb}."
        env["KV_CACHE_PIN_GB"] = repr(float(kv_cache_pin_gb))
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_path = worker_dir / f"servers_{capability}.log"
    logger.info(
        "Worker: starting %s capability servers via %s (KV_CACHE_PIN_GB=%s per server, log: %s)",
        capability, script, env.get("KV_CACHE_PIN_GB", "unset"), log_path,
    )
    log_fh = open(log_path, "a")
    return subprocess.Popen(
        ["bash", str(REPO_ROOT / script)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )


#: Default ceiling for :func:`wait_until_ready`. A 70B KV server loads its weights and
#: then pre-compresses a cache for every row of the dataset before it answers /status,
#: which is tens of minutes on a cold cache rather than the few minutes a plain HTTP
#: readiness poll would assume.
DEFAULT_READY_TIMEOUT_S = 3600.0


def wait_until_ready(
    capability: str,
    timeout_s: float = DEFAULT_READY_TIMEOUT_S,
    poll_interval_s: float = 5.0,
) -> bool:
    """Poll every port this capability's servers must expose until all answer
    ``GET /status`` with 200, or ``timeout_s`` elapses. Model loading is slow (large
    KV models can take tens of minutes), hence the generous default timeout - a worker
    must never register with the coordinator (and start claiming jobs) before its local
    servers can actually serve them.

    Progress is logged as ports come up so a long wait is visible in the worker log.
    """
    assert capability in CAPABILITY_SCRIPTS, (
        f"Unknown capability {capability!r}; expected one of {WORKER_CAPABILITY_CHOICES}."
    )
    _, ports = CAPABILITY_SCRIPTS[capability]
    deadline = time.monotonic() + timeout_s
    pending = set(ports)
    started = time.monotonic()
    logger.info(
        "Worker: waiting up to %.0fs for capability %r servers on port(s) %s.",
        timeout_s, capability, sorted(pending),
    )
    while pending and time.monotonic() < deadline:
        for port in list(pending):
            try:
                r = requests.get(f"http://localhost:{port}/status", timeout=2.0)
                if r.status_code == 200:
                    pending.discard(port)
                    logger.info(
                        "Worker: port %d ready after %.0fs; still waiting on %s.",
                        port, time.monotonic() - started, sorted(pending) or "nothing",
                    )
            except requests.RequestException:
                pass
        if pending:
            time.sleep(poll_interval_s)
    if pending:
        logger.error(
            "Worker: capability %r servers on port(s) %s never answered /status within "
            "%.0fs. Not registering - a worker that claims jobs its servers cannot "
            "serve fails them instantly and burns all their attempts.",
            capability,
            sorted(pending),
            timeout_s,
        )
    return not pending
