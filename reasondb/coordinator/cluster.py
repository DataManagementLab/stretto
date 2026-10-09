"""Declarative launch plan for a cluster of coordinators and their workers.

``scripts/start-configurators.sh`` and ``scripts/start-workers.sh`` both read one YAML
file - see ``scripts/cluster.yaml`` for the annotated defaults - which names the
experiments *and* the coordinator flags each one is launched with, so a new experiment
is a config entry rather than an edit to two shell scripts.

The two scripts must agree on the node assignment (a worker addresses the coordinator on
its own node as ``localhost`` and every other one by node name), which is exactly why
that assignment lives here, computed once from a file both of them read, rather than in
each script's own defaults.

Shell contract
--------------
The scripts consume this module through ``python -m reasondb.coordinator.cluster``:

    eval "$(python -m reasondb.coordinator.cluster --config F --emit settings)"
    python -m reasondb.coordinator.cluster --config F --emit coordinators   # TSV
    python -m reasondb.coordinator.cluster --config F --emit workers        # TSV

The rendered arguments end up inside a *single-quoted* remote command
(``ssh host "bash -ic '...'"``), which cannot carry a quote of either kind: a single one
closes the remote string, a double one is eaten by the local shell before ssh ever sees
it. Arguments are therefore restricted to :data:`FORBIDDEN_ARG_CHARS`-free tokens and
spaces are backslash-escaped for the innermost ``bash -ic``. ``$`` is deliberately
*allowed* and *not* escaped, so a config may write a path through a variable the node
defines (``$HOME/...``) and have it expand there rather than here.

Imports stay light on purpose - this runs on the launching machine, which needs to build
a plan, not a torch. The one heavy import (the producer registry, for validating
``producer:`` before ssh'ing anywhere) is optional and best-effort.
"""

import argparse
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from reasondb.coordinator.models import WORKER_CAPABILITY_CHOICES

#: Where the scripts look when ``--config`` is not passed, relative to the repository root.
DEFAULT_CONFIG_PATH = "scripts/cluster.yaml"

#: Characters an argument token may not contain. Quotes and backslashes cannot survive
#: the ssh -> remote shell -> ``bash -ic`` sandwich described in the module docstring;
#: the control characters would break the TSV the scripts read this back as; the
#: metacharacters would be interpreted by the innermost bash instead of reaching argparse.
FORBIDDEN_ARG_CHARS = "'\"\\\n\r\t`;&|<>()"

#: Section defaults. A key absent from these is a typo in the config, not an extra
#: setting - unknown keys are rejected rather than silently ignored.
CLUSTER_DEFAULTS: Dict[str, Any] = {
    "prefix": "sweep-",
    "hosts": None,
    "port": 5099,
    "repo": None,
    "bashrc": None,
    "home": None,
    "cache_dir": None,
    "conda_env": "reasondb",
    "user": None,
}

#: RAM budget in GB handed to *each* backend server a worker starts, i.e. what
#: ``KV_CACHE_PIN_GB`` sizes: the caches an ``-in-memory`` operator pins at ``prepare()``.
#: Per server process, not per node - a `both` worker starts four KV servers and each one
#: gets this budget, so a node hosting them needs the multiple.
DEFAULT_KV_CACHE_PIN_GB = 50.0

WORKER_DEFAULTS: Dict[str, Any] = {
    "nodes": None,
    "per_node": 1,
    "capability": "simulate",
    "device": "cuda:0",
    #: Sized rather than left at the start scripts' own 0, because 0 means "no -in-memory
    #: operator can be served" and the deployed suite holds two of them: a fleet launched
    #: from this file would refuse them at setup(), before any query.
    "kv_cache_pin_gb": DEFAULT_KV_CACHE_PIN_GB,
    #: Off by default: the whole fleet queues on the first experiment, finishes it, and
    #: moves on together, which is what "N workers on M coordinators" is normally meant
    #: to do. Rotation spreads the fleet over every coordinator at once instead, so with
    #: 15 workers and 5 experiments each experiment gets 3 - fine when the goal is to
    #: have all of them progressing, wrong when the goal is one finished experiment.
    "rotate": False,
    "args": None,
}

COORDINATOR_DEFAULTS: Dict[str, Any] = {
    "args": None,
}

EXPERIMENT_KEYS = frozenset({"task_id", "producer", "node", "host", "port", "args"})


# ── Argument rendering ───────────────────────────────────────────────────────────


def _stringify(value: Any, where: str) -> str:
    """One YAML scalar as the token argparse will see.

    Booleans become ``true``/``false`` rather than Python's ``True``/``False`` because
    that is the vocabulary of the flags that take them (``--tune-parameters true false``).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        raise ValueError(f"{where}: null is not a value; drop the key or use false to omit a flag.")
    if isinstance(value, (Mapping, list, tuple)):
        raise ValueError(f"{where}: expected a scalar, got {type(value).__name__}.")
    return str(value)


def _check_token(token: str, where: str) -> str:
    if not token:
        raise ValueError(f"{where}: empty argument token.")
    bad = sorted({c for c in token if c in FORBIDDEN_ARG_CHARS})
    if bad:
        printable = ", ".join(repr(c) for c in bad)
        raise ValueError(
            f"{where}: argument {token!r} contains {printable}, which cannot survive the "
            "single-quoted remote command these launchers build (see "
            "reasondb.coordinator.cluster). Move the value into a file, or set it in the "
            "node's .bashrc and reference it as $VAR - those are expanded on the node."
        )
    return token


def _flag(name: str) -> str:
    """``sample_sizes`` / ``sample-sizes`` / ``--sample-sizes`` all mean the flag."""
    name = str(name).strip()
    if name.startswith("-"):
        return name
    return "--" + name.replace("_", "-")


def render_args(spec: Any, where: str) -> List[str]:
    """A config ``args:`` value as the argv tokens it stands for.

    Three shapes, so a config can be as readable or as literal as it needs to be:

    * a mapping - ``benchmarks: [movie_random, artwork_random_medium]`` becomes
      ``--benchmarks movie_random artwork_random_medium``; ``true`` a bare flag,
      ``false``/``null`` nothing at all (so an experiment can switch a shared flag back
      off); a *nested* mapping the ``NAME=VALUE`` pairs ``--simulate`` and
      ``--precompute`` take.
    * a list of strings - each is split like a shell would, so both
      ``["--benchmarks", "movie_random"]`` and ``["--benchmarks movie_random"]`` work.
    * one string, split the same way.
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = [spec]
    tokens: List[str] = []
    if isinstance(spec, Mapping):
        for key, value in spec.items():
            flag = _flag(key)
            if value is None or value is False:
                continue
            if value is True:
                tokens.append(flag)
                continue
            tokens.append(flag)
            if isinstance(value, Mapping):
                tokens.extend(
                    f"{name}={_stringify(item, f'{where}.{key}')}" for name, item in value.items()
                )
            elif isinstance(value, (list, tuple)):
                if not value:
                    raise ValueError(f"{where}.{key}: empty list; drop the key or use false.")
                tokens.extend(_stringify(item, f"{where}.{key}") for item in value)
            else:
                tokens.append(_stringify(value, f"{where}.{key}"))
    elif isinstance(spec, (list, tuple)):
        for item in spec:
            if not isinstance(item, str):
                raise ValueError(
                    f"{where}: a list of args holds strings; got {type(item).__name__}. "
                    "Use the mapping form (flag: value) for anything else."
                )
            tokens.extend(shlex.split(item))
    else:
        raise ValueError(f"{where}: expected a mapping, a list or a string; got {type(spec).__name__}.")
    return [_check_token(token, where) for token in tokens]


def parse_node_selection(spec: Any, where: str = "workers.nodes") -> Tuple[int, ...]:
    """``workers.nodes`` as the node indices the fleet runs on.

    A **lone integer is a count**, so ``7`` is the first seven nodes (1..7), while
    anything carrying a ``-`` or a ``,`` is a list of node *ids* - ``8-12`` is the five
    nodes 8..12, ``1,3,5-7`` is four particular ones. ``1-7`` is the same fleet as
    ``7``, spelled unambiguously.

    Ids are 1-based (:func:`host_for_node`'s indexing, so with an explicit
    ``cluster.hosts`` node *i* is the *i*-th host). A repeated id is an error rather
    than deduplicated: two worker sets on one node would share that node's backend
    servers, which is precisely what per-node worker counts (``per_node``) exist to
    express.
    """
    if isinstance(spec, bool):
        raise ValueError(f"{where}: expected a node count or a range like 8-12; got {spec!r}.")
    if isinstance(spec, int):
        return _node_count(spec, where)
    text = str(spec).strip()
    if not text:
        raise ValueError(f"{where}: empty; give a count (7), a range (8-12) or a list (1,3,5-7).")
    if text.isdigit():
        return _node_count(int(text), where)

    ids: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"{where}: empty entry in {text!r}; expected a count (7), a range (8-12) or a list (1,3,5-7).")
        low_text, sep, high_text = part.partition("-")
        low = _node_id(low_text, part, where)
        high = _node_id(high_text, part, where) if sep else low
        if high < low:
            raise ValueError(f"{where}: range {part!r} counts down; write it as {high}-{low}.")
        ids.extend(range(low, high + 1))
    duplicates = sorted({node for node in ids if ids.count(node) > 1})
    if duplicates:
        raise ValueError(
            f"{where}: node(s) {duplicates} named twice in {text!r}. One entry per node - "
            "use workers.per_node to put several workers on one node, since they share its "
            "backend servers."
        )
    return tuple(ids)


def _node_count(count: int, where: str) -> Tuple[int, ...]:
    if count < 1:
        raise ValueError(f"{where} must be at least 1; got {count}.")
    return tuple(range(1, count + 1))


def _node_id(text: str, part: str, where: str) -> int:
    text = text.strip()
    if not text.isdigit():
        raise ValueError(f"{where}: {part!r} is not a node id or a range of them (e.g. 8-12).")
    node = int(text)
    if node < 1:
        raise ValueError(f"{where}: node ids are 1-based; got {node} in {part!r}.")
    return node


def format_node_selection(node_ids: Sequence[int]) -> str:
    """The inverse of :func:`parse_node_selection`, for messages: ``8-12``, ``1,3,5-7``.

    A single node comes out as ``9-9`` rather than ``9``, which is the one output that
    would not read back as itself: a lone integer is the count form, so ``9`` means the
    first nine nodes. Round-tripping matters more here than the extra character - this
    string is what a launcher prints for a node list to be re-passed to ``--nodes``.
    """
    parts: List[str] = []
    for node in node_ids:
        if parts and node == _run_end(parts[-1]) + 1:
            parts[-1] = f"{parts[-1].partition('-')[0]}-{node}"
        else:
            parts.append(str(node))
    if len(parts) == 1 and "-" not in parts[0]:
        return f"{parts[0]}-{parts[0]}"
    return ",".join(parts)


def _run_end(part: str) -> int:
    _, sep, high = part.partition("-")
    return int(high if sep else part)


def shell_join(tokens: Sequence[str]) -> str:
    """Tokens as one command-line fragment for the remote ``bash -ic``.

    Backslash-escaped spaces rather than quotes: see the module docstring for why no
    quoting character can be used here. Every other character is already known safe by
    :func:`_check_token`, ``$`` included - and left alone, so it expands on the node.
    """
    return " ".join(token.replace(" ", r"\ ") for token in tokens)


# ── The plan ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Experiment:
    """One coordinator: which task, on which node, with which flags."""

    task_id: str
    producer: str
    node: int
    host: str
    port: int
    args: List[str] = field(default_factory=list)

    def argv(self) -> List[str]:
        """Everything after ``python scripts/run_coordinator.py``.

        ``--port`` is always passed, never left to the default: the workers' URLs are
        built from this same number, and a node with ``REASONDB_MONITOR_PORT`` set in its
        environment would otherwise move the coordinator out from under them.

        Every flag, including the ``--simulate`` mapping, comes from the config file.
        """
        argv = ["--task-id", self.task_id, "--producer", self.producer, "--port", str(self.port)]
        argv.extend(self.args)
        return argv

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def format_gb(value: float) -> str:
    """A GB budget as the token argparse sees: ``50``, not ``50.0``, and ``12.5`` intact."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


@dataclass(frozen=True)
class Worker:
    """One worker process: which node it runs on and which tasks it drains, in order."""

    worker_id: str
    host: str
    node: int
    capability: str
    device: str
    kv_cache_pin_gb: float = DEFAULT_KV_CACHE_PIN_GB
    targets: List[Tuple[str, str]] = field(default_factory=list)
    args: List[str] = field(default_factory=list)

    def argv(self) -> List[str]:
        """Everything after ``python scripts/run_worker.py``.

        ``--kv-cache-pin-gb`` sits *before* ``args`` so a ``workers.args`` copy of the
        flag still wins (argparse keeps the last occurrence) - the same "shared first,
        specific last" rule :meth:`Experiment.argv` follows for the coordinator's flags.
        """
        argv = ["--tasks"]
        argv.extend(f"{task_id}={url}" for task_id, url in self.targets)
        argv.extend(["--worker-id", self.worker_id, "--capability", self.capability, "--device", self.device])
        argv.extend(["--kv-cache-pin-gb", format_gb(self.kv_cache_pin_gb)])
        argv.extend(self.args)
        return argv


@dataclass(frozen=True)
class ClusterConfig:
    """A whole cluster launch: the shared environment, the coordinators, the workers."""

    prefix: str
    port: int
    repo: str
    bashrc: Optional[str]
    home: Optional[str]
    cache_dir: Optional[str]
    conda_env: str
    user: Optional[str]
    hosts: Optional[List[str]]
    #: The node indices to start workers on, in launch order - ``(1, ..., 7)`` for a bare
    #: ``nodes: 7``, ``(8, ..., 12)`` for ``nodes: 8-12``. Kept as the ids rather than a
    #: count because the two only coincide for a fleet that starts at node 1.
    worker_node_ids: Tuple[int, ...]
    workers_per_node: int
    capability: str
    device: str
    kv_cache_pin_gb: float
    rotate: bool
    worker_args: List[str]
    experiments: List[Experiment]

    @property
    def worker_nodes(self) -> int:
        """How many nodes the fleet spans - not the highest id, which a range moves apart."""
        return len(self.worker_node_ids)

    def host_for(self, node: int) -> str:
        return host_for_node(node, self.prefix, self.hosts)

    def workers(self) -> List[Worker]:
        """One :class:`Worker` per process to start, in launch order.

        By default every worker gets the experiments in config order, so the whole fleet
        drains the first one, then the second, and the last worker to finish an experiment
        is the only thing holding the fleet back.

        With ``rotate``, worker *w* (counted across the whole fleet, so several workers on
        one node still spread) starts at task ``w mod len(experiments)`` and wraps: every
        coordinator gets work at once - 3 workers each, for 15 workers and 5 experiments -
        instead of the fleet queueing on the first task and then idling on its tail. Every
        experiment then progresses, and none of them finishes early.
        """
        workers: List[Worker] = []
        count = len(self.experiments)
        for node in self.worker_node_ids:
            host = self.host_for(node)
            for k in range(self.workers_per_node):
                index = len(workers)
                worker_id = host if self.workers_per_node == 1 else f"{host}-w{k + 1}"
                targets = []
                for j in range(count):
                    experiment = self.experiments[(j + index) % count if self.rotate else j]
                    # A worker sharing a node with its coordinator talks to it directly
                    # rather than through the network proxy.
                    target_host = "localhost" if experiment.host == host else experiment.host
                    targets.append((experiment.task_id, f"http://{target_host}:{experiment.port}"))
                workers.append(
                    Worker(
                        worker_id=worker_id,
                        host=host,
                        node=node,
                        capability=self.capability,
                        device=self.device,
                        kv_cache_pin_gb=self.kv_cache_pin_gb,
                        targets=targets,
                        args=list(self.worker_args),
                    )
                )
        return workers


def host_for_node(node: int, prefix: str, hosts: Optional[Sequence[str]]) -> str:
    """Node index (1-based) to hostname: an explicit ``hosts:`` list, else ``<prefix><i>``."""
    assert node >= 1, f"Node indices are 1-based; got {node}."
    if hosts is None:
        return f"{prefix}{node}"
    if node > len(hosts):
        raise ValueError(f"node {node} requested but cluster.hosts lists only {len(hosts)} host(s).")
    return hosts[node - 1]


# ── Loading ──────────────────────────────────────────────────────────────────────


def _section(raw: Mapping[str, Any], name: str, defaults: Mapping[str, Any]) -> Dict[str, Any]:
    section = raw.get(name) or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"{name}: expected a mapping; got {type(section).__name__}.")
    unknown = sorted(set(section) - set(defaults))
    if unknown:
        raise ValueError(f"{name}: unknown key(s) {unknown}; known keys are {sorted(defaults)}.")
    merged = dict(defaults)
    merged.update(section)
    return merged


def _coerce_like(value: str, template: Any, key: str) -> Any:
    """A ``--set key=value`` string as the type the config already holds there."""
    if isinstance(template, bool):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"--set {key}: expected a boolean, got {value!r}.")
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def apply_overrides(raw: Dict[str, Any], overrides: Mapping[str, str]) -> Dict[str, Any]:
    """``section.key=value`` pairs (the launcher scripts' own flags) over the file.

    Only the scalar settings are reachable this way; the experiment list is a config
    concern, not a command-line one - use ``select`` to run a subset of it.
    """
    sections = {"cluster": CLUSTER_DEFAULTS, "workers": WORKER_DEFAULTS, "coordinator": COORDINATOR_DEFAULTS}
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if section not in sections or not key:
            raise ValueError(f"--set {dotted}: expected <section>.<key> with section in {sorted(sections)}.")
        if key not in sections[section]:
            raise ValueError(f"--set {dotted}: unknown key; {section} takes {sorted(sections[section])}.")
        if value == "":
            # How a script says "the flag was left at its own empty default" - it must not
            # clobber a value the config file set.
            continue
        current = raw.setdefault(section, {})
        if not isinstance(current, Mapping):
            raise ValueError(f"{section}: expected a mapping; got {type(current).__name__}.")
        current[key] = _coerce_like(value, sections[section][key], dotted)
    return raw


def _validate_producers(experiments: Sequence[Experiment]) -> None:
    """Reject a typo'd producer here rather than in a launch log on a remote node.

    Best-effort: the registry pulls in the whole evaluation stack, which a machine that
    only launches jobs is not required to have installed.
    """
    try:
        from reasondb.coordinator.producers import PRODUCERS
    except Exception:  # pragma: no cover - depends on the launching machine's install
        return
    for experiment in experiments:
        if experiment.producer not in PRODUCERS:
            raise ValueError(
                f"experiments.{experiment.task_id}: unknown producer {experiment.producer!r}; "
                f"expected one of {sorted(PRODUCERS)}."
            )


def parse_cluster_config(
    raw: Mapping[str, Any],
    overrides: Optional[Mapping[str, str]] = None,
    select: Optional[Sequence[str]] = None,
) -> ClusterConfig:
    """Validate a loaded config into the plan both launcher scripts read."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"The config must be a mapping of sections; got {type(raw).__name__}.")
    known_sections = {"cluster", "workers", "coordinator", "experiments"}
    unknown = sorted(set(raw) - known_sections)
    if unknown:
        raise ValueError(f"Unknown top-level section(s) {unknown}; known sections are {sorted(known_sections)}.")

    raw = apply_overrides(dict(raw), overrides or {})
    cluster = _section(raw, "cluster", CLUSTER_DEFAULTS)
    workers = _section(raw, "workers", WORKER_DEFAULTS)
    coordinator = _section(raw, "coordinator", COORDINATOR_DEFAULTS)

    if not cluster["repo"]:
        raise ValueError(
            "cluster.repo is required: an ssh command lands in the container's own workdir, "
            "so the repository has to be named absolutely."
        )
    hosts = cluster["hosts"]
    if hosts is not None:
        if not isinstance(hosts, (list, tuple)) or not hosts:
            raise ValueError("cluster.hosts, when given, is a non-empty list of node hostnames.")
        hosts = [str(host) for host in hosts]

    entries = raw.get("experiments") or []
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError("experiments: at least one entry is required; there is nothing to launch otherwise.")

    shared_args = render_args(coordinator["args"], "coordinator.args")
    experiments: List[Experiment] = []
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            raise ValueError(f"experiments[{position}]: expected a mapping; got {type(entry).__name__}.")
        unknown = sorted(set(entry) - EXPERIMENT_KEYS)
        if unknown:
            raise ValueError(f"experiments[{position}]: unknown key(s) {unknown}; known keys are {sorted(EXPERIMENT_KEYS)}.")
        task_id = str(entry.get("task_id") or "").strip()
        producer = str(entry.get("producer") or "").strip()
        if not task_id or not producer:
            raise ValueError(f"experiments[{position}]: both task_id and producer are required.")
        node = int(entry.get("node", position))
        host = str(entry["host"]) if entry.get("host") else host_for_node(node, cluster["prefix"], hosts)
        experiments.append(
            Experiment(
                task_id=task_id,
                producer=producer,
                node=node,
                host=host,
                port=int(entry.get("port", cluster["port"])),
                # Shared flags first so an experiment's own copy of a flag is the one
                # argparse keeps.
                args=shared_args + render_args(entry.get("args"), f"experiments[{position}].args"),
            )
        )

    seen_tasks: Dict[str, int] = {}
    seen_endpoints: Dict[Tuple[str, int], str] = {}
    for experiment in experiments:
        if experiment.task_id in seen_tasks:
            raise ValueError(f"experiments: task id {experiment.task_id!r} used twice; a task id namespaces a whole sweep.")
        seen_tasks[experiment.task_id] = experiment.node
        endpoint = (experiment.host, experiment.port)
        if endpoint in seen_endpoints:
            raise ValueError(
                f"experiments: {seen_endpoints[endpoint]!r} and {experiment.task_id!r} would both bind "
                f"{experiment.host}:{experiment.port}. One coordinator serves one task - give them "
                "different nodes, or different ports on the same node."
            )
        seen_endpoints[endpoint] = experiment.task_id
    _validate_producers(experiments)

    if select is not None:
        wanted = list(dict.fromkeys(select))
        missing = [task_id for task_id in wanted if task_id not in seen_tasks]
        if missing:
            raise ValueError(f"--experiments: {missing} not in the config; it defines {sorted(seen_tasks)}.")
        # Filtered, never renumbered: the node an experiment runs on is a property of the
        # config, so launching a subset must not move the coordinators the other scripts
        # (and any already-running worker) expect to find.
        experiments = [experiment for experiment in experiments if experiment.task_id in wanted]

    if workers["nodes"] is None:
        if hosts is None:
            raise ValueError("workers.nodes is required unless cluster.hosts names the nodes.")
        worker_node_ids = parse_node_selection(len(hosts))
    else:
        worker_node_ids = parse_node_selection(workers["nodes"])
    if hosts is not None and max(worker_node_ids) > len(hosts):
        raise ValueError(
            f"workers.nodes reaches node {max(worker_node_ids)} but cluster.hosts lists only "
            f"{len(hosts)} host(s)."
        )
    per_node = int(workers["per_node"])
    if per_node < 1:
        raise ValueError(f"workers.per_node must be at least 1; got {per_node}.")
    if workers["capability"] not in WORKER_CAPABILITY_CHOICES:
        raise ValueError(f"workers.capability {workers['capability']!r} not in {list(WORKER_CAPABILITY_CHOICES)}.")
    try:
        kv_cache_pin_gb = float(workers["kv_cache_pin_gb"])
    except (TypeError, ValueError):
        raise ValueError(
            f"workers.kv_cache_pin_gb must be a number of GB; got {workers['kv_cache_pin_gb']!r}."
        ) from None
    if kv_cache_pin_gb < 0:
        raise ValueError(f"workers.kv_cache_pin_gb must be >= 0 (0 = disk-served operators only); got {kv_cache_pin_gb}.")

    return ClusterConfig(
        prefix=str(cluster["prefix"]),
        port=int(cluster["port"]),
        repo=str(cluster["repo"]),
        bashrc=str(cluster["bashrc"]) if cluster["bashrc"] else None,
        home=str(cluster["home"]) if cluster["home"] else None,
        cache_dir=str(cluster["cache_dir"]) if cluster["cache_dir"] else None,
        conda_env=str(cluster["conda_env"]),
        user=str(cluster["user"]) if cluster["user"] else None,
        hosts=hosts,
        worker_node_ids=worker_node_ids,
        workers_per_node=per_node,
        capability=str(workers["capability"]),
        device=str(workers["device"]),
        kv_cache_pin_gb=kv_cache_pin_gb,
        rotate=bool(workers["rotate"]),
        worker_args=render_args(workers["args"], "workers.args"),
        experiments=experiments,
    )


def load_cluster_config(
    path,
    overrides: Optional[Mapping[str, str]] = None,
    select: Optional[Sequence[str]] = None,
) -> ClusterConfig:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"No cluster config at {path}; pass --config, or copy {DEFAULT_CONFIG_PATH} and edit it.")
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    try:
        return parse_cluster_config(raw, overrides=overrides, select=select)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None


# ── Emission (what the launcher scripts read) ────────────────────────────────────


def emit_settings(config: ClusterConfig) -> str:
    """``KEY=value`` lines for the launcher scripts to ``eval``.

    Shell-quoted here, unlike the argument tokens: these never enter the remote
    command string, they are the local script's own variables.
    """
    values = {
        "PREFIX": config.prefix,
        "PORT": str(config.port),
        "REPO": config.repo,
        "BASHRC": config.bashrc or "",
        "HOME_DIR": config.home or "",
        "CACHE_DIR": config.cache_dir or "",
        "CONDA_ENV": config.conda_env,
        "SSH_USER": config.user or "",
        "WORKER_NODES": str(config.worker_nodes),
        # Which nodes, not just how many: a fleet asked for 8-12 spans five nodes whose
        # names the count alone cannot reconstruct.
        "WORKER_NODE_IDS": format_node_selection(config.worker_node_ids),
        "WORKERS_PER_NODE": str(config.workers_per_node),
        "CAPABILITY": config.capability,
        "DEVICE": config.device,
        "NUM_EXPERIMENTS": str(len(config.experiments)),
        "NUM_WORKERS": str(config.worker_nodes * config.workers_per_node),
        "TASK_IDS": " ".join(experiment.task_id for experiment in config.experiments),
        # task@host:port per experiment - what the "watch the queues" hints loop over, so
        # they stay right for a config with explicit hosts or per-experiment ports.
        "ENDPOINTS": " ".join(
            f"{experiment.task_id}@{experiment.host}:{experiment.port}" for experiment in config.experiments
        ),
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())


def emit_coordinators(config: ClusterConfig) -> str:
    """One tab-separated line per coordinator: host, task id, producer, port, arguments."""
    return "\n".join(
        "\t".join([experiment.host, experiment.task_id, experiment.producer, str(experiment.port), shell_join(experiment.argv())])
        for experiment in config.experiments
    )


def emit_workers(config: ClusterConfig) -> str:
    """One tab-separated line per worker: host, worker id, arguments."""
    return "\n".join(
        "\t".join([worker.host, worker.worker_id, shell_join(worker.argv())]) for worker in config.workers()
    )


EMITTERS = {"settings": emit_settings, "coordinators": emit_coordinators, "workers": emit_workers}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG_PATH))
    parser.add_argument("--emit", choices=sorted(EMITTERS), required=True)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="Override one config setting, e.g. --set workers.nodes=4. An empty VALUE is "
        "ignored, so a launcher script can forward a flag it was not given.",
    )
    parser.add_argument(
        "--experiments", type=str, nargs="+", default=None, metavar="TASK_ID",
        help="Launch only these experiments. Node assignment is unchanged - a subset does "
        "not renumber the nodes the rest of the fleet is already pointed at.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: Dict[str, str] = {}
    for item in args.overrides:
        key, sep, value = item.partition("=")
        if not sep:
            print(f"--set takes SECTION.KEY=VALUE; got {item!r}.", file=sys.stderr)
            return 2
        overrides[key.strip()] = value
    try:
        config = load_cluster_config(args.config, overrides=overrides, select=args.experiments)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    rendered = EMITTERS[args.emit](config)
    if rendered:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
