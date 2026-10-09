"""The single definition of the inference-statistics wire format.

Every KV cache server attaches a ``"stats"`` object to every inference response, and
every client parses it. This module owns that contract so the servers and the clients
cannot drift apart, and so the monitor (:mod:`reasondb.monitor`) has exactly one schema
to consume.

Two design rules keep this cheap enough to sit next to LLM inference:

* :func:`build_inference_stats` only *reports* values the server already computed while
  serving the request (batch size, the per-batch ``torch.cuda`` numbers the
  per-batch debug logs collect, the KV cache load/route/wait accumulators). It
  performs no measurement of its own and issues no CUDA calls.
* Keys are always present. A field that does not apply on a given path is ``None``
  rather than absent, so no consumer ever has to branch on key presence.

This module deliberately does not import :mod:`reasondb.monitor`: the servers run as
separate processes and must not depend on the monitoring package.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Module import, never `from ... import record_kv_inference`: the monitor rebinds its
# global sink at run start. `reasondb.monitor.collector` is standard-library only, so the
# KV servers can import this module without pulling in flask, pandas or torch.
from reasondb.monitor import collector as _monitor_collector

# Bump when a key is removed or its meaning changes. Adding an optional key does not
# require a bump; ``parse_inference_stats`` only requires the keys it knows about.
STATS_SCHEMA_VERSION = 2

SERVERS = ("kv_text_qa", "kv_image_qa", "kv_audio_qa")
#: ``prepare`` is not inference, but cache materialization is a large slice of a run's
#: time, and including it keeps "every response carries stats" true across every route
#: of every KV server.
PATHS = ("kv", "vanilla", "direct", "join", "prepare")

# Every key ``build_inference_stats`` emits. Pinned by tests so the servers and the
# monitor agree on one shape across all three modalities and all four paths.
STATS_KEYS = frozenset(
    {
        "schema",
        "server",
        "path",
        "model_name",
        "n_items",
        "n_batches",
        "batch_size",
        "effective_compression_ratio",
        "materialized_compression_ratio",
        "vanilla",
        "keep_in_memory",
        "server_elapsed_s",
        "cache_load_s",
        "cache_route_s",
        "cache_wait_s",
        "n_caches",
        "gpu",
        "min_free_gb",
        "pinned_hits",
        "pinned_misses",
        "pinned_gb",
        "n_errors",
    }
)

GPU_KEYS = frozenset({"index", "peak_allocated_gb", "free_gb", "total_gb"})


def gpu_entry(
    index: int,
    peak_allocated_gb: Optional[float],
    free_gb: Optional[float],
    total_gb: Optional[float],
) -> Dict[str, Any]:
    """One per-device record for the ``"gpu"`` list.

    ``peak_allocated_gb`` is ``None`` on paths that never reset the peak counter; the
    free/total pair comes from ``torch.cuda.mem_get_info`` the server already called.
    """
    assert index >= 0, f"GPU index must be non-negative; got {index}."
    return {
        "index": int(index),
        "peak_allocated_gb": _opt_float(peak_allocated_gb),
        "free_gb": _opt_float(free_gb),
        "total_gb": _opt_float(total_gb),
    }


def gpu_snapshot(
    peak_allocated_gb: Optional[Dict[int, float]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Per-device free/total memory plus the caller's per-batch peaks.

    ``torch.cuda.mem_get_info`` is a driver query (``cudaMemGetInfo``), not a stream
    synchronize, and the KV servers already call it once per batch for their
    per-batch debug logs. Calling it once more per *request* is a few tens of
    microseconds next to a ``model.generate`` measured in seconds.

    Returns ``([], None)`` when CUDA is unavailable, so CPU-only and simulated runs get
    the same shape as GPU runs.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep of the servers
        return [], None
    if not torch.cuda.is_available():
        return [], None

    peaks = peak_allocated_gb or {}
    entries: List[Dict[str, Any]] = []
    min_free: Optional[float] = None
    for index in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(index)
        except Exception:  # pragma: no cover - a device can vanish mid-run
            continue
        free_gb = free / 1e9
        entries.append(
            gpu_entry(
                index=index,
                peak_allocated_gb=peaks.get(index),
                free_gb=free_gb,
                total_gb=total / 1e9,
            )
        )
        min_free = free_gb if min_free is None else min(min_free, free_gb)
    return entries, min_free


def build_inference_stats(
    *,
    server: str,
    path: str,
    model_name: str,
    n_items: int,
    server_elapsed_s: float,
    n_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    effective_compression_ratio: Optional[float] = None,
    materialized_compression_ratio: Optional[float] = None,
    vanilla: bool = False,
    cache_load_s: Optional[float] = None,
    cache_route_s: Optional[float] = None,
    cache_wait_s: Optional[float] = None,
    n_caches: Optional[int] = None,
    gpu: Optional[Sequence[Dict[str, Any]]] = None,
    min_free_gb: Optional[float] = None,
    keep_in_memory: bool = False,
    pinned_hits: Optional[int] = None,
    pinned_misses: Optional[int] = None,
    pinned_gb: Optional[float] = None,
    n_errors: int = 0,
) -> Dict[str, Any]:
    """Assemble the stats object a server attaches to an inference response.

    All arguments are values the caller already has in scope; nothing is measured here.
    """
    assert server in SERVERS, f"Unknown server {server!r}; expected one of {SERVERS}."
    assert path in PATHS, f"Unknown inference path {path!r}; expected one of {PATHS}."
    assert model_name, "model_name must be a non-empty string."
    assert n_items >= 0, f"n_items must be non-negative; got {n_items}."
    assert server_elapsed_s >= 0.0, (
        f"server_elapsed_s must be non-negative; got {server_elapsed_s}."
    )
    assert batch_size is None or batch_size >= 1, (
        f"batch_size must be at least 1 when reported; got {batch_size}."
    )
    assert n_batches is None or n_batches >= 0, (
        f"n_batches must be non-negative when reported; got {n_batches}."
    )
    assert n_errors >= 0, f"n_errors must be non-negative; got {n_errors}."
    _assert_ratio("effective_compression_ratio", effective_compression_ratio)
    _assert_ratio("materialized_compression_ratio", materialized_compression_ratio)

    gpu_list: List[Dict[str, Any]] = list(gpu) if gpu is not None else []
    for entry in gpu_list:
        assert set(entry) == GPU_KEYS, (
            f"GPU entry keys {sorted(entry)} do not match {sorted(GPU_KEYS)}; "
            "build them with gpu_entry()."
        )

    stats = {
        "schema": STATS_SCHEMA_VERSION,
        "server": server,
        "path": path,
        "model_name": model_name,
        "n_items": int(n_items),
        "n_batches": None if n_batches is None else int(n_batches),
        "batch_size": None if batch_size is None else int(batch_size),
        "effective_compression_ratio": _opt_float(effective_compression_ratio),
        "materialized_compression_ratio": _opt_float(materialized_compression_ratio),
        "vanilla": bool(vanilla),
        "keep_in_memory": bool(keep_in_memory),
        "server_elapsed_s": float(server_elapsed_s),
        "cache_load_s": _opt_float(cache_load_s),
        "cache_route_s": _opt_float(cache_route_s),
        "cache_wait_s": _opt_float(cache_wait_s),
        "n_caches": None if n_caches is None else int(n_caches),
        "gpu": gpu_list,
        "min_free_gb": _opt_float(min_free_gb),
        "pinned_hits": None if pinned_hits is None else int(pinned_hits),
        "pinned_misses": None if pinned_misses is None else int(pinned_misses),
        "pinned_gb": _opt_float(pinned_gb),
        "n_errors": int(n_errors),
    }
    assert set(stats) == STATS_KEYS, (
        f"build_inference_stats emitted {sorted(set(stats) ^ STATS_KEYS)} "
        "off the pinned key set."
    )
    return stats


def parse_inference_stats(
    response_json: Dict[str, Any],
    *,
    client_elapsed_s: float,
    endpoint: str,
) -> Dict[str, Any]:
    """Validate the ``"stats"`` object of a server response and enrich it client-side.

    Adds ``client_elapsed_s`` (wall clock around the HTTP call, so the gap against
    ``server_elapsed_s`` exposes transport and queueing) and ``endpoint``.

    Asserts rather than degrades: every KV server in this repo attaches stats to every
    response, so a missing object means a server is out of date with its client, and
    silently dropping the telemetry would hide that.
    """
    assert isinstance(response_json, dict), (
        f"Expected a JSON object from {endpoint}, got {type(response_json).__name__}."
    )
    assert "stats" in response_json, (
        f"Response from {endpoint} carries no 'stats' object. Every KV server attaches "
        "one to every inference response - the server is likely running older code than "
        "this client."
    )
    stats = response_json["stats"]
    assert isinstance(stats, dict), (
        f"'stats' from {endpoint} must be an object; got {type(stats).__name__}."
    )
    assert stats.get("schema") == STATS_SCHEMA_VERSION, (
        f"'stats' from {endpoint} has schema {stats.get('schema')!r}, this client "
        f"expects {STATS_SCHEMA_VERSION}."
    )
    missing = STATS_KEYS - set(stats)
    assert not missing, (
        f"'stats' from {endpoint} is missing {sorted(missing)}; keys are always present "
        "in this schema (inapplicable ones are None)."
    )
    assert client_elapsed_s >= 0.0, (
        f"client_elapsed_s must be non-negative; got {client_elapsed_s}."
    )

    enriched = dict(stats)
    enriched["client_elapsed_s"] = float(client_elapsed_s)
    enriched["endpoint"] = endpoint
    return enriched


def forward_stats(
    response_json: Dict[str, Any], client_elapsed_s: float, endpoint: str
) -> None:
    """Client-side: hand a server's ``stats`` block to the run monitor.

    The disabled case returns before touching ``response_json`` at all, so a run without
    monitoring pays one module lookup and one identity comparison per HTTP call - against
    a call that just waited on an LLM.

    ``parse_inference_stats`` asserts on a malformed or missing ``stats`` block, since a
    mismatch indicates a client/server version skew. Telemetry must never abort a
    benchmark, so the assertion is caught here and reported as a monitor-side error
    event instead.
    """
    if not _monitor_collector.is_enabled():
        return
    try:
        stats = parse_inference_stats(
            response_json, client_elapsed_s=client_elapsed_s, endpoint=endpoint
        )
    except (AssertionError, TypeError, KeyError) as exc:
        _monitor_collector.record_error(f"inference_stats:{endpoint}", str(exc))
        return
    _monitor_collector.record_kv_inference(stats)


def record_simulated_call(
    *,
    model_id: str,
    modality: str,
    n_items: int,
    runtime_s: float,
    endpoint: str,
) -> None:
    """Record a call that ``--simulate`` replayed instead of sending to a server.

    Without this the dashboard is empty for exactly the runs that are easiest to inspect
    (no GPU, no servers). The record is marked ``path="simulate"`` via ``endpoint`` and
    carries no GPU data, because none was involved.
    """
    if not _monitor_collector.is_enabled():
        return
    _monitor_collector.record_kv_inference(
        {
            "schema": STATS_SCHEMA_VERSION,
            "server": modality,
            "path": "kv",
            "model_name": model_id,
            "n_items": int(n_items),
            "n_batches": None,
            "batch_size": None,
            "effective_compression_ratio": None,
            "materialized_compression_ratio": None,
            "vanilla": False,
            "keep_in_memory": False,
            "server_elapsed_s": float(runtime_s),
            "client_elapsed_s": 0.0,
            "cache_load_s": None,
            "cache_route_s": None,
            "cache_wait_s": None,
            "n_caches": None,
            "gpu": [],
            "min_free_gb": None,
            "pinned_hits": None,
            "pinned_misses": None,
            "pinned_gb": None,
            "n_errors": 0,
            "endpoint": endpoint,
            "simulated": True,
        }
    )


def _assert_ratio(name: str, value: Optional[float]) -> None:
    assert value is None or 0.0 <= value <= 1.0, (
        f"{name} must lie in [0, 1] when reported; got {value}."
    )


def _opt_float(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)
