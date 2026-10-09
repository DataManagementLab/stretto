from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Sequence
import gc
import json
import os
import threading
import yaml
import torch
import logging

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

from reasondb.memory_footprint.memory_report import (
    compute_memory_footprints,
    get_biggest_file_size_gb,
    get_yaml_path,
)

logger = logging.getLogger(__name__)


def validate_kv_compression_ratios(
    effective_compression_ratio: float,
    materialized_compression_ratio: float,
    vanilla: bool,
    keep_in_memory: bool = False,
) -> None:
    """Validate the client-side KV cache-serving arguments.

    Compression ratio is the fraction of KV entries DROPPED (0.0 = no compression,
    1.0 = maximum compression). The MATERIALIZED cache is the physical KV cache
    stored on disk; the EFFECTIVE ratio is what is actually applied at inference,
    derived by indexing into the materialized cache. Indexing can only drop more
    entries, never add them back, so the effective cache must be at least as
    compressed as the materialized one → effective >= materialized.

    When ``vanilla`` is set, no pre-computed KV cache is used at all, so both ratios
    must be 0.0.

    When ``keep_in_memory`` is set, the server holds this operator's caches in CPU RAM
    from ``prepare()`` on, so inference pays only the CPU→GPU copy. That claim is only
    true of a *physical* cache, hence the effective == materialized requirement below.
    """
    assert (
        0.0 <= effective_compression_ratio <= 1.0
    ), f"effective_compression_ratio must be in [0, 1], got {effective_compression_ratio}"
    assert (
        0.0 <= materialized_compression_ratio <= 1.0
    ), f"materialized_compression_ratio must be in [0, 1], got {materialized_compression_ratio}"
    assert effective_compression_ratio >= materialized_compression_ratio, (
        "effective_compression_ratio must be >= materialized_compression_ratio "
        "(the effective cache is derived by indexing into the materialized cache and "
        f"can only be more compressed); got effective={effective_compression_ratio}, "
        f"materialized={materialized_compression_ratio}"
    )
    if vanilla:
        assert (
            effective_compression_ratio == 0.0 and materialized_compression_ratio == 0.0
        ), (
            "vanilla=True uses no pre-computed KV cache, so both compression ratios must be "
            f"0.0; got effective={effective_compression_ratio}, "
            f"materialized={materialized_compression_ratio}"
        )
    if keep_in_memory:
        assert not vanilla, (
            "keep_in_memory=True holds this operator's pre-computed KV caches in the "
            "server's RAM, and vanilla=True runs no pre-computed cache at all — there is "
            "nothing to hold. The two are mutually exclusive."
        )
        assert effective_compression_ratio == materialized_compression_ratio, (
            "keep_in_memory=True requires a directly materialized cache "
            "(effective == materialized); got "
            f"effective={effective_compression_ratio}, "
            f"materialized={materialized_compression_ratio}. Under relative indices the "
            "resident artifact would be the larger, less-compressed baseline, and every "
            "query would still pay a CPU gather plus a GPU rerotate on top of the copy — "
            "not the 'only the CPU→GPU copy' this mode promises. Materialize the effective "
            "ratio directly instead."
        )


# Per-item VRAM reserved for one KV cache = KV_MEM_SAFETY_FACTOR × its resident bf16
# footprint (what get_biggest_file_size_gb measures). The cache itself is ~1× that; the
# rest is cushion for forward-pass activations over the appended query tokens, the caching
# allocator's reserved-vs-allocated slack, and fragmentation across a long run.
KV_MEM_SAFETY_FACTOR = 2.3

# Headroom reserved out of MEASURED free memory, as opposed to KV_MEM_SAFETY_FACTOR, which
# inflates the MODELED per-item cost: reserve the LARGER of a fixed floor or a fraction of
# what is free. Allocator fragmentation and reserved-but-unallocated slop are roughly an
# absolute quantity, driven by block-size mismatches and per-request churn rather than by
# how much memory is installed, so a fraction alone reserves 8 GB on an 80 GB GPU and
# 1.6 GB on a 16 GB one for the same behaviour. The floor makes small-device headroom
# realistic; the fraction still scales the reserve up for large, high-churn batches.
HEADROOM_FRACTION = 0.9
HEADROOM_FLOOR_GB = 2.0

# Per-item floor for the batch-size estimators below, only to keep per_item_gb off zero or
# negative (an all-compressed cache with zero question tokens), NOT to model a real memory
# cost. Where a pipe is available, _llm_activation_gb(pipe, seq_len) already supplies the
# real floor, since max_question_tokens cannot shrink under compression.
_MIN_PER_ITEM_GB = 0.05

# Fallback per-item floor used ONLY when there is no pipe/model config to size a real
# activation cost from (so _llm_activation_gb could not run). Use of this fallback is
# logged as a warning.
_UNCOSTED_ACTIVATION_FALLBACK_GB = 0.5


def _usable_free_gb(
    free_gb: float,
    floor_gb: float = HEADROOM_FLOOR_GB,
    fraction: float = HEADROOM_FRACTION,
) -> float:
    """Free memory (GPU or CPU) to actually budget a batch-size estimate against.

    Reserves max(floor_gb, (1 - fraction) * free_gb) out of the measured free memory and
    returns the rest, floored at 0 so a near-empty device never yields a negative budget.
    """
    reserved = max(floor_gb, (1.0 - fraction) * free_gb)
    return max(free_gb - reserved, 0.0)


def remap_cache_dir_to_local_mirror(cache_dir: str) -> str:
    """Serve KV caches from a node-local mirror when the job script staged one.

    KV_CACHE_LOCAL_MIRROR="<shared_prefix>=<local_prefix>" (exported after rsync-ing the
    cache tree to node-local disk) redirects any incoming cache_dir under <shared_prefix>
    to the mirrored path. The remap only applies when the mirrored directory actually
    exists, so an un-staged job falls back to the shared filesystem unchanged. Writes into
    the remapped dir (footprint YAML, ERRORS.json) land on the ephemeral local disk and
    are recomputed per job.
    """
    spec = os.environ.get("KV_CACHE_LOCAL_MIRROR", "")
    if "=" not in spec:
        return cache_dir
    shared_prefix, local_prefix = spec.split("=", 1)
    if not shared_prefix or not cache_dir.startswith(shared_prefix):
        return cache_dir
    candidate = local_prefix + cache_dir[len(shared_prefix) :]
    if os.path.isdir(candidate):
        logger.info(f"KV cache dir remapped to node-local mirror: {candidate}")
        return candidate
    logger.warning(
        f"KV_CACHE_LOCAL_MIRROR is set but mirror dir does not exist "
        f"({candidate}); falling back to {cache_dir}"
    )
    return cache_dir


def describe_error_response(response) -> str:
    """A readable message out of a non-200 model-server response.

    The servers report an unservable request as JSON (``{"error": ..., "traceback": ...}``)
    — notably a ``keep_in_memory`` request they have no pin budget for, which carries the
    restart instruction the caller needs. Anything else falls back to the raw body.
    """
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            body = response.json()
        except ValueError:
            return response.text
        message = body.get("error", response.text)
        traceback = body.get("traceback")
        return f"{message}\n{traceback}" if traceback else str(message)
    return response.text


def _obj_nbytes(obj) -> int:
    """Best-effort CPU byte size of a loaded KV-cache artifact.

    Handles the shapes we deserialize: a bare tensor (index files), per-layer tensor
    lists/tuples, a ``{"key_cache", "value_cache"}`` dict, and DynamicCache in both the
    transformers 5.x (``.layers[i].keys/.values``) and 4.x (``.key_cache``) layouts.
    Unknown objects size to 0 (they just don't count against the budget).
    """
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_obj_nbytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_obj_nbytes(v) for v in obj.values())
    layers = getattr(obj, "layers", None)
    if layers is not None:
        return sum(
            _obj_nbytes(getattr(layer, "keys", None))
            + _obj_nbytes(getattr(layer, "values", None))
            for layer in layers
        )
    if hasattr(obj, "key_cache"):
        return _obj_nbytes(obj.key_cache) + _obj_nbytes(obj.value_cache)
    return 0


def pin_cache_inplace(cache):
    """Replace every layer's (keys, values) with CUDA-pinned copies, in place.

    Handles the layouts ``_obj_nbytes`` documents: transformers 5.x (``.layers[i].keys``),
    4.50.x (``._cache[i].key_states``) and 4.x (``.key_cache``). ``pin_memory()`` passes an
    already-pinned tensor through, so re-pinning is a no-op.

    Pinned host memory is what makes the per-query H2D copy a DMA transfer rather than a
    staged pageable one — roughly 2x — which is the whole point of holding a cache in RAM.
    The pinning copy is paid once, at ``prepare()``. Both the prepare-time pin and the
    serve-time load MUST go through this one function: they build the object that gets
    stored and the object that gets read, and if the two disagree in layout the serve-time
    lookup silently produces something the batcher cannot use.
    """
    if not torch.cuda.is_available():
        # No CUDA → pin_memory() would raise. The cache is still held in RAM; only the
        # DMA speedup is unavailable (CPU-only runs are tests, not measurements).
        return cache
    if hasattr(cache, "layers"):  # transformers 5.x
        for layer in cache.layers:
            layer.keys, layer.values = layer.keys.pin_memory(), layer.values.pin_memory()
    elif (
        hasattr(cache, "_cache")
        and cache._cache
        and hasattr(cache._cache[0], "key_states")
    ):
        for item in cache._cache:  # transformers 4.50.x
            item.key_states = item.key_states.pin_memory()
            item.value_states = item.value_states.pin_memory()
    else:
        d = cache.__dict__
        d["key_cache"] = [k.pin_memory() for k in d["key_cache"]]
        d["value_cache"] = [v.pin_memory() for v in d["value_cache"]]
    return cache


class PinnedKVUnavailable(RuntimeError):
    """A ``keep_in_memory`` request cannot be served from RAM.

    Never a data problem. Either a configuration problem — no pin budget, or a budget too
    small for the column — or a lifecycle one: the caches were released for a previous
    dataset (see :meth:`PinnedKVStore.release_all`) and not re-pinned, e.g. because the
    client released without also calling ``prepare_memo.reset_prepare_memo``.

    Raised rather than degraded to a disk read, so an in-memory operator never silently
    serves from disk.
    """


class PinnedKVStore:
    """Process-wide CPU-RAM store of the KV caches an ``-in-memory`` operator pinned.

    Nothing lands here unless an operator asked for it *by name* at ``prepare()``. There is
    no eviction and no opportunistic caching: an entry is held until the pinning client
    explicitly gives it up via :meth:`release_all`, and nothing is ever dropped to make room
    for anything else, so the only way to exceed the budget is to raise. Explicit pinning
    (rather than an LRU) guarantees that an in-memory operator is always served from RAM,
    and that a disk operator sharing the same server and cache files is never.

    The loader runs OUTSIDE the lock so concurrent pin threads never serialize on I/O; two
    threads racing on the same first touch may both load, and one result is kept (benign —
    same file). Stored objects are shared across queries, so callers must treat them as
    read-only and copy to GPU out of place.
    """

    def __init__(self, budget_gb: Optional[float] = None):
        self._pinned: Dict[str, tuple] = {}
        self._pinned_bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.budget_bytes = 0
        if budget_gb is not None:
            self.configure(budget_gb)

    def configure(self, budget_gb: Optional[float] = None) -> None:
        """Set the pin budget once, at server startup.

        ``budget_gb=None`` reads ``KV_CACHE_PIN_GB``. Reconfiguring a non-empty store would
        orphan whatever is already held, so it is rejected rather than handled.
        """
        assert not self._pinned, (
            f"PinnedKVStore.configure() called with {len(self._pinned)} entries already "
            "pinned; the budget must be set once, before serving."
        )
        assert "KV_CACHE_RAM_RESIDENT_GB" not in os.environ, (
            "KV_CACHE_RAM_RESIDENT_GB is set, but the opportunistic resident-KV LRU it "
            "controlled has been removed. Caches are now held in RAM only when an "
            "-in-memory operator pins them at prepare(); use KV_CACHE_PIN_GB (or "
            "--kv-cache-pin-gb) to size that. Unset KV_CACHE_RAM_RESIDENT_GB."
        )
        if budget_gb is None:
            try:
                budget_gb = float(os.environ.get("KV_CACHE_PIN_GB", "0") or 0)
            except ValueError:
                raise AssertionError(
                    f"KV_CACHE_PIN_GB must be a number, got "
                    f"{os.environ.get('KV_CACHE_PIN_GB')!r}"
                )
        assert budget_gb >= 0, f"pin budget must be >= 0, got {budget_gb}"
        self.budget_bytes = int(budget_gb * 1e9)
        if self.configured:
            logger.info(f"[pinned-kv] configured with a {budget_gb:.0f} GB pin budget")

    @property
    def configured(self) -> bool:
        return self.budget_bytes > 0

    def require_configured(self, context: str) -> None:
        """Raise unless this server can hold caches in RAM. Never auto-sizes."""
        if not self.configured:
            raise PinnedKVUnavailable(
                f"keep_in_memory=True was requested for {context}, but this model server "
                "has no KV pin budget configured. Restart it with KV_CACHE_PIN_GB=<gb> or "
                "--kv-cache-pin-gb <gb>, sized to hold this column's whole cache set. No "
                "budget is inferred automatically."
            )

    def pin(self, path: str, loader) -> None:
        """Hold ``path``'s deserialized cache in RAM until the next :meth:`release_all`.

        Idempotent: one backend is shared by four operators and ``prepare()`` runs per
        query, so the same file is offered many times over.
        """
        with self._lock:
            if path in self._pinned:
                return
        obj = pin_cache_inplace(loader())
        nbytes = _obj_nbytes(obj)
        with self._lock:
            if path in self._pinned:  # lost a benign race; keep the first
                return
            if self._pinned_bytes + nbytes > self.budget_bytes:
                raise PinnedKVUnavailable(
                    f"Cannot hold {os.path.basename(path)} ({nbytes / 1e9:.2f} GB) in RAM: "
                    f"{self._pinned_bytes / 1e9:.1f} GB of a "
                    f"{self.budget_bytes / 1e9:.1f} GB pin budget is already held. Raise "
                    "KV_CACHE_PIN_GB / --kv-cache-pin-gb, or stop requesting "
                    "keep_in_memory for this column. Nothing is evicted: a pinned cache is "
                    "a promise, not a cache entry. A client releases between datasets "
                    "(POST /release_pinned_kv), so hitting this partway through a "
                    "multi-dataset run means that release did not happen, not that the "
                    "budget is too small for any one column."
                )
            self._pinned[path] = (obj, nbytes)
            self._pinned_bytes += nbytes

    def contains(self, path: str) -> bool:
        """Whether ``path`` is pinned, without counting a hit or a miss.

        For prepare-time accounting; ``get`` is the serve-time accessor and its counters
        are what say whether a measurement was actually served from RAM.
        """
        with self._lock:
            return path in self._pinned

    def get(self, path: str):
        """The pinned cache for ``path``, or None. NEVER loads from disk."""
        with self._lock:
            entry = self._pinned.get(path)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            return entry[0]

    def n_pinned(self) -> int:
        with self._lock:
            return len(self._pinned)

    def release_all(self) -> tuple:
        """Drop every pinned cache; return ``(n_released, released_gb)``.

        The one way an entry leaves this store, and not eviction: nothing is released to
        make room for anything (``pin`` still raises on overflow). A client calls this
        between datasets, when the column it pinned for will never be asked for again —
        without it a long-lived server accumulates one dataset's column after another
        until a later ``pin`` overflows a budget correctly sized for any single one.

        ``hits``/``misses`` are deliberately NOT reset: the serve paths bracket one request
        with two :meth:`stats` reads and act on the difference (see the ``_pin_h0``/
        ``_pin_h1`` pairs in the KV servers), which a concurrent reset would corrupt.

        Safe under concurrent traffic: :meth:`get` hands out the object itself, so a request
        already holding one keeps it alive by refcount. What a release changes for a
        concurrent peer is that its *next* lookup misses — i.e. it fails loudly rather than
        being served from disk.
        """
        with self._lock:
            entries, freed = self._pinned, self._pinned_bytes
            self._pinned, self._pinned_bytes = {}, 0
        n = len(entries)
        # Outside the lock on purpose: cudaFreeHost on page-locked memory synchronizes the
        # device and is slow at a 50 GB budget, and holding _lock across it would stall
        # every concurrent get(). PyTorch exposes no un-pin, so dropping the last reference
        # is the only way back. gc.collect() because the cache objects sit in reference
        # cycles and the resource at stake is page-locked host memory the OS cannot swap,
        # which directly bounds the next dataset's budget. NOT torch.cuda.empty_cache():
        # that is device memory, unrelated to what is held here.
        del entries
        gc.collect()
        logger.info(f"[pinned-kv] released {n} cache(s), {freed / 1e9:.1f} GB")
        return n, freed / 1e9

    def stats(self) -> tuple:
        """(hits, misses, pinned_gb).

        ``hits``/``misses`` are cumulative over the process lifetime; ``pinned_gb`` is
        current occupancy and drops back to 0 on :meth:`release_all`.
        """
        with self._lock:
            return self.hits, self.misses, self._pinned_bytes / 1e9


# Both servers pin into this singleton at prepare() and read it at serve time. Until a
# server calls configure(), any keep_in_memory request fails loudly.
PINNED_KV_STORE = PinnedKVStore()


def pin_caches_in_ram(paths: Sequence[str], context: str) -> int:
    """Pin every distinct cache file in ``paths``; return how many are now resident.

    Callers pass only paths they have already resolved as usable — never a hash with a
    recorded generation error and never a missing one. Pinning an unusable path would turn
    the servers' existing per-item skip into a hard load failure, which is the one way this
    mode could stop an already-running benchmark.

    Loads run on a small pool, mirroring the serve-time loaders: the cost here is I/O.
    """
    targets = list(dict.fromkeys(paths))
    if not targets:
        return 0

    def _load(path: str):
        PINNED_KV_STORE.pin(
            path,
            lambda p=path: torch.load(p, map_location="cpu", weights_only=False),
        )

    with ThreadPoolExecutor(max_workers=min(len(targets), 4)) as pool:
        # list() so a PinnedKVUnavailable from any worker propagates to the caller.
        list(pool.map(_load, targets))
    n_resident = sum(1 for path in targets if PINNED_KV_STORE.contains(path))
    _, _, pinned_gb = PINNED_KV_STORE.stats()
    logger.info(
        f"[pinned-kv] {n_resident}/{len(targets)} caches held in RAM for {context} "
        f"({pinned_gb:.1f} GB of a {PINNED_KV_STORE.budget_bytes / 1e9:.1f} GB budget)"
    )
    return n_resident


def release_pinned_kv_response() -> Dict[str, object]:
    """Release every pinned cache and build the ``/release_pinned_kv`` payload.

    Shared by the text and image servers — which otherwise duplicate their Resource classes
    — so the two cannot drift apart in a field the client asserts on.

    ``n_pinned``/``pinned_gb`` are the state *after* the release, so a caller can check the
    server is actually empty rather than trusting ``n_released``.
    """
    n_released, released_gb = PINNED_KV_STORE.release_all()
    _, _, pinned_gb = PINNED_KV_STORE.stats()
    return {
        "status": "released",
        "n_released": n_released,
        "released_gb": released_gb,
        "n_pinned": PINNED_KV_STORE.n_pinned(),
        "pinned_gb": pinned_gb,
        "pin_budget_gb": PINNED_KV_STORE.budget_bytes / 1e9,
    }


class KVCachingBackendBase:
    def _pin_prepared_caches(
        self,
        usable_paths: Sequence[str],
        *,
        column_name: str,
        compression_ratio: float,
    ) -> Dict[str, float]:
        """Pin a prepared column's caches and report what a client can assert on.

        ``usable_paths`` must already exclude missing and generation-errored items (see
        ``pin_caches_in_ram``). Returns ``n_pin_targets`` / ``n_resident`` — equal by
        construction, since ``pin`` raises rather than falling short — plus the budget
        numbers for the log line. The client asserts the two are equal rather than deriving
        an expected count from its row count, which duplicate items would break.
        """
        context = (
            f"column {column_name!r} on {getattr(self, 'model_name', '?')} "
            f"(cr={compression_ratio})"
        )
        targets = list(dict.fromkeys(usable_paths))
        n_resident = pin_caches_in_ram(targets, context)
        assert n_resident == len(targets), (
            f"pinned {n_resident}/{len(targets)} caches for {context}; "
            "every usable cache must be resident before an -in-memory operator serves"
        )
        _, _, pinned_gb = PINNED_KV_STORE.stats()
        return {
            "n_pin_targets": len(targets),
            "n_resident": n_resident,
            "pinned_gb": pinned_gb,
            "pin_budget_gb": PINNED_KV_STORE.budget_bytes / 1e9,
        }

    @staticmethod
    def _snap_batch(max_batch: int) -> int:
        """Largest multiple of 8 <= max_batch, capped at 1024, floored at 1.

        Shared by every batch-size path so the snapping/cap rule is defined once.
        Snapping (rather than using max_batch directly) keeps batch shapes from
        varying request-to-request, which helps the CUDA caching allocator reuse
        blocks instead of fragmenting over a long-running server, and keeps
        batches BF16-tensor-core-aligned. Unlike snapping to a power of two, a
        multiple of 8 discards at most 7 of the available batch slots.
        """
        if max_batch < 1:
            return 1
        batch = (max_batch // 8) * 8 if max_batch >= 8 else max_batch
        return int(max(1, min(batch, 1024)))

    @staticmethod
    def _min_free_gb(layer_devices: Sequence["torch.device"]) -> float:
        """Minimum free VRAM (GB) across the distinct CUDA devices the model occupies.

        The bottleneck GPU bounds the batch: free memory is NOT summed across GPUs — the
        callers instead divide the per-item cost by num_gpus (the even layer-shard), so the
        budget is "_usable_free_gb(tightest GPU)" against "per-item cost / num_gpus".
        """
        unique_cuda_indices = sorted(
            {
                d.index
                for d in set(layer_devices)
                if hasattr(d, "type") and d.type == "cuda" and d.index is not None
            }
        )
        if unique_cuda_indices:
            min_free_bytes = min(
                torch.cuda.mem_get_info(i)[0] for i in unique_cuda_indices
            )
        else:
            min_free_bytes = min(torch.cuda.mem_get_info(d)[0] for d in layer_devices)
        return min_free_bytes / 1e9

    @staticmethod
    def _llm_activation_gb(pipe, seq_len: float) -> float:
        """One item's LLM forward-pass activation memory (GB) for a seq_len-token pass.

        ``seq_len`` should be the FULL sequence length actually forward-passed, i.e.
        context + newly-appended tokens combined — NOT just the new tokens. Every serving
        path here builds ``input_ids`` by concatenating (dummy-token-padded) context with
        the real question tokens, even when a KV cache already holds the context (see
        kv_cache_*_qa_server.py's batch-building code), so the model still runs Q/K/V
        projections and FFN over the WHOLE padded sequence every request, not just the new
        tokens the cache doesn't yet cover.

        With flash-attention, attention weights are never fully materialised, so only the
        per-layer projection/FFN tensors need counting, in bf16/fp16 (2 bytes/element) — for
        ONE layer, not summed across all layers. All nine tensors a decoder layer
        materialises are enumerated explicitly below. Q/K/V are sized off the real head
        counts rather than ``hidden_size``, since under GQA (``num_key_value_heads`` <
        ``num_attention_heads``) K and V are much smaller than Q.

        Under ``torch.no_grad()`` inference there is no backward pass to keep earlier layers'
        activations alive: each layer's temporaries are freed when its forward() returns,
        and since every decoder layer works on identically-shaped tensors, the caching
        allocator reuses the same blocks layer to layer. So peak activation memory is one
        layer's footprint, not num_layers of them. KV_MEM_SAFETY_FACTOR covers residuals,
        layer-norm buffers, output projections, and unpad/pad index-tensor overhead — the
        same shared dial every other cost term in this file uses.
        """
        if pipe is None or seq_len <= 0:
            return 0.0
        try:
            cfg = pipe.model.config
            # Multimodal (LLaVA) nests the LLM settings under text_config.
            cfg = getattr(cfg, "text_config", cfg)
            hidden = cfg.hidden_size
            intermediate = getattr(cfg, "intermediate_size", 4 * hidden)
            # Same GQA-aware head sizing the bytes_per_kv_token blocks in this file use.
            n_heads = cfg.num_attention_heads
            n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
            head_dim = getattr(cfg, "head_dim", hidden // n_heads)
        except Exception as exc:
            logger.debug(f"Could not compute LLM activation memory: {exc}")
            return 0.0
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        # Elements per token, one decoder layer — every tensor the forward pass materialises:
        elements_per_token = (
            2 * q_dim  # q_proj output, flash-attention output
            + 2 * kv_dim  # k_proj output, v_proj output (GQA: smaller than q_dim)
            + 2 * hidden  # o_proj output, down_proj output
            + 3 * intermediate  # gate_proj, up_proj, silu(gate) * up
        )
        return KV_MEM_SAFETY_FACTOR * (seq_len * elements_per_token * 2) / 1e9

    def _get_max_batch_size(
        self,
        column_name: str,
        batch_size: Optional[int],
        compression_ratio: float,
        cache_dir: str,
        file_paths: Sequence[str],
        layer_devices: Sequence["torch.device"],
        model_name: Optional[str] = None,
        max_question_tokens: int = 0,
        baseline_compression_ratio: Optional[float] = None,
        extra_activation_gb: float = 0.0,
    ) -> int:
        """Estimate the largest safe batch size given available GPU memory.

        KV caches are distributed across layer_devices (one shard per GPU), so the
        per-item cost on each GPU is kv_size_total / num_gpus.  The bottleneck GPU
        (minimum free memory) determines the maximum batch size.

        When max_question_tokens is provided the method also accounts for the
        activation memory of the query tokens processed on top of the KV cache
        during the forward pass.

        ``baseline_compression_ratio`` matters only for the relative-index path, where
        the sized ``file_paths`` are the *baseline* caches (compression ``cr_base``) but
        the cache actually made resident per item is the *target* (compression
        ``compression_ratio > cr_base``, i.e. smaller).  When given, the measured baseline
        size is scaled by ``(1 - compression_ratio) / (1 - cr_base)`` so the batch estimate
        reflects the reconstructed target size instead of the larger baseline — letting the
        index path scale the batch up rather than under-shoot.

        ``extra_activation_gb`` is an undivided per-item cost this formula doesn't itself
        model — e.g. a vision tower's forward-pass activations, which have nothing to do
        with the KV cache or the LLM decoder this method already accounts for. Callers
        with such a cost pass it in rather than this method growing modality-specific
        knowledge; text/audio callers omit it.
        """
        yaml_path = get_yaml_path(cache_dir)
        if batch_size is not None:
            return batch_size

        # Use model name from self if not provided
        if model_name is None and hasattr(self, "model_name"):
            model_name = self.model_name

        # Retrieve max per-item KV size (GB). Prefer the precomputed footprint YAML; a
        # missing or zero entry is not trusted — fall through to file sizing.
        kv_size_gb = None
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}

            dataset = data.get(column_name)
            if dataset is None:
                logger.warning(
                    f"Column '{column_name}' not found in YAML file. Computing memory footprints on the fly..."
                )
                dataset = compute_memory_footprints(
                    cache_dir,
                    column_name,
                    [f.split("/")[-1] for f in file_paths],
                    store=False,
                    model_name=model_name,
                )[column_name]

            # If model_name is provided, look for data for that specific model
            compression_data = dataset.get(model_name) or {}
            kv_size_gb = compression_data.get(compression_ratio) or None
            if kv_size_gb is None:
                logger.warning(
                    f"Footprint YAML at {yaml_path} has no usable entry for "
                    f"'{column_name}'/{model_name}/cr={compression_ratio} "
                    f"(available: {sorted(compression_data)}); sizing from cache files instead."
                )

        if kv_size_gb is None:
            # No usable YAML entry (file absent, or the model/CR key missing/zero — e.g.
            # serving from relative indices before any prepare wrote the estimate). Size
            # directly from the passed cache files (for relative reconstruction these are
            # the baseline caches, an upper bound on the reconstructed target → the batch
            # estimate stays safe).
            if file_paths:
                folder = os.path.dirname(file_paths[0])
                names = [os.path.basename(f) for f in file_paths]
                kv_size_gb = get_biggest_file_size_gb(folder, names)
            else:
                kv_size_gb = 0.0
            if not kv_size_gb:
                logger.warning(
                    f"No footprint YAML at {yaml_path} and no sizable cache files; defaulting batch_size=1"
                )
                return 1
            baseline_kv_size_gb = kv_size_gb
            # Relative-index path: the sized file is the (larger) baseline; the resident
            # per-item cache is the reconstructed target. Scale down by the kept-fraction
            # ratio so the batch estimate matches what actually lives on the GPU. Guard the
            # ratio (need 0 <= cr_base < 1 and cr_base <= cr_target) or fall back to the
            # conservative baseline size.
            if (
                baseline_compression_ratio is not None
                and 0.0 <= baseline_compression_ratio < 1.0
                and baseline_compression_ratio <= compression_ratio < 1.0
            ):
                scale = (1.0 - compression_ratio) / (1.0 - baseline_compression_ratio)
                kv_size_gb *= scale
                logger.info(
                    f"No usable footprint entry; sized baseline={baseline_kv_size_gb:.4f} GB "
                    f"(cr={baseline_compression_ratio}) → target kv_size_gb={kv_size_gb:.4f} GB "
                    f"(cr={compression_ratio}, scale={scale:.3f}) from {len(file_paths)} baseline "
                    f"cache file(s)."
                )
            else:
                logger.info(
                    f"No usable footprint entry; sized kv_size_gb={kv_size_gb:.4f} GB from "
                    f"{len(file_paths)} baseline/physical cache file(s)."
                )
        # Count CUDA devices only (mirrors _get_max_batch_size_vanilla): an offloaded model
        # can leave layers on cpu/meta, which must not inflate num_gpus and over-size the
        # batch — KV for offloaded layers doesn't live in VRAM at all.
        num_gpus = max(
            1, len({d for d in layer_devices if getattr(d, "type", None) == "cuda"})
        )

        # GPU memory cost per batch item, per GPU.
        #
        # Every serving path places each cache layer directly on that layer's own device:
        # physical caches load to CPU (map_location="cpu") and scatter layer-by-layer;
        # relative-index Path A slices the pinned payload per device range and rerotates
        # locally (rerotate_payload_sharded — no whole-cache device-0 staging). So at
        # steady state each cache is split evenly across GPUs and the resident cost per
        # item per GPU is kv_size_gb / num_gpus. KV_MEM_SAFETY_FACTOR (see its definition)
        # is the shared cushion covering the pad/concat batching transient, query-token
        # activations, and allocator slack.
        #
        kv_mem_per_gpu = KV_MEM_SAFETY_FACTOR * kv_size_gb / num_gpus

        # q_mem is the forward-pass activation cost for this item's request. Even though
        # the KV cache already holds the context, the batch-building code (see
        # kv_cache_*_qa_server.py) concatenates dummy-padded context with the real question
        # tokens as input_ids, so the model forward-passes Q/K/V/FFN over the FULL
        # context+question length every request — not just the newly appended question
        # tokens. context length isn't passed to this function directly, but kv_size_gb (the
        # measured resident cache size) lets it be recovered: dividing by the same
        # bytes-per-token used elsewhere in this file. This is a transient cost, unlike
        # kv_mem (loaded directly per-layer, evenly split); which device bears it is not
        # predictable, so every device is checked as if it could bear the full, undivided
        # q_mem_full_gb, against whichever device is tightest on free memory.
        q_mem_full_gb = 0.0
        activation_costed = False
        pipe = getattr(self, "pipe", None)
        if pipe is not None:
            try:
                cfg = pipe.model.config
                cfg = getattr(cfg, "text_config", cfg)
                n_kv_heads = getattr(
                    cfg, "num_key_value_heads", cfg.num_attention_heads
                )
                head_dim = getattr(
                    cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
                )
                bytes_per_kv_token = (
                    cfg.num_hidden_layers * 2 * n_kv_heads * head_dim * 2
                )
                context_tokens = (
                    kv_size_gb * 1e9 / bytes_per_kv_token
                    if bytes_per_kv_token > 0
                    else 0
                )
                seq_len_tokens = context_tokens + max_question_tokens
                q_mem_full_gb = self._llm_activation_gb(pipe, seq_len_tokens)
                activation_costed = True
            except Exception as exc:
                logger.debug(f"Could not compute q_mem_full_gb: {exc}")

        # Per-item cost, checked against the tightest GPU across the whole pipeline. kv_mem
        # is evenly divided (each cache shard loads directly to its own layer's device);
        # q_mem is NOT divided, since any single device may have to absorb it in full.
        #
        # The floor is model-aware whenever possible (see _MIN_PER_ITEM_GB); it falls back
        # to the flat constant only when there was no pipe/config to size q_mem_full_gb from.
        if activation_costed:
            floor_gb = _MIN_PER_ITEM_GB
        else:
            floor_gb = _UNCOSTED_ACTIVATION_FALLBACK_GB / num_gpus
            logger.warning(
                f"No model config available to size per-item activation cost "
                f"(pipe={'present' if pipe is not None else 'unavailable'}); falling back "
                f"to flat floor={floor_gb:.3f}GB/item for the batch-size estimate."
            )
        per_item_gb = max(
            kv_mem_per_gpu + q_mem_full_gb + extra_activation_gb, floor_gb
        )
        min_free_gb = self._min_free_gb(layer_devices)
        max_batch = int(_usable_free_gb(min_free_gb) // per_item_gb)
        logger.info(
            f"Batch size estimate ({num_gpus} GPU(s), factor={KV_MEM_SAFETY_FACTOR}): "
            f"kv/gpu={kv_mem_per_gpu:.2f}GB q_mem={q_mem_full_gb:.2f}GB "
            f"extra_activation={extra_activation_gb:.2f}GB "
            f"per_item={per_item_gb:.2f}GB min_free={min_free_gb:.2f}GB → {max_batch}"
        )

        batch = self._snap_batch(max_batch)

        # CPU RAM sanity check.
        #
        # Caches are loaded directly to GPU via map_location, but torch.load still
        # deserializes one cache through a transient CPU buffer (~2× kv_size_gb for
        # the zip read buffer + tensor storage, freed immediately after GPU transfer).
        # Batch size does not affect CPU peak — only one cache is in CPU RAM at a time.
        # Just warn if even a single cache won't fit.
        if kv_size_gb > 0 and _psutil is not None:
            try:
                available_cpu_gb = _psutil.virtual_memory().available / 1e9
                if 2 * kv_size_gb > _usable_free_gb(available_cpu_gb):
                    logger.warning(
                        f"CPU RAM may be tight for deserialization: "
                        f"one cache needs ~{2 * kv_size_gb:.2f} GB but only "
                        f"{available_cpu_gb:.1f} GB available."
                    )
            except Exception as exc:
                logger.debug(f"CPU RAM check failed: {exc}")

        logger.info(f"Using batch size: {batch}")

        return int(max(1, batch))

    def _get_max_batch_size_join(
        self,
        column_name: str,
        batch_size: Optional[int],
        compression_ratio: float,
        cache_dir: str,
        file_paths: Sequence[str],
        layer_devices: Sequence["torch.device"],
        max_question_tokens: int,
        max_context_tokens: int,
        model_name: Optional[str] = None,
        baseline_compression_ratio: Optional[float] = None,
        extra_activation_gb: float = 0.0,
    ) -> int:
        """Like _get_max_batch_size, but for join-mode forward passes.

        In join mode each forward pass receives (B, max_ctx + q_len) tokens, so
        the question-token activations add memory proportional to q_len / ctx_len
        on top of the KV-cache loading cost.  Both costs are split across GPUs.

        ``baseline_compression_ratio``: see ``_get_max_batch_size`` — set on the
        relative-index path, where ``file_paths`` are the (larger) baseline caches
        and the measured size must be scaled down to the reconstructed target.

        ``extra_activation_gb``: see ``_get_max_batch_size`` — an undivided per-item cost
        (e.g. a vision tower's forward pass) this formula doesn't itself model. Present so
        all three estimators share one signature.
        """
        yaml_path = get_yaml_path(cache_dir)
        if batch_size is not None:
            return batch_size

        if model_name is None and hasattr(self, "model_name"):
            model_name = self.model_name

        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        dataset = data.get(column_name)
        if dataset is None:
            logger.warning(
                f"Column '{column_name}' not found in YAML file. Computing memory footprints on the fly..."
            )
            dataset = compute_memory_footprints(
                cache_dir,
                column_name,
                [f.split("/")[-1] for f in file_paths],
                store=False,
                model_name=model_name,
            )[column_name]

        compression_data = dataset.get(model_name) or {}
        kv_size_gb = compression_data.get(compression_ratio) or None
        if kv_size_gb is None:
            # Missing or zero entry: size from the cache files passed in — the
            # physical targets, or (relative-index path) the baseline caches.
            if file_paths:
                folder = os.path.dirname(file_paths[0])
                names = [os.path.basename(f) for f in file_paths]
                kv_size_gb = get_biggest_file_size_gb(folder, names)
            logger.warning(
                f"[join] Footprint YAML at {yaml_path} has no usable entry for "
                f"'{column_name}'/{model_name}/cr={compression_ratio} "
                f"(available: {sorted(compression_data)}); sized from cache files: "
                f"{kv_size_gb or 0.0:.4f} GB"
            )
            if not kv_size_gb:
                logger.warning(
                    "[join] No sizable cache files either; defaulting batch_size=1"
                )
                return 1
            # Relative-index path: the sized file is the (larger) baseline; the cache made
            # resident per item is the reconstructed target. Same guarded scale-down as
            # _get_max_batch_size.
            if (
                baseline_compression_ratio is not None
                and 0.0 <= baseline_compression_ratio < 1.0
                and baseline_compression_ratio <= compression_ratio < 1.0
            ):
                scale = (1.0 - compression_ratio) / (1.0 - baseline_compression_ratio)
                logger.info(
                    f"[join] Sized baseline={kv_size_gb:.4f} GB (cr={baseline_compression_ratio}) "
                    f"→ target={kv_size_gb * scale:.4f} GB (cr={compression_ratio}, scale={scale:.3f})"
                )
                kv_size_gb *= scale
        # CUDA devices only — same rationale as _get_max_batch_size.
        num_gpus = max(
            1, len({d for d in layer_devices if getattr(d, "type", None) == "cuda"})
        )

        kv_mem_per_gpu = KV_MEM_SAFETY_FACTOR * kv_size_gb / num_gpus

        # q_mem is the forward-pass activation cost, modeled off the real Q/K/V + FFN tensor
        # sizes for the full context+question sequence (see _llm_activation_gb). It is
        # transient and NOT divided by num_gpus: which device bears it is not reliably
        # first_device, so it is checked against whichever is tightest on free memory.
        pipe = getattr(self, "pipe", None)
        seq_len_tokens = max_context_tokens + max_question_tokens
        q_mem_full_gb = self._llm_activation_gb(pipe, seq_len_tokens)

        # Per-item cost, checked against the tightest GPU. kv_mem is evenly divided
        # (resident cache, loaded directly per-layer); q_mem is not (see above).
        #
        # See _get_max_batch_size for the floor rationale: whenever a pipe is available,
        # q_mem_full_gb already implicitly floors at _llm_activation_gb(pipe,
        # max_question_tokens), so only a tiny numerical floor is needed here. The flat
        # fallback only applies when there's no pipe to size a real floor from at all.
        if pipe is not None:
            floor_gb = _MIN_PER_ITEM_GB
        else:
            floor_gb = _UNCOSTED_ACTIVATION_FALLBACK_GB / num_gpus
            logger.warning(
                "[join] No pipe available to size per-item activation cost; falling back "
                f"to flat floor={floor_gb:.3f}GB/item for the batch-size estimate."
            )
        per_item_gb = max(kv_mem_per_gpu + q_mem_full_gb + extra_activation_gb, floor_gb)
        min_free_gb = self._min_free_gb(layer_devices)
        max_batch = int(_usable_free_gb(min_free_gb) // per_item_gb)
        logger.info(
            f"[join] Batch size estimate ({num_gpus} GPU(s), factor={KV_MEM_SAFETY_FACTOR}): "
            f"kv/gpu={kv_mem_per_gpu:.2f}GB q_mem={q_mem_full_gb:.2f}GB "
            f"extra_activation={extra_activation_gb:.2f}GB "
            f"per_item={per_item_gb:.2f}GB min_free={min_free_gb:.2f}GB → {max_batch}"
        )

        batch = self._snap_batch(max_batch)

        # CPU RAM constraint — same formula as _get_max_batch_size (see that method).
        if kv_size_gb > 0 and _psutil is not None:
            try:
                available_cpu_gb = _psutil.virtual_memory().available / 1e9
                cpu_max_batch = max(
                    1, int(_usable_free_gb(available_cpu_gb) / kv_size_gb) - 1
                )
                if cpu_max_batch < batch:
                    cpu_max_batch_p2 = self._snap_batch(cpu_max_batch)
                    logger.info(
                        f"[join] CPU RAM constraint: batch {batch} → {cpu_max_batch_p2}  "
                        f"(available_cpu={available_cpu_gb:.1f} GB, "
                        f"kv_per_item={kv_size_gb:.3f} GB)"
                    )
                    batch = cpu_max_batch_p2
            except Exception as exc:
                logger.debug(f"[join] CPU RAM check failed: {exc}")
        elif kv_size_gb > 0 and _psutil is None:
            logger.debug("[join] psutil not available — CPU RAM constraint skipped")

        logger.info(f"[join] Using batch size: {batch}")

        return int(max(1, batch))

    def _get_max_batch_size_vanilla(
        self,
        max_prompt_tokens: int,
        layer_devices: Sequence["torch.device"],
        batch_size: Optional[int] = None,
        extra_activation_gb: float = 0.0,
    ) -> int:
        """Memory-safe batch size for the vanilla path (full prefill, NO cache file).

        The uncompressed methods (kv*00) have no cache on disk, so — unlike the cache
        paths that MEASURE ``kv_size_gb`` from a file — we DERIVE the per-item cost: the
        full KV cache built during prefill is ``max_prompt_tokens × bytes_per_kv_token``,
        computed analytically from the model config.

        Unlike the cache paths, EVERY token of the prompt is freshly forward-passed here
        (there is no resident cache to lean on). That forward pass runs sequentially
        through the pipeline shard, and the transient activation cost can concentrate
        almost entirely on a single (not necessarily the first) GPU rather than spreading
        evenly. The per-item cost here is therefore NOT divided across GPUs, and is checked
        against whichever GPU has the least free memory across the whole pipeline.

        ``batch_size``: explicit override; a positive value short-circuits the estimate,
        ``None``/``<=0`` means auto-estimate.

        ``extra_activation_gb``: see ``_get_max_batch_size`` — an undivided per-item cost
        (e.g. a vision tower's forward pass) this formula doesn't itself model.
        """
        if batch_size is not None and batch_size > 0:
            return batch_size

        pipe = getattr(self, "pipe", None)
        # Count CUDA devices only: an offloaded model (vanilla tolerates offload) can leave
        # some layers on `meta`, which must not inflate num_gpus and over-size the batch.
        num_gpus = max(
            1, len({d for d in layer_devices if getattr(d, "type", None) == "cuda"})
        )
        if pipe is None or max_prompt_tokens <= 0:
            return 1

        cfg = pipe.model.config
        # Multimodal (LLaVA) nests the LLM settings under text_config.
        cfg = getattr(cfg, "text_config", cfg)
        n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        bytes_per_kv_token = cfg.num_hidden_layers * 2 * n_kv_heads * head_dim * 2

        # Resident KV cache built during prefill (kept for the decode steps that follow)
        # PLUS the transient Q/K/V/FFN activation memory of the prefill forward pass itself,
        # over the full max_prompt_tokens length (see _llm_activation_gb). Any single GPU in
        # the pipeline may have to absorb this in full, so it's checked against the tightest
        # GPU, not divided.
        kv_cache_gb = (
            KV_MEM_SAFETY_FACTOR * max_prompt_tokens * bytes_per_kv_token / 1e9
        )
        activation_gb = self._llm_activation_gb(pipe, max_prompt_tokens)
        # Unlike _get_max_batch_size/_get_max_batch_size_join, pipe and max_prompt_tokens are
        # guaranteed valid here (see the early return above), so kv_cache_gb and
        # activation_gb are real model-derived costs and the floor is only the numerical
        # backstop.
        per_item_gb = max(
            kv_cache_gb + activation_gb + extra_activation_gb, _MIN_PER_ITEM_GB
        )

        min_free_gb = self._min_free_gb(layer_devices)
        max_batch = int(_usable_free_gb(min_free_gb) // per_item_gb)
        batch = self._snap_batch(max_batch)
        logger.info(
            f"[vanilla] Batch size estimate ({num_gpus} GPU(s), factor={KV_MEM_SAFETY_FACTOR}): "
            f"max_prompt_tokens={max_prompt_tokens} kv_cache={kv_cache_gb:.3f}GB "
            f"activation={activation_gb:.3f}GB extra_activation={extra_activation_gb:.3f}GB "
            f"per_item={per_item_gb:.3f}GB min_free={min_free_gb:.2f}GB → {max_batch} → {batch}"
        )
        return batch

    @staticmethod
    def _normalize_error_key(key: str) -> str:
        """Reduce an ERRORS.json key to the bare content hash.

        Different generation paths key ERRORS.json differently: some write the bare
        hash (e.g. ``prepare_indices_relative``), others write the full cache file path
        (e.g. the physical ``prepare_caches``, ``"{save_dir}/cache_entry_{hash}.pt"``).
        Callers look items up by bare hash, so normalize both forms to that here rather
        than requiring every writer to agree on a format.
        """
        name = os.path.basename(key)
        if name.startswith("cache_entry_") and name.endswith(".pt"):
            return name[len("cache_entry_") : -len(".pt")]
        return name

    @classmethod
    def _load_known_errors(
        cls, press_dir: str, target_tag: str, base_tag: str
    ) -> dict[str, str]:
        """Merge ERRORS.json from the relative baseline and the target dir.

        An item can be absent from a relative-index target for two different reasons:
        (a) nobody has run the offline generator for it yet — a setup mistake the
        caller should crash on — or (b) the offline generator DID run and recorded a
        per-item failure in ERRORS.json (e.g. a malformed text/corrupted image). (b) is
        a known, already-tolerated data issue, not a setup mistake, so callers should
        report it as n_generation_errors, not n_missing. The baseline dir is resolved
        via the target's ``_meta.json`` (key ``"from"``), read from the nested layout
        ``{press_dir}/{base_tag}/indices/{target_tag}`` (any legacy flat dir has already been
        relocated there by ``migrate_legacy_index_dirs`` at startup).
        """
        known_errors: dict[str, str] = {}
        meta_path = os.path.join(press_dir, base_tag, "indices", target_tag, "_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    base_tag = json.load(f)["from"]
                base_errors_path = f"{press_dir}/{base_tag}/ERRORS.json"
                if os.path.exists(base_errors_path):
                    with open(base_errors_path) as f:
                        known_errors.update(
                            {
                                cls._normalize_error_key(k): v
                                for k, v in json.load(f).items()
                            }
                        )
            except (OSError, ValueError, KeyError, TypeError) as e:
                logger.warning(f"[relative] could not read baseline ERRORS.json: {e}")
        target_errors_path = f"{press_dir}/{target_tag}/ERRORS.json"
        if os.path.exists(target_errors_path):
            try:
                with open(target_errors_path) as f:
                    known_errors.update(
                        {
                            cls._normalize_error_key(k): v
                            for k, v in json.load(f).items()
                        }
                    )
            except (OSError, ValueError) as e:
                logger.warning(f"[relative] could not read physical ERRORS.json: {e}")
        return known_errors
