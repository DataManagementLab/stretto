"""Production KV-cache reconstruction from a baseline cache + relative indices.

When a target compression level (e.g. kv8B05 → dir ``comp0_5``) is stored as relative
indices into a less-compressed physical baseline (e.g. ``comp0_2``) rather than as its own
physical cache, this module reconstructs the target cache on the fly:

    gather the kept rows from the baseline cache + re-rotate them to compacted RoPE
    positions (row i of any rerotated cache carries phase i, so the baseline's rows can be
    re-selected and re-rotated to the target's positions).

The reconstruction is split into a CPU producer (``gather_payload_cpu`` — safe to run in a
loader thread) and a GPU consumer (``rerotate_payload_gpu`` — main thread / default stream),
so CPU loading overlaps GPU work. ``resolve_relative_source`` locates
the baseline + index files for a target/hash via the ``_meta.json`` sidecar written by
``compress_and_save_relative``. Index dirs are nested under their materialized baseline —
``{press}/comp{base}/indices/comp{target}/`` — so the same effective target can be indexed
from several baselines at once. A flat ``{press}/indices/comp{target}/`` layout, if
present, is relocated into the nested layout once at startup by
``migrate_legacy_index_dirs``, so all readers here only ever look at the nested layout.

Every artifact-level failure (missing/corrupt index, missing baseline, dtype/shape/bounds
mismatch) is raised as ``RelativeReconstructError`` so callers can fall back gracefully
instead of crashing the request.
"""

import json
import logging
import os
import shutil

import torch
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

from kvpress import KeyRerotationPress


logger = logging.getLogger(__name__)

# Single switch for ALL host→GPU cache copies across the serving/reconstruction paths
# (text + image servers). Blocking by default: it's race-free and, since
# the prefetch thread already overlaps the next batch's CPU load with this batch's GPU work,
# async copies buy little here. Flip to True to A/B async H2D — note it only actually
# overlaps from PINNED sources; on pageable sources it's effectively a no-op.
H2D_NON_BLOCKING = False


class RelativeReconstructError(Exception):
    """Raised when a relative-index artifact is missing or unusable.

    Callers should catch this and fall back (physical cache if present, else a per-text
    error) — it must never propagate into a silently-wrong answer.
    """


# --------------------------------------------------------------------------------------
# Low-level primitives
# --------------------------------------------------------------------------------------

def _open_kv0_mmap(baseline_path, mmap=True):
    """Memory-map a (baseline) cache on CPU; rows are paged in lazily on gather.

    Handles every on-disk shape: a plain ``{"key_cache", "value_cache"}`` dict, a
    transformers 5.x ``DynamicCache`` (``.layers[i].keys/.values``), and a legacy
    transformers 4.x ``DynamicCache`` reloaded under 5.x (``.key_cache``/``.value_cache``
    still present in ``__dict__``). Returns two per-layer lists of tensors.

    ``mmap=False`` materializes the tensors in RAM instead — used when the baseline is
    retained in the resident store (where mmap pages would silently fall back to disk
    under page-cache pressure), and when the gather would touch most of the file anyway
    (see ``_touched_page_fraction``).
    """
    data = torch.load(baseline_path, map_location="cpu", mmap=mmap, weights_only=False)
    if isinstance(data, dict):
        return data["key_cache"], data["value_cache"]
    if hasattr(data, "layers"):  # transformers 5.x DynamicCache
        return [l.keys for l in data.layers], [l.values for l in data.layers]
    return data.key_cache, data.value_cache  # legacy 4.x object


# Fraction of the baseline's pages the gather may fault in before it is cheaper to read
# the file sequentially instead. Override per run with KV_INDEX_SEQ_READ_TOUCHED; set it
# above 1.0 to always mmap, to 0.0 to always read whole.
#
# At the default of 0.2, mmap is used only where the gather skips >85% of the file. This
# trades a small mean slowdown for runtimes that do not depend on page-cache state.
_SEQ_READ_TOUCHED_ENV = "KV_INDEX_SEQ_READ_TOUCHED"
_SEQ_READ_TOUCHED_DEFAULT = 0.2
_PAGE_BYTES = 4096

# One INFO line per distinct baseline geometry, not per cache: within a run the ratios are
# fixed, so this is exactly one line per served compression ratio. `set.add` is atomic
# under the GIL, so the 4 loader threads need no extra locking.
_seq_read_logged = set()


def _touched_page_fraction(n_kept, baseline_len, head_dim, element_size):
    """Fraction of the baseline's 4 KB pages that a gather of `n_kept` rows must fault in.

    A kept "row" is a contiguous ``head_dim``-element block, so a page holds
    ``4096 / (head_dim * itemsize)`` rows (16 at the head_dim=128 every model here uses).
    The kept rows are spread through the sequence, so a page survives untouched only if
    *none* of its rows was kept — hence ``1 - (1 - keep)**rows_per_page``.

    This is what the read actually costs, and it is very different from the keep fraction:
    keeping 20% of the rows (effective 0.9 out of a 0.5 baseline) still touches 97% of the
    pages, i.e. mmap skips almost nothing while giving up readahead.
    """
    if baseline_len <= 0 or head_dim <= 0 or element_size <= 0:
        return 1.0
    rows_per_page = max(1, _PAGE_BYTES // (head_dim * element_size))
    keep = min(1.0, max(0.0, n_kept / baseline_len))
    return 1.0 - (1.0 - keep) ** rows_per_page


def _should_read_sequentially(n_kept, baseline_len, head_dim, element_size):
    """Whether to re-read the baseline whole instead of faulting it in page by page.

    Page-by-page reads are not just fewer bytes, they are also slower per byte — each
    fault is a round trip, and the scattered pattern defeats readahead (we measured
    ~0.8 GB/s faulting from a cold page cache vs ~2.5 GB/s for a sequential `torch.load`).
    So mmap only pays when it skips a large majority of the file; past the threshold the
    whole-file read is both faster and insensitive to how much of the file happens to be
    in the page cache.
    """
    touched = _touched_page_fraction(n_kept, baseline_len, head_dim, element_size)
    try:
        threshold = float(os.environ.get(_SEQ_READ_TOUCHED_ENV, _SEQ_READ_TOUCHED_DEFAULT))
    except ValueError:
        logger.warning(
            f"{_SEQ_READ_TOUCHED_ENV}={os.environ.get(_SEQ_READ_TOUCHED_ENV)!r} is not a "
            f"float; using {_SEQ_READ_TOUCHED_DEFAULT}"
        )
        threshold = _SEQ_READ_TOUCHED_DEFAULT
    sequential = touched >= threshold

    key = (n_kept, baseline_len, head_dim, element_size, threshold)
    if key not in _seq_read_logged:
        _seq_read_logged.add(key)
        logger.info(
            f"[relative] baseline read mode: {'sequential' if sequential else 'mmap'} — "
            f"gather keeps {n_kept}/{baseline_len} rows ({n_kept / max(baseline_len, 1):.1%}) "
            f"→ touches {touched:.1%} of pages (threshold {threshold:.2f})"
        )
    return sequential


def _gather_rows_cpu(kc, vc, idx):
    """Select kept rows from per-layer (1, H, seq, d) baselines → two (L, H, n_kept, d) buffers.

    A kept "row" is a contiguous d-element block, so this is a row-copy problem, not an
    elementwise one: ``index_select`` on an int64 view of the same bytes moves whole rows
    with memcpy, writing straight into the final output buffer — no per-layer temporaries,
    no cat, no separate pin copy. Output bytes are identical to a ``gather``
    formulation.

    Op granularity matters as much as the kernel: this runs inside 4 concurrent loader
    threads, so the selection is flattened to ONE index_select per layer per tensor
    (heads folded into the row index) — 64 big GIL-releasing calls per cache. A
    per-(layer, head) loop (512 tiny calls) would serialize the loader threads AND the
    main thread's route loop on the GIL.

    The buffers are pinned when CUDA is available (the serving case, enabling fast H2D);
    plain CPU memory otherwise (offline/CPU-only tests, where pinning is impossible).
    ``idx``: (L, H, n_kept) int64, CPU. mmap'd baselines page in only the touched rows.
    """
    n_layers, n_heads, n_kept = idx.shape
    head_dim = kc[0].shape[-1]
    pin = torch.cuda.is_available()
    k_out = torch.empty((n_layers, n_heads, n_kept, head_dim), dtype=kc[0].dtype, pin_memory=pin)
    v_out = torch.empty((n_layers, n_heads, n_kept, head_dim), dtype=vc[0].dtype, pin_memory=pin)
    # View rows as 8-byte words when the row size allows: 4× fewer elements to index.
    wide = torch.int64 if (head_dim * k_out.element_size()) % 8 == 0 else None

    def _as_rows(t, n_rows):
        # (…, X, head_dim) contiguous → (n_rows, row_words) view, no copy
        return (t.view(wide) if wide else t).reshape(n_rows, -1)

    k_dst = _as_rows(k_out, n_layers * n_heads * n_kept).view(n_layers, n_heads * n_kept, -1)
    v_dst = _as_rows(v_out, n_layers * n_heads * n_kept).view(n_layers, n_heads * n_kept, -1)
    # Rows in flattened (H·seq) space, one vectorized op for all layers (layers share
    # seq_len in every cache we produce; assert rather than silently corrupt).
    seq_len = kc[0].shape[2]
    flat_idx = (idx + (torch.arange(n_heads) * seq_len)[None, :, None]).reshape(n_layers, -1)
    for li in range(n_layers):
        assert kc[li].shape[2] == seq_len, f"layer {li} seq {kc[li].shape[2]} != {seq_len}"
        torch.index_select(_as_rows(kc[li], n_heads * seq_len), 0, flat_idx[li], out=k_dst[li])
        torch.index_select(_as_rows(vc[li], n_heads * seq_len), 0, flat_idx[li], out=v_dst[li])
    return k_out, v_out


def _reconstruct_from_indices(kc, vc, indices, rotary_emb, device) -> DynamicCache:
    """Gather kept rows (one H2D) + one batched re-rotation over all layers → DynamicCache.

    ``indices``: (n_layers, n_heads, n_kept) int, CPU, sorted ascending. Result is
    bit-identical to the absolute reconstruction.
    """
    n_layers = len(kc)
    indices = indices.long()

    # select only kept rows on CPU (mmap pages in just those bytes)
    k_cpu, v_cpu = _gather_rows_cpu(kc, vc, indices)

    # ONE host→GPU transfer + ONE batched re-rotation over all layers
    k_kept = k_cpu.to(device, non_blocking=H2D_NON_BLOCKING)
    v_kept = v_cpu.to(device, non_blocking=H2D_NON_BLOCKING)
    idx_dev = indices.to(device, non_blocking=H2D_NON_BLOCKING)
    cos, sin = KeyRerotationPress._rerotate_cos_sin(k_kept, rotary_emb.inv_freq, idx_dev)
    k_kept = (k_kept * cos) + (rotate_half(k_kept) * sin)

    cache = DynamicCache()
    for li in range(n_layers):
        cache.update(k_kept[li:li + 1].contiguous(), v_kept[li:li + 1].contiguous(), li)
    return cache


def reconstruct_cache_indices(baseline_path, indices, rotary_emb, device) -> DynamicCache:
    """Convenience: mmap the baseline + reconstruct in one call (non-overlapped callers / tests)."""
    kc, vc = _open_kv0_mmap(baseline_path)
    return _reconstruct_from_indices(kc, vc, indices, rotary_emb, device)


# --------------------------------------------------------------------------------------
# Production helpers: locate, gather (CPU), rerotate (GPU)
# --------------------------------------------------------------------------------------

def _relative_index_dir(press_dir, target_tag, base_tag):
    """Return the nested relative-index dir for a target, or None if absent.

    The layout is ``{press_dir}/{base_tag}/indices/{target_tag}`` (any legacy flat dir has
    already been relocated here by ``migrate_legacy_index_dirs`` at startup). A dir counts as
    present only when it carries a ``_meta.json`` sidecar (the relative layout);
    absolute-index dirs have no sidecar and are intentionally not matched here.
    """
    index_dir = os.path.join(press_dir, base_tag, "indices", target_tag)
    if os.path.exists(os.path.join(index_dir, "_meta.json")):
        return index_dir
    return None


def resolve_relative_source(press_dir, target_tag, hashed_text, base_tag):
    """Locate the baseline cache + relative index file for a target/hash, or None.

    ``press_dir``  : ``{cache_dir}/{model}/{press}`` (no trailing comp tag).
    ``target_tag`` : e.g. ``comp0_5``.
    ``base_tag``   : the materialized baseline dir, e.g. ``comp0_2``. Indices live at
        ``{press_dir}/{base_tag}/indices/{target_tag}/``. This nested layout disambiguates a
        target reconstructed from several materialized baselines.

    Returns ``(baseline_cache_path, idx_path)`` when a relative layout exists for this
    target, or ``None`` when there is no relative layout (no ``_meta.json`` — e.g. absolute
    indices, which share the dir but carry no sidecar). Raises ``RelativeReconstructError``
    if the sidecar exists but is unreadable / malformed (a corrupt relative layout must
    degrade, not be silently ignored). File existence of the returned paths is validated
    later in ``gather_payload_cpu`` (single I/O attempt, no TOCTOU double-stat).
    """
    index_dir = _relative_index_dir(press_dir, target_tag, base_tag)
    if index_dir is None:
        return None  # no relative layout for this target (normal: absolute / none)
    meta_path = os.path.join(index_dir, "_meta.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        resolved_base_tag = meta["from"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise RelativeReconstructError(f"unusable _meta.json at {meta_path}: {e}") from e

    baseline_cache_path = os.path.join(
        press_dir, resolved_base_tag, f"cache_entry_{hashed_text}.pt"
    )
    idx_path = os.path.join(index_dir, f"idx_{hashed_text}.pt")
    return baseline_cache_path, idx_path


def migrate_legacy_index_dirs(press_dir):
    """Best-effort one-shot MOVE of every legacy flat relative-index dir into the nested layout.

    Legacy generators wrote ``{press_dir}/indices/{target_tag}`` keyed only by the effective
    ratio, so only one materialized baseline's indices could exist per target. The nested
    layout ``{press_dir}/{base_tag}/indices/{target_tag}`` keys by baseline too. Called once at
    startup: for each legacy target dir, read its ``_meta.json`` (key ``"from"``) to learn the
    baseline it belongs to and move it under that baseline, so all readers thereafter look only
    at the nested layout. One atomic rename per dir when on the same filesystem.

    Never fatal: any problem is logged and that legacy dir is left untouched. A legacy dir with
    no ``_meta.json`` (an absolute-index dir, not a relative one) is left in place. When the
    nested target already exists, the legacy dir is a stale duplicate and is left untouched.
    """
    legacy_root = os.path.join(press_dir, "indices")
    if not os.path.isdir(legacy_root):
        return
    for target_tag in os.listdir(legacy_root):
        old_dir = os.path.join(legacy_root, target_tag)
        meta_path = os.path.join(old_dir, "_meta.json")
        if not os.path.isdir(old_dir) or not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path) as f:
                base_tag = json.load(f)["from"]
            new_dir = os.path.join(press_dir, base_tag, "indices", target_tag)
            if os.path.exists(new_dir):
                continue  # nested layout already has this target; legacy dir is a stale dup
            os.makedirs(os.path.dirname(new_dir), exist_ok=True)
            shutil.move(old_dir, new_dir)
            logger.info(f"[relative] migrated legacy index dir {old_dir} → {new_dir}")
        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"[relative] could not migrate {old_dir}: {e}")


def gather_payload_cpu(baseline_cache_path, idx_path):
    """CPU-only producer: load indices + mmap baseline + select kept rows → pinned tensors.

    Returns ``(k_cpu_pinned, v_cpu_pinned, idx)`` ready for one blocking H2D + rerotate.
    Safe to run in a loader thread (torch CPU ops + mmap release the GIL; pinning is
    thread-safe). ANY failure (missing/corrupt index, missing baseline, dtype/shape/bounds
    mismatch) is raised as ``RelativeReconstructError`` so the caller can fall back.
    """
    try:
        # Indices are a plain tensor (saved by _to_index_dtype) → weights_only is safe.
        idx = torch.load(idx_path, map_location="cpu", weights_only=True)
        if idx.dtype not in (torch.int16, torch.int32, torch.int64):
            raise ValueError(f"index dtype {idx.dtype} is not an integer type")
        if idx.dim() != 3:
            raise ValueError(f"index shape {tuple(idx.shape)} is not (n_layers, n_heads, n_kept)")
        idx = idx.long()  # index_select needs int64

        # Nothing on this path is ever held in RAM: keep_in_memory requires a directly
        # materialized cache (effective == materialized), so a relatively-indexed cache is
        # always reconstructed per query.
        kc, vc = _open_kv0_mmap(baseline_cache_path)
        n_layers = len(kc)
        baseline_len = kc[0].shape[2]
        if idx.shape[0] != n_layers:
            raise ValueError(f"index n_layers {idx.shape[0]} != baseline {n_layers}")
        if idx.shape[1] != kc[0].shape[1]:
            raise ValueError(f"index n_heads {idx.shape[1]} != baseline {kc[0].shape[1]}")
        if int(idx.max()) >= baseline_len:
            raise ValueError(
                f"index max {int(idx.max())} >= baseline seq_len {baseline_len} (out of bounds)"
            )

        if _should_read_sequentially(
            idx.shape[2], baseline_len, kc[0].shape[-1], kc[0].element_size()
        ):
            # Dense gather: re-open the same file with one streaming read instead of
            # faulting nearly every page in individually. The mapping above cost ~2 ms and
            # faulted nothing (only tensor metadata was read), so this is not a double read.
            kc, vc = _open_kv0_mmap(baseline_cache_path, mmap=False)

        k_cpu, v_cpu = _gather_rows_cpu(kc, vc, idx)
        return k_cpu, v_cpu, idx
    except RelativeReconstructError:
        raise
    except Exception as e:  # noqa: BLE001 — any failure must degrade, not crash
        raise RelativeReconstructError(
            f"relative gather failed (idx={idx_path}, baseline={baseline_cache_path}): {e}"
        ) from e


def rerotate_payload_gpu(k_cpu, v_cpu, idx, rotary_emb, device) -> DynamicCache:
    """GPU consumer (main thread / default stream): blocking H2D + batched rerotate → cache.

    The H2D is blocking (the rerotate needs the tensor immediately; measured race-free).
    Produces a per-text ``DynamicCache`` shape-identical to a physical compressed cache.
    Single-device variant; multi-GPU serving should prefer ``rerotate_payload_sharded``.
    """
    k_kept = k_cpu.to(device, non_blocking=H2D_NON_BLOCKING)
    v_kept = v_cpu.to(device, non_blocking=H2D_NON_BLOCKING)
    idx_dev = idx.to(device, non_blocking=H2D_NON_BLOCKING)
    cos, sin = KeyRerotationPress._rerotate_cos_sin(k_kept, rotary_emb.inv_freq, idx_dev)
    k_kept = (k_kept * cos) + (rotate_half(k_kept) * sin)
    n_layers = k_kept.shape[0]
    cache = DynamicCache()
    for li in range(n_layers):
        cache.update(k_kept[li:li + 1].contiguous(), v_kept[li:li + 1].contiguous(), li)
    return cache


def rerotate_payload_sharded(k_cpu, v_cpu, idx, rotary_emb, layer_devices) -> DynamicCache:
    """Consumer that lands each layer DIRECTLY on its own device: sliced H2D + local rerotate.

    The rerotation is per-layer independent (cos/sin derive from that layer's own indices,
    followed by an elementwise multiply-add — no cross-layer interaction), so splitting the
    batched call by device range is bit-identical to ``rerotate_payload_gpu``. The payoff on
    multi-GPU: no whole-cache staging on device 0 and no GPU0→GPUn second hop — each GPU
    receives exactly its ``kv_size/num_gpus`` share straight from the pinned CPU buffer,
    which is precisely the peak the batch-size estimator budgets for.

    ``layer_devices``: one target device per layer (``device_map="auto"`` assigns contiguous
    ranges; consecutive equal devices are grouped, so any assignment works). With a single
    device this degenerates to exactly the ``rerotate_payload_gpu`` behavior.
    """
    n_layers = k_cpu.shape[0]
    cache = DynamicCache()
    start = 0
    while start < n_layers:
        dev = layer_devices[start]
        end = start + 1
        while end < n_layers and layer_devices[end] == dev:
            end += 1
        k_kept = k_cpu[start:end].to(dev, non_blocking=H2D_NON_BLOCKING)
        v_kept = v_cpu[start:end].to(dev, non_blocking=H2D_NON_BLOCKING)
        idx_dev = idx[start:end].to(dev, non_blocking=H2D_NON_BLOCKING)
        cos, sin = KeyRerotationPress._rerotate_cos_sin(k_kept, rotary_emb.inv_freq, idx_dev)
        k_kept = (k_kept * cos) + (rotate_half(k_kept) * sin)
        for li in range(start, end):
            j = li - start
            cache.update(k_kept[j:j + 1].contiguous(), v_kept[j:j + 1].contiguous(), li)
        start = end
    return cache
