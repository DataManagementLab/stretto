"""Shared per-CR compression + save logic for multi-CR KV-cache generation.

Both the text (`KvTextQaModelWrapper`) and image (`KvImageQaModelWrapper`) servers run
ONE prefill per item, capture the full uncompressed cache + token-importance scores, and
then apply top-k selection (+ key re-rotation) per compression ratio. This module holds
that per-CR loop so the two servers don't duplicate it.

If `save_indices` is True, the kept-token indices (the SAME prefill's top-k selection that
produced the compressed cache) are written next to each cache as
`indices/comp{tag}/idx_{hash}.pt` — a single (n_layers, n_heads, n_kept) int tensor — which
makes masked reconstruction bit-exact (no cross-run top-k divergence). Only for cr > 0 and
non-Finch presses.

`compress_and_save_relative` (hierarchical mode) stores ONE physical baseline cache (e.g.
kv8B02) and, for each more-compressed target (e.g. kv8B05), saves only indices RELATIVE to
the baseline cache (positions into it), so the target can be reconstructed by gathering
from the baseline instead of the full kv8B00 — which then never has to be stored.
"""

import json
import logging
import os

import torch
from transformers import DynamicCache

from kvpress import KeyRerotationPress

from reasondb.backends.mrope_rerotate import canonicalize_keys_mrope, rerotate_keys_mrope

logger = logging.getLogger(__name__)


def _cache_num_layers(cache):
    """Layer count for a DynamicCache across transformers 5.x (.layers) and 4.x (.key_cache)."""
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache.key_cache)


def _cache_kv(cache, layer_idx):
    """Return (keys, values) for a layer across transformers 5.x (.layers) and 4.x (.key_cache)."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def _to_index_dtype(idx_tensor):
    """Down-cast a kept-index tensor to int16 when it fits, else int32 (gather needs int)."""
    int16_max = torch.iinfo(torch.int16).max
    dtype = torch.int16 if int(idx_tensor.max()) <= int16_max else torch.int32
    return idx_tensor.to(dtype).contiguous()


def compress_and_save_multi_cr(
    *,
    full_cache,
    layer_scores,
    layer_modules,
    compression_ratios,
    cache_filenames_by_cr,
    first_device,
    save_indices=False,
    is_finch=False,
    uses_rerotation=True,
    window_size=0,
    context_tokens_count=0,
    mrope_info=None,
):
    """Apply per-CR top-k compression to a captured full cache and save each compressed cache.

    Parameters mirror the variables computed during a single prefill:
      full_cache          : DynamicCache with the full (uncompressed) keys/values.
      layer_scores        : dict[layer_idx -> Tensor(1, n_heads, seq_len)] token scores.
      layer_modules       : dict[layer_idx -> attention module] (provides head_dim, rotary_emb).
      compression_ratios  : list of CRs to produce.
      cache_filenames_by_cr : dict[cr -> output .pt path].
      first_device        : device to do the compression on.
      save_indices        : also write indices/comp{tag}/idx_{hash}.pt (cr>0, non-Finch).
      is_finch / uses_rerotation / window_size / context_tokens_count : press-specific knobs.
      mrope_info          : None for standard-RoPE models. For interleaved-M-RoPE decoders
                            (Qwen3-VL), {"pos_3d": (3, S), "inv_freq": (D/2,),
                            "component_map": (D/2,)} — keys are then rerotated with the
                            M-RoPE-aware math, and the CR=0.0 cache is canonicalized to
                            1D phases instead of stored verbatim (see mrope_rerotate.py).
    """
    if mrope_info is not None:
        assert uses_rerotation and not is_finch, (
            "M-RoPE-aware compression is only supported for rerotation presses "
            "(expected_attention); gather-only or Finch caches would keep mixed "
            "t/h/w phases that the 1D serving math cannot handle."
        )
    num_layers = _cache_num_layers(full_cache)

    for cr in compression_ratios:
        compressed_cache = DynamicCache()
        # Per-layer kept indices for optional index saving (cr > 0, non-Finch).
        collect_indices = save_indices and cr > 0.0 and not is_finch
        layer_indices: list = []

        for layer_idx in range(num_layers):
            full_k, full_v = _cache_kv(full_cache, layer_idx)
            full_keys = full_k.to(first_device)
            full_values = full_v.to(first_device)

            if cr == 0.0:
                if mrope_info is not None:
                    # M-RoPE: even the uncompressed cache must be re-phased so image-token
                    # keys obey the "row i carries 1D phase i" contract of the serving path.
                    k = canonicalize_keys_mrope(
                        full_keys,
                        mrope_info["pos_3d"].to(first_device),
                        mrope_info["inv_freq"].to(first_device),
                        mrope_info["component_map"].to(first_device),
                    )
                    v = full_values.clone()
                else:
                    # No compression: just trim window tokens for Finch
                    k = full_keys[:, :, :-window_size, :] if window_size > 0 else full_keys.clone()
                    v = full_values[:, :, :-window_size, :] if window_size > 0 else full_values.clone()
            else:
                module = layer_modules[layer_idx]
                head_dim = module.head_dim
                scores = layer_scores[layer_idx].to(first_device)
                total_len = full_keys.shape[2]

                if is_finch:
                    n_kept_context = int(context_tokens_count * (1 - cr))
                    n_kept_total = min(n_kept_context + window_size, total_len)
                else:
                    n_kept_total = int(total_len * (1 - cr))

                indices = scores.topk(n_kept_total, dim=-1).indices

                if uses_rerotation or is_finch:
                    # Sort required for key rerotation
                    indices = torch.sort(indices, dim=2).values
                    if mrope_info is not None:
                        k = rerotate_keys_mrope(
                            full_keys,
                            indices,
                            mrope_info["pos_3d"].to(first_device),
                            mrope_info["inv_freq"].to(first_device),
                            mrope_info["component_map"].to(first_device),
                        )
                    else:
                        k = KeyRerotationPress.rerotate_keys(module, indices, full_keys)
                    idx_exp = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    v = full_values.gather(2, idx_exp).contiguous()
                else:
                    # kvzip-style: no rerotation
                    idx_exp = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    k = full_keys.gather(2, idx_exp).contiguous()
                    v = full_values.gather(2, idx_exp).contiguous()

                # Finch: trim window tokens appended at the end after sorting
                if is_finch and window_size > 0:
                    k = k[:, :, :-window_size, :]
                    v = v[:, :, :-window_size, :]

                if collect_indices:
                    layer_indices.append(indices[0].detach().cpu())  # (n_heads, n_kept)

            compressed_cache.update(k.detach().cpu(), v.detach().cpu(), layer_idx)

        cache_filename = cache_filenames_by_cr[cr]
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
        torch.save(compressed_cache, cache_filename)
        logger.info(f"Saved cache CR={cr} → {cache_filename}")
        del compressed_cache

        # Save the same prefill's kept indices next to the cache (bit-exact source).
        if collect_indices and layer_indices:
            idx_tensor = torch.stack(layer_indices, dim=0)  # (n_layers, n_heads, n_kept)
            int16_max = torch.iinfo(torch.int16).max
            idx_tensor = idx_tensor.to(
                torch.int16 if int(idx_tensor.max()) <= int16_max else torch.int32
            ).contiguous()
            comp_dir = os.path.dirname(cache_filename)  # .../{press}/comp{tag}
            idx_dir = os.path.join(
                os.path.dirname(comp_dir), "indices", os.path.basename(comp_dir)
            )
            idx_filename = os.path.join(
                idx_dir, os.path.basename(cache_filename).replace("cache_entry_", "idx_")
            )
            os.makedirs(idx_dir, exist_ok=True)
            torch.save(idx_tensor, idx_filename)
            logger.info(f"Saved indices CR={cr} → {idx_filename}")


def compress_and_save_relative(
    *,
    full_cache,
    layer_scores,
    layer_modules,
    base_cr,
    index_crs,
    base_cache_filename,
    rel_idx_filenames_by_cr,
    first_device,
    uses_rerotation=True,
    mrope_info=None,
):
    """Hierarchical compression: store ONE physical baseline cache at `base_cr`, and for each
    target cr in `index_crs` (each strictly > base_cr) save only the kept-token indices
    RELATIVE to the baseline cache (positions into it) — not absolute into the full kv8B00.

    Because every CR ranks tokens by the SAME prefill scores, a more-compressed target's
    kept set is a strict subset of the baseline's; its relative indices select exactly the
    tokens the absolute method would, expressed as rows of the baseline cache. A target is
    then reconstructed by gathering those rows from the baseline cache and rerotating — row i
    of any rerotated cache carries RoPE phase i, so the existing index reconstruction works
    unchanged when pointed at the baseline.

    Parameters
    ----------
    full_cache          : DynamicCache with the full (uncompressed) keys/values.
    layer_scores        : dict[layer_idx -> Tensor(1, n_heads, seq_len)] token scores.
    layer_modules       : dict[layer_idx -> attention module] (head_dim, rotary_emb).
    base_cr             : float, the physically-stored baseline CR (e.g. 0.2).
    index_crs           : list[float], each strictly > base_cr (e.g. [0.5, 0.9]).
    base_cache_filename : str, .../{press}/comp{base_tag}/cache_entry_{hash}.pt.
    rel_idx_filenames_by_cr : dict[cr -> .../{press}/indices/comp{tgt_tag}/idx_{hash}.pt].
    first_device        : device to do the compression on.
    uses_rerotation     : True for expected_attention (rerotate baseline keys); False = gather only.
    mrope_info          : None for standard-RoPE models; M-RoPE dict for interleaved-M-RoPE
                          decoders (see compress_and_save_multi_cr). The baseline is then
                          canonicalized to 1D phases at generation time, so the serve-time
                          relative reconstruction keeps its existing 1D math unchanged.
    """
    if mrope_info is not None:
        assert uses_rerotation, (
            "M-RoPE-aware relative compression requires a rerotation press "
            "(expected_attention); a gather-only baseline would keep mixed t/h/w phases."
        )
    # base_cr == 0.0 is allowed (experimental): the "baseline" is then the full uncompressed
    # cache — topk over all positions is the identity permutation, so the rerotation is an
    # exact identity (for M-RoPE: a pure canonicalization of the image-token phases) and the
    # relative indices degenerate to absolute positions.
    assert base_cr >= 0.0, "base_cr must be >= 0 for relative compression"
    for cr in index_crs:
        assert cr > base_cr, f"index cr {cr} must be strictly > base cr {base_cr}"

    num_layers = _cache_num_layers(full_cache)
    compressed_cache = DynamicCache()
    rel_idx_by_cr: dict = {cr: [] for cr in index_crs}

    for layer_idx in range(num_layers):
        full_k, full_v = _cache_kv(full_cache, layer_idx)
        full_keys = full_k.to(first_device)
        full_values = full_v.to(first_device)
        module = layer_modules[layer_idx]
        head_dim = module.head_dim
        scores = layer_scores[layer_idx].to(first_device)  # (1, n_heads, seq_len)
        total_len = full_keys.shape[2]

        # --- baseline kept tokens: absolute positions, sorted ascending (same recipe as multi_cr) ---
        n_base = int(total_len * (1 - base_cr))
        a_base = scores.topk(n_base, dim=-1).indices            # (1, n_heads, n_base)
        a_base = torch.sort(a_base, dim=2).values

        if uses_rerotation:
            if mrope_info is not None:
                k = rerotate_keys_mrope(
                    full_keys,
                    a_base,
                    mrope_info["pos_3d"].to(first_device),
                    mrope_info["inv_freq"].to(first_device),
                    mrope_info["component_map"].to(first_device),
                )
            else:
                k = KeyRerotationPress.rerotate_keys(module, a_base, full_keys)
        else:
            k = full_keys.gather(2, a_base.unsqueeze(-1).expand(-1, -1, -1, head_dim)).contiguous()
        v = full_values.gather(2, a_base.unsqueeze(-1).expand(-1, -1, -1, head_dim)).contiguous()
        compressed_cache.update(k.detach().cpu(), v.detach().cpu(), layer_idx)

        # --- relative indices for each more-compressed target ---
        # baseline row r corresponds to a_base[..., r]; rank those rows' scores and keep top n_tgt.
        base_kept_scores = scores.gather(2, a_base)             # (1, n_heads, n_base)
        for cr in index_crs:
            n_tgt = int(total_len * (1 - cr))
            rel = base_kept_scores.topk(n_tgt, dim=-1).indices  # positions in 0..n_base-1
            rel = torch.sort(rel, dim=2).values
            rel_idx_by_cr[cr].append(rel[0].detach().cpu())     # (n_heads, n_tgt)

    # --- save the one physical baseline cache ---
    os.makedirs(os.path.dirname(base_cache_filename), exist_ok=True)
    torch.save(compressed_cache, base_cache_filename)
    logger.info(f"Saved baseline cache CR={base_cr} → {base_cache_filename}")
    del compressed_cache

    # --- save relative indices + a per-dir _meta.json naming the baseline to gather from ---
    base_tag = os.path.basename(os.path.dirname(base_cache_filename))  # e.g. comp02
    for cr in index_crs:
        idx_tensor = _to_index_dtype(torch.stack(rel_idx_by_cr[cr], dim=0))  # (n_layers, n_heads, n_tgt)
        idx_filename = rel_idx_filenames_by_cr[cr]
        idx_dir = os.path.dirname(idx_filename)
        os.makedirs(idx_dir, exist_ok=True)
        torch.save(idx_tensor, idx_filename)
        with open(os.path.join(idx_dir, "_meta.json"), "w") as f:
            json.dump({"from": base_tag}, f)
        logger.info(f"Saved relative indices CR={cr} (from {base_tag}) → {idx_filename}")
