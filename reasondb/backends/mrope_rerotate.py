"""M-RoPE-aware key rerotation for interleaved-MRoPE VL decoders (Qwen3-VL).

The standard 1D rerotation (``KeyRerotationPress._rerotate_cos_sin``) assumes every
frequency dim of a cached key was rotated by ``token_position × inv_freq_i``.
Interleaved M-RoPE violates this for image tokens: frequency dim ``i`` is driven by
one of the three position components (t, h, w) selected by the interleave pattern
(``[THWTHW…TT]``, see ``Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope``), and an
image token's t/h/w differ from each other and from its sequence index. Rerotating
such keys with a scalar 1D delta silently mis-rotates the h/w-driven dims.

This module rerotates kept keys from their original 3D phases to the *canonical 1D
text phases* the rest of the system assumes ("row i of a stored cache carries RoPE
phase i"): for kept row ``r`` originating from sequence index ``j_r``,

    delta_i(r) = (r − pos_{c(i)}(j_r)) · inv_freq_i

where ``c(i) ∈ {t, h, w}`` is the component driving dim ``i``. For text tokens
(t = h = w = sequence position) this reduces exactly to the 1D formula. Once a cache
is canonicalized this way — including the CR=0.0 "uncompressed" cache, whose image
tokens still need their h/w dims re-phased — every downstream path (serving, joins,
relative-index Path A reconstruction, question continuation at cache_position) keeps
using the existing 1D math unchanged, because future question tokens are text tokens
whose three components coincide.

Note the same positional-compaction approximation the 1D pipeline already makes:
compressed caches place kept tokens at consecutive canonical positions, so query→key
relative distances shrink relative to the uncompressed forward pass.
"""

import torch
from transformers.models.llama.modeling_llama import rotate_half


def mrope_component_map(mrope_section, head_dim_half: int) -> torch.Tensor:
    """Which position component (0=t, 1=h, 2=w) drives each frequency dim.

    Mirrors ``Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope``: all dims default
    to t; h overwrites dims 1, 4, 7, … below ``mrope_section[1] * 3``; w overwrites
    dims 2, 5, 8, … below ``mrope_section[2] * 3``.

    Returns a LongTensor of shape (head_dim_half,).
    """
    assert sum(mrope_section) == head_dim_half, (
        f"mrope_section {mrope_section} does not cover head_dim//2 = {head_dim_half}"
    )
    comp = torch.zeros(head_dim_half, dtype=torch.long)
    for dim, offset in enumerate((1, 2), start=1):  # h, w
        comp[offset : mrope_section[dim] * 3 : 3] = dim
    return comp


def rerotate_keys_mrope(
    keys: torch.Tensor,
    indices: torch.Tensor,
    pos_3d: torch.Tensor,
    inv_freq: torch.Tensor,
    component_map: torch.Tensor,
) -> torch.Tensor:
    """Gather kept keys and rerotate them from interleaved-M-RoPE phases to canonical
    1D phases 0..n_kept-1.

    Parameters
    ----------
    keys          : (B, n_kv_heads, S, D) keys as cached from the prefill (M-RoPE-rotated).
    indices       : (B, n_kv_heads, n_kept) kept original sequence indices, sorted ascending.
    pos_3d        : (3, S) the prefill's M-RoPE position_ids (t/h/w rows, batch squeezed).
    inv_freq      : (D/2,) ``rotary_emb.inv_freq``.
    component_map : (D/2,) from ``mrope_component_map``.

    Returns the rerotated keys, shape (B, n_kv_heads, n_kept, D).
    """
    bsz, n_heads, n_kept = indices.shape
    head_dim = keys.shape[-1]
    device, dtype = keys.device, keys.dtype

    pos_3d = pos_3d.to(device)
    # Per-dim original position of every token: (S, D/2)
    pos_per_dim = pos_3d[component_map.to(device), :].transpose(0, 1)

    with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=False):
        # (B, H, n_kept, D/2): the position that rotated each kept token's dim i
        orig = pos_per_dim[indices].float()
        new = torch.arange(n_kept, device=device, dtype=torch.float32)[None, None, :, None]
        freqs = (new - orig) * inv_freq.to(device).float()[None, None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().contiguous()
        sin = emb.sin().contiguous()
    cos, sin = cos.to(dtype), sin.to(dtype)

    keys = keys.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)).contiguous()
    return (keys * cos) + (rotate_half(keys) * sin)


def canonicalize_keys_mrope(
    keys: torch.Tensor,
    pos_3d: torch.Tensor,
    inv_freq: torch.Tensor,
    component_map: torch.Tensor,
) -> torch.Tensor:
    """Rerotate a FULL (uncompressed) cache's keys to canonical 1D phases.

    Identity for text tokens (their phase already equals their sequence index); image
    tokens get their per-dim t/h/w phases re-phased to the sequence index. Used for the
    CR=0.0 cache so it obeys the same "row i carries phase i" contract as compressed ones.
    """
    bsz, n_heads, seq_len, _ = keys.shape
    idx = (
        torch.arange(seq_len, device=keys.device)[None, None, :].expand(bsz, n_heads, seq_len)
    )
    return rerotate_keys_mrope(keys, idx, pos_3d, inv_freq, component_map)
