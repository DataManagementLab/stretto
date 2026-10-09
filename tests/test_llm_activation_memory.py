"""LLM forward-pass activation memory — `KVCachingBackendBase._llm_activation_gb`.

This helper is the single per-item activation term behind every batch-size estimator in
the codebase (`_get_max_batch_size`, `_get_max_batch_size_join`,
`_get_max_batch_size_vanilla` in kv_cache_base.py, plus the image server's
`_get_max_batch_size_image_join`), so an error here scales straight into an OOM.

A decoder layer materialises 9 tensors: Q/K/V, the flash-attention output, the o_proj
output, the FFN gate/up projections, the SiLU product and the down_proj output, with K/V
sized by GQA rather than a flat `3 * hidden_size`. Counting only Q/K/V and gate/up (5 of
the 9) under-estimates memory measurably (0.060 vs. ~0.075 GB/item for Llama-3.1-8B in the
configuration below). These tests pin the complete enumeration.
"""

import types

import pytest

from reasondb.backends import kv_cache_base as kvb
from reasondb.backends.kv_cache_base import KV_MEM_SAFETY_FACTOR, KVCachingBackendBase


# ── Fakes ────────────────────────────────────────────────────────────────────


class _Config:
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


class _Pipe:
    def __init__(self, config):
        self.model = types.SimpleNamespace(config=config)


def _llama_31_8b():
    return _Config(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )


def _llama_31_70b():
    return _Config(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
    )


def _expected_gb(cfg, seq_len, factor=None):
    """Mirror of the implementation, so the test pins the formula independently."""
    factor = KV_MEM_SAFETY_FACTOR if factor is None else factor
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) * cfg.head_dim
    elements = (
        2 * q_dim  # q_proj output, flash-attention output
        + 2 * kv_dim  # k_proj, v_proj outputs
        + 2 * cfg.hidden_size  # o_proj output, down_proj output
        + 3 * cfg.intermediate_size  # gate, up, silu(gate) * up
    )
    return factor * (seq_len * elements * 2) / 1e9


def _partial_formula_gb(cfg, seq_len, factor=None):
    """A 5-of-9-tensor count: QKV as a flat 3*hidden, FFN gate+up only."""
    factor = KV_MEM_SAFETY_FACTOR if factor is None else factor
    attn = 3 * seq_len * cfg.hidden_size * 2
    ffn = 2 * seq_len * cfg.intermediate_size * 2
    return factor * (attn + ffn) / 1e9


# ── Exact formula ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cfg_fn, expected_elements_per_token",
    [(_llama_31_8b, 61440), (_llama_31_70b, 120832)],
    ids=["llama-3.1-8b", "llama-3.1-70b"],
)
def test_matches_complete_tensor_enumeration(cfg_fn, expected_elements_per_token):
    cfg = cfg_fn()
    got = KVCachingBackendBase._llm_activation_gb(_Pipe(cfg), 1000)
    assert got == pytest.approx(_expected_gb(cfg, 1000))
    # Independently spelled-out element count, so a change to both the implementation and
    # the _expected_gb mirror still has to justify itself against a hand-computed number.
    assert got == pytest.approx(
        KV_MEM_SAFETY_FACTOR * 1000 * expected_elements_per_token * 2 / 1e9
    )


def test_gqa_sizes_kv_smaller_than_q():
    """Under GQA k/v are num_key_value_heads wide, not hidden_size wide."""
    gqa = _llama_31_8b()
    mha = _llama_31_8b()
    mha.num_key_value_heads = mha.num_attention_heads
    assert KVCachingBackendBase._llm_activation_gb(
        _Pipe(gqa), 1000
    ) < KVCachingBackendBase._llm_activation_gb(_Pipe(mha), 1000)


def test_missing_num_key_value_heads_falls_back_to_mha():
    cfg = _llama_31_8b()
    del cfg.num_key_value_heads
    assert KVCachingBackendBase._llm_activation_gb(_Pipe(cfg), 500) == pytest.approx(
        _expected_gb(cfg, 500)
    )


def test_missing_head_dim_is_derived_from_hidden_size():
    cfg = _llama_31_8b()
    del cfg.head_dim
    # hidden_size // num_attention_heads == 128, i.e. the value of the deleted head_dim
    assert KVCachingBackendBase._llm_activation_gb(_Pipe(cfg), 500) == pytest.approx(
        _expected_gb(_llama_31_8b(), 500)
    )


def test_unwraps_multimodal_text_config():
    inner = _llama_31_8b()
    outer = _Config(text_config=inner)
    assert KVCachingBackendBase._llm_activation_gb(
        _Pipe(outer), 500
    ) == pytest.approx(_expected_gb(inner, 500))


# ── Comparison against an under-counting formula ─────────────────────────────


@pytest.mark.parametrize(
    "cfg_fn", [_llama_31_8b, _llama_31_70b], ids=["llama-3.1-8b", "llama-3.1-70b"]
)
def test_exceeds_the_partial_tensor_count(cfg_fn):
    """The complete enumeration must exceed the 5-of-9-tensor count.

    ~1.5x for both Llama-3.1 shapes: the four added tensors outweigh the GQA correction
    that shrinks k/v.
    """
    cfg = cfg_fn()
    new = KVCachingBackendBase._llm_activation_gb(_Pipe(cfg), 1000)
    old = _partial_formula_gb(cfg, 1000)
    assert new > old
    assert new / old == pytest.approx(1.5, abs=0.05)


def test_predicts_the_measured_per_item_cost():
    """End-to-end sanity check against a measured out-of-memory configuration.

    Llama-3.1-8B, 868-item batch, ~16.3 GB of weights resident on the GPU. Peak allocated
    memory was 81.44 GB, i.e. ~0.075 GB/item actually needed; the 5-of-9 count budgets
    0.060 GB/item (kv/gpu=0.03 + q_mem=0.03). The activation half of that per-item cost
    is what this helper produces.
    """
    cfg = _llama_31_8b()
    # seq_len back-solved from a q_mem of 0.03 GB, reproducing the measured configuration.
    seq_len = 183
    # Pinned to factor=2.0, the factor of the measured configuration, so the recorded
    # numbers do not drift when the live KV_MEM_SAFETY_FACTOR changes.
    old_activation_gb = _partial_formula_gb(cfg, seq_len, factor=2.0)
    assert old_activation_gb == pytest.approx(0.03, abs=0.001)

    # Also pinned to factor=2.0 via the verified mirror (test_matches_complete_tensor_enumeration
    # checks it tracks the real _llm_activation_gb), so raising the live factor does not move it.
    activation_gb = _expected_gb(cfg, seq_len, factor=2.0)
    # kv/gpu is also 0.03 GB/item. A ratio-proxy total of 0.060 undershoots what the batch
    # needed (~0.075 GB/item); the modeled total must reach it.
    assert 0.03 + old_activation_gb == pytest.approx(0.060, abs=0.001)
    assert 0.03 + activation_gb == pytest.approx(0.075, abs=0.001)


# ── Guards ───────────────────────────────────────────────────────────────────


def test_scales_linearly_with_seq_len():
    cfg = _llama_31_8b()
    single = KVCachingBackendBase._llm_activation_gb(_Pipe(cfg), 100)
    double = KVCachingBackendBase._llm_activation_gb(_Pipe(cfg), 200)
    assert double == pytest.approx(single * 2)


def test_scales_linearly_with_safety_factor(monkeypatch):
    cfg = _llama_31_8b()
    baseline = KVCachingBackendBase._llm_activation_gb(_Pipe(cfg), 500)
    monkeypatch.setattr(kvb, "KV_MEM_SAFETY_FACTOR", KV_MEM_SAFETY_FACTOR * 2)
    assert KVCachingBackendBase._llm_activation_gb(_Pipe(cfg), 500) == pytest.approx(
        baseline * 2
    )


@pytest.mark.parametrize("seq_len", [0, -1])
def test_zero_for_non_positive_seq_len(seq_len):
    assert KVCachingBackendBase._llm_activation_gb(_Pipe(_llama_31_8b()), seq_len) == 0.0


def test_zero_for_missing_pipe():
    assert KVCachingBackendBase._llm_activation_gb(None, 500) == 0.0


def test_zero_for_unusable_config():
    """A config without the fields we need must degrade to 0.0, not raise."""
    assert KVCachingBackendBase._llm_activation_gb(
        _Pipe(_Config(hidden_size=4096)), 500
    ) == 0.0
