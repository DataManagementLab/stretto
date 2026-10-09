"""ViT activation memory in the image server's batch-size estimators.

`_get_max_batch_size` / `_get_max_batch_size_vanilla` (kv_cache_base.py) model KV-cache
residency and LLM forward-pass activations. For image models the vision tower's forward
pass also needs memory; omitting it over-sizes batches and runs out of GPU memory.

`extra_activation_gb` is an optional term on both shared base estimators, and
`KvImageQaModelWrapper._vit_activation_gb` is passed into all three image batch-size call
sites (generate, vanilla and join). These tests pin:

  - `_vit_activation_gb`'s formula in isolation.
  - `extra_activation_gb` actually shrinks the estimated batch in both shared base
    estimators (and defaults to a no-op, so text/audio callers that don't pass it are
    unaffected).
  - the image-join path accounts for ViT cost through the shared helper.
"""

import sys
import types

# kv_cache_image_qa_server.py imports concrete press classes from kvpress at module level.
# These tests only exercise the pure batch-size arithmetic and need none of them, so fall
# back to a stub when kvpress isn't installed rather than requiring the full CUDA/kvpress
# stack just to run these tests.
try:
    import kvpress  # noqa: F401

    kvpress.KeyRerotationPress
    kvpress.ExpectedAttentionPress
    kvpress.KVzipPress
    kvpress.FinchPress
except (ImportError, AttributeError):
    _stub = types.ModuleType("kvpress")
    for _name in ("KeyRerotationPress", "ExpectedAttentionPress", "KVzipPress", "FinchPress"):
        setattr(_stub, _name, object)
    sys.modules["kvpress"] = _stub

import pytest
import torch

from reasondb.backends import kv_cache_base as kvb
from reasondb.backends.kv_cache_base import KV_MEM_SAFETY_FACTOR, KVCachingBackendBase
from reasondb.backends.kv_cache_image_qa_server import KvImageQaModelWrapper


# ── Fakes ────────────────────────────────────────────────────────────────────


class _TextConfig:
    """Enough of a decoder LLM config for _llm_activation_gb / the q_mem term."""

    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim


class _VisionConfig:
    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        patch_size=14,
        intermediate_size=None,
        image_size=336,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.patch_size = patch_size
        if intermediate_size is not None:
            self.intermediate_size = intermediate_size
        self.image_size = image_size


class _MultimodalConfig:
    def __init__(self, text_config, vision_config, image_grid_pinpoints=None):
        self.text_config = text_config
        self.vision_config = vision_config
        if image_grid_pinpoints is not None:
            self.image_grid_pinpoints = image_grid_pinpoints


#: Patch count the stubbed processor probe reports, standing in for one worst-case
#: serving image. The real count comes from `_vit_num_patches`, which loads the model's
#: own image processor from the hub — these tests pin the arithmetic downstream of it, so
#: they stub it out rather than reach the network.
STUB_NUM_PATCHES = 2885


def _expected_vit_gb(
    hidden=1024, intermediate=None, num_patches=STUB_NUM_PATCHES, factor=None
):
    """Mirror of _vit_activation_gb, so the tests pin the formula independently."""
    intermediate = 4 * hidden if intermediate is None else intermediate
    factor = KV_MEM_SAFETY_FACTOR if factor is None else factor
    elements_per_token = 3 * hidden + 2 * hidden + 2 * intermediate + hidden
    return (factor * num_patches * elements_per_token * 2) / 1e9


class _Model:
    def __init__(self, config, num_params=1):
        self.config = config
        self._params = [types.SimpleNamespace(device=torch.device("cpu")) for _ in range(num_params)]

    def parameters(self):
        return iter(self._params)


class _Pipe:
    def __init__(self, model):
        self.model = model


def _text_pipe():
    return _Pipe(_Model(_TextConfig()))


def _image_pipe(vision_config=None, image_grid_pinpoints=None):
    return _Pipe(
        _Model(
            _MultimodalConfig(
                _TextConfig(),
                vision_config or _VisionConfig(),
                image_grid_pinpoints=image_grid_pinpoints,
            )
        )
    )


def _bare_base():
    return object.__new__(KVCachingBackendBase)


def _bare_image():
    return object.__new__(KvImageQaModelWrapper)


def _vit_wrapper(vision_config=None, num_patches=STUB_NUM_PATCHES):
    """A wrapper whose only live parts are `pipe` and a pinned patch count.

    The patch count is probed from the model's own image processor. Seeding the cache
    the probe fills keeps it off the network, and keeps these tests about the per-token
    arithmetic the probe feeds.
    """
    obj = _bare_image()
    obj.pipe = _image_pipe(vision_config)
    obj._vit_num_patches_cached = num_patches
    return obj


# ── _vit_activation_gb ───────────────────────────────────────────────────────


def test_vit_activation_gb_matches_formula():
    """CLIP ViT-L/14-336 shape: every per-layer tensor counted, once, over every patch."""
    obj = _vit_wrapper()
    assert obj._vit_activation_gb(obj.pipe) == pytest.approx(_expected_vit_gb())


def test_vit_activation_gb_positive():
    obj = _vit_wrapper()
    assert obj._vit_activation_gb(obj.pipe) > 0.0


def test_vit_activation_gb_counts_the_whole_layer_not_just_qkv():
    """The MLP block, the attention output and out_proj are counted, not just Q/K/V.

    A Q/K/V-only count is 3 * hidden. With the CLIP-L default MLP ratio of 4x, the complete
    tensor list is 14336 elements/token against 3072 — so a QKV-only count fails loudly here
    rather than by OOM.
    """
    obj = _vit_wrapper()
    qkv_only = (KV_MEM_SAFETY_FACTOR * STUB_NUM_PATCHES * 3 * 1024 * 2) / 1e9
    assert obj._vit_activation_gb(obj.pipe) == pytest.approx(
        qkv_only * (14336 / 3072)
    )


def test_vit_activation_gb_is_independent_of_layer_count():
    """One layer's activations, not num_hidden_layers of them (see the docstring rationale).

    Activations are freed layer by layer, so the layer count must not move the number.
    """
    shallow = _vit_wrapper(_VisionConfig(num_hidden_layers=2))
    deep = _vit_wrapper(_VisionConfig(num_hidden_layers=64))
    assert shallow._vit_activation_gb(shallow.pipe) == pytest.approx(
        deep._vit_activation_gb(deep.pipe)
    )


def test_vit_activation_gb_scales_with_patch_count():
    """The tower's cost is linear in the patches it forward-passes.

    The patch count itself is the probe's business (`_vit_num_patches` asks the model's
    own processor, so tiled and native-resolution towers are both covered there); what
    this pins is that the activation term tracks it proportionally rather than being
    computed from a CLIP-shaped constant.
    """
    one = _vit_wrapper(num_patches=577)
    five = _vit_wrapper(num_patches=577 * 5)
    assert one._vit_activation_gb(one.pipe) == pytest.approx(
        _expected_vit_gb(num_patches=577)
    )
    assert five._vit_activation_gb(five.pipe) == pytest.approx(
        one._vit_activation_gb(one.pipe) * 5
    )


#: One 336x336 CLIP tile at patch_size 14, plus the CLS token.
_PATCHES_PER_TILE = (336 // 14) ** 2 + 1


def test_vit_num_patches_falls_back_when_the_processor_probe_fails():
    """An unloadable image processor degrades to the CLIP-shaped estimate, never raises.

    Batch sizing must not hard-error on an exotic processor; `_bare_image()` has no
    `model_name`, so the probe raises and the fallback is what comes back. No
    image_grid_pinpoints on the config -> the documented LLaVA-NeXT worst case, 4 + base.
    """
    obj = _bare_image()
    obj.pipe = _image_pipe()
    assert obj._vit_num_patches() == 5 * _PATCHES_PER_TILE


def test_vit_num_patches_fallback_counts_every_tile():
    """The fallback is per *image*, like the probe it stands in for — not per tile.

    The probe returns every patch the processor batches through the tower, tiles
    included. A per-tile fallback would under-count anyres images by the tile factor,
    and an under-counted ViT term over-sizes the batch — the OOM direction.
    """
    one_tile = _bare_image()
    one_tile.pipe = _image_pipe(image_grid_pinpoints=[[336, 336]])  # 1 tile + base
    four_tile = _bare_image()
    four_tile.pipe = _image_pipe(image_grid_pinpoints=[[672, 672]])  # 4 tiles + base

    assert one_tile._vit_num_patches() == 2 * _PATCHES_PER_TILE
    assert four_tile._vit_num_patches() == 5 * _PATCHES_PER_TILE


def test_vit_activation_gb_falls_back_to_defaults_when_config_fields_missing():
    """An unrecognized vision backbone (missing config fields) must not crash the estimator."""
    obj = _bare_image()
    obj.pipe = _Pipe(_Model(_MultimodalConfig(_TextConfig(), types.SimpleNamespace())))
    obj._vit_num_patches_cached = STUB_NUM_PATCHES
    # defaults: hidden 1024, intermediate 4*hidden
    assert obj._vit_activation_gb(obj.pipe) == pytest.approx(_expected_vit_gb())


def test_vit_activation_gb_scales_linearly_with_safety_factor(monkeypatch):
    import reasondb.backends.kv_cache_image_qa_server as img_mod

    obj = _vit_wrapper()
    baseline = obj._vit_activation_gb(obj.pipe)
    monkeypatch.setattr(img_mod, "KV_MEM_SAFETY_FACTOR", KV_MEM_SAFETY_FACTOR * 2)
    doubled = obj._vit_activation_gb(obj.pipe)
    assert doubled == pytest.approx(baseline * 2)


# ── extra_activation_gb plumbing in the shared base estimators ─────────────


def test_get_max_batch_size_extra_activation_gb_shrinks_batch(monkeypatch, tmp_path):
    obj = _bare_base()
    obj.model_name = "fake-model"
    obj.pipe = _text_pipe()
    monkeypatch.setattr(obj, "_min_free_gb", lambda layer_devices: 100.0)
    monkeypatch.setattr(kvb, "get_biggest_file_size_gb", lambda folder, names: 0.01)

    kwargs = dict(
        column_name="c",
        batch_size=None,
        compression_ratio=0.5,
        cache_dir=str(tmp_path),  # no footprint YAML present -> file-sizing fallback
        file_paths=["fake.pt"],
        layer_devices=[torch.device("cpu")],
    )
    baseline = obj._get_max_batch_size(**kwargs)
    with_extra = obj._get_max_batch_size(**kwargs, extra_activation_gb=50.0)

    assert with_extra < baseline


def test_get_max_batch_size_extra_activation_gb_defaults_to_zero(monkeypatch, tmp_path):
    obj = _bare_base()
    obj.model_name = "fake-model"
    obj.pipe = _text_pipe()
    monkeypatch.setattr(obj, "_min_free_gb", lambda layer_devices: 100.0)
    monkeypatch.setattr(kvb, "get_biggest_file_size_gb", lambda folder, names: 0.01)

    kwargs = dict(
        column_name="c",
        batch_size=None,
        compression_ratio=0.5,
        cache_dir=str(tmp_path),
        file_paths=["fake.pt"],
        layer_devices=[torch.device("cpu")],
    )
    omitted = obj._get_max_batch_size(**kwargs)
    explicit_zero = obj._get_max_batch_size(**kwargs, extra_activation_gb=0.0)

    assert omitted == explicit_zero


def test_get_max_batch_size_vanilla_extra_activation_gb_shrinks_batch(monkeypatch):
    obj = _bare_base()
    obj.pipe = _text_pipe()
    monkeypatch.setattr(obj, "_min_free_gb", lambda layer_devices: 100.0)

    kwargs = dict(max_prompt_tokens=500, layer_devices=[torch.device("cpu")], batch_size=None)
    baseline = obj._get_max_batch_size_vanilla(**kwargs)
    with_extra = obj._get_max_batch_size_vanilla(**kwargs, extra_activation_gb=50.0)

    assert with_extra < baseline


def test_get_max_batch_size_vanilla_extra_activation_gb_defaults_to_zero(monkeypatch):
    obj = _bare_base()
    obj.pipe = _text_pipe()
    monkeypatch.setattr(obj, "_min_free_gb", lambda layer_devices: 100.0)

    kwargs = dict(max_prompt_tokens=500, layer_devices=[torch.device("cpu")], batch_size=None)
    omitted = obj._get_max_batch_size_vanilla(**kwargs)
    explicit_zero = obj._get_max_batch_size_vanilla(**kwargs, extra_activation_gb=0.0)

    assert omitted == explicit_zero


# ── image-join path: ViT-aware via the shared helper ───────────────────────


def test_image_join_batch_size_accounts_for_vit_activation(monkeypatch):
    """The join estimate must shrink relative to a (hypothetical) zero-ViT-cost
    estimate, i.e. the ViT term is included.
    """
    obj = _bare_image()
    # A large-enough vision tower that its activation cost clears the per-item floor every
    # cost is clamped to — otherwise both runs would floor to the same value and the
    # comparison below would be a false pass.
    obj.pipe = _image_pipe(_VisionConfig(hidden_size=4096, num_hidden_layers=48, patch_size=14))
    obj._vit_num_patches_cached = STUB_NUM_PATCHES
    monkeypatch.setattr(obj, "_min_free_gb", lambda layer_devices: 100.0)
    monkeypatch.setattr(
        "reasondb.memory_footprint.memory_report.get_biggest_file_size_gb",
        lambda folder, names: 0.01,
    )

    kwargs = dict(file_paths=["fake.pt"], max_right_tokens=10, max_context_tokens=100)

    with_vit = obj._get_max_batch_size_image_join(**kwargs)

    monkeypatch.setattr(obj, "_vit_activation_gb", lambda pipe: 0.0)
    without_vit = obj._get_max_batch_size_image_join(**kwargs)

    assert with_vit < without_vit


def test_image_join_uses_the_shared_vit_helper(monkeypatch):
    """The join path's vit_mem_gb term must come from the shared helper, not a local
    copy, so the two cannot drift apart.
    """
    obj = _vit_wrapper()
    monkeypatch.setattr(obj, "_min_free_gb", lambda layer_devices: 100.0)
    monkeypatch.setattr(
        "reasondb.memory_footprint.memory_report.get_biggest_file_size_gb",
        lambda folder, names: 0.01,
    )

    calls = []
    real_vit_activation_gb = KvImageQaModelWrapper._vit_activation_gb

    def spy(pipe):
        calls.append(pipe)
        return real_vit_activation_gb(obj, pipe)

    monkeypatch.setattr(obj, "_vit_activation_gb", spy)
    obj._get_max_batch_size_image_join(
        file_paths=["fake.pt"], max_right_tokens=10, max_context_tokens=100
    )

    assert calls == [obj.pipe]


def test_image_join_does_not_double_count_right_side_tokens(monkeypatch):
    """No q_mem_gb proxy term (kv_size * right/ctx) is added on top of llm_mem_gb.

    llm_mem_gb already covers the whole max_ctx+right_len forward pass, so a second term
    scaled off the KV-cache byte size would double-count. Varying
    max_right_tokens may only move the estimate through _llm_activation_gb — which is
    monkeypatched to a constant here, so the batch size must not change at all.
    """
    obj = _bare_image()
    obj.pipe = _image_pipe()
    monkeypatch.setattr(obj, "_min_free_gb", lambda layer_devices: 100.0)
    monkeypatch.setattr(
        "reasondb.memory_footprint.memory_report.get_biggest_file_size_gb",
        lambda folder, names: 0.01,
    )
    monkeypatch.setattr(obj, "_llm_activation_gb", lambda pipe, seq_len: 0.02)

    small_right = obj._get_max_batch_size_image_join(
        file_paths=["fake.pt"], max_right_tokens=10, max_context_tokens=100
    )
    large_right = obj._get_max_batch_size_image_join(
        file_paths=["fake.pt"], max_right_tokens=90, max_context_tokens=100
    )

    assert small_right == large_right
