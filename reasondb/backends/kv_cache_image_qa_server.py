import argparse
import json
import asyncio
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from flask import Flask, request
from flask_restful import Resource, Api
from typing import Any, Dict, List, Optional, Tuple

from reasondb.backends.kv_cache_base import (
    KVCachingBackendBase,
    KV_MEM_SAFETY_FACTOR,
    PINNED_KV_STORE,
    PinnedKVUnavailable,
    release_pinned_kv_response,
    remap_cache_dir_to_local_mirror,
    validate_kv_compression_ratios,
    _MIN_PER_ITEM_GB,
    _UNCOSTED_ACTIVATION_FALLBACK_GB,
    _usable_free_gb,
)
from reasondb.backends.kv_cache_multi_cr import compress_and_save_multi_cr, compress_and_save_relative
from reasondb.backends.kv_cache_reconstruct import (
    H2D_NON_BLOCKING,
    RelativeReconstructError,
    gather_payload_cpu,
    migrate_legacy_index_dirs,
    rerotate_payload_sharded,
    resolve_relative_source,
)
from reasondb.backends.mrope_rerotate import mrope_component_map
from reasondb.backends.inference_stats import build_inference_stats, gpu_snapshot
from reasondb.backends.vision_model import PORT_KV_VISION
from reasondb.config.model_registry import ModelRegistry as _ModelRegistry
from PIL import Image, ImageFile
import torch
from kvpress import ExpectedAttentionPress, KeyRerotationPress

from transformers import DynamicCache, pipeline  # type: ignore
import os
import contextlib

import base64
from io import BytesIO
from transformers import AutoProcessor

from reasondb.memory_footprint.memory_report import (
    compute_memory_footprints,
    update_compressed_cache_footprint,
)
import numpy as np


def _iter_cache_layers(cache):
    """Yield (key_tensor, value_tensor) per layer for old and new DynamicCache."""
    if hasattr(cache, "layers"):
        # transformers 5.x
        for layer in cache.layers:
            yield layer.keys, layer.values
    elif hasattr(cache, "_cache") and cache._cache and hasattr(cache._cache[0], "key_states"):
        # transformers 4.50.3
        for item in cache._cache:
            yield item.key_states, item.value_states
    else:
        yield from zip(cache.__dict__["key_cache"], cache.__dict__["value_cache"])


def _cache_kv(cache, layer_idx):
    """Return (keys, values) for one layer across transformers 5.x (.layers) and 4.x (.key_cache)."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


MODEL_NAME = "llava-hf/llava-next-72b-hf"
IMAGE_MAX_PIXELS = 1400 * 1400

_vl_registry = _ModelRegistry.get()

MODEL_TAG = {
    **{
        name: _vl_registry.spec_by_model_name(name, modality="vision").key
        for name in _vl_registry.all_model_names(modality="vision")
    },
    "llava-hf/llava-next-72b-hf": "llava-70B",
    "llava-hf/llama3-llava-next-8b-hf": "llava-8B",
}
PRESS = {
    "expected_attention": lambda compression_ratio: ExpectedAttentionPress(
        compression_ratio=compression_ratio
    ),
}
# Enable loading of truncated images
# Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)


class KvImageQaModelWrapper(KVCachingBackendBase):
    def __init__(
        self,
        model_name,
        device_id,
        compression_ratios=(0.0, 0.3, 0.5, 0.6, 0.8, 0.9, 0.99),
        batch_sizes=None,
        press_name="expected_attention",
        vanilla_batch_size=None,
        use_relative_indices=False,
    ):
        self.device_id = device_id
        self.compression_ratios = compression_ratios
        batch_sizes = batch_sizes or (None,) * len(compression_ratios)
        self.compression_ratio_to_batch_size = {
            cr: bs for cr, bs in zip(compression_ratios, batch_sizes)
        }
        self.model_name = model_name
        self.press_name = press_name
        # None/0 → auto-estimate the vanilla batch on the shared memory budget; >0 → force.
        self.vanilla_batch_size = vanilla_batch_size
        # Path A: reconstruct a target comp{tag} cache on the fly from a baseline cache +
        # relative indices instead of loading a physical comp{tag} cache. Default off.
        self.use_relative_indices = use_relative_indices

        self.model_tag = MODEL_TAG.get(model_name, "unknown_model")
        
        _spec = _vl_registry.spec_by_model_name(model_name, modality="vision")
        self.is_large_model = (
            _spec.size_class == "large" if _spec is not None else "72b" in model_name
        )

        # Will be initialized in init()
        self.init()

    def compute_image_qa_response(
        self,
        column_name: str,
        image_paths: List[str],
        question: str,
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
        materialized_compression_ratio: float,
        vanilla: bool,
        keep_in_memory: bool = False,
    ) -> Dict[str, Dict]:
        self._validate_client_crs(
            compression_ratio, materialized_compression_ratio, vanilla, keep_in_memory
        )
        _t0 = time.perf_counter()
        responses, log_odds, load_stats = asyncio.run(
            self._run_kv_cache_multimodal(
                column_name=column_name,
                image_paths=image_paths,
                question=question,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
                materialized_compression_ratio=materialized_compression_ratio,
                vanilla=vanilla,
                keep_in_memory=keep_in_memory,
            )
        )
        # ``answers``/``log_odds`` stay path-keyed dicts here (unlike the text server's
        # lists); ``stats`` is a sibling key, so the shape difference never leaks into
        # the statistics contract.
        return {
            "answers": responses,
            "log_odds": log_odds,
            "stats": self._request_stats(
                path="vanilla" if vanilla else "kv",
                n_items=len(image_paths),
                elapsed_s=time.perf_counter() - _t0,
                effective_compression_ratio=compression_ratio,
                materialized_compression_ratio=materialized_compression_ratio,
                vanilla=vanilla,
                keep_in_memory=keep_in_memory,
                **load_stats,
            ),
        }

    def _request_stats(
        self,
        *,
        path: str,
        n_items: int,
        elapsed_s: float,
        effective_compression_ratio=None,
        materialized_compression_ratio=None,
        vanilla: bool = False,
        n_errors: int = 0,
        cache_load_s: Optional[float] = None,
        cache_route_s: Optional[float] = None,
        cache_wait_s: Optional[float] = None,
        keep_in_memory: bool = False,
        pinned_hits: Optional[int] = None,
        pinned_misses: Optional[int] = None,
        pinned_gb: Optional[float] = None,
        dynamic_batch_size: Optional[int] = None,
        peaks: Optional[Dict[int, float]] = None,
    ) -> dict:
        """Request-level statistics attached to every response of this server.

        The per-batch debug logs stay where they are; this reports the
        request as a whole plus one GPU probe, which is what a dashboard can actually
        plot without turning every batch into an event.

        ``cache_load_s``/``cache_route_s``/``cache_wait_s``/``pinned_*``/``peaks`` are
        the reconstruction-path telemetry ``_run_kv_cache_multimodal`` already measures
        internally; callers forward it here instead of it being computed twice.
        ``dynamic_batch_size`` is the batch size ``_get_max_batch_size`` actually picked
        for this request, which is what's meaningful for the relative-index path — it
        overrides the static per-ratio ``compression_ratio_to_batch_size`` lookup (which
        only applies to the physical, non-indexed path) when the caller has it.
        """
        gpu, min_free = gpu_snapshot(peaks)
        batch_size = (
            dynamic_batch_size
            if dynamic_batch_size is not None
            else self.compression_ratio_to_batch_size.get(effective_compression_ratio)
        )
        return build_inference_stats(
            server="kv_image_qa",
            path=path,
            model_name=self.model_name,
            n_items=n_items,
            batch_size=batch_size,
            n_batches=(n_items + batch_size - 1) // batch_size if batch_size else None,
            effective_compression_ratio=effective_compression_ratio,
            materialized_compression_ratio=materialized_compression_ratio,
            vanilla=vanilla,
            server_elapsed_s=elapsed_s,
            cache_load_s=cache_load_s,
            cache_route_s=cache_route_s,
            cache_wait_s=cache_wait_s,
            keep_in_memory=keep_in_memory,
            n_caches=n_items,
            gpu=gpu,
            min_free_gb=min_free,
            pinned_hits=pinned_hits,
            pinned_misses=pinned_misses,
            pinned_gb=pinned_gb,
            n_errors=n_errors,
        )

    def _validate_client_crs(
        self,
        effective_compression_ratio: float,
        materialized_compression_ratio: float,
        vanilla: bool,
        keep_in_memory: bool = False,
    ) -> None:
        """Validate the client-supplied compression ratios against what this server serves."""
        validate_kv_compression_ratios(
            effective_compression_ratio,
            materialized_compression_ratio,
            vanilla,
            keep_in_memory,
        )
        if not vanilla:
            assert effective_compression_ratio in self.compression_ratios, (
                f"effective compression ratio {effective_compression_ratio} not in "
                f"supported ratios: {self.compression_ratios}"
            )
            assert materialized_compression_ratio in self.compression_ratios, (
                f"materialized compression ratio {materialized_compression_ratio} not in "
                f"supported ratios: {self.compression_ratios}"
            )

    def hash_path(self, path: str) -> str:
        """Generate a sha256hash for a given path."""
        import hashlib

        return hashlib.sha256(path.encode()).hexdigest()

    def _reset_mrope_state(self):
        """Clear stale per-batch M-RoPE state before a generate() call.

        Qwen3-VL caches ``rope_deltas`` (one per batch row, derived from each
        image's grid) on the inner model after any forward that saw images.
        A later generate() without pixel inputs cannot recompute them and
        falls back to the cached ones: a batch-size mismatch crashes
        position-id preparation, and a coincidental match silently applies the
        *previous* images' position offsets. Our canonicalized caches use plain
        sequential text positions, so the correct state for cache-fed or
        text-only generation is no deltas at all; paths that do pass pixels
        recompute fresh deltas when the cached value is None.
        """
        if self.mrope_component_map is None:
            return
        inner = getattr(self.pipe.model, "model", None)
        if inner is not None and getattr(inner, "rope_deltas", None) is not None:
            inner.rope_deltas = None

    async def prepare_caches(
        self,
        column_name: str,
        image_paths: List[str],
        cache_dir: str,
        compression_ratio: float,
        materialized_compression_ratio: float,
        vanilla: bool,
        keep_in_memory: bool = False,
    ):
        # Vanilla mode uses no pre-computed KV cache, so there is nothing to prepare.
        if vanilla:
            logger.info(
                f"Skipping cache preparation for vanilla mode (cr={compression_ratio})"
            )
            return {"n_images": 0, "n_missing": 0, "missing_hashes": []}

        # Fail before any filesystem work: a server with no pin budget can never serve
        # this operator (see the text server for the rationale).
        if keep_in_memory:
            PINNED_KV_STORE.require_configured(
                f"column {column_name!r} on {self.model_name} (cr={compression_ratio})"
            )

        # In relative-indices mode `compression_ratio` (the effective ratio) names the
        # index target dir and the physical baseline lives at
        # materialized_compression_ratio; in physical mode materialized == effective.
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"

        # Relative-index mode: the compressed target is reconstructed on the fly at serve time
        # from an offline baseline + relative indices (generate_kv_caches_image_indices.py), so
        # prepare must NOT prefill/generate physical caches. Verify the artifacts exist and
        # report back which are missing — the caller (PrepareCaches) surfaces this to the
        # client, which crashes loudly at setup instead of silently degrading to "Not sure" at
        # serve time. Layout is press-aware and matches the serving path:
        # {cache_dir}/{model}/{press}/comp{base}/indices/comp{tag} (any legacy flat
        # {press}/indices/comp{tag} is relocated into that layout at startup).
        if self.use_relative_indices:
            press_dir = f"{cache_dir}/{self.model_name}/{self.press_name}"
            target_tag = f"comp{self.to_compression_tag(compression_ratio)}"
            base_tag = f"comp{self.to_compression_tag(materialized_compression_ratio)}"

            # Index dirs live nested under their materialized baseline
            # ({press}/{base_tag}/indices/{target_tag}); relocate any legacy flat dirs
            # ({press}/indices/{target_tag}) into that layout once, so every reader below
            # only ever looks at the nested layout.
            migrate_legacy_index_dirs(press_dir)

            # An image can be absent for two different reasons: (a) nobody has run the
            # offline generator for it yet — a setup mistake the caller should crash on —
            # or (b) the offline generator (prepare_indices_relative / physical
            # prepare_caches) DID run and recorded a per-image failure in ERRORS.json (e.g.
            # a corrupted image). (b) is a known, already-tolerated data issue, not a setup
            # mistake, so it must be reported as n_generation_errors, not n_missing —
            # mirroring physical mode below. Load both possible ERRORS.json locations: the
            # relative baseline dir (resolved via the target's _meta.json) and the target
            # dir itself (physical generation).
            known_errors = self._load_known_errors(press_dir, target_tag, base_tag)

            cache_basenames = []
            missing_hashes = []
            generation_error_hashes = []
            for image_path in image_paths:
                hash_name = self.hash_path(image_path)
                cache_basenames.append(f"cache_entry_{hash_name}.pt")
                try:
                    relative = resolve_relative_source(
                        press_dir, target_tag, hash_name, base_tag
                    )
                except RelativeReconstructError as e:
                    logger.warning(f"[relative] _meta unusable for {hash_name[:12]}…: {e}")
                    relative = None
                physical = f"{press_dir}/{target_tag}/cache_entry_{hash_name}.pt"
                if relative is None and not os.path.exists(physical):
                    if hash_name in known_errors:
                        generation_error_hashes.append(hash_name)
                    else:
                        missing_hashes.append(hash_name)
            n_ready = len(image_paths) - len(missing_hashes) - len(generation_error_hashes)
            logger.info(
                f"[relative] CR {compression_ratio}: {n_ready}/{len(image_paths)} images have "
                f"indices or physical caches under {press_dir}; no physical generation performed."
            )
            if generation_error_hashes:
                logger.warning(
                    f"[relative] CR {compression_ratio}: {len(generation_error_hashes)}/"
                    f"{len(image_paths)} images have a known generation error recorded "
                    f"during offline generation and will be skipped at serve time (sample: "
                    f"{generation_error_hashes[:20]})"
                )
            # Refresh the footprint YAML (best-effort): process_dataset measures the
            # physical baseline dirs and sizes the index-backed target CRs proportionally
            # from the baseline named in their _meta.json. Without this the batch-size
            # estimator finds no entry for the effective CR and falls back to stat-ing the
            # baseline cache files, which is both slower and less accurate. Mirrors the
            # text server's relative branch.
            try:
                compute_memory_footprints(
                    cache_dir, column_name, cache_basenames, model_name=self.model_name
                )
            except Exception as e:
                logger.warning(
                    f"[relative] footprint accounting failed (non-fatal): {e}"
                )
            response = {
                "n_images": len(image_paths),
                "n_missing": len(missing_hashes),
                "missing_hashes": missing_hashes[:20],
                "n_generation_errors": len(generation_error_hashes),
                "generation_error_hashes": generation_error_hashes[:20],
            }
            if keep_in_memory:
                # Only a physical cache can be pinned; a relatively-indexed one is
                # reconstructed per query and the client-side validator forbids that
                # combination, so anything reaching here resolved to a physical file.
                unusable = set(missing_hashes) | set(generation_error_hashes)
                response.update(
                    self._pin_prepared_caches(
                        [
                            f"{press_dir}/{target_tag}/cache_entry_{h}.pt"
                            for h in (self.hash_path(p) for p in image_paths)
                            if h not in unusable
                        ],
                        column_name=column_name,
                        compression_ratio=compression_ratio,
                    )
                )
            return response

        # Check if caches exist and generate if not, one by one. The press-aware layout
        # ({model}/{press}/comp{tag}) MUST match the serving path (_run_kv_cache_multimodal)
        # and the generation scripts, or prepare misses existing caches and regenerates them
        # into a directory serving ignores.
        save_dir = (
            f"{cache_dir}/{self.model_name}/{self.press_name}"
            f"/comp{self.to_compression_tag(compression_ratio)}"
        )
        os.makedirs(save_dir, exist_ok=True)

        errors = {}
        cache_filenames = []

        for i, image_path in tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc=f"Preparing caches for CR {compression_ratio}",
        ):
            hash_name = self.hash_path(image_path)
            cache_filename = f"{save_dir}/cache_entry_{hash_name}.pt"
            cache_filenames.append(cache_filename.split("/")[-1])

            if os.path.exists(cache_filename):
                continue

            try:
                image = self.deserialize_image(image_path)
                await self._generate_cache_for_image(
                    image, cache_filename, compression_ratio
                )
            except Exception as e:
                logger.warning(
                    f"Error processing image {i} {image_path} with hash {hash_name}: {str(e)}",
                )
                errors[cache_filename] = str(e)
        compute_memory_footprints(cache_dir, column_name, cache_filenames, model_name=self.model_name)

        with open(f"{save_dir}/ERRORS.json", "w") as f:
            json.dump(errors, f, indent=4)

        # Per-image generation failures (e.g. a corrupted image file) are an expected,
        # already-tolerated data issue, NOT a "forgot to pregenerate" setup mistake: the
        # serve path skips these per-image using the recorded error and answers "Not sure"
        # for just that image, while the rest of the batch proceeds normally. So this must
        # NOT surface as "n_missing" (the client only crashes on that key) — report it under
        # a distinct, non-fatal key purely for visibility.
        response = {
            "n_images": len(image_paths),
            "n_generation_errors": len(errors),
            "generation_error_hashes": [os.path.basename(p) for p in list(errors)[:20]],
        }
        if keep_in_memory:
            # An image whose generation just failed, or failed on an earlier run, has no
            # file — `errors` and the existence check both exclude it, so the serve path
            # skips it exactly as it does for a disk-served operator.
            response.update(
                self._pin_prepared_caches(
                    [
                        path
                        for path in (
                            f"{save_dir}/cache_entry_{self.hash_path(p)}.pt"
                            for p in image_paths
                        )
                        if path not in errors and os.path.exists(path)
                    ],
                    column_name=column_name,
                    compression_ratio=compression_ratio,
                )
            )
        return response

    def to_compression_tag(self, compression_ratio: float) -> str:
        """Convert compression ratio to a string tag for directory naming."""
        return (
            str(compression_ratio).replace(".", "_")
            if compression_ratio != 0.0
            else "0"
        )

    @staticmethod
    def _comp_dir_to_cr(comp_dir: str) -> Optional[float]:
        """Inverse of to_compression_tag for a ``comp{tag}`` directory name.

        ``comp0_9`` → 0.9, ``comp0`` → 0.0. Returns None if it can't be parsed (the caller
        then falls back to the conservative baseline size for batch estimation).
        """
        name = os.path.basename(comp_dir.rstrip("/"))
        if not name.startswith("comp"):
            return None
        try:
            return float(name[len("comp"):].replace("_", "."))
        except ValueError:
            return None

    def init(self):
        """Initialize the llava-next model pipeline and compression settings."""
        logger.info("Setting up KV Cache Filter with llava-next model...")

        # Set up device
        self.device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"

        # Set PIL image size limit
        Image.MAX_IMAGE_PIXELS = 250000000

        # Initialize the pipeline
        args = [{"attn_implementation": "flash_attention_2"}, {}]
        self.pipe = None
        for x in args:
            try:
                self.pipe = pipeline(
                    "kv-press-text-generation",  # type: ignore
                    model=self.model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    model_kwargs=x,  # type: ignore
                )
                break
            except Exception as e:
                logger.warning(
                    f"Error initializing model with args {x}: {str(e)}", exc_info=True
                )
        assert self.pipe is not None, "Failed to initialize the model pipeline."
        self.pipe.model.eval()

        # Interleaved M-RoPE detection (Qwen3-VL): cached keys carry per-dim t/h/w
        # phases, so compression must use the M-RoPE-aware rerotation and even the
        # CR=0 cache must be canonicalized (see mrope_rerotate.py). None for
        # standard-RoPE VL models (LLaVA, Mistral3).
        text_cfg = getattr(self.pipe.model.config, "text_config", self.pipe.model.config)
        rope_cfg = (
            getattr(text_cfg, "rope_scaling", None)
            or getattr(text_cfg, "rope_parameters", None)
            or {}
        )
        mrope_section = rope_cfg.get("mrope_section") if isinstance(rope_cfg, dict) else None
        self.mrope_component_map = None
        if mrope_section is not None:
            assert rope_cfg.get("mrope_interleaved", False), (
                "Only interleaved M-RoPE (Qwen3-VL style) is supported; chunked M-RoPE "
                "(Qwen2-VL/Qwen2.5-VL) needs a different frequency-component map."
            )
            head_dim = getattr(
                text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads
            )
            self.mrope_component_map = mrope_component_map(mrope_section, head_dim // 2)
            logger.info(
                f"Interleaved M-RoPE model detected (mrope_section={mrope_section}); "
                "using M-RoPE-aware key rerotation for cache generation."
            )

        # Set up compression press
        if self.press_name not in PRESS:
            raise ValueError(
                f"Unknown press_name '{self.press_name}'. Available options: {list(PRESS.keys())}"
            )
        self.presses = {
            cr: KeyRerotationPress(PRESS[self.press_name](compression_ratio=cr))
            for cr in self.compression_ratios
        }

        logger.info(
            f"Using press {self.press_name} with compression_ratios={self.compression_ratios}",
        )
        logger.info(
            f"Model {self.model_name} loaded on {next(self.pipe.model.parameters()).device}",
        )

    def _get_decoder(self):
        """Return the LLM decoder (LlamaModel) exposing `.layers` and `.rotary_emb`.

        Old transformers: model.language_model is the decoder.
        New transformers: model.model.language_model is the decoder.
        """
        model = self.pipe.model
        if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
            return model.language_model
        return model.model.language_model

    def _get_llm_layers(self):
        """Return the transformer layers of the LLM part, compatible with old and new transformers APIs."""
        return self._get_decoder().layers

    def _mrope_info_for_prefill(self, inputs, device):
        """M-RoPE payload for ONE image prefill, or None for standard-RoPE models.

        Asks the VL model (e.g. Qwen3VLModel.get_rope_index) for the (3, S) t/h/w
        position ids the prefill will use, and bundles them with the decoder's
        inv_freq + the frequency-component map for the M-RoPE-aware rerotation in
        kv_cache_multi_cr (see mrope_rerotate.py).
        """
        if self.mrope_component_map is None:
            return None
        vl_model = self.pipe.model.model  # Qwen3VLModel exposes get_rope_index
        grid = inputs.get("image_grid_thw")
        attn = inputs.get("attention_mask")
        position_ids, _ = vl_model.get_rope_index(
            input_ids=inputs["context_ids"].to(device),
            image_grid_thw=grid.to(device) if grid is not None else None,
            attention_mask=attn.to(device) if attn is not None else None,
        )
        return {
            "pos_3d": position_ids[:, 0, :],  # (3, B, S) → (3, S)
            "inv_freq": self._get_decoder().rotary_emb.inv_freq,
            "component_map": self.mrope_component_map,
        }

    async def prepare_caches_multi_cr(
        self,
        column_name: str,
        image_paths: List[str],
        cr_to_cache_dir: dict,
        save_indices: bool = False,
    ):
        """Like prepare_caches, but generates ALL compression ratios in a SINGLE prefill
        per image — mirroring the text server's multi-CR path.

        Writes the SAME directory layout as text:
        {cache_dir}/{model}/{press}/comp{tag}/cache_entry_{hash}.pt, plus (when
        save_indices=True) {cache_dir}/{model}/{press}/indices/comp{tag}/idx_{hash}.pt.
        Because the indices come from the same prefill as the caches, masked
        reconstruction is bit-exact.
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        compression_ratios = list(cr_to_cache_dir.keys())
        for cr in compression_ratios:
            assert cr in self.compression_ratios, (
                f"Compression ratio {cr} not in supported ratios: {self.compression_ratios}"
            )

        save_dirs: dict = {}
        for cr, cache_dir in cr_to_cache_dir.items():
            save_dir = (
                f"{cache_dir}/{self.model_name}/{self.press_name}"
                f"/comp{self.to_compression_tag(cr)}"
            )
            os.makedirs(save_dir, exist_ok=True)
            save_dirs[cr] = save_dir

        errors: dict = {}
        for i, image_path in tqdm(
            enumerate(image_paths), total=len(image_paths), desc="Preparing multi-CR image caches"
        ):
            hash_name = self.hash_path(image_path)

            cache_filenames_by_cr: dict = {}
            missing_crs: list = []
            missing_idx = False
            for cr in compression_ratios:
                cache_filename = f"{save_dirs[cr]}/cache_entry_{hash_name}.pt"
                cache_filenames_by_cr[cr] = cache_filename
                if not os.path.exists(cache_filename):
                    missing_crs.append(cr)
                elif save_indices and cr > 0.0:
                    idx_path = os.path.join(
                        os.path.dirname(save_dirs[cr]), "indices",
                        os.path.basename(save_dirs[cr]), f"idx_{hash_name}.pt",
                    )
                    if not os.path.exists(idx_path):
                        missing_idx = True

            if not missing_crs and not missing_idx:
                continue
            # With save_indices, regenerate ALL CRs so cache+indices share one prefill.
            crs_to_generate = compression_ratios if save_indices else missing_crs

            try:
                image = self.deserialize_image(image_path)
                await self._generate_caches_for_image_multi_cr(
                    image=image,
                    cache_filenames_by_cr={cr: cache_filenames_by_cr[cr] for cr in crs_to_generate},
                    compression_ratios=crs_to_generate,
                    save_indices=save_indices,
                )
            except Exception as e:
                logger.warning(
                    f"Error processing image {i} {image_path} (hash={hash_name}): {e}", exc_info=True
                )
                errors[hash_name] = str(e)

        # Serving (_run_kv_cache_multimodal) requires {comp_dir}/ERRORS.json — as written
        # by prepare_caches — to know which images may lack a cache. Write it per CR dir
        # even when empty or when every cache already existed, so offline multi-CR
        # generation is sufficient to serve from without a separate prepare pass.
        for cr in compression_ratios:
            with open(f"{save_dirs[cr]}/ERRORS.json", "w") as f:
                json.dump(errors, f, indent=4)

    async def _generate_caches_for_image_multi_cr(
        self, image, cache_filenames_by_cr, compression_ratios, save_indices=False
    ):
        """ONE multimodal prefill for an image → full cache + scores → per-CR compress+save."""
        assert self.pipe is not None
        uses_rerotation = self.press_name == "expected_attention"

        # ---- 1. Preprocess (image context) ----
        inputs = self.pipe.preprocess(
            context=" ", questions=[""], answer_prefix="Answer: ",
            max_context_length=128000, image=image,
        )

        # ---- 2. Score-capturing hook (unwrap KeyRerotationPress → inner scoring press) ----
        decoder = self._get_decoder()
        layers = decoder.layers
        has_nonzero = any(cr > 0.0 for cr in compression_ratios)
        scoring_press = None
        if has_nonzero:
            any_nonzero_cr = next(cr for cr in self.compression_ratios if cr > 0.0)
            scoring_press = (
                self.presses[any_nonzero_cr].press if uses_rerotation else self.presses[any_nonzero_cr]
            )

        layer_scores: dict = {}
        layer_modules: dict = {}

        def capturing_forward_hook(module, _input, kwargs, output):
            if scoring_press is None:
                return output
            hidden_states = kwargs["hidden_states"]
            cache = kwargs.get("past_key_value") or kwargs.get("past_key_values")
            q_len = hidden_states.shape[1]
            if kwargs["cache_position"][-1] > q_len:  # only during prefill
                return output
            keys, values = _cache_kv(cache, module.layer_idx)
            scores = scoring_press.score(module, hidden_states, keys, values, output[1], kwargs)
            layer_scores[module.layer_idx] = scores.detach().cpu()
            layer_modules[module.layer_idx] = module
            return output

        # ---- 3. Single multimodal prefill (no compression → full cache; no decoding) ----
        first_device = next(self.pipe.model.parameters()).device
        for layer in layers:
            layer.self_attn.rotary_emb = decoder.rotary_emb
        hooks = [
            layer.self_attn.register_forward_hook(capturing_forward_hook, with_kwargs=True)
            for layer in layers
        ]

        full_cache = DynamicCache()
        prefill_kwargs = {
            "input_ids": inputs["context_ids"].to(first_device),
            "past_key_values": full_cache,
            "use_cache": True,
        }
        for key in ("pixel_values", "image_sizes", "image_grid_thw", "attention_mask"):
            val = inputs.get(key)
            if val is not None:
                prefill_kwargs[key] = val.to(first_device)

        # M-RoPE models: the (3, S) t/h/w positions this prefill rotates keys with.
        mrope_info = self._mrope_info_for_prefill(inputs, first_device)

        try:
            with torch.inference_mode():
                self.pipe.model(**prefill_kwargs)  # full vision+language forward
        finally:
            for h in hooks:
                h.remove()

        # ---- 4. Per-CR compression + save (+ indices), shared with text server ----
        compress_and_save_multi_cr(
            full_cache=full_cache,
            layer_scores=layer_scores,
            layer_modules=layer_modules,
            compression_ratios=compression_ratios,
            cache_filenames_by_cr=cache_filenames_by_cr,
            first_device=first_device,
            save_indices=save_indices,
            is_finch=False,
            uses_rerotation=uses_rerotation,
            window_size=0,
            context_tokens_count=0,
            mrope_info=mrope_info,
        )
        del full_cache
        torch.cuda.empty_cache()

    async def _prefill_and_score_image(self, image):
        """Run ONE multimodal prefill (expected_attention) and capture the full cache +
        per-layer scores. Image analogue of the text server's `_prefill_and_score`; used
        by the relative (hierarchical) index generation. Returns
        (full_cache, layer_scores, layer_modules, first_device, mrope_info) — mrope_info
        is None for standard-RoPE models (see _mrope_info_for_prefill).
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.press_name == "expected_attention", (
            "Relative index generation supports only expected_attention."
        )
        inputs = self.pipe.preprocess(
            context=" ", questions=[""], answer_prefix="Answer: ",
            max_context_length=128000, image=image,
        )
        decoder = self._get_decoder()
        layers = decoder.layers
        any_nonzero_cr = next(cr for cr in self.compression_ratios if cr > 0.0)
        scoring_press = self.presses[any_nonzero_cr].press  # inner ScorerPress of the rerotation wrapper

        layer_scores: dict = {}
        layer_modules: dict = {}

        def capturing_forward_hook(module, _input, kwargs, output):
            hidden_states = kwargs["hidden_states"]
            cache = kwargs.get("past_key_value") or kwargs.get("past_key_values")
            q_len = hidden_states.shape[1]
            if kwargs["cache_position"][-1] > q_len:  # prefill only
                return output
            keys, values = _cache_kv(cache, module.layer_idx)
            scores = scoring_press.score(module, hidden_states, keys, values, output[1], kwargs)
            layer_scores[module.layer_idx] = scores.detach().cpu()
            layer_modules[module.layer_idx] = module
            return output

        first_device = next(self.pipe.model.parameters()).device
        for layer in layers:
            layer.self_attn.rotary_emb = decoder.rotary_emb
        hooks = [
            layer.self_attn.register_forward_hook(capturing_forward_hook, with_kwargs=True)
            for layer in layers
        ]
        full_cache = DynamicCache()
        prefill_kwargs = {
            "input_ids": inputs["context_ids"].to(first_device),
            "past_key_values": full_cache,
            "use_cache": True,
        }
        for key in ("pixel_values", "image_sizes", "image_grid_thw", "attention_mask"):
            val = inputs.get(key)
            if val is not None:
                prefill_kwargs[key] = val.to(first_device)
        # M-RoPE models: the (3, S) t/h/w positions this prefill rotates keys with.
        mrope_info = self._mrope_info_for_prefill(inputs, first_device)
        try:
            with torch.inference_mode():
                self.pipe.model(**prefill_kwargs)
        finally:
            for h in hooks:
                h.remove()
        return full_cache, layer_scores, layer_modules, first_device, mrope_info

    async def _generate_relative_for_image(
        self, image, base_cache_filename, rel_idx_filenames_by_cr, base_cr, index_crs
    ):
        """One prefill → save the physical baseline cache + relative indices for each target."""
        full_cache, layer_scores, layer_modules, first_device, mrope_info = (
            await self._prefill_and_score_image(image)
        )
        try:
            compress_and_save_relative(
                full_cache=full_cache,
                layer_scores=layer_scores,
                layer_modules=layer_modules,
                base_cr=base_cr,
                index_crs=index_crs,
                base_cache_filename=base_cache_filename,
                rel_idx_filenames_by_cr=rel_idx_filenames_by_cr,
                first_device=first_device,
                uses_rerotation=True,  # expected_attention
                mrope_info=mrope_info,
            )
        finally:
            del full_cache
            torch.cuda.empty_cache()

    async def prepare_indices_relative(
        self,
        column_name: str,
        image_paths: List[str],
        cache_dir: str,
        base_cr: float,
        index_crs: list,
    ):
        """Hierarchical generation for IMAGES: store ONE physical baseline cache at `base_cr`
        and, for each target in `index_crs` (strictly more compressed), only indices RELATIVE
        to the baseline. Mirrors the text server's `prepare_indices_relative`.

        Layout (under {cache_dir}/{model}/{press}/):
          comp{base_tag}/cache_entry_{hash}.pt              — physical baseline cache
          comp{base_tag}/indices/comp{tgt_tag}/idx_{hash}.pt — indices into the baseline
          comp{base_tag}/indices/comp{tgt_tag}/_meta.json    — {"from": "comp{base_tag}"}
        Nesting the index dirs under their baseline lets several baselines index the same
        effective target at once (any legacy flat indices/comp{tgt_tag}/ is migrated into
        this layout at startup by migrate_legacy_index_dirs).
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.press_name == "expected_attention", (
            "prepare_indices_relative supports only expected_attention."
        )
        for cr in (base_cr, *index_crs):
            assert cr in self.compression_ratios, (
                f"Compression ratio {cr} not in supported ratios: {self.compression_ratios}"
            )
        for cr in index_crs:
            assert cr > base_cr, (
                f"index method cr {cr} must be strictly more compressed than baseline cr {base_cr}"
            )

        press_dir = f"{cache_dir}/{self.model_name}/{self.press_name}"
        base_dir = f"{press_dir}/comp{self.to_compression_tag(base_cr)}"
        # Index dirs nest under their materialized baseline so several baselines can index
        # the same effective target without colliding: {base_dir}/indices/comp{tgt_tag}.
        idx_dirs = {
            cr: f"{base_dir}/indices/comp{self.to_compression_tag(cr)}" for cr in index_crs
        }
        os.makedirs(base_dir, exist_ok=True)
        for d in idx_dirs.values():
            os.makedirs(d, exist_ok=True)

        errors: dict = {}
        base_cache_basenames: list[str] = []
        for i, image_path in tqdm(
            enumerate(image_paths), total=len(image_paths), desc="Preparing relative image indices"
        ):
            hash_name = self.hash_path(image_path)
            base_cache_filename = f"{base_dir}/cache_entry_{hash_name}.pt"
            base_cache_basenames.append(f"cache_entry_{hash_name}.pt")
            rel_idx_filenames_by_cr = {
                cr: f"{idx_dirs[cr]}/idx_{hash_name}.pt" for cr in index_crs
            }

            missing = not os.path.exists(base_cache_filename) or any(
                not os.path.exists(p) for p in rel_idx_filenames_by_cr.values()
            )
            if not missing:
                continue

            try:
                image = self.deserialize_image(image_path)
                await self._generate_relative_for_image(
                    image=image,
                    base_cache_filename=base_cache_filename,
                    rel_idx_filenames_by_cr=rel_idx_filenames_by_cr,
                    base_cr=base_cr,
                    index_crs=index_crs,
                )
            except Exception as e:
                logger.warning(
                    f"Error processing image {i} {image_path} (hash={hash_name}): {e}", exc_info=True
                )
                errors[hash_name] = str(e)

        with open(f"{base_dir}/ERRORS.json", "w") as f:
            json.dump(errors, f, indent=4)

        # Footprint accounting for the single physical baseline cache (best-effort).
        # Mirrors the text server's prepare_indices_relative: compute_memory_footprints
        # records the per-item baseline size and estimates every index-backed target CR
        # from it, so the serving batch-size estimator has a YAML entry to read.
        try:
            compute_memory_footprints(
                cache_dir, column_name, base_cache_basenames, model_name=self.model_name
            )
            update_compressed_cache_footprint(
                cache_path=cache_dir,
                compression_ratio=base_cr,
                cache_filenames=base_cache_basenames,
                model_name=self.model_name,
                column_name=column_name,
                press_name=self.press_name,
            )
        except Exception as e:
            logger.warning(f"Footprint accounting failed (non-fatal): {e}")

    def _vit_num_patches(self) -> int:
        """Patch tokens the vision tower forward-passes for ONE worst-case serving image.

        Probed from the model's OWN image processor with a dummy image at the serving
        cap (IMAGE_MAX_PIXELS — the deserialize_image bound, the largest image any
        serving path can feed the tower). The processor implements each family's
        resize/tile/merge logic, so the count tracks reality with no per-family
        knowledge here: e.g. LLaVA-NeXT batches ~5 tiles through the tower per image
        (~2.9k patches), while native-resolution families (Qwen3-VL, Pixtral) run the
        capped image whole (~7.7k / ~10k patches).

        Counting: ``image_grid_thw.prod()`` where the processor returns it (Qwen-VL —
        the exact pre-merge tower token count); otherwise total pixels in
        ``pixel_values`` / patch_size² (exact for tiled CLIP too, since tiles are
        batched through the tower and their patch counts add; drops per-tile CLS
        tokens, which KV_MEM_SAFETY_FACTOR dwarfs).

        Probed once and cached on self; falls back to a CLIP-shaped per-tile constant
        times a config-derived worst-case tile count if the probe fails, so batch sizing
        never hard-errors on an exotic processor.
        """
        cached = getattr(self, "_vit_num_patches_cached", None)
        if cached is not None:
            return cached
        vision_cfg = self.pipe.model.config.vision_config
        patch_size = getattr(vision_cfg, "patch_size", 14)
        try:
            from transformers import AutoImageProcessor

            side = int(np.sqrt(IMAGE_MAX_PIXELS))
            image_processor = AutoImageProcessor.from_pretrained(self.model_name)
            out = image_processor(
                images=Image.new("RGB", (side, side)), return_tensors="pt"
            )
            grid = out.get("image_grid_thw")
            if grid is not None:
                num_patches = int(grid.prod().item())
            else:
                num_patches = int(
                    out["pixel_values"].numel() // 3 // (patch_size * patch_size)
                )
            assert num_patches > 0
        except Exception as exc:
            # The probe counts patches per *image* — every tile the processor batches
            # through the tower — so the fallback has to as well. The CLIP-shaped
            # constant alone is ONE tile, and using it bare would under-count LLaVA-NeXT
            # by the tile factor (~5x). Worst-case anyres grid: the largest
            # the processor may pick plus the base resize that always accompanies it,
            # falling back in turn to the standard 2x2+base when the config carries no
            # pinpoints.
            crop = getattr(vision_cfg, "image_size", 336)
            pinpoints = getattr(self.pipe.model.config, "image_grid_pinpoints", None)
            try:
                max_tiles = 1 + max((h // crop) * (w // crop) for h, w in pinpoints)
            except Exception:
                max_tiles = 5
            num_patches = max_tiles * ((crop // patch_size) ** 2 + 1)
            logger.warning(
                f"ViT patch-count probe failed ({type(exc).__name__}: {exc}); falling "
                f"back to the CLIP-shaped estimate of {num_patches} patches "
                f"({max_tiles} tiles x {(crop // patch_size) ** 2 + 1}) — still an "
                f"under-estimate for native-resolution vision towers (Qwen3-VL, "
                f"Pixtral), so batch sizes may come out too large there."
            )
        self._vit_num_patches_cached = num_patches
        logger.info(
            f"[vit] worst-case patch count for batch sizing: {num_patches} "
            f"(patch_size={patch_size}, cap={IMAGE_MAX_PIXELS} px)"
        )
        return num_patches

    def _vit_activation_gb(self, pipe) -> float:
        """One item's ViT forward-pass activation memory (GB).

        Mirrors kv_cache_base._llm_activation_gb: with flash-attn the attention weights are
        never fully materialised, so only the per-layer projection/MLP tensors count, in
        float16, for ONE layer — under no_grad each layer's temporaries are freed as its
        forward() returns and the caching allocator reuses the same blocks (see that method
        for the full rationale). KV_MEM_SAFETY_FACTOR covers residuals and layer-norm buffers
        — the same shared dial every other cost term in this file uses.

        The cost is the complete per-layer tensor list × the patch count the tower really
        runs. The tile count is folded into ``_vit_num_patches``, which probes the model's
        own image processor at the serving cap and so covers tiled (LLaVA-NeXT) and
        native-resolution (Qwen3-VL, Pixtral) towers alike.

        This is a fixed per-item cost independent of KV compression or context length —
        every batch-size path that forward-passes an image through the vision tower
        (join and plain generate alike) pays it, so it's shared rather than duplicated. The
        join path caps its images to a single tile, so the probe's worst case is
        conservative there rather than wrong.
        """
        vision_cfg = pipe.model.config.vision_config
        vit_hidden = getattr(vision_cfg, "hidden_size", 1024)
        vit_intermediate = getattr(vision_cfg, "intermediate_size", 4 * vit_hidden)
        num_patches = self._vit_num_patches()

        # Elements per patch token, one ViT layer — every tensor the forward pass
        # materialises (no GQA in a CLIP vision tower, so Q/K/V are all vit_hidden):
        elements_per_token = (
            3 * vit_hidden  # q_proj, k_proj, v_proj outputs
            + 2 * vit_hidden  # flash-attention output, out_proj output
            + 2 * vit_intermediate  # fc1 output, activation output
            + vit_hidden  # fc2 output
        )
        return (KV_MEM_SAFETY_FACTOR * num_patches * elements_per_token * 2) / 1e9

    def _get_max_batch_size_image_join(
        self,
        file_paths: List[str],
        max_right_tokens: int,
        max_context_tokens: int,
        compression_ratio: Optional[float] = None,
        baseline_compression_ratio: Optional[float] = None,
    ) -> int:
        """Compute the max safe batch size for image join.

        The generic text-join formula only accounts for KV-cache file sizes and
        ignores two additional costs that dominate for images:
          - ViT activation memory:  one vision-encoder layer's tensors, over every
                                    anyres tile (see _vit_activation_gb)
          - LLM activation memory:  one decoder layer's tensors over the full
                                    (max_ctx + right_len) input sequence
                                    (see kv_cache_base._llm_activation_gb)

        With flash-attention, attention weights are never fully materialised, so only the
        per-layer projection/FFN tensors need to be counted; both helpers enumerate them.

        Parameters
        ----------
        file_paths         : paths to all left-side KV-cache .pt files (physical targets,
                             or on the relative-index path the baseline caches)
        max_right_tokens   : token count of one right-side input (right_ids length)
        max_context_tokens : token count of one left KV cache
        baseline_compression_ratio : set on the relative-index path — the measured
                             baseline size is scaled down to the target CR (see
                             kv_cache_base._get_max_batch_size).
        """
        from reasondb.memory_footprint.memory_report import get_biggest_file_size_gb

        save_dir = os.path.dirname(file_paths[0])
        filenames = [os.path.basename(f) for f in file_paths]
        kv_size_gb = get_biggest_file_size_gb(save_dir, filenames)

        # Relative-index path: the sized files are the (larger) baseline caches; the cache
        # actually made resident per item is the reconstructed target. Same guarded
        # scale-down as the text estimators in kv_cache_base.
        if (
            compression_ratio is not None
            and baseline_compression_ratio is not None
            and 0.0 <= baseline_compression_ratio < 1.0
            and baseline_compression_ratio <= compression_ratio < 1.0
        ):
            scale = (1.0 - compression_ratio) / (1.0 - baseline_compression_ratio)
            logger.info(
                f"[image_join] Sized baseline={kv_size_gb:.4f} GB (cr={baseline_compression_ratio}) "
                f"→ target={kv_size_gb * scale:.4f} GB (cr={compression_ratio}, scale={scale:.3f})"
            )
            kv_size_gb *= scale

        # (1) KV cache residency: KV_MEM_SAFETY_FACTOR × the resident bf16 footprint — the
        #     same shared guardrail every batch-size path uses. Unlike the generate paths,
        #     this join formula ALSO models ViT + LLM activations explicitly below, so this
        #     term is purely the KV cache + its allocator/fragmentation cushion.
        kv_mem_gb = KV_MEM_SAFETY_FACTOR * kv_size_gb

        # (2) ViT activation memory for one batch item — shared with the plain generate
        #     path's batch-size estimate (see _vit_activation_gb).
        vit_mem_gb = self._vit_activation_gb(self.pipe)

        # (3) LLM activation memory for one batch item during the first generate step.
        #     Input sequence length = max_ctx + right_len (the most expensive step).
        #     Shared with the other batch-size estimators (kv_cache_base._llm_activation_gb)
        #     instead of a separate local formula, so the model stays in sync everywhere.
        seq_len = max_context_tokens + max_right_tokens
        llm_mem_gb = self._llm_activation_gb(self.pipe, seq_len)

        # layer_devices covers every parameter's device (vision tower, projector,
        # embedding/lm_head, decoder layers), as in the other batch-size estimators, so a
        # GPU holding only non-decoder weight is still a bottleneck candidate.
        torch.cuda.empty_cache()
        layer_devices = list({p.device for p in self.pipe.model.parameters()})
        num_gpus = max(
            1, len({d for d in layer_devices if getattr(d, "type", None) == "cuda"})
        )

        logger.debug("layer_devices set: %s", sorted({str(d) for d in layer_devices}))
        logger.debug("first_device (input placement): %s", next(self.pipe.model.parameters()).device)
        for _i in range(torch.cuda.device_count()):
            _free, _total = torch.cuda.mem_get_info(_i)
            logger.debug(f"GPU {_i} free (pre-estimate): {_free/1e9:.2f} GB / {_total/1e9:.2f} GB total")

        # kv_mem is the resident cache — loaded directly per-layer onto each layer's own
        # device (no pipeline hand-off), so it genuinely splits evenly: divide by num_gpus.
        # vit_mem/llm_mem are both freshly forward-passed activation cost (ViT encoder, LLM
        # prefill over max_ctx+right_len) — a transient cost that may concentrate on any
        # single device, not necessarily first_device. So these terms are NOT divided by
        # num_gpus, and are checked against whichever device is tightest on free memory.
        #
        # There is no separate term for the right-side token activations: llm_mem_gb already
        # covers the whole max_ctx+right_len forward pass, so adding one double-counts.
        #
        # The floor matches the other estimators (see kv_cache_base._get_max_batch_size):
        # model-derived costs need only the tiny numerical backstop; the flat fallback
        # applies solely when there's no pipe to size a real activation cost from.
        if self.pipe is not None:
            floor_gb = _MIN_PER_ITEM_GB
        else:
            floor_gb = _UNCOSTED_ACTIVATION_FALLBACK_GB / num_gpus
            logger.warning(
                "[image_join] No pipe available to size per-item activation cost; falling "
                f"back to flat floor={floor_gb:.3f}GB/item for the batch-size estimate."
            )
        per_item_gb = max(kv_mem_gb / num_gpus + vit_mem_gb + llm_mem_gb, floor_gb)
        min_free_gb = self._min_free_gb(layer_devices)
        max_batch = int(_usable_free_gb(min_free_gb) // per_item_gb)
        logger.info(
            f"[image_join] Batch size estimate ({num_gpus} GPU(s), "
            f"factor={KV_MEM_SAFETY_FACTOR}): "
            f"kv/gpu={kv_mem_gb / num_gpus:.2f}GB "
            f"vit_mem={vit_mem_gb:.2f}GB llm_mem={llm_mem_gb:.2f}GB "
            f"per_item={per_item_gb:.2f}GB min_free={min_free_gb:.2f}GB → {max_batch}"
        )

        batch = self._snap_batch(max_batch)
        logger.info(f"[image_join] Using batch size: {batch}")
        logger.debug(f"computed batch_size={batch}")
        return batch

    # Convert DataFrame images to PIL Images
    @staticmethod
    def deserialize_image(b64_or_path):
        """Loads an image and downsizes it if it's too large."""
        try:
            if isinstance(b64_or_path, str) and b64_or_path.startswith("data:image"):
                # Handle base64 encoded images
                image_data = base64.b64decode(b64_or_path.split(",")[1])
                image = Image.open(BytesIO(image_data))
                image.load()  # Force load to catch truncation errors early
                image = image.convert("RGB")
            elif isinstance(b64_or_path, str):
                # Handle file paths
                image = Image.open(b64_or_path)
                image.load()  # Force load to catch truncation errors early
                image = image.convert("RGB")
            else:
                # Assume it's already a PIL Image or bytes
                if hasattr(b64_or_path, "convert"):
                    image = b64_or_path.convert("RGB")
                else:
                    image = Image.open(BytesIO(b64_or_path))
                    image.load()  # Force load to catch truncation errors early
                    image = image.convert("RGB")

            # Downsize image if too large
            if int(np.prod(image.size)) > IMAGE_MAX_PIXELS:
                ratio = np.sqrt(IMAGE_MAX_PIXELS / np.prod(image.size))
                image.thumbnail(
                    (int(image.size[0] * ratio), int(image.size[1] * ratio))
                )

            return image
        except Exception as e:
            logger.error(f"Error deserializing image {b64_or_path}: {str(e)}")
            raise

    async def _run_vanilla_inference(
        self,
        image_paths: List[str],
        question: str,
        boolean_question: bool,
    ):
        """Run vanilla inference without KV caching."""
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."

        # Set up multimodal processing parameters
        if boolean_question:
            context = "Answer the following question based on the image with '1' or '0'. Do not add any other comments."
        else:
            context = "Answer the following question based on the image. Do not add any other comments."

        full_question = context + " " + question
        max_new_tokens = 4 if boolean_question else 64

        # Initialize processor
        processor = AutoProcessor.from_pretrained(self.model_name)
        answers = []
        log_odds = []

        # Left-pad the text so generated tokens can be sliced uniformly across a batch.
        try:
            processor.tokenizer.padding_side = "left"
        except Exception:
            pass

        first_device = next(self.pipe.model.parameters()).device
        id0 = self.pipe.tokenizer.convert_tokens_to_ids("0")
        id1 = self.pipe.tokenizer.convert_tokens_to_ids("1")

        # Same prompt for every image (only the image differs).
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_question},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        def _run_batch(pil_images):
            """Run one batch of already-deserialized PIL images → (answers, log_odds)."""
            inputs = processor(
                images=pil_images,
                text=[prompt] * len(pil_images),
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
            self._reset_mrope_state()
            with torch.no_grad():
                generated = self.pipe.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            log_probs = torch.nn.functional.log_softmax(generated.scores[0], dim=-1)
            los = (log_probs[:, id1] - log_probs[:, id0]).cpu().tolist()
            input_len = inputs["input_ids"].shape[1]  # uniform width (left-padded)
            decoded = self.pipe.tokenizer.batch_decode(
                generated.sequences[:, input_len:], skip_special_tokens=True
            )
            del inputs, generated
            return decoded, los

        # Batch size: the uncompressed method has no cache file, so size from the sample
        # prompt length (which INCLUDES the expanded vision tokens) × model config, on the
        # same shared budget as the cache methods. self.vanilla_batch_size (None/0 → auto,
        # >0 → force) lets an A/B run pin it.
        if self.vanilla_batch_size and self.vanilla_batch_size > 0:
            batch_size = self.vanilla_batch_size
        else:
            # layer_devices covers every parameter's device (vision tower, projector,
            # embedding/lm_head included), not just the LLM decoder layers, so a GPU
            # holding only non-LLM-decoder weight is still a bottleneck candidate.
            torch.cuda.empty_cache()
            layer_devices = list({p.device for p in self.pipe.model.parameters()})

            logger.debug("layer_devices set: %s", sorted({str(d) for d in layer_devices}))
            try:
                logger.debug("hf_device_map: %s", getattr(self.pipe.model, "hf_device_map", None))
            except Exception as e:
                logger.debug("could not read hf_device_map: %s", e)
            try:
                logger.debug("lm_head device: %s", self.pipe.model.lm_head.weight.device)
            except Exception as e:
                logger.debug("could not read lm_head device: %s", e)
            logger.debug("embed/first param device: %s", next(self.pipe.model.parameters()).device)
            logger.debug("first_device (input placement): %s", first_device)
            for _i in range(torch.cuda.device_count()):
                _free, _total = torch.cuda.mem_get_info(_i)
                logger.debug(f"GPU {_i} free (pre-estimate): {_free/1e9:.2f} GB / {_total/1e9:.2f} GB total")
            logger.debug("_min_free_gb(layer_devices): %s GB", self._min_free_gb(layer_devices))

            sample_tokens = 0
            for p in image_paths:
                try:
                    _img = self.deserialize_image(p)
                    _si = processor(images=_img, text=prompt, return_tensors="pt")
                    sample_tokens = _si["input_ids"].shape[1]
                    del _img, _si
                    break
                except Exception:
                    continue
            batch_size = self._get_max_batch_size_vanilla(
                max_prompt_tokens=sample_tokens,
                layer_devices=layer_devices,
                batch_size=None,
                extra_activation_gb=self._vit_activation_gb(self.pipe),
            )

            logger.debug(f"sample_tokens={sample_tokens} computed batch_size={batch_size}")

        # Process images in batches; a batch failure (e.g. batched-multimodal edge cases)
        # degrades to one image at a time so answers/log_odds are never lost or misaligned.
        for start in tqdm(
            range(0, len(image_paths), batch_size),
            desc=f"Processing vanilla batches (bs={batch_size})",
        ):
            batch_paths = image_paths[start : start + batch_size]
            imgs, ok_idx = [], []
            for j, p in enumerate(batch_paths):
                try:
                    imgs.append(self.deserialize_image(p))
                    ok_idx.append(j)
                except Exception as e:
                    logger.warning(f"Error deserializing image {p}: {e}")

            # Default (deserialization failures keep these) — aligned to batch_paths order.
            batch_answers = ["Not sure"] * len(batch_paths)
            batch_los = [0.0] * len(batch_paths)

            if imgs:
                for _i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(_i)
                try:
                    decoded, los = _run_batch(imgs)
                    for k, j in enumerate(ok_idx):
                        batch_answers[j] = decoded[k]
                        batch_los[j] = los[k]

                    logger.debug(
                        f"batch_start={start} rows={len(imgs)}"
                    )
                    for _i in range(torch.cuda.device_count()):
                        _free, _total = torch.cuda.mem_get_info(_i)
                        _peak = torch.cuda.max_memory_allocated(_i) / 1e9
                        logger.debug(
                            f"  GPU {_i}: peak_allocated={_peak:.2f} GB "
                            f"free_after={_free/1e9:.2f} GB"
                        )
                except Exception as e:
                    logger.debug(f"EXCEPTION at batch_start={start}: {e}")
                    for _i in range(torch.cuda.device_count()):
                        try:
                            _free, _total = torch.cuda.mem_get_info(_i)
                            logger.debug(
                                f"  GPU {_i} free at failure: "
                                f"{_free/1e9:.2f} GB / {_total/1e9:.2f} GB total"
                            )
                        except Exception as _mem_err:
                            logger.debug(f"  GPU {_i} mem_get_info failed: {_mem_err}")
                    logger.warning(
                        f"Batched vanilla generation failed ({e}); "
                        f"falling back to one image at a time."
                    )
                    torch.cuda.empty_cache()
                    for k, j in enumerate(ok_idx):
                        try:
                            d1, l1 = _run_batch([imgs[k]])
                            batch_answers[j] = d1[0]
                            batch_los[j] = l1[0]
                        except Exception as e2:
                            logger.warning(
                                f"Per-image vanilla failed for {batch_paths[j]}: {e2}"
                            )

            answers.extend(batch_answers)
            log_odds.extend(batch_los)

        # Once per request, not per batch (see _run_kv_cache_multimodal for the rationale).
        torch.cuda.empty_cache()

        # Create result dictionary
        result_answers = {
            img_path: answer for img_path, answer in zip(image_paths, answers)
        }
        result_log_odds = {
            img_path: log_odd for img_path, log_odd in zip(image_paths, log_odds)
        }

        return result_answers, result_log_odds

    async def _run_kv_cache_multimodal(
        self,
        column_name: str,
        image_paths: List[str],
        question: str,
        compression_ratio: float,
        cache_dir: str,
        boolean_question: bool,
        materialized_compression_ratio: float,
        vanilla: bool,
        keep_in_memory: bool = False,
    ) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, Any]]:
        """Run KV cache-based multimodal inference on the data.

        The third return value carries the reconstruction-path telemetry this function
        measures internally (cache load/route/wait time, resident-store hit/miss deltas,
        the dynamically-picked batch size, per-GPU peak memory) so the caller can attach
        it to the response's ``stats`` block instead of it being logged and discarded.
        Empty (``{}``) on the vanilla path and the no-valid-caches early-out, since
        neither goes through the reconstruction/cache-load machinery.
        """

        if vanilla:
            logger.info(f"Using vanilla mode for inference (cr={compression_ratio})")
            answers, log_odds = await self._run_vanilla_inference(
                image_paths=image_paths,
                question=question,
                boolean_question=boolean_question,
            )
            return answers, log_odds, {}

        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."

        # Check if caches exist and generate them if not, one item at a time
        # Store mapping of row indices to cache files. The layout is press-aware
        # ({cache_dir}/{model}/{press}/comp{tag}/...), matching generation and the text
        # server, so physical Path-B and relative Path-A live under the same press_dir.
        press_dir = f"{cache_dir}/{self.model_name}/{self.press_name}"
        target_tag = f"comp{self.to_compression_tag(compression_ratio)}"
        save_dir = f"{press_dir}/{target_tag}"
        base_tag = f"comp{self.to_compression_tag(materialized_compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        recorded_errors = {}
        try:
            with open(f"{save_dir}/ERRORS.json", "r") as f:
                recorded_errors = json.load(f)
        except FileNotFoundError:
            if not self.use_relative_indices:
                raise  # ERRORS.json is required when serving physical caches only

        # Per served-image load plan, parallel to image_paths_with_caches.
        load_plans = []
        cache_files = []  # sizing/counting paths (physical or baseline) for the estimator
        image_paths_with_caches = []
        image_paths_without_caches = []
        for i, image_path in tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc=f"Preparing caches for CR {compression_ratio}",
        ):
            cache_name = self.hash_path(image_path)
            cache_filename = f"{save_dir}/cache_entry_{cache_name}.pt"

            relative = None
            if self.use_relative_indices:
                try:
                    relative = resolve_relative_source(
                        press_dir, target_tag, cache_name, base_tag
                    )
                except RelativeReconstructError as e:
                    logger.warning(f"[image {i}] relative _meta unusable, will try physical: {e}")
                    relative = None

            has_physical = os.path.exists(cache_filename)
            if relative is None and not has_physical:
                if not self.use_relative_indices:
                    assert (
                        cache_filename in recorded_errors
                    ), f"Cache file {cache_filename} missing but no recorded error found."
                    logger.warning(
                        f"Skipping image {i} {image_path} due to previous error: {recorded_errors[cache_filename]}",
                    )
                else:
                    logger.warning(
                        f"Skipping image {i} {image_path}: no relative source and no physical cache"
                    )
                image_paths_without_caches.append(image_path)
                continue

            load_plans.append({
                "image_path": image_path,
                "hash": cache_name,
                "physical": cache_filename if has_physical else None,
                "relative": relative,
            })
            cache_files.append(relative[0] if relative is not None else cache_filename)
            image_paths_with_caches.append(image_path)

        if not load_plans:
            logger.warning("No valid caches or images found")
            return {}, {}, {}

        # Set up multimodal processing parameters
        if boolean_question:
            context = "Answer the following question based on the image with '1' or '0'. Do not add any other comments."
        else:
            context = "Answer the following question based on the image. Do not add any other comments."
        question = context + " " + question
        answer_prefix = "Answer: "
        max_question_tokens = len(
            self.pipe.tokenizer.encode(question + answer_prefix, add_special_tokens=False)
        )
        batch_size = self.compression_ratio_to_batch_size[compression_ratio]
        llm_layers = self._get_llm_layers()
        layer_devices = [
            llm_layers[i].self_attn.q_proj.weight.device for i in range(len(llm_layers))
        ]
        # layer_devices (ordered, one entry per decoder layer) is reused below for indexed
        # per-layer cache placement, so it must stay decoder-layers-only. The batch-size
        # estimator instead gets every parameter's device (vision tower/projector/embedding/
        # lm_head included), so a GPU holding only non-LLM-decoder weight is still a
        # bottleneck candidate.
        torch.cuda.empty_cache()
        estimator_layer_devices = list({p.device for p in self.pipe.model.parameters()})
        # Relative-index path: cache_files hold baseline caches (larger than the reconstructed
        # target). Recover the baseline CR from a relative plan's baseline dir so the estimator
        # scales the baseline size down to the target and the batch can grow.
        baseline_cr = None
        if self.use_relative_indices:
            for p in load_plans:
                rel = p.get("relative")
                if rel is not None:
                    baseline_cr = self._comp_dir_to_cr(os.path.dirname(rel[0]))
                    break
            # Client drives materialized selection; verify against the resolved baseline.
            if baseline_cr is not None:
                assert baseline_cr == materialized_compression_ratio, (
                    f"client declared materialized_compression_ratio "
                    f"{materialized_compression_ratio} but effective cr "
                    f"{compression_ratio} resolves to baseline cr {baseline_cr} via "
                    f"relative indices"
                )

        logger.debug("layer_devices set: %s", sorted({str(d) for d in estimator_layer_devices}))
        logger.debug("first_device (input placement): %s", next(self.pipe.model.parameters()).device)
        for _i in range(torch.cuda.device_count()):
            _free, _total = torch.cuda.mem_get_info(_i)
            logger.debug(f"GPU {_i} free (pre-estimate): {_free/1e9:.2f} GB / {_total/1e9:.2f} GB total")

        batch_size = self._get_max_batch_size(
            column_name=column_name,
            batch_size=batch_size,
            compression_ratio=compression_ratio,
            layer_devices=estimator_layer_devices,
            file_paths=cache_files,
            cache_dir=cache_dir,
            max_question_tokens=max_question_tokens,
            baseline_compression_ratio=baseline_cr,
            extra_activation_gb=self._vit_activation_gb(self.pipe),
        )

        logger.debug(f"computed batch_size={batch_size}")
        max_new_tokens = 4 if boolean_question else 64

        # Initialize processor
        processor = AutoProcessor.from_pretrained(self.model_name)
        answers = []
        log_odds = []
        served_image_paths = []  # parallel to `answers`; images that produced an answer
        failed_at_load = []      # relative failed AND no physical → "Not sure" (rare edge)

        # Path A needs the decoder's rotary embedding (only inv_freq) to re-rotate kept
        # keys. Rerotation runs on the GPU on the main thread (mirroring the text server):
        # the elementwise rerotate is orders of magnitude faster there than on the CPU.
        # Loader threads stay CUDA-free (gather + pin only).
        rotary_emb = self._get_decoder().rotary_emb if self.use_relative_indices else None

        # CUDA-free per-batch loader (4-worker pool + 1-worker prefetch, mirroring the text
        # server): Path A gathers kept rows into pinned CPU buffers (H2D + rerotation happen
        # on the main thread), Path B loads the physical cache to CPU; any unusable image
        # returns an error marker. No CUDA here → safe in worker threads.
        def _load_batch_to_cpu(batch_plans):
            def _load_one(plan):
                _t0 = time.perf_counter()
                if plan["relative"] is not None:
                    baseline_path, idx_path = plan["relative"]
                    try:
                        k, v, idx = gather_payload_cpu(baseline_path, idx_path)
                        return (
                            {"needs_rerotate": True, "k": k, "v": v, "idx": idx,
                             "image_path": plan["image_path"]},
                            time.perf_counter() - _t0,
                        )
                    except RelativeReconstructError as e:
                        logger.warning(f"Path A failed for {plan['hash'][:12]}…, trying physical: {e}")
                physical = plan["physical"]
                if physical is not None and keep_in_memory:
                    # Pinned at prepare() and shared across queries; the route below only
                    # reads it (per-layer .to() copies), never mutates it. A miss is a
                    # setup bug, not a data issue (those are already error-marked), so it
                    # must NOT fall back to a disk read — that would quietly turn this
                    # into a disk-served operator.
                    c = PINNED_KV_STORE.get(physical)
                    if c is None:
                        logger.warning(f"keep_in_memory cache not pinned: {physical}")
                        return {"error": True, "image_path": plan["image_path"]}, time.perf_counter() - _t0
                    return {"cache": c, "image_path": plan["image_path"]}, time.perf_counter() - _t0
                if physical is not None and os.path.exists(physical):
                    try:
                        c = torch.load(physical, map_location="cpu", weights_only=False)
                        return {"cache": c, "image_path": plan["image_path"]}, time.perf_counter() - _t0
                    except Exception as e:  # noqa: BLE001 — a corrupt file must degrade, not crash
                        logger.warning(f"Physical cache unreadable for {plan['hash'][:12]}…: {e}")
                logger.warning(f"No usable cache for image hash={plan['hash'][:12]}… — skipping")
                return {"error": True, "image_path": plan["image_path"]}, time.perf_counter() - _t0
            with ThreadPoolExecutor(max_workers=min(len(batch_plans), 4)) as pool:
                return list(pool.map(_load_one, batch_plans))

        batch_starts = list(range(0, len(load_plans), batch_size))
        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        prefetch_future = prefetch_pool.submit(
            _load_batch_to_cpu, load_plans[0 : min(batch_size, len(load_plans))]
        )
        llm_layers = self._get_llm_layers()

        # Overlap instrumentation (mirrors the text server's "KV cache timing" line):
        # thread-sum loader work vs. how much of it was NOT hidden behind GPU batches.
        total_disk_load_time = 0.0
        total_route_time = 0.0
        total_load_wait_time = 0.0
        _pin_h0, _pin_m0, _ = PINNED_KV_STORE.stats()
        _loop_t0 = time.perf_counter()
        _peaks: dict = {}  # device index -> peak allocated GB, for the response stats

        # Process images in batches
        for batch_start in tqdm(
            batch_starts,
            desc=f"Processing batches for CR {compression_ratio}",
        ):
            batch_input_ids = []
            batch_attention_masks = []
            caches = []
            context_lengths = []
            batch_served = []  # image_paths whose cache made it into `caches`, in order

            # Wait for the prefetched batch, then immediately prefetch the next one (overlaps
            # this batch's GPU work with the next batch's CPU load/reconstruct).
            _wait_t0 = time.perf_counter()
            cpu_results = prefetch_future.result()
            total_load_wait_time += time.perf_counter() - _wait_t0
            total_disk_load_time += sum(r[1] for r in cpu_results)
            next_start = batch_start + batch_size
            if next_start < len(load_plans):
                prefetch_future = prefetch_pool.submit(
                    _load_batch_to_cpu,
                    load_plans[next_start : min(next_start + batch_size, len(load_plans))],
                )

            _route_t0 = time.perf_counter()
            for r, _dt in cpu_results:
                if r.get("error"):
                    failed_at_load.append(r["image_path"])
                    continue
                if r.get("needs_rerotate"):
                    # Path A: sliced H2D + per-device rerotate — every layer lands directly
                    # on its own GPU straight from the pinned buffer (no batch×cache
                    # staging on device 0, no second hop); the batching loop's per-layer
                    # .to() below is a no-op for these caches.
                    cache = rerotate_payload_sharded(
                        r["k"], r["v"], r["idx"], rotary_emb, layer_devices
                    )
                else:
                    cache = r["cache"]
                caches.append(cache)
                batch_served.append(r["image_path"])
                context_lengths.append(next(_iter_cache_layers(cache))[0].shape[2])

            if not caches:
                continue

            max_context_len = max(context_lengths)

            # Prepare inputs for each image in the batch
            for i, ctx_len in enumerate(context_lengths):
                padded_context_ids = torch.full(
                    (1, ctx_len),
                    self.pipe.tokenizer.pad_token_id + 1,
                    device=self.device,
                )
                pad_len = max_context_len - ctx_len
                padding_ids = torch.full(
                    (1, pad_len), self.pipe.tokenizer.pad_token_id, device=self.device
                )
                padded_context = torch.cat([padding_ids, padded_context_ids], dim=1)

                # Create the prompt structure
                separator = "\n" + "#" * ctx_len if ctx_len > 0 else "\n" + "#"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": context + separator},
                        ],
                    }
                ]

                # Apply chat template
                prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                _, question_suffix = prompt.split(separator)

                question_text = question + question_suffix + answer_prefix
                question_ids = self.pipe.tokenizer.encode(
                    question_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)

                input_ids = torch.cat([padded_context, question_ids], dim=1)

                # Create attention masks
                context_mask = torch.ones_like(padded_context_ids)
                padding_mask = torch.zeros_like(padding_ids)
                question_mask = torch.ones_like(question_ids)
                attention_mask = torch.cat(
                    [padding_mask, context_mask, question_mask], dim=1
                )

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)

            # Batch the inputs
            batched_inputs = torch.cat(batch_input_ids, dim=0)
            batched_attention_mask = torch.cat(batch_attention_masks, dim=0)

            # Batch the caches ON the GPU: move each (k, v) to its layer's device, then pad
            # + concatenate there (GPU batching, matching the text server). The transient 2x
            # (per-image inputs alongside the concatenated output) is covered by the
            # batch-size estimate's safety margin; it avoids the slower CPU pad/cat.
            batched_cache = []
            for layer_idx, layers in enumerate(zip(*[_iter_cache_layers(c) for c in caches])):
                layer_device = llm_layers[layer_idx].self_attn.q_proj.weight.device
                max_seq_len = max(k.shape[2] for k, _ in layers)
                keys_padded = []
                values_padded = []

                for k, v in layers:
                    k = k.to(layer_device, non_blocking=H2D_NON_BLOCKING)
                    v = v.to(layer_device, non_blocking=H2D_NON_BLOCKING)
                    pad_len = max_seq_len - k.shape[2]
                    if pad_len > 0:
                        k = torch.nn.functional.pad(k, (0, 0, pad_len, 0))
                        v = torch.nn.functional.pad(v, (0, 0, pad_len, 0))
                    keys_padded.append(k.contiguous())
                    values_padded.append(v.contiguous())

                batched_cache.append(
                    (torch.cat(keys_padded, dim=0), torch.cat(values_padded, dim=0))
                )

            padded_cache = DynamicCache()
            for layer_idx, (keys, values) in enumerate(batched_cache):
                padded_cache.update(keys, values, layer_idx)

            # Route ends once the batched cache is resident on the per-layer GPUs.
            torch.cuda.synchronize()
            total_route_time += time.perf_counter() - _route_t0

            # Move inputs to the device of the embedding layer (first layer). The cache is
            # already on the per-layer GPUs (built there above) — no post-batch move needed.
            first_device = next(self.pipe.model.parameters()).device
            batched_inputs = batched_inputs.to(first_device)
            batched_attention_mask = batched_attention_mask.to(first_device)

            # Generate responses
            for _i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(_i)
            # No pixel inputs here: without this reset, Qwen3-VL would reuse the
            # previous request's per-image rope_deltas (see _reset_mrope_state).
            self._reset_mrope_state()
            try:
                with torch.no_grad():
                    generated = self.pipe.model.generate(
                        input_ids=batched_inputs,
                        attention_mask=batched_attention_mask,
                        past_key_values=padded_cache,
                        pad_token_id=self.pipe.tokenizer.eos_token_id,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
            except Exception as e:
                logger.debug(
                    f"EXCEPTION during generate() at batch of "
                    f"{len(caches)} item(s): {e}"
                )
                for _i in range(torch.cuda.device_count()):
                    try:
                        _free, _total = torch.cuda.mem_get_info(_i)
                        _peak = torch.cuda.max_memory_allocated(_i) / 1e9
                        logger.debug(
                            f"  GPU {_i}: peak_allocated={_peak:.2f} GB "
                            f"free_at_failure={_free/1e9:.2f} GB / {_total/1e9:.2f} GB total"
                        )
                    except Exception as e2:
                        logger.debug(f"  GPU {_i}: could not read memory stats: {e2}")
                raise

            logger.debug(f"batch of {len(caches)} item(s) succeeded")
            for _i in range(torch.cuda.device_count()):
                _free, _total = torch.cuda.mem_get_info(_i)
                _peak = torch.cuda.max_memory_allocated(_i) / 1e9
                _peaks[_i] = max(_peaks.get(_i, 0.0), _peak)
                logger.debug(
                    f"  GPU {_i}: peak_allocated={_peak:.2f} GB "
                    f"free_after={_free/1e9:.2f} GB / {_total/1e9:.2f} GB total"
                )

            logits = generated.scores[0]  # type: ignore
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            id0 = self.pipe.tokenizer.convert_tokens_to_ids("0")
            id1 = self.pipe.tokenizer.convert_tokens_to_ids("1")
            log_probs_0 = log_probs[:, id0]
            log_probs_1 = log_probs[:, id1]
            log_odds_1_vs_0 = log_probs_1 - log_probs_0

            # Decode the generated tokens
            decoded = self.pipe.tokenizer.batch_decode(
                generated.sequences[:, batched_inputs.shape[1] :],  # type: ignore
                skip_special_tokens=True,
            )

            answers.extend(decoded)
            log_odds.extend(log_odds_1_vs_0.cpu().tolist())
            served_image_paths.extend(batch_served)  # keep answers aligned to served images

            # Cleanup batch resources
            del (
                caches,
                batched_inputs,
                batched_attention_mask,
                padded_cache,
                generated,
                logits,
                log_probs,
            )
            del batch_input_ids, batch_attention_masks, batched_cache

        prefetch_pool.shutdown(wait=False)
        # Once per request, not per batch: batches within a request reuse identical
        # shapes, so per-batch empty_cache only added a sync + cudaMalloc churn. The
        # end-of-request release keeps mem_get_info (used by the batch estimators)
        # honest for the next request.
        torch.cuda.empty_cache()

        total_elapsed = time.perf_counter() - _loop_t0
        n_caches = max(len(load_plans), 1)
        pinned_note = ""
        _pin_hits = _pin_misses = _pin_held = None
        if keep_in_memory:
            _pin_h1, _pin_m1, _pin_gb = PINNED_KV_STORE.stats()
            _pin_hits, _pin_misses, _pin_held = (
                _pin_h1 - _pin_h0,
                _pin_m1 - _pin_m0,
                _pin_gb,
            )
            pinned_note = (
                f" | pinned-RAM: {_pin_hits} hits / {_pin_misses} misses, "
                f"{_pin_gb:.1f} GB held"
            )
        logger.info(
            f"KV cache timing ({n_caches} caches) — "
            f"load caches: {total_disk_load_time:.2f}s ({total_disk_load_time / n_caches:.3f}s/cache, thread-sum) | "
            f"distribute to GPUs: {total_route_time:.2f}s ({total_route_time / n_caches:.3f}s/cache) | "
            f"load overhead (not hidden): {total_load_wait_time:.2f}s | "
            f"total: {total_elapsed:.2f}s{pinned_note}"
        )
        if keep_in_memory and _pin_misses:
            # Every miss is a cache prepare() should have pinned. The per-image degrade
            # above kept the request answering, but those images would answer "Not sure"
            # without ever having been served from RAM — a plausible-looking result whose
            # premise is false. Fail the request instead. (Never fires on a recorded
            # generation error: those never reach the store, see prepare_caches.)
            raise PinnedKVUnavailable(
                f"{_pin_misses} of {n_caches} caches were not resident while serving "
                f"column {column_name!r} on {self.model_name} (cr={compression_ratio}) as "
                "an -in-memory operator. prepare() must run for this column before it "
                "serves; nothing is ever loaded from disk on this path."
            )


        # Images with no cache (skipped at prepare) or whose reconstruction failed with no
        # physical fallback (skipped at load) get "Not sure"; served images keyed to answers.
        not_served = image_paths_without_caches + failed_at_load
        result_answers = {img_path: "Not sure" for img_path in not_served}
        result_answers.update(
            {
                img_path: answer
                for img_path, answer in zip(served_image_paths, answers)
            }
        )
        result_log_odds = {img_path: 0.0 for img_path in not_served}
        result_log_odds.update(
            {
                img_path: log_odd
                for img_path, log_odd in zip(served_image_paths, log_odds)
            }
        )

        load_stats = {
            "cache_load_s": total_disk_load_time,
            "cache_route_s": total_route_time,
            "cache_wait_s": total_load_wait_time,
            "pinned_hits": _pin_hits,
            "pinned_misses": _pin_misses,
            "pinned_gb": _pin_held,
            "dynamic_batch_size": batch_size,
            "peaks": _peaks,
        }
        return result_answers, result_log_odds, load_stats

    def compute_image_qa_join_response(
        self,
        left_column_name: str,
        pairs: List[Dict[str, str]],
        question: str,
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
        materialized_compression_ratio: float,
        vanilla: bool,
        keep_in_memory: bool = False,
    ) -> Dict[str, Dict]:
        """Run join inference: image A is KV-cached, image B is passed inline."""
        # The image-join path has no vanilla (no-cache) implementation; reject it
        # explicitly rather than silently loading a cache the client asked to skip.
        # keep_in_memory, by contrast, IS supported: this path loads the same physical
        # caches, and two of the four operators sharing an -in-memory image backend are
        # join predicates.
        assert not vanilla, "vanilla mode is not supported for the image-join path"
        self._validate_client_crs(
            compression_ratio, materialized_compression_ratio, vanilla, keep_in_memory
        )
        _t0 = time.perf_counter()
        answers, log_odds, load_stats = asyncio.run(
            self._run_kv_cache_multimodal_join(
                left_column_name=left_column_name,
                pairs=pairs,
                question=question,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
                materialized_compression_ratio=materialized_compression_ratio,
                keep_in_memory=keep_in_memory,
            )
        )
        return {
            "answers": answers,
            "log_odds": log_odds,
            "stats": self._request_stats(
                path="join",
                n_items=len(pairs),
                elapsed_s=time.perf_counter() - _t0,
                effective_compression_ratio=compression_ratio,
                materialized_compression_ratio=materialized_compression_ratio,
                vanilla=False,
                keep_in_memory=keep_in_memory,
                **load_stats,
            ),
        }

    async def _run_kv_cache_multimodal_join(
        self,
        left_column_name: str,
        pairs: List[Dict[str, str]],
        question: str,
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
        materialized_compression_ratio: float,
        keep_in_memory: bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
        """
        Batched join: load B left image KV caches at once, then for each right
        image run one batch-B generate() call — mirrors the TextQA join structure.

        Outer loop: batches of B left images (KV cache on GPU, stacked to (B, heads, ctx, dim)).
        Inner loop: M unique right images, each processed as a single batch-B forward pass.
        Total generate() calls: ceil(N/B) × M  instead of  N × M  for per-pair calls.
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."

        # Press-aware layout ({model}/{press}/comp{tag}) — matches generation, prepare and
        # the single-image serving path.
        press_dir = f"{cache_dir}/{self.model_name}/{self.press_name}"
        target_tag = f"comp{self.to_compression_tag(compression_ratio)}"
        save_dir = f"{press_dir}/{target_tag}"
        base_tag = f"comp{self.to_compression_tag(materialized_compression_ratio)}"

        join_instruction = (
            "Answer the following question about the two images with '1' or '0'. "
            "Do not add any other comments."
        )
        answer_prefix = "Answer: "
        max_new_tokens = 4 if boolean_question else 64

        processor = AutoProcessor.from_pretrained(self.model_name)
        first_device = next(self.pipe.model.parameters()).device
        llm_layers = self._get_llm_layers()
        layer_devices = [
            llm_layers[i].self_attn.q_proj.weight.device for i in range(len(llm_layers))
        ]
        id0 = self.pipe.tokenizer.convert_tokens_to_ids("0")
        id1 = self.pipe.tokenizer.convert_tokens_to_ids("1")

        result_answers: Dict[str, str] = {}
        result_log_odds: Dict[str, float] = {}

        # Build ordered unique left / right lists and a fast pair-membership set.
        unique_lefts = list(dict.fromkeys(p["left"] for p in pairs))
        unique_rights = list(dict.fromkeys(p["right"] for p in pairs))
        requested_pairs = {(p["left"], p["right"]) for p in pairs}

        # Pre-build the right-side prompt template (identical for every right image).
        full_question = join_instruction + " " + question
        messages_template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_question},
                ],
            }
        ]
        prompt_template = (
            processor.apply_chat_template(
                messages_template, add_generation_prompt=True, tokenize=False
            )
            + answer_prefix
        )

        # Per-left load plan, mirroring _run_kv_cache_multimodal: with --use-relative-indices,
        # prefer reconstructing from a baseline + relative indices (Path A), falling back to a
        # physical cache (Path B). A left with neither degrades all its pairs to "Not sure".
        load_plans = []
        sizing_paths = []  # real on-disk files (physical or baseline) for the estimator
        for left_path in unique_lefts:
            hash_name = self.hash_path(left_path)
            cache_filename = f"{save_dir}/cache_entry_{hash_name}.pt"
            relative = None
            if self.use_relative_indices:
                try:
                    relative = resolve_relative_source(
                        press_dir, target_tag, hash_name, base_tag
                    )
                except RelativeReconstructError as e:
                    logger.warning(
                        f"ImageQAJoin: relative _meta unusable for {hash_name[:12]}…, will try physical: {e}"
                    )
                    relative = None
            has_physical = os.path.exists(cache_filename)
            if relative is None and not has_physical:
                logger.warning(
                    f"ImageQAJoin: skipping left image {left_path} — no cache and no relative source"
                )
                for right_path in unique_rights:
                    if (left_path, right_path) in requested_pairs:
                        key = f"{left_path}|{right_path}"
                        result_answers[key] = "Not sure"
                        result_log_odds[key] = 0.0
                continue
            load_plans.append({
                "image_path": left_path,
                "hash": hash_name,
                "physical": cache_filename if has_physical else None,
                "relative": relative,
            })
            sizing_paths.append(relative[0] if relative is not None else cache_filename)

        N = len(load_plans)
        M = len(unique_rights)

        if not load_plans:
            logger.warning("ImageQAJoin: no left image has a usable cache or relative source")
            return result_answers, result_log_odds, {}

        # --- Dynamic batch size (mirrors TextQA join) ---
        # Step 1: measure right-side token count from the first right image (CPU only).
        _sample_right = self.deserialize_image(unique_rights[0])
        _sample_right.thumbnail((336, 336))
        _sample_inputs = processor(
            images=_sample_right, text=prompt_template, return_tensors="pt"
        )
        sample_right_len = _sample_inputs.input_ids.shape[1] - 1  # strip BOS
        del _sample_right, _sample_inputs

        # Step 2: measure left context length from the first plan (CPU only). For a
        # relative plan the reconstructed length = kept-token count = the index file's
        # last dim (the tiny idx file avoids deserializing a whole baseline cache here).
        if load_plans[0]["relative"] is not None:
            _idx = torch.load(
                load_plans[0]["relative"][1], map_location="cpu", weights_only=True
            )
            sample_ctx_len = _idx.shape[2]
            del _idx
        else:
            _sample_cache = torch.load(
                load_plans[0]["physical"],
                map_location="cpu",
                weights_only=False,
            )
            if hasattr(_sample_cache, "layers") and _sample_cache.layers:
                sample_ctx_len = _sample_cache.layers[0].keys.shape[2]
            elif hasattr(_sample_cache, "_cache") and _sample_cache._cache:
                sample_ctx_len = _sample_cache._cache[0].key_states.shape[2]
            else:
                sample_ctx_len = _sample_cache.__dict__["key_cache"][0].shape[2]
            del _sample_cache

        # Step 3: compute batch size — use the explicit per-ratio override if set,
        # otherwise call the image-specific formula that accounts for ViT + LLM activations.
        # Relative plans are sized against the (larger) baseline files; hand the estimator
        # the baseline CR so it scales down to the reconstructed target size.
        baseline_cr = None
        if self.use_relative_indices:
            for p in load_plans:
                if p["relative"] is not None:
                    baseline_cr = self._comp_dir_to_cr(os.path.dirname(p["relative"][0]))
                    break
            # Client drives materialized selection; verify against the resolved baseline.
            if baseline_cr is not None:
                assert baseline_cr == materialized_compression_ratio, (
                    f"client declared materialized_compression_ratio "
                    f"{materialized_compression_ratio} but effective cr "
                    f"{compression_ratio} resolves to baseline cr {baseline_cr} via "
                    f"relative indices"
                )
        configured = self.compression_ratio_to_batch_size.get(compression_ratio)
        if configured is not None:
            batch_size = configured
        else:
            try:
                batch_size = self._get_max_batch_size_image_join(
                    file_paths=sizing_paths,
                    max_right_tokens=sample_right_len,
                    max_context_tokens=sample_ctx_len,
                    compression_ratio=compression_ratio,
                    baseline_compression_ratio=baseline_cr,
                )
            except Exception as e:
                logger.warning(
                    f"[image_join] Could not compute batch size: {e}. Falling back to 1."
                )
                batch_size = 1

        logger.info(
            f"[image_join] N={N} valid left images, M={M} right images, "
            f"batch_size={batch_size}, compression_ratio={compression_ratio}"
        )

        # Path A needs the decoder's rotary embedding (only inv_freq) to re-rotate kept keys.
        rotary_emb = self._get_decoder().rotary_emb if self.use_relative_indices else None

        # Same split as the text server's join loader (_load_join_to_cpu): "load" is disk
        # I/O (mmap'd baseline gather, or torch.load of a physical cache); "route" is
        # getting it onto the layer GPUs (rerotate, or the per-layer .to() transfer).
        total_disk_load_time = 0.0
        total_route_time = 0.0
        _pin_h0, _pin_m0, _ = PINNED_KV_STORE.stats()
        _peaks: dict = {}  # device index -> peak allocated GB, for the response stats

        for batch_start in tqdm(
            range(0, N, batch_size),
            desc="ImageQA join: processing batches of left images",
        ):
            batch_plans = load_plans[batch_start : batch_start + batch_size]

            # ------------------------------------------------------------------
            # Step 1: Load B KV caches, each layer on its own GPU device. Physical
            # plans deserialize + route per layer; relative plans (Path A) gather
            # kept rows from the baseline and rerotate directly onto the layer
            # devices — shape-identical to a physical cache. A load failure
            # degrades that left's pairs to "Not sure" instead of aborting.
            # ------------------------------------------------------------------
            caches = []  # list of (key_layers, val_layers) — each layer already on GPU
            context_lens = []
            batch_lefts = []  # left paths whose cache made it into `caches`, in order
            for plan in batch_plans:
                left_path = plan["image_path"]
                keys = vals = None
                if plan["relative"] is not None:
                    try:
                        _load_t0 = time.perf_counter()
                        k_cpu, v_cpu, ridx = gather_payload_cpu(*plan["relative"])
                        total_disk_load_time += time.perf_counter() - _load_t0
                        _route_t0 = time.perf_counter()
                        cache = rerotate_payload_sharded(
                            k_cpu, v_cpu, ridx, rotary_emb, layer_devices
                        )
                        total_route_time += time.perf_counter() - _route_t0
                        keys, vals = [], []
                        for k, v in _iter_cache_layers(cache):
                            keys.append(k)
                            vals.append(v)
                    except RelativeReconstructError as e:
                        logger.warning(
                            f"ImageQAJoin: Path A failed for {plan['hash'][:12]}…, trying physical: {e}"
                        )
                        keys = vals = None
                if keys is None and plan["physical"] is not None:
                    _load_t0 = time.perf_counter()
                    if keep_in_memory:
                        # Pinned at prepare(); never re-read from disk. The per-layer
                        # .to() copies below leave the shared object untouched.
                        cache_data = PINNED_KV_STORE.get(plan["physical"])
                        if cache_data is None:
                            raise PinnedKVUnavailable(
                                f"keep_in_memory cache not pinned: {plan['physical']}. "
                                f"prepare() must run for column {left_column_name!r} on "
                                f"{self.model_name} before it serves a join."
                            )
                    else:
                        cache_data = torch.load(
                            plan["physical"], map_location="cpu", weights_only=False
                        )
                    total_disk_load_time += time.perf_counter() - _load_t0
                    _route_t0 = time.perf_counter()
                    # Normalise across DynamicCache format versions
                    if hasattr(cache_data, "layers") and cache_data.layers:
                        keys = [
                            layer.keys.to(layer_devices[j])
                            for j, layer in enumerate(cache_data.layers)
                        ]
                        vals = [
                            layer.values.to(layer_devices[j])
                            for j, layer in enumerate(cache_data.layers)
                        ]
                    elif hasattr(cache_data, "_cache") and cache_data._cache:
                        keys = [
                            item.key_states.to(layer_devices[j])
                            for j, item in enumerate(cache_data._cache)
                        ]
                        vals = [
                            item.value_states.to(layer_devices[j])
                            for j, item in enumerate(cache_data._cache)
                        ]
                    else:
                        key_cache = cache_data.__dict__["key_cache"]
                        val_cache = cache_data.__dict__["value_cache"]
                        keys = [k.to(layer_devices[j]) for j, k in enumerate(key_cache)]
                        vals = [v.to(layer_devices[j]) for j, v in enumerate(val_cache)]
                    total_route_time += time.perf_counter() - _route_t0
                    del cache_data
                if keys is None:
                    logger.warning(
                        f"ImageQAJoin: no usable cache for left image {left_path} — skipping"
                    )
                    for right_path in unique_rights:
                        if (left_path, right_path) in requested_pairs:
                            key = f"{left_path}|{right_path}"
                            result_answers[key] = "Not sure"
                            result_log_odds[key] = 0.0
                    continue
                caches.append((keys, vals))
                context_lens.append(keys[0].shape[2])
                batch_lefts.append(left_path)

            if not caches:
                continue
            B = len(batch_lefts)

            max_ctx = max(context_lens)

            # ------------------------------------------------------------------
            # Step 2: Left-pad and stack B caches → (B, heads, max_ctx, head_dim)
            # per layer.  Shorter caches get zero-padded on the left (attention
            # mask will mask those positions out during the forward pass).
            # ------------------------------------------------------------------
            batched_gpu_keys = []
            batched_gpu_values = []
            for layer_idx, layer_kv_pairs in enumerate(
                zip(*[zip(kc, vc) for kc, vc in caches])
            ):
                keys_padded, values_padded = [], []
                for k, v in layer_kv_pairs:
                    pad_len = max_ctx - k.shape[2]
                    k_pad = (
                        torch.nn.functional.pad(k, (0, 0, pad_len, 0)).contiguous()
                        if pad_len > 0
                        else k
                    )
                    v_pad = (
                        torch.nn.functional.pad(v, (0, 0, pad_len, 0)).contiguous()
                        if pad_len > 0
                        else v
                    )
                    keys_padded.append(k_pad)
                    values_padded.append(v_pad)
                batched_gpu_keys.append(torch.cat(keys_padded, dim=0))
                batched_gpu_values.append(torch.cat(values_padded, dim=0))

            # ------------------------------------------------------------------
            # Step 3: For each right image, run one batch-B generate() call.
            # ------------------------------------------------------------------
            for right_path in unique_rights:
                # Skip right images not paired with any left in this batch.
                if not any(
                    (left_path, right_path) in requested_pairs
                    for left_path in batch_lefts
                ):
                    continue

                try:
                    image_B = self.deserialize_image(right_path)
                    # Cap to a single tile (336×336) to avoid a transformers 5.x
                    # LLaVA-NeXT bug where the feature count disagrees with tokens.
                    image_B.thumbnail((336, 336))

                    inputs_B = processor(
                        images=image_B, text=prompt_template, return_tensors="pt"
                    )

                    # Strip leading BOS — it is already covered by the left KV cache.
                    right_ids_1 = inputs_B.input_ids[:, 1:].to(first_device)  # (1, R)
                    pixel_values_1 = inputs_B.pixel_values.to(first_device)   # (1, P, C, H, W)
                    image_sizes_1 = inputs_B.image_sizes.to(first_device)     # (1, 2)
                    right_len = right_ids_1.shape[1]

                    # Expand right-side tensors to batch size B — each left context
                    # is compared against the same right image.
                    pixel_values_B = pixel_values_1.expand(
                        B, *pixel_values_1.shape[1:]
                    ).contiguous()
                    image_sizes_B = image_sizes_1.expand(B, -1).contiguous()

                    # Build (B, max_ctx + right_len) input_ids and attention_mask.
                    # Layout per item: [left_padding | ctx_placeholder | right_tokens]
                    # Attention mask: [0…0 | 1…1 | 1…1]
                    batch_input_ids = []
                    batch_attn_mask = []
                    for ctx_len in context_lens:
                        pad_len = max_ctx - ctx_len
                        pad_ids = torch.full(
                            (1, pad_len),
                            self.pipe.tokenizer.pad_token_id,
                            device=first_device,
                        )
                        ctx_ids = torch.full(
                            (1, ctx_len),
                            self.pipe.tokenizer.pad_token_id + 1,
                            device=first_device,
                        )
                        input_ids_i = torch.cat(
                            [pad_ids, ctx_ids, right_ids_1.expand(1, -1)], dim=1
                        )
                        attn_mask_i = torch.cat(
                            [
                                torch.zeros_like(pad_ids),
                                torch.ones_like(ctx_ids),
                                torch.ones((1, right_len), device=first_device),
                            ],
                            dim=1,
                        )
                        batch_input_ids.append(input_ids_i)
                        batch_attn_mask.append(attn_mask_i)

                    batched_inputs = torch.cat(batch_input_ids, dim=0)  # (B, max_ctx+R)
                    batched_attn = torch.cat(batch_attn_mask, dim=0)    # (B, max_ctx+R)

                    # Fresh DynamicCache for this forward pass — generate() mutates
                    # it in-place, so we must not reuse across right images.
                    batch_cache = DynamicCache()
                    for layer_idx, (k, v) in enumerate(
                        zip(batched_gpu_keys, batched_gpu_values)
                    ):
                        batch_cache.update(k.clone(), v.clone(), layer_idx)

                    for _i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(_i)
                    # Stale rope_deltas would shadow the fresh ones this batch's
                    # pixel inputs should produce (see _reset_mrope_state).
                    self._reset_mrope_state()
                    try:
                        with torch.no_grad():
                            generated = self.pipe.model.generate(
                                input_ids=batched_inputs,
                                attention_mask=batched_attn,
                                past_key_values=batch_cache,
                                pixel_values=pixel_values_B,
                                image_sizes=image_sizes_B,
                                pad_token_id=self.pipe.tokenizer.eos_token_id,
                                do_sample=False,
                                max_new_tokens=max_new_tokens,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                    except Exception as e:
                        logger.debug(
                            f"[image_join] EXCEPTION during generate() at "
                            f"batch of {len(batch_lefts)} item(s): {e}"
                        )
                        for _i in range(torch.cuda.device_count()):
                            try:
                                _free, _total = torch.cuda.mem_get_info(_i)
                                _peak = torch.cuda.max_memory_allocated(_i) / 1e9
                                logger.debug(
                                    f"  GPU {_i}: peak_allocated={_peak:.2f} GB "
                                    f"free_at_failure={_free/1e9:.2f} GB / {_total/1e9:.2f} GB total"
                                )
                            except Exception as e2:
                                logger.debug(f"  GPU {_i}: could not read memory stats: {e2}")
                        raise

                    logger.debug(f"[image_join] batch of {len(batch_lefts)} item(s) succeeded")
                    for _i in range(torch.cuda.device_count()):
                        _free, _total = torch.cuda.mem_get_info(_i)
                        _peak = torch.cuda.max_memory_allocated(_i) / 1e9
                        _peaks[_i] = max(_peaks.get(_i, 0.0), _peak)
                        logger.debug(
                            f"  GPU {_i}: peak_allocated={_peak:.2f} GB "
                            f"free_after={_free/1e9:.2f} GB / {_total/1e9:.2f} GB total"
                        )

                    logits = generated.scores[0]  # (B, vocab_size)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    log_odds_vals = (log_probs[:, id1] - log_probs[:, id0]).tolist()

                    decoded_all = self.pipe.tokenizer.batch_decode(
                        generated.sequences[:, batched_inputs.shape[1] :],
                        skip_special_tokens=True,
                    )

                    for b, left_path in enumerate(batch_lefts):
                        if (left_path, right_path) in requested_pairs:
                            key = f"{left_path}|{right_path}"
                            result_answers[key] = (
                                decoded_all[b] if decoded_all[b] else "Not sure"
                            )
                            result_log_odds[key] = log_odds_vals[b]

                    del inputs_B, right_ids_1, pixel_values_1, image_sizes_1
                    del pixel_values_B, image_sizes_B
                    del batch_input_ids, batch_attn_mask, batched_inputs, batched_attn
                    del batch_cache, generated, logits, log_probs

                except Exception as e:
                    logger.warning(
                        f"ImageQAJoin: error processing right image {right_path} "
                        f"against batch {batch_start}–{batch_start+B-1}: {e}"
                    )
                    for left_path in batch_lefts:
                        if (left_path, right_path) in requested_pairs:
                            key = f"{left_path}|{right_path}"
                            result_answers[key] = "Not sure"
                            result_log_odds[key] = 0.0

            # Clean up batch GPU tensors
            del caches, batched_gpu_keys, batched_gpu_values

        # Once per request, not per batch (see _run_kv_cache_multimodal for the rationale).
        torch.cuda.empty_cache()

        pinned_note = ""
        _pin_hits = _pin_misses = _pin_held = None
        if keep_in_memory:
            _pin_h1, _pin_m1, _pin_gb = PINNED_KV_STORE.stats()
            _pin_hits, _pin_misses, _pin_held = (
                _pin_h1 - _pin_h0,
                _pin_m1 - _pin_m0,
                _pin_gb,
            )
            pinned_note = (
                f" | pinned-RAM: {_pin_hits} hits / {_pin_misses} misses, "
                f"{_pin_gb:.1f} GB held"
            )
        logger.info(
            f"KV cache timing join ({N} left images) — "
            f"load caches: {total_disk_load_time:.2f}s ({total_disk_load_time / max(N, 1):.3f}s/cache) | "
            f"distribute to GPUs: {total_route_time:.2f}s ({total_route_time / max(N, 1):.3f}s/cache)"
            f"{pinned_note}"
        )

        load_stats = {
            "cache_load_s": total_disk_load_time,
            "cache_route_s": total_route_time,
            "pinned_hits": _pin_hits,
            "pinned_misses": _pin_misses,
            "pinned_gb": _pin_held,
            "dynamic_batch_size": batch_size,
            "peaks": _peaks,
        }
        return result_answers, result_log_odds, load_stats

    def compute_text_only_qa_response(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
    ) -> Dict:
        """Synchronous wrapper for _run_text_only_qa."""
        _t0 = time.perf_counter()
        log_odds = asyncio.run(
            self._run_text_only_qa(
                questions=questions,
                contexts=contexts,
                boolean_question=boolean_question,
            )
        )
        return {
            "log_odds": log_odds,
            "stats": self._request_stats(
                path="direct",
                n_items=len(questions),
                elapsed_s=time.perf_counter() - _t0,
                vanilla=True,
            ),
        }

    async def _run_text_only_qa(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
    ) -> List[float]:
        """Text-only LLaVA inference — no images, no KV caches.

        For each (context, question) pair, builds a text prompt and runs generate().
        Returns log_odds as a list ordered identically to `questions`.
        """
        assert len(questions) == len(contexts)
        if boolean_question:
            instruction = (
                "Answer the following question with '1' or '0'. "
                "Do not add any other comments."
            )
        else:
            instruction = "Answer the following question. Do not add any other comments."
        answer_prefix = "Answer: "
        max_new_tokens = 4 if boolean_question else 64

        processor = AutoProcessor.from_pretrained(self.model_name)
        first_device = next(self.pipe.model.parameters()).device
        id0 = self.pipe.tokenizer.convert_tokens_to_ids("0")
        id1 = self.pipe.tokenizer.convert_tokens_to_ids("1")

        log_odds_out = []
        for context, question in zip(contexts, questions):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{context}\n{instruction}\n{question}"},
                    ],
                }
            ]
            prompt = (
                processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                + answer_prefix
            )

            inputs = processor(text=prompt, return_tensors="pt")
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

            # Text-only prompt: stale per-image rope_deltas must not leak in
            # (see _reset_mrope_state).
            self._reset_mrope_state()
            with torch.no_grad():
                generated = self.pipe.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            logits = generated.scores[0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            lo = (log_probs[0, id1] - log_probs[0, id0]).item()
            log_odds_out.append(lo)

            del inputs, generated, logits, log_probs

        # Once per request, not per item (see _run_kv_cache_multimodal for the rationale).
        torch.cuda.empty_cache()
        return log_odds_out

    async def _generate_cache_for_image(self, image, cache_filename, compression_ratio):
        """Generate and save multimodal KV cache for a single image."""
        assert self.pipe is not None, "Model pipeline is not initialized."
        # This single-CR path compresses inside the kvpress press hooks, whose 1D key
        # rerotation silently mis-rotates M-RoPE image-token keys. The multi-CR path
        # applies the M-RoPE-aware rerotation instead.
        assert self.mrope_component_map is None, (
            f"{self.model_name} uses interleaved M-RoPE: generate caches via the "
            "multi-CR path (prepare_caches_multi_cr / prepare_indices_relative), "
            "not the legacy single-CR press path."
        )

        try:
            # Standard context and answer format
            context = " "
            answer_prefix = "Answer: "

            # Generate cache
            inputs = self.pipe.preprocess(
                context=context,
                questions=[""],
                answer_prefix=answer_prefix,
                max_context_length=128000,
                image=image,
            )

            cache = DynamicCache()
            press = self.presses[compression_ratio]
            with torch.inference_mode():
                with (
                    press(self.pipe.model)
                    if press is not None
                    else contextlib.nullcontext()
                ):
                    _ = self.pipe._forward(inputs, press=press, cache=cache)

            # Save cache to disk — move to CPU in-place
            if hasattr(cache, "layers"):
                for layer in cache.layers:
                    layer.keys = layer.keys.detach().cpu()
                    layer.values = layer.values.detach().cpu()
            elif hasattr(cache, "_cache") and cache._cache:
                for item in cache._cache:
                    item.key_states = item.key_states.detach().cpu()
                    item.value_states = item.value_states.detach().cpu()
            else:
                for layer_idx in range(len(cache.key_cache)):
                    cache.key_cache[layer_idx] = cache.key_cache[layer_idx].detach().cpu()
                    cache.value_cache[layer_idx] = cache.value_cache[layer_idx].detach().cpu()

            logger.info(f"Saving cache to: {cache_filename}")

            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
            torch.save(cache, cache_filename)

            if not os.path.exists(cache_filename):
                logger.error(f"Cache file was not actually created: {cache_filename}")

            # Cleanup
            del cache
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error generating cache for item {cache_filename}: {str(e)}")
            raise


model_wrapper = None


class Status(Resource):
    def get(self):
        assert model_wrapper is not None
        return {
            "status": "alive",
            "model_name": model_wrapper.model_name,
            "compression_ratios": list(model_wrapper.compression_ratios),
            # See the same field on kv_cache_text_qa_server.Status: which half of the
            # USE_INDICES/--use-indexes pairing this server belongs to, so a worker
            # reusing it can refuse a mismatch before claiming a job.
            "use_relative_indices": bool(model_wrapper.use_relative_indices),
            # See the same field on kv_cache_text_qa_server.Status: the RAM budget for
            # caches an -in-memory operator pins. 0 means it cannot be served here.
            "kv_cache_pin_gb": PINNED_KV_STORE.budget_bytes / 1e9,
        }, 200


class PrepareCaches(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "image_paths": [
                "/path/to/image1.jpg",
                "/path/to/image2.jpg",
            ],
            "cache_dir": "/path/to/cache/dir"
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        img_paths = data["image_paths"]
        cache_dir = remap_cache_dir_to_local_mirror(data["cache_dir"])
        assert model_wrapper is not None
        _t0 = time.perf_counter()
        try:
            prep_result = asyncio.run(
                model_wrapper.prepare_caches(
                    column_name=column_name,
                    image_paths=img_paths,
                    cache_dir=cache_dir,
                    compression_ratio=data["effective_compression_ratio"],
                    materialized_compression_ratio=data[
                        "materialized_compression_ratio"
                    ],
                    vanilla=data["vanilla"],
                    keep_in_memory=data["keep_in_memory"],
                )
            )
        except PinnedKVUnavailable as e:
            # A configuration problem the operator must fix; returned as JSON so the
            # client can print it rather than a bare "Internal Server Error".
            logger.error(str(e))
            return {"status": "error", "error": str(e)}, 503
        return {
            "status": "cache_ready",
            **(prep_result or {}),
            "stats": model_wrapper._request_stats(
                path="prepare",
                n_items=len(img_paths),
                elapsed_s=time.perf_counter() - _t0,
                effective_compression_ratio=data["effective_compression_ratio"],
                materialized_compression_ratio=data["materialized_compression_ratio"],
                vanilla=data["vanilla"],
                keep_in_memory=data["keep_in_memory"],
                n_errors=(prep_result or {}).get("n_generation_errors", 0) or 0,
            ),
        }, 200


class ReleasePinnedKV(Resource):
    def post(self):
        """Drop every KV cache an -in-memory operator pinned here. Takes no body.

        See the same Resource on kv_cache_text_qa_server: called by a worker when its
        dataset changes, because pins are never evicted and would otherwise accumulate one
        column per dataset until a later /prepare_caches overflowed the budget.

        Deliberately does not read the request body: `requests.post(url)` with no json=
        sends an empty one, which `request.get_json(force=True)` — what every other handler
        in this file uses — would raise on.
        """
        return release_pinned_kv_response(), 200


class ImageQA(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "image_paths": [
                "/path/to/image1.jpg",
                "/path/to/image2.jpg",
            ],
            "question": "How many cats are there?",
            "effective_compression_ratio": 0.9,
            "materialized_compression_ratio": 0.5,
            "vanilla": false,
            "boolean": true,
            "cache_dir": "/path/to/cache/dir",

        }
        """
        data = request.get_json(force=True)
        img_paths = data["image_paths"]
        column_name = data["column_name"]
        question = data["question"]
        boolean_question = data["boolean"]
        cache_dir = remap_cache_dir_to_local_mirror(data["cache_dir"])
        assert model_wrapper is not None
        try:
            responses = model_wrapper.compute_image_qa_response(
                column_name=column_name,
                image_paths=img_paths,
                question=question,
                compression_ratio=data["effective_compression_ratio"],
                boolean_question=boolean_question,
                cache_dir=cache_dir,
                materialized_compression_ratio=data["materialized_compression_ratio"],
                vanilla=data["vanilla"],
                keep_in_memory=data["keep_in_memory"],
            )
        except PinnedKVUnavailable as e:
            # See the text server's TextQA route.
            logger.error(str(e))
            return {"status": "error", "error": str(e)}, 503
        return responses, 200


class ImageQAJoin(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "left_column_name": "artworks.image",
            "pairs": [
                {"left": "/path/to/a.jpg", "right": "/path/to/b.jpg"},
                ...
            ],
            "question": "Do both images show a painting by the same artist?",
            "effective_compression_ratio": 0.9,
            "materialized_compression_ratio": 0.9,
            "vanilla": false,
            "boolean": true,
            "cache_dir": "/path/to/cache/dir"
        }
        Returns:
        {
            "answers": {"left|right": "1", ...},
            "log_odds": {"left|right": 0.42, ...}
        }
        """
        data = request.get_json(force=True)
        left_column_name = data["left_column_name"]
        pairs = data["pairs"]
        question = data["question"]
        boolean_question = data["boolean"]
        cache_dir = remap_cache_dir_to_local_mirror(data["cache_dir"])
        assert model_wrapper is not None
        responses = model_wrapper.compute_image_qa_join_response(
            left_column_name=left_column_name,
            pairs=pairs,
            question=question,
            compression_ratio=data["effective_compression_ratio"],
            boolean_question=boolean_question,
            cache_dir=cache_dir,
            materialized_compression_ratio=data["materialized_compression_ratio"],
            vanilla=data["vanilla"],
            keep_in_memory=data["keep_in_memory"],
        )
        return responses, 200


class TextQADirect(Resource):
    def post(self):
        """
        Expects JSON:
        {
            "questions": ["Is it the same style as 'Impressionism'?", ...],
            "contexts": ["Cubism", ...],
            "boolean": true
        }
        Returns: {"log_odds": [0.42, ...]}
        """
        data = request.get_json(force=True)
        assert model_wrapper is not None
        responses = model_wrapper.compute_text_only_qa_response(
            questions=data["questions"],
            contexts=data["contexts"],
            boolean_question=data["boolean"],
        )
        return responses, 200


api.add_resource(Status, "/status")
api.add_resource(ImageQA, "/image_qa")
api.add_resource(ImageQAJoin, "/image_qa_join")
api.add_resource(PrepareCaches, "/prepare_caches")
api.add_resource(ReleasePinnedKV, "/release_pinned_kv")
api.add_resource(TextQADirect, "/text_qa_direct")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        choices=sorted(PORT_KV_VISION.keys()),
        help="Name of the model to use",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID for GPU to use",
    )
    parser.add_argument(
        "--vanilla-batch-size",
        type=int,
        default=None,
        help="Batch size for vanilla inference (client vanilla=True). Default: auto-estimate "
        "from prompt length (incl. vision tokens) × model config on the same memory budget "
        "as the cache methods. Pass a positive value to force a fixed batch size.",
    )
    parser.add_argument(
        "--use-relative-indices",
        action="store_true",
        help="Path A: reconstruct a target comp{tag} cache on the fly from a baseline cache "
        "+ relative indices when available; otherwise fall back to the physical cache, or "
        "skip just that image. Default off.",
    )
    parser.add_argument(
        "--kv-cache-pin-gb",
        type=float,
        default=None,
        help="RAM budget (GB) for KV caches an -in-memory operator pins at prepare(). "
        "Defaults to $KV_CACHE_PIN_GB, i.e. 0 = no -in-memory operator can be served "
        "here. Nothing is cached in RAM unless an operator asks for it by name, and "
        "nothing is ever evicted, so size this to hold the whole column.",
    )
    args = parser.parse_args()
    PINNED_KV_STORE.configure(args.kv_cache_pin_gb)
    device_id = args.device_id
    model_wrapper = KvImageQaModelWrapper(
        args.model_name,
        device_id,
        compression_ratios=(0.0, 0.3, 0.5, 0.6, 0.8, 0.9, 0.99),
        vanilla_batch_size=args.vanilla_batch_size,
        use_relative_indices=args.use_relative_indices,
    )
    app.run(host="127.0.0.1", port=PORT_KV_VISION.get(args.model_name), debug=False, threaded=False)