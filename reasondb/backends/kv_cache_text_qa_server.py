import argparse
import json
import yaml
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from flask import Flask, request
from flask_restful import Resource, Api
from typing import List, Optional


from reasondb.backends.kv_cache_base import (
    KVCachingBackendBase,
    PINNED_KV_STORE,
    PinnedKVUnavailable,
    pin_cache_inplace,
    release_pinned_kv_response,
    remap_cache_dir_to_local_mirror,
    validate_kv_compression_ratios,
)
from reasondb.backends.kv_cache_multi_cr import (
    compress_and_save_multi_cr,
    compress_and_save_relative,
)
from reasondb.backends.kv_cache_reconstruct import (
    H2D_NON_BLOCKING,
    RelativeReconstructError,
    gather_payload_cpu,
    migrate_legacy_index_dirs,
    rerotate_payload_sharded,
    resolve_relative_source,
)
from reasondb.backends.inference_stats import build_inference_stats, gpu_snapshot
from reasondb.backends.text_qa import PORT_KV_TEXT_QA
import torch
from kvpress import ExpectedAttentionPress, KeyRerotationPress, KVzipPress, FinchPress

from transformers import DynamicCache, pipeline  # type: ignore
import os
import contextlib

from reasondb.memory_footprint.memory_report import (
    compute_memory_footprints,
    update_compressed_cache_footprint,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _cache_num_layers(cache):
    """Layer count for a DynamicCache across transformers 5.x (.layers) and 4.x (.key_cache)."""
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache.key_cache)


def _cache_kv(cache, layer_idx):
    """Return (keys, values) for one layer across transformers 5.x (.layers) and 4.x (.key_cache)."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def _cache_set_kv(cache, layer_idx, keys, values):
    """Write (keys, values) for one layer across transformers 5.x (.layers) and 4.x (.key_cache)."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        layer.keys, layer.values = keys, values
    else:
        cache.key_cache[layer_idx] = keys
        cache.value_cache[layer_idx] = values


def _cache_seq_length(cache, layer_idx=0):
    """Sequence length for one layer across transformers 5.x (.layers) and 4.x (.key_cache).

    Avoids calling cache.get_seq_length() directly: on 5.x it reads len(self.layers),
    which raises AttributeError on caches deserialized from disk under the older
    key_cache/value_cache-based DynamicCache layout.
    """
    if hasattr(cache, "layers"):
        return cache.get_seq_length(layer_idx)
    if len(cache.key_cache) <= layer_idx or cache.key_cache[layer_idx] is None:
        return 0
    return cache.key_cache[layer_idx].shape[-2]


app = Flask(__name__)
api = Api(app)

from reasondb.config.model_registry import ModelRegistry as _ModelRegistry

_registry = _ModelRegistry.get()
MODEL_NAME = _registry.all_model_names()[0]

PRESS = {
    "expected_attention": lambda compression_ratio: ExpectedAttentionPress(
        compression_ratio=compression_ratio
    ),
    "kvzip": lambda compression_ratio: KVzipPress(compression_ratio=compression_ratio),
    "finch": lambda compression_ratio: FinchPress(compression_ratio=compression_ratio),
    "finch-cachenotes": lambda compression_ratio: FinchPress(
        compression_ratio=compression_ratio
    ),
}

CPT_PATH = {
    "movie": "reasondb/evaluation/benchmarks/files/reviews_1000.csv",
    "rotowire": "reasondb/evaluation/benchmarks/files/reports.csv",
    "email": "reasondb/evaluation/benchmarks/files/emails_with_cpt.csv",
}

# Maps the text column name (used as column_name at runtime) to (task key, CSV text column)
# column_name at runtime is the full dotted name like "reviews.reviewtext"
CPT_COLUMN_MAP = {
    "reviews.reviewtext": ("movie", "reviewtext"),
    "reviewtext": ("movie", "reviewtext"),
    "reports.report": ("rotowire", "report"),
    "report": ("rotowire", "report"),
    "emails.text": ("email", "text"),
    "text": ("email", "text"),
}


class KvTextQaModelWrapper(KVCachingBackendBase):
    def __init__(
        self,
        model_name,
        device_id: int,
        compression_ratios=(0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 0.99),
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
        self.vanilla_batch_size = vanilla_batch_size
        # Path A: when set, reconstruct a target compressed cache on the fly from a baseline
        # cache + relative indices (comp{base}/indices/comp{tag}/_meta.json) instead of
        # loading a physical comp{tag} cache. Default off.
        self.use_relative_indices = use_relative_indices
        self.init()

    def compute_text_qa_response(
        self,
        column_name: str,
        texts: List[str],
        questions: List[str],
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
        materialized_compression_ratio: float,
        vanilla: bool,
        keep_in_memory: bool = False,
    ):
        self._validate_client_crs(
            compression_ratio, materialized_compression_ratio, vanilla, keep_in_memory
        )
        return asyncio.run(
            self._run_kv_cache_text(
                column_name=column_name,
                texts=texts,
                all_questions=questions,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
                materialized_compression_ratio=materialized_compression_ratio,
                vanilla=vanilla,
                keep_in_memory=keep_in_memory,
            )
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

    def hash_text(self, text: str) -> str:
        """Generate a sha256hash for a given path."""
        import hashlib

        return hashlib.sha256(text.encode()).hexdigest()

    async def prepare_caches(
        self,
        column_name: str,
        texts: List[str],
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
            return {"n_texts": 0, "n_missing": 0, "missing_hashes": []}

        # Fail before any filesystem work: a server with no pin budget can never serve
        # this operator, and saying so in milliseconds beats saying so after a full scan.
        if keep_in_memory:
            PINNED_KV_STORE.require_configured(
                f"column {column_name!r} on {self.model_name} (cr={compression_ratio})"
            )

        # In relative-indices mode `compression_ratio` (the effective ratio) names the
        # index target dir, and the physical baseline lives at
        # materialized_compression_ratio; in physical mode materialized == effective, so
        # the effective ratio already names the cache dir. Either way the paths below key
        # off `compression_ratio`.
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"

        # Relative-index mode: the compressed target is reconstructed on the fly at serve
        # time from an offline baseline + relative indices (generate_kv_caches_indices.py),
        # so prepare must NOT prefill/generate physical caches. Verify the artifacts exist
        # and report back which are missing — the caller (PrepareCaches) surfaces this to
        # the client, which crashes loudly at setup instead of silently degrading per-text
        # at serve time. Refresh the footprint YAML (it sizes index-backed CRs
        # proportionally from the baseline). Layout is press-aware, mirrors the image server.
        if self.use_relative_indices:
            press_dir = f"{cache_dir}/{self.model_name}/{self.press_name}"
            target_tag = f"comp{self.to_compression_tag(compression_ratio)}"
            base_tag = f"comp{self.to_compression_tag(materialized_compression_ratio)}"

            # Index dirs live nested under their materialized baseline
            # ({press}/{base_tag}/indices/{target_tag}); relocate any legacy flat dirs
            # ({press}/indices/{target_tag}) into that layout once, so every reader below
            # only ever looks at the nested layout.
            migrate_legacy_index_dirs(press_dir)

            # A text can be absent for two different reasons: (a) nobody has run the
            # offline generator for it yet — a setup mistake the caller should crash on —
            # or (b) the offline generator (prepare_indices_relative / physical
            # prepare_caches) DID run and recorded a per-text failure in ERRORS.json (e.g.
            # a malformed text). (b) is a known, already-tolerated data issue, not a setup
            # mistake, so it must be reported as n_generation_errors, not n_missing —
            # mirroring physical mode below. Load both possible ERRORS.json locations: the
            # relative baseline dir (resolved via the target's _meta.json) and the target
            # dir itself (physical generation).
            known_errors = self._load_known_errors(press_dir, target_tag, base_tag)

            cache_basenames = []
            missing_hashes = []
            generation_error_hashes = []
            for text in texts:
                hash_name = self.hash_text(text)
                cache_basenames.append(f"cache_entry_{hash_name}.pt")
                try:
                    relative = resolve_relative_source(
                        press_dir, target_tag, hash_name, base_tag
                    )
                except RelativeReconstructError as e:
                    logger.warning(
                        f"[relative] _meta unusable for {hash_name[:12]}…: {e}"
                    )
                    relative = None
                physical = f"{press_dir}/{target_tag}/cache_entry_{hash_name}.pt"
                if relative is None and not os.path.exists(physical):
                    if hash_name in known_errors:
                        generation_error_hashes.append(hash_name)
                    else:
                        missing_hashes.append(hash_name)
            n_ready = len(texts) - len(missing_hashes) - len(generation_error_hashes)
            logger.info(
                f"[relative] CR {compression_ratio}: {n_ready}/{len(texts)} texts have "
                f"indices or physical caches under {press_dir}; no physical generation "
                f"performed."
            )
            if generation_error_hashes:
                logger.warning(
                    f"[relative] CR {compression_ratio}: {len(generation_error_hashes)}/"
                    f"{len(texts)} texts have a known generation error recorded during "
                    f"offline generation and will be skipped at serve time (sample: "
                    f"{generation_error_hashes[:20]})"
                )
            try:
                compute_memory_footprints(
                    cache_dir, column_name, cache_basenames, model_name=self.model_name
                )
            except Exception as e:
                logger.warning(
                    f"[relative] footprint accounting failed (non-fatal): {e}"
                )
            response = {
                "n_texts": len(texts),
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
                            for h in (self.hash_text(t) for t in texts)
                            if h not in unusable
                        ],
                        column_name=column_name,
                        compression_ratio=compression_ratio,
                    )
                )
            return response

        # Check if caches exist and generate them if not, one item at a time
        # Store mapping of row indices to cache files
        save_dir = f"{cache_dir}/{self.model_name}/{self.press_name}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        errors = {}
        cache_filenames = []

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc=f"Preparing caches {compression_ratio}",
        ):
            hash_name = self.hash_text(text)
            cache_filename = f"{save_dir}/cache_entry_{hash_name}.pt"
            cache_filenames.append(cache_filename.split("/")[-1])

            if os.path.exists(cache_filename):
                continue

            try:
                await self._generate_cache_for_text(
                    text, cache_filename, compression_ratio, column_name
                )
            except Exception as e:
                logger.warning(
                    f"Error processing text {i} with hash {hash_name}: {str(e)}",
                )
                errors[cache_filename] = str(e)

        compute_memory_footprints(
            cache_dir, column_name, cache_filenames, model_name=self.model_name
        )
        update_compressed_cache_footprint(
            cache_path=cache_dir,
            compression_ratio=compression_ratio,
            cache_filenames=cache_filenames,
            model_name=self.model_name,
            column_name=column_name,
            press_name=self.press_name,
        )
        with open(
            f"{save_dir}/ERRORS.json",
            "w",
        ) as f:
            json.dump(errors, f, indent=4)

        # Per-text generation failures (e.g. a malformed/corrupt text) are an expected,
        # already-tolerated data issue, NOT a "forgot to pregenerate" setup mistake: the
        # serve path (_run_kv_cache_text) skips these per-text using the recorded error and
        # keeps answering the rest of the batch. So this must NOT surface as "n_missing"
        # (the client only crashes on that key) — report it under a distinct, non-fatal key
        # purely for visibility.
        response = {
            "n_texts": len(texts),
            "n_generation_errors": len(errors),
            "generation_error_hashes": [os.path.basename(p) for p in list(errors)[:20]],
        }
        if keep_in_memory:
            # A text whose generation just failed, or failed on an earlier run, has no
            # file — `errors` and the existence check both exclude it, so the serve path
            # degrades exactly as it does for a disk-served operator.
            response.update(
                self._pin_prepared_caches(
                    [
                        path
                        for path in (
                            f"{save_dir}/cache_entry_{self.hash_text(t)}.pt"
                            for t in texts
                        )
                        if path not in errors and os.path.exists(path)
                    ],
                    column_name=column_name,
                    compression_ratio=compression_ratio,
                )
            )
        return response

    async def prepare_caches_multi_cr(
        self,
        column_name: str,
        texts: List[str],
        cr_to_cache_dir: dict,
        save_indices: bool = False,
    ):
        """
        Like prepare_caches, but generates all compression ratios in a SINGLE prefill
        pass per text. Scores are computed once and top-k selection is applied per CR.

        Parameters
        ----------
        cr_to_cache_dir : dict[float, str]
            Mapping from compression ratio to its base cache directory.
            Each CR may have its own directory (e.g. one per method name).
        save_indices : bool
            If True, also save the per-CR kept-token indices (from the same prefill) as
            indices/comp{tag}/idx_{hash}.pt, enabling bit-exact masked reconstruction.
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."
        compression_ratios = list(cr_to_cache_dir.keys())
        for cr in compression_ratios:
            assert (
                cr in self.compression_ratios
            ), f"Compression ratio {cr} not in supported ratios: {self.compression_ratios}"

        # Set up per-CR save directories (one per CR, each under its own cache_dir)
        save_dirs: dict[float, str] = {}
        for cr, cache_dir in cr_to_cache_dir.items():
            save_dir = (
                f"{cache_dir}/{self.model_name}/{self.press_name}"
                f"/comp{self.to_compression_tag(cr)}"
            )
            os.makedirs(save_dir, exist_ok=True)
            save_dirs[cr] = save_dir

        errors: dict[str, str] = {}
        all_cache_filenames: dict[float, list[str]] = {
            cr: [] for cr in compression_ratios
        }

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc="Preparing multi-CR caches",
        ):
            hash_name = self.hash_text(text)
            is_finch = self.press_name in ("finch", "finch-cachenotes")

            # Determine which CRs (and, when save_indices, which index files) are missing
            cache_filenames_by_cr: dict[float, str] = {}
            missing_crs: list[float] = []
            missing_idx = False
            for cr in compression_ratios:
                cache_filename = f"{save_dirs[cr]}/cache_entry_{hash_name}.pt"
                all_cache_filenames[cr].append(cache_filename.split("/")[-1])
                cache_filenames_by_cr[cr] = cache_filename
                if not os.path.exists(cache_filename):
                    missing_crs.append(cr)
                elif save_indices and cr > 0.0 and not is_finch:
                    idx_path = os.path.join(
                        os.path.dirname(save_dirs[cr]),
                        "indices",
                        os.path.basename(save_dirs[cr]),
                        f"idx_{hash_name}.pt",
                    )
                    if not os.path.exists(idx_path):
                        missing_idx = True

            if not missing_crs and not missing_idx:
                continue

            # With save_indices, regenerate ALL CRs together so every cache and its index
            # come from one prefill (bit-exact); otherwise only the missing CRs.
            crs_to_generate = compression_ratios if save_indices else missing_crs

            try:
                await self._generate_caches_for_text_multi_cr(
                    text_content=text,
                    cache_filenames_by_cr={
                        cr: cache_filenames_by_cr[cr] for cr in crs_to_generate
                    },
                    compression_ratios=crs_to_generate,
                    column_name=column_name,
                    save_indices=save_indices,
                )
            except Exception as e:
                logger.warning(
                    f"Error processing text {i} with hash {hash_name}: {str(e)}",
                )
                errors[hash_name] = str(e)

        # Write error logs and memory footprints for each CR
        for cr in compression_ratios:
            save_dir = save_dirs[cr]
            cache_dir = cr_to_cache_dir[cr]
            with open(f"{save_dir}/ERRORS.json", "w") as f:
                json.dump(errors, f, indent=4)
            compute_memory_footprints(
                cache_dir,
                column_name,
                all_cache_filenames[cr],
                model_name=self.model_name,
            )
            update_compressed_cache_footprint(
                cache_path=cache_dir,
                compression_ratio=cr,
                cache_filenames=all_cache_filenames[cr],
                model_name=self.model_name,
                column_name=column_name,
                press_name=self.press_name,
            )

    async def generate_scores_and_masks_multi_cr(
        self,
        column_name: str,
        texts: List[str],
        masks_dir_by_cr: dict,
        scores_dir: str | None = None,
    ):
        """
        Like prepare_caches_multi_cr, but saves binary keep-masks instead of
        compressed KV caches.

        Parameters
        ----------
        masks_dir_by_cr : dict[float, str]
            Mapping from compression ratio to the directory for that CR's masks.
            Each file: mask_{hash}.pt  →  dict[layer_idx, Tensor(1, n_heads, seq_len)] bool (True=keep)
        scores_dir : str | None
            Directory where per-text score files are saved. If None, scores are
            computed in-memory to build masks but never written to disk.
            Each file: scores_{hash}.pt  →  dict[layer_idx, Tensor(1, n_heads, seq_len)]
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."
        compression_ratios = list(masks_dir_by_cr.keys())

        if scores_dir is not None:
            os.makedirs(scores_dir, exist_ok=True)
        for cr, masks_dir in masks_dir_by_cr.items():
            os.makedirs(masks_dir, exist_ok=True)

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc="Generating scores and masks",
        ):
            hash_name = self.hash_text(text)
            scores_file = (
                f"{scores_dir}/scores_{hash_name}.pt"
                if scores_dir is not None
                else None
            )
            scores_missing = scores_file is not None and not os.path.exists(scores_file)

            missing_mask_crs = [
                cr
                for cr in compression_ratios
                if not os.path.exists(f"{masks_dir_by_cr[cr]}/mask_{hash_name}.pt")
            ]
            if not scores_missing and not missing_mask_crs:
                continue

            try:
                await self._generate_scores_and_masks_for_text(
                    text_content=text,
                    hash_name=hash_name,
                    column_name=column_name,
                    compression_ratios=compression_ratios
                    if scores_missing
                    else missing_mask_crs,
                    scores_dir=scores_dir,
                    masks_dir_by_cr=masks_dir_by_cr,
                    scores_file=scores_file,
                    missing_mask_crs=missing_mask_crs,
                )
            except Exception as e:
                logger.warning(
                    f"Error processing text {i} (hash={hash_name}): {e}", exc_info=True
                )

    async def _generate_scores_and_masks_for_text(
        self,
        text_content: str,
        hash_name: str,
        column_name: str,
        compression_ratios: list,
        scores_dir: str,
        masks_dir_by_cr: dict,
        scores_file: str,
        missing_mask_crs: list,
    ):
        """Run one prefill, save scores once, and save per-CR binary masks."""
        assert self.pipe is not None

        is_finch = self.press_name in ("finch", "finch-cachenotes")
        uses_rerotation = self.press_name == "expected_attention"

        try:
            answer_prefix = "Answer: "
            window_text = None

            # ---- 1. Prepare context (mirrors _generate_caches_for_text_multi_cr) ----
            if self.press_name == "finch":
                yaml_path = "queries_workloads/_workload.yaml"
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)
                query_list = data["queries"]
                queries_workload = (
                    "Pay attention to these examples of questions:\n"
                    + "\n".join(f"- {q}" for q in query_list)
                )
                sample_press = next(
                    p for cr, p in self.presses.items() if p is not None
                )
                context = text_content[: min(128000, len(text_content))]
                context_aware = (
                    context + sample_press.delimiter_token + queries_workload
                )
                window_text = queries_workload

            elif self.press_name == "finch-cachenotes":
                import re
                import hashlib as _hashlib
                import pandas as _pd

                def _normalize(t):
                    t = t.strip()
                    t = re.sub(r"[ \t]+", " ", t)
                    t = re.sub(r"\n\s*\n", "\n\n", t)
                    t = t.replace("\r\n", "\n").replace("\r", "\n")
                    t = t.replace("​", "")
                    return t

                assert column_name is not None
                assert (
                    column_name in CPT_COLUMN_MAP
                ), f"Unknown column_name '{column_name}' for finch-cachenotes."
                task_key, text_col = CPT_COLUMN_MAP[column_name]
                cpt_path = CPT_PATH[task_key]
                df = _pd.read_csv(cpt_path)
                norm_content = _normalize(text_content)
                text_hash = _hashlib.sha256(text_content.encode()).hexdigest()
                matching_row = None
                for _, row in df.iterrows():
                    row_text = str(row.get(text_col, ""))
                    if (
                        row_text == text_content
                        or _normalize(row_text) == norm_content
                        or _hashlib.sha256(row_text.encode()).hexdigest() == text_hash
                    ):
                        matching_row = row
                        break
                sample_press = next(
                    p for cr, p in self.presses.items() if p is not None
                )
                if matching_row is not None:
                    cpt = str(matching_row["cpt"])
                else:
                    logger.warning(
                        f"No matching CPT found for column '{column_name}' — using fallback."
                    )
                    cpt = "Your task is to answer questions based on the context."
                context = text_content[: min(128000, len(text_content))]
                context_aware = context + sample_press.delimiter_token + cpt
                window_text = cpt

            else:
                context_aware = text_content[: min(128000, len(text_content))]

            # ---- 2. Tokenize ----
            inputs = self.pipe.preprocess(
                context=context_aware,
                questions=[""],
                answer_prefix=answer_prefix,
                max_context_length=128000,
            )
            context_ids = inputs["context_ids"]
            actual_context_ids_length = context_ids.shape[1]

            if is_finch and window_text is not None:
                window_tokens_count = len(
                    self.tokenizer.encode(window_text, add_special_tokens=False)
                )
                context_tokens_count = (
                    actual_context_ids_length - window_tokens_count - 1
                )
            else:
                window_tokens_count = 0
                context_tokens_count = actual_context_ids_length

            # ---- 3. Build score-capturing hook ----
            has_nonzero_crs = any(cr > 0.0 for cr in compression_ratios)
            scoring_press = None
            if has_nonzero_crs:
                if is_finch:
                    scoring_press = FinchPress(compression_ratio=0.0)
                    scoring_press.update_model_and_tokenizer(
                        self.pipe.model, self.pipe.tokenizer
                    )
                elif uses_rerotation:
                    any_nonzero_cr = next(
                        cr for cr in self.compression_ratios if cr > 0.0
                    )
                    scoring_press = self.presses[any_nonzero_cr].press
                else:
                    any_nonzero_cr = next(
                        cr for cr in self.compression_ratios if cr > 0.0
                    )
                    scoring_press = self.presses[any_nonzero_cr]

            layer_scores: dict[int, torch.Tensor] = {}
            layer_modules: dict[int, object] = {}

            def capturing_forward_hook(module, input, kwargs, output):
                if scoring_press is None:
                    return output
                hidden_states = kwargs["hidden_states"]
                cache = kwargs.get("past_key_value") or kwargs.get("past_key_values")
                q_len = hidden_states.shape[1]
                if kwargs["cache_position"][-1] > q_len:
                    return output
                keys, values = _cache_kv(cache, module.layer_idx)
                scores = scoring_press.score(
                    module, hidden_states, keys, values, output[1], kwargs
                )
                layer_scores[module.layer_idx] = scores.detach().cpu()
                layer_modules[module.layer_idx] = module
                return output

            # ---- 4. Run a single prefill ----
            first_device = next(self.pipe.model.parameters()).device
            context_ids = context_ids.to(first_device)
            full_cache = DynamicCache()

            for layer in self.pipe.model.model.layers:
                layer.self_attn.rotary_emb = self.pipe.model.model.rotary_emb

            hooks = [
                layer.self_attn.register_forward_hook(
                    capturing_forward_hook, with_kwargs=True
                )
                for layer in self.pipe.model.model.layers
            ]
            embed_hook = None
            if is_finch and scoring_press is not None:
                embed_hook = self.pipe.model.model.embed_tokens.register_forward_hook(
                    scoring_press.embed_token_forward_hook
                )

            try:
                with torch.inference_mode():
                    self.pipe.model.model(
                        input_ids=context_ids,
                        past_key_values=full_cache,
                        use_cache=True,
                        output_attentions=False,
                    )
            finally:
                for hook in hooks:
                    hook.remove()
                if embed_hook is not None:
                    embed_hook.remove()

            window_size = 0
            if is_finch:
                window_size = scoring_press.window_size
                assert window_size is not None and window_size > 0

            # Capture shape info before freeing the cache
            num_layers = _cache_num_layers(full_cache)
            full_seq_len = (
                _cache_kv(full_cache, 0)[0].shape[2]
                if num_layers > 0
                else actual_context_ids_length
            )
            del full_cache
            torch.cuda.empty_cache()

            # ---- 5. Save scores (once, CR-independent) ----
            if (
                scores_file is not None
                and not os.path.exists(scores_file)
                and layer_scores
            ):
                torch.save(layer_scores, scores_file)
                logger.info(f"Saved scores → {scores_file}")

            # ---- 6. Compute and save per-CR binary masks ----
            for cr in missing_mask_crs:
                mask_file = f"{masks_dir_by_cr[cr]}/mask_{hash_name}.pt"
                if os.path.exists(mask_file):
                    continue

                layer_masks: dict[int, torch.Tensor] = {}

                for layer_idx in range(num_layers):
                    if layer_idx in layer_scores:
                        scores = layer_scores[layer_idx]  # (1, n_heads, seq_len)
                        seq_len = scores.shape[-1]
                        n_heads = scores.shape[1]
                    else:
                        # CR=0.0 only run — scores not captured
                        seq_len = full_seq_len
                        n_heads = self.pipe.model.config.num_key_value_heads
                        scores = None

                    if cr == 0.0:
                        mask = torch.ones((1, n_heads, seq_len), dtype=torch.bool)
                        if is_finch and window_size > 0:
                            # cr=0 for Finch trims window tokens from full cache
                            mask[:, :, -window_size:] = False
                    else:
                        total_len = seq_len
                        if is_finch:
                            n_kept_context = int(context_tokens_count * (1 - cr))
                            n_kept_total = min(n_kept_context + window_size, total_len)
                        else:
                            n_kept_total = int(total_len * (1 - cr))

                        indices = scores.topk(n_kept_total, dim=-1).indices
                        mask = torch.zeros(scores.shape, dtype=torch.bool)
                        mask.scatter_(
                            -1, indices, torch.ones_like(indices, dtype=torch.bool)
                        )

                    layer_masks[layer_idx] = mask

                # Save mask + metadata sidecar
                torch.save(layer_masks, mask_file)
                meta = {
                    "compression_ratio": cr,
                    "seq_len": full_seq_len,
                    "window_size": window_size,
                    "context_tokens_count": context_tokens_count,
                    "press_name": self.press_name,
                    "model_name": self.model_name,
                }
                meta_file = mask_file.replace(".pt", "_meta.json")
                with open(meta_file, "w") as f:
                    json.dump(meta, f)
                logger.info(f"Saved mask CR={cr} → {mask_file}")

        except StopIteration as e:
            logger.error("StopIteration escaped coroutine", exc_info=True)
            raise RuntimeError("StopIteration escaped coroutine") from e
        except Exception as e:
            logger.error(f"Error in score/mask generation: {str(e)}", exc_info=True)
            raise

    async def _generate_caches_for_text_multi_cr(
        self,
        text_content: str,
        cache_filenames_by_cr: dict,
        compression_ratios: list,
        column_name: str = None,
        save_indices: bool = False,
    ):
        """
        Run ONE prefill pass and save compressed caches for multiple CRs.

        If `save_indices` is True, the per-CR kept-token indices (the same prefill's
        topk selection that produced the compressed cache) are also written next to the
        cache as indices/comp{tag}/idx_{hash}.pt — a single (n_layers, n_heads, n_kept)
        int tensor. Because they come from the SAME prefill as the cache, the masked
        reconstruction is bit-exact (no cross-run topk divergence). Only for cr > 0 and
        non-Finch presses.

        Strategy:
          1. Prepare context (same logic as _generate_cache_for_text).
          2. Run prefill with a score-capturing hook — no compression applied,
             full uncompressed KV cache is kept in memory.
          3. For each CR: apply topk(n_kept) on the captured scores + key
             rerotation (when applicable), then save the compressed cache.

        This avoids N separate prefill passes for N compression ratios.
        """
        assert self.pipe is not None, "Model pipeline is not initialized."

        is_finch = self.press_name in ("finch", "finch-cachenotes")
        uses_rerotation = (
            self.press_name == "expected_attention"
        )  # kvzip: no rerotation

        try:
            answer_prefix = "Answer: "
            window_text = None  # set only for Finch

            # ---- 1. Prepare context (mirrors _generate_cache_for_text) ----
            if self.press_name == "finch":
                yaml_path = "queries_workloads/_workload.yaml"
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)
                query_list = data["queries"]
                queries_workload = (
                    "Pay attention to these examples of questions:\n"
                    + "\n".join(f"- {q}" for q in query_list)
                )
                sample_press = next(
                    p for cr, p in self.presses.items() if p is not None
                )
                context = text_content[: min(128000, len(text_content))]
                context_aware = (
                    context + sample_press.delimiter_token + queries_workload
                )
                window_text = queries_workload

            elif self.press_name == "finch-cachenotes":
                import re
                import hashlib as _hashlib
                import pandas as _pd

                def _normalize(t):
                    t = t.strip()
                    t = re.sub(r"[ \t]+", " ", t)
                    t = re.sub(r"\n\s*\n", "\n\n", t)
                    t = t.replace("\r\n", "\n").replace("\r", "\n")
                    t = t.replace("\u200b", "")
                    return t

                assert column_name is not None
                assert (
                    column_name in CPT_COLUMN_MAP
                ), f"Unknown column_name '{column_name}' for finch-cachenotes."
                task_key, text_col = CPT_COLUMN_MAP[column_name]
                cpt_path = CPT_PATH[task_key]
                df = _pd.read_csv(cpt_path)
                norm_content = _normalize(text_content)
                text_hash = _hashlib.sha256(text_content.encode()).hexdigest()

                matching_row = None
                for _, row in df.iterrows():
                    row_text = str(row.get(text_col, ""))
                    if (
                        row_text == text_content
                        or _normalize(row_text) == norm_content
                        or _hashlib.sha256(row_text.encode()).hexdigest() == text_hash
                    ):
                        matching_row = row
                        break

                sample_press = next(
                    p for cr, p in self.presses.items() if p is not None
                )
                if matching_row is not None:
                    cpt = str(matching_row["cpt"])
                else:
                    logger.warning(
                        f"No matching CPT found for column '{column_name}' — using fallback."
                    )
                    cpt = "Your task is to answer questions based on the context."
                context = text_content[: min(128000, len(text_content))]
                context_aware = context + sample_press.delimiter_token + cpt
                window_text = cpt

            else:
                context_aware = text_content[: min(128000, len(text_content))]

            # ---- 2. Tokenize ----
            inputs = self.pipe.preprocess(
                context=context_aware,
                questions=[""],
                answer_prefix=answer_prefix,
                max_context_length=128000,
            )
            context_ids = inputs["context_ids"]
            actual_context_ids_length = context_ids.shape[1]

            # For Finch: compute token counts needed for per-CR n_kept calculation
            if is_finch and window_text is not None:
                window_tokens_count = len(
                    self.tokenizer.encode(window_text, add_special_tokens=False)
                )
                context_tokens_count = (
                    actual_context_ids_length - window_tokens_count - 1
                )
            else:
                window_tokens_count = 0
                context_tokens_count = actual_context_ids_length

            # ---- 3. Build a score-capturing hook (only needed when compressing) ----
            has_nonzero_crs = any(cr > 0.0 for cr in compression_ratios)
            scoring_press = None
            if has_nonzero_crs:
                if is_finch:
                    scoring_press = FinchPress(compression_ratio=0.0)
                    scoring_press.update_model_and_tokenizer(
                        self.pipe.model, self.pipe.tokenizer
                    )
                elif uses_rerotation:
                    any_nonzero_cr = next(
                        cr for cr in self.compression_ratios if cr > 0.0
                    )
                    scoring_press = self.presses[any_nonzero_cr].press
                else:
                    any_nonzero_cr = next(
                        cr for cr in self.compression_ratios if cr > 0.0
                    )
                    scoring_press = self.presses[any_nonzero_cr]

            layer_scores: dict[int, torch.Tensor] = {}
            layer_modules: dict[int, object] = {}

            def capturing_forward_hook(module, input, kwargs, output):
                if scoring_press is None:
                    return output
                hidden_states = kwargs["hidden_states"]
                cache = kwargs.get("past_key_value") or kwargs.get("past_key_values")
                q_len = hidden_states.shape[1]
                # Only during prefill
                if kwargs["cache_position"][-1] > q_len:
                    return output
                keys, values = _cache_kv(cache, module.layer_idx)
                # output[1] is attention weights (None when output_attentions=False)
                scores = scoring_press.score(
                    module, hidden_states, keys, values, output[1], kwargs
                )
                layer_scores[module.layer_idx] = scores.detach().cpu()
                layer_modules[module.layer_idx] = module
                # Return output unchanged — keep full uncompressed cache
                return output

            # ---- 4. Run a single prefill ----
            first_device = next(self.pipe.model.parameters()).device
            context_ids = context_ids.to(first_device)
            full_cache = DynamicCache()

            # Set rotary embeddings (required by scoring methods)
            for layer in self.pipe.model.model.layers:
                layer.self_attn.rotary_emb = self.pipe.model.model.rotary_emb

            hooks = [
                layer.self_attn.register_forward_hook(
                    capturing_forward_hook, with_kwargs=True
                )
                for layer in self.pipe.model.model.layers
            ]
            embed_hook = None
            if is_finch and scoring_press is not None:
                embed_hook = self.pipe.model.model.embed_tokens.register_forward_hook(
                    scoring_press.embed_token_forward_hook
                )

            try:
                logger.info(
                    f"Single prefill for {actual_context_ids_length} tokens "
                    f"→ generating {len(compression_ratios)} cache(s)"
                )
                with torch.inference_mode():
                    self.pipe.model.model(
                        input_ids=context_ids,
                        past_key_values=full_cache,
                        use_cache=True,
                        output_attentions=False,
                    )
            finally:
                for hook in hooks:
                    hook.remove()
                if embed_hook is not None:
                    embed_hook.remove()

            # Capture Finch window_size (set by embed_token_hook during prefill)
            window_size = 0
            if is_finch:
                window_size = scoring_press.window_size
                assert (
                    window_size is not None and window_size > 0
                ), "Finch window_size was not detected during prefill."
                logger.info(f"Finch window_size = {window_size}")

            # ---- 5. Apply per-CR compression and save (shared with image server) ----
            compress_and_save_multi_cr(
                full_cache=full_cache,
                layer_scores=layer_scores,
                layer_modules=layer_modules,
                compression_ratios=compression_ratios,
                cache_filenames_by_cr=cache_filenames_by_cr,
                first_device=first_device,
                save_indices=save_indices,
                is_finch=is_finch,
                uses_rerotation=uses_rerotation,
                window_size=window_size,
                context_tokens_count=context_tokens_count,
            )

            del full_cache
            torch.cuda.empty_cache()

        except StopIteration as e:
            logger.error("StopIteration escaped coroutine", exc_info=True)
            raise RuntimeError("StopIteration escaped coroutine") from e
        except Exception as e:
            logger.error(f"Error in multi-CR cache generation: {str(e)}", exc_info=True)
            raise

    async def _prefill_and_score(self, text_content: str):
        """Run ONE expected_attention prefill and capture the full cache + per-layer scores.

        Slimmed version of `_generate_caches_for_text_multi_cr`'s prefill seam, scoped to
        expected_attention (no Finch window / embed hook). Returns
        (full_cache, layer_scores, layer_modules, first_device). Used by the relative
        (hierarchical) index generation.
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert (
            self.press_name == "expected_attention"
        ), "Relative index generation supports only expected_attention."
        context_aware = text_content[: min(128000, len(text_content))]
        inputs = self.pipe.preprocess(
            context=context_aware,
            questions=[""],
            answer_prefix="Answer: ",
            max_context_length=128000,
        )
        context_ids = inputs["context_ids"]

        # scoring press = the inner ScorerPress of the rerotation wrapper (same as multi_cr)
        any_nonzero_cr = next(cr for cr in self.compression_ratios if cr > 0.0)
        scoring_press = self.presses[any_nonzero_cr].press

        layer_scores: dict[int, torch.Tensor] = {}
        layer_modules: dict[int, object] = {}

        def capturing_forward_hook(module, input, kwargs, output):
            hidden_states = kwargs["hidden_states"]
            cache = kwargs.get("past_key_value") or kwargs.get("past_key_values")
            q_len = hidden_states.shape[1]
            if kwargs["cache_position"][-1] > q_len:  # prefill only
                return output
            keys, values = _cache_kv(cache, module.layer_idx)
            scores = scoring_press.score(
                module, hidden_states, keys, values, output[1], kwargs
            )
            layer_scores[module.layer_idx] = scores.detach().cpu()
            layer_modules[module.layer_idx] = module
            return output

        first_device = next(self.pipe.model.parameters()).device
        context_ids = context_ids.to(first_device)
        full_cache = DynamicCache()
        for layer in self.pipe.model.model.layers:
            layer.self_attn.rotary_emb = self.pipe.model.model.rotary_emb
        hooks = [
            layer.self_attn.register_forward_hook(
                capturing_forward_hook, with_kwargs=True
            )
            for layer in self.pipe.model.model.layers
        ]
        try:
            with torch.inference_mode():
                self.pipe.model.model(
                    input_ids=context_ids,
                    past_key_values=full_cache,
                    use_cache=True,
                    output_attentions=False,
                )
        finally:
            for hook in hooks:
                hook.remove()
        return full_cache, layer_scores, layer_modules, first_device

    async def _generate_relative_for_text(
        self,
        text_content: str,
        base_cache_filename: str,
        rel_idx_filenames_by_cr: dict,
        base_cr: float,
        index_crs: list,
    ):
        """One prefill → save the physical baseline cache + relative indices for each target."""
        (
            full_cache,
            layer_scores,
            layer_modules,
            first_device,
        ) = await self._prefill_and_score(text_content)
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
            )
        finally:
            del full_cache
            torch.cuda.empty_cache()

    async def prepare_indices_relative(
        self,
        column_name: str,
        texts: List[str],
        cache_dir: str,
        base_cr: float,
        index_crs: list,
    ):
        """Hierarchical generation: store ONE physical baseline cache at `base_cr` and, for
        each target in `index_crs` (strictly more compressed), only indices RELATIVE to the
        baseline. The full kv8B00 is never stored.

        Layout (under {cache_dir}/{model}/{press}/):
          comp{base_tag}/cache_entry_{hash}.pt              — physical baseline cache
          comp{base_tag}/indices/comp{tgt_tag}/idx_{hash}.pt — indices into the baseline
          comp{base_tag}/indices/comp{tgt_tag}/_meta.json    — {"from": "comp{base_tag}"}
        Nesting the index dirs under their baseline lets several baselines index the same
        effective target at once (any legacy flat indices/comp{tgt_tag}/ is migrated into
        this layout at startup by migrate_legacy_index_dirs).
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."
        assert (
            self.press_name == "expected_attention"
        ), "prepare_indices_relative supports only expected_attention."
        for cr in (base_cr, *index_crs):
            assert (
                cr in self.compression_ratios
            ), f"Compression ratio {cr} not in supported ratios: {self.compression_ratios}"
        for cr in index_crs:
            assert (
                cr > base_cr
            ), f"index method cr {cr} must be strictly more compressed than baseline cr {base_cr}"

        press_dir = f"{cache_dir}/{self.model_name}/{self.press_name}"
        base_dir = f"{press_dir}/comp{self.to_compression_tag(base_cr)}"
        # Index dirs nest under their materialized baseline so several baselines can index
        # the same effective target without colliding: {base_dir}/indices/comp{tgt_tag}.
        idx_dirs = {
            cr: f"{base_dir}/indices/comp{self.to_compression_tag(cr)}"
            for cr in index_crs
        }
        os.makedirs(base_dir, exist_ok=True)
        for d in idx_dirs.values():
            os.makedirs(d, exist_ok=True)

        errors: dict[str, str] = {}
        base_cache_basenames: list[str] = []

        for i, text in tqdm(
            enumerate(texts), total=len(texts), desc="Preparing relative indices"
        ):
            hash_name = self.hash_text(text)
            base_cache_filename = f"{base_dir}/cache_entry_{hash_name}.pt"
            base_cache_basenames.append(f"cache_entry_{hash_name}.pt")
            rel_idx_filenames_by_cr = {
                cr: f"{idx_dirs[cr]}/idx_{hash_name}.pt" for cr in index_crs
            }

            # Regenerate iff the baseline cache OR any target's relative-idx file is missing.
            # When regenerating, baseline + all relative indices come from ONE prefill
            # (shared scores → bit-exact subset selection).
            missing = not os.path.exists(base_cache_filename) or any(
                not os.path.exists(p) for p in rel_idx_filenames_by_cr.values()
            )
            if not missing:
                continue

            try:
                await self._generate_relative_for_text(
                    text_content=text,
                    base_cache_filename=base_cache_filename,
                    rel_idx_filenames_by_cr=rel_idx_filenames_by_cr,
                    base_cr=base_cr,
                    index_crs=index_crs,
                )
            except Exception as e:
                logger.warning(
                    f"Error processing text {i} with hash {hash_name}: {str(e)}"
                )
                errors[hash_name] = str(e)

        with open(f"{base_dir}/ERRORS.json", "w") as f:
            json.dump(errors, f, indent=4)

        # Footprint accounting for the single physical baseline cache (best-effort).
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
            return float(name[len("comp") :].replace("_", "."))
        except ValueError:
            return None

    def init(self):
        """Initialize the text model pipeline and compression settings."""
        logger.info("Setting up KV Cache Text Filter...")

        # Set up device
        self.device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"

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
                    f"Error initializing model with args {x}: {str(type(e))}({str(e)})",
                    exc_info=True,
                )

        assert self.pipe is not None, "Failed to initialize the model pipeline."
        self.pipe.model.eval()
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer.pad_token = self.pipe.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Set up compression press
        if self.press_name not in PRESS:
            raise ValueError(
                f"Unknown press_name '{self.press_name}'. Available options: {list(PRESS.keys())}"
            )
        if self.press_name in ("finch", "kvzip", "finch-cachenotes"):
            self.presses = {
                cr: PRESS[self.press_name](compression_ratio=cr) if cr > 0.0 else None
                for cr in self.compression_ratios
            }
        else:
            self.presses = {
                cr: KeyRerotationPress(PRESS[self.press_name](compression_ratio=cr))
                if cr > 0.0
                else None
                for cr in self.compression_ratios
            }

        # Initialize Finch presses with model and tokenizer if using Finch
        if self.press_name in ("finch", "finch-cachenotes"):
            for cr, press in self.presses.items():
                if press is not None and hasattr(press, "update_model_and_tokenizer"):
                    press.update_model_and_tokenizer(
                        self.pipe.model, self.pipe.tokenizer
                    )
                    logger.info(
                        f"Initialized Finch press for CR={cr} with delimiter token"
                    )

        logger.info(
            f"Using press {self.press_name} with compression_ratios {self.compression_ratios}",
        )
        logger.info(
            f"Model {self.model_name} loaded on {next(self.pipe.model.parameters()).device}",
        )

    async def _run_vanilla_inference(
        self,
        texts: List[str],
        all_questions: List[str],
        boolean_question: bool,
    ):
        """Run vanilla inference without KV caching."""
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."
        # Every response carries an inference-statistics block; see
        # reasondb/backends/inference_stats.py for the contract.
        _stats_t0 = time.perf_counter()
        _peaks: dict = {}

        # Mirror the instruction prefix applied in _run_kv_cache_text (which adds it
        # after the early-return that routes here, so we must add it ourselves).
        if boolean_question:
            instruction = "Answer the following question based on the context with '1' or '0'. Do not add any other comments."
        else:
            instruction = "Answer the following question based on the context. Do not add any other comments."
        all_questions = [instruction + " " + q for q in all_questions]

        answer_prefix = "Answer: "
        max_new_tokens = 4 if boolean_question else 64
        id0 = self.tokenizer.convert_tokens_to_ids("0")
        id1 = self.tokenizer.convert_tokens_to_ids("1")
        first_device = next(self.pipe.model.parameters()).device

        # Compute question suffix once (it does not vary per row)
        if self.tokenizer.chat_template is None:
            question_suffix = "\n"
        else:
            template_context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "Example of context\n###"}],
                add_generation_prompt=True,
                tokenize=False,
            )
            _, question_suffix = template_context.split("\n###", 1)

        full_prompts = [
            text + "\n" + question + question_suffix + answer_prefix
            for text, question in zip(texts, all_questions)
        ]

        # Batch size: no cache file exists for the uncompressed (kv*00) method, so derive
        # the per-item cost from the prompt length × model config (the full KV built during
        # prefill) and size on the same shared budget as classic/indices. self.vanilla_batch_size
        # (None/0 → auto-estimate; >0 → force) lets an A/B run pin it.
        #
        # layer_devices covers every parameter's device (embedding/lm_head included), not just
        # decoder layers (as in _get_max_batch_size), so a GPU that holds only the
        # embedding/lm_head (and few or no decoder layers) is still a bottleneck candidate.
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

        max_prompt_tokens = max(
            (
                len(self.tokenizer.encode(p, add_special_tokens=False))
                for p in full_prompts
            ),
            default=1,
        )
        batch_size = self._get_max_batch_size_vanilla(
            max_prompt_tokens=max_prompt_tokens,
            layer_devices=layer_devices,
            batch_size=self.vanilla_batch_size,
        )

        logger.debug(f"max_prompt_tokens={max_prompt_tokens} computed batch_size={batch_size}")

        answers = []
        log_odds = []

        for batch_start in tqdm(
            range(0, len(full_prompts), batch_size),
            desc=f"Processing vanilla batches (bs={batch_size})",
        ):
            batch_prompts = full_prompts[batch_start : batch_start + batch_size]
            try:
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                )
                input_len = inputs["input_ids"].shape[1]
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

                for _i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(_i)

                with torch.no_grad():
                    generated = self.pipe.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                logger.debug(
                    f"batch_start={batch_start} rows={len(batch_prompts)} "
                    f"input_len(tokens)={input_len}"
                )
                for _i in range(torch.cuda.device_count()):
                    _free, _total = torch.cuda.mem_get_info(_i)
                    _peak = torch.cuda.max_memory_allocated(_i) / 1e9
                    _peaks[_i] = max(_peaks.get(_i, 0.0), _peak)
                    logger.debug(
                        f"  GPU {_i}: peak_allocated={_peak:.2f} GB "
                        f"free_after={_free/1e9:.2f} GB"
                    )

                logits = generated.scores[0]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                log_odds.extend((log_probs[:, id1] - log_probs[:, id0]).cpu().tolist())

                decoded = self.tokenizer.batch_decode(
                    generated.sequences[:, input_len:],
                    skip_special_tokens=True,
                )
                answers.extend(decoded)

                del inputs, generated, logits, log_probs

            except Exception as e:
                logger.debug(f"EXCEPTION at batch_start={batch_start}: {e}")
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
                    f"Error processing vanilla batch at index {batch_start}: {e}"
                )
                for _ in batch_prompts:
                    answers.append("Not sure")
                    log_odds.append(0.0)

        # Once per request, not per batch: batches reuse identical shapes, so per-batch
        # empty_cache only added a sync + cudaMalloc churn. The end-of-request release
        # keeps mem_get_info (used by the batch estimators) honest for the next request.
        torch.cuda.empty_cache()
        _gpu, _min_free = gpu_snapshot(_peaks)
        return {
            "answers": answers,
            "log_odds": log_odds,
            "stats": build_inference_stats(
                server="kv_text_qa",
                path="vanilla",
                model_name=self.model_name,
                n_items=len(texts),
                n_batches=(len(full_prompts) + batch_size - 1) // batch_size,
                batch_size=batch_size,
                vanilla=True,
                server_elapsed_s=time.perf_counter() - _stats_t0,
                gpu=_gpu,
                min_free_gb=_min_free,
            ),
        }

    async def _run_kv_cache_text(
        self,
        column_name: str,
        texts: List[str],
        all_questions: List[str],
        compression_ratio: float,
        cache_dir: str,
        boolean_question: bool,
        materialized_compression_ratio: float,
        vanilla: bool,
        keep_in_memory: bool = False,
    ):
        """Run KV cache-based text inference on the data."""

        if vanilla:
            logger.info(f"Using vanilla mode for inference (cr={compression_ratio})")
            return await self._run_vanilla_inference(
                texts=texts,
                all_questions=all_questions,
                boolean_question=boolean_question,
            )

        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."

        # Check if caches exist and generate if not
        save_dir = f"{cache_dir}/{self.model_name}/{self.press_name}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        # Per-text load plan. Default (flag off): one physical cache per text, asserted to
        # exist. With --use-relative-indices:
        # prefer reconstructing from a baseline + relative indices (Path A), fall back to a
        # physical cache if present (Path B), else mark the text errored (graceful per-text
        # degrade — the rest of the batch still answers).
        press_dir = f"{cache_dir}/{self.model_name}/{self.press_name}"
        target_tag = f"comp{self.to_compression_tag(compression_ratio)}"
        base_tag = f"comp{self.to_compression_tag(materialized_compression_ratio)}"
        load_plans = []
        sizing_paths = []  # real on-disk paths for the batch-size estimator's fallback branch

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc=f"Preparing caches for cr {compression_ratio}",
        ):
            hashed_text = self.hash_text(text)
            cache_filename = f"{save_dir}/cache_entry_{hashed_text}.pt"

            relative = None
            if self.use_relative_indices:
                try:
                    relative = resolve_relative_source(
                        press_dir, target_tag, hashed_text, base_tag
                    )
                except RelativeReconstructError as e:
                    logger.warning(
                        f"[text {i}] relative _meta unusable, will try physical: {e}"
                    )
                    relative = None

            has_physical = os.path.exists(cache_filename)
            if relative is None and not has_physical:
                if self.use_relative_indices:
                    logger.warning(
                        f"[text {i}] no relative source and no physical cache: {cache_filename}"
                    )
                    load_plans.append(
                        {
                            "index": i,
                            "hash": hashed_text,
                            "error": f"no cache for text {i}",
                        }
                    )
                    continue
                assert (
                    has_physical
                ), f"Cache file does not exist for text at index {i}: {cache_filename}"

            load_plans.append(
                {
                    "index": i,
                    "hash": hashed_text,
                    "physical": cache_filename if has_physical else None,
                    "relative": relative,
                    "error": None,
                }
            )
            sizing_paths.append(relative[0] if relative is not None else cache_filename)

        if not load_plans:
            logger.warning("No valid caches or texts found")
            return []

        # Set up text processing parameters
        if boolean_question:
            context = "Answer the following question based on the context with '1' or '0'. Do not add any other comments."
        else:
            context = "Answer the following question based on the context. Do not add any other comments."
        all_questions = [context + " " + q for q in all_questions]
        answer_prefix = "Answer: "
        batch_size = self.compression_ratio_to_batch_size[compression_ratio]
        layer_devices = list({p.device for p in self.pipe.model.parameters()})

        # Tokenise the longest question (including the answer prefix that is
        # always appended) so the batch-size estimator can account for the
        # question-token activation overhead.  The per-item chat-template
        # suffix is not known yet, but it is short (< 10 tokens) and omitting
        # it has no meaningful effect on the estimate.
        max_question_tokens = (
            max(
                len(self.tokenizer.encode(q + answer_prefix, add_special_tokens=False))
                for q in all_questions
            )
            if all_questions
            else 0
        )

        # When serving relative indices, sizing_paths point at the (larger) baseline caches
        # while the resident per-item cache is the reconstructed target. Recover the baseline
        # CR from a relative plan's baseline dir (…/comp{base_tag}/cache_entry_*.pt) so the
        # estimator can scale the baseline size down to the target and let the batch grow.
        baseline_cr = None
        if self.use_relative_indices:
            for p in load_plans:
                rel = p.get("relative")
                if rel is not None:
                    baseline_cr = self._comp_dir_to_cr(os.path.dirname(rel[0]))
                    break
            # Client drives which materialized cache is used; verify its declared
            # materialized ratio matches the baseline the _meta.json actually points at.
            if baseline_cr is not None:
                assert baseline_cr == materialized_compression_ratio, (
                    f"client declared materialized_compression_ratio "
                    f"{materialized_compression_ratio} but effective cr "
                    f"{compression_ratio} resolves to baseline cr {baseline_cr} via "
                    f"relative indices"
                )

        logger.debug("layer_devices set: %s", sorted({str(d) for d in layer_devices}))
        logger.debug("first_device (input placement): %s", next(self.pipe.model.parameters()).device)
        for _i in range(torch.cuda.device_count()):
            _free, _total = torch.cuda.mem_get_info(_i)
            logger.debug(f"GPU {_i} free (pre-estimate): {_free/1e9:.2f} GB / {_total/1e9:.2f} GB total")

        batch_size = self._get_max_batch_size(
            column_name=column_name,
            batch_size=batch_size,
            compression_ratio=compression_ratio,
            layer_devices=layer_devices,
            cache_dir=cache_dir,
            file_paths=sizing_paths
            or [p["physical"] for p in load_plans if p.get("physical")],
            max_question_tokens=max_question_tokens,
            baseline_compression_ratio=baseline_cr,
        )

        logger.debug(f"computed batch_size={batch_size}")
        max_new_tokens = 4 if boolean_question else 64

        answers = []
        log_odds = []
        errors_out = {}  # original-text-index -> message, for texts that could not be served
        total_disk_load_time = 0.0
        total_route_time = 0.0
        total_load_wait_time = 0.0
        first_device = next(self.pipe.model.parameters()).device
        layer_devices = [
            self.pipe.model.model.layers[i].self_attn.q_proj.weight.device
            for i in range(len(self.pipe.model.model.layers))
        ]
        # Path A needs the decoder's rotary embedding (only inv_freq) to re-rotate kept keys.
        rotary_emb = (
            self.pipe.model.model.rotary_emb if self.use_relative_indices else None
        )

        def _load_batch_to_cpu(batch_plans):
            def _load_one(plan):
                _t0 = time.perf_counter()
                idx = plan["index"]
                if plan.get("error"):  # marked errored at prepare time
                    return {"error": plan["error"], "index": idx}, 0.0
                # Path A: reconstruct from a baseline + relative indices (CPU gather + pin).
                if plan.get("relative") is not None:
                    baseline_path, idx_path = plan["relative"]
                    try:
                        k_cpu, v_cpu, ridx = gather_payload_cpu(baseline_path, idx_path)
                        return (
                            {
                                "needs_rerotate": True,
                                "k": k_cpu,
                                "v": v_cpu,
                                "idx": ridx,
                                "index": idx,
                            },
                            time.perf_counter() - _t0,
                        )
                    except RelativeReconstructError as e:
                        logger.warning(
                            f"Path A failed for {plan['hash'][:12]}…, falling back to physical: {e}"
                        )
                # Path B: physical cache — from RAM for an -in-memory operator, off disk
                # otherwise.
                physical = plan.get("physical")
                if physical and keep_in_memory:
                    # The pinned copy is authoritative: it was loaded at prepare() and
                    # stays valid whatever happens to the file afterwards. A miss is a
                    # setup bug (prepare never ran for this column), never a data issue —
                    # those short-circuited above — so it must NOT fall back to a disk
                    # read, which would quietly turn this into a disk-served operator.
                    c = PINNED_KV_STORE.get(physical)
                    if c is None:
                        return {
                            "error": f"keep_in_memory cache not pinned: {physical}",
                            "index": idx,
                        }, time.perf_counter() - _t0
                    return {
                        "needs_rerotate": False,
                        "cache": c,
                        "index": idx,
                    }, time.perf_counter() - _t0
                if physical and os.path.exists(physical):
                    c = torch.load(physical, map_location="cpu", weights_only=False)
                    return {
                        "needs_rerotate": False,
                        "cache": c,
                        "index": idx,
                    }, time.perf_counter() - _t0
                # Neither available → graceful per-text error.
                return {
                    "error": f"no usable cache for text {idx}",
                    "index": idx,
                }, time.perf_counter() - _t0

            with ThreadPoolExecutor(max_workers=min(len(batch_plans), 4)) as pool:
                return list(pool.map(_load_one, batch_plans))

        batch_starts = list(range(0, len(load_plans), batch_size))
        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        prefetch_future = prefetch_pool.submit(
            _load_batch_to_cpu,
            load_plans[0 : min(batch_size, len(load_plans))],
        )

        _pin_h0, _pin_m0, _ = PINNED_KV_STORE.stats()
        _loop_t0 = time.perf_counter()
        _peaks: dict = {}  # device index -> peak allocated GB, for the response stats

        # Process texts in batches
        for batch_start in tqdm(
            batch_starts,
            desc=f"Processing batches for CR {compression_ratio}",
        ):
            batch_plans = load_plans[
                batch_start : min(batch_start + batch_size, len(load_plans))
            ]
            batch_questions = all_questions[
                batch_start : min(batch_start + batch_size, len(load_plans))
            ]

            # Wait for prefetched CPU data (should already be ready since inference ran)
            _wait_t0 = time.perf_counter()
            cpu_results = prefetch_future.result()
            total_load_wait_time += time.perf_counter() - _wait_t0
            total_disk_load_time += sum(r[1] for r in cpu_results)

            # Immediately kick off prefetch of next batch
            next_start = batch_start + batch_size
            if next_start < len(load_plans):
                prefetch_future = prefetch_pool.submit(
                    _load_batch_to_cpu,
                    load_plans[
                        next_start : min(next_start + batch_size, len(load_plans))
                    ],
                )

            # Build the GPU caches for VIABLE texts; record errored texts as response slots.
            # Path A payloads are H2D'd + rerotated here (main thread); Path B payloads are
            # routed per-layer. `slots` preserves batch order for reassembly.
            _route_t0 = time.perf_counter()
            payloads = [r[0] for r in cpu_results]
            caches = []
            questions = []
            slots = []  # ("V",) for a viable text, ("E", orig_index, msg) for an errored one
            for p, q in zip(payloads, batch_questions):
                if p.get("error"):
                    slots.append(("E", p["index"], p["error"]))
                    continue
                if p.get("needs_rerotate"):
                    # Path A: sliced H2D + per-device rerotate — every layer lands directly
                    # on its own GPU (no whole-cache staging on device 0, no GPU0→GPUn
                    # second hop), so no scatter pass is needed afterwards.
                    c = rerotate_payload_sharded(
                        p["k"], p["v"], p["idx"], rotary_emb, layer_devices
                    )
                else:
                    # Path B: physical cache on CPU → copy each layer to its device, OUT
                    # OF PLACE. `src` may be the pinned, process-wide object shared by
                    # every query over this text; writing the GPU tensors back into it
                    # would hand the next query a GPU-resident "CPU cache" that is never
                    # freed (and that the pin budget counted as CPU bytes).
                    # H2D mode is the single shared switch (H2D_NON_BLOCKING, default blocking).
                    src = p["cache"]
                    c = DynamicCache()
                    for li in range(_cache_num_layers(src)):
                        ld = layer_devices[li]
                        k, v = _cache_kv(src, li)
                        c.update(
                            k.to(ld, non_blocking=H2D_NON_BLOCKING),
                            v.to(ld, non_blocking=H2D_NON_BLOCKING),
                            li,
                        )
                caches.append(c)
                questions.append(q)
                slots.append(("V",))
            del payloads

            if not caches:
                # Whole batch errored — emit markers, no generation.
                for slot in slots:
                    _, gidx, msg = slot
                    answers.append("")
                    log_odds.append(None)
                    errors_out[gidx] = msg
                continue

            context_lengths = [_cache_seq_length(c) for c in caches]
            max_context_len = max(context_lengths)

            question_ids_list = []
            context_ids_list = []
            padded_context_ids_mask_list = []

            for i, (ctx_len, q) in enumerate(zip(context_lengths, questions)):
                # Pad context to align with KV cache
                padded_context_ids = torch.full(
                    (1, ctx_len),
                    self.tokenizer.pad_token_id + 1,
                    device=self.device,
                )
                pad_len = max_context_len - ctx_len
                padding_ids = torch.full(
                    (1, pad_len), self.tokenizer.pad_token_id, device=self.device
                )
                padded_context = torch.cat([padding_ids, padded_context_ids], dim=1)
                padding_mask = torch.zeros_like(padding_ids)
                padded_context_mask = torch.ones_like(padded_context_ids)
                padded_context_ids_mask = torch.cat(
                    [padding_mask, padded_context_mask], dim=1
                )

                # Determine question suffix
                if self.tokenizer.chat_template is None:
                    question_suffix = "\n"
                else:
                    separator = "\n" + "#" * ctx_len if ctx_len > 0 else "\n#"
                    template_context = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": "Example of context" + separator}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    _, question_suffix = template_context.split(separator)

                # Tokenize question (variable-length)
                complete_question = q + question_suffix + answer_prefix
                question_ids = self.tokenizer.encode(
                    complete_question,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)

                # Store for later padding
                context_ids_list.append(padded_context)
                question_ids_list.append(question_ids)
                padded_context_ids_mask_list.append(padded_context_ids_mask)

            # Compute max question length across batch
            max_question_len = max(q.shape[1] for q in question_ids_list)

            # Pad questions and build batch tensors
            batch_input_ids = []
            batch_attention_masks = []

            for padded_context, question_ids, padded_context_ids_mask in zip(
                context_ids_list, question_ids_list, padded_context_ids_mask_list
            ):
                q_len = question_ids.shape[1]
                q_pad_len = max_question_len - q_len

                # Pad question to match longest in batch
                if q_pad_len > 0:
                    q_padding = torch.full(
                        (1, q_pad_len),
                        self.tokenizer.pad_token_id,
                        device=self.device,
                    )
                    padded_question = torch.cat([q_padding, question_ids], dim=1)
                    padded_question_mask = torch.cat(
                        [torch.zeros_like(q_padding), torch.ones_like(question_ids)],
                        dim=1,
                    )
                else:
                    padded_question = question_ids
                    padded_question_mask = torch.ones_like(question_ids)

                # Concatenate full input
                input_ids = torch.cat([padded_context, padded_question], dim=1)

                # Build attention mask: padding + context + padding + question
                attention_mask = torch.cat(
                    [padded_context_ids_mask, padded_question_mask], dim=1
                )

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)

            # Stack batched tensors
            batched_inputs = torch.cat(batch_input_ids, dim=0)
            batched_attention_mask = torch.cat(batch_attention_masks, dim=0)

            # Ensure all async GPU transfers are complete before touching the tensors
            torch.cuda.synchronize()
            total_route_time += time.perf_counter() - _route_t0

            # Batch the caches. Iterate layers by index and pull (keys, values) via the
            # 5.x-safe accessor: iterating a transformers-5.x DynamicCache yields DynamicLayer
            # objects rather than (key, value) tuples.
            batched_cache = []
            for layer_idx in range(_cache_num_layers(caches[0])):
                layer_kvs = [_cache_kv(c, layer_idx) for c in caches]
                max_seq_len = max(k.shape[2] for k, _ in layer_kvs)
                keys_padded = []
                values_padded = []

                for k, v in layer_kvs:
                    seq_len = k.shape[2]
                    pad_len = max_seq_len - seq_len
                    k_padded = (
                        torch.nn.functional.pad(k, (0, 0, pad_len, 0))
                        if pad_len > 0
                        else k
                    )
                    v_padded = (
                        torch.nn.functional.pad(v, (0, 0, pad_len, 0))
                        if pad_len > 0
                        else v
                    )
                    k_padded = k_padded.contiguous()
                    v_padded = v_padded.contiguous()
                    keys_padded.append(k_padded)
                    values_padded.append(v_padded)

                keys_cat = torch.cat(keys_padded, dim=0)
                values_cat = torch.cat(values_padded, dim=0)
                batched_cache.append((keys_cat, values_cat))

                # Release this layer's per-item tensors immediately. Unconditional: both
                # paths build a per-request DynamicCache, so `caches` never holds the
                # pinned object itself.
                for c in caches:
                    _cache_set_kv(c, layer_idx, None, None)

            padded_cache = DynamicCache()
            for layer_idx, (keys, values) in enumerate(batched_cache):
                padded_cache.update(keys, values, layer_idx)

            # Move inputs to the device of the embedding layer (first layer)
            batched_inputs = batched_inputs.to(first_device)
            batched_attention_mask = batched_attention_mask.to(first_device)

            # Generate responses
            for _i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(_i)
            try:
                with torch.no_grad():
                    generated = self.pipe.model.generate(
                        input_ids=batched_inputs,
                        attention_mask=batched_attention_mask,
                        past_key_values=padded_cache,
                        pad_token_id=self.tokenizer.eos_token_id,
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
            id0 = self.tokenizer.convert_tokens_to_ids("0")
            id1 = self.tokenizer.convert_tokens_to_ids("1")
            log_probs_0 = log_probs[:, id0]
            log_probs_1 = log_probs[:, id1]
            log_odds_1_vs_0 = log_probs_1 - log_probs_0

            # Decode the generated tokens
            decoded = self.tokenizer.batch_decode(
                generated.sequences[:, batched_inputs.shape[1] :],  # type: ignore
                skip_special_tokens=True,
            )

            # Reassemble in batch order: viable texts consume `decoded` in order; errored
            # texts (Path A failed AND no physical) get a marker so the response stays
            # aligned to the input texts and the request never crashes.
            lo_list = log_odds_1_vs_0.cpu().tolist()
            di = 0
            for slot in slots:
                if slot[0] == "V":
                    answers.append(decoded[di])
                    log_odds.append(lo_list[di])
                    di += 1
                else:
                    _, gidx, msg = slot
                    answers.append("")
                    log_odds.append(None)
                    errors_out[gidx] = msg

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
            del (
                batch_input_ids,
                batch_attention_masks,
                context_ids_list,
                question_ids_list,
                padded_context_ids_mask_list,
                batched_cache,
            )

        prefetch_pool.shutdown(wait=False)
        # Once per request, not per batch (see _run_vanilla_inference for the rationale).
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
        if errors_out:
            logger.warning(
                f"{len(errors_out)} text(s) could not be served: {sorted(errors_out)}"
            )
        if keep_in_memory and _pin_misses:
            # Every miss here is a cache prepare() should have pinned: the per-item
            # degrade above kept the request answering, but the answers it produced were
            # not served from RAM, so the measurement this operator exists to make is
            # invalid. Fail the request rather than return a plausible-looking result.
            raise PinnedKVUnavailable(
                f"{_pin_misses} of {n_caches} caches were not resident while serving "
                f"column {column_name!r} on {self.model_name} (cr={compression_ratio}) as "
                "an -in-memory operator. prepare() must run for this column before it "
                "serves; nothing is ever loaded from disk on this path."
            )

        _gpu, _min_free = gpu_snapshot(_peaks)
        result = {
            "answers": answers,
            "log_odds": log_odds,
            # Reports only what this request already measured above; see
            # reasondb/backends/inference_stats.py.
            "stats": build_inference_stats(
                server="kv_text_qa",
                path="kv",
                model_name=self.model_name,
                n_items=len(texts),
                n_batches=(len(load_plans) + batch_size - 1) // batch_size
                if batch_size
                else None,
                batch_size=batch_size,
                effective_compression_ratio=compression_ratio,
                materialized_compression_ratio=materialized_compression_ratio,
                vanilla=False,
                keep_in_memory=keep_in_memory,
                server_elapsed_s=total_elapsed,
                cache_load_s=total_disk_load_time,
                cache_route_s=total_route_time,
                cache_wait_s=total_load_wait_time,
                n_caches=n_caches,
                gpu=_gpu,
                min_free_gb=_min_free,
                pinned_hits=_pin_hits,
                pinned_misses=_pin_misses,
                pinned_gb=_pin_held,
                n_errors=len(errors_out),
            ),
        }
        if (
            errors_out
        ):  # only present when something degraded
            result["errors"] = errors_out
        return result

    async def _generate_cache_for_text(
        self, text_content, cache_filename, compression_ratio, column_name=None
    ):
        """Generate and save text KV cache for a single text."""
        assert (
            self.pipe is not None
        ), "Model pipeline for cache generation is not initialized."

        try:
            # Set the press
            press = self.presses[compression_ratio]

            # For Finch, prepare context with delimiter token and query workload
            if self.press_name == "finch" and press is not None:
                # Set the few-shot questions for the query-aware compression
                yaml_path = "queries_workloads/_workload.yaml"
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)

                query_list = data["queries"]
                queries_workload = (
                    "Pay attention to these examples of questions:\n"
                    + "\n".join(f"- {q}" for q in query_list)
                )

                context = text_content[
                    : min(128000, len(text_content))
                ]  # Limit context length
                context_aware = context + press.delimiter_token + queries_workload

            elif self.press_name == "finch-cachenotes" and press is not None:
                import pandas as pd
                import hashlib
                import re

                def normalize_text(text):
                    """Normalize text by removing extra whitespace and standardizing characters."""
                    text = text.strip()
                    # Normalize whitespace but preserve newlines
                    text = re.sub(r"[ \t]+", " ", text)  # Only collapse spaces and tabs
                    text = re.sub(
                        r"\n\s*\n", "\n\n", text
                    )  # Collapse multiple blank lines
                    text = text.replace("\r\n", "\n").replace(
                        "\r", "\n"
                    )  # Normalize line endings
                    text = text.replace("\u200b", "")  # Remove zero-width spaces
                    return text

                # Resolve task and text column from column_name
                assert (
                    column_name is not None
                ), "column_name is required for finch-cachenotes"
                assert column_name in CPT_COLUMN_MAP, (
                    f"Unknown column_name '{column_name}' for finch-cachenotes. "
                    f"Supported: {list(CPT_COLUMN_MAP.keys())}"
                )
                task_key, text_col = CPT_COLUMN_MAP[column_name]
                cpt_path = CPT_PATH[task_key]

                # Load CSV file
                df = pd.read_csv(cpt_path)

                # Normalize text_content for comparison
                normalized_text_content = normalize_text(text_content)
                text_hash = hashlib.sha256(text_content.encode()).hexdigest()

                # Try to find matching row by comparing text content with the correct column
                matching_row = None
                for _, row in df.iterrows():
                    row_text = str(row.get(text_col, ""))
                    normalized_row_text = normalize_text(row_text)

                    # Try exact match first, then normalized match, then hash match
                    if (
                        row_text == text_content
                        or normalized_row_text == normalized_text_content
                        or hashlib.sha256(row_text.encode()).hexdigest() == text_hash
                    ):
                        matching_row = row
                        break

                # Get the CPT from the matching row
                if matching_row is not None:
                    cpt = str(matching_row["cpt"])
                    context = text_content[: min(128000, len(text_content))]
                    context_aware = context + press.delimiter_token + cpt
                else:
                    logger.warning(
                        f"No matching CPT found for column '{column_name}' "
                        f"(task={task_key}, text_col={text_col})"
                    )
                    cpt = "Your task is to answer questions based on the context."
                    context = text_content[: min(128000, len(text_content))]
                    context_aware = context + press.delimiter_token + cpt
            else:
                context_aware = text_content[
                    : min(128000, len(text_content))
                ]  # Limit context length

            answer_prefix = "Answer: "

            # Generate cache using preprocess method
            inputs = self.pipe.preprocess(
                context=context_aware,
                questions=[""],
                answer_prefix=answer_prefix,
                max_context_length=128000,
            )

            context_ids = inputs["context_ids"]

            # For Finch: Adjust compression ratio to ensure fair comparison
            # Goal: After removing window tokens, retain same % of context as other methods
            if (
                self.press_name in ("finch", "finch-cachenotes")
                and press is not None
                and compression_ratio > 0
            ):
                # Use the actual tokenized length from context_ids instead of re-encoding
                actual_context_ids_length = context_ids.shape[1]

                # Tokenize to get exact counts of components
                if self.press_name == "finch":
                    window_tokens_count = len(
                        self.tokenizer.encode(
                            queries_workload, add_special_tokens=False
                        )
                    )
                else:
                    window_tokens_count = len(
                        self.tokenizer.encode(cpt, add_special_tokens=False)
                    )

                context_tokens_count = (
                    actual_context_ids_length - window_tokens_count - 1
                )

                # Calculate adjusted ratio
                desired_context_kept = int(
                    context_tokens_count * (1 - compression_ratio)
                )
                # Total to keep before window removal: desired_context + window
                total_to_keep = desired_context_kept + window_tokens_count
                # Effective compression ratio
                adjusted_ratio = 1.0 - (total_to_keep / actual_context_ids_length)
                adjusted_ratio = max(0.0, min(1.0, adjusted_ratio))

                # Create a new press instance with adjusted ratio
                press = FinchPress(compression_ratio=adjusted_ratio)
                press.update_model_and_tokenizer(self.pipe.model, self.pipe.tokenizer)

            cache = DynamicCache()

            # ----------------------------------------------------------------
            first_device = next(self.pipe.model.parameters()).device
            context_ids = context_ids.to(first_device)
            # ----------------------------------------------------------------

            logger.info(
                f"Generating cache for text with initial length {context_ids.shape[1]} tokens"
            )
            with torch.inference_mode():
                with (
                    press(self.pipe.model)
                    if press is not None
                    else contextlib.nullcontext()
                ):
                    # Run the model without the lm head for pre-filling. None of the
                    # supported presses needs attention weights, matching the multi-CR
                    # prefill paths above.
                    self.pipe.model.model(
                        input_ids=context_ids,
                        past_key_values=cache,
                        use_cache=True,
                        output_attentions=False,
                    )

            # For Finch: Remove window tokens from cache (keep only compressed context)
            # The window guides compression but should not be stored
            if (
                self.press_name in ("finch", "finch-cachenotes")
                and press is not None
                and hasattr(press, "window_size")
            ):
                window_size = press.window_size
                if window_size is not None and window_size > 0:
                    # Trim the last window_size tokens from each cache layer
                    for li in range(_cache_num_layers(cache)):
                        k, v = _cache_kv(cache, li)
                        _cache_set_kv(
                            cache,
                            li,
                            k[:, :, :-window_size, :],
                            v[:, :, :-window_size, :],
                        )

            # Save cache to disk
            for li in range(_cache_num_layers(cache)):
                k, v = _cache_kv(cache, li)
                _cache_set_kv(cache, li, k.detach().cpu(), v.detach().cpu())

            logger.info(f"Saving cache to: {cache_filename}")

            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
            torch.save(cache, cache_filename)

            if not os.path.exists(cache_filename):
                logger.error(f"Cache file was not actually created: {cache_filename}")

            # Cleanup
            del cache
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error generating cache for text {cache_filename}: {str(e)}")
            raise

    def _unique_layer_devices(self) -> List:
        """Return deduplicated list of GPU devices that hold model layers (ordered)."""
        seen: set = set()
        devices = []
        for layer in self.pipe.model.model.layers:
            d = layer.self_attn.q_proj.weight.device
            key = (d.type, d.index)
            if key not in seen:
                seen.add(key)
                devices.append(d)
        return devices

    def compute_text_qa_join_response(
        self,
        column_name: str,
        unique_texts: List[str],
        questions_per_text: List[List[str]],
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
        materialized_compression_ratio: float,
        vanilla: bool,
        keep_in_memory: bool = False,
    ) -> dict:
        self._validate_client_crs(
            compression_ratio, materialized_compression_ratio, vanilla, keep_in_memory
        )
        return asyncio.run(
            self._run_kv_cache_text_join(
                column_name=column_name,
                unique_texts=unique_texts,
                questions_per_text=questions_per_text,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
                materialized_compression_ratio=materialized_compression_ratio,
                vanilla=vanilla,
                keep_in_memory=keep_in_memory,
            )
        )

    async def _run_kv_cache_text_join(
        self,
        column_name: str,
        unique_texts: List[str],
        questions_per_text: List[List[str]],
        compression_ratio: float,
        cache_dir: str,
        boolean_question: bool,
        materialized_compression_ratio: float,
        vanilla: bool,
        keep_in_memory: bool = False,
    ) -> dict:
        """Batch KV-cache join: for N left texts × M right-side questions per text."""
        assert self.pipe is not None
        assert self.tokenizer is not None
        assert len(unique_texts) == len(questions_per_text)
        assert len(questions_per_text) > 0

        if vanilla:
            logger.info(
                f"Using vanilla mode for join inference (cr={compression_ratio})"
            )
            flat_texts, flat_questions, pair_indices = [], [], []
            for i, (text, questions) in enumerate(
                zip(unique_texts, questions_per_text)
            ):
                for j, q in enumerate(questions):
                    flat_texts.append(text)
                    flat_questions.append(q)
                    pair_indices.append((i, j))
            result = await self._run_vanilla_inference(
                texts=flat_texts,
                all_questions=flat_questions,
                boolean_question=boolean_question,
            )
            answers_per_text = [[""] * len(qs) for qs in questions_per_text]
            log_odds_per_text = [[0.0] * len(qs) for qs in questions_per_text]
            for (i, j), answer, log_odd in zip(
                pair_indices, result["answers"], result["log_odds"]
            ):
                answers_per_text[i][j] = answer
                log_odds_per_text[i][j] = log_odd
            # This path delegates to vanilla inference, so reuse its measurements and
            # only correct the label describing which route served the request.
            _stats = dict(result["stats"])
            _stats["path"] = "join"
            _stats["n_items"] = len(unique_texts)
            return {
                "answers_per_text": answers_per_text,
                "log_odds_per_text": log_odds_per_text,
                "stats": _stats,
            }

        canonical_questions = questions_per_text[0]
        questions_are_uniform = all(
            qs == canonical_questions for qs in questions_per_text
        )

        save_dir = f"{cache_dir}/{self.model_name}/{self.press_name}/comp{self.to_compression_tag(compression_ratio)}"
        if boolean_question:
            context_prefix = "Answer the following question based on the context above with '1' or '0'. Do not add any other comments."
        else:
            context_prefix = "Answer the following question based on both the context above. Do not add any other comments."
        answer_prefix = "Answer: "
        max_new_tokens = 4 if boolean_question else 64
        id0 = self.tokenizer.convert_tokens_to_ids("0")
        id1 = self.tokenizer.convert_tokens_to_ids("1")
        first_device = next(self.pipe.model.parameters()).device

        N = len(unique_texts)
        max_M = max(len(qs) for qs in questions_per_text)

        # Per-text load plan, mirroring _run_kv_cache_text: with --use-relative-indices,
        # prefer reconstructing from a baseline + relative indices (Path A), falling back
        # to a physical cache (Path B). Joins have no per-text degrade slot (every text in
        # the batch must answer), so a text with neither source is a hard error.
        press_dir = f"{cache_dir}/{self.model_name}/{self.press_name}"
        target_tag = f"comp{self.to_compression_tag(compression_ratio)}"
        base_tag = f"comp{self.to_compression_tag(materialized_compression_ratio)}"
        load_plans = []
        sizing_paths = []  # real on-disk files for the batch-size estimator
        for i, text in enumerate(unique_texts):
            hashed_text = self.hash_text(text)
            cache_filename = f"{save_dir}/cache_entry_{hashed_text}.pt"
            relative = None
            if self.use_relative_indices:
                try:
                    relative = resolve_relative_source(
                        press_dir, target_tag, hashed_text, base_tag
                    )
                except RelativeReconstructError as e:
                    logger.warning(
                        f"[join] [text {i}] relative _meta unusable, will try physical: {e}"
                    )
                    relative = None
            has_physical = os.path.exists(cache_filename)
            assert (
                relative is not None or has_physical
            ), f"[join] No physical cache and no relative source for text at index {i}: {cache_filename}"
            load_plans.append(
                {
                    "index": i,
                    "hash": hashed_text,
                    "physical": cache_filename if has_physical else None,
                    "relative": relative,
                }
            )
            sizing_paths.append(relative[0] if relative is not None else cache_filename)

        longest_question = max(canonical_questions, key=len)
        sample_q_tokens = len(
            self.tokenizer.encode(
                context_prefix + " " + longest_question, add_special_tokens=False
            )
        )
        if load_plans[0]["relative"] is not None:
            # Reconstructed context length = kept-token count = the index file's last dim
            # (the tiny idx file avoids deserializing a whole baseline cache here).
            _idx = torch.load(
                load_plans[0]["relative"][1], map_location="cpu", weights_only=True
            )
            sample_ctx_tokens = _idx.shape[2]
            del _idx
        else:
            _sc = torch.load(
                load_plans[0]["physical"], map_location="cpu", weights_only=False
            )
            if isinstance(_sc, dict):
                sample_ctx_tokens = _sc["key_cache"][0].shape[2]
            else:
                sample_ctx_tokens = _cache_kv(_sc, 0)[0].shape[2]
            del _sc

        layer_devices = [
            self.pipe.model.model.layers[i].self_attn.q_proj.weight.device
            for i in range(len(self.pipe.model.model.layers))
        ]
        # layer_devices (ordered, one entry per decoder layer) is reused below for indexed
        # per-layer cache placement, so it must stay decoder-layers-only. The batch-size
        # estimator instead gets every parameter's device (embedding/lm_head included), as
        # in the non-join compressed path, so a GPU holding only non-decoder weight is still
        # a bottleneck candidate.
        torch.cuda.empty_cache()
        estimator_layer_devices = list({p.device for p in self.pipe.model.parameters()})

        # Path A needs the decoder's rotary embedding (only inv_freq) to re-rotate kept keys.
        rotary_emb = (
            self.pipe.model.model.rotary_emb if self.use_relative_indices else None
        )

        # Relative plans are sized against the (larger) baseline files; hand the estimator
        # the baseline CR so it scales down to the reconstructed target size.
        baseline_cr = None
        if self.use_relative_indices:
            for p in load_plans:
                if p["relative"] is not None:
                    baseline_cr = self._comp_dir_to_cr(
                        os.path.dirname(p["relative"][0])
                    )
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

        batch_size = self._get_max_batch_size_join(
            column_name=column_name,
            batch_size=self.compression_ratio_to_batch_size[compression_ratio],
            compression_ratio=compression_ratio,
            layer_devices=estimator_layer_devices,
            cache_dir=cache_dir,
            file_paths=sizing_paths,
            max_question_tokens=sample_q_tokens,
            max_context_tokens=sample_ctx_tokens,
            baseline_compression_ratio=baseline_cr,
        )

        logger.debug(f"computed batch_size={batch_size}")

        answers_per_text = [[""] * len(qs) for qs in questions_per_text]
        log_odds_per_text = [[0.0] * len(qs) for qs in questions_per_text]
        total_disk_load_time = 0.0
        total_route_time = 0.0
        total_wall_time = 0.0
        _peaks: dict = {}  # device index -> peak allocated GB, for the response stats
        _pin_h0, _pin_m0, _ = PINNED_KV_STORE.stats()

        for batch_start in tqdm(range(0, N, batch_size), desc="TextQA join: batches"):
            batch_plans = load_plans[batch_start : batch_start + batch_size]
            B = len(batch_plans)

            caches = []
            context_lens = []

            def _load_join_to_cpu(plan):
                """CPU half of one cache load (loader thread). Physical: deserialize + scatter
                each layer directly to its own device — the same topology as the main serving
                path and the image server (no whole-cache GPU-0 staging spike). Relative
                (Path A): gather kept rows from the baseline into pinned buffers; the GPU
                rerotate runs on the main thread, mirroring _run_kv_cache_text."""
                _t0 = time.perf_counter()
                if plan["relative"] is not None:
                    baseline_path, idx_path = plan["relative"]
                    try:
                        k_cpu, v_cpu, ridx = gather_payload_cpu(baseline_path, idx_path)
                        return (
                            {
                                "needs_rerotate": True,
                                "k": k_cpu,
                                "v": v_cpu,
                                "idx": ridx,
                            },
                            time.perf_counter() - _t0,
                            0.0,
                        )
                    except RelativeReconstructError as e:
                        logger.warning(
                            f"[join] Path A failed for {plan['hash'][:12]}…, falling back to physical: {e}"
                        )
                fname = plan["physical"]
                if keep_in_memory:
                    # Pinned at prepare(); never re-read from disk (see _run_kv_cache_text).
                    cache_data = PINNED_KV_STORE.get(fname)
                    if cache_data is None:
                        raise PinnedKVUnavailable(
                            f"keep_in_memory cache not pinned: {fname}. prepare() must run "
                            f"for column {column_name!r} on {self.model_name} before it "
                            "serves a join."
                        )
                else:
                    assert fname is not None and os.path.exists(
                        fname
                    ), f"Cache file missing: {fname}"
                    cache_data = torch.load(
                        fname, map_location="cpu", weights_only=False
                    )
                _t1 = time.perf_counter()
                if isinstance(cache_data, dict):
                    kc = cache_data["key_cache"]
                    vc = cache_data["value_cache"]
                elif hasattr(cache_data, "layers"):
                    kc = [l.keys for l in cache_data.layers]
                    vc = [l.values for l in cache_data.layers]
                else:
                    kc = cache_data.key_cache
                    vc = cache_data.value_cache
                kc = [k.to(layer_devices[j]) for j, k in enumerate(kc)]
                vc = [v.to(layer_devices[j]) for j, v in enumerate(vc)]
                _t2 = time.perf_counter()
                return (
                    {"needs_rerotate": False, "kc": kc, "vc": vc},
                    _t1 - _t0,
                    _t2 - _t1,
                )

            _wall_t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=min(B, 4)) as pool:
                _results = list(pool.map(_load_join_to_cpu, batch_plans))

            for payload, _disk_t, _route_t in _results:
                total_disk_load_time += _disk_t
                total_route_time += _route_t
                if payload["needs_rerotate"]:
                    # Path A GPU half: sliced H2D + per-device rerotate — each layer lands
                    # directly on its own GPU, shape-identical to a physical cache.
                    _rt0 = time.perf_counter()
                    c = rerotate_payload_sharded(
                        payload["k"],
                        payload["v"],
                        payload["idx"],
                        rotary_emb,
                        layer_devices,
                    )
                    total_route_time += time.perf_counter() - _rt0
                    n_layers = _cache_num_layers(c)
                    kc = [_cache_kv(c, li)[0] for li in range(n_layers)]
                    vc = [_cache_kv(c, li)[1] for li in range(n_layers)]
                else:
                    kc, vc = payload["kc"], payload["vc"]
                caches.append((kc, vc))
                context_lens.append(kc[0].shape[2])
            torch.cuda.synchronize()
            total_wall_time += time.perf_counter() - _wall_t0

            max_ctx = max(context_lens)
            batched_gpu_keys = []
            batched_gpu_values = []
            for layer_idx, layer_kv_pairs in enumerate(
                zip(*[zip(kc, vc) for kc, vc in caches])
            ):
                keys_padded, values_padded = [], []
                for k, v in layer_kv_pairs:
                    pad_len = max_ctx - k.shape[2]
                    k = (
                        torch.nn.functional.pad(k, (0, 0, pad_len, 0)).contiguous()
                        if pad_len > 0
                        else k
                    )
                    v = (
                        torch.nn.functional.pad(v, (0, 0, pad_len, 0)).contiguous()
                        if pad_len > 0
                        else v
                    )
                    keys_padded.append(k)
                    values_padded.append(v)
                batched_gpu_keys.append(torch.cat(keys_padded, dim=0))
                batched_gpu_values.append(torch.cat(values_padded, dim=0))

            batch_qs = questions_per_text[batch_start : batch_start + batch_size]
            batch_max_j = max(len(qs) for qs in batch_qs)
            question_iter = (
                enumerate(canonical_questions)
                if questions_are_uniform
                else ((j, None) for j in range(batch_max_j))
            )

            for j, _uniform_question in question_iter:
                if questions_are_uniform:
                    full_question = context_prefix + " " + _uniform_question
                    valid_bs = list(range(B))
                    batch_input_ids, batch_attn_mask = [], []
                    for ctx_len in context_lens:
                        pad_len = max_ctx - ctx_len
                        if self.tokenizer.chat_template is None:
                            question_suffix = "\n"
                        else:
                            sep = "\n" + "#" * ctx_len if ctx_len > 0 else "\n#"
                            tmpl = self.tokenizer.apply_chat_template(
                                [
                                    {
                                        "role": "user",
                                        "content": "Example of context" + sep,
                                    }
                                ],
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                            _, question_suffix = tmpl.split(sep)
                        complete_q = full_question + question_suffix + answer_prefix
                        q_ids = self.tokenizer.encode(
                            complete_q, return_tensors="pt", add_special_tokens=False
                        ).to(first_device)
                        ctx_ids = torch.full(
                            (1, ctx_len),
                            self.tokenizer.pad_token_id + 1,
                            device=self.device,
                        )
                        pad_ids = torch.full(
                            (1, pad_len),
                            self.tokenizer.pad_token_id,
                            device=self.device,
                        )
                        input_ids_i = torch.cat([pad_ids, ctx_ids, q_ids], dim=1)
                        mask_i = torch.cat(
                            [
                                torch.zeros_like(pad_ids),
                                torch.ones_like(ctx_ids),
                                torch.ones_like(q_ids),
                            ],
                            dim=1,
                        )
                        batch_input_ids.append(input_ids_i)
                        batch_attn_mask.append(mask_i)
                else:
                    valid_bs = [b for b in range(B) if j < len(batch_qs[b])]
                    per_item_questions = [batch_qs[b][j] for b in valid_bs]
                    per_item_q_ids = []
                    for b, question in zip(valid_bs, per_item_questions):
                        ctx_len = context_lens[b]
                        full_q = context_prefix + " " + question
                        if self.tokenizer.chat_template is None:
                            question_suffix = "\n"
                        else:
                            sep = "\n" + "#" * ctx_len if ctx_len > 0 else "\n#"
                            tmpl = self.tokenizer.apply_chat_template(
                                [
                                    {
                                        "role": "user",
                                        "content": "Example of context" + sep,
                                    }
                                ],
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                            _, question_suffix = tmpl.split(sep)
                        q_ids = self.tokenizer.encode(
                            full_q + question_suffix + answer_prefix,
                            return_tensors="pt",
                            add_special_tokens=False,
                        ).to(first_device)
                        per_item_q_ids.append(q_ids)
                    max_q_len = max(q.shape[1] for q in per_item_q_ids)
                    batch_input_ids, batch_attn_mask = [], []
                    for vi, b in enumerate(valid_bs):
                        ctx_len = context_lens[b]
                        ctx_pad = max_ctx - ctx_len
                        q_ids = per_item_q_ids[vi]
                        q_pad = max_q_len - q_ids.shape[1]
                        ctx_ids = torch.full(
                            (1, ctx_len),
                            self.tokenizer.pad_token_id + 1,
                            device=self.device,
                        )
                        pad_ids = torch.full(
                            (1, ctx_pad),
                            self.tokenizer.pad_token_id,
                            device=self.device,
                        )
                        q_pad_ids = torch.full(
                            (1, q_pad), self.tokenizer.pad_token_id, device=self.device
                        )
                        input_ids_i = torch.cat(
                            [pad_ids, ctx_ids, q_pad_ids, q_ids], dim=1
                        )
                        mask_i = torch.cat(
                            [
                                torch.zeros_like(pad_ids),
                                torch.ones_like(ctx_ids),
                                torch.zeros_like(q_pad_ids),
                                torch.ones_like(q_ids),
                            ],
                            dim=1,
                        )
                        batch_input_ids.append(input_ids_i)
                        batch_attn_mask.append(mask_i)

                batched_inputs = torch.cat(batch_input_ids, dim=0)
                batched_attn = torch.cat(batch_attn_mask, dim=0)

                cache = DynamicCache()
                if questions_are_uniform:
                    for layer_idx, (k, v) in enumerate(
                        zip(batched_gpu_keys, batched_gpu_values)
                    ):
                        cache.update(k.clone(), v.clone(), layer_idx)
                else:
                    for layer_idx, (k, v) in enumerate(
                        zip(batched_gpu_keys, batched_gpu_values)
                    ):
                        cache.update(
                            k[valid_bs].clone(), v[valid_bs].clone(), layer_idx
                        )

                for _i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(_i)
                try:
                    with torch.no_grad():
                        generated = self.pipe.model.generate(
                            input_ids=batched_inputs,
                            attention_mask=batched_attn,
                            past_key_values=cache,
                            pad_token_id=self.tokenizer.eos_token_id,
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )
                except Exception as e:
                    logger.debug(
                        f"[join] EXCEPTION during generate() at batch of "
                        f"{len(valid_bs)} item(s): {e}"
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

                logger.debug(f"[join] batch of {len(valid_bs)} item(s) succeeded")
                for _i in range(torch.cuda.device_count()):
                    _free, _total = torch.cuda.mem_get_info(_i)
                    _peak = torch.cuda.max_memory_allocated(_i) / 1e9
                    _peaks[_i] = max(_peaks.get(_i, 0.0), _peak)
                    logger.debug(
                        f"  GPU {_i}: peak_allocated={_peak:.2f} GB "
                        f"free_after={_free/1e9:.2f} GB / {_total/1e9:.2f} GB total"
                    )
                logits = generated.scores[0]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                log_odds_batch = (log_probs[:, id1] - log_probs[:, id0]).cpu()
                decoded_batch = self.tokenizer.batch_decode(
                    generated.sequences[:, batched_inputs.shape[1] :],
                    skip_special_tokens=True,
                )
                for vi, b in enumerate(valid_bs):
                    answers_per_text[batch_start + b][j] = decoded_batch[vi]
                    log_odds_per_text[batch_start + b][j] = log_odds_batch[vi].item()

                del cache, batched_inputs, batched_attn, generated, logits, log_probs

            del batched_gpu_keys, batched_gpu_values

        # Once per request, not per batch (see _run_vanilla_inference for the rationale).
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
            f"KV cache timing join ({N} contexts) — "
            f"load caches: {total_disk_load_time:.2f}s ({total_disk_load_time / N:.3f}s/context, thread-sum) | "
            f"distribute to GPUs: {total_route_time:.2f}s ({total_route_time / N:.3f}s/context) | "
            f"total: {total_wall_time:.2f}s{pinned_note}"
        )
        _gpu, _min_free = gpu_snapshot(_peaks)
        return {
            "answers_per_text": answers_per_text,
            "log_odds_per_text": log_odds_per_text,
            "stats": build_inference_stats(
                server="kv_text_qa",
                path="join",
                model_name=self.model_name,
                n_items=N,
                n_batches=(N + batch_size - 1) // batch_size if batch_size else None,
                batch_size=batch_size,
                effective_compression_ratio=compression_ratio,
                materialized_compression_ratio=materialized_compression_ratio,
                vanilla=False,
                keep_in_memory=keep_in_memory,
                server_elapsed_s=total_wall_time,
                cache_load_s=total_disk_load_time,
                cache_route_s=total_route_time,
                n_caches=N,
                gpu=_gpu,
                min_free_gb=_min_free,
                pinned_hits=_pin_hits,
                pinned_misses=_pin_misses,
                pinned_gb=_pin_held,
            ),
        }

    def compute_direct_qa_response(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        batch_size: int = 128,
    ) -> dict:
        """Run direct LLM inference without KV caches."""
        assert self.pipe is not None
        assert self.tokenizer is not None
        assert len(questions) == len(contexts)
        _stats_t0 = time.perf_counter()

        max_new_tokens = 4 if boolean_question else 64
        if boolean_question:
            instruction = (
                "Answer '1' if BOTH the context above AND the following text satisfy the condition, '0' otherwise. "
                "Do not add any other comments."
            )
        else:
            instruction = "Answer the following question based on both the context above and the following text. Do not add any other comments."

        answer_prefix = "Answer: "
        first_device = next(self.pipe.model.parameters()).device
        id0 = self.tokenizer.convert_tokens_to_ids("0")
        id1 = self.tokenizer.convert_tokens_to_ids("1")

        full_prompts = []
        for context, question in zip(contexts, questions):
            full_content = context + instruction + " " + question
            if self.tokenizer.chat_template is None:
                full_prompts.append(full_content + "\n" + answer_prefix)
            else:
                full_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": full_content}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    + answer_prefix
                )

        answers = []
        log_odds_list = []
        for batch_start in tqdm(
            range(0, len(full_prompts), batch_size),
            desc=f"Direct QA batches (bs={batch_size})",
        ):
            batch_prompts = full_prompts[batch_start : batch_start + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_len = inputs["input_ids"].shape[1]
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = self.pipe.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            logits = generated.scores[0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            log_odds_list.extend((log_probs[:, id1] - log_probs[:, id0]).cpu().tolist())
            answers.extend(
                self.tokenizer.batch_decode(
                    generated.sequences[:, input_len:], skip_special_tokens=True
                )
            )
            del inputs, generated

        # Once per request, not per batch (see _run_vanilla_inference for the rationale).
        torch.cuda.empty_cache()
        _gpu, _min_free = gpu_snapshot()
        return {
            "answers": answers,
            "log_odds": log_odds_list,
            "stats": build_inference_stats(
                server="kv_text_qa",
                path="direct",
                model_name=self.model_name,
                n_items=len(questions),
                n_batches=(len(full_prompts) + batch_size - 1) // batch_size
                if batch_size
                else None,
                batch_size=batch_size,
                vanilla=True,
                server_elapsed_s=time.perf_counter() - _stats_t0,
                gpu=_gpu,
                min_free_gb=_min_free,
            ),
        }


model_wrapper = None


class Status(Resource):
    def get(self):
        assert model_wrapper is not None
        return {
            "status": "alive",
            "model_name": model_wrapper.model_name,
            "compression_ratios": list(model_wrapper.compression_ratios),
            # Whether this server was started with --use-relative-indices, i.e. which
            # half of the USE_INDICES/--use-indexes pairing it belongs to. Reported so a
            # worker reusing an already-running server can refuse a mismatch up front;
            # otherwise the mismatch only surfaces per job, as a client-side "no usable
            # cache or relative index" (indices on, --use-indexes off) or as silent
            # re-prefilling of every ratio (indices off, --use-indexes on).
            "use_relative_indices": bool(model_wrapper.use_relative_indices),
            # How much RAM this server may hold pinned KV caches in. 0 means an
            # -in-memory operator cannot be served here at all, which its setup()
            # asserts on before any query runs.
            "kv_cache_pin_gb": PINNED_KV_STORE.budget_bytes / 1e9,
        }, 200


class PrepareCaches(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "texts": [
                "text1",
                "text2",
            ],
            "cache_dir": "/path/to/cache/dir"
            "compression_ratio": 0.5
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        texts = data["texts"]
        cache_dir = remap_cache_dir_to_local_mirror(data["cache_dir"])
        assert model_wrapper is not None
        _t0 = time.perf_counter()
        try:
            prep_result = asyncio.run(
                model_wrapper.prepare_caches(
                    column_name=column_name,
                    texts=texts,
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
            # A configuration problem the operator must fix (no pin budget, or a budget
            # too small). Returned as JSON so the client can print it; an uncaught
            # exception would surface as a bare "Internal Server Error".
            logger.error(str(e))
            return {"status": "error", "error": str(e)}, 503
        _gpu, _min_free = gpu_snapshot()
        return {
            "status": "cache_ready",
            **(prep_result or {}),
            "stats": build_inference_stats(
                server="kv_text_qa",
                path="prepare",
                model_name=model_wrapper.model_name,
                n_items=len(texts),
                effective_compression_ratio=data["effective_compression_ratio"],
                materialized_compression_ratio=data["materialized_compression_ratio"],
                vanilla=data["vanilla"],
                keep_in_memory=data["keep_in_memory"],
                server_elapsed_s=time.perf_counter() - _t0,
                n_caches=(prep_result or {}).get("n_texts"),
                gpu=_gpu,
                min_free_gb=_min_free,
                n_errors=(prep_result or {}).get("n_generation_errors", 0) or 0,
            ),
        }, 200


class ReleasePinnedKV(Resource):
    def post(self):
        """Drop every KV cache an -in-memory operator pinned here. Takes no body.

        Called by a worker when its dataset changes. Pins are never evicted, so without
        this a server that outlives one benchmark accumulates the next one's column on top
        of it, until a later /prepare_caches overflows a budget that was sized correctly
        for any single dataset.

        Deliberately does not read the request body: `requests.post(url)` with no json=
        sends an empty one, which `request.get_json(force=True)` — what every other handler
        in this file uses — would raise on.
        """
        return release_pinned_kv_response(), 200


class TextQA(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "texts": [
                "text 1",
                "text 2",
            ],
            "questions": [
                "How many cats are there?"
                "How many dogs are there?"
            ],
            "effective_compression_ratio": 0.9,
            "materialized_compression_ratio": 0.5,
            "vanilla": false,
            "boolean": true,
            "cache_dir": "/path/to/cache/dir"
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        texts = data["texts"]
        questions = data["questions"]
        boolean_question = data["boolean"]
        cache_dir = remap_cache_dir_to_local_mirror(data["cache_dir"])
        assert model_wrapper is not None
        assert (
            len(texts) == len(questions)
        ), f"Number of texts {len(texts)} must match number of questions {len(questions)}"
        try:
            responses = model_wrapper.compute_text_qa_response(
                column_name=column_name,
                texts=texts,
                questions=questions,
                compression_ratio=data["effective_compression_ratio"],
                boolean_question=boolean_question,
                cache_dir=cache_dir,
                materialized_compression_ratio=data["materialized_compression_ratio"],
                vanilla=data["vanilla"],
                keep_in_memory=data["keep_in_memory"],
            )
        except PinnedKVUnavailable as e:
            # Loud either way, but a 503 carries the reason; an uncaught exception
            # reaches the client as a bare "Internal Server Error".
            logger.error(str(e))
            return {"status": "error", "error": str(e)}, 503
        return responses, 200


class TextQADirect(Resource):
    def post(self):
        """Direct QA without KV caches.

        Expects JSON: {"questions": [...], "contexts": [...], "boolean": true, "batch_size": 128}
        Returns {"answers": [...], "log_odds": [...]}
        """
        data = request.get_json(force=True)
        assert model_wrapper is not None
        responses = model_wrapper.compute_direct_qa_response(
            questions=data["questions"],
            contexts=data["contexts"],
            boolean_question=data["boolean"],
            batch_size=data.get("batch_size", 128),
        )
        return responses, 200


class TextQAJoin(Resource):
    def post(self):
        data = request.get_json(force=True)
        assert model_wrapper is not None
        assert len(data["unique_texts"]) == len(data["questions_per_text"])
        responses = model_wrapper.compute_text_qa_join_response(
            column_name=data["column_name"],
            unique_texts=data["unique_texts"],
            questions_per_text=data["questions_per_text"],
            compression_ratio=data["effective_compression_ratio"],
            boolean_question=data["boolean"],
            cache_dir=remap_cache_dir_to_local_mirror(data["cache_dir"]),
            materialized_compression_ratio=data["materialized_compression_ratio"],
            vanilla=data["vanilla"],
            keep_in_memory=data["keep_in_memory"],
        )
        return responses, 200


api.add_resource(Status, "/status")
api.add_resource(TextQA, "/text_qa")
api.add_resource(PrepareCaches, "/prepare_caches")
api.add_resource(ReleasePinnedKV, "/release_pinned_kv")
api.add_resource(TextQADirect, "/text_qa_direct")
api.add_resource(TextQAJoin, "/text_qa_join")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "from prompt length × model config on the same memory budget as the cache methods. "
        "Pass a positive value to force a fixed batch size (e.g. for A/B timing).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        choices=sorted(_registry.all_model_names()),
        default=MODEL_NAME,
        help="Name of the model to use",
    )
    parser.add_argument(
        "--press-name",
        type=str,
        choices=sorted(PRESS.keys()),
        default="expected_attention",
        help="Name of the compression press to use",
    )
    parser.add_argument(
        "--use-relative-indices",
        action="store_true",
        help="Path A: reconstruct a target comp{tag} cache on the fly from a baseline cache "
        "+ relative indices (indices/comp{tag}/_meta.json) when available; otherwise fall "
        "back to the physical cache, or error just that text. Default off.",
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
    model_name = args.model_name
    press_name = args.press_name
    vanilla_batch_size = args.vanilla_batch_size
    model_wrapper = KvTextQaModelWrapper(
        model_name,
        device_id,
        batch_sizes=[None, None, None, None, None, None, None, None, None, None],
        compression_ratios=[
            0.0,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.99,
        ],
        press_name=press_name,
        vanilla_batch_size=vanilla_batch_size,
        use_relative_indices=args.use_relative_indices,
    )
    app.run(host="127.0.0.1", port=PORT_KV_TEXT_QA.get(model_name), debug=False)
