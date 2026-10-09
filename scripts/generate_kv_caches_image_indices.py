"""
Hierarchical / relative KV-cache index generation for IMAGE datasets (llava).

Image analogue of scripts/generate_kv_caches_indices.py (text), built on the image pipeline
of scripts/generate_kv_cache_image.py. Stores exactly ONE physical baseline cache for
`--kv-method` (e.g. kv8B02) and, for each more-compressed `--index-methods` (e.g. kv8B05),
only the kept-token indices RELATIVE to that baseline — so a target cache is reconstructed
by gathering from the baseline, and the uncompressed kv8B00 never has to be stored.

Layout (under {CACHE_DIR}/{dataset}_{split}/kv-image-qa-cache/{model}/{press}/):
    comp{base_tag}/cache_entry_{hash}.pt                — physical baseline cache
    comp{base_tag}/indices/comp{tgt_tag}/idx_{hash}.pt  — indices into the baseline cache
    comp{base_tag}/indices/comp{tgt_tag}/_meta.json     — {"from": "comp{base_tag}"}
Index dirs nest under their materialized baseline so the same effective target can be
indexed from several baselines at once; the servers also accept a flat
indices/comp{tgt_tag}/ layout (and migrate it in place).

Scope: expected_attention only. As in generate_kv_cache_image.py, the method tags
(kv8B*) supply only the compression ratio; the model is the dataset's vision model (llava).

Usage example:
    python scripts/generate_kv_caches_image_indices.py \
        --benchmark artwork_random --press-name expected_attention \
        --kv-method kv8B02 --index-methods kv8B05 kv8B09 --device-id 0
"""

import argparse
import asyncio
import csv
import glob
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLAVA_8B = "llava-hf/llama3-llava-next-8b-hf"
LLAVA_72B = "llava-hf/llava-next-72b-hf"

from reasondb.evaluation.benchmarks.artwork import ArtworkRandom, ArtworkRandomMedium

# dataset → local images subdir, logical column name, CSV (for computing which images are
# actually referenced), and the CSV column holding the image URL. Dataset names match the
# text generator / benchmark names (e.g. artwork_random); the cache subdir is derived as
# {dataset}_{split}, mirroring generate_kv_caches_indices.py.
# The vision model is NOT fixed per dataset: it is derived from the requested methods
# (kv8B* → 8B, kv70B* → 72B) via VISION_MODEL_FOR below.
IMAGE_DATASETS = {
    "artwork_random": {
        "images_subdir": "artworks_files",  # local image files under {CACHE_DIR}/{dataset}_{split}/
        "column_name": "artworks.image",
        "csv_path": ArtworkRandom.get_csv(),
        "url_column": "image_url",
    },
    "artwork_random_medium": {
        "images_subdir": "artworks_files",
        "column_name": "artworks.image",
        "csv_path": ArtworkRandomMedium.get_csv(),
        "url_column": "image_url",
    },
}

from reasondb.config.model_registry import ModelRegistry as _ModelRegistry

# method tag → (model, compression_ratio); modality=None so both the text-LLM-keyed
# methods (kv8B*/kv70B*) and native VL methods (kvQwenVL8B*, …) resolve.
METHOD_CONFIG = _ModelRegistry.get().method_config(modality=None)

# Which vision model generates the caches for a given method's registry model.
# Registry-native VL models (kvLlava*, kvQwenVL*, kvMistralVL*, …) are their own vision
# model; the text-LLM-keyed methods (kv8B*/kv70B*) map to the matching LLaVA.
VISION_MODEL_FOR = {
    **{name: name for name in _ModelRegistry.get().all_model_names(modality="vision")},
    "meta-llama/Llama-3.1-8B-Instruct": LLAVA_8B,
    "meta-llama/Llama-3.1-70B-Instruct": LLAVA_72B,
}

SUPPORTED_PRESS = ["expected_attention"]


def resolve_and_validate(kv_method: str, index_methods: list[str]) -> tuple[str, float, list[float]]:
    """Resolve methods to (registry_model, base_cr, index_crs) and enforce the constraints.

    All methods must map to a single registry model (don't mix e.g. kv8B/kv70B); that model
    then selects the LLaVA vision model to load (see VISION_MODEL_FOR). The compression ratios
    come from METHOD_CONFIG. Returns (base_model, base_cr, sorted_index_crs). Raises SystemExit
    with a clear message on violation.
    """
    if kv_method not in METHOD_CONFIG:
        raise SystemExit(f"Unknown --kv-method '{kv_method}'. Known: {sorted(METHOD_CONFIG)}")
    base_model, base_cr = METHOD_CONFIG[kv_method]
    if base_cr <= 0.0:
        # Uncompressed baseline: mathematically exact (identity rerotation; the relative
        # indices degenerate to absolute positions in the full cache), but it stores the
        # FULL kv*00 cache per image — allowed for experiments, so warn instead of block.
        logger.warning(
            f"--kv-method '{kv_method}' has cr={base_cr}: the UNCOMPRESSED cache will be "
            f"stored as the physical baseline (large storage footprint)."
        )

    index_crs: list[float] = []
    bad_unknown, bad_model, bad_cr = [], [], []
    for m in index_methods:
        if m not in METHOD_CONFIG:
            bad_unknown.append(m)
            continue
        m_model, m_cr = METHOD_CONFIG[m]
        if m_model != base_model:
            bad_model.append(f"{m} (registry-model={m_model})")
        elif m_cr <= base_cr:
            bad_cr.append(f"{m} (cr={m_cr})")
        else:
            index_crs.append(m_cr)

    if bad_unknown:
        raise SystemExit(f"Unknown --index-methods: {bad_unknown}. Known: {sorted(METHOD_CONFIG)}")
    if bad_model:
        raise SystemExit(
            f"--index-methods must share the baseline's registry model ({base_model}). "
            f"Offenders: {bad_model}"
        )
    if bad_cr:
        raise SystemExit(
            f"--index-methods must be strictly MORE compressed than the baseline "
            f"'{kv_method}' (cr={base_cr}). Offenders: {bad_cr}"
        )
    if not index_crs:
        raise SystemExit("No valid --index-methods after validation.")

    return base_model, base_cr, sorted(set(index_crs))


def referenced_image_basenames(csv_path: Path, url_column: str) -> set[str]:
    """Basenames of images the CSV references, keyed the same way as the download path
    (reasondb/database/external_table.py:download_file: `url.split("/")[-1]`)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row[url_column].split("/")[-1] for row in csv.DictReader(f)}


def record_sizes(benchmark: str, split: str, press_name: str, model_name: str, written_under) -> None:
    """Record what this model's caches now weigh, so a replay can plan without them.

    See ``reasondb.evaluation.parameter_sweep.record_generated_levels``; it logs and
    swallows its own failures, so this never fails a generation run.
    """
    from reasondb.evaluation.parameter_sweep import record_generated_levels

    record_generated_levels(
        benchmark, split, press_name, [model_name], written_under=Path(written_under)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", "--dataset", dest="benchmark",
        choices=list(IMAGE_DATASETS.keys()), default="artwork_random",
        help="Benchmark to generate for. --dataset is kept as a deprecated alias.",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--press-name", type=str, choices=SUPPORTED_PRESS, default="expected_attention")
    parser.add_argument(
        "--kv-method", type=str, required=True,
        help="Single baseline method stored as a physical cache, e.g. kv8B02.",
    )
    parser.add_argument(
        "--index-methods", type=str, nargs="+", required=True,
        help="More-compressed methods stored as relative indices only, e.g. kv8B05 kv8B09.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N images (useful for quick verification).",
    )
    args = parser.parse_args()

    base_model, base_cr, index_crs = resolve_and_validate(args.kv_method, args.index_methods)

    cfg = IMAGE_DATASETS[args.benchmark]
    if base_model not in VISION_MODEL_FOR:
        raise SystemExit(
            f"No LLaVA vision model mapped for registry model '{base_model}' "
            f"(from method '{args.kv_method}'). Known: {sorted(VISION_MODEL_FOR)}"
        )
    model_name = VISION_MODEL_FOR[base_model]

    from reasondb.utils.cache import CACHE_DIR

    subdir = f"{args.benchmark}_{args.split}"  # same convention as generate_kv_caches_indices.py
    images_dir = str(CACHE_DIR / subdir / cfg["images_subdir"])
    # Must match the image server's cache root (vision_model appends /kv-image-qa-cache),
    # otherwise serving looks for the relative indices in a different tree and never finds them.
    cache_dir = str(CACHE_DIR / subdir) + "/kv-image-qa-cache"

    image_paths = sorted(p for p in glob.glob(os.path.join(images_dir, "*")) if os.path.isfile(p))
    logger.info(f"{len(image_paths)} local images in {images_dir}")
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    referenced = referenced_image_basenames(cfg["csv_path"], cfg["url_column"])
    local_basenames = {os.path.basename(p) for p in image_paths}
    missing = referenced - local_basenames  # referenced in the CSV but not downloaded locally
    extra = local_basenames - referenced  # present locally but not referenced by the CSV
    if missing:
        logger.warning(
            f"{len(missing)} image(s) referenced in {cfg['csv_path']} are missing from "
            f"{images_dir}, e.g. {sorted(missing)[:5]}"
        )
    if extra:
        logger.warning(
            f"{len(extra)} image(s) in {images_dir} are not referenced in {cfg['csv_path']} "
            f"and will be skipped, e.g. {sorted(extra)[:5]}"
        )
    if missing or extra:
        image_paths = [p for p in image_paths if os.path.basename(p) in referenced]
        logger.info(f"{len(image_paths)} images remain after intersecting with the CSV")

    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images left after intersecting local files with {cfg['csv_path']}")

    from reasondb.backends.kv_cache_image_qa_server import KvImageQaModelWrapper

    logger.info(
        f"Baseline {args.kv_method} (cr={base_cr}) | index targets cr={index_crs} | "
        f"model={model_name} | press={args.press_name}"
    )
    wrapper = KvImageQaModelWrapper(
        model_name=model_name,
        device_id=args.device_id,
        compression_ratios=(base_cr, *index_crs),
        press_name=args.press_name,
    )
    logger.info(f"Writing baseline cache + relative indices under {cache_dir}")

    asyncio.run(
        wrapper.prepare_indices_relative(
            column_name=cfg["column_name"],
            image_paths=image_paths,
            cache_dir=cache_dir,
            base_cr=base_cr,
            index_crs=index_crs,
        )
    )
    logger.info("All baseline image caches + relative indices generated successfully.")
    # The relative indices are recorded with the baseline they sit under, so a replay
    # under --use-indexes gets the footprint the live walk would have given it.
    record_sizes(args.benchmark, args.split, args.press_name, model_name, cache_dir)


if __name__ == "__main__":
    main()
