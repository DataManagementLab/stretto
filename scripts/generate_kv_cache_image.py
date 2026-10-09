"""
Pre-generate KV caches (+ optional indices) for IMAGE datasets, mirroring
scripts/generate_kv_cache.py but for the image pipeline.

As in the text script, the vision model is selected BY the requested methods:
registry VL methods (kvLlava8B*, kvQwenVL8B*, kvMistralVL24B*, …) load their own
model, and the text-LLM-keyed methods (kv8B*/kv70B*) map to the matching
LLaVA model. Methods are grouped per model so each model is loaded once.

Writes the same directory layout as the text datasets:
  {CACHE_DIR}/{dataset}_{split}/kv-text-qa-cache/{model}/{press}/comp{tag}/cache_entry_{hash}.pt
  {CACHE_DIR}/{dataset}_{split}/kv-text-qa-cache/{model}/{press}/indices/comp{tag}/idx_{hash}.pt   (--indices)

A single prefill per image produces all requested CRs; with --indices the kept-token
indices come from that same prefill (bit-exact reconstruction).

Usage:
    python scripts/generate_kv_cache_image.py \\
        --benchmark artwork_random --kv-methods kvQwenVL8B00 kvQwenVL8B05 --device-id 0
    python scripts/generate_kv_cache_image.py \\
        --benchmark artwork_random --kv-methods kv8B00 kv8B05 --indices   # → LLaVA-8B

Timing-only runs (--tmp) write the caches to a throwaway directory and delete them once
the timing row is on disk, so already-materialized caches are neither read nor
overwritten and every (image, CR) pair is regenerated from scratch:

    python scripts/generate_kv_cache_image.py \\
        --benchmark artwork_random --kv-methods kvQwenVL8B05 \\
        --tmp tmp_caches --timing-csv benchmark_results/cache_generation_times_image_tmp.csv

Two --tmp runs of the same dataset+split must not share a --tmp directory concurrently:
they map to the same throwaway subtree and would delete each other's work.
"""

import argparse
import asyncio
import csv
import glob
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLAVA_8B = "llava-hf/llama3-llava-next-8b-hf"
LLAVA_72B = "llava-hf/llava-next-72b-hf"

DEFAULT_TIMING_CSV = Path("benchmark_results/cache_generation_times_image.csv")

# Columns of the timing CSV, in order. One row per (dataset, model) generation pass.
# Mirrors scripts/generate_kv_cache.py, with images as the unit of work.
TIMING_FIELDS = [
    "timestamp",
    "dataset",
    "split",
    "press_name",
    "model",
    "methods",
    "compression_ratios",
    "indices",
    "device_id",
    "n_images",
    "n_caches_expected",
    "n_caches_before",
    "n_caches_written",
    "n_errors",
    "model_load_seconds",
    "generation_seconds",
    "seconds_per_image",
    "seconds_per_cache_written",
    "load_images_seconds",
    "skipped_model_load",
    # Non-empty only for --tmp runs: the throwaway dir the caches were written to
    # (and deleted from), i.e. the row measures generation cost, not a materialization.
    "tmp_dir",
]

# dataset → where its images come from, logical column name. Dataset names match the
# benchmark names (e.g. artwork_random); the cache subdir is derived as {dataset}_{split},
# mirroring generate_kv_caches_image_indices.py.
#
# Entries with ``from_database`` read the image paths from the benchmark's own database
# column, the same values the operators send to the server, so the cache file names
# (hashes of those paths) match what the server looks up. The others glob a local images
# subdir under {CACHE_DIR}/{dataset}_{split}/.
IMAGE_DATASETS = {
    "artwork_random": {
        "images_subdir": "artworks_files",
        "column_name": "artworks.image",
    },
    "artwork_random_medium": {
        "from_database": True,
        "column_name": "artworks.image",
    },
    "ecommerce_random_large": {
        "from_database": True,
        "column_name": "products.product_image",
    },
}

from reasondb.config.model_registry import ModelRegistry as _ModelRegistry
from reasondb.utils.tmp_cache_dir import (
    DEFAULT_TMP_ROOT,
    TMP_HELP,
    dir_size_gb,
    purge_tmp_tree,
    setup_tmp_cache_dir,
)

# method tag → (model, compression_ratio); modality=None so both the text-LLM-keyed
# methods (kv8B*) and native VL methods (kvQwenVL8B*, …) resolve.
METHOD_CONFIG = _ModelRegistry.get().method_config(modality=None)

# Which vision model generates the caches for a given method's registry model.
# Registry-native VL models are their own vision model; the text-LLM-keyed
# methods (kv8B*/kv70B*) map to the matching LLaVA. Same mapping as
# generate_kv_caches_image_indices.py.
VISION_MODEL_FOR = {
    **{name: name for name in _ModelRegistry.get().all_model_names(modality="vision")},
    "meta-llama/Llama-3.1-8B-Instruct": LLAVA_8B,
    "meta-llama/Llama-3.1-70B-Instruct": LLAVA_72B,
}


def _compression_tag(cr: float) -> str:
    """Mirror of KvImageQaModelWrapper.to_compression_tag (pure dir-name formatting)."""
    return str(cr).replace(".", "_") if cr != 0.0 else "0"


def _hash_path(path: str) -> str:
    """Mirror of KvImageQaModelWrapper.hash_path (names the cache/index files)."""
    return hashlib.sha256(path.encode()).hexdigest()


def _comp_dir(cache_dir: str, model_name: str, press_name: str, cr: float) -> Path:
    """The server's per-CR cache directory: {cache_dir}/{model}/{press}/comp{tag}."""
    return Path(cache_dir) / model_name / press_name / f"comp{_compression_tag(cr)}"


def cache_status(
    cache_dir: str,
    model_name: str,
    press_name: str,
    compression_ratios,
    hashes: list[str],
    save_indices: bool,
) -> tuple[int, int, int]:
    """Inspect on-disk state for one model, without loading it.

    Returns (n_present, n_expected, n_todo) counted across every requested CR.
    An image counts as done when its cache exists, or when a previous run recorded a
    permanent generation error for it (ERRORS.json) — otherwise a dataset with a few
    unprocessable images could never be reported as complete. With ``save_indices`` the
    per-CR index file must exist too, matching prepare_caches_multi_cr's own condition
    for re-running an image.
    """
    n_present = 0
    n_expected = len(compression_ratios) * len(hashes)
    for cr in compression_ratios:
        cr_dir = _comp_dir(cache_dir, model_name, press_name, cr)
        try:
            with open(cr_dir / "ERRORS.json") as f:
                known_errors = set(json.load(f))
        except (OSError, ValueError):
            known_errors = set()
        idx_dir = cr_dir.parent / "indices" / cr_dir.name
        wants_idx = save_indices and cr > 0.0
        for h in hashes:
            if h in known_errors:
                n_present += 1
            elif (cr_dir / f"cache_entry_{h}.pt").exists() and (
                not wants_idx or (idx_dir / f"idx_{h}.pt").exists()
            ):
                n_present += 1
    return n_present, n_expected, n_expected - n_present


def load_image_paths_from_database(
    benchmark_name: str, split: str, column_name: str
) -> tuple[list[str], str]:
    """Load a benchmark's image paths through its own database.

    Preparing the database downloads remote images first, as at query time. Returns the
    distinct non-null values of ``column_name`` (exactly the paths the operators send to
    the server) and the benchmark's KV image cache dir.
    """
    from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS
    from reasondb.utils.logging import FileLogger

    # Benchmark.load() generates random queries as a side effect; they are not
    # needed for cache generation, so don't require filter stats for them.
    os.environ.setdefault("REASONDB_ALLOW_MISSING_FILTER_STATS", "1")
    benchmark = ALL_BENCHMARKS[benchmark_name].load(split)
    table_name, col = column_name.split(".")

    async def _paths_from_db() -> list[str]:
        await benchmark.database.prepare(FileLogger())
        rows = benchmark.database.sql(f"SELECT {col} FROM {table_name}").fetchall()
        await benchmark.database.wind_down()
        return sorted({str(Path(row[0])) for row in rows if row[0] is not None})

    paths = asyncio.run(_paths_from_db())
    return paths, str(benchmark.database.cache_dir) + "/kv-image-qa-cache"


def count_caches(cache_dir: str, model_name: str, press_name: str, compression_ratios) -> int:
    """Number of physical cache files on disk across the requested CRs."""
    return sum(
        len(list(_comp_dir(cache_dir, model_name, press_name, cr).glob("cache_entry_*.pt")))
        for cr in compression_ratios
    )


def count_errors(cache_dir: str, model_name: str, press_name: str, compression_ratios) -> int:
    """Distinct images that failed generation, as recorded in the per-CR ERRORS.json."""
    failed: set[str] = set()
    for cr in compression_ratios:
        cr_dir = _comp_dir(cache_dir, model_name, press_name, cr)
        try:
            with open(cr_dir / "ERRORS.json") as f:
                failed.update(json.load(f))
        except (OSError, ValueError):
            continue
    return len(failed)


def append_timing_row(csv_path: Path, row: dict) -> None:
    """Append one timing row, writing the header when the file is new.

    Written after each model rather than at the end so an interrupted campaign still
    leaves the timings of the models that did complete. A pre-existing CSV keeps its
    own header (columns added since it was created are dropped rather than silently
    shifting every value one column to the left).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    fields = TIMING_FIELDS
    if not is_new:
        with open(csv_path, newline="") as f:
            existing = next(csv.reader(f), None)
        if existing and existing != TIMING_FIELDS:
            fields = existing
            dropped = [k for k in TIMING_FIELDS if k not in existing]
            if dropped:
                logger.warning(
                    f"{csv_path} predates the columns {dropped}; they are not recorded. "
                    f"Point --timing-csv at a new file to capture them."
                )
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


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
    parser.add_argument("--press-name", type=str, default="expected_attention")
    parser.add_argument("--kv-methods", type=str, nargs="+", default=["kv8B00", "kv8B05"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--indices",
        action="store_true",
        help="Also save per-CR kept indices (from the same prefill) for bit-exact reconstruction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images (useful for quick verification). "
        "Mirrors --limit in scripts/generate_kv_caches_image_indices.py.",
    )
    parser.add_argument(
        "--timing-csv",
        type=Path,
        default=DEFAULT_TIMING_CSV,
        help=f"CSV to append per-model generation timings to (default: {DEFAULT_TIMING_CSV}).",
    )
    parser.add_argument(
        "--tmp",
        type=Path,
        nargs="?",
        const=DEFAULT_TMP_ROOT,
        default=None,
        metavar="DIR",
        help=TMP_HELP,
    )
    parser.add_argument(
        "--force-model-load",
        action="store_true",
        help="Load the model even when every requested cache already exists on disk. "
        "By default that model is skipped entirely (no weights loaded), which makes "
        "re-running a finished campaign nearly free.",
    )
    args = parser.parse_args()

    cfg = IMAGE_DATASETS[args.benchmark]

    # Group requested methods by VISION model so we only load each model once.
    model_to_methods: dict[str, list[str]] = defaultdict(list)
    for m in args.kv_methods:
        if m not in METHOD_CONFIG:
            logger.warning(f"Unknown method '{m}', skipping.")
            continue
        registry_model, _ = METHOD_CONFIG[m]
        vision_model = VISION_MODEL_FOR.get(registry_model)
        if vision_model is None:
            logger.warning(
                f"Method '{m}' maps to '{registry_model}', which has no vision model; skipping."
            )
            continue
        model_to_methods[vision_model].append(m)
    if not model_to_methods:
        raise SystemExit("No valid --kv-methods given.")

    from reasondb.utils.cache import CACHE_DIR

    t_images = time.perf_counter()
    if cfg.get("from_database"):
        image_paths, cache_dir = load_image_paths_from_database(
            args.benchmark, args.split, cfg["column_name"]
        )
        source = f"column {cfg['column_name']} of {args.benchmark}"
    else:
        # Local images + the text-like cache dir the test reads from.
        subdir = f"{args.benchmark}_{args.split}"
        images_dir = str(CACHE_DIR / subdir / cfg["images_subdir"])
        cache_dir = str(CACHE_DIR / subdir) + "/kv-image-qa-cache"
        image_paths = sorted(
            p for p in glob.glob(os.path.join(images_dir, "*")) if os.path.isfile(p)
        )
        source = images_dir
    load_images_seconds = time.perf_counter() - t_images
    logger.info(f"{len(image_paths)} images from {source}")
    if not image_paths:
        raise SystemExit(f"No images found in {source}")

    # --tmp: redirect every write (caches, indices, ERRORS.json — the wrapper puts all of
    # them under the cache dir it is given) into a throwaway tree. Images are still read
    # from their real location; only the generated caches move.
    tmp_root: Path | None = None
    tmp_stop_at: Path | None = None
    if args.tmp is not None:
        tmp_root, tmp_stop_at = setup_tmp_cache_dir(cache_dir, args.tmp)
        cache_dir = str(tmp_root)

    if args.limit is not None:
        image_paths = image_paths[: args.limit]
        logger.info(f"--limit {args.limit}: generating for {len(image_paths)} images only.")

    from reasondb.backends.kv_cache_image_qa_server import KvImageQaModelWrapper

    # Hashes name the cache files, so they are all that is needed to check on-disk state.
    hashes = [_hash_path(p) for p in image_paths]

    for model_name, methods in model_to_methods.items():
        crs = tuple(sorted({METHOD_CONFIG[m][1] for m in methods}))

        if tmp_root is not None:
            # An earlier interrupted --tmp run may have left this model's caches here;
            # start cold so the timing covers every image.
            purge_tmp_tree(tmp_root / model_name, tmp_root)

        n_present, n_expected, n_todo = cache_status(
            cache_dir, model_name, args.press_name, crs, hashes, args.indices
        )
        n_caches_before = count_caches(cache_dir, model_name, args.press_name, crs)
        logger.info(
            f"{model_name}: {n_present}/{n_expected} (image, CR) pairs already on disk; "
            f"{n_todo} to generate."
        )

        base_row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": args.benchmark,
            "split": args.split,
            "press_name": args.press_name,
            "model": model_name,
            "methods": " ".join(sorted(methods)),
            "compression_ratios": " ".join(str(cr) for cr in crs),
            "indices": int(args.indices),
            "device_id": args.device_id,
            "n_images": len(image_paths),
            "n_caches_expected": n_expected,
            "n_caches_before": n_caches_before,
            "load_images_seconds": round(load_images_seconds, 3),
            "tmp_dir": str(tmp_root) if tmp_root is not None else "",
        }

        # Nothing to do → don't pay the model load (minutes of weight I/O for a 72B).
        if n_todo == 0 and not args.force_model_load:
            logger.info(
                f"Skipping model={model_name}: all requested caches already exist "
                f"(use --force-model-load to load it anyway)."
            )
            append_timing_row(
                args.timing_csv,
                {
                    **base_row,
                    "n_caches_written": 0,
                    "n_errors": count_errors(cache_dir, model_name, args.press_name, crs),
                    "model_load_seconds": 0.0,
                    "generation_seconds": 0.0,
                    "seconds_per_image": "",
                    "seconds_per_cache_written": "",
                    "skipped_model_load": 1,
                },
            )
            # Nothing was generated, but the caches are all here - which is exactly when a
            # recording that predates them is cheapest to bring up to date.
            if tmp_root is None:
                record_sizes(
                    args.benchmark, args.split, args.press_name, model_name, cache_dir
                )
            continue

        logger.info(
            f"Loading model {model_name} | press={args.press_name} | "
            f"methods={methods} | CRs={crs}"
        )
        t_load = time.perf_counter()
        wrapper = KvImageQaModelWrapper(
            model_name=model_name,
            device_id=args.device_id,
            compression_ratios=crs,
            press_name=args.press_name,
        )
        model_load_seconds = time.perf_counter() - t_load
        logger.info(f"Model loaded in {model_load_seconds:.1f}s")

        logger.info(f"Writing caches to {cache_dir}/{model_name}/{args.press_name}/comp*")
        t_gen = time.perf_counter()
        asyncio.run(
            wrapper.prepare_caches_multi_cr(
                column_name=cfg["column_name"],
                image_paths=image_paths,
                cr_to_cache_dir={cr: cache_dir for cr in crs},
                save_indices=args.indices,
            )
        )
        generation_seconds = time.perf_counter() - t_gen

        n_written = count_caches(cache_dir, model_name, args.press_name, crs) - n_caches_before
        n_errors = count_errors(cache_dir, model_name, args.press_name, crs)
        # Per-image cost is only meaningful over the images that actually ran a prefill,
        # which is n_written / n_CRs (all CRs of an image come from one prefill).
        n_images_generated = n_written / len(crs) if crs else 0
        logger.info(
            f"Done with model={model_name}: {generation_seconds:.1f}s, "
            f"{n_written} cache files written, {n_errors} error(s)."
        )

        append_timing_row(
            args.timing_csv,
            {
                **base_row,
                "n_caches_written": n_written,
                "n_errors": n_errors,
                "model_load_seconds": round(model_load_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "seconds_per_image": (
                    round(generation_seconds / n_images_generated, 4)
                    if n_images_generated
                    else ""
                ),
                "seconds_per_cache_written": (
                    round(generation_seconds / n_written, 4) if n_written else ""
                ),
                "skipped_model_load": 0,
            },
        )

        # Per model rather than at the end, so an interrupted multi-model run still
        # leaves the models it finished recorded. --tmp caches are about to be deleted,
        # so recording them would describe caches that no longer exist.
        if tmp_root is None:
            record_sizes(args.benchmark, args.split, args.press_name, model_name, cache_dir)

        # Timing row is on disk → the caches themselves are disposable. Deleting per
        # model (not at the end) keeps peak usage at one model's worth of caches.
        if tmp_root is not None:
            model_tmp_dir = tmp_root / model_name
            logger.info(
                f"--tmp: deleting {dir_size_gb(model_tmp_dir):.2f} GB of throwaway "
                f"caches at {model_tmp_dir}"
            )
            purge_tmp_tree(model_tmp_dir, tmp_root)

    if tmp_root is not None:
        # Removes what is left at the top level (empty dirs).
        purge_tmp_tree(tmp_root, tmp_stop_at)
        logger.info(f"--tmp: throwaway caches deleted. Timings → {args.timing_csv}")
    else:
        logger.info(f"All image caches generated successfully. Timings → {args.timing_csv}")


if __name__ == "__main__":
    main()
