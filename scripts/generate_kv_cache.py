"""
Standalone script to pre-generate KV caches for TEXT datasets — the text counterpart
of scripts/generate_kv_cache_image.py.

The model is selected BY the requested methods (kv8B*, kvMistral8B*, kvQwen7B*, …);
methods are grouped per model so each model is loaded once. Texts and the cache
directory are obtained by loading the benchmark's own database, so both the
directory ({CACHE_DIR}/{db_name}_{split}/kv-text-qa-cache/{model}/{press}/comp{tag}/,
with db_name = benchmark class name lowercased, e.g. enronemailrandom) and the
text bytes (which determine the sha256 cache filenames) match exactly what
run_benchmark_single_operator.py's servers look up at serve time.

Usage example:
    python scripts/generate_kv_cache.py \
        --benchmark rotowire_random \
        --split dev \
        --press-name expected_attention \
        --kv-methods kv8B09 kv8B05 kv70B00 \
        --device-id 0

Timing-only runs (--tmp) write the caches to a throwaway directory and delete them
once the timing row is on disk, so already-materialized caches are neither read nor
overwritten and every (text, CR) pair is regenerated from scratch:

    python scripts/generate_kv_cache.py \
        --benchmark rotowire_random --kv-methods kv8B05 \
        --tmp tmp_caches --timing-csv benchmark_results/cache_generation_times_tmp.csv

Two --tmp runs of the same benchmark+split must not share a --tmp directory
concurrently: they map to the same throwaway subtree and would delete each other's work.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TIMING_CSV = Path("benchmark_results/cache_generation_times.csv")

# Columns of the timing CSV, in order. One row per (benchmark, model) generation pass.
TIMING_FIELDS = [
    "timestamp",
    "benchmark",
    "split",
    "press_name",
    "model",
    "methods",
    "compression_ratios",
    "indices",
    "device_id",
    "n_texts",
    "n_caches_expected",
    "n_caches_before",
    "n_caches_written",
    "n_errors",
    "model_load_seconds",
    "generation_seconds",
    "seconds_per_text",
    "seconds_per_cache_written",
    "load_texts_seconds",
    "skipped_model_load",
    # Non-empty only for --tmp runs: the throwaway dir the caches were written to
    # (and deleted from), i.e. the row measures generation cost, not a materialization.
    "tmp_dir",
]


# Maps benchmark name -> (CSV path, text column in CSV, column_name used at runtime)
BENCHMARK_DATASET_CONFIG = {
    "rotowire_random": (
        "reasondb/evaluation/benchmarks/files/reports.csv",
        "report",
        "reports.report",
    ),
    "movie_random": (
        "reasondb/evaluation/benchmarks/files/reviews_1000.csv",
        "reviewtext",
        "reviews.reviewtext",
    ),
    "movie_random_huge": (
        "reasondb/evaluation/benchmarks/files/reviews_10000.csv",
        "reviewtext",
        "reviews.reviewtext",
    ),
    "email_random": (
        "reasondb/evaluation/benchmarks/files/emails_with_cpt.csv",
        "text",
        "emails.text",
    ),
    "ecommerce_random_large": (
        "reasondb/evaluation/benchmarks/files/ecommerce_products_large.csv",
        "description",
        "products.description",
    ),
}

from reasondb.config.model_registry import ModelRegistry as _ModelRegistry
from reasondb.utils.tmp_cache_dir import (
    DEFAULT_TMP_ROOT,
    TMP_HELP,
    dir_size_gb,
    purge_tmp_tree,
    setup_tmp_cache_dir,
)

METHOD_CONFIG = _ModelRegistry.get().method_config()

AVAILABLE_PRESS = ["expected_attention", "kvzip", "finch", "finch-cachenotes"]


def _compression_tag(cr: float) -> str:
    """Mirror of KvTextQaModelWrapper.to_compression_tag (pure dir-name formatting)."""
    return str(cr).replace(".", "_") if cr != 0.0 else "0"


def _hash_text(text: str) -> str:
    """Mirror of KvTextQaModelWrapper.hash_text (names the cache/index files)."""
    return hashlib.sha256(text.encode()).hexdigest()


def _comp_dir(shared_cache_dir: str, model_name: str, press_name: str, cr: float) -> Path:
    """The server's per-CR cache directory: {cache_dir}/{model}/{press}/comp{tag}."""
    return Path(shared_cache_dir) / model_name / press_name / f"comp{_compression_tag(cr)}"


def cache_status(
    shared_cache_dir: str,
    model_name: str,
    press_name: str,
    compression_ratios,
    hashes: list[str],
    save_indices: bool,
) -> tuple[int, int, int]:
    """Inspect on-disk state for one model, without loading it.

    Returns (n_present, n_expected, n_todo) counted across every requested CR.
    A text counts as done when its cache exists, or when a previous run recorded a
    permanent generation error for it (ERRORS.json) — otherwise a dataset with a few
    unprocessable texts could never be reported as complete. With ``save_indices`` the
    per-CR index file must exist too, matching prepare_caches_multi_cr's own condition
    for re-running a text.
    """
    n_present = 0
    n_expected = len(compression_ratios) * len(hashes)
    for cr in compression_ratios:
        comp_dir = _comp_dir(shared_cache_dir, model_name, press_name, cr)
        try:
            with open(comp_dir / "ERRORS.json") as f:
                known_errors = set(json.load(f))
        except (OSError, ValueError):
            known_errors = set()
        idx_dir = comp_dir.parent / "indices" / comp_dir.name
        wants_idx = save_indices and cr > 0.0
        for h in hashes:
            if h in known_errors:
                n_present += 1
            elif (comp_dir / f"cache_entry_{h}.pt").exists() and (
                not wants_idx or (idx_dir / f"idx_{h}.pt").exists()
            ):
                n_present += 1
    return n_present, n_expected, n_expected - n_present


def count_caches(
    shared_cache_dir: str, model_name: str, press_name: str, compression_ratios
) -> int:
    """Number of physical cache files on disk across the requested CRs."""
    return sum(
        len(list(_comp_dir(shared_cache_dir, model_name, press_name, cr).glob("cache_entry_*.pt")))
        for cr in compression_ratios
    )


def count_errors(
    shared_cache_dir: str, model_name: str, press_name: str, compression_ratios
) -> int:
    """Distinct texts that failed generation, as recorded in the per-CR ERRORS.json."""
    failed: set[str] = set()
    for cr in compression_ratios:
        comp_dir = _comp_dir(shared_cache_dir, model_name, press_name, cr)
        try:
            with open(comp_dir / "ERRORS.json") as f:
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


def load_texts(benchmark_name: str, split: str) -> tuple[list[str], str, str]:
    """Load texts through the benchmark's own database — the same code path
    run_benchmark_single_operator.py uses at runtime. This guarantees both the
    cache directory ({db_name}_{split}) and the exact text bytes (and therefore
    the sha256 cache filenames) match what the servers will look up.

    Returns (texts, column_name, kv_cache_dir).
    """
    if benchmark_name not in BENCHMARK_DATASET_CONFIG:
        raise ValueError(
            f"No text dataset configured for benchmark '{benchmark_name}'. "
            f"Supported: {list(BENCHMARK_DATASET_CONFIG.keys())}"
        )
    column_name = BENCHMARK_DATASET_CONFIG[benchmark_name][2]

    from reasondb.evaluation.benchmarks.ecommerce import EcommerceRandomLarge
    from reasondb.evaluation.benchmarks.email import EnronEmailRandom
    from reasondb.evaluation.benchmarks.movie import MovieRandom, MovieRandomHuge
    from reasondb.evaluation.benchmarks.rotowire import RotowireRandom
    from reasondb.utils.logging import FileLogger

    benchmark_classes = {
        "rotowire_random": RotowireRandom,
        "movie_random": MovieRandom,
        "movie_random_huge": MovieRandomHuge,
        "email_random": EnronEmailRandom,
        "ecommerce_random_large": EcommerceRandomLarge,
    }

    # Benchmark.load() generates random queries as a side effect; they are not
    # needed for cache generation, so don't require filter stats for them.
    os.environ.setdefault("REASONDB_ALLOW_MISSING_FILTER_STATS", "1")
    benchmark = benchmark_classes[benchmark_name].load(split)

    table_name, col = column_name.split(".")

    async def _texts_from_db() -> list[str]:
        await benchmark.database.prepare(FileLogger())
        rows = benchmark.database.sql(f"SELECT {col} FROM {table_name}").fetchall()
        await benchmark.database.wind_down()
        return [row[0] for row in rows if row[0] is not None]

    texts = asyncio.run(_texts_from_db())
    kv_cache_dir = str(benchmark.database.cache_dir) + "/kv-text-qa-cache"
    logger.info(
        f"Loaded {len(texts)} texts from table {table_name} (column: {col}); "
        f"caches will be written to {kv_cache_dir}"
    )
    return texts, column_name, kv_cache_dir


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
    parser = argparse.ArgumentParser(
        description="Pre-generate KV caches using the same directory layout as run_benchmark_single_operator.py."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=list(BENCHMARK_DATASET_CONFIG.keys()),
        required=True,
        help="Benchmark to generate caches for.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "test"],
        default="dev",
        help="Dataset split (must match the split used during inference).",
    )
    parser.add_argument(
        "--press-name",
        type=str,
        choices=AVAILABLE_PRESS,
        default="expected_attention",
        help="KV compression press to use.",
    )
    parser.add_argument(
        "--kv-methods",
        type=str,
        nargs="+",
        default=list(METHOD_CONFIG.keys()),
        help="KV-cache methods to pre-generate caches for.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device ID to use.",
    )
    parser.add_argument(
        "--indices",
        action="store_true",
        help="Also save the per-CR kept-token indices (from the same prefill as the "
        "caches) as indices/comp{tag}/idx_{hash}.pt, for bit-exact masked reconstruction. "
        "With this flag, texts are (re)generated across all CRs so cache+indices share a "
        "prefill.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N texts (useful for quick verification). "
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

    t_texts = time.perf_counter()
    texts, column_name, shared_cache_dir = load_texts(args.benchmark, args.split)
    load_texts_seconds = time.perf_counter() - t_texts

    # --tmp: redirect every write (caches, indices, ERRORS.json, footprint YAMLs — the
    # wrapper puts all of them under the cache dir it is given) into a throwaway tree.
    tmp_root: Path | None = None
    tmp_stop_at: Path | None = None
    if args.tmp is not None:
        tmp_root, tmp_stop_at = setup_tmp_cache_dir(shared_cache_dir, args.tmp)
        shared_cache_dir = str(tmp_root)

    if args.limit is not None:
        texts = texts[: args.limit]
        logger.info(f"--limit {args.limit}: generating for {len(texts)} texts only.")

    # Group requested methods by model so we only load each model once
    from collections import defaultdict
    model_to_methods: dict[str, list[str]] = defaultdict(list)
    for method_name in args.kv_methods:
        if method_name not in METHOD_CONFIG:
            logger.warning(f"Unknown method '{method_name}', skipping.")
            continue
        model_name, _ = METHOD_CONFIG[method_name]
        model_to_methods[model_name].append(method_name)

    # Import here so GPU / model loading only happens when needed
    from reasondb.backends.kv_cache_text_qa_server import KvTextQaModelWrapper

    # Hashes name the cache files, so they are all that is needed to check on-disk state.
    hashes = [_hash_text(t) for t in texts]

    for model_name, methods in model_to_methods.items():
        # dedup: two methods can map to the same CR, and generating it twice is wasted work
        compression_ratios = tuple(sorted({METHOD_CONFIG[m][1] for m in methods}))

        if tmp_root is not None:
            # An earlier interrupted --tmp run may have left this model's caches here;
            # start cold so the timing covers every text.
            purge_tmp_tree(tmp_root / model_name, tmp_root)

        n_present, n_expected, n_todo = cache_status(
            shared_cache_dir,
            model_name,
            args.press_name,
            compression_ratios,
            hashes,
            args.indices,
        )
        n_caches_before = count_caches(
            shared_cache_dir, model_name, args.press_name, compression_ratios
        )
        logger.info(
            f"{model_name}: {n_present}/{n_expected} (text, CR) pairs already on disk; "
            f"{n_todo} to generate."
        )

        # Nothing to do → don't pay the model load (minutes of weight I/O for a 70B).
        if n_todo == 0 and not args.force_model_load:
            logger.info(
                f"Skipping model={model_name}: all requested caches already exist "
                f"(use --force-model-load to load it anyway)."
            )
            append_timing_row(
                args.timing_csv,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "benchmark": args.benchmark,
                    "split": args.split,
                    "press_name": args.press_name,
                    "model": model_name,
                    "methods": " ".join(sorted(methods)),
                    "compression_ratios": " ".join(str(cr) for cr in compression_ratios),
                    "indices": int(args.indices),
                    "device_id": args.device_id,
                    "n_texts": len(texts),
                    "n_caches_expected": n_expected,
                    "n_caches_before": n_caches_before,
                    "n_caches_written": 0,
                    "n_errors": count_errors(
                        shared_cache_dir, model_name, args.press_name, compression_ratios
                    ),
                    "model_load_seconds": 0.0,
                    "generation_seconds": 0.0,
                    "seconds_per_text": "",
                    "seconds_per_cache_written": "",
                    "load_texts_seconds": round(load_texts_seconds, 3),
                    "skipped_model_load": 1,
                    "tmp_dir": str(tmp_root) if tmp_root is not None else "",
                },
            )
            # Nothing was generated, but the caches are all here - which is exactly when a
            # recording that predates them is cheapest to bring up to date.
            if tmp_root is None:
                record_sizes(
                    args.benchmark, args.split, args.press_name, model_name,
                    shared_cache_dir,
                )
            continue

        logger.info(
            f"Loading model {model_name} | press={args.press_name} | "
            f"compression_ratios={compression_ratios}"
        )
        t_load = time.perf_counter()
        wrapper = KvTextQaModelWrapper(
            model_name=model_name,
            device_id=args.device_id,
            compression_ratios=compression_ratios,
            press_name=args.press_name,
        )
        model_load_seconds = time.perf_counter() - t_load
        logger.info(f"Model loaded in {model_load_seconds:.1f}s")

        # Build {cr: cache_dir} mapping.
        # All methods share the same top-level cache dir (from the benchmark's
        # database); model/press/comp differentiation happens inside the
        # server's subdirectory structure.
        cr_to_cache_dir = {cr: shared_cache_dir for cr in compression_ratios}
        logger.info(
            f"Generating caches for model={model_name} | "
            f"methods={methods} | cr_to_cache_dir={cr_to_cache_dir}"
        )
        # Single prefill per text → all CRs saved in one shot
        t_gen = time.perf_counter()
        asyncio.run(
            wrapper.prepare_caches_multi_cr(
                column_name=column_name,
                texts=texts,
                cr_to_cache_dir=cr_to_cache_dir,
                save_indices=args.indices,
            )
        )
        generation_seconds = time.perf_counter() - t_gen

        n_caches_after = count_caches(
            shared_cache_dir, model_name, args.press_name, compression_ratios
        )
        n_written = n_caches_after - n_caches_before
        n_errors = count_errors(
            shared_cache_dir, model_name, args.press_name, compression_ratios
        )
        # Per-text cost is only meaningful over the texts that actually ran a prefill,
        # which is n_written / n_CRs (all CRs of a text come from one prefill).
        n_texts_generated = n_written / len(compression_ratios) if compression_ratios else 0
        logger.info(
            f"Done with model={model_name}: {generation_seconds:.1f}s, "
            f"{n_written} cache files written, {n_errors} error(s)."
        )

        append_timing_row(
            args.timing_csv,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "benchmark": args.benchmark,
                "split": args.split,
                "press_name": args.press_name,
                "model": model_name,
                "methods": " ".join(sorted(methods)),
                "compression_ratios": " ".join(str(cr) for cr in compression_ratios),
                "indices": int(args.indices),
                "device_id": args.device_id,
                "n_texts": len(texts),
                "n_caches_expected": n_expected,
                "n_caches_before": n_caches_before,
                "n_caches_written": n_written,
                "n_errors": n_errors,
                "model_load_seconds": round(model_load_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "seconds_per_text": (
                    round(generation_seconds / n_texts_generated, 4)
                    if n_texts_generated
                    else ""
                ),
                "seconds_per_cache_written": (
                    round(generation_seconds / n_written, 4) if n_written else ""
                ),
                "load_texts_seconds": round(load_texts_seconds, 3),
                "skipped_model_load": 0,
                "tmp_dir": str(tmp_root) if tmp_root is not None else "",
            },
        )

        # Per model rather than at the end, so an interrupted multi-model run still
        # leaves the models it finished recorded. --tmp caches are about to be deleted,
        # so recording them would describe caches that no longer exist.
        if tmp_root is None:
            record_sizes(
                args.benchmark, args.split, args.press_name, model_name, shared_cache_dir
            )

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
        # Removes what is left at the top level (footprint YAMLs, empty dirs).
        purge_tmp_tree(tmp_root, tmp_stop_at)
        logger.info(f"--tmp: throwaway caches deleted. Timings → {args.timing_csv}")
    else:
        logger.info(f"All caches generated successfully. Timings → {args.timing_csv}")


if __name__ == "__main__":
    main()
