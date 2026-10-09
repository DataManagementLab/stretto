"""
Hierarchical / relative KV-cache index generation.

Unlike `generate_kv_cache.py` (which stores a full compressed cache per method, each
selected from the uncompressed kv8B00), this script stores exactly ONE physical baseline
cache for `--kv-method` (e.g. kv8B02) and, for each more-compressed `--index-methods`
(e.g. kv8B05 kv8B09), stores only the kept-token indices RELATIVE to that baseline. A
target cache is then reconstructed by gathering from the baseline (not kv8B00), so kv8B00
never has to be retained.

This is valid because all CRs rank tokens by the same single-prefill scores, so a
more-compressed method's kept set is a strict subset of a less-compressed one's; the
relative indices select exactly the same tokens the absolute method would, expressed as
rows of the baseline cache. Reconstruction works with the existing index machinery because
row i of any rerotated cache carries RoPE phase i.

Layout (under {CACHE_DIR}/{benchmark}_{split}/kv-text-qa-cache/{model}/{press}/):
    comp{base_tag}/cache_entry_{hash}.pt                — physical baseline cache
    comp{base_tag}/indices/comp{tgt_tag}/idx_{hash}.pt  — indices into the baseline cache
    comp{base_tag}/indices/comp{tgt_tag}/_meta.json     — {"from": "comp{base_tag}"}
Index dirs nest under their materialized baseline so the same effective target can be
indexed from several baselines at once; the servers also accept a flat
indices/comp{tgt_tag}/ layout (and migrate it in place).

Scope: expected_attention only (rerotation press; exact subset-nesting). finch and
kvzip are rejected.

Usage example:
    python scripts/generate_kv_caches_indices.py \
        --benchmark email_random --split dev --press-name expected_attention \
        --kv-method kv8B02 --index-methods kv8B05 kv8B09 --device-id 0
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
}

from reasondb.config.model_registry import ModelRegistry as _ModelRegistry

METHOD_CONFIG = _ModelRegistry.get().method_config()

# Only the rerotation press is supported, since there subset-nesting is exact.
SUPPORTED_PRESS = ["expected_attention"]


def load_texts(benchmark_name: str, split: str) -> tuple[list[str], str, str]:
    """Load texts through the benchmark's own database, exactly as
    scripts/generate_kv_cache.py does. Returns (texts, column_name, kv_cache_dir).

    The sha256 of the text bytes names the cache and index files, and the database is
    what the operator reads and pushes to /prepare_caches, so loading from it (rather
    than from the raw CSV) makes generation agree with serving by construction —
    including the cache directory, which the database also owns.
    """
    if benchmark_name not in BENCHMARK_DATASET_CONFIG:
        raise ValueError(
            f"No text dataset configured for benchmark '{benchmark_name}'. "
            f"Supported: {list(BENCHMARK_DATASET_CONFIG.keys())}"
        )
    column_name = BENCHMARK_DATASET_CONFIG[benchmark_name][2]

    from reasondb.evaluation.benchmarks.email import EnronEmailRandom
    from reasondb.evaluation.benchmarks.movie import MovieRandom, MovieRandomHuge
    from reasondb.evaluation.benchmarks.rotowire import RotowireRandom
    from reasondb.utils.logging import FileLogger

    benchmark_classes = {
        "rotowire_random": RotowireRandom,
        "movie_random": MovieRandom,
        "movie_random_huge": MovieRandomHuge,
        "email_random": EnronEmailRandom,
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
        f"baseline caches + indices will be written under {kv_cache_dir}"
    )
    return texts, column_name, kv_cache_dir


def resolve_and_validate(kv_method: str, index_methods: list[str]) -> tuple[str, float, list[float]]:
    """Resolve methods to (model, cr) and enforce the relative-compression constraints.

    Returns (base_model, base_cr, sorted_index_crs). Raises SystemExit with a clear
    message on any violation.
    """
    if kv_method not in METHOD_CONFIG:
        raise SystemExit(f"Unknown --kv-method '{kv_method}'. Known: {sorted(METHOD_CONFIG)}")
    base_model, base_cr = METHOD_CONFIG[kv_method]
    if base_cr <= 0.0:
        # Uncompressed baseline: mathematically exact (identity rerotation; the relative
        # indices degenerate to absolute positions in the full cache), but it stores the
        # FULL kv*00 cache per item — allowed for experiments, so warn instead of block.
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
            bad_model.append(f"{m} (model={m_model})")
        elif m_cr <= base_cr:
            bad_cr.append(f"{m} (cr={m_cr})")
        else:
            index_crs.append(m_cr)

    if bad_unknown:
        raise SystemExit(f"Unknown --index-methods: {bad_unknown}. Known: {sorted(METHOD_CONFIG)}")
    if bad_model:
        raise SystemExit(
            f"--index-methods must share the baseline model ({base_model}). Offenders: {bad_model}"
        )
    if bad_cr:
        raise SystemExit(
            f"--index-methods must be strictly MORE compressed than the baseline "
            f"'{kv_method}' (cr={base_cr}). Offenders: {bad_cr}"
        )
    if not index_crs:
        raise SystemExit("No valid --index-methods after validation.")

    return base_model, base_cr, sorted(set(index_crs))


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
        "--benchmark", "--dataset", dest="benchmark", type=str,
        choices=list(BENCHMARK_DATASET_CONFIG.keys()), required=True,
        help="Benchmark to generate for. --dataset is kept as a deprecated alias.",
    )
    parser.add_argument("--split", type=str, choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/filter_stats")
    )
    parser.add_argument(
        "--press-name", type=str, choices=SUPPORTED_PRESS, default="expected_attention",
        help="Only expected_attention is supported in v1.",
    )
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
        help="Only process the first N texts (useful for quick verification).",
    )
    args = parser.parse_args()

    base_model, base_cr, index_crs = resolve_and_validate(args.kv_method, args.index_methods)

    result_dir = args.output_dir / args.benchmark / args.press_name / args.split
    result_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Result dir: {result_dir}")

    texts, column_name, shared_cache_dir = load_texts(args.benchmark, args.split)
    if args.limit is not None:
        texts = texts[: args.limit]
        logger.info(f"Limiting to first {len(texts)} texts.")

    # Import here so GPU / model loading only happens when needed
    from reasondb.backends.kv_cache_text_qa_server import KvTextQaModelWrapper

    logger.info(
        f"Baseline {args.kv_method} (cr={base_cr}) | index targets cr={index_crs} | "
        f"model={base_model} | press={args.press_name}"
    )
    wrapper = KvTextQaModelWrapper(
        model_name=base_model,
        device_id=args.device_id,
        compression_ratios=(base_cr, *index_crs),
        press_name=args.press_name,
    )

    logger.info(f"Writing baseline cache + relative indices under {shared_cache_dir}")

    asyncio.run(
        wrapper.prepare_indices_relative(
            column_name=column_name,
            texts=texts,
            cache_dir=shared_cache_dir,
            base_cr=base_cr,
            index_crs=index_crs,
        )
    )
    logger.info("All baseline caches + relative indices generated successfully.")
    # The relative indices are recorded with the baseline they sit under, so a replay
    # under --use-indexes gets the footprint the live walk would have given it.
    record_sizes(args.benchmark, args.split, args.press_name, base_model, shared_cache_dir)


if __name__ == "__main__":
    main()
