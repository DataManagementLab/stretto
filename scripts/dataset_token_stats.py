"""Pin how long a benchmark's items are, in tokens, and how many of them there are.

These statistics normalize KV cache footprints: differences in storage cost between
benchmarks are largely explained by their row counts and item lengths.

The counts are pinned into the package rather than recomputed, so that
``scripts/plot_sweep.py`` can use them on a machine without the datasets and without
re-running any experiment.

Needs the benchmark files and a tokenizer - a vocabulary, not a network, so this is
CPU-only and loads no model weights. The Llama tokenizers are gated on Hugging Face; log
in first, or point ``--tokenizer`` at any tokenizer that shares the vocabulary.

    python scripts/dataset_token_stats.py                      # every random benchmark
    python scripts/dataset_token_stats.py --benchmarks movie_random --dry-run

Re-runnable and additive: a benchmark that is not named keeps whatever was pinned for it.
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from reasondb.evaluation import dataset_stats
from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS, RANDOM_BENCHMARKS
from reasondb.interface.default_operator_toolbox import TEXT_MODEL_8B

logger = logging.getLogger("dataset_token_stats")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmarks", nargs="+", choices=sorted(ALL_BENCHMARKS),
        help="Default: every random benchmark, which is what the sweeps run.",
    )
    parser.add_argument("--split", type=str, choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--tokenizer", type=str, default=TEXT_MODEL_8B,
        help="The 8B and 70B text models share one vocabulary, so one pass covers both.",
    )
    parser.add_argument(
        "--output", type=Path, default=dataset_stats.TOKEN_STATS_PATH,
        help="Where to pin the result. The default is what the plotting layer reads.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report the statistics, write nothing."
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Resolved through the registry so a hyphenated alias and its underscored name write
    # one entry rather than two describing the same files.
    names = args.benchmarks or sorted(RANDOM_BENCHMARKS)
    resolved = {}
    for name in names:
        benchmark_cls = ALL_BENCHMARKS[name]
        resolved.setdefault(benchmark_cls, benchmark_cls.name())

    payload = {"benchmarks": {}}
    if args.output.exists():
        try:
            payload = json.loads(args.output.read_text())
            payload.setdefault("benchmarks", {})
        except ValueError:
            logger.warning("%s is not readable JSON; starting a fresh file.", args.output)

    for benchmark_cls, canonical in resolved.items():
        logger.info("Measuring %s ...", canonical)
        stats = dataset_stats.compute_benchmark_stats(
            benchmark_cls, args.split, tokenizer
        )
        headline = dataset_stats.tokens_per_item({canonical: stats}, canonical)
        logger.info(
            "%s: %s tuples, %s, %s",
            canonical,
            stats["num_tuples"],
            stats["modality"],
            (
                f"{headline['tokens_per_item']:.0f} tokens/item "
                f"(p95 {headline['tokens_per_item_p95']:.0f})"
                if headline
                else "no text columns"
            ),
        )
        payload["benchmarks"][canonical] = stats

    payload["generated"] = date.today().isoformat()
    payload["tokenizer"] = args.tokenizer

    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote %s.", args.output)


if __name__ == "__main__":
    main()
