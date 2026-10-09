"""Build each dataset's filter stats out of its ``--precompute`` file, calling no model.

A ``RandomBenchmark`` samples its queries from a predicate-overlap matrix: which tuples
each pool filter keeps, according to the gold (vanilla 70B) model. A recording made by an
earlier ``--precompute`` pass already contains those answers, keyed by the question the
pinned operator config produced - so the matrix can be derived from the file instead of
asking the models for it again.

What it writes, per dataset:

  1. the ``filter_stats`` bucket, into the precompute JSON itself;
  2. ``benchmark_results/filter_stats/<benchmark>/<split>/stats.json``, to read by eye;
  3. ``data/<benchmark>/benchmark/<split>/queries.json`` - the query set drawn from the
     matrix, which every later job reads instead of sampling its own.

With (1) and (3) in place the coordinator enumerates no filter-stats job for that dataset
at all, so run this once before starting a task and the whole first stage disappears.

Requirements: no KV servers (the operators replay from the file). A benchmark with image
columns needs the two embedding servers up, because ``ImageEmbedFilter.prepare`` embeds
its images locally rather than replaying them. ``OPENAI_API_KEY`` is needed either way -
the configurator asks GPT-4o for each step's operator definitions before the recorded
pins overwrite their parameters.

An answer the recording does not hold is an error, not a fallback to a live model: an
incomplete file should be reported so it can be completed by a real ``--precompute`` run.

Examples
--------
    python scripts/filter_stats_from_precompute.py \\
        movie_random=movie_precompute_kv.json \\
        artwork_random_medium=artwork_precompute_kv.json

    # Recompute even though the file already carries a matrix
    python scripts/filter_stats_from_precompute.py --force movie_random=movie.json
"""

import argparse
import logging
from pathlib import Path

from reasondb.evaluation.benchmark import RandomBenchmark
from reasondb.evaluation.benchmark_registry import RANDOM_BENCHMARKS
from reasondb.evaluation.filter_stats import (
    derive_filter_stats_from_recordings,
    pin_query_set,
    write_stats_file,
)
from reasondb.utils.benchmark_args import parse_dataset_path_mapping

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="BENCH=PATH",
        help="The same mapping --precompute/--simulate take, one file per dataset.",
    )
    parser.add_argument("--split", type=str, choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the matrix even if the file already carries one. The query set is "
        "redrawn from it, which changes what every later job runs.",
    )
    args = parser.parse_args()

    mapping = parse_dataset_path_mapping(
        args.datasets,
        flag="datasets",
        known_benchmarks=RANDOM_BENCHMARKS,
        must_exist=True,
    )

    for benchmark_name, store_path in mapping.items():
        benchmark_cls = RANDOM_BENCHMARKS[benchmark_name]
        assert issubclass(benchmark_cls, RandomBenchmark), (
            f"{benchmark_name} is not a RandomBenchmark; only those sample their "
            "queries from filter stats."
        )
        logger.info("=== %s (%s) ===", benchmark_name, store_path)

        payload = derive_filter_stats_from_recordings(
            benchmark_cls, args.split, store_path, force=args.force
        )
        stats_path = write_stats_file(
            benchmark_cls.filter_stats_dir(args.split), payload
        )
        n_queries = pin_query_set(benchmark_cls, args.split, payload)

        covered = payload.get("covered_filters", {})
        for key, expressions in sorted(covered.items()):
            logger.info(
                "  pool %-8s %d filters covered", repr(key), len(expressions)
            )
        logger.info("  stats     -> %s", stats_path)
        logger.info(
            "  queries   -> %s (%d queries)",
            benchmark_cls.benchmark_dir() / args.split / "queries.json",
            n_queries,
        )
        logger.info(
            "  %s/%s is complete; the coordinator will enumerate no filter-stats job "
            "for it.",
            benchmark_name, args.split,
        )


if __name__ == "__main__":
    main()
