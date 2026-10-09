"""Compute one benchmark's filter stats outside the coordinator.

The coordinator normally schedules this as a phase-0 job per benchmark (see
``reasondb/coordinator/producers/filter_stats_jobs.py``): a ``RandomBenchmark``'s query
set is sampled from the predicate-overlap matrix it produces, so nothing downstream can be
enumerated before it. This script regenerates the stats of a single dataset without
starting a coordinator task.

The work is done by :func:`reasondb.evaluation.filter_stats.run_filter_stats_pass`, which
needs a ``--precompute`` file to write into: the pass pins the question phrasing and
records the gold responses into that file, so that later runs ask exactly the prompts the
matrix describes. Requires the model servers to be up.
"""

import argparse
import logging
from pathlib import Path

from reasondb.evaluation.benchmark import RandomBenchmark
from reasondb.evaluation.benchmark_registry import RANDOM_BENCHMARKS
from reasondb.evaluation.filter_stats import run_filter_stats_pass

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmark", type=str, choices=sorted(RANDOM_BENCHMARKS), required=True
    )
    parser.add_argument("--split", type=str, choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--precompute",
        type=Path,
        required=True,
        metavar="OUTPUT_JSON",
        help="This dataset's precompute file. Read as well as written: an existing file "
        "supplies the operator config pins, so a re-run asks the same questions it did "
        "the first time.",
    )
    args = parser.parse_args()

    benchmark_cls = RANDOM_BENCHMARKS[args.benchmark]
    assert issubclass(benchmark_cls, RandomBenchmark), (
        f"{args.benchmark} is not a RandomBenchmark; only those sample their queries "
        "from filter stats."
    )

    summary = run_filter_stats_pass(
        benchmark_cls,
        args.split,
        simulate=False,
        store_path=args.precompute,
        stats_dir=benchmark_cls.filter_stats_dir(args.split),
    )
    logger.info("Done: %s", summary)


if __name__ == "__main__":
    main()
