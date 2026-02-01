import argparse
import json
import numpy as np
import logging
from pathlib import Path

from reasondb.evaluation.benchmark import RandomBenchmark
from reasondb.evaluation.benchmarks.artwork import ArtworkRandom, ArtworkRandomMedium
from reasondb.evaluation.benchmarks.ecommerce import (
    EcommerceRandom,
    EcommerceRandomLarge,
)
from reasondb.evaluation.benchmarks.email import EnronEmailRandom
from reasondb.evaluation.benchmarks.movie import MovieRandom
from reasondb.evaluation.benchmarks.rotowire import RotowireRandom
from reasondb.executor import Executor
from reasondb.interface.config import get_default_configurator
from reasondb.optimizer.label_optimizer import LabelOptimizer
from reasondb.query_plan.logical_plan import ALL_LOGICAL_OPERATORS_TOOLBOX
from reasondb.reasoning.few_shot_database import DUMMY_FEW_SHOT_DATABASE
from reasondb.reasoning.llm import GPT4o
from reasondb.reasoning.reasoners.self_correction import SelfCorrectionReasoner

logger = logging.getLogger(__name__)


BENCHMARKS = {
    # CAESURA
    "artwork_random": ArtworkRandom,
    "artwork_random_medium": ArtworkRandomMedium,
    "rotowire_random": RotowireRandom,
    "movie_random": MovieRandom,
    "email_random": EnronEmailRandom,
    "ecommerce_random": EcommerceRandom,
    "ecommerce_random_large": EcommerceRandomLarge,
}


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=BENCHMARKS.keys(),
        default=["artwork_random"],
        help="The benchmark to run.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "test"],
        default="dev",
        help="The split of the benchmark to run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/filter_stats")
    )
    args = parser.parse_args()

    logger.info(f"Analyzing benchmarks: {args.benchmark}")
    benchmark_name = args.benchmark

    result_dir = args.output_dir / benchmark_name / args.split
    result_dir.mkdir(parents=True, exist_ok=True)

    benchmark_class = BENCHMARKS[benchmark_name]
    split = args.split
    print(f"Analyzing benchmark {benchmark_name} on split {split}...")
    benchmark = benchmark_class.load(split)
    assert isinstance(benchmark, RandomBenchmark)

    configurator = get_default_configurator()
    reasoner = SelfCorrectionReasoner(
        llm=GPT4o(),
        configurator=configurator,
        logical_operators=ALL_LOGICAL_OPERATORS_TOOLBOX,
        few_shot_database=DUMMY_FEW_SHOT_DATABASE,
    )
    executor = Executor(
        name="silver",
        database=benchmark.database,
        reasoner=reasoner,
        optimizer=LabelOptimizer(),
        configurator=configurator,
    )
    with executor as e:
        queries = benchmark.single_filter_queries
        benchmark_result = e.execute_benchmark(
            queries,
            results_cache_dir=result_dir / "cache",
            reset_db_before_each_query=True,
        )

    query_to_index = {}
    all_index = set()
    for i, (query, df) in enumerate(benchmark_result.results.items()):
        logger.info(f"{i}) Num results: {len(df)} - Query: {query}")
        query_to_index[query] = df.index
        all_index = all_index.union(set(df.index.values))
    all_index_sorted = sorted(list(all_index))
    index_to_id = {idx: i for i, idx in enumerate(all_index_sorted)}
    matrix = np.zeros((len(benchmark_result.results), len(all_index)), dtype=int)
    query_to_matrix_id = {}
    matrix_id_to_query = {}
    for i, (query, df) in enumerate(benchmark_result.results.items()):
        for idx in df.index.values:
            matrix_id = index_to_id[idx]
            matrix[i, matrix_id] = 1
        query_to_matrix_id[query] = i
        matrix_id_to_query[i] = query

    output_json = {
        "overlap_matrix": matrix.tolist(),
        "predicate_to_matrix_id": query_to_matrix_id,
        "matrix_id_to_predicate": matrix_id_to_query,
    }
    with open(result_dir / "stats.json", "w") as f:
        json.dump(output_json, f)


if __name__ == "__main__":
    main()
