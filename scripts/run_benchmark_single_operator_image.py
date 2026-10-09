import argparse
import json
import os
import numpy as np
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from reasondb.evaluation.benchmark import RandomBenchmark
from reasondb.evaluation.single_operator_queries import (
    get_extract_single_operator_queries,
    get_join_single_operator_queries,
)
from reasondb.evaluation.benchmarks.artwork import ArtworkRandom
from reasondb.evaluation.benchmarks.email import EnronEmailRandom
from reasondb.evaluation.benchmarks.movie import MovieRandom
from reasondb.evaluation.benchmarks.rotowire import RotowireRandom
from reasondb.executor import CostSummary, Executor
from reasondb.interface.config import get_default_configurator
from reasondb.utils.benchmark_args import (
    add_benchmark_argument,
    add_labels_argument,
    add_materialized_cr_argument,
    add_output_dir_argument,
    add_single_operator_arguments,
    add_split_argument,
    add_use_indexes_argument,
    add_monitor_arguments,
)
from reasondb.monitor.session import monitor_session
from reasondb.optimizer.label_optimizer import LabelOptimizer
from reasondb.query_plan.logical_plan import ALL_LOGICAL_OPERATORS_TOOLBOX
from reasondb.query_plan.physical_operator import CostType, PhysicalOperatorToolbox
from reasondb.reasoning.few_shot_database import DUMMY_FEW_SHOT_DATABASE
from reasondb.reasoning.llm import GPT4o, GPT4oMini
from reasondb.reasoning.reasoners.self_correction import SelfCorrectionReasoner
from reasondb.evaluation.metrics.metrics_manager import MetricsManager
from reasondb.operators.aggregate.aggregate import Aggregate
from reasondb.operators.aggregate.groupby import GroupBy
from reasondb.operators.limit.limit import Limit
from reasondb.operators.perfect_operators.perfect_extract import PerfectExtract
from reasondb.operators.perfect_operators.perfect_filter import PerfectFilter
from reasondb.operators.perfect_operators.perfect_transform import PerfectTransform
from reasondb.operators.project.project import Project
from reasondb.operators.rename.rename import Rename
from reasondb.operators.sorting.sort import Sort
from reasondb.operators.filter.traditional_filter import TraditionalFilter
from reasondb.operators.join.traditional_join import TraditionalJoin
from reasondb.operators.join.qa_filter_join import QaFilterJoin
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.backends.image_qa import VisionModelImageQABackend
from reasondb.backends.python_codegen import LLMPythonCodegenBackend
from reasondb.backends.text_qa import KvTextQABackend, LLMTextQABackend
from reasondb.backends.vision_model import KvVisionModel, LlmVisionModel
from reasondb.operators.extract.image_qa_extract import ImageQaExtract
from reasondb.operators.extract.python_extract import PythonExtract
from reasondb.operators.extract.text_qa_extract import TextQaExtract
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.operators.filter.text_qa_filter import TextQaFilter
from reasondb.operators.filter.extract_and_match import ExtractAndMatchFilter
from reasondb.operators.filter.extract_and_qa_filter import ExtractAndQaFilter
from reasondb.operators.filter.extract_and_match_image import ExtractAndMatchImageFilter
from reasondb.operators.filter.extract_and_qa_image import ExtractAndQaImageFilter
from reasondb.operators.filter.image_qa_join_filter import ImageQaJoinFilter
from reasondb.operators.filter.raw_text_qa_filter import RawTextQaFilter
from reasondb.operators.filter.kv_hidden_state_filter import KvHiddenStateFilter
from reasondb.operators.tranform.python_transform import PythonTransform
from reasondb.backends.text_embeddings import TextSimilarityBackend

logger = logging.getLogger(__name__)


BENCHMARKS = {
    # CAESURA
    "artwork_random": ArtworkRandom,
    "rotowire_random": RotowireRandom,
    "movie_random": MovieRandom,
    "email_random": EnronEmailRandom,
}

DATASET_LENGTHS = {
    "artwork_random": 65,
    "rotowire_random": 1000,
    "movie_random": 1000,
    "email_random": 1000,
}


def get_all_single_operator_queries(benchmark: RandomBenchmark):
    """
    Generate queries for all single operators (both filters and extracts) 
    from the benchmark's operator options.
    """
    from reasondb.query_plan.query import Queries, QueryShape
    from reasondb.query_plan.logical_plan import LogicalFilter, LogicalExtract
    from reasondb.database.indentifier import VirtualTableIdentifier
    from reasondb.query_plan.query import OperatorPlaceholder
    import random
    
    random.seed(42)
    queries = []
    
    # Get the base table name from the benchmark's single filter shape
    single_filter_shapes = benchmark.single_filter_shape()
    # Get the first shape to extract the base table name
    first_shape = next(iter(single_filter_shapes.values()))
    base_table_name = first_shape.shape[0].inputs[0].name
    
    # Get operator options for the benchmark
    operator_options = benchmark.get_operator_options()
    
    # Process each category of operators
    for _, ops_by_type in operator_options.items():
        # Handle LogicalFilter operators
        if LogicalFilter in ops_by_type:
            filter_shape = QueryShape(
                OperatorPlaceholder(
                    LogicalFilter,
                    inputs=[VirtualTableIdentifier(base_table_name)],
                    output=VirtualTableIdentifier("output"),
                ),
            )
            for option in ops_by_type[LogicalFilter]:
                query = filter_shape.instantiate({LogicalFilter: [option]})
                queries.append(query)
        
        # Handle LogicalExtract operators
        if LogicalExtract in ops_by_type:
            extract_shape = QueryShape(
                OperatorPlaceholder(
                    LogicalExtract,
                    inputs=[VirtualTableIdentifier(base_table_name)],
                    output=VirtualTableIdentifier("output"),
                ),
            )
            for option in ops_by_type[LogicalExtract]:
                query = extract_shape.instantiate({LogicalExtract: [option]})
                queries.append(query)
    
    return Queries(*queries)






def postprocess_string(s: str) -> str:
    """
    Post-process a single string value:
    1. Convert to lowercase
    2. Remove " and ' symbols
    3. Remove "STRING:" prefix if present
    4. Strip leading/trailing whitespace
    """
    if not isinstance(s, str):
        return s
    
    # 1. Convert to lowercase
    s = s.lower()
    
    # 2. Remove " and ' symbols
    s = s.replace('"', '').replace("'", '')
    
    # 3. Remove "STRING:" prefix if present
    if s.startswith('string:'):
        s = s[7:]  # Remove "string:" (7 characters)
    
    # 4. Strip leading/trailing whitespace
    s = s.strip()
    
    return s


def postprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process a dataframe by applying the following transformations to string columns:
    1. Drop the auxiliary 'cpt' column if present (emitted by some cache presses;
       must not leak into evaluation or debug outputs)
    2. Convert to lowercase
    3. Remove " and ' symbols
    4. Remove "STRING:" prefix if present
    5. Strip leading/trailing whitespace (keep it between words)
    """
    df = df.copy()
    
    # Drop the auxiliary column so it doesn't interfere with metrics or
    # get mistakenly renamed to 'answer' by _rename_last_col_to_answer
    if "cpt" in df.columns:
        df = df.drop(columns=["cpt"])
    
    for col in df.columns:
        # Only process columns that contain strings
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            df[col] = df[col].apply(lambda x: postprocess_string(x) if isinstance(x, str) else x)
    
    return df


def _rename_last_col_to_answer(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the last column of a DataFrame to 'answer' so that
    every debug CSV has a uniform column name for the extracted/filtered value.

    Some operators can legitimately return an empty DataFrame with zero columns.
    In that case there is no last column to rename.
    """
    df = df.copy()

    # Prevent IndexError on df.columns[-1] when there are zero columns
    if df.columns.size == 0:
        # Optional but useful: keep schema consistent for debug CSV consumers
        df["answer"] = pd.Series(dtype="object")
        return df

    last_col = df.columns[-1]
    if last_col != "answer":
        df = df.rename(columns={last_col: "answer"})
    return df


def _get_join_subdir(configurator) -> str:
    """Return a short directory name for the highest-quality join predicate."""
    preds = list(configurator.physical_operators.join_predicates) + list(configurator.physical_operators.filter_operators)
    if not preds:
        return "unknown"
    best = max(preds, key=lambda op: op.quality)
    return type(best).__name__.lower().replace("filter", "")


def save_debug_csvs(
    method_name: str,
    all_results: dict,
    query_to_id: dict,
    debug_base_dir: Path,
    join_subdir: str = "",
):
    """Save one CSV per query for a given method under
    ``debug_base_dir / method_name / [join_subdir] /``.  The last column of every
    DataFrame is renamed to ``answer``."""
    method_dir = debug_base_dir / method_name
    if join_subdir:
        method_dir = method_dir / join_subdir
    method_dir.mkdir(parents=True, exist_ok=True)

    for query, df in all_results.items():
        query_id = query_to_id[query]
        df_processed = postprocess_dataframe(df)
        df_out = _rename_last_col_to_answer(df_processed)
        csv_file = method_dir / f"{query_id}.csv"
        df_out.to_csv(csv_file, index=True)

    logger.info(f"Debug CSVs for {method_name} written to {method_dir}")


def evaluate_results(
    approach_name: str,
    all_predictions: dict,
    all_labels: dict,
    all_costs: dict,
    debug_dir: Path,
) -> pd.DataFrame:
    """
    Evaluate predictions against labels and compute metrics.
    """
    all_metrics = dict()
    for query, ground_truth_df in all_labels.items():
        predictions_df = all_predictions[query]
        
        # Post-process ground truth and predictions
        ground_truth_df_processed = postprocess_dataframe(ground_truth_df)
        predictions_df_processed = postprocess_dataframe(predictions_df)
        
        # Evaluate metrics
        evaluator = MetricsManager(predictions_df_processed, ground_truth_df_processed)
        metrics = evaluator.evaluate_all()
        
        # Add cost information
        cost = all_costs[query]
        for cost_type in CostType:
            metrics[f"execution_cost_{cost_type.value}"] = (
                cost.execution_cost.get_cost(cost_type)
            )
            metrics[f"tuning_cost_{cost_type.value}"] = cost.tuning_cost.get_cost(
                cost_type
            )
            metrics[f"total_cost_{cost_type.value}"] = cost.total_cost.get_cost(
                cost_type
            )
        
        all_metrics[approach_name, query] = metrics
    
    if not all_metrics:
        raise ValueError("No metrics were computed. Check the inputs.")
    
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.names = ["approach_name", "query"]
    
    return metrics_df


def get_label_configurator():
    """Configurator for gold/perfect labels."""
    return PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[],
            filter_operators=[PerfectFilter(quality=1, fake_cost=0)],
            extract_operators=[PerfectExtract(quality=1, fake_cost=0)],
            transform_operators=[PerfectTransform(quality=1, fake_cost=0)],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )


def get_kv8B09_configurator():
    """KV-cache configurator with 0.9 compression ratio."""
    kv09_cost = 0.1
    kvtextqa8B_backendcr09 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", effective_compression_ratio=0.9, materialized_compression_ratio=0.9
    )
    kvimageqa_backendcr09 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", effective_compression_ratio=0.9, materialized_compression_ratio=0.9)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.9, materialized_compression_ratio=0.9,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv09_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.9, materialized_compression_ratio=0.9,
                        )
                    ),
                    quality=5,
                    fake_cost=kv09_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.9, materialized_compression_ratio=0.9,
                        )
                    ),
                    quality=3,
                    fake_cost=kv09_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa_backendcr09, quality=3, fake_cost=kv09_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv8B05_configurator():
    """KV-cache configurator with 0.5 compression ratio."""
    kv05_cost = 0.6
    kvtextqa8B_backendcr05 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", effective_compression_ratio=0.5, materialized_compression_ratio=0.5
    )
    kvimageqa_backendcr05 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", effective_compression_ratio=0.5, materialized_compression_ratio=0.5)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.5, materialized_compression_ratio=0.5,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv05_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.5, materialized_compression_ratio=0.5,
                        )
                    ),
                    quality=5,
                    fake_cost=kv05_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.5, materialized_compression_ratio=0.5,
                        )
                    ),
                    quality=5,
                    fake_cost=kv05_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa_backendcr05, quality=5, fake_cost=kv05_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv8B099_configurator():
    """KV-cache configurator with 0.99 compression ratio."""
    kv099_cost = 0.01
    kvtextqa8B_backendcr099 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", effective_compression_ratio=0.99, materialized_compression_ratio=0.99
    )
    kvimageqa_backendcr099 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", effective_compression_ratio=0.99, materialized_compression_ratio=0.99)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.99, materialized_compression_ratio=0.99,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv099_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.99, materialized_compression_ratio=0.99,
                        )
                    ),
                    quality=5,
                    fake_cost=kv099_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.99, materialized_compression_ratio=0.99,
                        )
                    ),
                    quality=1,
                    fake_cost=kv099_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa_backendcr099, quality=1, fake_cost=kv099_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv8B08_configurator():
    """KV-cache configurator with 0.8 compression ratio."""
    kv08_cost = 0.3
    kvtextqa8B_backendcr08 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", effective_compression_ratio=0.8, materialized_compression_ratio=0.8
    )
    kvimageqa_backendcr08 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", effective_compression_ratio=0.8, materialized_compression_ratio=0.8)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.8, materialized_compression_ratio=0.8,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv08_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.8, materialized_compression_ratio=0.8,
                        )
                    ),
                    quality=5,
                    fake_cost=kv08_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.8, materialized_compression_ratio=0.8,
                        )
                    ),
                    quality=4,
                    fake_cost=kv08_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa_backendcr08, quality=4, fake_cost=kv08_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv8B06_configurator():
    """KV-cache configurator with 0.6 compression ratio."""
    kv06_cost = 0.45
    kvtextqa8B_backendcr06 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", effective_compression_ratio=0.6, materialized_compression_ratio=0.6
    )
    kvimageqa_backendcr06 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", effective_compression_ratio=0.6, materialized_compression_ratio=0.6)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.6, materialized_compression_ratio=0.6,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv06_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.6, materialized_compression_ratio=0.6,
                        )
                    ),
                    quality=5,
                    fake_cost=kv06_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.6, materialized_compression_ratio=0.6,
                        )
                    ),
                    quality=4,
                    fake_cost=kv06_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa_backendcr06, quality=4, fake_cost=kv06_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv8B04_configurator():
    """KV-cache configurator with 0.4 compression ratio."""
    kv04_cost = 0.75
    kvtextqa8B_backendcr04 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", effective_compression_ratio=0.4, materialized_compression_ratio=0.4
    )
    kvimageqa_backendcr04 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", effective_compression_ratio=0.4, materialized_compression_ratio=0.4)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.4, materialized_compression_ratio=0.4,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv04_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.4, materialized_compression_ratio=0.4,
                        )
                    ),
                    quality=5,
                    fake_cost=kv04_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.4, materialized_compression_ratio=0.4,
                        )
                    ),
                    quality=5,
                    fake_cost=kv04_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa_backendcr04, quality=5, fake_cost=kv04_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv8B03_configurator():
    """KV-cache configurator with 0.3 compression ratio."""
    kv03_cost = 0.85
    kvtextqa8B_backendcr03 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", effective_compression_ratio=0.3, materialized_compression_ratio=0.3
    )
    kvimageqa_backendcr03 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", effective_compression_ratio=0.3, materialized_compression_ratio=0.3)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.3, materialized_compression_ratio=0.3,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv03_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.3, materialized_compression_ratio=0.3,
                        )
                    ),
                    quality=5,
                    fake_cost=kv03_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.3, materialized_compression_ratio=0.3,
                        )
                    ),
                    quality=6,
                    fake_cost=kv03_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa_backendcr03, quality=6, fake_cost=kv03_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv8B00_configurator():
    """KV-cache configurator with 0.0 compression ratio (no compression)."""
    kv00_cost = 0.9
    kvtextqa8B_backendcr00 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", effective_compression_ratio=0.0, materialized_compression_ratio=0.0
    )
    kvimageqa_backendcr00 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", effective_compression_ratio=0.0, materialized_compression_ratio=0.0)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.0, materialized_compression_ratio=0.0,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv00_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.0, materialized_compression_ratio=0.0,
                        )
                    ),
                    quality=5,
                    fake_cost=kv00_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            effective_compression_ratio=0.0, materialized_compression_ratio=0.0,
                        )
                    ),
                    quality=3,
                    fake_cost=kv00_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa_backendcr00, quality=5, fake_cost=kv00_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_gpt_configurator():
    """GPT-4o Mini configurator."""
    gpt_cost = 2
    textqa_backend = LLMTextQABackend(GPT4oMini())
    imageqa_backend = VisionModelImageQABackend(LlmVisionModel(GPT4oMini()))
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(LlmVisionModel(GPT4oMini())),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=gpt_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(LlmVisionModel(GPT4oMini())),
                    quality=5,
                    fake_cost=gpt_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(LlmVisionModel(GPT4oMini())),
                    quality=10,
                    fake_cost=gpt_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(imageqa_backend, quality=10, fake_cost=gpt_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv70B09_configurator():
    """KV-cache 70B configurator with 0.9 compression ratio."""
    kv70B09_cost = 0.1 + 0.5
    kvtextqa70B_backendcr09 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", effective_compression_ratio=0.9, materialized_compression_ratio=0.9
    )
    kvimageqa70B_backendcr09 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.9, materialized_compression_ratio=0.9)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.9, materialized_compression_ratio=0.9,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv70B09_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.9, materialized_compression_ratio=0.9,
                        )
                    ),
                    quality=5,
                    fake_cost=kv70B09_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.9, materialized_compression_ratio=0.9,
                        )
                    ),
                    quality=7,
                    fake_cost=kv70B09_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa70B_backendcr09, quality=7, fake_cost=kv70B09_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv70B099_configurator():
    """KV-cache 70B configurator with 0.99 compression ratio."""
    kv70B099_cost = 0.01 + 0.5
    kvtextqa70B_backendcr099 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", effective_compression_ratio=0.99, materialized_compression_ratio=0.99
    )
    kvimageqa70B_backendcr099 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.99, materialized_compression_ratio=0.99)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.99, materialized_compression_ratio=0.99,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv70B099_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.99, materialized_compression_ratio=0.99,
                        )
                    ),
                    quality=5,
                    fake_cost=kv70B099_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.99, materialized_compression_ratio=0.99,
                        )
                    ),
                    quality=5,
                    fake_cost=kv70B099_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa70B_backendcr099, quality=5, fake_cost=kv70B099_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv70B08_configurator():
    """KV-cache 70B configurator with 0.8 compression ratio."""
    kv70B08_cost = 0.3 + 0.5
    kvtextqa70B_backendcr08 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", effective_compression_ratio=0.8, materialized_compression_ratio=0.8
    )
    kvimageqa70B_backendcr08 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.8, materialized_compression_ratio=0.8)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.8, materialized_compression_ratio=0.8,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv70B08_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.8, materialized_compression_ratio=0.8,
                        )
                    ),
                    quality=5,
                    fake_cost=kv70B08_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.8, materialized_compression_ratio=0.8,
                        )
                    ),
                    quality=8,
                    fake_cost=kv70B08_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa70B_backendcr08, quality=8, fake_cost=kv70B08_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv70B06_configurator():
    """KV-cache 70B configurator with 0.6 compression ratio."""
    kv70B06_cost = 0.45 + 0.5
    kvtextqa70B_backendcr06 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", effective_compression_ratio=0.6, materialized_compression_ratio=0.6
    )
    kvimageqa70B_backendcr06 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.6, materialized_compression_ratio=0.6)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.6, materialized_compression_ratio=0.6,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv70B06_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.6, materialized_compression_ratio=0.6,
                        )
                    ),
                    quality=5,
                    fake_cost=kv70B06_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.6, materialized_compression_ratio=0.6,
                        )
                    ),
                    quality=8,
                    fake_cost=kv70B06_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa70B_backendcr06, quality=8, fake_cost=kv70B06_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv70B05_configurator():
    """KV-cache 70B configurator with 0.5 compression ratio."""
    kv70B05_cost = 0.6 + 0.5
    kvtextqa70B_backendcr05 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", effective_compression_ratio=0.5, materialized_compression_ratio=0.5
    )
    kvimageqa70B_backendcr05 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.5, materialized_compression_ratio=0.5)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.5, materialized_compression_ratio=0.5,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv70B05_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.5, materialized_compression_ratio=0.5,
                        )
                    ),
                    quality=5,
                    fake_cost=kv70B05_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.5, materialized_compression_ratio=0.5,
                        )
                    ),
                    quality=9,
                    fake_cost=kv70B05_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa70B_backendcr05, quality=9, fake_cost=kv70B05_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv70B04_configurator():
    """KV-cache 70B configurator with 0.4 compression ratio."""
    kv70B04_cost = 0.75 + 0.5
    kvtextqa70B_backendcr04 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", effective_compression_ratio=0.4, materialized_compression_ratio=0.4
    )
    kvimageqa70B_backendcr04 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.4, materialized_compression_ratio=0.4)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.4, materialized_compression_ratio=0.4,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv70B04_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.4, materialized_compression_ratio=0.4,
                        )
                    ),
                    quality=5,
                    fake_cost=kv70B04_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.4, materialized_compression_ratio=0.4,
                        )
                    ),
                    quality=9,
                    fake_cost=kv70B04_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa70B_backendcr04, quality=9, fake_cost=kv70B04_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv70B03_configurator():
    """KV-cache 70B configurator with 0.3 compression ratio."""
    kv70B03_cost = 0.85 + 0.5
    kvtextqa70B_backendcr03 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", effective_compression_ratio=0.3, materialized_compression_ratio=0.3
    )
    kvimageqa70B_backendcr03 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.3, materialized_compression_ratio=0.3)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.3, materialized_compression_ratio=0.3,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv70B03_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.3, materialized_compression_ratio=0.3,
                        )
                    ),
                    quality=5,
                    fake_cost=kv70B03_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.3, materialized_compression_ratio=0.3,
                        )
                    ),
                    quality=10,
                    fake_cost=kv70B03_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa70B_backendcr03, quality=10, fake_cost=kv70B03_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_kv70B00_configurator():
    """KV-cache 70B configurator with 0.0 compression ratio (no compression)."""
    kv70B00_cost = 0.9 + 0.5
    kvtextqa70B_backendcr00 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", effective_compression_ratio=0.0, materialized_compression_ratio=0.0
    )
    kvimageqa70B_backendcr00 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.0, materialized_compression_ratio=0.0)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.0, materialized_compression_ratio=0.0,
                        )
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=kv70B00_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.0, materialized_compression_ratio=0.0,
                        )
                    ),
                    quality=5,
                    fake_cost=kv70B00_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            effective_compression_ratio=0.0, materialized_compression_ratio=0.0,
                        )
                    ),
                    quality=7,
                    fake_cost=kv70B00_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa70B_backendcr00, quality=9, fake_cost=kv70B00_cost),
            ],
            transform_operators=[
                PythonTransform(
                    LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0
                )
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def get_vanilla70B_configurator():
    """Vanilla 70B configurator: full prefill at inference, no pre-generated KV caches.
    Sets vanilla=True so its cache dir is separate from kv70B00."""
    vanilla_cost = 0.9 + 0.5
    kvimageqa70B_vanilla = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.0, materialized_compression_ratio=0.0, vanilla=True)
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.0, materialized_compression_ratio=0.0, vanilla=True)
                    ),
                    text_similarity_backend,
                    quality=4,
                    fake_cost=vanilla_cost,
                ),
                ExtractAndQaImageFilter(
                    VisionModelImageQABackend(
                        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.0, materialized_compression_ratio=0.0, vanilla=True)
                    ),
                    quality=5,
                    fake_cost=vanilla_cost,
                ),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel("llava-hf/llava-next-72b-hf", effective_compression_ratio=0.0, materialized_compression_ratio=0.0, vanilla=True)
                    ),
                    quality=7,
                    fake_cost=vanilla_cost,
                ),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa70B_vanilla, quality=9, fake_cost=vanilla_cost),
            ],
            transform_operators=[
                PythonTransform(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0)
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )
    return configurator


def make_kv_image_configurator(
    model_name: str,
    compression_ratio: float,
    cost: float,
    size_class: str,
    vanilla: bool = False,
    materialized_cr: Optional[float] = None,
):
    """Registry-driven factory for image KV-cache configurators — the image analogue of
    the text script's make_kv_configurator. Mirrors the hardcoded per-CR llava blocks:
    the image QA filter/extract qualities follow the model's size class (large=7, small=3),
    join predicates and the Python operators are identical across models. ``vanilla=True``
    (CR 0.0 only) runs full prefill with no stored caches; the backend's model_id gets a
    ``-vanilla`` suffix so its result cache stays separate from the kv*00 method.

    ``materialized_cr`` is the ratio the physical cache on disk was generated at (the
    ``--kv-method`` baseline of generate_kv_caches_image_indices.py). The server derives
    the relative-index dir from it — ``{press}/comp{materialized}/indices/comp{effective}``
    — so it must name the baseline, not the effective ratio. Clamping with ``min`` keeps
    ``effective >= materialized`` (asserted server-side) and makes one value correct for a
    whole run: the baseline method itself and every vanilla configurator clamp to their own
    ratio and load the physical cache, while more-compressed methods index out of the
    baseline. ``None`` (the default) means materialized == effective, i.e. physical
    serving. It must NOT default to 0.0, which would
    clamp every method down to a cr-0 baseline."""
    backend = VisionModelImageQABackend(
        KvVisionModel(
            model_name,
            effective_compression_ratio=compression_ratio,
            materialized_compression_ratio=(
                compression_ratio
                if materialized_cr is None
                else min(materialized_cr, compression_ratio)
            ),
            vanilla=vanilla,
        )
    )
    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")
    image_quality = 7 if size_class == "large" else 3
    return PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0), QaFilterJoin()],
            join_predicates=[
                ExtractAndMatchImageFilter(
                    backend, text_similarity_backend, quality=4, fake_cost=cost
                ),
                ExtractAndQaImageFilter(backend, quality=5, fake_cost=cost),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(backend, quality=image_quality, fake_cost=cost),
            ],
            extract_operators=[
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(backend, quality=image_quality, fake_cost=cost),
            ],
            transform_operators=[
                PythonTransform(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0)
            ],
            limit_operators=[Limit(quality=1, fake_cost=0)],
            project_operators=[Project(quality=1, fake_cost=0)],
            sorting_operators=[Sort(quality=1, fake_cost=0)],
            groupby_operators=[GroupBy(quality=1, fake_cost=0)],
            aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
            rename_operators=[Rename(quality=1, fake_cost=0)],
        ),
    )


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    add_benchmark_argument(parser, BENCHMARKS.keys(), ["artwork_random"])
    add_split_argument(parser)
    add_use_indexes_argument(parser)
    add_materialized_cr_argument(parser)
    add_output_dir_argument(parser, Path("benchmark_results"))
    add_labels_argument(parser, ["silver"])
    add_single_operator_arguments(
        parser,
        kv_methods=[
            "kv8B09", "kv8B095", "kv8B099", "kv8B0995", "kv8B08", "kv8B06", "kv8B05", "kv8B04", "kv8B03", "kv8B00",
            "kv70B09", "kv70B095", "kv70B099", "kv70B0995", "kv70B08", "kv70B06", "kv70B05", "kv70B04", "kv70B03", "kv70B00",
            "gpt"
        ],
    )
    add_monitor_arguments(parser)
    args = parser.parse_args()

    with monitor_session(args, output_dir=args.output_dir, script="run_benchmark_single_operator_image.py"):
        _run(args)


def _run(args):

    # Convert --sample percentage to an absolute row count using the known dataset length.
    if args.sample is not None:
        if not (0 < args.sample <= 100):
            parser.error("--sample must be between 0 (exclusive) and 100 (inclusive)")
        dataset_total = DATASET_LENGTHS.get(args.benchmark)
        if dataset_total is None:
            parser.error(f"Unknown dataset length for benchmark '{args.benchmark}'")
        num_items = max(1, int(dataset_total * args.sample / 100))
    else:
        num_items = None

    if args.only_join:
        os.environ["REASONDB_FORCE_JOIN"] = "1"

    # Validate mutually exclusive options
    if args.all_operators and args.only_extracts:
        parser.error("--all-operators and --only-extracts cannot be used together")
    if args.only_join and (args.all_operators or args.only_extracts):
        parser.error("--only-join cannot be used with --all-operators or --only-extracts")

    # Ensure the reference method is always included
    if args.reference_method not in args.kv_methods:
        logger.warning("Adding %s to methods list as it's required as the reference model", args.reference_method)
        args.kv_methods.append(args.reference_method)

    logger.info(f"Analyzing benchmarks: {args.benchmark}")
    benchmark_name = args.benchmark

    # Determine stats subdirectory from operator mode
    if args.only_extracts:
        stats_subdir = "extract_stats"
    elif args.only_join:
        stats_subdir = "join_stats"
    elif args.all_operators:
        stats_subdir = "all_stats"
    else:
        stats_subdir = "filter_stats"

    # Start from --output-dir, optionally suffix with sample pct, then append stats subdir
    base = args.output_dir
    if args.sample is not None:
        base = base.parent / f"{base.name}_{args.sample:g}pct"
    output_dir = base / stats_subdir

    logger.info(f"Output directory: {output_dir}")

    result_dir = output_dir / benchmark_name / args.press_name / args.split
    result_dir.mkdir(parents=True, exist_ok=True)

    benchmark_class = BENCHMARKS[benchmark_name]
    split = args.split
    print(f"Analyzing benchmark {benchmark_name} on split {split}...")
    benchmark = benchmark_class.load(split)
    assert isinstance(benchmark, RandomBenchmark)

    if num_items is not None:
        _original_prepare = benchmark.database.prepare

        async def _prepare_limited(logger):
            await _original_prepare(logger)
            conn = benchmark.database._connection
            for table in benchmark.database.external_tables:
                conn.execute(f"DROP VIEW IF EXISTS {table.name};")
                conn.execute(
                    f"CREATE VIEW {table.name} AS "
                    f"SELECT * FROM {table.name}.{table.name} LIMIT {num_items};"
                )
            logger.info(__name__, f"Applied LIMIT {num_items} to all database views.")

        benchmark.database.prepare = _prepare_limited
        logger.info("Database views will be limited to first %d rows (%.4g%% of %d).",
                    num_items, args.sample, DATASET_LENGTHS.get(benchmark_name, 0))

    configurator = get_default_configurator(use_indexes=args.use_indexes)
    reasoner = SelfCorrectionReasoner(
        llm=GPT4o(),
        configurator=configurator,
        logical_operators=ALL_LOGICAL_OPERATORS_TOOLBOX,
        few_shot_database=DUMMY_FEW_SHOT_DATABASE,
    )

    # Get single operator queries with optional range filtering
    from reasondb.query_plan.query import Queries as _Queries

    need_ranges = args.range_filter is not None or args.range_extract is not None

    if args.only_join:
        logger.info("Running only join operators")
        queries = get_join_single_operator_queries(benchmark)
    elif need_ranges:
        filter_qs = list(benchmark.single_filter_queries)
        extract_qs = list(get_extract_single_operator_queries(benchmark))
        if args.range_filter:
            s, e = args.range_filter
            filter_qs = filter_qs[s - 1 : e]
            logger.info("Applying --range-filter %d %d: %d filter queries", s, e, len(filter_qs))
        if args.range_extract:
            s, e = args.range_extract
            extract_qs = extract_qs[s - 1 : e]
            logger.info("Applying --range-extract %d %d: %d extract queries", s, e, len(extract_qs))
        if args.only_extracts:
            combined = extract_qs
        elif args.range_extract and not args.range_filter:
            combined = extract_qs
        elif args.range_filter and not args.range_extract:
            combined = filter_qs
        else:
            combined = filter_qs + extract_qs
        queries = _Queries(*combined)
    elif args.all_operators:
        logger.info("Running ALL single operators (filters + extracts)")
        queries = get_all_single_operator_queries(benchmark)
    elif args.only_extracts:
        logger.info("Running only extract operators")
        queries = get_extract_single_operator_queries(benchmark)
    else:
        logger.info("Running only filter operators")
        queries = benchmark.single_filter_queries

    logger.info(f"Total number of queries to run: {len(queries)}")

    # Define configurators for different methods
    method_configurators = {

        "silver": configurator,
        "gold": get_label_configurator(),
        "kv8B09": get_kv8B09_configurator(),
        "kv8B05": get_kv8B05_configurator(),
        "kv8B099": get_kv8B099_configurator(),
        "kv8B08": get_kv8B08_configurator(),
        "kv8B06": get_kv8B06_configurator(),
        "kv8B04": get_kv8B04_configurator(),
        "kv8B03": get_kv8B03_configurator(),
        "kv8B00": get_kv8B00_configurator(),
        "kv70B09": get_kv70B09_configurator(),
        "kv70B099": get_kv70B099_configurator(),
        "kv70B08": get_kv70B08_configurator(),
        "kv70B06": get_kv70B06_configurator(),
        "kv70B05": get_kv70B05_configurator(),
        "kv70B04": get_kv70B04_configurator(),
        "kv70B03": get_kv70B03_configurator(),
        "kv70B00": get_kv70B00_configurator(),
        "vanilla70B": get_vanilla70B_configurator(),
    }

    # Registry-driven VL methods (kvLlava8B*, kvQwenVL8B*, kvMistralVL24B*, …): every
    # vision model in the registry gets a configurator per compression ratio. The llava
    # entries above keep their kv8B*/kv70B* method names (result and debug dirs are keyed
    # by them), so setdefault never overrides them.
    from reasondb.config.model_registry import ModelRegistry
    _registry = ModelRegistry.get()
    for _method_name, (_model_name, _cr) in _registry.method_config(modality="vision").items():
        _spec = _registry.spec_by_model_name(_model_name, modality="vision")
        _cost = _registry.cost_for(_cr, _spec.size_class)
        method_configurators.setdefault(
            _method_name,
            make_kv_image_configurator(
                _model_name,
                _cr,
                _cost,
                _spec.size_class,
                materialized_cr=args.materialized_cr,
            ),
        )
    # Vanilla variants (full prefill, no stored caches): one per vision model, named by
    # swapping the "kv" method prefix for "vanilla" (kvLlava72B -> vanillaLlava72B,
    # kvMistralVL24B -> vanillaMistralVL24B, ...). The "vanilla70B" (llava-72b) entry
    # above keeps its name since setdefault never overrides it.
    for _spec_key in _registry.all_keys(modality="vision"):
        _spec = _registry.spec_by_key(_spec_key)
        _vanilla_name = "vanilla" + _spec.method_prefix.removeprefix("kv")
        _cost = _registry.cost_for(0.0, _spec.size_class)
        method_configurators.setdefault(
            _vanilla_name,
            make_kv_image_configurator(
                _spec.model_name, 0.0, _cost, _spec.size_class, vanilla=True
            ),
        )

    # Collect predictions from KV-cache methods
    all_predictions = {}
    all_costs = {}
    
    for method_name in args.kv_methods:
        if method_name not in method_configurators:
            logger.warning(f"Unknown method: {method_name}, skipping...")
            continue
        
        logger.info(f"Running predictions with {method_name}...")
        method_config = method_configurators[method_name]
        
        executor = Executor(
            name=method_name,
            database=benchmark.database,
            reasoner=reasoner,
            optimizer=LabelOptimizer(),
            configurator=method_config,
        )
        
        with executor as e:
            benchmark_result = e.execute_benchmark(
                queries,
                results_cache_dir=result_dir / "cache" / method_name,
                reset_db_before_each_query=True,
            )
        
        all_predictions[method_name] = benchmark_result.results
        all_costs[method_name] = benchmark_result.costs
    
    # Use the reference method as the silver model
    if args.reference_method not in all_predictions:
        logger.error("%s must be run to serve as the reference model!", args.reference_method)
        return

    logger.info("Using %s as the reference (silver) model...", args.reference_method)
    
    # Filter out high selectivity queries if requested
    if args.no_high_selectivity:
        logger.info("Filtering out queries with high selectivity (keeping < 5% of samples)...")
        total_samples = DATASET_LENGTHS.get(benchmark_name, 0)
        logger.info(f"Total samples in database: {total_samples}")
        
        queries_to_keep = {}
        for query, df in all_predictions[args.reference_method].items():
            num_kept = len(df)
            selectivity = num_kept / total_samples if total_samples > 0 else 0

            if selectivity >= 0.05:  # Keep if >= 5%
                queries_to_keep[query] = df
                logger.info(f"KEPT: Query keeps {num_kept}/{total_samples} samples ({selectivity*100:.2f}%): {query}")
            else:
                logger.info(f"FILTERED OUT: Query keeps {num_kept}/{total_samples} samples ({selectivity*100:.2f}%): {query}")

        # Filter all predictions to only include kept queries
        logger.info(f"Filtered {len(all_predictions[args.reference_method]) - len(queries_to_keep)} queries out of {len(all_predictions[args.reference_method])}")
        for method_name in all_predictions.keys():
            filtered_predictions = {q: all_predictions[method_name][q] for q in queries_to_keep.keys() if q in all_predictions[method_name]}
            all_predictions[method_name] = filtered_predictions
            
            filtered_costs = {q: all_costs[method_name][q] for q in queries_to_keep.keys() if q in all_costs[method_name]}
            all_costs[method_name] = filtered_costs
    
    collected_labels = {
        f"{args.reference_method}_silver": all_predictions[args.reference_method]
    }
    
    # Compute overlap matrix using the first method's results
    if all_predictions:
        first_method = list(all_predictions.keys())[0]
        first_results = all_predictions[first_method]
        
        query_to_index = {}
        all_index = set()
        for i, (query, df) in enumerate(first_results.items()):
            logger.info(f"{i}) Num results: {len(df)} - Query: {query}")
            query_to_index[query] = df.index
            all_index = all_index.union(set(df.index.values))
        all_index_sorted = sorted(list(all_index))
        index_to_id = {idx: i for i, idx in enumerate(all_index_sorted)}
        matrix = np.zeros((len(first_results), len(all_index)), dtype=int)
        query_to_matrix_id = {}
        matrix_id_to_query = {}
        for i, (query, df) in enumerate(first_results.items()):
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
    
    # ------------------------------------------------------------------
    # Save debug CSVs under:
    #   debug_outputs/{press_name}/{benchmark_name}/{split}/
    #     query_mapping.txt
    #     kv70B00/query_000.csv  (ground truth, saved once)
    #     {method}/query_000.csv (predictions, one dir per method)
    # ------------------------------------------------------------------
    debug_base_dir = Path("debug_outputs") / args.press_name / benchmark_name / split
    debug_base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load existing query mapping (if any) so we don't overwrite it.
    # Only *new* queries get a fresh ID; existing ones keep their old ID.
    # ------------------------------------------------------------------
    mapping_file = debug_base_dir / "query_mapping.txt"
    # text_to_id holds ALL mappings (old + new) keyed by query string repr
    text_to_id: dict[str, str] = {}

    if mapping_file.exists():
        with open(mapping_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("query_") and ": " in line:
                    query_id_str, query_text = line.split(": ", 1)
                    text_to_id[query_text] = query_id_str.strip()
        logger.info(
            f"Loaded {len(text_to_id)} existing query mappings from {mapping_file}"
        )

    # Determine the next free numeric id
    existing_ids = [
        int(v.replace("query_", ""))
        for v in text_to_id.values()
    ]
    next_id = max(existing_ids, default=-1) + 1

    # Map current queries, reusing existing IDs where the query text matches
    reference_labels = collected_labels[f"{args.reference_method}_silver"]
    query_obj_to_id: dict[object, str] = {}
    for query in reference_labels.keys():
        query_str = str(query)
        if query_str in text_to_id:
            query_obj_to_id[query] = text_to_id[query_str]
        else:
            new_id = f"query_{next_id:03d}"
            query_obj_to_id[query] = new_id
            text_to_id[query_str] = new_id
            next_id += 1

    # query_to_id maps query *objects* (current run) → id  (used downstream)
    query_to_id = query_obj_to_id

    # Re-write the full mapping file (preserves old + adds new entries)
    with open(mapping_file, "w") as f:
        f.write("Query ID to Query Name Mapping\n")
        f.write("=" * 80 + "\n\n")
        # Sort by numeric id so the file stays readable
        for q_text, q_id in sorted(text_to_id.items(), key=lambda x: int(x[1].replace("query_", ""))):
            f.write(f"{q_id}: {q_text}\n")
    logger.info(f"Query mapping written to {mapping_file}")

    # Save the reference model's ground-truth CSVs once
    save_debug_csvs(
        method_name=args.reference_method,
        all_results=all_predictions[args.reference_method],
        query_to_id=query_to_id,
        debug_base_dir=debug_base_dir,
        join_subdir=_get_join_subdir(method_configurators[args.reference_method]),
    )

    # Save predictions for every method and evaluate
    for method_name, predictions in all_predictions.items():
        # Save method predictions as CSVs
        save_debug_csvs(
            method_name=method_name,
            all_results=predictions,
            query_to_id=query_to_id,
            debug_base_dir=debug_base_dir,
            join_subdir=_get_join_subdir(method_configurators[method_name]),
        )

        for label_type, labels in collected_labels.items():
            logger.info(f"Evaluating {method_name} against {label_type} labels...")
            
            # Evaluate
            metrics_df = evaluate_results(
                approach_name=method_name,
                all_predictions=predictions,
                all_labels=labels,
                all_costs=all_costs[method_name],
                debug_dir=debug_base_dir,
            )
            
            # Save metrics with appropriate filename
            suffix = "_no_high_selectivity" if args.no_high_selectivity else ""
            metrics_output = result_dir / f"{method_name}_vs_{label_type}_metrics{suffix}.csv"
            metrics_df.to_csv(metrics_output, index=True)
            logger.info(f"Saved metrics to {metrics_output}")
            
            print()
            print(f"{method_name.upper()} vs {label_type.upper()} Metrics:")
            print(metrics_df)
            print()


if __name__ == "__main__":
    main()
