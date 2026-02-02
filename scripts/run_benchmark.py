from collections import defaultdict
import logging
import shutil
import argparse
from pathlib import Path
import tracemalloc
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import torch
import yaml
import json
from reasondb.backends.image_qa import VisionModelImageQABackend
from reasondb.backends.image_similarity import ImageSimilarityBackend
from reasondb.backends.python_codegen import LLMPythonCodegenBackend
from reasondb.backends.text_qa import KvTextQABackend, LLMTextQABackend
from reasondb.backends.vision_model import KvVisionModel, LlmVisionModel
from reasondb.evaluation.benchmarks.artwork import (
    Artwork,
    ArtworkLarge,
    ArtworkRandom,
    ArtworkRandomMedium,
)
from reasondb.evaluation.benchmarks.ecommerce import (
    EcommerceLarge,
    EcommerceRandom,
    EcommerceRandomLarge,
)
from reasondb.evaluation.benchmarks.movie import Movie, MovieRandom
from reasondb.evaluation.benchmarks.email import EnronEmail, EnronEmailRandom
from reasondb.evaluation.benchmarks.real_estate import RealEstate
from reasondb.evaluation.benchmarks.rotowire import Rotowire, RotowireRandom
from reasondb.executor import CostSummary, Executor
from reasondb.interface.config import get_default_configurator
from reasondb.operators.aggregate.aggregate import Aggregate
from reasondb.operators.extract.image_qa_extract import ImageQaExtract
from reasondb.operators.extract.python_extract import PythonExtract
from reasondb.operators.extract.text_qa_extract import TextQaExtract
from reasondb.operators.filter.image_embed_filter import ImageSimilarityFilter
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.operators.filter.text_qa_filter import TextQaFilter
from reasondb.operators.filter.traditional_filter import TraditionalFilter
from reasondb.operators.join.qa_filter_join import QaFilterJoin
from reasondb.operators.join.traditional_join import TraditionalJoin
from reasondb.operators.limit.limit import Limit
from reasondb.operators.perfect_operators.perfect_extract import PerfectExtract
from reasondb.operators.perfect_operators.perfect_filter import PerfectFilter
from reasondb.operators.perfect_operators.perfect_transform import PerfectTransform
from reasondb.operators.project.project import Project
from reasondb.operators.aggregate.groupby import GroupBy
from reasondb.operators.rename.rename import Rename
from reasondb.operators.sorting.sort import Sort
from reasondb.operators.tranform.python_transform import PythonTransform
from reasondb.optimizer.baselines.abacus_optimizer import ParetoCascades
from reasondb.optimizer.baselines.lotus_optimizer import LotusOptimizer
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.optimizer.gd_optimizer import (
    GlobalOptimizationMode,
    GradientDescentOptimizer,
    OptimizationConfig,
)
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
from reasondb.optimizer.label_optimizer import LabelOptimizer
from reasondb.query_plan.logical_plan import ALL_LOGICAL_OPERATORS_TOOLBOX
from reasondb.query_plan.physical_operator import CostType, PhysicalOperatorToolbox
from reasondb.reasoning.few_shot_database import DUMMY_FEW_SHOT_DATABASE
from reasondb.reasoning.llm import GPT4o, GPT4oMini
from reasondb.reasoning.reasoners.self_correction import SelfCorrectionReasoner
from reasondb.evaluation.metrics.metrics_manager import MetricsManager

# from reasondb.reasoning.llm import GPT35Turbo
# from reasondb.evaluation.benchmarks.fever import Fever

tracemalloc.start()
# warnings.filterwarnings("error", message="Using a slow.*")


logger = logging.getLogger(__name__)

BENCHMARKS = {
    # CAESURA
    "artwork": Artwork,
    "artwork_large": ArtworkLarge,
    "artwork_random": ArtworkRandom,
    "artwork_random_medium": ArtworkRandomMedium,
    "rotowire": Rotowire,
    "rotowire_random": RotowireRandom,
    #
    # Palimpzest
    "real_estate": RealEstate,
    "enron_email": EnronEmail,
    "email_random": EnronEmailRandom,
    #
    # Sembench
    "movie": Movie,
    "movie_random": MovieRandom,
    "ecommerce_large": EcommerceLarge,
    "ecommerce_random": EcommerceRandom,
    "ecommerce_random_large": EcommerceRandomLarge,
}
BENCHMARKS.update({k.replace("_", "-"): v for k, v in BENCHMARKS.items()})


def get_label_configurator():
    return PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0)],
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
    kv09_cost = 0.1
    kvtextqa8B_backendcr09 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.9
    )
    kvimageqa_backendcr09 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.9)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(kvtextqa8B_backendcr09, quality=3, fake_cost=kv09_cost),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            compression_ratio=0.9,
                        )
                    ),
                    quality=3,
                    fake_cost=kv09_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(kvtextqa8B_backendcr09, quality=3, fake_cost=kv09_cost),
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


def get_kv8B095_configurator():
    kv095_cost = 0.05
    kvtextqa8B_backendcr095 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.95
    )
    kvimageqa_backendcr095 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.95)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(kvtextqa8B_backendcr095, quality=2, fake_cost=kv095_cost),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            compression_ratio=0.95,
                        )
                    ),
                    quality=2,
                    fake_cost=kv095_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(kvtextqa8B_backendcr095, quality=2, fake_cost=kv095_cost),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa_backendcr095, quality=2, fake_cost=kv095_cost),
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
    kv099_cost = 0.01
    kvtextqa8B_backendcr099 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.99
    )
    kvimageqa_backendcr099 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.99)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(kvtextqa8B_backendcr099, quality=1, fake_cost=kv099_cost),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            compression_ratio=0.99,
                        )
                    ),
                    quality=1,
                    fake_cost=kv099_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(kvtextqa8B_backendcr099, quality=1, fake_cost=kv099_cost),
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


def get_kv8B0995_configurator():
    kv0995_cost = 0.005
    kvtextqa8B_backendcr0995 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.995
    )
    kvimageqa_backendcr0995 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.995)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(
                    kvtextqa8B_backendcr0995, quality=1, fake_cost=kv0995_cost
                ),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            compression_ratio=0.995,
                        )
                    ),
                    quality=1,
                    fake_cost=kv0995_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(
                    kvtextqa8B_backendcr0995, quality=1, fake_cost=kv0995_cost
                ),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa_backendcr0995, quality=1, fake_cost=kv0995_cost
                ),
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
    kv08_cost = 0.3
    kvtextqa8B_backendcr08 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.8
    )
    kvimageqa_backendcr08 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.8)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
                # ExtractAndMatchJoin(textqa_backend),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(kvtextqa8B_backendcr08, quality=4, fake_cost=kv08_cost),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            compression_ratio=0.8,
                        )
                    ),
                    quality=4,
                    fake_cost=kv08_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(kvtextqa8B_backendcr08, quality=4, fake_cost=kv08_cost),
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


def get_kv8B05_configurator():
    kv05_cost = 0.6
    kvtextqa8B_backendcr05 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.5
    )
    kvimageqa_backendcr05 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.5)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
                # ExtractAndMatchJoin(textqa_backend),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(kvtextqa8B_backendcr05, quality=5, fake_cost=kv05_cost),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            compression_ratio=0.5,
                        )
                    ),
                    quality=5,
                    fake_cost=kv05_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(kvtextqa8B_backendcr05, quality=5, fake_cost=kv05_cost),
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


def get_gpt_configurator():
    gpt_cost = 2
    textqa_backend = LLMTextQABackend(GPT4oMini())
    imageqa_backend = VisionModelImageQABackend(LlmVisionModel(GPT4oMini()))
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(textqa_backend, quality=3, fake_cost=gpt_cost),
                ImageQaFilter(
                    VisionModelImageQABackend(LlmVisionModel(GPT4oMini())),
                    quality=10,
                    fake_cost=gpt_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(textqa_backend, quality=3, fake_cost=gpt_cost),
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


def get_kv8B00_configurator():
    kv00_cost = 0.9
    kvtextqa8B_backendcr00 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.0
    )
    kvimageqa_backendcr00 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.0)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TextQaFilter(kvtextqa8B_backendcr00, quality=3, fake_cost=kv00_cost),
                TraditionalFilter(quality=0, fake_cost=0),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            compression_ratio=0.0,
                        )
                    ),
                    quality=3,
                    fake_cost=kv00_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(kvtextqa8B_backendcr00, quality=3, fake_cost=kv00_cost),
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


def get_kv8B06_configurator():
    kv06_cost = 0.45
    kvtextqa8B_backendcr06 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.6
    )
    kvimageqa_backendcr06 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.6)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(kvtextqa8B_backendcr06, quality=4, fake_cost=kv06_cost),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            compression_ratio=0.6,
                        )
                    ),
                    quality=4,
                    fake_cost=kv06_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(kvtextqa8B_backendcr06, quality=4, fake_cost=kv06_cost),
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
    kv04_cost = 0.75
    kvtextqa8B_backendcr04 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.4
    )
    kvimageqa_backendcr04 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.4)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(kvtextqa8B_backendcr04, quality=4.5, fake_cost=kv04_cost),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            compression_ratio=0.4,
                        )
                    ),
                    quality=4.5,
                    fake_cost=kv04_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(kvtextqa8B_backendcr04, quality=4.5, fake_cost=kv04_cost),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa_backendcr04, quality=4.5, fake_cost=kv04_cost),
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
    kv03_cost = 0.85
    kvtextqa8B_backendcr03 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.3
    )
    kvimageqa_backendcr03 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.3)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(kvtextqa8B_backendcr03, quality=4.7, fake_cost=kv03_cost),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llama3-llava-next-8b-hf",
                            compression_ratio=0.3,
                        )
                    ),
                    quality=4.7,
                    fake_cost=kv03_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(kvtextqa8B_backendcr03, quality=4.7, fake_cost=kv03_cost),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(kvimageqa_backendcr03, quality=4.7, fake_cost=kv03_cost),
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
    kv70B09_cost = 0.1 + 0.5
    kvtextqa70B_backendcr09 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.9
    )
    kvimageqa70B_backendcr09 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.9)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
                # ExtractAndMatchJoin(textqa_backend),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(
                    kvtextqa70B_backendcr09, quality=7, fake_cost=kv70B09_cost
                ),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            compression_ratio=0.9,
                        )
                    ),
                    quality=7,
                    fake_cost=kv70B09_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(
                    kvtextqa70B_backendcr09, quality=7, fake_cost=kv70B09_cost
                ),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa70B_backendcr09, quality=7, fake_cost=kv70B09_cost
                ),
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
    kv70B08_cost = 0.3 + 0.5
    kvtextqa70B_backendcr08 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.8
    )
    kvimageqa70B_backendcr08 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.8)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
                # ExtractAndMatchJoin(textqa_backend),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(
                    kvtextqa70B_backendcr08, quality=8, fake_cost=kv70B08_cost
                ),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            compression_ratio=0.8,
                        )
                    ),
                    quality=8,
                    fake_cost=kv70B08_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(
                    kvtextqa70B_backendcr08, quality=8, fake_cost=kv70B08_cost
                ),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa70B_backendcr08, quality=8, fake_cost=kv70B08_cost
                ),
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
    kv70B05_cost = 0.6 + 0.5
    kvtextqa70B_backendcr05 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.5
    )
    kvimageqa70B_backendcr05 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.5)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
                # ExtractAndMatchJoin(textqa_backend),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(
                    kvtextqa70B_backendcr05, quality=9, fake_cost=kv70B05_cost
                ),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            compression_ratio=0.5,
                        )
                    ),
                    quality=9,
                    fake_cost=kv70B05_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(
                    kvtextqa70B_backendcr05, quality=9, fake_cost=kv70B05_cost
                ),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa70B_backendcr05, quality=9, fake_cost=kv70B05_cost
                ),
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
    kv70B00_cost = 0.9 + 0.5
    kvtextqa70B_backendcr00 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.0
    )
    kvimageqa70B_backendcr00 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.0)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
                # ExtractAndMatchJoin(textqa_backend),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(
                    kvtextqa70B_backendcr00, quality=9, fake_cost=kv70B00_cost
                ),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            compression_ratio=0.0,
                        )
                    ),
                    quality=9,
                    fake_cost=kv70B00_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(
                    kvtextqa70B_backendcr00, quality=9, fake_cost=kv70B00_cost
                ),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa70B_backendcr00, quality=9, fake_cost=kv70B00_cost
                ),
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
    kv70B06_cost = 0.45 + 0.5
    kvtextqa70B_backendcr06 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.6
    )
    kvimageqa70B_backendcr06 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.6)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(
                    kvtextqa70B_backendcr06, quality=8.5, fake_cost=kv70B06_cost
                ),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            compression_ratio=0.6,
                        )
                    ),
                    quality=8.5,
                    fake_cost=kv70B06_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(
                    kvtextqa70B_backendcr06, quality=8.5, fake_cost=kv70B06_cost
                ),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa70B_backendcr06, quality=8.5, fake_cost=kv70B06_cost
                ),
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
    kv70B04_cost = 0.75 + 0.5
    kvtextqa70B_backendcr04 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.4
    )
    kvimageqa70B_backendcr04 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.4)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(
                    kvtextqa70B_backendcr04, quality=9, fake_cost=kv70B04_cost
                ),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            compression_ratio=0.4,
                        )
                    ),
                    quality=9,
                    fake_cost=kv70B04_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(
                    kvtextqa70B_backendcr04, quality=9, fake_cost=kv70B04_cost
                ),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa70B_backendcr04, quality=9, fake_cost=kv70B04_cost
                ),
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
    kv70B03_cost = 0.85 + 0.5
    kvtextqa70B_backendcr03 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.3
    )
    kvimageqa70B_backendcr03 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.3)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(
                    kvtextqa70B_backendcr03, quality=9.5, fake_cost=kv70B03_cost
                ),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            compression_ratio=0.3,
                        )
                    ),
                    quality=9.5,
                    fake_cost=kv70B03_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(
                    kvtextqa70B_backendcr03, quality=9.5, fake_cost=kv70B03_cost
                ),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa70B_backendcr03, quality=9.5, fake_cost=kv70B03_cost
                ),
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


def get_kv70B0995_configurator():
    kv70B0995_cost = 0.005 + 0.5
    kvtextqa70B_backendcr0995 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.995
    )
    kvimageqa70B_backendcr0995 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.995)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(
                    kvtextqa70B_backendcr0995, quality=5, fake_cost=kv70B0995_cost
                ),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            compression_ratio=0.995,
                        )
                    ),
                    quality=5,
                    fake_cost=kv70B0995_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(
                    kvtextqa70B_backendcr0995, quality=5, fake_cost=kv70B0995_cost
                ),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa70B_backendcr0995, quality=5, fake_cost=kv70B0995_cost
                ),
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
    kv70B099_cost = 0.01 + 0.5
    kvtextqa70B_backendcr099 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.99
    )
    kvimageqa70B_backendcr099 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.99)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(
                    kvtextqa70B_backendcr099, quality=5, fake_cost=kv70B099_cost
                ),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            compression_ratio=0.99,
                        )
                    ),
                    quality=5,
                    fake_cost=kv70B099_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(
                    kvtextqa70B_backendcr099, quality=5, fake_cost=kv70B099_cost
                ),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa70B_backendcr099, quality=5, fake_cost=kv70B099_cost
                ),
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


def get_kv70B095_configurator():
    kv70B095_cost = 0.05 + 0.5
    kvtextqa70B_backendcr095 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.95
    )
    kvimageqa70B_backendcr095 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.95)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
            ],
            filter_operators=[
                TraditionalFilter(quality=0, fake_cost=0),
                TextQaFilter(
                    kvtextqa70B_backendcr095, quality=6, fake_cost=kv70B095_cost
                ),
                ImageQaFilter(
                    VisionModelImageQABackend(
                        KvVisionModel(
                            "llava-hf/llava-next-72b-hf",
                            compression_ratio=0.95,
                        )
                    ),
                    quality=6,
                    fake_cost=kv70B095_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(
                    kvtextqa70B_backendcr095, quality=6, fake_cost=kv70B095_cost
                ),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa70B_backendcr095, quality=6, fake_cost=kv70B095_cost
                ),
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


def get_no_compr_configurator():
    kv00_cost = 0.9
    kvtextqa8B_backendcr00 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.0
    )
    kvtextqa70B_backendcr00 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.0
    )
    kvimageqa8B_backendcr00 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.0)
    )
    kvimageqa70B_backendcr00 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.0)
    )
    configurator = PlanConfigurator(
        llm=GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[
                TraditionalJoin(quality=1, fake_cost=0),
                QaFilterJoin(),
            ],
            filter_operators=[
                # TextQaFilter(textqa_backend, quality=10, fake_cost=gpt_cost),
                TraditionalFilter(quality=1, fake_cost=0),
                ImageSimilarityFilter(
                    ImageSimilarityBackend("Salesforce/blip-itm-base-coco"),
                    quality=2,
                    fake_cost=0,
                ),
                TextQaFilter(kvtextqa8B_backendcr00, quality=5, fake_cost=kv00_cost),
                TextQaFilter(kvtextqa70B_backendcr00, quality=8, fake_cost=kv00_cost),
                ImageQaFilter(
                    kvimageqa8B_backendcr00,
                    quality=5,
                    fake_cost=kv00_cost,
                ),
                ImageQaFilter(
                    kvimageqa70B_backendcr00,
                    quality=9,
                    fake_cost=kv00_cost,
                ),
            ],
            extract_operators=[
                TextQaExtract(kvtextqa8B_backendcr00, quality=5, fake_cost=kv00_cost),
                TextQaExtract(kvtextqa70B_backendcr00, quality=8, fake_cost=kv00_cost),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                ImageQaExtract(
                    kvimageqa8B_backendcr00,
                    quality=5,
                    fake_cost=kv00_cost,
                ),
                ImageQaExtract(
                    kvimageqa70B_backendcr00, quality=9, fake_cost=kv00_cost
                ),
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
    s = s.replace('"', "").replace("'", "")

    # 3. Remove "STRING:" prefix if present
    if s.startswith("string:"):
        s = s[7:]  # Remove "string:" (7 characters)

    # 4. Strip leading/trailing whitespace
    s = s.strip()

    return s


def postprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process a dataframe by applying the following transformations to string columns:
    1. Convert to lowercase
    2. Remove " and ' symbols
    3. Remove "STRING:" prefix if present
    4. Strip leading/trailing whitespace (keep it between words)
    """
    df = df.copy()

    for col in df.columns:
        # Only process columns that contain strings
        if df[col].dtype == "object" or df[col].dtype.name == "string":
            df[col] = df[col].apply(
                lambda x: postprocess_string(x) if isinstance(x, str) else x
            )

    return df


def evaluate(
    benchmark_name: str,
    approach_name: str,
    all_predictions: Union[
        Dict[str, Dict[Tuple[float, float], pd.DataFrame]], Dict[str, pd.DataFrame]
    ],  # query -> guarantee -> df
    all_labels: Dict[str, pd.DataFrame],  # query -> df
    all_costs: Union[
        Dict[str, Dict[Tuple[float, float], CostSummary]], Dict[str, CostSummary]
    ],
    debug_query: Optional[str],
) -> pd.DataFrame:
    # Create debug directory for storing predictions and ground truth
    debug_dir = Path("debug_outputs") / benchmark_name / approach_name
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Create a mapping from query names to short IDs
    query_to_id = {query: f"query_{i:03d}" for i, query in enumerate(all_labels.keys())}

    # Write the query mapping to a file for reference
    mapping_file = debug_dir / "query_mapping.txt"
    with open(mapping_file, "w") as f:
        f.write("Query ID to Query Name Mapping\n")
        f.write("=" * 80 + "\n\n")
        for query, query_id in query_to_id.items():
            f.write(f"{query_id}: {query}\n")

    all_metrics = dict()
    for query, ground_truth_df in all_labels.items():
        if debug_query is not None and query != debug_query:
            continue

        if ground_truth_df.empty:
            logger.warning(f"Ground truth for query {query} is empty. Skipping.")
            continue
        query_id = query_to_id[query]
        predictions_df_or_per_target = all_predictions[query]

        # Post-process ground truth
        ground_truth_df_processed = postprocess_dataframe(ground_truth_df)

        # Write processed ground truth to file (once per query)
        gt_file = debug_dir / f"{query_id}_ground_truth.txt"
        gt_file_csv = debug_dir / f"{query_id}_ground_truth.csv"
        with open(gt_file, "w") as f:
            f.write(f"Ground Truth for query: {query} (POST-PROCESSED)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Shape: {ground_truth_df_processed.shape}\n")
            f.write(f"Columns: {list(ground_truth_df_processed.columns)}\n\n")
            f.write("\n\n")
        ground_truth_df_processed.to_csv(gt_file_csv)

        if isinstance(predictions_df_or_per_target, pd.DataFrame):
            predictions_df = predictions_df_or_per_target

            # Post-process predictions
            predictions_df_processed = postprocess_dataframe(predictions_df)

            # Write processed predictions to file (no guarantees case)
            pred_file = debug_dir / f"{query_id}_predictions_no_guarantees.txt"
            pred_file_csv = debug_dir / f"{query_id}_predictions_no_guarantees.csv"
            with open(pred_file, "w") as f:
                f.write(
                    f"Predictions for query: {query} (no guarantees) (POST-PROCESSED)\n"
                )
                f.write("=" * 80 + "\n\n")
                f.write(f"Shape: {predictions_df_processed.shape}\n")
                f.write(f"Columns: {list(predictions_df_processed.columns)}\n\n")
                f.write("\n\n")
            predictions_df_processed.to_csv(pred_file_csv)

            evaluator = MetricsManager(
                predictions_df_processed, ground_truth_df_processed
            )
            metrics = evaluator.evaluate_all()
            cost = all_costs[query]
            assert isinstance(cost, CostSummary)
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
                metrics["true_output_cardinality"] = len(ground_truth_df)
                metrics["predicted_output_cardinality"] = len(predictions_df)
            all_metrics[approach_name, query, None, None] = metrics
            continue
        else:
            for prec_rec, predictions_df in predictions_df_or_per_target.items():
                # Post-process predictions
                predictions_df_processed = postprocess_dataframe(predictions_df)

                # Write processed predictions to file (with guarantees)
                pred_file = (
                    debug_dir
                    / f"{query_id}_predictions_prec_{prec_rec[0]}_rec_{prec_rec[1]}.txt"
                )
                pred_file_csv = (
                    debug_dir
                    / f"{query_id}_predictions_prec_{prec_rec[0]}_rec_{prec_rec[1]}.csv"
                )
                with open(pred_file, "w") as f:
                    f.write(f"Predictions for query: {query} (POST-PROCESSED)\n")
                    f.write(
                        f"Precision guarantee: {prec_rec[0]}, Recall guarantee: {prec_rec[1]}\n"
                    )
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Shape: {predictions_df_processed.shape}\n")
                    f.write(f"Columns: {list(predictions_df_processed.columns)}\n\n")
                    f.write("\n\n")
                predictions_df_processed.to_csv(pred_file_csv)

                evaluator = MetricsManager(
                    predictions_df_processed, ground_truth_df_processed
                )
                metrics = evaluator.evaluate_all()
                costs = all_costs[query]
                assert isinstance(costs, dict)
                cost = costs[prec_rec]
                for cost_type in CostType:
                    metrics[f"execution_cost_{cost_type.value}"] = (
                        cost.execution_cost.get_cost(cost_type)
                    )
                    metrics[f"tuning_cost_{cost_type.value}"] = (
                        cost.tuning_cost.get_cost(cost_type)
                    )
                    metrics[f"total_cost_{cost_type.value}"] = cost.total_cost.get_cost(
                        cost_type
                    )
                    metrics["true_output_cardinality"] = len(ground_truth_df)
                    metrics["predicted_output_cardinality"] = len(predictions_df)
                all_metrics[approach_name, query, prec_rec[0], prec_rec[1]] = metrics

    logger.info(f"Debug files written to {debug_dir}")

    if not all_metrics:
        raise ValueError("No metrics were computed. Check the inputs.")
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.names = [
        "approach_name",
        "query",
        "precision_guarantee",
        "recall_guarantee",
    ]

    # Add a column with the YAML key for easy reference
    def get_yaml_key(row):
        if pd.isna(row["precision_guarantee"]):
            return f"{row['approach_name']}-{row['query']}-no_guarantees"
        else:
            return f"{row['approach_name']}-{row['query']}-precision_{row['precision_guarantee']}_recall_{row['recall_guarantee']}"

    metrics_df.reset_index(inplace=True)
    metrics_df["pipeline_track_key"] = metrics_df.apply(get_yaml_key, axis=1)
    metrics_df.set_index(
        ["approach_name", "query", "precision_guarantee", "recall_guarantee"],
        inplace=True,
    )

    return metrics_df


def collect_results_all_guarantees(
    out_dir,
    executor_name,
    executor,
    benchmark,
    precision_guarantees,
    recall_guarantees,
    all_combinations,
    debug_query,
):
    queries = benchmark.queries
    force_running_queries = []
    if debug_query is not None:
        queries = [q for q in queries if q.query == debug_query]
        force_running_queries = [debug_query]

    collected_results = {}
    collected_pipeline_tracks = {}
    collected_costs = {}
    if all_combinations:
        guarantees = [
            (prec, rec) for prec in precision_guarantees for rec in recall_guarantees
        ]
    else:
        guarantees = list(zip(precision_guarantees, recall_guarantees))
    for prec, rec in guarantees:
        benchmark_result = executor.execute_benchmark(
            queries,
            PrecisionGuarantee(prec),
            RecallGuarantee(rec),
            results_cache_dir=out_dir / "cache",
            reset_db_before_each_query=True,
            force_running_queries=force_running_queries,
        )
        print()
        print(
            f"Results for {executor_name}, Precision Guarentee {prec}, Recall Guarantee {rec}:"
        )
        for query, table in benchmark_result.results.items():
            print(f"Query: {query}")
            print(table)
            print()
            print("-" * 80)

            if query not in collected_results:
                collected_results[query] = {}
            if query not in collected_pipeline_tracks:
                collected_pipeline_tracks[query] = {}
            if query not in collected_costs:
                collected_costs[query] = {}

            collected_results[query][prec, rec] = table
            collected_pipeline_tracks[query][prec, rec] = (
                benchmark_result.tuned_pipelines.get(query, None)
            )
            collected_costs[query][prec, rec] = benchmark_result.costs.get(query, None)
    return collected_results, collected_pipeline_tracks, collected_costs


def collect_result_no_guarantees(
    out_dir: Path,
    executor_name,
    executor,
    benchmark,
    debug_query: Optional[str],
):
    queries = benchmark.queries
    force_running_queries = []
    if debug_query is not None:
        queries = [q for q in queries if q.query == debug_query]
        force_running_queries = [debug_query]

    benchmark_result = executor.execute_benchmark(
        queries,
        results_cache_dir=out_dir / "cache",
        reset_db_before_each_query=True,
        force_running_queries=force_running_queries,
    )
    print()
    print(f"Results for {executor_name}")
    collected_results = {}
    collected_pipeline_tracks = {}
    collected_costs = {}
    for query, table in benchmark_result.results.items():
        print(f"Query: {query}")
        print(table)
        print()
        print("-" * 80)

        if query not in collected_results:
            collected_results[query] = {}
        if query not in collected_pipeline_tracks:
            collected_pipeline_tracks[query] = {}
        if query not in collected_costs:
            collected_costs[query] = {}

        collected_results[query] = table
        collected_pipeline_tracks[query] = benchmark_result.tuned_pipelines.get(
            query, None
        )
        collected_costs[query] = benchmark_result.costs.get(query, None)
    return collected_results, collected_pipeline_tracks, collected_costs


def save_pipeline_tracks_to_yaml(
    pipeline_tracks: Dict[str, Dict[str, Union[Dict[Tuple[float, float], str], str]]],
    output_path: Path,
) -> None:
    """
    Save pipeline tracks to a YAML file with flattened keys in the format:
    approach_name-query-precision_X_recall_Y (or no_guarantees)

    If the YAML file already exists, it loads existing data and updates/adds keys.

    Args:
        pipeline_tracks: Dictionary mapping approach -> query -> (prec, rec) -> pipeline_track
        output_path: Path where to save the YAML file
    """
    # Load existing YAML file if it exists
    if output_path.exists():
        with open(output_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}
    else:
        yaml_data = {}

    for approach_name, queries in pipeline_tracks.items():
        for query, guarantee_or_track in queries.items():
            if isinstance(guarantee_or_track, dict):
                # Has precision/recall guarantees
                for (prec, rec), pipeline_track in guarantee_or_track.items():
                    key = f"{approach_name}-{query}-precision_{prec}_recall_{rec}"
                    yaml_data[key] = _parse_pipeline_track(pipeline_track)
            else:
                # No guarantees
                pipeline_track = guarantee_or_track
                key = f"{approach_name}-{query}-no_guarantees"
                yaml_data[key] = _parse_pipeline_track(pipeline_track)

    with open(output_path, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )


def _parse_pipeline_track(pipeline_track):
    """
    Parse pipeline_track which can be:
    - A list of strings (where the second element is a JSON string to parse)
    - A string (JSON to parse)
    - Already parsed data

    Returns the parsed data structure suitable for YAML output.
    """
    if isinstance(pipeline_track, list):
        # It's a list, typically ['[]', '<json_string>']
        # Parse the second element if it exists and is a string
        if len(pipeline_track) > 1 and isinstance(pipeline_track[1], str):
            try:
                return json.loads(pipeline_track[1])
            except json.JSONDecodeError:
                return pipeline_track
        return pipeline_track
    elif isinstance(pipeline_track, str):
        # It's a string, try to parse as JSON
        try:
            return json.loads(pipeline_track)
        except json.JSONDecodeError:
            return pipeline_track
    else:
        # Already parsed or other type
        return pipeline_track


def compute_operator_stats(collected_pipeline_tracks, optimized_executors):
    operator_stats = {}
    for approach, tracks_per_query in collected_pipeline_tracks.items():
        if approach not in optimized_executors:
            continue
        for query, trackes_per_target in tracks_per_query.items():
            for (precision_target, recall_target), tracks in trackes_per_target.items():
                operator_counts = defaultdict(int)
                for section in tracks:
                    for operator_def in json.loads(section):
                        if "operator" not in operator_def:
                            continue
                        operator = operator_def["operator"]
                        operator_counts[operator] += 1

                for operator, count in operator_counts.items():
                    operator_stats[
                        (approach, query, precision_target, recall_target, operator)
                    ] = {"count": count}

    # Return empty DataFrame with proper structure if no stats were collected
    if not operator_stats:
        df = pd.DataFrame(columns=["count"])
        df.index = pd.MultiIndex.from_tuples(
            [],
            names=[
                "approach",
                "query",
                "precision_target",
                "recall_target",
                "operator",
            ],
        )
        return df

    df = pd.DataFrame.from_dict(operator_stats, orient="index").fillna(0).astype(int)
    df.index.names = [
        "approach",
        "query",
        "precision_target",
        "recall_target",
        "operator",
    ]
    return df


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=BENCHMARKS.keys(),
        default=["enron_email"],
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
        "--labels",
        type=str,
        nargs="+",
        default=["silver", "gold"],
        help="Labels to use for evaluation.",
    )
    parser.add_argument(
        "--skip-executors",
        type=str,
        nargs="+",
        default=[],
        help="Executors to select.",
    )
    parser.add_argument(
        "--select-executors",
        type=str,
        nargs="+",
        default=[],
        help="Executors to select.",
    )
    parser.add_argument(
        "--clean-result-cache",
        action="store_true",
        help="Whether to clean all previously cached results",
    )
    parser.add_argument(
        "--precision-guarantees",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.9],
    )
    parser.add_argument(
        "--recall-guarantees",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.9],
    )
    parser.add_argument(
        "--all-guarantee-combinations",
        action="store_true",
        help="Whether to run all combinations of precision and recall guarantees.",
    )
    parser.add_argument(
        "--cost-type",
        type=str,
        choices=[c.value for c in CostType],
        default=CostType.RUNTIME.value,
        help="The cost type to optimize for.",
    )
    parser.add_argument(
        "--debug-query",
        type=str,
        default=None,
        help="If set, only run the benchmark for the specified query.",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        required=True,
        help="The device to run optimization on",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    logger.info(f"Running benchmarks: {args.benchmarks}")
    for benchmark_name in args.benchmarks:
        result_dir = args.output_dir / benchmark_name / args.split
        result_dir.mkdir(parents=True, exist_ok=True)

        if args.clean_result_cache:
            shutil.rmtree(str(result_dir / "cache"))

        benchmark_class = BENCHMARKS[benchmark_name]
        split = args.split
        print(f"Running benchmark {benchmark_name} on split {split}...")
        benchmark = benchmark_class.load(split)
        # benchmark.pprint()

        configurator = get_default_configurator()
        configurator_no_compr = get_no_compr_configurator()
        kv8B09configurator = get_kv8B09_configurator()
        kv8B095configurator = get_kv8B095_configurator()
        kv8B099configurator = get_kv8B099_configurator()
        kv8B0995configurator = get_kv8B0995_configurator()
        kv8B08configurator = get_kv8B08_configurator()
        kv8B06configurator = get_kv8B06_configurator()
        kv8B05configurator = get_kv8B05_configurator()
        kv8B04configurator = get_kv8B04_configurator()
        kv8B03configurator = get_kv8B03_configurator()
        kv8B00configurator = get_kv8B00_configurator()
        kv70B09configurator = get_kv70B09_configurator()
        kv70B08configurator = get_kv70B08_configurator()
        kv70B05configurator = get_kv70B05_configurator()
        kv70B00configurator = get_kv70B00_configurator()
        kv70B0995configurator = get_kv70B0995_configurator()
        kv70B099configurator = get_kv70B099_configurator()
        kv70B095configurator = get_kv70B095_configurator()
        kv70B06configurator = get_kv70B06_configurator()
        kv70B04configurator = get_kv70B04_configurator()
        kv70B03configurator = get_kv70B03_configurator()
        gpt_configurator = get_gpt_configurator()
        label_configurator = get_label_configurator()

        reasoner = SelfCorrectionReasoner(
            llm=GPT4o(),
            configurator=configurator,
            logical_operators=ALL_LOGICAL_OPERATORS_TOOLBOX,
            few_shot_database=DUMMY_FEW_SHOT_DATABASE,
        )

        executors = {
            "optim_global": Executor(
                name="optim_global",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=GradientDescentOptimizer(
                    OptimizationConfig(
                        cost_type=CostType(args.cost_type),
                        global_optimization_mode=GlobalOptimizationMode.GLOBAL,
                        device=args.device,
                    )
                ),
                configurator=configurator,
            ),
            "abacus": Executor(
                name="abacus",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=ParetoCascades(CostType(args.cost_type)),
                configurator=configurator,
            ),
            "lotus": Executor(
                name="lotus",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LotusOptimizer(
                    CostType(args.cost_type),
                    proxy_operators=[
                        "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0",
                        "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0",
                        # "ImageSimilarityFilter-ImageSimilarityBackend-Salesforce/blip-itm-base-coco",
                        "AudioQaFilter-AudioQABackend-Qwen/Qwen2-Audio-7B-Instruct-cr0.9",  # cr00 is silver model
                    ],
                ),
                configurator=configurator,
            ),
            "lotus-no-guarantee": Executor(
                name="lotus-no-guarantee",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LotusOptimizer(
                    CostType(args.cost_type),
                    proxy_operators=[
                        "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0",
                        "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0",
                        # "ImageSimilarityFilter-ImageSimilarityBackend-Salesforce/blip-itm-base-coco",
                        "AudioQaFilter-AudioQABackend-Qwen/Qwen2-Audio-7B-Instruct-cr0.9",  # cr00 is silver model
                    ],
                    guarantee_targets=False,
                ),
                configurator=configurator,
            ),
            "optim_combo": Executor(
                name="optim_combo",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=GradientDescentOptimizer(
                    OptimizationConfig(
                        cost_type=CostType(args.cost_type),
                        global_optimization_mode=GlobalOptimizationMode.COMBO,
                        device=args.device,
                    )
                ),
                configurator=configurator,
            ),
            "optim_no_guarantee": Executor(
                name="optim_no_guarantee",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=GradientDescentOptimizer(
                    OptimizationConfig(
                        cost_type=CostType(args.cost_type),
                        global_optimization_mode=GlobalOptimizationMode.COMBO,
                        guarantee_targets=False,
                        device=args.device,
                    )
                ),
                configurator=configurator,
            ),
            "optim_global_no_compr": Executor(
                name="optim_global_no_compr",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=GradientDescentOptimizer(
                    OptimizationConfig(
                        cost_type=CostType(args.cost_type),
                        global_optimization_mode=GlobalOptimizationMode.GLOBAL,
                        device=args.device,
                    )
                ),
                configurator=configurator_no_compr,
            ),
            "optim_shift_budget": Executor(
                name="optim_shift_budget",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=GradientDescentOptimizer(
                    OptimizationConfig(
                        cost_type=CostType(args.cost_type),
                        global_optimization_mode=GlobalOptimizationMode.SHIFT_BUDGET,
                        device=args.device,
                    )
                ),
                configurator=configurator,
            ),
            "optim_local": Executor(
                name="optim_local",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=GradientDescentOptimizer(
                    OptimizationConfig(
                        cost_type=CostType(args.cost_type),
                        global_optimization_mode=GlobalOptimizationMode.LOCAL,
                        device=args.device,
                    )
                ),
                configurator=configurator,
            ),
            "kv8B09": Executor(
                name="kv8B09",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv8B09configurator,
            ),
            "kv8B08": Executor(
                name="kv8B08",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv8B08configurator,
            ),
            "kv8B05": Executor(
                name="kv8B05",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv8B05configurator,
            ),
            "kv8B06": Executor(
                name="kv8B06",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv8B06configurator,
            ),
            "kv8B04": Executor(
                name="kv8B04",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv8B04configurator,
            ),
            "kv8B03": Executor(
                name="kv8B03",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv8B03configurator,
            ),
            "kv8B00": Executor(
                name="kv8B00",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv8B00configurator,
            ),
            "kv8B095": Executor(
                name="kv8B095",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv8B095configurator,
            ),
            "kv8B099": Executor(
                name="kv8B099",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv8B099configurator,
            ),
            "kv8B0995": Executor(
                name="kv8B0995",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv8B0995configurator,
            ),
            "kv70B09": Executor(
                name="kv70B09",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv70B09configurator,
            ),
            "kv70B08": Executor(
                name="kv70B08",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv70B08configurator,
            ),
            "kv70B06": Executor(
                name="kv70B06",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv70B06configurator,
            ),
            "kv70B05": Executor(
                name="kv70B05",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv70B05configurator,
            ),
            "kv70B04": Executor(
                name="kv70B04",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv70B04configurator,
            ),
            "kv70B03": Executor(
                name="kv70B03",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv70B03configurator,
            ),
            "kv70B00": Executor(
                name="kv70B00",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv70B00configurator,
            ),
            "kv70B0995": Executor(
                name="kv70B0995",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv70B0995configurator,
            ),
            "kv70B099": Executor(
                name="kv70B099",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv70B099configurator,
            ),
            "kv70B095": Executor(
                name="kv70B095",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=kv70B095configurator,
            ),
            "gpt": Executor(
                name="gpt",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=gpt_configurator,
            ),
            "silver": Executor(
                name="silver",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=configurator,
            ),
            "gold": Executor(
                name="gold",
                database=benchmark.database,
                reasoner=reasoner,
                optimizer=LabelOptimizer(),
                configurator=label_configurator,
            ),
        }
        optimized_executors = {
            "optim_global",
            "abacus",
            "lotus",
            "lotus-no-guarantee",
            "optim_combo",
            "optim_shift_budget",
            "optim_local",
            "optim_no_guarantee",
            "optim_global_no_compr",
        }
        non_optimized_baselines = {
            "kv8B09",
            "kv8B08",
            "kv8B06",
            "kv8B05",
            "kv8B04",
            "kv8B03",
            "kv8B00",
            "kv8B095",
            "kv8B099",
            "kv8B0995",
            "kv70B09",
            "kv70B08",
            "kv70B05",
            "kv70B00",
            "kv70B0995",
            "kv70B099",
            "kv70B095",
            "kv70B06",
            "kv70B04",
            "kv70B03",
            "gpt",
        }
        if not benchmark.has_ground_truth and "gold" in executors:
            del executors["gold"]
        skip_executors = (
            args.skip_executors
            if args.skip_executors
            else [
                executor
                for executor in executors.keys()
                if args.select_executors
                and executor not in (args.select_executors + ["silver", "gold"])
            ]
        )
        for skip_executor in skip_executors:
            if skip_executor in executors:
                del executors[skip_executor]
                logger.info(f"Skipping executor {skip_executor} as requested.")
            else:
                logger.warning(
                    f"Requested to skip executor {skip_executor}, but it was not found among the executors."
                )
                logger.warning(f"Available executors: {list(executors.keys())}")
            if skip_executor in optimized_executors:
                optimized_executors.remove(skip_executor)
            if skip_executor in non_optimized_baselines:
                non_optimized_baselines.remove(skip_executor)

        collected_results_predictions = {}
        collected_results_labels = {}
        collected_pipeline_tracks = {}
        collected_exec_costs = {}
        for name, executor in executors.items():
            logger.info("*" * 80)
            logger.info(f"Running executor: {name}")
            logger.info("*" * 80)
            with executor as e:
                if name in optimized_executors:
                    r, pt, c = collect_results_all_guarantees(
                        out_dir=result_dir,
                        executor_name=name,
                        executor=e,
                        benchmark=benchmark,
                        precision_guarantees=args.precision_guarantees,
                        recall_guarantees=args.recall_guarantees,
                        all_combinations=args.all_guarantee_combinations,
                        debug_query=args.debug_query,
                    )
                    collected_results_predictions[name] = r
                    collected_pipeline_tracks[name] = pt
                    collected_exec_costs[name] = c
                elif name in non_optimized_baselines:
                    r, pt, c = collect_result_no_guarantees(
                        out_dir=result_dir,
                        executor_name=name,
                        executor=e,
                        benchmark=benchmark,
                        debug_query=args.debug_query,
                    )
                    collected_results_predictions[name] = r
                    collected_pipeline_tracks[name] = pt
                    collected_exec_costs[name] = c
                elif name in args.labels:
                    r, pt, c = collect_result_no_guarantees(
                        out_dir=result_dir,
                        executor_name=name,
                        executor=e,
                        benchmark=benchmark,
                        debug_query=args.debug_query,
                    )
                    collected_results_labels[name] = r
                    # Label executors pipeline tracks are also saved but separately
                    collected_pipeline_tracks[name] = pt
                    collected_exec_costs[name] = c
                else:
                    logger.info(f"Skipping executor {name}...")
                    input("Press Enter to continue...")

        all_silver_metrics = []
        all_gold_metrics = []
        for approach in optimized_executors.union(non_optimized_baselines):
            silver_metrics = evaluate(
                benchmark_name,
                approach,
                collected_results_predictions[approach],
                collected_results_labels["silver"],
                collected_exec_costs[approach],
                debug_query=args.debug_query,
            )
            all_silver_metrics.append(silver_metrics)
            if "gold" in collected_results_labels:
                gold_metrics = evaluate(
                    benchmark_name,
                    approach,
                    collected_results_predictions[approach],
                    collected_results_labels["gold"],
                    collected_exec_costs[approach],
                    debug_query=args.debug_query,
                )
                all_gold_metrics.append(gold_metrics)

        additional_info_dict = {
            query.query: query.additional_info for query in benchmark.queries
        }
        additional_info_df = pd.DataFrame(additional_info_dict)
        additional_info_df = additional_info_df.T
        additional_info_df.index.name = "query"
        print()

        all_silver_metrics_df = pd.concat(all_silver_metrics)
        all_silver_metrics_df = all_silver_metrics_df.merge(
            additional_info_df, left_index=True, right_index=True
        )
        print()
        print("Silver Metrics:")
        print(all_silver_metrics_df)
        all_silver_metrics_df.to_csv(result_dir / "silver_metrics.csv", index=True)

        if all_gold_metrics:
            all_gold_metrics_df = pd.concat(all_gold_metrics)
            all_gold_metrics_df = all_gold_metrics_df.merge(
                additional_info_df, left_index=True, right_index=True
            )
            print()
            print("Gold Metrics:")
            print(all_gold_metrics_df)
            all_gold_metrics_df.to_csv(result_dir / "gold_metrics.csv", index=True)

        # Save pipeline tracks to YAML
        save_pipeline_tracks_to_yaml(
            collected_pipeline_tracks, result_dir / "pipeline_tracks.yaml"
        )
        operator_stats = compute_operator_stats(
            collected_pipeline_tracks, optimized_executors
        )
        operator_stats.to_csv(result_dir / "operator_stats.csv", index=True)


if __name__ == "__main__":
    main()


