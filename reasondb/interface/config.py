from reasondb.database.database import Database
from reasondb.executor import Executor
from reasondb.operators.extract.image_qa_extract import ImageQaExtract
from reasondb.operators.join.qa_filter_join import QaFilterJoin
from reasondb.operators.limit.limit import Limit
from reasondb.optimizer.baselines.abacus_optimizer import ParetoCascades
from reasondb.optimizer.baselines.lotus_optimizer import LotusOptimizer
from reasondb.optimizer.gd_optimizer import (
    GlobalOptimizationMode,
    GradientDescentOptimizer,
    OptimizationConfig,
)
from reasondb.query_plan.logical_plan import ALL_LOGICAL_OPERATORS_TOOLBOX
from reasondb.reasoning.few_shot_database import DUMMY_FEW_SHOT_DATABASE
from reasondb.reasoning.llm import GPT4o
from reasondb.reasoning.reasoners.self_correction import SelfCorrectionReasoner
from reasondb.backends.image_qa import VisionModelImageQABackend
from reasondb.backends.audio_qa import AudioModelAudioQABackend
from reasondb.backends.image_similarity import ImageSimilarityBackend
from reasondb.backends.python_codegen import (
    LLMPythonCodegenBackend,
)
from reasondb.backends.text_qa import KvTextQABackend, LLMTextQABackend
from reasondb.backends.vision_model import KvVisionModel, LlmVisionModel
from reasondb.backends.audio_model import KvAudioModel
from reasondb.operators.aggregate.aggregate import Aggregate
from reasondb.operators.extract.python_extract import PythonExtract
from reasondb.operators.extract.text_qa_extract import TextQaExtract
from reasondb.operators.filter.image_embed_filter import ImageSimilarityFilter
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.operators.filter.text_qa_filter import TextQaFilter
from reasondb.operators.filter.audio_qa_filter import AudioQaFilter
from reasondb.operators.filter.traditional_filter import TraditionalFilter
from reasondb.operators.join.traditional_join import TraditionalJoin
from reasondb.operators.project.project import Project
from reasondb.operators.aggregate.groupby import GroupBy
from reasondb.operators.rename.rename import Rename
from reasondb.operators.sorting.sort import Sort
from reasondb.operators.tranform.python_transform import PythonTransform
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.query_plan.physical_operator import CostType, PhysicalOperatorToolbox
from reasondb.reasoning.llm import GPT4oMini


def get_default_configurator():
    kv09_cost = 0.1
    kv08_cost = 0.3
    kv06_cost = 0.45
    kv05_cost = 0.6
    kv04_cost = 0.75
    kv03_cost = 0.85
    kv00_cost = 0.9
    gpt_cost = 2
    textqa_backend = LLMTextQABackend(GPT4oMini())
    imageqa_backend = VisionModelImageQABackend(LlmVisionModel(GPT4oMini()))
    kvtextqa8B_backendcr09 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.9
    )
    kvtextqa8B_backendcr08 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.8
    )
    kvtextqa8B_backendcr06 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.6
    )
    kvtextqa8B_backendcr05 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.5
    )
    kvtextqa8B_backendcr04 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.4
    )
    kvtextqa8B_backendcr03 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.3
    )
    kvtextqa8B_backendcr00 = KvTextQABackend(
        "meta-llama/Llama-3.1-8B-Instruct", compression_ratio=0.0
    )
    kvtextqa70B_backendcr09 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.9
    )
    kvtextqa70B_backendcr08 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.8
    )
    kvtextqa70B_backendcr06 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.6
    )
    kvtextqa70B_backendcr05 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.5
    )
    # Unsupported compression ratios - commented out
    kvtextqa70B_backendcr04 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.4
    )
    kvtextqa70B_backendcr03 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.3
    )
    kvtextqa70B_backendcr00 = KvTextQABackend(
        "meta-llama/Llama-3.1-70B-Instruct", compression_ratio=0.0
    )
    kvimageqa8B_backendcr0995 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.995)
    )
    kvimageqa8B_backendcr099 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.99)
    )
    kvimageqa8B_backendcr095 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.95)
    )
    kvimageqa8B_backendcr09 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.9)
    )
    kvimageqa8B_backendcr08 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.8)
    )
    kvimageqa8B_backendcr05 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.5)
    )
    kvimageqa8B_backendcr00 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llama3-llava-next-8b-hf", compression_ratio=0.0)
    )
    kvimageqa70B_backendcr0995 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.995)
    )
    kvimageqa70B_backendcr099 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.99)
    )
    kvimageqa70B_backendcr095 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.95)
    )
    kvimageqa70B_backendcr09 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.9)
    )
    kvimageqa70B_backendcr08 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.8)
    )
    kvimageqa70B_backendcr05 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.5)
    )
    kvimageqa70B_backendcr00 = VisionModelImageQABackend(
        KvVisionModel("llava-hf/llava-next-72b-hf", compression_ratio=0.0)
    )
    kvaudioqa_backend_cr09 = AudioModelAudioQABackend(
        KvAudioModel("Qwen/Qwen2-Audio-7B-Instruct", compression_ratio=0.9)
    )
    kvaudioqa_backend_cr08 = AudioModelAudioQABackend(
        KvAudioModel("Qwen/Qwen2-Audio-7B-Instruct", compression_ratio=0.8)
    )
    kvaudioqa_backend_cr05 = AudioModelAudioQABackend(
        KvAudioModel("Qwen/Qwen2-Audio-7B-Instruct", compression_ratio=0.5)
    )
    kvaudioqa_backend_cr00 = AudioModelAudioQABackend(
        KvAudioModel("Qwen/Qwen2-Audio-7B-Instruct", compression_ratio=0.0)
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
                # TextQaFilter(kvtextqa8B_backendcr09, quality=3, fake_cost=kv09_cost),
                TextQaFilter(kvtextqa8B_backendcr08, quality=3.5, fake_cost=kv08_cost),
                # TextQaFilter(kvtextqa8B_backendcr06, quality=4, fake_cost=kv06_cost),
                TextQaFilter(kvtextqa8B_backendcr05, quality=4.2, fake_cost=kv05_cost),
                # TextQaFilter(kvtextqa8B_backendcr04, quality=4.5, fake_cost=kv04_cost),
                # TextQaFilter(kvtextqa8B_backendcr03, quality=4.7, fake_cost=kv03_cost),
                TextQaFilter(kvtextqa8B_backendcr00, quality=5, fake_cost=kv00_cost),
                # TextQaFilter(kvtextqa70B_backendcr09, quality=6, fake_cost=kv09_cost),
                TextQaFilter(kvtextqa70B_backendcr08, quality=6.5, fake_cost=kv08_cost),
                TextQaFilter(kvtextqa70B_backendcr06, quality=7, fake_cost=kv06_cost),
                # TextQaFilter(kvtextqa70B_backendcr05, quality=7.2, fake_cost=kv05_cost),
                # TextQaFilter(kvtextqa70B_backendcr04, quality=7.5, fake_cost=kv04_cost),
                TextQaFilter(kvtextqa70B_backendcr03, quality=7.7, fake_cost=kv03_cost),
                TextQaFilter(kvtextqa70B_backendcr00, quality=8, fake_cost=kv00_cost),
                # ImageQaFilter(
                #     VisionModelImageQABackend(
                #         LocalVisionModel("Salesforce/blip2-opt-2.7b")
                #     )
                # ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr0995,
                #     quality=2,
                #     fake_cost=kv09_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr099,
                #     quality=2.5,
                #     fake_cost=kv09_cost,
                # ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr095,
                #     quality=2.7,
                #     fake_cost=kv09_cost,
                # ),
                ImageQaFilter(
                    kvimageqa8B_backendcr09,
                    quality=3,
                    fake_cost=kv09_cost,
                ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr08,
                #     quality=4,
                #     fake_cost=kv08_cost,
                # ),
                ImageQaFilter(
                    kvimageqa8B_backendcr05,
                    quality=4.5,
                    fake_cost=kv05_cost,
                ),
                ImageQaFilter(
                    kvimageqa8B_backendcr00,
                    quality=5,
                    fake_cost=kv00_cost,
                ),
                # ImageQaFilter(
                #     kvimageqa70B_backendcr0995,
                #     quality=2,
                #     fake_cost=kv09_cost,
                # ),
                ImageQaFilter(
                    kvimageqa70B_backendcr099,
                    quality=6,
                    fake_cost=kv09_cost,
                ),
                # ImageQaFilter(
                #     kvimageqa70B_backendcr095,
                #     quality=2.7,
                #     fake_cost=kv09_cost,
                # ),
                ImageQaFilter(
                    kvimageqa70B_backendcr09,
                    quality=7,
                    fake_cost=kv09_cost,
                ),
                # ImageQaFilter(
                #     kvimageqa70B_backendcr08,
                #     quality=4,
                #     fake_cost=kv08_cost,
                # ),
                ImageQaFilter(
                    kvimageqa70B_backendcr05,
                    quality=8,
                    fake_cost=kv05_cost,
                ),
                ImageQaFilter(
                    kvimageqa70B_backendcr00,
                    quality=9,
                    fake_cost=kv00_cost,
                ),
                # ImageQaFilter(
                #     imageqa_backend,
                #     quality=11,
                #     fake_cost=gpt_cost,
                # ),
                # # AudioQaFilter(
                #     kvaudioqa_backend_cr09,
                #     quality=3,
                #     fake_cost=kv09_cost,
                # ),
                # AudioQaFilter(
                #     kvaudioqa_backend_cr08,
                #     quality=4,
                #     fake_cost=kv08_cost,
                # ),
                # AudioQaFilter(
                #     kvaudioqa_backend_cr05,
                #     quality=4.5,
                #     fake_cost=kv05_cost,
                # ),
                # AudioQaFilter(
                #     kvaudioqa_backend_cr00,
                #     quality=5,
                #     fake_cost=kv00_cost,
                # ),
            ],
            extract_operators=[
                # TextQaExtract(textqa_backend, quality=10, fake_cost=gpt_cost),
                # TextQaExtract(kvtextqa8B_backendcr09, quality=3, fake_cost=kv09_cost),
                TextQaExtract(
                    kvtextqa8B_backendcr08, quality=3.5, fake_cost=kv08_cost
                ),
                TextQaExtract(kvtextqa8B_backendcr06, quality=4, fake_cost=kv06_cost),
                TextQaExtract(
                    kvtextqa8B_backendcr05, quality=4.2, fake_cost=kv05_cost
                ),
                # TextQaExtract(
                #     kvtextqa8B_backendcr04, quality=4.5, fake_cost=kv04_cost
                # ),
                # TextQaExtract(
                #     kvtextqa8B_backendcr03, quality=4.7, fake_cost=kv03_cost
                # ),
                TextQaExtract(kvtextqa8B_backendcr00, quality=5, fake_cost=kv00_cost),
                # TextQaExtract(kvtextqa70B_backendcr09, quality=6, fake_cost=kv09_cost),
                TextQaExtract(
                    kvtextqa70B_backendcr08, quality=6.5, fake_cost=kv08_cost
                ),
                TextQaExtract(kvtextqa70B_backendcr06, quality=7, fake_cost=kv06_cost),
                # TextQaExtract(
                #     kvtextqa70B_backendcr05, quality=7.2, fake_cost=kv05_cost
                # ),
                # TextQaExtract(
                #     kvtextqa70B_backendcr04, quality=7.5, fake_cost=kv04_cost
                # ),
                TextQaExtract(
                    kvtextqa70B_backendcr03, quality=7.7, fake_cost=kv03_cost
                ),
                TextQaExtract(kvtextqa70B_backendcr00, quality=8, fake_cost=kv00_cost),
                PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0),
                # ImageQaExtract(imageqa_backend, quality=10, fake_cost=gpt_cost),
                ImageQaExtract(
                    kvimageqa8B_backendcr09,
                    quality=3,
                    fake_cost=kv09_cost,
                ),
                # ImageQaFilter(
                #     kvimageqa8B_backendcr08,
                #     quality=4,
                #     fake_cost=kv08_cost,
                # ),
                # ImageQaExtract(
                #     kvimageqa8B_backendcr05,
                #     quality=4.5,
                #     fake_cost=kv05_cost,
                # ),
                ImageQaExtract(
                    kvimageqa8B_backendcr00,
                    quality=5,
                    fake_cost=kv00_cost,
                ),
                # ImageQaExtract(
                #     kvimageqa70B_backendcr0995, quality=2, fake_cost=kv09_cost
                # ),
                ImageQaExtract(
                    kvimageqa70B_backendcr099, quality=6, fake_cost=kv09_cost
                ),
                # ImageQaExtract(
                #     kvimageqa70B_backendcr095, quality=2.7, fake_cost=kv09_cost
                # ),
                ImageQaExtract(
                    kvimageqa70B_backendcr09, quality=7, fake_cost=kv09_cost
                ),
                # ImageQaExtract(
                #     kvimageqa70B_backendcr08, quality=4, fake_cost=kv08_cost
                # ),
                ImageQaExtract(
                    kvimageqa70B_backendcr05, quality=8, fake_cost=kv05_cost
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


class Config:
    def __init__(
        self, identifier_for_caching: str, working_dir: str = "./.working_dir"
    ):
        self.working_dir = working_dir
        self.db_file = None
        self.identifier_for_caching = identifier_for_caching

    def construct_executor(self):
        configurator = get_default_configurator()
        reasoner = SelfCorrectionReasoner(
            llm=GPT4o(),
            configurator=configurator,
            logical_operators=ALL_LOGICAL_OPERATORS_TOOLBOX,
            few_shot_database=DUMMY_FEW_SHOT_DATABASE,
        )
        database = Database(identifier_for_caching=self.identifier_for_caching)
        gd_optimizer = GradientDescentOptimizer(
            OptimizationConfig(
                cost_type=CostType.RUNTIME,
                global_optimization_mode=GlobalOptimizationMode.COMBO,
            )
        )
        abacus_optimizer = ParetoCascades(CostType.RUNTIME)
        lotus_optimizer = LotusOptimizer(
            CostType.RUNTIME,
            proxy_operators=[
                "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0",
                "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0",
                "AudioQaFilter-AudioQABackend-Qwen/Qwen2-Audio-7B-Instruct-cr0.0",
            ],
        )
        executor = Executor(
            database=database,
            reasoner=reasoner,
            optimizer=gd_optimizer,
            configurator=configurator,
        )
        return executor
