import torch
from reasondb.database.database import Database
from reasondb.executor import Executor
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
from reasondb.interface.default_operator_toolbox import (
    build_toolbox,
    default_lotus_proxy_operators,
)
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.query_plan.physical_operator import CostType


def get_default_configurator(
    use_indexes: bool = False, use_human_labels: bool = False
):
    """Build the default operator suite.

    :param use_indexes: When True, each model family materializes a single KV cache at its
        least-compressed enabled ratio and every higher effective ratio is indexed out of
        that one cache (materialized_cr <= effective_cr). When False (the default), every
        effective ratio gets its own materialized cache (effective_cr == materialized_cr),
        which costs more storage but needs no indexing.
    :param use_human_labels: When True, steps carrying a ``LabelsDefinition`` get a
        label-only operator appended, so guarantees are measured against those labels
        rather than against the highest-quality model's own verdicts. See
        :class:`~reasondb.optimizer.configurator.PlanConfigurator`.
    """
    return PlanConfigurator(
        llm=GPT4o(),
        physical_operators=build_toolbox(use_indexes=use_indexes),
        use_human_labels=use_human_labels,
    )


def get_no_index_configurator():
    """The default operator suite without KV cache indexing.

    Same operators, costs and qualities as :func:`get_default_configurator`, but every KV
    backend reads a cache materialized at exactly its effective compression ratio instead
    of indexing into a less-compressed one.
    """
    return get_default_configurator(use_indexes=False)


class Config:
    def __init__(
        self,
        identifier_for_caching: str,
        working_dir: str = "./.working_dir",
        device: torch.device = torch.device("cpu"),
        use_indexes: bool = False,
    ):
        self.working_dir = working_dir
        self.db_file = None
        self.identifier_for_caching = identifier_for_caching
        self.device = device
        self.use_indexes = use_indexes

    def construct_executor(self):
        configurator = get_default_configurator(use_indexes=self.use_indexes)
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
                device=self.device,
            )
        )
        abacus_optimizer = ParetoCascades(CostType.RUNTIME)
        lotus_optimizer = LotusOptimizer(
            CostType.RUNTIME,
            # Derived from the default suite, since Lotus matches proxies by exact
            # operator identifier.
            proxy_operators=default_lotus_proxy_operators(),
        )
        executor = Executor(
            database=database,
            reasoner=reasoner,
            optimizer=gd_optimizer,
            configurator=configurator,
        )
        return executor

