from collections.abc import Callable
import math
import pandas as pd
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Type, List
from reasondb.backends.simulate_store import SimulateStore
from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
from reasondb.query_plan.capabilities import Capability
from reasondb.query_plan.logical_plan import (
    LogicalAggregate,
    LogicalExtract,
    LogicalFilter,
    LogicalPlan,
    LogicalPlanStep,
)
from reasondb.query_plan.llm_parameters import LlmParameterTemplate
from reasondb.query_plan.unoptimized_physical_plan import (
    OutputCardinalityTooSmall,
    UnoptimizedPhysicalPlan,
    UnoptimizedPhysicalPlanStep,
)
from reasondb.query_plan.query import Query

# Module import, never by value: the monitor rebinds its global sink at run start.
from reasondb.monitor import collector as _monitor

# The same compression-ratio extraction `operator_run` events use, so a candidate in the
# search space and the same operator's later runs describe their backend identically.
from reasondb.query_plan.physical_operator import (
    LABEL_OPERATOR_QUALITY,
    _extract_cr_info,
)
from reasondb.reasoning.exceptions import Mistake
from reasondb.reasoning.llm import LargeLanguageModel, Message, Prompt, PromptTemplate
from reasondb.utils.logging import FileLogger
from reasondb.utils import precompute_modalities
from reasondb.reasoning.observation import Observation

if TYPE_CHECKING:
    from reasondb.query_plan.physical_operator import (
        PhysicalOperatorToolbox,
        PhysicalOperatorsWithPseudos,
    )

TEST_OUTPUT_LEN = 3


def _label_operator_for(logical_step_type: Type[LogicalPlanStep]):
    """The label-only operator that can supply ground truth for a logical operator type.

    Imported lazily: the perfect operators pull in ``reasondb.evaluation.benchmark`` for
    ``LabelsDefinition``, and that package reaches back into the executor and this module.
    """
    from reasondb.operators.perfect_operators.perfect_extract import PerfectExtract
    from reasondb.operators.perfect_operators.perfect_filter import PerfectFilter

    # `LogicalFilter` covers join predicates too: a join predicate is a filter over two
    # input tables, which is why `get_label_configurator` registers `PerfectFilter` as
    # both `filter_operators` and `join_predicates`.
    #
    # Matched by `issubclass`, not by exact type: `implements_logical_operator` is
    # compared the same way throughout the toolbox, so a subclassed logical operator must
    # not silently lose its label source.
    for logical_type, operator_cls in (
        (LogicalFilter, PerfectFilter),
        (LogicalExtract, PerfectExtract),
    ):
        if issubclass(logical_step_type, logical_type):
            return operator_cls
    return None


class ConfigurationState:
    """A class for storing the current state of the configuration process."""

    def __init__(
        self,
        input_cardinalities: Sequence[int],
        min_output_cardinalities: Sequence[int],
    ):
        """Initialize the configuration state.
        :param input_cardinalities: The input cardinalities of each step for configuration.
        :param min_output_cardinalities: The minimum acceptable output cardinalities of each step configuration.
        """

        self.input_cardinalities = input_cardinalities
        self.output_cardinalities = min_output_cardinalities
        plan_len = len(min_output_cardinalities)
        self.observations: List[Optional[Observation]] = [None] * plan_len
        self.physical_steps: List[Optional[UnoptimizedPhysicalPlanStep]] = [
            None
        ] * plan_len
        self.input_cardinality_factor = 1

    def add_step(
        self,
        step_idx: int,
        observation: Observation,
        physical_step: UnoptimizedPhysicalPlanStep,
    ):
        """Add a step to the configuration state.
        :param step_idx: The index of the step in the logical plan.
        :param observation: The observation resulting from the configuration of the step.
        :param physical_step: The configured physical step.
        """
        i = step_idx
        self.observations[i] = observation
        self.physical_steps[i] = physical_step

    def update_cardinalities(
        self,
        input_cardinalities: Sequence[int],
        output_cardinalities: Sequence[int],
    ):
        """Update the cardinalities of the configuration state.
        :param input_cardinalities: The new input cardinalities of each step for configuration.
        :param output_cardinalities: The new minimum acceptable output cardinalities of each step configuration.
        """
        input_cardinalities = [
            v * self.input_cardinality_factor for v in input_cardinalities
        ]
        old_plan_len = len(self.output_cardinalities)
        self.input_cardinalities = input_cardinalities
        self.output_cardinalities = output_cardinalities
        new_plan_len = len(output_cardinalities)
        num_add_values = new_plan_len - old_plan_len
        self.observations += [None] * num_add_values
        self.physical_steps += [None] * num_add_values

    def increase_input_cardinalities(self, logger: FileLogger):
        """Increase the input cardinalities because some output cardinalities were too small."""
        if not any(c > 1000 for c in self.input_cardinalities):
            self.input_cardinalities = [v * 2 for v in self.input_cardinalities]
            self.input_cardinality_factor *= 2
        else:
            logger.warning(__name__, "Input Cardinalities reached maximum")
            self.output_cardinalities = [v // 2 for v in self.output_cardinalities]


class PlanConfigurator:
    """Configures the physical operators that are available to execute the steps of a logical plan."""

    def __init__(
        self,
        llm: LargeLanguageModel,
        physical_operators: "PhysicalOperatorToolbox",
        use_human_labels: bool = False,
    ):
        """Initialize the PlanConfigurator.
        :param llm: The large language model that sets the configuration parameters.
        :param physical_operators: The toolbox of physical operators that can be used to execute the logical plan.
        :param use_human_labels: When True, any step whose logical operator carries a
            ``LabelsDefinition`` gets a label-only operator appended as its last
            candidate, so the optimizer measures precision and recall against those
            human labels instead of against the highest-quality model's own verdicts.
            The label operator is profiled but never executed -- see
            ``UnoptimizedPhysicalPlanStep.attach_label_operator``. Steps without a
            ``LabelsDefinition`` are unaffected and keep model-derived labels, so a plan
            can mix the two.
        """
        self._database: Optional[Database] = None
        self.physical_operators: "PhysicalOperatorToolbox" = physical_operators
        self.llm = llm
        self.use_human_labels = use_human_labels

    def setup(self, database: Database, logger: FileLogger):
        """Setup the available physical operators.
        :param database: The database that stores all data.
        :param logger: The logger used to log the configuration process.
        """
        for operator in self.physical_operators:
            operator.setup(database=database, logger=logger)

    async def prepare(self, database: Database, logger: FileLogger):
        """Setup the available physical operators.
        :param database: The database that stores all data.
        :param logger: The logger used to log the configuration process.
        """
        await self.llm.prepare()
        # During --precompute, `prepare()` contacts the operator's modality server
        # (e.g. for KV cache materialization), so modalities listed in
        # REASONDB_PRECOMPUTE_SKIP_MODALITIES are skipped here as well. Only takes
        # effect during --precompute; regular execution and --simulate ignore it.
        skip_modalities = (
            precompute_modalities.PRECOMPUTE_SKIP_MODALITIES
            if SimulateStore.get_precompute() is not None
            else frozenset()
        )
        for operator in self.physical_operators:
            modality = operator.get_modality()
            if modality in skip_modalities:
                logger.info(
                    __name__,
                    f"[precompute] skipping prepare() for {operator.get_operation_identifier()} "
                    f"(modality {modality!r} in REASONDB_PRECOMPUTE_SKIP_MODALITIES)",
                )
                continue
            await operator.prepare(database=database, logger=logger)

    async def wind_down(self):
        """Setup the available physical operators.
        :param database: The database that stores all data.
        :param logger: The logger used to log the configuration process.
        """
        await self.llm.close()
        for operator in self.physical_operators:
            await operator.wind_down()

    def shutdown(self, logger: FileLogger):
        """Shutdown the available physical operators.
        :param logger: The logger used to log the shutdown process.
        """
        for operator in self.physical_operators:
            operator.shutdown(logger=logger)

    @property
    def database(self) -> Database:
        """Get the database that stores all data."""
        assert self._database is not None
        return self._database

    def set_database(self, database: Database):
        """Set the database that stores all data."""
        self._database = database

    async def test_logical_plan(
        self, query: Query, logical_plan: LogicalPlan, logger: FileLogger
    ):
        """Test the logical plan by configuring and running it on a small sample."""
        await self.llm_configure(query, logical_plan, logger=logger)

    async def test_logical_plan_prefix(
        self, query: Query, plan_prefix: LogicalPlan, logger: FileLogger
    ) -> IntermediateState:
        """Test the logical plan prefix by configuring and running it on a small sample.
        :param query: The query to be executed.
        :param plan_prefix: The logical plan prefix to be configured.
        :param logger: The logger used to log the configuration process.
        :return: The state of the database after executing the plan prefix.
        """
        unoptimized_physical_plan = await self.llm_configure(
            query, plan_prefix, logger=logger
        )
        return IntermediateState(self.database, plan_prefix=unoptimized_physical_plan)

    async def get_test_output(
        self, logical_plan: LogicalPlan, logger: FileLogger
    ) -> Tuple[pd.DataFrame, Callable[[], int]]:
        """Get some test output of a configured logical plan.
        :param logical_plan: The logical plan to be configured.
        :param logger: The logger used to log the configuration process.
        :return: The test output as a pandas DataFrame and a callable that returns the estimated length of the output.
        """

        unoptimized_physical_plan = logical_plan.get_unoptimized_physical_plan()
        assert unoptimized_physical_plan is not None
        final_state = IntermediateState(
            self.database, plan_prefix=unoptimized_physical_plan
        )
        final_table_identifier = unoptimized_physical_plan.plan_steps[-1].output
        final_table = final_state.get_virtual_table(final_table_identifier)
        data = pd.DataFrame(
            [
                row
                for _, _, row, _ in await final_table.get_data(
                    limit=TEST_OUTPUT_LEN, logger=logger, for_prompt=True
                )
            ][:TEST_OUTPUT_LEN]
        )
        get_estimated_len = final_table.estimated_len
        return data, get_estimated_len

    async def llm_configure(
        self, query: Optional[Query], logical_plan: LogicalPlan, logger: FileLogger
    ) -> UnoptimizedPhysicalPlan:
        """Configure the logical plan by using a LLM to set the configuration parameters.
        First assigns goal cardinalities of the sample data that we present to the LLM as additional context for configuring the plan.
        :param query: The query to be executed.
        :param logical_plan: The logical plan to be configured.
        :param logger: The logger used to log the configuration process.
        :return: The unoptimized physical plan that can be executed on the database.
        """

        configuration_state = logical_plan.get_configuration_state()
        input_cardinalities = self.compute_input_cardinalities(logical_plan)
        min_output_cardinalities = self.compute_min_output_cardinalities(
            logical_plan, input_cardinalities
        )
        if configuration_state is None:
            configuration_state = ConfigurationState(
                input_cardinalities=input_cardinalities,
                min_output_cardinalities=min_output_cardinalities,
            )
        else:
            configuration_state.update_cardinalities(  # continue configuring plan prefix (in tree of thoughts)
                input_cardinalities=input_cardinalities,
                output_cardinalities=min_output_cardinalities,
            )
        assert len(min_output_cardinalities) == len(configuration_state.observations)

        while True:
            try:
                unoptimized_physical_plan = (
                    await self.llm_configure_assigned_cardinalities(
                        query=query,
                        logical_plan=logical_plan,
                        configuration_state=configuration_state,
                        logger=logger,
                    )
                )
                break
            except OutputCardinalityTooSmall as e:
                logger.info(__name__, str(e))
                configuration_state.increase_input_cardinalities(logger=logger)
        logical_plan.set_configuration_state(
            unoptimized_physical_plan, configuration_state
        )
        unoptimized_physical_plan.set_observations(configuration_state.observations)
        return unoptimized_physical_plan

    async def llm_configure_assigned_cardinalities(
        self,
        query: Optional[Query],
        logical_plan: LogicalPlan,
        configuration_state: ConfigurationState,
        logger: FileLogger,
    ) -> UnoptimizedPhysicalPlan:
        """Configure the logical plan by using a LLM to set the configuration parameters.
        :param query: The query to be executed.
        :param logical_plan: The logical plan to be configured.
        :param configuration_state: The current state of the configuration process. Contains the input and output cardinalities of each step.
        :param logger: The logger used to log the configuration process.
        :return: The unoptimized physical plan that can be executed on the database.
        """
        assert self._database is not None
        database_state: IntermediateState = self._database.get_initial_state()
        mistake: Optional[Mistake] = None
        prompt: Optional[Prompt] = None
        response: Optional[str] = None
        unoptimized_physical_plan = UnoptimizedPhysicalPlan()
        i = 0
        sql_queries = {t.identifier: t.sql() for t in database_state.virtual_tables}
        while i < len(logical_plan.plan_steps):
            logical_step = logical_plan.plan_steps[i]
            has_logical_transforms, new_steps = (
                logical_step.perform_logical_transformations(database_state)
            )
            if has_logical_transforms:
                logical_plan._plan_steps = (
                    logical_plan._plan_steps[:i]
                    + new_steps
                    + logical_plan._plan_steps[i + 1 :]
                )
                configuration_state.observations = (
                    configuration_state.observations[:i]
                    + [None] * len(new_steps)
                    + configuration_state.observations[i + 1 :]
                )
                configuration_state.physical_steps = (
                    configuration_state.physical_steps[:i]
                    + [None] * len(new_steps)
                    + configuration_state.physical_steps[i + 1 :]
                )
                configuration_state.input_cardinalities = (
                    list(configuration_state.input_cardinalities[:i])
                    + [configuration_state.input_cardinalities[i]] * len(new_steps)
                    + list(configuration_state.input_cardinalities[i + 1 :])
                )
                configuration_state.output_cardinalities = (
                    list(configuration_state.output_cardinalities[:i])
                    + [configuration_state.output_cardinalities[i]] * len(new_steps)
                    + list(configuration_state.output_cardinalities[i + 1 :])
                )
                logical_plan.validate(database_state.database)
                continue
            if configuration_state.observations[i] is None:
                assert isinstance(logical_step, LogicalPlanStep)
                (
                    prompt,
                    response,
                    mistake,
                ) = await self.llm_configure_single_step(
                    step_idx=i,
                    configuration_state=configuration_state,
                    database_state=database_state,
                    query=query,
                    previous_prompt=prompt,
                    previous_response=response,
                    previous_mistake=mistake,
                    logical_step=logical_step,
                    logger=logger / f"llm-configure-{i}",
                )
                if mistake is not None:
                    continue

            physical_step = configuration_state.physical_steps[i]
            observation = configuration_state.observations[i]
            assert physical_step is not None and observation is not None
            # Record the final candidate set this logical step will be optimized over
            # (built, pseudo-replaced and quality-sorted) for monitoring; a tuned
            # pipeline keeps only the chosen operators.
            _record_search_space(physical_step, step_index=i, query=query)
            input_sql_queries = [
                sql_queries[input_table] for input_table in logical_step.inputs
            ]
            output_sql_query = observation.get_sql(input_sql_queries)

            await physical_step.potentially_run_outside_db(
                op_idx=physical_step.chosen_operator_idx,
                database_state=database_state,
                output_sql_query=output_sql_query,
                logger=logger / "get-observation",
                input_cardinality=configuration_state.input_cardinalities[i],
                check_output_cardinality=configuration_state.output_cardinalities[i],
            )
            assert self._database is not None
            database_state = unoptimized_physical_plan.append(
                step=physical_step,
                observation=observation,
                database=database_state,
            )
            observation_str = await observation.to_str(
                input_sql_queries=input_sql_queries,
                output_table_identifier=physical_step.output,
                database_state=database_state,
                logger=logger / "observation-str",
            )
            sql_queries[logical_step.output] = output_sql_query
            logger.info(__name__, observation_str)
            i += 1
            prompt = mistake = response = None
        return unoptimized_physical_plan

    async def llm_configure_single_step(
        self,
        step_idx: int,
        configuration_state: ConfigurationState,
        database_state: IntermediateState,
        query: Optional[Query],
        previous_prompt: Optional[Prompt],
        previous_response: Optional[str],
        previous_mistake: Optional[Mistake],
        logical_step: LogicalPlanStep,
        logger: FileLogger,
    ) -> Tuple[
        Prompt,
        str,
        Optional[Mistake],
    ]:
        """Configure a single step of the logical plan by using a LLM to set the configuration parameters.
        :param step_idx: The index of the step in the logical plan.
        :param configuration_state: The current state of the configuration process. Contains the input and output cardinalities of each step.
        :param database_state: The current state of the database.
        :param query: The query to be executed.
        :param previous_prompt: The prompt used to configure the previous step.
        :param previous_response: The response of the LLM for the previous step.
        :param previous_mistake: The mistake made by the LLM for the previous step.
        :param logical_step: The logical step to be configured.
        :param logger: The logger used to log the configuration process.
        :return: The prompt used to configure the step, the response of the LLM for the step, and any mistake made by the LLM.
        """
        options = self.physical_operators.get_options(
            logical_step, database_state, logger
        )
        physical_step = mistake = observation = None
        prompt = await self.generate_prompt(
            query=query,
            logical_step=logical_step,
            options=options,
            database_state=database_state,
            prompt=previous_prompt,
            response=previous_response,
            mistake=previous_mistake,
            logger=logger / "generate-prompt",
        )
        response = await self.llm.invoke(prompt, logger=logger)
        try:
            physical_step = self.map_logical_to_physical(
                logical_step=logical_step,
                options=options,
                database_state=database_state,
                response=response,
            )
            observation = await physical_step.get_observation_for_operator(
                op_idx=physical_step.chosen_operator_idx,
                database_state=database_state,
                logger=logger / "get-observation",
            )

            configuration_state.add_step(
                step_idx=step_idx,
                observation=observation,
                physical_step=physical_step,
            )

        except Mistake as m:
            mistake = m
        return prompt, response, mistake

    def compute_input_cardinalities(
        self,
        logical_plan: LogicalPlan,
        result_goal_cardinality=3,
    ) -> Sequence[int]:
        """Compute the input cardinalities of each step for configuration.
        :param logical_plan: The logical plan to be configured.
        :param result_goal_cardinality: The goal cardinality of the result of the logical plan.
        :return: The input cardinalities of each step for configuration.
        """
        plan_len = len(logical_plan.plan_steps)
        input_cardinalities = ([0] * plan_len) + [result_goal_cardinality]
        for i, step in list(enumerate(logical_plan.plan_steps))[::-1]:
            assert isinstance(step, LogicalPlanStep)
            if isinstance(step, LogicalAggregate):
                input_cardinalities[i] = result_goal_cardinality
                continue
            parent_indexes = [p.index for p in step.parents] + [plan_len]
            parent_cardinalities = [input_cardinalities[c] for c in parent_indexes]
            this_goal_cardinality = max(parent_cardinalities) * math.ceil(
                1 / step.get_dummy_selectivity()
            )
            input_cardinalities[i] = this_goal_cardinality
        return input_cardinalities

    def compute_min_output_cardinalities(
        self,
        logical_plan: LogicalPlan,
        input_cardinalities: Sequence[int],
    ) -> Sequence[int]:
        """Compute the minimum acceptable output cardinalities of each step configuration.
        :param logical_plan: The logical plan to be configured.
        :param input_cardinalities: The input cardinalities of each step for configuration.
        :return: The minimum acceptable output cardinalities of each step configuration.
        """
        aggregate_encountered = False
        plan_len = len(logical_plan.plan_steps)
        output_cardinalities = [0] * plan_len
        for i, step in list(enumerate(logical_plan.plan_steps)):
            assert isinstance(step, LogicalPlanStep)
            if isinstance(step, LogicalAggregate):
                aggregate_encountered = True
            if aggregate_encountered:
                output_cardinalities[i] = 1
            else:
                output_cardinalities[i] = max(
                    1, math.floor(input_cardinalities[i] * step.get_dummy_selectivity())
                )
        return output_cardinalities

    def map_logical_to_physical(
        self,
        logical_step: LogicalPlanStep,
        options: "PhysicalOperatorsWithPseudos",
        database_state: IntermediateState,
        response: str,
    ) -> UnoptimizedPhysicalPlanStep:
        """Map the logical step to a physical step by using the LLM response.
        :param logical_step: The logical step to be configured.
        :param options: The available physical operators for the logical step.
        :param database_state: The current state of the database.
        :param response: The response of the LLM for the step.
        :return: The unoptimized physical plan step that can be executed on the database.
        """
        single_input_table = (
            logical_step.inputs[0] if len(logical_step.inputs) == 1 else None
        )
        physical_step = options.parse(logical_step, response, database_state)
        self._maybe_attach_label_operator(logical_step, physical_step)
        physical_step.validate_step(database_state, single_input_table)
        return physical_step

    def _maybe_attach_label_operator(
        self,
        logical_step: LogicalPlanStep,
        physical_step: UnoptimizedPhysicalPlanStep,
    ) -> None:
        """Append a human-label source to this step, when there is one to append.

        Deliberately after ``options.parse`` rather than via the toolbox: ``parse`` keeps
        only the operators the configuring LLM named, so a toolbox entry could be dropped,
        and ``get_capabilities`` derives the reasoner's capability list from the toolbox,
        which would leak ``PERFECT_FILTER`` into planning.
        """
        if not self.use_human_labels:
            return
        operator_cls = _label_operator_for(type(logical_step))
        if operator_cls is None:
            # No label source implemented for this logical operator (transforms, joins,
            # projections...), so there is nothing to attach and nothing to report.
            # Checked before the labels, since many such steps are traditional and
            # hold no model at all.
            return
        labels = logical_step.get_labels()
        if labels is None:
            # A semantic step that could carry ground truth and does not: it keeps
            # model-derived labels. A plan can legitimately mix the two, so this is a log
            # rather than an error; the wholly-unlabelled case is rejected up front
            # (see `producers/run_benchmark.enumerate_jobs`).
            _monitor.record_error(
                "human_labels",
                f"No LabelsDefinition on {type(logical_step).__name__} "
                f"{logical_step.expression!r}; this step keeps model-derived labels.",
            )
            return
        physical_step.attach_label_operator(
            operator_cls(quality=LABEL_OPERATOR_QUALITY, fake_cost=0.0),
            # The only key the perfect operators read. `validate_config` iterates the
            # config's own keys and skips `__expression__`, so a config with just this
            # one validates; `replace_input_table_name` rewrites LlmParameterTemplate
            # values, so it follows table renames.
            {"__expression__": LlmParameterTemplate(logical_step.expression)},
        )

    async def generate_prompt(
        self,
        query: Optional[Query],
        logical_step: LogicalPlanStep,
        options: "PhysicalOperatorsWithPseudos",
        database_state: IntermediateState,
        prompt: Optional[Prompt],
        response: Optional[str],
        mistake: Optional[Mistake],
        logger: FileLogger,
    ) -> Prompt:
        """Generate the prompt for the LLM to configure the logical step.
        :param query: The query to be executed.
        :param logical_step: The logical step to be configured.
        :param options: The available physical operators for the logical step.
        :param database_state: The current state of the database.
        :param prompt: The prompt used to configure the previous step.
        :param response: The response of the LLM for the previous step.
        :param mistake: The mistake made by the LLM for the previous step.
        :param logger: The logger used to log the configuration process.
        :return: The prompt used to configure the step.
        """
        if prompt is None:
            assert mistake is None and response is None
            prompt = await self.generate_init_prompt(
                options, logical_step, database_state, query, logger=logger
            )
        else:
            assert mistake is not None and response is not None
            prompt = self.generate_fix_prompt(mistake, response, prompt)
        return prompt

    async def generate_init_prompt(
        self,
        options: "PhysicalOperatorsWithPseudos",
        logical_step: LogicalPlanStep,
        database_state: IntermediateState,
        query: Optional[Query],
        logger: FileLogger,
    ):
        """Generate the very first prompt for the conversation with the LLM that configures the logical step.
        :param options: The available physical operators for the logical step.
        :param logical_step: The logical step to be configured.
        :param database_state: The current state of the database.
        :param query: The query to be executed.
        :param logger: The logger used to log the configuration process.
        :return: The prompt used to configure the step.
        """
        prompt = PromptTemplate(
            [
                Message(
                    "You are a smart and helpful assistant that helps the user choosing and configuring algorithms",
                    role="system",
                ),
                Message(LLM_CONFIGURE_PROMPT, role="user"),
            ]
        ).fill(
            database_state=await database_state.for_prompt(
                query,
                filter_columns=set(logical_step.get_input_columns()),
                logger=logger,
            ),
            logical_step=logical_step,
            physical_operators=options.to_prompt(logical_step, database_state),
            output_format=options.output_format(),
        )
        return prompt

    def generate_fix_prompt(
        self, mistake: Mistake, response: str, prompt: Prompt
    ) -> Prompt:
        """Generate a fix prompt for the LLM to correct the mistake made in the previous step.
        :param mistake: The mistake made by the LLM for the previous step.
        :param response: The response of the LLM for the previous step.
        :param prompt: The prompt used to configure the previous step.
        :return: The prompt used to configure the step.
        """

        prompt = PromptTemplate(
            prompt.messages
            + [
                Message(role="assistant", text=response),
                Message(
                    role="user",
                    text=LLM_CONFIGURE_FIX_PROMPT,
                ),
            ],
            temperature=0.0,
        ).fill(
            mistake=mistake,
        )
        return prompt

    def get_capabilities(
        self, logical_step: Type[LogicalPlanStep]
    ) -> Sequence[Capability]:
        """Get the available capabilities for a logical operation. These come from the available physical operators.
        :param logical_step: The logical step to be configured.
        :return: The capabilities of the physical operators for the logical step.
        """
        physical_operators = self.physical_operators.get_options_for_logical_operator(
            logical_step
        )
        collected_capabilities = dict()
        for operator in physical_operators:
            for base_capability in operator.get_capabilities():
                capability: Capability = Capability(base_capability)
                capability = collected_capabilities.get(capability, capability)
                capability.add_input_dtypes(*operator.get_input_datatypes())

                if logical_step.get_can_produce_output_columns():
                    capability.add_output_dtypes(*operator.get_output_datatypes())

                collected_capabilities[capability] = capability
        return list(collected_capabilities.values())


def _record_search_space(physical_step, step_index: int, query) -> None:
    """Report one logical step's candidate operators to the monitor.

    A no-op unless a run opted into monitoring. Wrapped in a blanket ``except``:
    telemetry must never be able to fail a run.
    """
    if not _monitor.is_enabled():
        return
    try:
        operators = list(physical_step.operators)
        # Gold is the *last* candidate, because UnoptimizedPhysicalPlanStep sorts them
        # ascending by quality on construction and the profiler derives labels from
        # `len(step.operators) - 1`.
        #
        # "gold" therefore means *label source* -- under --human-labels a label-only
        # operator rather than a model. `executable` is reported separately, since a
        # candidate can be the reference without ever appearing in a plan.
        gold_index = len(operators) - 1
        candidates = []
        for idx, operator in enumerate(operators):
            info = _extract_cr_info(operator)
            label_only = getattr(operator, "is_label_only", False)
            candidates.append(
                {
                    "operator": operator.get_operation_identifier(),
                    "operation_class": type(operator).__name__,
                    "interface": operator.get_llm_parameters().name,
                    "quality": getattr(operator, "quality", None),
                    "fake_cost": getattr(operator, "_fake_cost", None),
                    "gold": idx == gold_index,
                    "label_only": label_only,
                    "executable": not label_only,
                    "estimated_best": idx == physical_step.chosen_operator_idx,
                    **info,
                }
            )
        logical = physical_step.logical_plan_step
        _monitor.record_search_space(
            query=query.query if query is not None else None,
            step_index=step_index,
            logical_type=type(logical).__name__,
            logical_expression=getattr(logical, "expression", None),
            output=str(physical_step.output),
            inputs=[str(i) for i in physical_step.inputs],
            n_candidates=len(candidates),
            candidates=candidates,
        )
    except Exception as exc:  # pragma: no cover - telemetry must never break a run
        _monitor.record_error("search_space", f"{type(exc).__name__}: {exc}")


LLM_CONFIGURE_PROMPT = """
    You are an algorithm expert that knows many algorithms and knows how to configure them.

    You are given a logical operator and a database state.
    There are many available implementations for the logical operator as well as many possible configurations for each implementation.
    For each implementation, configure the relevant parameters and estimate the quality of the result as well as the runtime.

    These are the relevant tables of the database:
    {{database_state}}

    And the user would like to perform the following operation:
    ```json
    {{logical_step}}
    ```

    These are all the available physical implementations for the logical operator and their parameters:
    ```json
    {{physical_operators}}
    ```

    Use the following output format:
    ```json
    {{output_format}}
    ```
"""

LLM_CONFIGURE_FIX_PROMPT = """
    You made a mistake: {{mistake}}.
    Try again!
"""
