from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union
import pandas as pd
import torch
from torch import Tensor
from reasondb.database.indentifier import (
    DataType,
    HiddenColumnType,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
from reasondb.evaluation.benchmark import LabelsDefinition
from reasondb.operators.perfect_operators.label_lookup import (
    MissingLabelsError,
    lookup_labels,
)
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import PhysicalOperatorInterface
from reasondb.query_plan.logical_plan import LogicalFilter, LogicalPlanStep
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    ProfilingCost,
    RunOutsideResult,
)
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import FilterOnComputedDataObservation, Observation
from reasondb.utils.logging import FileLogger


class PerfectFilter(PhysicalOperator):
    is_label_only = True

    async def get_observation(
        self,
        database_state: IntermediateState,
        inputs: Sequence[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        output_columns: Sequence[VirtualColumnIdentifier],
        llm_parameters: Dict[str, Any],
        data_sample: Sequence[Optional[pd.DataFrame]],
        logical_plan_step: LogicalPlanStep,
        logger: FileLogger,
    ) -> Observation:
        # Two inputs when this serves a join predicate -- `get_label_configurator`
        # registers it as one, and `FilterOnComputedDataObservation.get_sql` already
        # has the `join_on_computed_data` branch for that case.
        assert len(inputs) in (1, 2)

        output_hidden_cols = await database_state.get_output_hidden_cols(
            operation=self,
            llm_configuration=llm_parameters,
            dependent_columns=llm_parameters["__expression__"].column_mentions(),
            data_type=DataType.BOOL,
            database_state=database_state,
            logger=logger / "get-skip-columns",
        )
        return FilterOnComputedDataObservation(
            hidden_columns=output_hidden_cols,
            logical_plan_step=logical_plan_step,
            quality=self.quality,
        )

    @property
    def prefers_run_outside_db(self) -> bool:
        return True

    async def _run_outside_db(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        input_data: pd.DataFrame,
        llm_parameters: Dict[str, Any],
        database_state: IntermediateState,
        observation: Observation,
        labels: Optional["LabelsDefinition"],
        logger: FileLogger,
    ):
        if labels is None:
            raise MissingLabelsError(
                "PerfectFilter requires ground truth labels, but the logical plan step "
                "carries no LabelsDefinition."
            )
        result_labels = lookup_labels(
            labels=labels, input_data=input_data, operator_name="PerfectFilter"
        )
        result_labels = result_labels.fillna(0)

        mask = [
            (data_id, label) for data_id, label in zip(input_data.index, result_labels)
        ]
        return RunOutsideResult(
            mask,
            # Human labels are not priced; the number of labels requested is tracked
            # separately in `ProfilingOutput.n_labels_requested`.
            ProfilingCost(0.0, 0.0, 0.0),
            input_data=input_data,
        )

    async def profile(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        database_state: IntermediateState,
        observation: Observation,
        llm_parameters: Dict[str, str],
        sample: "ProfilingSampleSpecification",
        data_sample: Sequence[pd.DataFrame],
        logger: FileLogger,
    ) -> Tuple[pd.DataFrame, Tensor, ProfilingCost]:
        """Emit the human verdicts as a saturated decision matrix.

        Mirrors `PhysicalOperator.profile`, which cannot be reused directly because it
        passes `labels=None` to `run_outside_db`. The labels come off the observation's
        logical plan step rather than being held on the operator: operator instances are
        shared across steps and queries, so per-instance label state would alias between
        two filters with different label columns.

        The ±1000 saturation is what `Profiler.get_labels` hard-argmaxes into the boolean
        `keep_labels` the optimizer measures precision and recall against.
        """
        assert isinstance(observation, FilterOnComputedDataObservation)
        run_result = await self.run_outside_db(
            inputs=inputs,
            input_data=data_sample,
            llm_parameters=llm_parameters,
            database_state=database_state,
            observation=observation,
            labels=observation.logical_plan_step.get_labels(),
            logger=logger,
        )
        # `run_outside_db` already formed the cartesian product for a join predicate, so
        # pass its frame rather than `data_sample` -- `transform_input` takes exactly one.
        in_data = run_result.input_data
        kept, _ = observation.transform_input(
            input_data=[in_data],
            transform_data=run_result.output_data,
            inputs=inputs,
            random_ids=None,  # keeps the gold-mixing branch inert
            database_state=database_state,
        )
        surviving_ids = set(kept.index)
        keep_mask = torch.from_numpy(
            in_data.index.map(lambda x: x in surviving_ids).values
        )

        m = torch.ones(len(in_data), 1, 3)
        m[:, 0, 0] = keep_mask * 1000  # Keep
        m[:, 0, 1] = (~keep_mask) * 1000  # Discard
        m[:, 0, 2] = -1000  # Unsure -- a human label is never unsure
        return in_data, m, run_result.cost

    def get_operation_identifier(self) -> str:
        return "PerfectFilter"

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="PerfectFilter",
            explanation="Always returns the perfect filter results based on ground truth labels.",
            parameters=[],
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalFilter

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.PERFECT_FILTER,)

    def get_free_form_equivalence_prompt(
        self, incoming_text: str, db_text: str
    ) -> Prompt:
        raise NotImplementedError()

    def get_is_expensive(self) -> bool:
        return True

    def get_is_potentially_flawed(self) -> bool:
        return True

    def get_hidden_column_type(self) -> HiddenColumnType:
        return HiddenColumnType.FILTER_COLUMN

    def setup(self, database: Database, logger: FileLogger):
        pass

    async def prepare(self, database: Database, logger: FileLogger):
        pass

    async def wind_down(self):
        pass

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True
