import asyncio
import os
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from copy import deepcopy

from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
from reasondb.optimizer.base_optimizer import Optimizer
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.optimizer.dependency_graph import DependencyGraph
from reasondb.optimizer.guarantees import Guarantee
from reasondb.query_plan.logical_plan import LogicalPlan
from reasondb.query_plan.optimized_physical_plan import ResultData, TunedPipeline
from reasondb.query_plan.physical_operator import ProfilingCost
from reasondb.query_plan.query import Queries, Query
from reasondb.query_plan.tuning_workflow import (
    AggregationSection,
    InitialTraditionalSection,
    TuningMaterializationPoint,
    TuningPipeline,
    TuningWorkflow,
)
from reasondb.reasoning.reasoner import Reasoner
from reasondb.utils.logging import FileLogger


@dataclass
class ExecuteBenchmarkResults:
    results: Dict[str, pd.DataFrame]
    tuned_pipelines: Dict[str, List[str]]
    costs: Dict[str, "CostSummary"]


@dataclass
class CostSummary:
    execution_cost: ProfilingCost
    tuning_cost: ProfilingCost

    @property
    def total_cost(self) -> ProfilingCost:
        return self.execution_cost + self.tuning_cost

    def to_json(self):
        return {
            "execution_cost": self.execution_cost.to_json(),
            "tuning_cost": self.tuning_cost.to_json(),
        }

    @staticmethod
    def from_json(json_obj):
        return CostSummary(
            execution_cost=ProfilingCost.from_json(json_obj["execution_cost"]),
            tuning_cost=ProfilingCost.from_json(json_obj["tuning_cost"]),
        )


class Executor:
    """Executes multi-modal queries by
    1. Translating the natural language query into a logical plan,
    2. Configuring the available physical operators to obtain an unoptimized physical plan,
    3. Preparing the physical plan for tuning to obtain a tuning plan,
    4. Tuning the physical plan to obtain tuned pipelines that are
    5. Executed one after the other
    """

    def __init__(
        self,
        database: Database,
        reasoner: Reasoner,
        optimizer: Optimizer,
        configurator: PlanConfigurator,
        name="default_executor",
    ):
        """Intialize the executor.
        :param database: The database where all the data is stored.
        :param reasoner: The reasoner that translates the natural language query into a logical plan.
        :param optimizer: The optimizer that tunes the physical plan.
        :param configurator: The configurator that configures the physical operators.
        :param working_dir: The directory where the temporary files are stored.
        """
        self.name = name
        self.database = database
        self.reasoner = reasoner
        self.configurator = configurator
        self.optimizer = optimizer
        self.configurator.set_database(self.database)
        self.reasoner.set_database(self.database)
        self.optimizer.set_database(self.database)
        self.logger = FileLogger()
        self.to_clean_up: List[str] = []
        self.last_tuned_pipelines: List[str] = []

    def check_cached_results(
        self,
        results_cache_dir: Optional[Path],
        query_str: str,
        guarantees: Sequence[Guarantee],
        force_running_queries: List[str],
        logger: FileLogger,
    ):
        pass
        if results_cache_dir is None:
            return
        if query_str in force_running_queries:
            return
        json_filepath, data_filepath, key = self.get_cache_filepath(
            dir=results_cache_dir, guarantees=guarantees, query_str=query_str
        )
        if os.path.exists(json_filepath) and os.path.exists(data_filepath):
            try:
                with open(json_filepath, "r") as jf:
                    json_data = json.load(jf)[key]
                with open(data_filepath, "rb") as df:
                    data = pd.read_parquet(df)
                return (
                    data,
                    CostSummary.from_json(json_data["cost"]),
                    json_data["pipelines"],
                )
            except Exception as e:
                logger.warning(
                    __name__,
                    f"Could not load cached results {json_filepath} or {data_filepath}: {e}",
                )

    def get_cache_filepath(
        self, dir: Path, guarantees: Sequence[Guarantee], query_str: str
    ):
        dir.mkdir(parents=True, exist_ok=True)
        guarantees_str = "-".join(str(g) for g in guarantees)
        filekey = "-".join([self.name])
        filekey = "".join(
            [e for e in filekey.replace(" ", "_") if e.isalnum() or e in "-_"]
        )
        jsonkey = "-".join([query_str, guarantees_str])
        jsonkey = "".join(
            [e for e in jsonkey.replace(" ", "_") if e.isalnum() or e in "-_"]
        )
        json_path = dir / (filekey + ".json")
        h = hashlib.sha256(jsonkey.encode("utf-8")).hexdigest()
        data_path = dir / (f"{filekey}_{h}.parquet")
        return json_path, data_path, jsonkey

    def cache_result(
        self,
        results_cache_dir: Optional[Path],
        query_str: str,
        guarantees: Sequence[Guarantee],
        results,
        costs,
        tuned_pipelines,
        logger: FileLogger,
    ):
        if results_cache_dir is None:
            return
        json_filepath, data_filepath, key = self.get_cache_filepath(
            dir=results_cache_dir, guarantees=guarantees, query_str=query_str
        )
        results.to_parquet(data_filepath)

        json_obj = {}
        if json_filepath.exists():
            with open(json_filepath, "r") as f:
                json_obj = json.load(f)

        json_obj[key] = {
            "cost": costs.to_json(),
            "pipelines": tuned_pipelines,
            "data_filepath": str(data_filepath),
        }
        with open(json_filepath, "w") as f:
            json.dump(json_obj, f, indent=4)
        logger.info(
            __name__,
            f"Cached results to {json_filepath} and {data_filepath} with key {key}",
        )

    def execute_benchmark(
        self,
        queries: Queries,
        *guarantees: Guarantee,
        skip_nl_translation: bool = True,
        results_cache_dir: Optional[Path] = None,
        reset_db_before_each_query: bool = False,
        force_running_queries: List[str] = [],
    ) -> ExecuteBenchmarkResults:
        """Execute the queries and return the results.
        :param queries: The queries to be executed.
        :return: The results of the queries. Mapping from query string to DataTable.
        """
        results = {}
        tuned_pipelines = {}
        costs = {}
        self.logger.info(__name__, f"Executing benchmark with {len(queries)} queries")
        for i, query in enumerate(queries):
            logger = self.logger / f"query-{i}"
            cached_result = self.check_cached_results(
                results_cache_dir=results_cache_dir,
                query_str=query.query,
                guarantees=guarantees,
                force_running_queries=force_running_queries,
                logger=logger,
            )
            if cached_result is not None:
                (
                    results[query.query],
                    costs[query.query],
                    tuned_pipelines[query.query],
                ) = cached_result
                continue

            if reset_db_before_each_query:
                self.database.reset()

            # self.initial_state.pprint(query)
            if not skip_nl_translation:
                r, c = asyncio.run(
                    self.execute_query(query, logger=logger),
                )
            else:
                logical_plan = query.get_gt_logical_plan()
                r, c = asyncio.run(
                    self.execute_logical_plan(
                        logical_plan=logical_plan,
                        guarantees=guarantees,
                        logger=logger,
                    )
                )

            d = asyncio.run(self.extract_data(r, logger=self.logger / f"extract-{i}"))
            index_names = [c.col_name for c in r.index_columns]
            df = pd.DataFrame(
                [x[2] for x in d],
                index=pd.MultiIndex.from_tuples([x[0] for x in d], names=index_names),
            )
            results[query.query] = df
            costs[query.query] = c
            tuned_pipelines[query.query] = self.last_tuned_pipelines
            self.cache_result(
                results_cache_dir=results_cache_dir,
                query_str=query.query,
                guarantees=guarantees,
                results=df,
                costs=c,
                tuned_pipelines=self.last_tuned_pipelines,
                logger=logger,
            )
            self.clean_up()
        return ExecuteBenchmarkResults(
            results=results, tuned_pipelines=tuned_pipelines, costs=costs
        )

    async def extract_data(
        self, query_results: TuningMaterializationPoint, logger
    ) -> List:
        result = []
        async for data in query_results.get_data(logger=logger):
            """Extract data from the query results."""
            result.append(data)
        return result

    async def setup(self):
        """Setup the executor by setting up the database and all phyiscal operators registered in the configurator."""
        self.configurator.setup(self.database, self.logger / "setup-optimizer")

    def shutdown(self):
        """Shutdown the executor by shutting down all phyiscal operators registered in the configurator."""
        self.configurator.shutdown(self.logger / "shutdown")

    def __enter__(self):
        """Setup as a context manager."""
        asyncio.run(self.setup())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shutdown as a context manager."""
        self.shutdown()

    async def prepare(self, logger: Optional[FileLogger] = None):
        """Prepare the executor by preparing the database and all phyiscal operators registered in the configurator."""
        # asyncio.get_event_loop().set_debug(True)
        logger = logger or self.logger
        await self.database.prepare(logger / "prepare-database")
        await self.configurator.prepare(self.database, logger / "prepare-configurator")
        await self.reasoner.prepare()

    async def wind_down(self):
        await self.database.wind_down()
        await self.configurator.wind_down()
        await self.reasoner.wind_down()

    async def execute_query(
        self,
        query: Query,
        logger: FileLogger,
        *guarantees: Guarantee,
    ) -> Tuple[TuningMaterializationPoint, CostSummary]:
        """Execute a query and return the result.
        :param query: The query to be executed.
        :param logger: The logger to be used.
        :return: The result of the query as a DataTable.
        """
        await self.prepare(logger)
        logical_plan = await self.reasoner.run(query, logger / "reasoner")
        logger.info(__name__, "Logical plan generated: ", str(logical_plan))
        tuning_workflow = await self.get_tuning_workflow(
            query, logical_plan, logger / "tuning_workflow"
        )
        cost = await self.interleaved_optimization_and_execution(
            tuning_workflow=tuning_workflow,
            guarantees=guarantees,
            logger=logger,
        )
        final_table = tuning_workflow.final_materialization_point
        await self.wind_down()
        return final_table, cost

    async def execute_logical_plan(
        self,
        logical_plan: LogicalPlan,
        guarantees: Iterable[Guarantee],
        logger: FileLogger,
    ):
        """Execute a logical plan and return the result.
        :param logical_plan: The logical plan to be executed.
        :param logger: The logger to be used.
        :return: The result of the query as a DataTable.
        """
        await self.prepare(logger)
        logical_plan = deepcopy(logical_plan)
        logical_plan.validate(self.database)
        tuning_workflow = await self.get_tuning_workflow(
            query=None, logical_plan=logical_plan, logger=logger / "tuning_workflow"
        )
        cost = await self.interleaved_optimization_and_execution(
            tuning_workflow=tuning_workflow,
            guarantees=guarantees,
            logger=logger / "interleaved_optimization_and_execution",
        )
        final_table = tuning_workflow.final_materialization_point
        await self.wind_down()
        return final_table, cost

    async def interleaved_optimization_and_execution(
        self,
        tuning_workflow: TuningWorkflow,
        guarantees: Iterable[Guarantee],
        logger: FileLogger,
    ) -> CostSummary:
        """Optimize and execute the plan section by section.
        :param tuning_workflow: The tuning plan to be executed
        :param logger: The logger to be used.
        """

        intermediate_state = self.initial_state
        self.last_tuned_pipelines = []
        collected_tuning_costs: List[ProfilingCost] = []
        collected_execution_costs: List[ProfilingCost] = []
        for materialization_stage in tuning_workflow.materialization_stages:
            for tuning_pipeline, materialization_point in materialization_stage:
                tuned_pipeline, tuning_cost = await self.tune_pipeline(
                    tuning_pipeline,
                    guarantees=guarantees,
                    intermediate_state=intermediate_state,
                    logger=logger / "tuning",
                )
                self.last_tuned_pipelines.append(tuned_pipeline.__str__())
                sql_query_or_data = await tuned_pipeline.execute(
                    intermediate_state=intermediate_state,
                    logger=logger / "execution",
                )
                logger.info(
                    __name__,
                    f"Materialized result in temporary table with name {materialization_point.identifier} ({materialization_point.tmp_table_name})",
                )
                if isinstance(sql_query_or_data, ResultData):
                    materialization_point.materialize_data(data=sql_query_or_data)
                    collected_execution_costs.append(sql_query_or_data.execution_cost)
                    collected_tuning_costs.append(tuning_cost)
                else:
                    materialization_point.materialize_sql(sql_query=sql_query_or_data)
                    collected_execution_costs.append(ProfilingCost(0.0, 0.0))
                    collected_tuning_costs.append(tuning_cost)
                self.to_clean_up.append(materialization_point.tmp_table_name)

                intermediate_state.add_materialization_point(materialization_point)

        return CostSummary(
            execution_cost=sum(collected_execution_costs, ProfilingCost(0.0, 0.0)),
            tuning_cost=sum(collected_tuning_costs, ProfilingCost(0.0, 0.0)),
        )

    def clean_up(self):
        """Clean up the temporary tables created during the execution."""
        for tbl in self.to_clean_up:
            self.database.drop_table(tbl)
            self.database.metadata.clean_up(tbl)

    async def tune_pipeline(
        self,
        pipeline: TuningPipeline,
        guarantees: Iterable[Guarantee],
        intermediate_state: IntermediateState,
        logger: FileLogger,
    ) -> Tuple[TunedPipeline, ProfilingCost]:
        """Tune a section of the plan and return it for execution.
        :param pipeline: The pipeline to be tuned.
        :param logger: The logger to be used.
        :return: The tuned pipeline.
        """
        if isinstance(pipeline, InitialTraditionalSection):
            return await TunedPipeline.from_traditional_section(
                pipeline, intermediate_state, logger=logger
            ), ProfilingCost(0.0, 0.0, 0.0)

        if isinstance(pipeline, AggregationSection):
            return await TunedPipeline.from_aggregation_section(
                pipeline, intermediate_state, logger=logger
            ), ProfilingCost(0.0, 0.0, 0.0)

        return await self.optimizer.tune_pipeline(
            pipeline=pipeline,
            intermediate_state=intermediate_state,
            guarantees=guarantees,
            logger=logger,
        )

    async def get_tuning_workflow(
        self, query: Optional[Query], logical_plan: LogicalPlan, logger: FileLogger
    ) -> TuningWorkflow:
        """Get the tuning plan for a query. A tuning plan divides the plan into sections that can be tuned and executed one after the other.
        :param query: The query to be executed.
        :param logical_plan: The logical plan to tune and execute.
        :param logger: The logger to be used.
        :return: The tuning plan for the query.
        """
        unoptimized_physical_plan = logical_plan.get_unoptimized_physical_plan()
        if unoptimized_physical_plan is None:
            unoptimized_physical_plan = await self.configurator.llm_configure(
                query, logical_plan, logger=logger
            )

        dependency_graph = DependencyGraph.from_unoptimized_physical_plan(
            unoptimized_physical_plan=unoptimized_physical_plan, database=self.database
        )
        tuning_workflow = dependency_graph.compute_tuning_workflow(self.database)
        logger.info(
            __name__,
            "Unoptimized physical plan: ",
            str(unoptimized_physical_plan),
        )
        return tuning_workflow

    @property
    def initial_state(self) -> IntermediateState:
        """Get the inital state of the database before execution."""
        return IntermediateState(self.database, plan_prefix=None)
