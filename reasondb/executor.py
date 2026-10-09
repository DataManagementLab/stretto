import asyncio
import os
import json
import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from copy import deepcopy

from reasondb.database.database import Database
from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.database.intermediate_state import IntermediateState
from reasondb.optimizer.base_optimizer import Optimizer
from reasondb.optimizer.label_optimizer import LabelOptimizer
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.optimizer.dependency_graph import DependencyGraph
from reasondb.optimizer.guarantees import Guarantee, PrecisionGuarantee, RecallGuarantee
from reasondb.query_plan.logical_plan import LogicalPlan
from reasondb.query_plan.optimized_physical_plan import ResultData, TunedPipeline
from reasondb.query_plan.physical_operator import (
    _CANONICAL_ALIAS_PREFIX,
    ProfilingCost,
    _canonicalize_expression,
)
from reasondb.evaluation import row_signature_sql as rss
from reasondb.evaluation.row_signature import RowSignature
from reasondb.utils import precompute_modalities
from reasondb.query_plan.query import Queries, Query
from reasondb.query_plan.tuning_workflow import (
    AggregationSection,
    InitialTraditionalSection,
    TuningMaterializationPoint,
    TuningPipeline,
    TuningWorkflow,
)
from reasondb.query_plan.unoptimized_physical_plan import UnoptimizedPhysicalPlan
from reasondb.backends.simulate_store import SimulateStore
from reasondb.monitor import collector as _monitor
from reasondb.reasoning.reasoner import Reasoner
from reasondb.utils.logging import FileLogger
from reasondb.utils.timing import measure, timing_session


def _answer_rows(answer: Union[pd.DataFrame, RowSignature]) -> int:
    """How many rows an answer has, whichever form it is in.

    ``n_rows`` on a signature is the count *before* dedup, so this reports the same
    number a frame would - the telemetry column must not change meaning with the form
    the answer happens to be stored in.
    """
    return answer.n_rows if isinstance(answer, RowSignature) else len(answer)


@dataclass
class ExecuteBenchmarkResults:
    #: One answer per query: the result frame, or - under
    #: ``execute_benchmark(reduce_results=True)`` - the :class:`RowSignature` it reduces
    #: to, so large answers are freed as soon as they have been hashed.
    results: Dict[str, Union[pd.DataFrame, RowSignature]]
    tuned_pipelines: Dict[str, List[str]]
    costs: Dict[str, "CostSummary"]
    #: Where each answer was cached, when a ``results_cache_dir`` was given. A job's
    #: shard records these paths instead of a second copy of the answers, so the hashes
    #: exist once on disk and the coordinator can load one query's at a time.
    result_paths: Dict[str, Path] = field(default_factory=dict)


@dataclass
class CostSummary:
    execution_cost: ProfilingCost
    tuning_cost: ProfilingCost
    # Wall-clock per phase (seconds), simulate-corrected. See utils/timing.py.
    component_times: Dict[str, float] = field(default_factory=dict)
    #: How many tuples a human had to label to answer this query, under
    #: ``--human-labels``. A count, not a cost, and not part of ``total_cost``:
    #: annotation is paid once, up front, not per tuple at query time. Zero when labels
    #: come from a model.
    n_labels_requested: int = 0
    #: What the optimizer concluded about this query's guarantees:
    #: ``guarantee_met`` and the achieved lower bounds. Empty for optimizers that report
    #: nothing (the baselines, the label pass).
    guarantee: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> ProfilingCost:
        return self.execution_cost + self.tuning_cost

    def to_json(self):
        return {
            "execution_cost": self.execution_cost.to_json(),
            "tuning_cost": self.tuning_cost.to_json(),
            "component_times": self.component_times,
            "n_labels_requested": self.n_labels_requested,
            "guarantee": self.guarantee,
        }

    @staticmethod
    def from_json(json_obj):
        return CostSummary(
            execution_cost=ProfilingCost.from_json(json_obj["execution_cost"]),
            tuning_cost=ProfilingCost.from_json(json_obj["tuning_cost"]),
            component_times=json_obj.get("component_times", {}),
            # Optional fields; default when absent from a cached result.
            n_labels_requested=json_obj.get("n_labels_requested") or 0,
            guarantee=json_obj.get("guarantee") or {},
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
        logger: Optional[FileLogger] = None,
        role: Optional[str] = None,
    ):
        """Initialize the executor.
        :param database: The database where all the data is stored.
        :param reasoner: The reasoner that translates the natural language query into a logical plan.
        :param optimizer: The optimizer that tunes the physical plan.
        :param configurator: The configurator that configures the physical operators.
        :param logger: Where operator/backend logs go. Defaults to ``FileLogger()``,
            which writes under ``logging/<timestamp>`` relative to cwd. Callers that need
            per-job/per-worker isolation (the coordinator) pass an explicit
            ``FileLogger(log_root_path=...)``.
        :param role: ``"sweep"`` (a measured point) or ``"label"`` (a labelling pass,
            run only to produce ground truth for scoring). It is attached to every
            telemetry event this executor emits, so the monitor can exclude label passes
            from its statistics. Derived from the optimizer when not given; pass it
            explicitly only to override that, e.g. to measure a ``LabelOptimizer`` run
            as an approach in its own right.
        """
        if role is None:
            role = "label" if isinstance(optimizer, LabelOptimizer) else "sweep"
        assert role in ("sweep", "label"), (
            f"role must be 'sweep' or 'label'; got {role!r}."
        )
        self.role = role
        self.name = name
        self.database = database
        self.reasoner = reasoner
        self.configurator = configurator
        self.optimizer = optimizer
        self.configurator.set_database(self.database)
        self.reasoner.set_database(self.database)
        self.optimizer.set_database(self.database)
        self.logger = logger if logger is not None else FileLogger()
        self.to_clean_up: List[str] = []
        self.last_tuned_pipelines: List[str] = []
        #: One per tuned pipeline of the most recent query, when the optimizer produces
        #: them. Says whether each pipeline's guarantees were met, and whether they were
        #: reachable at all.
        self.last_optimization_reports: List = []

    def check_cached_results(
        self,
        results_cache_dir: Optional[Path],
        query_str: str,
        guarantees: Sequence[Guarantee],
        force_running_queries: List[str],
        logger: FileLogger,
        reduced: bool = False,
    ):
        if results_cache_dir is None:
            return
        if query_str in force_running_queries:
            return
        json_filepath, data_filepath, key = self.get_cache_filepath(
            dir=results_cache_dir,
            guarantees=guarantees,
            query_str=query_str,
            reduced=reduced,
        )
        if os.path.exists(json_filepath) and os.path.exists(data_filepath):
            try:
                with open(json_filepath, "r") as jf:
                    json_data = json.load(jf)[key]
                with open(data_filepath, "rb") as df:
                    data = pickle.load(df) if reduced else pd.read_parquet(df)
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
        self,
        dir: Path,
        guarantees: Sequence[Guarantee],
        query_str: str,
        reduced: bool = False,
    ):
        """Where this (executor, query, guarantee) caches its answer and its metadata.

        The two forms of the answer get different extensions rather than a version field
        inside one: a reducing caller looks only for ``.sig2.pkl`` and a frame caller only
        for ``.parquet``, so a cache directory written in one mode is simply a miss in the
        other and can never be half-read.

        The ``2`` is the hash scheme version: signatures from a different scheme are
        incomparable, so encoding it in the filename turns them into a plain cache miss.
        Bump this alongside ``row_signature_sql.SCHEME``.
        """
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
        suffix = "sig2.pkl" if reduced else "parquet"
        data_path = dir / (f"{filekey}_{h}.{suffix}")
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
        """Write one query's answer and metadata, so a re-run of this job skips it.

        The answer is stored in whatever form the caller holds. A
        :class:`RowSignature` is all that scoring reads, so a reducing job caches that
        and avoids rehydrating a large parquet file into a frame on a cache hit.
        """
        if results_cache_dir is None:
            return
        reduced = isinstance(results, RowSignature)
        json_filepath, data_filepath, key = self.get_cache_filepath(
            dir=results_cache_dir,
            guarantees=guarantees,
            query_str=query_str,
            reduced=reduced,
        )
        if reduced:
            with open(data_filepath, "wb") as f:
                pickle.dump(results, f)
        else:
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
        reduce_results: bool = False,
    ) -> ExecuteBenchmarkResults:
        """Execute the queries and return the results.

        :param queries: The queries to be executed.
        :param reduce_results: return (and cache) each answer as a
            :class:`RowSignature` rather than as its frame. Every caller that only
            *scores* the answer wants this, so the frame does not stay in memory for
            the rest of the benchmark. The frame is needed only where something reads its
            cells or index (e.g. ``filter_stats.compute_filter_stats``).
        :return: The results of the queries. Mapping from query string to answer.
        """
        results = {}
        tuned_pipelines = {}
        costs = {}
        result_paths: Dict[str, Path] = {}
        self.logger.info(__name__, f"Executing benchmark with {len(queries)} queries")
        precision_guarantee = next(
            (g.value for g in guarantees if isinstance(g, PrecisionGuarantee)), None
        )
        recall_guarantee = next(
            (g.value for g in guarantees if isinstance(g, RecallGuarantee)), None
        )
        _monitor.record_executor_start(
            executor=self.name,
            role=self.role,
            precision_guarantee=precision_guarantee,
            recall_guarantee=recall_guarantee,
        )
        for i, query in enumerate(queries):
            logger = self.logger / f"query-{i}"
            _monitor.record_query_start(
                executor=self.name,
                role=self.role,
                query=query.query,
                query_index=i,
                n_queries=len(queries),
            )
            if results_cache_dir is not None:
                # Recorded whether this query is about to be executed or read back: the
                # path is a function of (executor, query, guarantees, form), so it is the
                # same either way, and the shard needs it in both cases.
                _, data_filepath, _ = self.get_cache_filepath(
                    dir=results_cache_dir,
                    guarantees=guarantees,
                    query_str=query.query,
                    reduced=reduce_results,
                )
                result_paths[query.query] = data_filepath
            cached_result = self.check_cached_results(
                results_cache_dir=results_cache_dir,
                query_str=query.query,
                guarantees=guarantees,
                force_running_queries=force_running_queries,
                logger=logger,
                reduced=reduce_results,
            )
            if cached_result is not None:
                (
                    results[query.query],
                    costs[query.query],
                    tuned_pipelines[query.query],
                ) = cached_result
                cached_cost = costs[query.query]
                _monitor.record_query_end(
                    executor=self.name,
                    role=self.role,
                    query=query.query,
                    query_index=i,
                    cached=True,
                    # Recorded on the cached path too, so the column does not mean two
                    # different things depending on cache state.
                    n_rows=_answer_rows(results[query.query]),
                    component_times=getattr(cached_cost, "component_times", {}),
                    tuned_pipeline=tuned_pipelines[query.query],
                )
                continue

            if reset_db_before_each_query:
                self.database.reset()

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

            if reduce_results:
                # The answer is never materialized in Python: the reduction the scorer
                # needs is one SQL statement over the relation the pipeline just wrote.
                answer = asyncio.run(
                    self.extract_signature(r, logger=self.logger / f"extract-{i}")
                )
                n_rows = answer.n_rows
            else:
                df = asyncio.run(
                    self.extract_data(r, logger=self.logger / f"extract-{i}")
                )
                answer = df
                n_rows = len(df)
                del df
            results[query.query] = answer
            costs[query.query] = c
            tuned_pipelines[query.query] = self.last_tuned_pipelines
            self.cache_result(
                results_cache_dir=results_cache_dir,
                query_str=query.query,
                guarantees=guarantees,
                results=answer,
                costs=c,
                tuned_pipelines=self.last_tuned_pipelines,
                logger=logger,
            )
            _monitor.record_query_end(
                executor=self.name,
                role=self.role,
                query=query.query,
                query_index=i,
                cached=False,
                n_rows=n_rows,
                component_times=getattr(c, "component_times", {}),
                tuned_pipeline=self.last_tuned_pipelines,
                **self._guarantee_telemetry(),
            )
            self.clean_up()
        return ExecuteBenchmarkResults(
            results=results,
            tuned_pipelines=tuned_pipelines,
            costs=costs,
            result_paths=result_paths,
        )

    def _guarantee_telemetry(self) -> Dict[str, Any]:
        """What the optimizer concluded about this query's guarantees, for the monitor.

        A query is feasible only if *every* one of its tuned pipelines was: the guarantee
        is end-to-end, so one unreachable step makes the whole query unreachable. The
        reported precision/recall are the worst across pipelines for the same reason.

        Empty when no pipeline produced a report (the baselines and the label pass);
        these keys are optional for the collector.
        """
        reports = [r for r in self.last_optimization_reports if r is not None]
        if not reports:
            return {}
        return {
            # False means the optimizer exhausted its sampling rounds without reaching
            # the target, and the plan fell back to the highest-quality executable
            # operator -- i.e. the guarantee was unreachable for this query.
            "guarantee_met": all(r.meets_targets for r in reports),
            "achieved_precision_lower": min(
                r.achieved_precision_lower for r in reports
            ),
            "achieved_recall_lower": min(r.achieved_recall_lower for r in reports),
        }

    async def extract_data(
        self, query_results: TuningMaterializationPoint, logger
    ) -> pd.DataFrame:
        data_iterator = await query_results.get_data(logger=logger)
        index_names = [c.col_name for c in query_results.index_columns]
        df = data_iterator.to_df_with_index(index_names)
        return df

    async def extract_signature(
        self, query_results: TuningMaterializationPoint, logger
    ) -> RowSignature:
        """What :meth:`extract_data` fetches, reduced without being fetched.

        Goes through the same ``get_data`` call so that the columns compared are the
        ``DataIterator``'s own ``column_names`` - the set ``extract_data`` would have
        projected, with the ``_index_`` / ``_flag_`` / ``__random__`` columns already
        split off, so both paths share a single projection.
        """
        data_iterator = await query_results.get_data(logger=logger)
        assert data_iterator.column_names is not None
        columns = list(data_iterator.column_names)
        state = getattr(self, "last_intermediate_state", None)
        points = list(state.materialization_points) if state is not None else []
        try:
            sides = rss.plan_decomposition(
                query_results, points, self.database, columns
            )
        except Exception as exc:  # noqa: BLE001
            # Planning is only an optimization: fall back to row-by-row hashing, but
            # log a warning since this is unexpected (unlike a declined decomposition).
            logger.warning(
                __name__, f"Could not plan a decomposed reduction ({exc!r}); "
                "hashing the answer row by row instead."
            )
            sides = None
        if sides:
            logger.info(
                __name__,
                "Reducing the answer by side: "
                + ", ".join(
                    f"{s.source_table}[{s.index_column}] x{len(s.columns)}"
                    for s in sides
                ),
            )
        return RowSignature.from_answer(
            self.database,
            data_iterator.sql_str,
            columns,
            sides=sides,
            index_columns=[c.col_name for c in query_results.index_columns],
        )

    async def setup(self):
        """Setup the executor by setting up the database and all physical operators registered in the configurator."""
        self.configurator.setup(self.database, self.logger / "setup-optimizer")

    def shutdown(self):
        """Shutdown the executor by shutting down all physical operators registered in the configurator."""
        self.configurator.shutdown(self.logger / "shutdown")
        self.database.cleanup_backups()

    def __enter__(self):
        """Setup as a context manager."""
        asyncio.run(self.setup())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shutdown as a context manager."""
        self.shutdown()

    async def prepare(self, logger: Optional[FileLogger] = None):
        """Prepare the executor by preparing the database and all physical operators registered in the configurator."""
        logger = logger or self.logger
        await self.database.prepare(logger / "prepare-database")
        await self.configurator.prepare(self.database, logger / "prepare-configurator")
        self.database.snapshot_external_tables()
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
        with timing_session() as session, measure("end_to_end"):
            await self.prepare(logger)
            with measure("reasoning"):
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
        cost.component_times = dict(session.durations)
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
        with timing_session() as session, measure("end_to_end"):
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
        cost.component_times = dict(session.durations)
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
        self.last_optimization_reports = []
        # Kept for `extract_signature`, which needs the materialization points this loop
        # accumulates. `initial_state` builds a fresh `IntermediateState` on every access,
        # so it cannot be re-read later.
        self.last_intermediate_state = intermediate_state
        collected_tuning_costs: List[ProfilingCost] = []
        collected_execution_costs: List[ProfilingCost] = []
        collected_n_labels: List[int] = []
        for materialization_stage in tuning_workflow.materialization_stages:
            for tuning_pipeline, materialization_point in materialization_stage:
                tuned_pipeline, tuning_cost = await self.tune_pipeline(
                    tuning_pipeline,
                    guarantees=guarantees,
                    intermediate_state=intermediate_state,
                    logger=logger / "tuning",
                )
                # Reported alongside the optimizer's costs rather than through
                # `tune_pipeline`'s return type - see
                # `Optimizer.last_n_labels_requested`.
                collected_n_labels.append(self.optimizer.last_n_labels_requested)
                if self.optimizer.last_optimization_report is not None:
                    self.last_optimization_reports.append(
                        self.optimizer.last_optimization_report
                    )
                self.last_tuned_pipelines.append(tuned_pipeline.__str__())
                with measure("execution"):
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
            n_labels_requested=sum(collected_n_labels),
            guarantee=self._guarantee_telemetry(),
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
        # "tuning" spans profiling + optimization; the profiler records its own
        # nested "profiling" span, so optimization = tuning - profiling.
        with measure("tuning"):
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
            with measure("configuring"):
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
        """Get the initial state of the database before execution."""
        return IntermediateState(self.database, plan_prefix=None)

    def precompute_query(self, query, index: int) -> None:
        """Run query planning and execute every multi-modal operator for a single query."""
        logger = self.logger / f"precompute-{index}"
        self.database.reset()
        try:
            logical_plan = query.get_gt_logical_plan()
        except Exception as e:
            logger.warning(__name__, f"[precompute] No GT logical plan for query {index}: {e}")
            return
        asyncio.run(self.precompute_logical_plan(logical_plan, logger))
        self.clean_up()

    def precompute_benchmark(self, queries: Queries) -> None:
        """Run query planning for each query, then execute every multi-modal operator
        on all data items without running any optimization.
        """
        for i, query in enumerate(queries):
            self.precompute_query(query, i)

    async def precompute_logical_plan(
        self,
        logical_plan: LogicalPlan,
        logger: FileLogger,
    ) -> None:
        """Plan the query and run every multi-modal operator on all data items.

        Unlike execute_logical_plan, this skips optimization. Traditional and
        aggregation sections are executed normally so that the intermediate state
        is populated for subsequent stages. Multi-modal pipeline stages run every
        operator on the full (unfiltered) input table so that the SimulateStore
        captures all (operator, row) combinations.
        """
        await self.prepare(logger)
        logical_plan = deepcopy(logical_plan)
        logical_plan.validate(self.database)
        tuning_workflow = await self.get_tuning_workflow(
            query=None, logical_plan=logical_plan, logger=logger / "tuning_workflow"
        )

        intermediate_state = self.initial_state

        for materialization_stage in tuning_workflow.materialization_stages:
            for tuning_pipeline, materialization_point in materialization_stage:
                if isinstance(tuning_pipeline, (InitialTraditionalSection, AggregationSection)):
                    tuned_pipeline, _ = await self.tune_pipeline(
                        tuning_pipeline,
                        guarantees=[],
                        intermediate_state=intermediate_state,
                        logger=logger,
                    )
                    result = await tuned_pipeline.execute(
                        intermediate_state=intermediate_state, logger=logger
                    )
                    if isinstance(result, ResultData):
                        materialization_point.materialize_data(data=result)
                    else:
                        materialization_point.materialize_sql(sql_query=result)
                    self.to_clean_up.append(materialization_point.tmp_table_name)
                    intermediate_state.add_materialization_point(materialization_point)
                else:
                    await self._precompute_pipeline(tuning_pipeline, intermediate_state, logger)

        await self.wind_down()

    async def _precompute_pipeline(
        self,
        pipeline: TuningPipeline,
        intermediate_state: IntermediateState,
        logger: FileLogger,
    ) -> None:
        """Run every operator with prefers_run_outside_db in a multi-modal pipeline
        on ALL rows of the pipeline's base input tables.

        For cascades with multiple levels, the base data (full, unfiltered) is reused
        at every level so that no model call is skipped due to filtering. After each
        level the internal intermediate state is updated (via UnoptimizedPhysicalPlan.append)
        so that virtual column references in later steps resolve correctly.
        """
        # Fetch ALL rows from each base input table of each cascade.
        base_input: Dict[VirtualTableIdentifier, pd.DataFrame] = {}
        for cascade in pipeline.steps_in_parallel:
            first_step = cascade[0]
            for tbl in first_step.inputs:
                if tbl in base_input:
                    continue
                [data] = await first_step.get_input_sample(
                    database_state=intermediate_state,
                    logger=logger,
                    limit=None,
                    for_prompt=True,
                    override_inputs=[tbl],
                )
                base_input[tbl] = data

        for cascade in pipeline.steps_in_parallel:
            profiling_pipeline = UnoptimizedPhysicalPlan()
            current_state = intermediate_state
            cascade_base_tables = cascade[0].inputs

            for step in cascade:
                # Map this step's inputs to base-table data. For level-0 steps the
                # mapping is direct. For level-1+ steps (whose inputs are intermediate
                # filtered tables not in base_input) fall back to the cascade's base
                # table at the same position.
                data_sample = []
                for j, tbl in enumerate(step.inputs):
                    if tbl in base_input:
                        data_sample.append(base_input[tbl])
                    else:
                        fallback = cascade_base_tables[min(j, len(cascade_base_tables) - 1)]
                        data_sample.append(base_input.get(fallback, pd.DataFrame()))

                if not any(len(d) > 0 for d in data_sample):
                    break

                precompute_store = SimulateStore.get_precompute()
                first_valid_observation = None
                for op_idx, operator in enumerate(step.operators):
                    if not operator.prefers_run_outside_db:
                        continue
                    if getattr(operator, "is_label_only", False):
                        # A label operator reads a CSV, not a model, so there is nothing
                        # to record for a later --simulate run.
                        continue
                    llm_parameters = step.llm_configurations[
                        operator.get_llm_parameters().name
                    ]
                    # No try/except here: a failure precomputing one operator aborts the
                    # whole precompute run, rather than silently leaving that operator
                    # uncached for later `--simulate` runs.
                    obs_data = [None] * len(step.inputs)
                    if operator.requires_data_sample():
                        obs_data = data_sample
                    observation = await operator.get_observation(
                        database_state=current_state,
                        inputs=step.inputs,
                        output=step.output,
                        output_columns=step.get_output_columns(),
                        llm_parameters=llm_parameters,
                        data_sample=obs_data,
                        logical_plan_step=step.logical_plan_step,
                        logger=logger,
                    )
                    step.observations[op_idx] = observation
                    if first_valid_observation is None:
                        first_valid_observation = observation

                    # Canonicalize on alias position rather than literal table name:
                    # the same operator/expression can recur against a base table
                    # accessed directly (e.g. "reviews") in one query and against an
                    # auto-generated intermediate materialization of the same
                    # underlying data in another (as in `_pin_operator_config`).
                    canonical_expression = _canonicalize_expression(
                        step.logical_plan_step.expression, step.inputs
                    )
                    canonical_base_tables = ",".join(
                        f"{_CANONICAL_ALIAS_PREFIX}{i}"
                        for i in range(len(cascade_base_tables))
                    )
                    precompute_key = (
                        f"{operator.get_operation_identifier()}"
                        f"|{canonical_expression}"
                        f"|{canonical_base_tables}"
                    )
                    if precompute_store is not None and precompute_store.is_op_precomputed(precompute_key):
                        logger.info(
                            __name__,
                            f"[precompute] skipping {operator.get_operation_identifier()} (already precomputed)",
                        )
                    elif operator.get_modality() in precompute_modalities.PRECOMPUTE_SKIP_MODALITIES:
                        # Not marked precomputed: a later run with this modality's
                        # server up must still fill it in.
                        logger.info(
                            __name__,
                            f"[precompute] skipping {operator.get_operation_identifier()} "
                            f"(modality {operator.get_modality()!r} in REASONDB_PRECOMPUTE_SKIP_MODALITIES)",
                        )
                    else:
                        await operator.run_outside_db(
                            inputs=step.inputs,
                            input_data=data_sample,
                            llm_parameters=llm_parameters,
                            database_state=current_state,
                            observation=observation,
                            labels=step.logical_plan_step.get_labels(),
                            logger=logger,
                        )
                        if precompute_store is not None:
                            precompute_store.mark_op_precomputed(precompute_key)

                # Register this step's output in current_state so the next level can
                # resolve virtual columns that reference this step's output table.
                if first_valid_observation is not None:
                    current_state = profiling_pipeline.append(
                        step=step,
                        observation=first_valid_observation,
                        database=current_state,
                    )
