"""Shared benchmark-evaluation helpers.

Metric computation, dataframe post-processing, wall-clock breakdown and the gold-label
configurator, shared by the benchmark and sweep entry points.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from reasondb.evaluation.metrics.metrics_manager import (
    confusion_counts,
    precision_recall_f1,
)
from reasondb.evaluation.row_signature import SAMPLE_ROWS, RowSignature
from reasondb.executor import CostSummary

# Module import, never by value: the monitor rebinds its global sink at run start.
from reasondb.monitor import collector as _monitor
from reasondb.operators.aggregate.aggregate import Aggregate
from reasondb.operators.aggregate.groupby import GroupBy
from reasondb.operators.join.traditional_join import TraditionalJoin
from reasondb.operators.limit.limit import Limit
from reasondb.operators.perfect_operators.perfect_extract import PerfectExtract
from reasondb.operators.perfect_operators.perfect_filter import PerfectFilter
from reasondb.operators.perfect_operators.perfect_transform import PerfectTransform
from reasondb.operators.project.project import Project
from reasondb.operators.rename.rename import Rename
from reasondb.operators.sorting.sort import Sort
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.query_plan.physical_operator import CostType, PhysicalOperatorToolbox
from reasondb.reasoning.llm import GPT4o
from reasondb.utils.answer_normalization import (
    postprocess_dataframe as _postprocess_dataframe,
    postprocess_string as _postprocess_string,
)

logger = logging.getLogger(__name__)


# Re-exported so this stays the import site for callers, while the definition lives in
# reasondb.utils.answer_normalization, which the profiler can import and this module
# cannot be imported from.
postprocess_string = _postprocess_string
postprocess_dataframe = _postprocess_dataframe



def _record_query_metrics(
    benchmark_name: str,
    approach_name: str,
    query: str,
    precision_guarantee: Optional[float],
    recall_guarantee: Optional[float],
    metrics: Dict[str, float],
    context: Optional[Dict[str, Any]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Report one query's achieved accuracy to the monitor.

    ``evaluate()`` is the only place per-query precision/recall is known, so emitting
    from here covers every caller without each plumbing its own telemetry.

    ``context`` carries what the caller knows and this function cannot infer:

    - ``labels``: which label set ("silver" or "gold") the measurement was scored
      against, so rows scored against different label sets stay distinguishable.
    - ``job_id``/``worker_id``: needed when scoring happens inside the coordinator
      process, whose events are not stamped by ``coordinator.ingest.ingest_events``.

    ``stats`` carries the query's shape statistics (``num_semops`` and friends, from
    ``QueryShape.additional_info``) so accuracy can be grouped by query complexity.

    A no-op unless monitoring is enabled; never fails an evaluation.
    """
    if not _monitor.is_enabled():
        return
    try:
        _monitor.record_query_metrics(
            benchmark=benchmark_name,
            executor=approach_name,
            query=query,
            precision_guarantee=precision_guarantee,
            recall_guarantee=recall_guarantee,
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1_score"),
            predicted_output_cardinality=metrics.get("predicted_output_cardinality"),
            true_output_cardinality=metrics.get("true_output_cardinality"),
            **(stats or {}),
            **(context or {}),
        )
    except Exception as exc:  # pragma: no cover - telemetry must never break evaluation
        logger.warning("Monitor: could not record query metrics: %s", exc)


#: Columns `guarantee_metrics` emits, in the order they appear. Named so the monitor and
#: the plotting scripts can select them without hardcoding strings twice.
GUARANTEE_COLUMNS = (
    "guarantee_met",
    "achieved_precision_lower",
    "achieved_recall_lower",
    "n_labels_requested",
)


def guarantee_metrics(cost: CostSummary) -> Dict[str, Any]:
    """What the optimizer achieved against this query's guarantees.

    ``guarantee_met=False`` means the optimizer used up its sampling rounds (see
    ``OptimizationConfig.max_sampling_rounds``, one by default) without reaching the
    target, and the plan fell back to the highest-quality executable operator everywhere.
    Read it as *unreachable at this sampling budget*, and read
    ``achieved_precision_lower`` / ``achieved_recall_lower`` for how far short it fell.

    ``None`` rather than a default when the optimizer produced no report (the baselines,
    the label pass): "not measured" and "measured and met" are different claims.
    """
    guarantee = getattr(cost, "guarantee", None) or {}
    return {
        "guarantee_met": guarantee.get("guarantee_met"),
        "achieved_precision_lower": guarantee.get("achieved_precision_lower"),
        "achieved_recall_lower": guarantee.get("achieved_recall_lower"),
        # Annotation burden as a count, kept out of the cost_* columns since pricing a
        # human label is an assumption. Zero unless --human-labels was used.
        "n_labels_requested": getattr(cost, "n_labels_requested", 0),
    }


def time_metrics(cost: CostSummary) -> Dict[str, float]:
    """Per-phase wall-clock metrics (seconds) derived from a CostSummary.

    ``optimization`` and ``engine_runtime`` are derived: the profiler nests a
    "profiling" span inside "tuning", and reasoning/configuring are the query
    planning phases excluded from the engine runtime.
    """
    t = cost.component_times or {}
    reasoning = t.get("reasoning", 0.0)
    configuring = t.get("configuring", 0.0)
    profiling = t.get("profiling", 0.0)
    tuning = t.get("tuning", 0.0)
    execution = t.get("execution", 0.0)
    end_to_end = t.get("end_to_end", 0.0)
    return {
        "time_reasoning": reasoning,
        "time_configuring": configuring,
        "time_profiling": profiling,
        "time_optimization": max(tuning - profiling, 0.0),
        "time_execution": execution,
        "time_end_to_end": end_to_end,
        # End-to-end excluding query planning (profiling + optimization + execution).
        "time_engine_runtime": max(end_to_end - reasoning - configuring, 0.0),
    }


def _write_debug_sample(
    txt_path: Path, csv_path: Path, header_lines: list, signature: "RowSignature"
) -> None:
    """One frame's shape and a sample of it, under ``debug_outputs``.

    The CSV holds only ``row_signature.SAMPLE_ROWS`` rows, since full answers (e.g. of a
    self-join) can be very large; the header records the true shape.
    """
    with open(txt_path, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Shape: ({signature.n_rows}, {signature.n_columns})\n")
        f.write(f"Distinct comparison rows: {signature.n_unique}\n")
        f.write(f"Columns: {signature.columns}\n\n")
        f.write(f"(CSV holds the first {SAMPLE_ROWS} rows.)\n\n")
    if signature.sample is not None:
        signature.sample.to_csv(csv_path)


def _score_signatures(
    query: str, predictions: "RowSignature", ground_truth: "RowSignature"
) -> Dict[str, float]:
    """Precision/recall/F1 for one (predictions, labels) pair.

    Two answers of different widths cannot share a row, so every metric is zero; this is
    logged as a warning because it indicates the two answers have different schemas.
    """
    if predictions.n_columns != ground_truth.n_columns and not predictions.is_empty:
        logger.warning(
            "Predictions and labels for query %r have different widths (%d vs %d "
            "columns: %s vs %s); no row can match and all metrics will be 0.",
            query, predictions.n_columns, ground_truth.n_columns,
            predictions.columns, ground_truth.columns,
        )
    if predictions.scheme != ground_truth.scheme:
        # An error rather than a warning: hashes from different schemes are not
        # comparable, so every metric would silently come out ~0.
        raise ValueError(
            f"Predictions for query {query!r} were hashed by scheme "
            f"{predictions.scheme!r} and their labels by {ground_truth.scheme!r}. "
            f"Row hashes from two schemes are not comparable; re-run whichever side is "
            f"stale (its cached answers are keyed by scheme, so they will be recomputed)."
        )
    if predictions.pandas_version != ground_truth.pandas_version:
        logger.warning(
            "Predictions for query %r were hashed by pandas %s and their labels by "
            "pandas %s. Row hashes are only comparable within one pandas version, so "
            "these metrics may be meaningless - re-run the older side.",
            query, predictions.pandas_version, ground_truth.pandas_version,
        )
    return precision_recall_f1(
        *confusion_counts(predictions.hashes, ground_truth.hashes)
    )


def evaluate(
    benchmark_name: str,
    approach_name: str,
    # query -> guarantee -> RowSignature (or query -> RowSignature when there are no
    # guarantees). Never a frame: `Executor.execute_benchmark(reduce_results=True)`
    # reduces each answer where it is produced, and a shard stores only the reduction.
    all_predictions: Union[
        Dict[str, Dict[Tuple[float, float], RowSignature]], Dict[str, RowSignature]
    ],
    all_labels: Dict[str, RowSignature],  # query -> signature
    all_costs: Union[
        Dict[str, Dict[Tuple[float, float], CostSummary]], Dict[str, CostSummary]
    ],
    debug_query: Optional[str] = None,
    debug_root: Path = Path("debug_outputs"),
    record_telemetry: bool = True,
    telemetry_context: Optional[Dict[str, Any]] = None,
    query_stats: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Compute precision/recall/F1 (+ cost/time columns) per (query, guarantee).

    Returns a DataFrame indexed by
    ``(approach_name, query, precision_guarantee, recall_guarantee)``. When an
    approach was run without guarantees, the two guarantee levels are ``None``.

    ``record_telemetry=False`` computes the same numbers without reporting them to the
    monitor. The coordinator uses it for its final merge pass, since each job was already
    reported when it was first scored (see ``coordinator.scoring``).

    ``telemetry_context`` is passed through to every emitted row - see
    :func:`_record_query_metrics` for what belongs in it and why.

    ``query_stats`` maps a query string to that query's shape statistics (``num_semops``,
    ``num_sem_filter``, ...). They are added to both the returned DataFrame and the
    telemetry row; queries missing from the mapping (or ``None``) get no such columns.
    """
    # Create debug directory for storing predictions and ground truth
    debug_dir = debug_root / benchmark_name / approach_name
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
    for query, gt_signature in tqdm(all_labels.items(), "Running Evaluation"):
        if debug_query is not None and query != debug_query:
            continue

        if gt_signature.is_empty:
            logger.warning(f"Ground truth for query {query} is empty. Skipping.")
            continue
        query_id = query_to_id[query]
        predictions_or_per_target = all_predictions[query]

        logger.info(f"Ground truth size: {gt_signature.n_rows} rows")
        _write_debug_sample(
            debug_dir / f"{query_id}_ground_truth.txt",
            debug_dir / f"{query_id}_ground_truth_sample.csv",
            [f"Ground Truth for query: {query} (POST-PROCESSED)"],
            gt_signature,
        )

        if not isinstance(predictions_or_per_target, dict):
            pred_signature = predictions_or_per_target
            _write_debug_sample(
                debug_dir / f"{query_id}_predictions_no_guarantees.txt",
                debug_dir / f"{query_id}_predictions_no_guarantees_sample.csv",
                [f"Predictions for query: {query} (no guarantees) (POST-PROCESSED)"],
                pred_signature,
            )

            cost = all_costs[query]
            assert isinstance(cost, CostSummary)
            metrics = _score_signatures(query, pred_signature, gt_signature)
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
                metrics["true_output_cardinality"] = gt_signature.n_rows
                metrics["predicted_output_cardinality"] = pred_signature.n_rows
            metrics.update(time_metrics(cost))
            metrics.update(guarantee_metrics(cost))
            stats = (query_stats or {}).get(query, {})
            metrics.update(stats)
            all_metrics[approach_name, query, None, None] = metrics
            if record_telemetry:
                _record_query_metrics(
                    benchmark_name, approach_name, query, None, None, metrics,
                    context=telemetry_context, stats=stats,
                )
            continue
        else:
            for prec_rec, predictions in tqdm(
                predictions_or_per_target.items(),
                position=1,
                leave=False,
                desc="Multiple Targets",
            ):
                pred_signature = predictions
                logger.info(f"Prediction size: {pred_signature.n_rows} rows")
                _write_debug_sample(
                    debug_dir
                    / f"{query_id}_predictions_prec_{prec_rec[0]}_rec_{prec_rec[1]}.txt",
                    debug_dir
                    / f"{query_id}_predictions_prec_{prec_rec[0]}_rec_{prec_rec[1]}_sample.csv",
                    [
                        f"Predictions for query: {query} (POST-PROCESSED)",
                        f"Precision guarantee: {prec_rec[0]}, Recall guarantee: {prec_rec[1]}",
                    ],
                    pred_signature,
                )

                logger.info("Compute Metrics now")
                metrics = _score_signatures(query, pred_signature, gt_signature)

                logger.info("Compute cost")
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
                    metrics["true_output_cardinality"] = gt_signature.n_rows
                    metrics["predicted_output_cardinality"] = pred_signature.n_rows
                metrics.update(time_metrics(cost))
                metrics.update(guarantee_metrics(cost))
                stats = (query_stats or {}).get(query, {})
                metrics.update(stats)
                all_metrics[approach_name, query, prec_rec[0], prec_rec[1]] = metrics
                if record_telemetry:
                    _record_query_metrics(
                        benchmark_name, approach_name, query, prec_rec[0], prec_rec[1],
                        metrics, context=telemetry_context, stats=stats,
                    )

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


def get_label_configurator(llm=None) -> PlanConfigurator:
    """Configurator of perfect (ground-truth) operators, used to produce gold labels.

    Pair with ``LabelOptimizer`` in an ``Executor`` to materialize the
    ground-truth answer for each query.
    """
    return PlanConfigurator(
        llm=llm or GPT4o(),
        physical_operators=PhysicalOperatorToolbox(
            join_operators=[TraditionalJoin(quality=1, fake_cost=0)],
            join_predicates=[PerfectFilter(quality=1, fake_cost=0)],
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
