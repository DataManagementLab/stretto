"""Running a benchmark for one executor, and turning the results into artifacts.

The two ``collect_*`` functions drive an ``Executor`` over every query of a
benchmark - with a guarantee sweep, or once without guarantees for a labelling
pass - and return the ``(answer_paths, pipeline_tracks, costs)`` triple everything
downstream is built from. Paths rather than answers: both ask the executor to reduce
each result to its row signature and cache it, so what comes back names where each one
is rather than carrying a second copy of it (see
:mod:`reasondb.evaluation.row_signature`). ``save_pipeline_tracks_to_yaml`` and
``compute_operator_stats`` turn that triple into the ``pipeline_tracks.yaml`` and
``operator_stats.csv`` artifacts ``scripts/plot_benchmark.py`` reads.

Both ``collect_*`` functions take an optional ``results_cache_dir`` (default
``out_dir / "cache"``). The result cache is keyed only by ``(query, guarantees)``, so
runs that differ in anything else must use separate cache directories.

This module deliberately does not import :mod:`reasondb.evaluation.evaluation`:
that one is about *scoring* predictions against labels, and it imports
``reasondb.executor``, so pairing the two here would build an import cycle.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import yaml

from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee

logger = logging.getLogger(__name__)


def collect_results_all_guarantees(
    out_dir,
    executor_name,
    executor,
    benchmark,
    precision_guarantees,
    recall_guarantees,
    all_combinations,
    debug_query,
    results_cache_dir: Optional[Path] = None,
):
    cache_dir = results_cache_dir if results_cache_dir is not None else out_dir / "cache"
    queries = benchmark.queries
    force_running_queries = []
    if debug_query is not None:
        queries = [q for q in queries if q.query == debug_query]
        force_running_queries = [debug_query]

    collected_paths = {}
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
            results_cache_dir=cache_dir,
            reset_db_before_each_query=True,
            force_running_queries=force_running_queries,
            # Answers are scored, never read cell by cell: the executor reduces each one
            # to its row signature and caches that. See Executor.execute_benchmark.
            reduce_results=True,
        )
        logger.info(
            "Results for %s, precision guarantee %s, recall guarantee %s",
            executor_name,
            prec,
            rec,
        )
        for query, answer in benchmark_result.results.items():
            logger.debug("Query: %s\n%s", query, answer)

            if query not in collected_paths:
                collected_paths[query] = {}
            if query not in collected_pipeline_tracks:
                collected_pipeline_tracks[query] = {}
            if query not in collected_costs:
                collected_costs[query] = {}

            collected_paths[query][prec, rec] = benchmark_result.result_paths[query]
            collected_pipeline_tracks[query][prec, rec] = (
                benchmark_result.tuned_pipelines.get(query, None)
            )
            collected_costs[query][prec, rec] = benchmark_result.costs.get(query, None)
    return collected_paths, collected_pipeline_tracks, collected_costs


def collect_result_no_guarantees(
    out_dir: Path,
    executor_name,
    executor,
    benchmark,
    debug_query: Optional[str],
    results_cache_dir: Optional[Path] = None,
):
    cache_dir = results_cache_dir if results_cache_dir is not None else out_dir / "cache"
    queries = benchmark.queries
    force_running_queries = []
    if debug_query is not None:
        queries = [q for q in queries if q.query == debug_query]
        force_running_queries = [debug_query]

    benchmark_result = executor.execute_benchmark(
        queries,
        results_cache_dir=cache_dir,
        reset_db_before_each_query=True,
        force_running_queries=force_running_queries,
        # As above: the row signature is the whole of what a label pass is read for.
        reduce_results=True,
    )
    logger.info("Results for %s", executor_name)
    collected_paths = {}
    collected_pipeline_tracks = {}
    collected_costs = {}
    for query, answer in benchmark_result.results.items():
        logger.debug("Query: %s\n%s", query, answer)

        collected_paths[query] = benchmark_result.result_paths[query]
        collected_pipeline_tracks[query] = benchmark_result.tuned_pipelines.get(
            query, None
        )
        collected_costs[query] = benchmark_result.costs.get(query, None)
    return collected_paths, collected_pipeline_tracks, collected_costs


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
        df = pd.DataFrame(columns=pd.Index(["count"]))
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


