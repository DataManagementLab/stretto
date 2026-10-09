"""Compute the predicate/tuple overlap matrix a ``RandomBenchmark`` samples queries from.

Every filter option in the pool is run as a single-filter query over its base table with
the gold (vanilla 70B) model, and cell ``(p, t)`` records whether that model kept tuple
``t`` for predicate ``p``. ``FilterStats.sample_overlapping`` then draws only combinations
whose conjunction covers at least one tuple, which is what stops the generator from
emitting queries that return nothing.

The pass must run with a :class:`~reasondb.backends.simulate_store.SimulateStore`
installed. With a precompute store live, ``_pin_operator_config`` pins the question
phrasing the LLM derives for each expression, and the text/vision backends record every
gold response and runtime into the *same file* the matrix is then written to. A later
``--precompute`` or ``--simulate`` run reuses those pins, so it asks the identical
question about the identical row, and the matrix predicts the real conjunction.

This is not built on :func:`reasondb.evaluation.precompute.run_precompute`, which runs
every operator variant over all rows to fill a replay cache; this is one gold-model pass
whose output is a matrix. The backends record whenever a precompute store is installed.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Type, cast

import numpy as np

from reasondb.backends.simulate_store import SimulateStore
from reasondb.evaluation.benchmark import Benchmark, FilterStats, RandomBenchmark
from reasondb.evaluation.precompute import load_or_create_store
from reasondb.executor import Executor
from reasondb.interface.config import get_default_configurator
from reasondb.operators.filter.image_embed_filter import ImageSimilarityFilter
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.optimizer.label_optimizer import LabelOptimizer
from reasondb.query_plan.logical_plan import ALL_LOGICAL_OPERATORS_TOOLBOX
from reasondb.query_plan.physical_operator import PhysicalOperatorToolbox
from reasondb.reasoning.few_shot_database import DUMMY_FEW_SHOT_DATABASE
from reasondb.reasoning.llm import GPT4o
from reasondb.reasoning.reasoners.self_correction import SelfCorrectionReasoner

logger = logging.getLogger(__name__)

#: Filename under ``benchmark_results/filter_stats/<benchmark>/<split>/``. The store is
#: authoritative; this copy exists to be looked at, and for entry points that install no
#: store (demos, the single-operator studies).
STATS_FILENAME = "stats.json"


#: Operators whose ``prepare()`` computes embeddings locally against the similarity
#: server, rather than replaying anything. ``LabelOptimizer`` always executes the
#: highest-quality candidate, so a label pass never runs these, but
#: ``PlanConfigurator.prepare`` prepares every operator in the suite. Leaving them out of
#: a replay-only run drops the server dependency and the embedding cost.
_NEEDS_LIVE_EMBEDDING_SERVER = (ImageSimilarityFilter,)

#: The toolbox lists ``PhysicalOperatorToolbox`` keeps, so a filtered copy can be
#: rebuilt without knowing which list an operator landed in.
_TOOLBOX_GROUPS = (
    "join_operators", "join_predicates", "filter_operators", "extract_operators",
    "transform_operators", "limit_operators", "project_operators", "sorting_operators",
    "groupby_operators", "aggregate_operators", "rename_operators",
)


def without_live_embedding_operators(
    configurator: PlanConfigurator,
) -> PlanConfigurator:
    """A copy of *configurator* that contacts no embedding server when it prepares."""
    toolbox = configurator.physical_operators
    return PlanConfigurator(
        llm=configurator.llm,
        physical_operators=PhysicalOperatorToolbox(
            **{
                group: [
                    op
                    for op in getattr(toolbox, group)
                    if not isinstance(op, _NEEDS_LIVE_EMBEDDING_SERVER)
                ]
                for group in _TOOLBOX_GROUPS
            }
        ),
    )


def build_filter_stats_executor(
    benchmark: Benchmark, *, replay_only: bool = False
) -> Executor:
    """The gold labeler: ``LabelOptimizer`` always picks the last *executable* operator
    candidate, which ``PlanConfigurator`` sorts to be the highest-quality one - the
    vanilla 70B.

    It builds ``get_default_configurator()`` without ``use_human_labels``: this pass
    produces model-derived labels, so using human labels here would be circular.

    ``replay_only`` drops the operators that would prepare against a live embedding
    server, for a pass whose every model call is a store lookup.
    """
    configurator = get_default_configurator()
    if replay_only:
        configurator = without_live_embedding_operators(configurator)
    reasoner = SelfCorrectionReasoner(
        llm=GPT4o(),
        configurator=configurator,
        logical_operators=ALL_LOGICAL_OPERATORS_TOOLBOX,
        few_shot_database=DUMMY_FEW_SHOT_DATABASE,
    )
    return Executor(
        name="silver",
        database=benchmark.database,
        reasoner=reasoner,
        optimizer=LabelOptimizer(),
        configurator=configurator,
    )


def _gold_operators(tuned_pipelines: Dict[str, Any]) -> Sequence[str]:
    """Operator identifiers the pass actually ran, for the payload's provenance.

    Best-effort: the pipelines are JSON strings whose shape is an executor implementation
    detail, and a matrix is still valid if we cannot name the operator that built it.
    """
    found = set()
    for pipelines in tuned_pipelines.values():
        for pipeline in pipelines or []:
            try:
                for step in json.loads(pipeline):
                    if isinstance(step, dict) and "operator" in step:
                        found.add(step["operator"])
            except (ValueError, TypeError):
                continue
    return sorted(found)


def _base_table_hashes(benchmark: Benchmark) -> Dict[str, str]:
    """Hash each base table file, so a changed CSV invalidates the matrix loudly.

    The matrix's columns are row ids of these tables. Editing one under
    ``evaluation/benchmarks/files/`` renumbers the rows and makes every cell refer to a
    different tuple.
    """
    hashes = {}
    for table in getattr(benchmark.database, "external_tables", []):
        path = Path(table.path)
        if path.exists():
            hashes[table.name] = Benchmark._get_hash_of_file(path)
    return hashes


def _json_row_id(index_entry: Any) -> Any:
    """One result-row identifier, in a form ``json.dump`` accepts.

    A materialized result is indexed by one ``_index_<table>`` column per base table, so
    pandas hands back a tuple per row even when there is only one table, and the values
    are numpy integers. Both have to be flattened to plain Python for the payload; a
    tuple becomes a list, which is what it reads back as.
    """
    if isinstance(index_entry, tuple):
        return [int(part) for part in index_entry]
    return int(index_entry)


def compute_filter_stats(
    benchmark: RandomBenchmark,
    executor: Executor,
    *,
    results_cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run every pool filter as a single-filter query and build the payload.

    ``results_cache_dir`` is the ordinary executor cache. Pass ``None`` when the point is
    to *record* into a store: a cache hit returns the previous run's rows without calling
    a model, so nothing would be recorded.
    """
    plan = benchmark.single_filter_plan()
    queries = benchmark.single_filter_queries

    with executor as e:
        benchmark_result = e.execute_benchmark(
            queries,
            results_cache_dir=results_cache_dir,
            reset_db_before_each_query=True,
        )

    results = list(benchmark_result.results.items())
    assert len(results) == len(plan), (
        f"{len(plan)} single-filter queries were planned but {len(results)} results came "
        "back; results are keyed by query text, so two pool options sharing an "
        "expression would collapse into one."
    )

    per_key: Dict[str, Dict[str, Any]] = {}
    for (key, option, _query), (_query_str, df) in zip(plan, results):
        bucket = per_key.setdefault(key, {"options": [], "indices": []})
        bucket["options"].append(option.expression)
        bucket["indices"].append(list(df.index.values))

    keys_payload = {}
    for key, bucket in per_key.items():
        expressions = bucket["options"]
        assert len(set(expressions)) == len(expressions), (
            f"pool key {key!r} has duplicate filter expressions; the matrix is keyed by "
            "expression, so duplicates would silently share a row."
        )
        row_ids = sorted({idx for indices in bucket["indices"] for idx in indices})
        row_to_column = {idx: i for i, idx in enumerate(row_ids)}
        matrix = np.zeros((len(expressions), len(row_ids)), dtype=int)
        for i, indices in enumerate(bucket["indices"]):
            for idx in indices:
                matrix[i, row_to_column[idx]] = 1
            logger.info(
                "[filter-stats] %s/%s: %d rows kept by %s",
                benchmark.name(), key or "-", len(indices), expressions[i],
            )
        keys_payload[key] = {
            "overlap_matrix": matrix.tolist(),
            "predicate_to_matrix_id": {expr: i for i, expr in enumerate(expressions)},
            "matrix_id_to_predicate": {str(i): expr for i, expr in enumerate(expressions)},
            "row_ids": [_json_row_id(idx) for idx in row_ids],
        }

    return {
        "keys": keys_payload,
        "gold_operators": _gold_operators(benchmark_result.tuned_pipelines),
        "base_table_hashes": _base_table_hashes(benchmark),
        "generated_at": time.time(),
    }


def derive_filter_stats_from_recordings(
    benchmark_cls: Type[RandomBenchmark],
    split: Literal["train", "dev", "test"],
    store_path: Path,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Build a benchmark's filter stats out of answers a ``--precompute`` file holds.

    The single-filter queries run exactly as they would against live models, except the
    store is installed as the *replay* store: every model call becomes a lookup in the
    file, so the operators apply their own thresholds and answer conversion and no model
    is contacted. An answer the file does not hold raises rather than falling back to a
    model.

    Writes the matrix back into ``store_path``, so the coordinator finds it there and
    enumerates no filter-stats job at all.

    Servers: none for a text-only benchmark. A benchmark with image columns needs the two
    embedding servers up, because ``ImageEmbedFilter.prepare`` embeds its images locally
    rather than replaying them. Needs ``OPENAI_API_KEY`` either way: the configurator asks
    GPT-4o for each step's operator definitions before the recorded pins overwrite their
    parameters.
    """
    store_path = Path(store_path)
    store = SimulateStore.load(store_path)
    name = benchmark_cls.name()

    existing = store.get_filter_stats(name, split)
    if existing is not None and not force:
        logger.info(
            "[filter-stats] %s/%s already has stats in %s; nothing to do.",
            name, split, store_path,
        )
        return existing

    SimulateStore.set_simulate(store)
    try:
        benchmark = cast(RandomBenchmark, benchmark_cls.load_without_queries(split))
        payload = compute_filter_stats(
            benchmark, build_filter_stats_executor(benchmark, replay_only=True)
        )
    finally:
        SimulateStore.set_simulate(None)

    payload["source"] = "recordings"
    payload["covered_filters"] = {
        key: sorted(stats["predicate_to_matrix_id"])
        for key, stats in payload["keys"].items()
    }
    store.record_filter_stats(name, split, payload)
    store.save(store_path)
    logger.info(
        "[filter-stats] %s/%s: derived %d filters from recordings, written to %s",
        name, split,
        sum(len(v) for v in payload["covered_filters"].values()),
        store_path,
    )
    return payload


def write_stats_file(stats_dir: Path, payload: Dict[str, Any]) -> Path:
    """The inspection copy. See :data:`STATS_FILENAME` for why it is not the source."""
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / STATS_FILENAME
    with path.open("w") as f:
        json.dump(payload, f)
    return path


def run_filter_stats_pass(
    benchmark_cls: Type[RandomBenchmark],
    split: Literal["train", "dev", "test"],
    *,
    simulate: bool,
    store_path: Optional[Path],
    stats_dir: Path,
    results_cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Produce (or replay) one benchmark's filter stats, then pin its query set.

    Three artifacts are written in order: first the store (the matrix together with the
    prompts and responses that produced it, in a single ``save``), then ``stats.json``
    and ``queries.json``, which are derived and can be regenerated from it.

    Under ``simulate`` nothing is executed: the matrix is read back from the store, so a
    simulate worker needs neither model servers nor an ``OPENAI_API_KEY``. A store
    without the bucket is an error.
    """
    store = SimulateStore.get_simulate() if simulate else None
    if simulate:
        assert store is not None, (
            "Simulate mode: no store installed. run_filter_stats_pass replays the matrix "
            "from the store and cannot compute one without model servers."
        )
        payload = store.get_filter_stats(benchmark_cls.name(), split)
        if payload is None:
            raise RuntimeError(
                f"Simulate mode: no filter stats recorded for {benchmark_cls.name()}/"
                f"{split}. This store predates the filter_stats bucket (or was recorded "
                "for another dataset); re-run --precompute for this benchmark."
            )
        logger.info(
            "[filter-stats] %s/%s: replaying the recorded matrix, nothing to execute.",
            benchmark_cls.name(), split,
        )
    else:
        assert store_path is not None, (
            "A recording pass needs the dataset's precompute path: the matrix and the "
            "responses it is built from are saved together."
        )
        store = load_or_create_store(Path(store_path))
        SimulateStore.set_precompute(store)
        try:
            benchmark = cast(
                RandomBenchmark, benchmark_cls.load_without_queries(split)
            )
            payload = compute_filter_stats(
                benchmark,
                build_filter_stats_executor(benchmark),
                results_cache_dir=results_cache_dir,
            )
            store.record_filter_stats(benchmark_cls.name(), split, payload)
            store.save(Path(store_path))
        finally:
            SimulateStore.set_precompute(None)
        logger.info(
            "[filter-stats] %s/%s: recorded into %s (%s)",
            benchmark_cls.name(), split, store_path, store.stats(),
        )

    write_stats_file(Path(stats_dir), payload)

    queries = benchmark_cls.generate_random_queries(
        split,
        num_queries_per_shape=benchmark_cls.num_queries_per_shape,
        keep_per_shape=benchmark_cls.queries_kept_per_shape,
        filter_stats=FilterStats.from_payload(payload),
    )
    queries.dump(benchmark_cls.benchmark_dir() / split)
    logger.info(
        "[filter-stats] %s/%s: pinned %d queries to %s",
        benchmark_cls.name(), split, len(queries),
        benchmark_cls.benchmark_dir() / split,
    )
    return {"n_queries": len(queries), **_payload_counts(payload)}


def pin_query_set(
    benchmark_cls: Type[RandomBenchmark],
    split: Literal["train", "dev", "test"],
    payload: Dict[str, Any],
) -> int:
    """Draw the query set from *payload* and write it to ``queries.json``.

    Every later job reads that file rather than sampling its own, so this is what makes
    one query set authoritative for a whole experiment. Returns how many were written.
    """
    queries = benchmark_cls.generate_random_queries(
        split,
        num_queries_per_shape=benchmark_cls.num_queries_per_shape,
        keep_per_shape=benchmark_cls.queries_kept_per_shape,
        filter_stats=FilterStats.from_payload(payload),
    )
    queries.dump(benchmark_cls.benchmark_dir() / split)
    return len(queries)


def _payload_counts(payload: Dict[str, Any]) -> Dict[str, int]:
    keys = payload.get("keys", {})
    return {
        "n_pool_keys": len(keys),
        "n_predicates": sum(len(k.get("overlap_matrix", [])) for k in keys.values()),
    }
