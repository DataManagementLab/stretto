"""Record every operator response a later ``--simulate`` run will replay.

The shared precompute loop: create-or-resume a
:class:`~reasondb.backends.simulate_store.SimulateStore`, install it as the
process's precompute sink, run the perfect plan over every query, and save after each
one. Executor construction stays at the call site, because it differs per experiment:
the storage sweep needs a configurator covering every baseline it will ever request,
the sample-size sweep the plain default suite.

The pass needs the model servers running - recording real responses is the entire
point - and it is resumable: the store is saved after every query, so a crashed
run reloads what it had and skips the operators already covered.

It reports progress through two kinds of events: ``precompute_progress`` drives the
monitor's Precompute tab (store growth, save cost), and ``query_start``/``query_end``
drive everything that counts *queries* (per-worker and coordinator progress bars, ETA).

One pass covers a whole benchmark, deliberately. Precomputed work is keyed by
``(operator_id, expression, base_tables)`` rather than by query, and a
``RandomBenchmark`` draws its queries from one shared operator pool, so a single
pass already skips every expression it has seen. Splitting the query range across
processes gives each its own store and therefore no shared reuse - the overlap is
recomputed, with real model calls, once per split. Splitting by *modality* is the
split that pays; see ``scripts/merge_precompute.py``.

The store this fills is shared with the phase-0 filter-stats pass
(``reasondb.evaluation.filter_stats``), which runs first and writes the same file: it
pins the operator configuration and records the gold model's responses, so this pass
reuses both rather than re-deriving the question phrasing.
"""

import logging
from pathlib import Path
from typing import Optional

from reasondb.backends.simulate_store import SimulateStore
from reasondb.monitor import collector as monitor

logger = logging.getLogger(__name__)

#: The ``role`` every telemetry event of this pass carries. Distinct from ``Executor``'s
#: "sweep"/"label": this pass runs every candidate operator over the full table and
#: measures nothing, so its counts must not be pooled with a sweep's.
PRECOMPUTE_ROLE = "precompute"


def load_or_create_store(output_path: Path) -> SimulateStore:
    """The store this pass appends to, resuming from its own partial file if present."""
    if output_path.exists():
        store = SimulateStore.load(output_path)
        logger.info("[precompute] Resuming from %s (%s)", output_path, store.stats())
        return store
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return SimulateStore()


def run_precompute(
    benchmark,
    executor,
    output_path: Path,
    *,
    debug_query: Optional[str] = None,
    progress_label: str = "precompute",
) -> SimulateStore:
    """Run *executor*'s perfect plan over the benchmark, recording every response.

    :param progress_label: identifies this pass in the monitor's precompute tab -
        the executor name for a multi-executor run, the job id under the coordinator.
    :returns: the store, already saved to ``output_path``.
    """
    output_path = Path(output_path)
    store = load_or_create_store(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    SimulateStore.set_precompute(store)

    queries = list(benchmark.queries)
    if debug_query is not None:
        queries = [q for q in queries if q.query == debug_query]

    try:
        with executor as e:
            for i, query in enumerate(queries):
                monitor.record_query_start(
                    executor=progress_label,
                    role=PRECOMPUTE_ROLE,
                    query=query.query,
                    query_index=i,
                    n_queries=len(queries),
                )
                before = store.counts()
                e.precompute_query(query, i)
                store.save(output_path)
                counts = store.counts()
                monitor.record_precompute_progress(
                    executor=progress_label,
                    query_index=i,
                    n_queries=len(queries),
                    **counts,
                )
                monitor.record_query_end(
                    executor=progress_label,
                    role=PRECOMPUTE_ROLE,
                    query=query.query,
                    query_index=i,
                    # A query whose operators were all recorded already adds nothing to
                    # the store and costs no inference: the precompute analogue of a
                    # cache hit.
                    cached=counts == before,
                    # No cost object: this pass does not tune or measure a plan. The key
                    # is kept because the event schema requires it.
                    component_times={},
                )
                logger.info(
                    "[%s] [%d/%d] %s",
                    progress_label,
                    i + 1,
                    len(queries),
                    store.stats(),
                )
    finally:
        SimulateStore.set_precompute(None)
    logger.info(
        "[%s] Done. Saved %s to %s", progress_label, store.stats(), output_path
    )
    return store
