"""Event vocabulary for the run monitor.

Events are plain dicts, not dataclasses, for one reason: the producer side runs next to
LLM inference and must stay allocation-cheap and serialization-free. A dict literal is
built at the call site, handed to the collector, and only ever touched again on the
background drain thread.

Every event the collector accepts carries, once the collector has stamped it:

``seq``    monotonically increasing integer, the cursor the UI pages with
``type``   one of :data:`EVENT_TYPES`
``t``      ``time.time()`` at the moment the producer called ``record_*``
``data``   the type-specific payload described below

Any event forwarded through a coordinator (see ``reasondb.coordinator.ingest``) also
carries ``worker_id``/``job_id`` in ``data`` - a single-process run never sets these,
so they're absent (not present as ``None``) there.

Payload shapes (documented, not enforced beyond the type name - a stray key costs
nothing and a missing one renders as blank in the UI rather than breaking a run):

``run_start``           script, argv, output_dir, benchmarks, split, mode, pid
``run_plan``            total_queries, total_iterations, dimensions - what the whole
                        sweep is going to run, announced once up front by the
                        ``run_benchmark*`` scripts so the dashboard's progress bar can
                        span every iteration instead of restarting each one. Absent for
                        a script that cannot know its total; the coordinator ignores it
                        entirely and uses its job queue (which is exact by construction).
``run_end``             status ("ok" | "error" | "interrupted"), error
``benchmark_start``     benchmark, split, n_queries
``executor_start``      executor, benchmark, precision_guarantee, recall_guarantee
``query_start``         executor, benchmark, query, query_index, n_queries
``query_end``           executor, query, query_index, cached, n_rows, component_times,
                        execution/tuning cost, tuned_pipeline (which physical plan was
                        picked - see reasondb.executor.Executor.execute_benchmark)
``query_metrics``       benchmark, executor, query, precision/recall guarantee, the
                        *achieved* precision/recall/F1 plus output cardinalities, and
                        ``labels`` - which label set scored it ("silver"/"gold"), since
                        both are scored and the two are otherwise indistinguishable
                        rows for the same (query, guarantee). Emitted from
                        ``reasondb.evaluation.evaluation.evaluate``, which is the only
                        place these exist - accuracy is scored against labels, so
                        nothing during execution can report it. Under the coordinator
                        it arrives per job, as soon as that job's labels land (see
                        ``reasondb.coordinator.scoring``), carrying its own job_id.
``search_space``        query, step_index, logical_type/expression, and one entry per
                        candidate operator the optimizer may choose between for that
                        logical step: operator id, quality, fake cost, compression
                        variant, plus which entry is ``gold`` (the highest-quality one,
                        from which the profiler derives that step's labels) and which
                        the configuring LLM guessed was best. Emitted once per logical
                        step per query, from ``PlanConfigurator.llm_configure``.
``phase``               name, seconds, parent (one per ``reasondb.utils.timing.measure``
                        span). ``parent`` is the enclosing span, or None at the top
                        level: spans nest (``end_to_end`` > ``tuning`` > ``profiling``),
                        so ``seconds`` is inclusive and only the parent link makes an
                        un-double-counted breakdown possible.
``operator_run``        operator, operation_class, n_input_rows, seconds,
                        phase (the enclosing ``measure()`` span - "execution" or
                        "profiling"; the same operator code path serves both),
                        step_expression (which plan step made the call, since one plan
                        can run the same operator at several positions - see
                        physical_operator._step_expression), cost fields, and - only
                        for operators backed by a KV-compressed model - model_name,
                        effective_compression_ratio,
                        materialized_compression_ratio, vanilla (see
                        physical_operator._extract_cr_info)
``optimizer_solve``     one GD solve's outcome, from
                        ``GradientDescentOptimizer.post_optimization_check``, and the
                        substance of the Optimizer tab. How large the discrete space was
                        (``n_pick_params`` -- the learnable subset-selection
                        coordinates, so the space is 2**that), how many of the parallel
                        restarts ended feasible (``n_feasible`` of
                        ``n_initializations``), how many *different* plans they resolved
                        to (``n_distinct_plans``, and ``n_distinct_plans_feasible`` over
                        just the eligible ones), and which restart won
                        (``winner_init_index``, ``winner_init_kind``,
                        ``winner_proxies_per_step``).
                        ``winner_init_index`` doubles as the violation-penalty level,
                        because the restart axis and the penalty sweep are the same axis
                        (see ``OptimizationConfig.violation_loss_first/last_initialization``,
                        recorded here as ``violation_first``/``violation_last`` so the
                        actual lambda is reconstructible offline).
                        ``n_slots_by_kind`` is what makes a win rate readable: a seed
                        kind holding 3/8 of the slots wins 3/8 of the time by chance, so
                        the number worth reading is win share over slot share.
                        ``meets_targets`` / ``ended`` / ``attempt`` separate the three
                        distinct failures -- targets unreachable, step budget exhausted,
                        and retried at a higher penalty ceiling.
                        Emitted once per solve, not once per check, so failing and
                        succeeding solves are counted equally.
``kv_inference``        the full ``reasondb.backends.inference_stats`` payload
``precompute_progress`` query_index, n_queries, n_text_qa, n_vision, n_ops, n_configs
``precompute_save``     path, seconds, size_bytes
``job_spec``            job_id, spec
``error``               where, message
"""

from typing import Any, Dict, List

EV_RUN_START = "run_start"
EV_RUN_PLAN = "run_plan"
EV_RUN_END = "run_end"
EV_JOB_SPEC = "job_spec"
EV_BENCHMARK_START = "benchmark_start"
EV_EXECUTOR_START = "executor_start"
EV_QUERY_START = "query_start"
EV_QUERY_END = "query_end"
EV_QUERY_METRICS = "query_metrics"
EV_SEARCH_SPACE = "search_space"
EV_PHASE = "phase"
EV_OPERATOR_RUN = "operator_run"
EV_OPTIMIZER_SOLVE = "optimizer_solve"
EV_KV_INFERENCE = "kv_inference"
EV_PRECOMPUTE_PROGRESS = "precompute_progress"
EV_PRECOMPUTE_SAVE = "precompute_save"
EV_ERROR = "error"

EVENT_TYPES = frozenset(
    {
        EV_RUN_START,
        EV_RUN_PLAN,
        EV_RUN_END,
        EV_JOB_SPEC,
        EV_BENCHMARK_START,
        EV_EXECUTOR_START,
        EV_QUERY_START,
        EV_QUERY_END,
        EV_QUERY_METRICS,
        EV_SEARCH_SPACE,
        EV_PHASE,
        EV_OPERATOR_RUN,
        EV_OPTIMIZER_SOLVE,
        EV_KV_INFERENCE,
        EV_PRECOMPUTE_PROGRESS,
        EV_PRECOMPUTE_SAVE,
        EV_ERROR,
    }
)

# Types the UI's live event log shows by default. High-volume per-call types are opt-in
# so the log stays readable during a long run.
DEFAULT_LOG_TYPES = (
    EV_RUN_START,
    EV_RUN_PLAN,
    EV_RUN_END,
    EV_BENCHMARK_START,
    EV_EXECUTOR_START,
    EV_QUERY_START,
    EV_QUERY_END,
    EV_PRECOMPUTE_PROGRESS,
    EV_PRECOMPUTE_SAVE,
    EV_ERROR,
)


#: The payload keys the collector actually *reads* to build its aggregates, per event
#: type. Deliberately not the full documented payload: a key is listed here only if a
#: number on the dashboard is wrong without it.
#:
#: Unknown keys are never rejected, because payloads are dicts precisely so they can be
#: extended, and a coordinator ingests events from workers that may be on a different
#: revision. A *missing* required key, however, is counted and named rather than
#: rendering as a silent blank.
#:
#: Notably absent for `operator_run`: model_name / effective_compression_ratio /
#: materialized_compression_ratio / vanilla. `_extract_cr_info` omits those by design
#: for an operator with no KV backend (TraditionalFilter, PythonExtract), so requiring
#: them would flag correct behaviour - see tests/test_monitor_operator_cr.py.
EVENT_SCHEMA: Dict[str, frozenset] = {
    EV_EXECUTOR_START: frozenset({"executor", "role"}),
    EV_QUERY_START: frozenset({"executor", "role", "query", "query_index"}),
    EV_QUERY_END: frozenset({"executor", "role", "query", "cached", "component_times"}),
    EV_BENCHMARK_START: frozenset({"benchmark", "split"}),
    EV_JOB_SPEC: frozenset({"job_id", "spec"}),
    EV_OPERATOR_RUN: frozenset(
        {"operator", "operation_class", "seconds", "n_input_rows", "phase"}
    ),
    EV_PHASE: frozenset({"name", "seconds"}),
    EV_QUERY_METRICS: frozenset({"benchmark", "executor", "query"}),
    EV_SEARCH_SPACE: frozenset({"query", "step_index", "candidates"}),
    EV_OPTIMIZER_SOLVE: frozenset(
        {
            "n_initializations",
            "n_feasible",
            "winner_init_index",
            "winner_init_kind",
            "n_slots_by_kind",
            "n_pick_params",
            "meets_targets",
        }
    ),
}


def missing_required_keys(event_type: str, data: Dict[str, Any]) -> List[str]:
    """Required keys this payload does not carry. Empty for an unknown event type."""
    required = EVENT_SCHEMA.get(event_type)
    if not required:
        return []
    return sorted(k for k in required if k not in data)


def is_event_type(name: str) -> bool:
    return name in EVENT_TYPES


def make_event(seq: int, event_type: str, t: float, data: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp a payload into the canonical envelope. Called on the drain thread only."""
    return {"seq": seq, "type": event_type, "t": t, "data": data}
