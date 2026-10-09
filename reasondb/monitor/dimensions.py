"""What each group-by / filter dimension is called in the dashboard.

The frontend derives its dimensions from the data - a coordinator job's ``spec`` becomes
groupable simply by being a flat scalar - which is what keeps a new producer flag
working with no UI change. An unlabelled key gets an auto-prettified version of its
own name, so ``sweep_to_gold`` reads as "Sweep To Gold", and two different keys could
prettify to the *same* string (e.g. ``executor`` vs. ``approach``, ``labels`` vs.
``label_set``). Explicit labels keep them distinct.

This module is the single table, served to the browser through ``/api/presentation``.
``tests/test_monitor_dimension_labels.py`` asserts every spec key a producer can emit
appears here, so a new flag fails a test instead of shipping an unlabelled chip.

Standard library only, for the same reason as ``phases.py``: ``collector.py`` imports
it and that module loads in every process of a benchmark.
"""

from typing import Dict, FrozenSet, Tuple

#: The query *shape* statistics a ``QueryShape.additional_info`` may declare, as the
#: columns they become on a metrics or sweep CSV.
#:
#: A single tuple shared by every consumer (``monitor.results``'s facet columns, the
#: sweep merge's stamp, the plotting registry), kept next to ``DIMENSION_LABELS`` so the
#: two cannot drift apart.
QUERY_STAT_COLUMNS: Tuple[str, ...] = (
    "num_semops",
    "num_sem_filter",
    "num_sem_extract",
    "num_sem_join",
    "num_tradops",
)

#: Dimension name -> the string the UI shows.
DIMENSION_LABELS: Dict[str, str] = {
    # Stamped on every record by the collector (CONFIG_DIMENSIONS).
    # "Run", not "Run id": a dashboard seeded from earlier sidecars shows several runs
    # of one output directory at once, and this is the chip that separates them.
    "run_id": "Run",
    "worker_id": "Worker",
    "job_id": "Job",
    "benchmark": "Benchmark",
    "split": "Split",
    # "Executor", not "Approach": `approach` is a separate spec key with its own values.
    "executor": "Executor",
    "role": "Pass",
    "precision": "Precision target",
    "recall": "Recall target",
    # Per-record.
    "query": "Query",
    "winner_init_kind": "Winning seed",
    "n_pick_params": "Pick coordinates",
    "used_method": "Optimization mode",
    "query_index": "Query #",
    # Query *shape* statistics, from the benchmark's QueryShape.additional_info. These
    # describe how hard a query is rather than how it was run.
    "num_semops": "Semantic operators",
    "num_sem_filter": "Semantic filters",
    "num_sem_extract": "Semantic extracts",
    "num_sem_join": "Semantic joins",
    "num_tradops": "Traditional operators",
    "phase": "Phase",
    "operator": "Operator",
    "operation_class": "Operator class",
    "model_name": "Model",
    "cr_label": "Compression",
    "cached": "Cached",
    # Which label set an accuracy row was scored against - not the job-spec key below.
    "labels": "Scored against",
    # Coordinator job/worker records.
    "state": "State",
    "producer": "Producer",
    # ── Job-spec keys, flattened onto records by facets.withJobDimensions ──
    "kind": "Job kind",
    "name": "Job target",
    "approach": "Optimizer",
    "label_set": "Label set (job)",
    "step_idx": "Sweep state",
    "state_plan": "State plan",
    "tune_parameters": "Parameter tuning",
    "sample_size": "Sample size",
    "adaptive_sampling": "Adaptive sampling",
    "reorder": "Operator reordering",
    "pruned": "Pruned",
    "protected": "Never prunable",
    "sweep_to_gold": "Sweeps to gold",
    # Precompute jobs only: which operators the recording covers. "State plan" above is
    # a different question on the same vocabulary - which states a *sweep* visits - so
    # the two need names that cannot be read for each other.
    "precompute_states": "Precompute coverage",
    # Which modality a precompute job records: one of them under
    # --split-both-capability-datasets, "all" (spec value None) for an ordinary pass.
    "precompute_modality": "Precompute modality",
    "use_indexes": "Index use",
    "human_labels": "Human labels",
    # `label_reference`'s two arms: which reference the optimizer tuned against. Named
    # for the question rather than "Arm", which says nothing on its own - and the values
    # ("model"/"human") are spelled out below for the same reason.
    "arm": "Optimized against",
    "press_name": "KV press",
    "cost_type": "Cost model",
    "simulate": "Simulated",
    "shard_index": "Shard",
    "n_shards": "Shards",
    "text_small_model": "Text model (small)",
    "text_large_model": "Text model (large)",
    "image_small_model": "Image model (small)",
    "image_large_model": "Image model (large)",
}

#: Dimension name -> {raw value: label}, for keys whose values are opaque on their own.
#:
#: ``kind`` is the case that needs it: each producer uses its own vocabulary for what a
#: job *is*, and the raw values ("step", "point", "approach") say little in a chip.
DIMENSION_VALUE_LABELS: Dict[str, Dict[str, str]] = {
    "kind": {
        "approach": "Approach run",
        "step": "Sweep state",
        "point": "Sample point",
        "label": "Label pass",
        "precompute": "Precompute",
    },
    "state_plan": {
        "greedy": "Greedy storage walk",
        "greedy_to_gold": "Greedy walk to gold",
        "default": "Default operator suite",
        "ablation": "Default suite vs vanilla only",
        "gold": "Gold operator only",
        "full": "Every materialized operator",
        "kv_operator": "Gold plus one KV operator",
        "kv_operator_pairs": "Gold plus one KV operator per modality",
        "kv_operator_marginal": "Vanilla suite plus one KV operator per modality",
    },
    "arm": {
        "model": "The best model's verdicts",
        "human": "Per-tuple ground truth",
    },
    #: Optimizer names, so a chip reads as the thing rather than as the CLI token.
    #: Every entry of ``kv_experiment_utils.APPROACHES`` is here and
    #: ``tests/test_monitor_dimension_labels.py`` asserts it stays that way.
    #:
    #: Prose, not `LABEL_MAP`'s strings: those carry newlines placed for a figure's tick
    #: geometry ("Stretto\n-independent"), which would break a chip mid-word.
    "approach": {
        "optim_global": "Stretto",
        "optim_local": "Stretto (per step)",
        "optim_shift_budget": "Stretto (shifted budget)",
        "lotus": "Lotus",
        "abacus": "Abacus",
        "no_optim": "No optimization",
        "no_optim_reorder": "Reordering only",
    },
    "role": {
        "sweep": "Sweep",
        "label": "Label pass",
        # Not an ``Executor.role`` - see PRECOMPUTE_ROLE in evaluation/precompute.py.
        "precompute": "Precompute pass",
    },
}

#: Job-spec keys that must never become dimensions.
#:
#: Not because they are unlabelled, but because they are not *knobs*: a filesystem path
#: or a query count partitions a chart into one bucket per job and says nothing. Paths
#: are the important case - ``precompute_path`` is an absolute path, so without this it
#: renders as a group-by chip listing directories.
SPEC_KEYS_NOT_DIMENSIONS: FrozenSet[str] = frozenset(
    {
        "simulate_path",
        "simulate_paths",
        "precompute_path",
        # The mapped file a per-modality half is merged back into: a path, and the same
        # one for both halves, so it groups nothing `precompute_modality` does not group
        # better. `precompute_split_parts` is bookkeeping for that merge (how many halves
        # to wait for), not a knob anybody set.
        "precompute_base_path",
        "precompute_split_parts",
        "seed_path",
        "output_path",
        "debug_query",
        "guarantee",
        "n_queries",
        # Like `guarantee`: a list-valued key that the run-configuration panel reads
        # specially, so it is listed explicitly.
        "operator_set",
    }
)


def presentation_payload() -> Dict[str, object]:
    """The dimension vocabulary, as the browser receives it from ``/api/presentation``."""
    return {
        "dimension_labels": dict(DIMENSION_LABELS),
        "dimension_value_labels": {k: dict(v) for k, v in DIMENSION_VALUE_LABELS.items()},
        "spec_keys_not_dimensions": sorted(SPEC_KEYS_NOT_DIMENSIONS),
    }
