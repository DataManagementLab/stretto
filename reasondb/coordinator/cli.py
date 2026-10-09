"""The coordinator's command line, as a module rather than a script.

``scripts/run_coordinator.py`` is the CLI; this is the parser behind it. Split out for
one reason: anything that wants to know what a set of coordinator flags *resolves to* -
``scripts/generate_experiment_report.py``, which turns every experiment in
``scripts/cluster.yaml`` into a table of what is fixed and what is swept - has to
parse those flags the same way the coordinator does, and a script may not import another
script. Shared code lives in the package.

That also makes the guarantee mechanical rather than remembered: the report describes the
flags the coordinator will really accept, because it is handed the same parser, and a new
flag reaches both the moment it is added here.
"""

import argparse
from pathlib import Path

import torch

from reasondb.coordinator.producers import PRODUCERS
from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS
from reasondb.evaluation.kv_experiment_utils import APPROACHES
from reasondb.evaluation.parameter_sweep import PRECOMPUTE_COVERAGE, STATE_PLANS
from reasondb.utils.benchmark_args import (
    add_benchmarks_argument,
    add_cost_type_argument,
    add_debug_query_argument,
    add_executor_selection_arguments,
    add_guarantee_arguments,
    add_human_labels_argument,
    add_labels_argument,
    add_precompute_simulate_arguments,
    add_split_argument,
    add_use_indexes_argument,
)
from reasondb.utils.benchmark_args import DEFAULT_BENCHMARKS  # re-exported: see its docstring

__all__ = ["DEFAULT_BENCHMARKS", "build_parser"]


def _all_benchmark_choices() -> list:
    """Every benchmark any producer could load.

    The registries differ: the KV sweeps only support the ``RandomBenchmark``
    variants, while ``run_benchmark`` also runs the fixed ones (RealEstate etc.).
    ``--benchmarks`` accepts the union; a name valid here but not in the SPECIFIC
    producer you picked still fails, loudly, inside that producer's
    ``enumerate_jobs`` - not worth producer-conditional argparse choices for.
    """
    return sorted(ALL_BENCHMARKS.keys())





def _add_shared_sweep_args(parser: argparse.ArgumentParser) -> None:
    """Flags every producer's enumerate_jobs/run_job reads off args, regardless of
    which one is selected."""
    add_benchmarks_argument(parser, _all_benchmark_choices(), DEFAULT_BENCHMARKS)
    add_split_argument(parser)
    add_use_indexes_argument(parser)
    add_human_labels_argument(parser)
    add_guarantee_arguments(parser)
    add_cost_type_argument(parser)
    add_debug_query_argument(parser)
    add_precompute_simulate_arguments(parser)
    parser.add_argument(
        "--split-both-capability-datasets",
        action="store_true",
        help=(
            "Record a mixed-modality dataset (ecommerce is the only one in the suite) as "
            "one --precompute job per modality instead of one job needing every KV "
            "server at once, so a --capability text and a --capability image node can "
            "record it side by side. Each half writes a sibling of the mapped file "
            "(ecomm.text.json, ecomm.image.json) because a store rewrites its whole file "
            "after every query and two writers on one path race; the halves are merged "
            "back into the mapped file once the task's jobs are all terminal. Datasets "
            "with a single modality are unaffected. Only meaningful with --precompute."
        ),
    )


def _add_parameter_sweep_args(parser: argparse.ArgumentParser) -> None:
    """parameter_sweep-only flags - see reasondb.evaluation.parameter_sweep."""
    parser.add_argument("--text-small-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--text-large-model", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--image-small-model", type=str, default="llava-hf/llama3-llava-next-8b-hf")
    parser.add_argument("--image-large-model", type=str, default="llava-hf/llava-next-72b-hf")
    parser.add_argument("--press-name", type=str, default="expected_attention")
    parser.add_argument(
        "--state-plan",
        type=str,
        choices=sorted(STATE_PLANS),
        default=None,
        help=(
            "Which operator sets the sweep visits: 'greedy' walks from every "
            "materialized level down to one per slot, 'greedy_to_gold' one step further, "
            "'default' is the single deployed suite, 'ablation' that suite then the "
            "vanilla-only state, 'gold' that vanilla-only state on its own with one "
            "operator per modality (what --producer reorder_only needs), 'full' is "
            "the walk's widest state on its own - every "
            "operator on disk, at one point instead of a curve over it - and "
            "'kv_operator' is that vanilla-only state followed by one state per "
            "materialized level, each holding the gold model and that level alone; "
            "'kv_operator_pairs' is the same with one level per modality per state, "
            "matched by model size and compression rank (single-modality benchmarks get "
            "exactly kv_operator's states), and 'kv_operator_marginal' visits those states "
            "keeping the small models' vanilla operators throughout, so a state adds a KV "
            "operator to the reference suite instead of replacing its small model. "
            "Omitted keeps the "
            "default behaviour, where --sweep-to-gold is the only thing that changes "
            "which states exist. Every wrapper producer pins this except --producer "
            "baselines, which defaults it to 'default' and takes any single-state plan; "
            "use --producer operator_count to sweep the axis rather than name a point on "
            "it."
        ),
    )
    parser.add_argument(
        "--sweep-to-gold",
        action="store_true",
        help=(
            "Continue the greedy sweep past its usual stop (one compressed cache per "
            "slot) to a state where nothing is materialized and only the gold "
            "(vanilla, uncompressed) models remain. Changes the number of steps, hence "
            "job ids and priorities - do not enable it partway through an existing "
            "task. Pinned on by --producer operator_count."
        ),
    )
    parser.add_argument(
        "--precompute-states",
        type=str,
        choices=list(PRECOMPUTE_COVERAGE),
        default="all",
        help=(
            "Which operators a --precompute pass records: 'all' (the default) is every "
            "materialized level on disk, i.e. what any experiment could later ask for. "
            "Naming a state plan instead records only the union of that plan's own "
            "states - 'default' covers the experiments that pin the default suite "
            "(baselines, sample_size, adaptive_sampling, reordering), "
            "'ablation' those plus the ablation's vanilla-only arm, 'gold' the vanilla "
            "operators alone (reorder_only, a subset of what 'ablation' records), and "
            "none of them covers operator_count or any "
            "run at --state-plan full (mode01), which reach every materialized level. "
            "'greedy'/'greedy_to_gold' equal 'all', and so does 'full' unless "
            "--use-indexes is on, where the state it names is one baseline per slot. "
            "Both models' vanilla operators are recorded either way. Narrow this only "
            "for a dataset you have decided not to run the wider experiments on: a "
            "replay that reaches an operator the store lacks fails hours in."
        ),
    )
    parser.add_argument(
        "--tune-parameters",
        type=str,
        nargs="+",
        choices=("true", "false"),
        default=None,
        help=(
            "Sweep axis: whether the optimizer tunes operator parameters or freezes "
            "them at their defaults and only chooses operators. Defaults to 'true' "
            "alone, i.e. parameters are always tuned. Pass both to "
            "cross the axis; only optim_global has a tuning phase to disable, so "
            "'false' cannot be combined with --approaches lotus/abacus. Swept by "
            "--producer tuning."
        ),
    )
    parser.add_argument(
        "--reorder",
        type=str,
        nargs="+",
        choices=("true", "false"),
        default=None,
        help=(
            "Sweep axis: whether the optimizer reorders the selected physical "
            "operators (Step 4) or runs the plan in the order it was built in. "
            "Defaults to 'true' alone. 'false' takes the whole feature off, including the cost model's "
            "order-awareness, so the optimizer never selects operators for an order it "
            "will not get. Only the gradient-descent approaches and no_optim have a "
            "reordering step this reaches. Swept by --producer reordering."
        ),
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Sweep axis: absolute optimizer profiling sample sizes. Defaults to a "
            "single point that leaves the optimizer's own budget alone "
            "(DEFAULT_SAMPLE_SIZE, 100 rows). Swept by --producer sample_size."
        ),
    )
    parser.add_argument(
        "--adaptive-sampling",
        type=str,
        nargs="+",
        choices=["true", "false"],
        default=None,
        help=(
            "Sweep axis: whether the optimizer draws its profiling sample in growing "
            "rounds, stopping once a larger sample stops paying for itself (extra "
            "profiling plus an extra solve, against the cheaper execution it buys). "
            "Defaults to a single 'false' point, i.e. one sample of --sample-sizes "
            "rows. Pass both "
            "to cross the axis; only the gradient-descent approaches (optim_global, "
            "optim_local, optim_shift_budget) have a sampling loop, so 'true' cannot "
            "be combined with --approaches lotus/abacus."
        ),
    )
    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        choices=APPROACHES,
        default=None,
        help=(
            "Sweep axis: which optimizers to run at every other point of the sweep. "
            "Defaults to optim_global alone, so widening it is always deliberate. "
            "Naming a baseline here compares it against optim_global on whichever "
            "axis the chosen producer sweeps - operator count and sample size "
            "included, not just one of them. optim_local and optim_shift_budget are "
            "the same gradient-descent optimizer in another global optimization mode, "
            "so naming them compares the scope of the search rather than the optimizer."
        ),
    )


def _add_run_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """run_benchmark-only flags.

    --select-executors/--skip-executors narrow the producer's five in-scope names
    (reasondb.coordinator.producers.run_benchmark.EXECUTOR_NAMES)."""
    add_labels_argument(parser)
    add_executor_selection_arguments(parser)


def build_parser(description: str = "") -> argparse.ArgumentParser:
    """The coordinator's full parser. ``description`` is the CLI's own ``__doc__``."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-id", type=str, required=True, help="Namespaces this sweep; workers must pass the same value to connect.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Shared-filesystem directory every job/the SQLite DB live under. Default: benchmark_results/<task-id>.",
    )
    parser.add_argument("--producer", type=str, default="parameter_sweep", choices=sorted(PRODUCERS.keys()))
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address that workers connect to.")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--local", action="store_true", help="Run every job in-process (no Flask/SQLite/workers) - fastest debug/test path.")
    parser.add_argument("--merge-now", action="store_true", help="Merge whatever's already 'done' for --task-id and exit, without enumerating or serving.")
    parser.add_argument("--job-heartbeat-timeout-s", type=float, default=300.0)
    parser.add_argument("--worker-heartbeat-timeout-s", type=float, default=120.0)
    parser.add_argument("--lease-sweep-interval-s", type=float, default=15.0)
    # Not add_device_argument (required=True): the coordinator itself never runs an
    # Executor - it enumerates, tracks, serves, and merges; only --local mode acts as
    # its own worker and needs one (see run_local's WorkerContext). Distributed-mode
    # coordinators would otherwise be forced to pass a flag they don't use.
    parser.add_argument("--device", type=torch.device, default=None, help="Required only with --local.")
    _add_shared_sweep_args(parser)
    _add_parameter_sweep_args(parser)
    _add_run_benchmark_args(parser)
    return parser

