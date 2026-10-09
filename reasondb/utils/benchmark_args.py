"""Argparse wiring shared by the ``run_benchmark*`` scripts.

Every flag defined here is used by at least two of those scripts. Where the scripts
disagree it is only about defaults (which benchmark, which output directory, which
guarantees), so defaults stay parameters of these helpers and remain visible at the call
site — only the flag names, types, choices and help texts are shared.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from reasondb.query_plan.physical_operator import CostType

# Mirrors reasondb.monitor.server.DEFAULT_PORT, duplicated rather than imported because
# that module imports flask, which is optional. Kept equal by
# tests/test_monitor_run_lifecycle.py.
_MONITOR_DEFAULT_PORT = 5099

SPLITS = ["dev", "test"]
PRESS_NAMES = ["expected_attention", "finch", "kvzip", "finch-cachenotes"]

USE_INDEXES_HELP = (
    "Let each model family materialize a single KV cache at its least-compressed enabled "
    "compression ratio and index the higher effective ratios out of that one cache. "
    "Without this flag every effective ratio reads a cache materialized at exactly that "
    "ratio (effective_cr == materialized_cr): more storage, no indexing."
)


HUMAN_LABELS_HELP = (
    "Measure precision/recall guarantees against the per-tuple ground-truth files a step's "
    "logical operator points at, instead of against the highest-quality model's own "
    "verdicts. The label source is profiled but never executed - so it cannot be used as a "
    "fallback for tuples the cheaper operators are unsure about, the highest-quality model "
    "becomes an ordinary pickable candidate, and targets above what the real operators can "
    "reach become genuinely unsatisfiable (reported as guarantee_met=False, with a "
    "best-effort plan, rather than silently met). Steps with no ground-truth file keep "
    "model-derived labels."
)


def add_use_indexes_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--use-indexes`` flag that selects the KV cache materialization scheme."""
    parser.add_argument("--use-indexes", action="store_true", help=USE_INDEXES_HELP)


def add_human_labels_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--human-labels`` flag that selects the guarantee reference."""
    parser.add_argument("--human-labels", action="store_true", help=HUMAN_LABELS_HELP)


MATERIALIZED_CR_HELP = (
    "Compression ratio the physical KV caches on disk were generated at — the "
    "--kv-method baseline passed to generate_kv_caches_indices.py / "
    "generate_kv_caches_image_indices.py. The server reads the relative indices from "
    "{press}/comp{materialized}/indices/comp{effective}/, so this must name the baseline "
    "for indexed serving to find them; each method's ratio is clamped to it, so the "
    "baseline method itself still loads its physical cache. Requires the servers to run "
    "with --use-relative-indices. Omit it (the default) to have every method read a cache "
    "materialized at exactly its own effective ratio, i.e. physical serving."
)


def add_materialized_cr_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--materialized-cr``, the baseline ratio the relative indices point into.

    Default ``None`` means "materialized == effective" (physical serving); it is NOT 0.0,
    which would clamp every method down to a cr-0 baseline.
    """
    parser.add_argument(
        "--materialized-cr", type=float, default=None, help=MATERIALIZED_CR_HELP
    )


#: What ``--benchmarks`` selects when the flag is omitted on the coordinator.
#:
#: Defined here (not beside the parser) to avoid an import cycle. Used by
#: ``resolve_precompute_simulate`` and ``producers/label_reference.py`` to distinguish
#: "left at the default" from "explicitly asked for exactly this".
DEFAULT_BENCHMARKS = ["movie_random"]


def add_benchmarks_argument(
    parser: argparse.ArgumentParser, choices: Iterable[str], default: Sequence[str]
) -> None:
    """Add ``--benchmarks``, the multi-benchmark selector."""
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=choices,
        default=list(default),
        help="The benchmarks to run.",
    )


def add_benchmark_argument(
    parser: argparse.ArgumentParser, choices: Iterable[str], default
) -> None:
    """Add ``--benchmark``, the single-benchmark selector used by the operator studies."""
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=choices,
        default=default,
        help="The benchmark to run.",
    )


def add_split_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        type=str,
        choices=SPLITS,
        default="dev",
        help="The split of the benchmark to run.",
    )


def add_labels_argument(
    parser: argparse.ArgumentParser, default: Sequence[str] = ("silver", "gold")
) -> None:
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=list(default),
        help="Labels to use for evaluation (silver and/or gold).",
    )


def add_executor_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the two complementary executor filters."""
    parser.add_argument(
        "--skip-executors",
        type=str,
        nargs="+",
        default=[],
        help="Executors to skip.",
    )
    parser.add_argument(
        "--select-executors",
        type=str,
        nargs="+",
        default=[],
        help="Executors to run; every other executor is skipped.",
    )


def add_clean_result_cache_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--clean-result-cache",
        action="store_true",
        help="Whether to clean all previously cached results",
    )


def add_guarantee_arguments(
    parser: argparse.ArgumentParser,
    precision_default: Sequence[float] = (0.5, 0.7, 0.9),
    recall_default: Sequence[float] = (0.5, 0.7, 0.9),
) -> None:
    """Add ``--precision-guarantees``/``--recall-guarantees`` and how to pair them up."""
    parser.add_argument(
        "--precision-guarantees",
        type=float,
        nargs="+",
        default=list(precision_default),
    )
    parser.add_argument(
        "--recall-guarantees",
        type=float,
        nargs="+",
        default=list(recall_default),
    )
    parser.add_argument(
        "--all-guarantee-combinations",
        action="store_true",
        help="Cross-product precision x recall instead of zipping them.",
    )


def add_cost_type_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cost-type",
        type=str,
        choices=[c.value for c in CostType],
        default=CostType.RUNTIME.value,
        help="The cost type to optimize for.",
    )


def add_debug_query_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--debug-query",
        type=str,
        default=None,
        help="If set, only run the benchmark for the specified query.",
    )


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        type=torch.device,
        required=True,
        help="The device to run optimization on",
    )


def add_output_dir_argument(parser: argparse.ArgumentParser, default: Path) -> None:
    parser.add_argument("--output-dir", type=Path, default=default)


def add_precompute_simulate_arguments(
    parser: argparse.ArgumentParser,
    precompute_note: str = "",
    simulate_note: str = "",
) -> None:
    """Add the record/replay pair. Both take ``BENCH=PATH`` pairs and are turned into
    ``{benchmark: path}`` mappings - and checked against each other and against
    ``--benchmarks`` - by :func:`resolve_precompute_simulate`.

    The notes are appended to the shared help texts for whatever a given script does
    differently (which queries it records, what it still reads off disk).
    """
    parser.add_argument(
        "--precompute",
        type=str,
        nargs="+",
        default=None,
        metavar="BENCH=PATH",
        help=(
            "Record every operator response into one JSON file per dataset, e.g. "
            "'--precompute movie_random=movie.json artwork_random_medium=art.json', then "
            "exit. Each file is read as well as written: an existing one supplies the "
            "resume markers and the pinned operator configs, so a re-run skips what it "
            "already did and asks the same questions it asked the first time. Requires "
            "the model servers. Set REASONDB_PRECOMPUTE_SKIP_MODALITIES (comma-separated, "
            "e.g. 'image,audio') to skip operators needing those modalities' servers; "
            "skipped operators aren't marked precomputed, so a later run with those "
            "servers up and the same file fills them in. " + precompute_note
        ).strip(),
    )
    parser.add_argument(
        "--simulate",
        type=str,
        nargs="+",
        default=None,
        metavar="BENCH=PATH",
        help=(
            "Replay recorded operator responses instead of calling models, one JSON file "
            "per dataset - the same mapping --precompute wrote, e.g. "
            "'--simulate movie_random=movie.json artwork_random_medium=art.json'. Each "
            "job runs one benchmark and is given only that benchmark's file. The KV cache "
            "servers are not needed, but the embedding servers (5005/5324) still are. "
            + simulate_note
        ).strip(),
    )


def add_query_range_arguments(
    parser: argparse.ArgumentParser, kinds: Sequence[str]
) -> None:
    """Add a ``--range-<kind>`` slicing flag per query kind (filter, extract, map, join)."""
    for kind in kinds:
        parser.add_argument(
            f"--range-{kind}",
            nargs=2,
            type=int,
            metavar=("START", "END"),
            default=None,
            help=(
                f"1-indexed inclusive range of {kind} queries to run "
                f"(e.g. --range-{kind} 1 10)."
            ),
        )


def add_single_operator_arguments(
    parser: argparse.ArgumentParser, kv_methods: Sequence[str]
) -> None:
    """Add the operator-study flags shared by the two ``run_benchmark_single_operator*``
    scripts. They differ only in which KV methods they can serve, hence the parameter."""
    parser.add_argument(
        "--all-operators",
        action="store_true",
        help="Run all single operators (filters + extracts). If not set, only runs filter operators.",
    )
    parser.add_argument(
        "--only-extracts",
        action="store_true",
        help="Run only extract operators. Cannot be used with --all-operators.",
    )
    parser.add_argument(
        "--only-join",
        action="store_true",
        help="Run only join operators. Cannot be used with --all-operators or --only-extracts.",
    )
    parser.add_argument(
        "--kv-methods",
        type=str,
        nargs="+",
        default=list(kv_methods),
        help="KV-cache methods to run predictions with.",
    )
    parser.add_argument(
        "--no_high_selectivity",
        action="store_true",
        help="Filter out queries where the silver model (kv70B00) keeps less than 5%% of samples.",
    )
    parser.add_argument(
        "--press-name",
        type=str,
        choices=PRESS_NAMES,
        default="expected_attention",
        help="Name of the compression press to use",
    )
    parser.add_argument(
        "--reference-method",
        type=str,
        default="kv70B00",
        help="Method used as silver reference for evaluation (default: kv70B00).",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Fraction of dataset rows to use, expressed as a percentage (e.g. 50 = first 50%%). "
        "Changes the output directory to benchmark_results_{sample}pct/filter_stats.",
    )
    add_query_range_arguments(parser, ["filter", "extract"])


def add_monitor_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags that control the run monitor dashboard (see ``reasondb.monitor``)."""
    parser.add_argument(
        "--monitor-port",
        type=int,
        default=None,
        metavar="PORT",
        help=(
            f"Port for the run monitor dashboard (default {_MONITOR_DEFAULT_PORT}, or "
            "$REASONDB_MONITOR_PORT). Scans a few ports upward if taken."
        ),
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable the run monitor dashboard and telemetry sidecar entirely.",
    )


def parse_dataset_path_mapping(
    values: Sequence[str],
    *,
    flag: str,
    known_benchmarks: Dict[str, Any],
    must_exist: bool,
) -> Dict[str, Path]:
    """Turn ``["movie_random=movie.json", ...]`` into ``{"movie_random": Path(...)}``.

    One dataset, one file: ``SimulateStore.save`` rewrites its whole file after every
    query, so two datasets sharing a path would race.

    The checks here fail early on errors that would otherwise surface only on a worker:
    unrecognized benchmark names, missing ``--simulate`` files, and duplicated keys.
    """
    mapping: Dict[str, Path] = {}
    spelled_as: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(
                f"{flag} takes BENCH=PATH pairs, one per dataset; got {value!r}. "
                f"Example: {flag} movie_random=movie.json artwork_random_medium=art.json"
            )
        # Split on the first '=' only: a path may contain one.
        name, _, raw_path = value.partition("=")
        benchmark = known_benchmarks.get(name)
        if benchmark is None:
            raise SystemExit(
                f"{flag}: unknown benchmark {name!r}. Known: "
                f"{', '.join(sorted(n for n in known_benchmarks if '-' not in n))}"
            )
        # 'movie-random' and 'movie_random' are the same benchmark under two spellings;
        # canonicalize so they collide as one key instead of mapping to two files.
        #
        # Canonicalize by un-hyphenating the *spelling*, not via the class's `name()`:
        # registry keys and `name()` can differ (e.g. enron_email -> enronemail), and the
        # result becomes `args.benchmarks`, which is looked up in the same registry.
        # `_with_hyphen_aliases` derives the hyphen keys from the underscore ones; the
        # assert checks that convention.
        canonical = name.replace("-", "_")
        assert canonical in known_benchmarks, (
            f"{flag}: {name!r} canonicalizes to {canonical!r}, which is not a benchmark "
            "key. Aliases are expected to differ from their primary key only by "
            "hyphens (see benchmark_registry._with_hyphen_aliases); an alias of another "
            "shape needs a real primary-key lookup here."
        )
        if canonical in mapping:
            raise SystemExit(
                f"{flag}: {canonical} given twice (as {spelled_as[canonical]!r} and "
                f"{name!r}); one dataset records into exactly one file."
            )
        path = Path(raw_path)
        if not raw_path:
            raise SystemExit(f"{flag}: no path given for {name!r}.")
        if path.is_dir():
            raise SystemExit(f"{flag}: {path} is a directory, not a JSON file.")
        if must_exist and not path.exists():
            raise SystemExit(
                f"{flag}: {path} does not exist. Record it first with "
                f"--precompute {canonical}={path}."
            )
        mapping[canonical] = path
        spelled_as[canonical] = name

    by_path: Dict[Path, str] = {}
    for canonical, path in mapping.items():
        resolved = path.resolve()
        if resolved in by_path:
            raise SystemExit(
                f"{flag}: {by_path[resolved]} and {canonical} both map to {path}. "
                "A store rewrites its whole file after every query, so sharing one path "
                "loses whichever writer finishes first."
            )
        by_path[resolved] = canonical
    return mapping


def resolve_precompute_simulate(
    args: argparse.Namespace,
    known_benchmarks: Dict[str, Any],
    default_benchmarks: Sequence[str] = (),
    *,
    simulate_files_must_exist: bool = True,
) -> None:
    """Parse both flags into ``{benchmark: path}`` and reconcile them with ``--benchmarks``.

    Recording and replaying cannot happen in the same run. Beyond that, the mapping and
    ``--benchmarks`` must describe the same set: a benchmark with no mapping would record
    to nowhere (or replay from a store that cannot serve it), and a mapping key that is
    not being run is a typo that would otherwise pass silently. When ``--benchmarks`` was
    left at its default the mapping simply *is* the selection, so
    ``--precompute a=a.json b=b.json`` needs no second list.

    ``simulate_files_must_exist=False`` keeps every check above but drops the one that
    reads the filesystem, for callers that only inspect what an argv would enumerate
    (e.g. ``scripts/generate_experiment_report.py``) on a machine without the recordings.

    Mutates ``args`` in place.
    """
    assert not (args.precompute is not None and args.simulate is not None), (
        "--precompute and --simulate are mutually exclusive: the first records "
        "operator responses (servers required), the second replays them."
    )
    for flag, must_exist in (
        ("precompute", False),
        ("simulate", simulate_files_must_exist),
    ):
        values = getattr(args, flag, None)
        if values is None:
            continue
        mapping = parse_dataset_path_mapping(
            values, flag=f"--{flag}", known_benchmarks=known_benchmarks,
            must_exist=must_exist,
        )
        setattr(args, flag, mapping)

        selected = getattr(args, "benchmarks", None)
        if selected is None or set(selected) == set(default_benchmarks):
            args.benchmarks = list(mapping)
            continue
        # Same canonicalization as the mapping it is compared against (see
        # parse_dataset_path_mapping), so both sets use the same keys.
        canonical_selected = {
            name.replace("-", "_") for name in selected if name in known_benchmarks
        }
        assert canonical_selected == set(mapping), (
            f"--benchmarks {sorted(canonical_selected)} and --{flag} "
            f"{sorted(mapping)} must describe the same datasets: every benchmark being "
            f"run needs its own file, and a file for a benchmark that is not being run "
            "is a typo."
        )


def resolve_guarantees(args: argparse.Namespace) -> List[Tuple[float, float]]:
    """Validate the guarantee arguments and pair them into (precision, recall) tuples.

    Zipped by default, cross-producted under ``--all-guarantee-combinations``.
    """
    assert all(0.0 <= p <= 1.0 for p in args.precision_guarantees), (
        f"--precision-guarantees must all be in [0, 1]; got {args.precision_guarantees}."
    )
    assert all(0.0 <= r <= 1.0 for r in args.recall_guarantees), (
        f"--recall-guarantees must all be in [0, 1]; got {args.recall_guarantees}."
    )
    if args.all_guarantee_combinations:
        return [
            (p, r) for p in args.precision_guarantees for r in args.recall_guarantees
        ]
    assert len(args.precision_guarantees) == len(args.recall_guarantees), (
        "When zipping guarantees, --precision-guarantees and --recall-guarantees "
        f"must have equal length ({len(args.precision_guarantees)} vs "
        f"{len(args.recall_guarantees)}); pass --all-guarantee-combinations to "
        "cross-product them instead."
    )
    return list(zip(args.precision_guarantees, args.recall_guarantees))
