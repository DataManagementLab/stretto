"""Per-job scoring material, and the two passes that read it.

A sweep-point job writes what it measured; a *label* job writes what that will be scored
against; nothing scores inside a job. That split is why this module exists.

Scoring happens in the coordinator: :func:`score_shard` is called from
``coordinator.scoring`` for the live dashboard and again from each producer's ``merge``
for the CSVs, reading pickles in a process that never builds a database and never runs a
labelling pass. Scoring inside a worker would instead require replaying every query of
the benchmark through a full ``execute_benchmark`` loop once per job just to read the
labels, and would attribute the resulting accuracy rows to that labelling pass
(``collector._apply_query_metrics`` stamps ``role`` from the emitting worker's context).
"""

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from reasondb.coordinator.models import Job
from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS
from reasondb.evaluation.evaluation import evaluate
from reasondb.evaluation.row_signature import RowSignature
from reasondb.monitor.dimensions import QUERY_STAT_COLUMNS

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def query_stats_for(benchmark_name: str, split: str) -> Dict[str, Dict[str, Any]]:
    """Shape statistics per query, for whichever benchmark a shard came from.

    Reads the pinned ``queries.json`` only - the scorer runs in the coordinator process,
    which has no database and no benchmark instance, and this must stay that cheap. A
    benchmark the registry does not know, or one whose set is not pinned, scores without
    the columns rather than failing: they are a grouping convenience, not a result.

    Memoized because a merge calls it once per (sweep-point shard x label shard) pair and
    once more per job to stamp the rows, all for the same handful of benchmarks. Every
    caller only ever *reads* the mapping - :func:`evaluate` does ``.get(query, {})`` and
    copies - so one shared dict per benchmark is safe.
    """
    benchmark_cls = ALL_BENCHMARKS.get(benchmark_name)
    if benchmark_cls is None:
        logger.warning(
            "No benchmark %r in the registry; scoring without query shape statistics.",
            benchmark_name,
        )
        return {}
    # `query_stats` is a RandomBenchmark method: the statistics describe the QueryShape a
    # query was instantiated from, and a fixed benchmark's queries are hand-written, so
    # there is no shape and nothing to report. Returning {} is the same degradation the
    # docstring promises for an unpinned set.
    if not hasattr(benchmark_cls, "query_stats"):
        return {}
    return benchmark_cls.query_stats(split)


#: Key under which a loaded shard remembers the directory its manifest paths are
#: relative to. Not written to disk - it is derived from where the shard was read.
BASE_DIR_KEY = "_base_dir"


def manifest_base(job_output_dir) -> Path:
    """What a shard's manifest paths are relative to: the *task* directory.

    Not the job's own directory, though most caches are under it (``cache_<point>``,
    ``cache``): a benchmark's label cache is deliberately beside the job directories
    rather than inside one (``<task>/label_cache/<benchmark>``, see
    ``producers.parameter_sweep._label_dir``) because every step job of a benchmark reads
    it and it must outlive whichever job filled it. One level up covers both, and keeps
    the paths relative so a finished task directory can be moved or read from another
    mount.
    """
    return Path(job_output_dir).parent


def write_shard(job: Job, payload: dict) -> Path:
    """Pickle one job's scoring material into its own directory.

    ``benchmark``/``split`` are stamped here rather than by each caller so a shard is
    self-describing: a merge pass reads a flat list of directories and has to pair sweep
    shards with the label shard of the *same* benchmark, with no job queue to ask.

    Pickle rather than parquet because this half genuinely is not tabular: costs are
    ``CostSummary`` objects and the manifest is nested dicts keyed by guarantee tuples.
    The tidy rows a producer plots from stay in their own ``rows.parquet``.

    ``results`` is a **manifest**, not the answers: each query maps to the path of the
    :class:`~reasondb.evaluation.row_signature.RowSignature` the executor already cached
    for it, relative to :func:`manifest_base`. The answers themselves are hashes and do
    not compress, so a second copy here would double what a saturating self-join costs on
    disk - and a shard holding every query's hashes has to be unpickled whole to score
    one of them.
    """
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / "shard.pkl"
    if "results" in payload:
        payload = {
            **payload,
            "results": relative_manifest(
                payload["results"], manifest_base(job.output_dir)
            ),
        }
    with open(shard_path, "wb") as f:
        pickle.dump({"benchmark": job.benchmark, "split": job.split, **payload}, f)
    return shard_path


def relative_manifest(manifest: Dict[str, Any], base: Path) -> Dict[str, Any]:
    """The two shapes a producer writes, with every path made relative to *base*.

    ``query -> path`` for a label job, ``query -> guarantee -> path`` for a sweep point or
    an approach job. Every result cache lives under the task directory, so a path that
    will not relativize is a wiring bug and says so rather than being silently
    absolutized into something another machine cannot read.
    """
    def one(path) -> str:
        try:
            return str(Path(path).resolve().relative_to(base.resolve()))
        except ValueError as exc:
            raise ValueError(
                f"Cached answer {path} is not under the task directory {base}; a "
                "shard's manifest can only point inside the task that wrote it."
            ) from exc

    out: Dict[str, Any] = {}
    for query, value in manifest.items():
        out[query] = (
            {key: one(path) for key, path in value.items()}
            if isinstance(value, dict)
            else one(value)
        )
    return out


def load_shards(job_output_dirs: List[str]) -> List[dict]:
    """Every ``shard.pkl`` among *job_output_dirs*, in order.

    A missing one is skipped rather than raising: filter-stats and precompute jobs write
    none by design, and a job that failed before its write leaves none either.

    Each shard remembers the directory it came from (:data:`BASE_DIR_KEY`), which is what
    :func:`load_answer` resolves its manifest against.
    """
    shards = []
    for d in job_output_dirs:
        shard_path = Path(d) / "shard.pkl"
        if shard_path.is_file():
            with open(shard_path, "rb") as f:
                shard = pickle.load(f)
            shard[BASE_DIR_KEY] = str(manifest_base(d))
            shards.append(shard)
    return shards


def load_answer(shard: dict, query: str, key: Any = None) -> RowSignature:
    """One query's answer, read from the file the shard's manifest points at.

    Loaded per query rather than per shard: a sweep job over a benchmark with self-joins
    has answers of tens of millions of rows, and scoring only ever needs the two under
    comparison.
    """
    entry = shard["results"][query]
    relative = entry if key is None else entry[key]
    path = Path(shard[BASE_DIR_KEY]) / relative
    with open(path, "rb") as f:
        return pickle.load(f)


def label_name_from_shard(output_dir: str) -> Optional[str]:
    """Which label set a finished label job produced, read from what it wrote.

    The job spec is the first place to look for this (``coordinator.scoring`` does), but
    not every producer can put it there: ``label_set_for`` needs a benchmark *instance*
    to read ``has_ground_truth``, and enumeration deliberately avoids constructing one
    (``Benchmark.count_queries`` exists precisely so counting jobs does not build a
    DuckDB). The shard is written by the job that knows, so it is the authority.
    """
    for shard in load_shards([output_dir]):
        if shard.get("kind") == "label" and isinstance(shard.get("name"), str):
            return shard["name"]
    return None


def score_shard(
    point_shard: dict,
    label_shard: dict,
    debug_root: Path,
    *,
    record_telemetry: bool,
    telemetry_context: Optional[dict] = None,
) -> Optional[pd.DataFrame]:
    """``evaluate()`` one sweep-point shard against one label shard, or ``None``.

    ``evaluate()`` iterates the *labels* and indexes predictions by query, so a query in
    one and not the other raises rather than being skipped. The two normally match
    exactly; scoring the overlap is how a label job enqueued before the query set changed
    degrades - to fewer rows, not to a failed pass.

    Both sides' answers are read here, from the files their shards' manifests name.
    """
    if point_shard.get("benchmark") != label_shard.get("benchmark"):
        return None
    label_results = label_shard.get("results") or {}
    shared = [q for q in label_results if q in point_shard.get("results", {})]
    if not shared:
        return None
    return evaluate(
        benchmark_name=point_shard["benchmark"],
        approach_name=point_shard["name"],
        all_predictions={
            q: {
                key: load_answer(point_shard, q, key)
                for key in point_shard["results"][q]
            }
            for q in shared
        },
        all_labels={q: load_answer(label_shard, q) for q in shared},
        all_costs={q: point_shard["costs"][q] for q in shared},
        debug_query=point_shard.get("debug_query"),
        debug_root=debug_root,
        record_telemetry=record_telemetry,
        telemetry_context=telemetry_context,
        query_stats=query_stats_for(
            point_shard["benchmark"], point_shard.get("split", "dev")
        ),
    )


def score_point_job(job: Job, label_output_dirs: List[str], kind: str) -> List[str]:
    """Score one finished sweep-point job against whatever label shards exist, now.

    Returns the label set names actually scored, so ``coordinator.scoring`` can remember
    them and not re-emit. A job whose labels have not landed yet returns ``[]`` and emits
    nothing - the caller retries on its next pass. That silence is deliberate: a sweep
    point routinely finishes before the (expensive) silver pass it is scored against, and
    accuracy scored against absent or partial labels would be worse than none.

    ``kind`` is the producer's own name for a sweep-point shard ("step", "point", ...),
    so a label shard sitting in the same directory list is never mistaken for one.
    """
    point_shards = load_shards([job.output_dir])
    if not point_shards or point_shards[0].get("kind") != kind:
        return []
    point_shard = point_shards[0]

    scored: List[str] = []
    for label_shard in load_shards(label_output_dirs):
        if label_shard.get("kind") != "label":
            continue
        metrics = score_shard(
            point_shard,
            label_shard,
            # Under the job's own directory, not the CWD-relative default the merge pass
            # uses: the two write the same filenames for the same queries.
            debug_root=Path(job.output_dir) / "debug_outputs",
            record_telemetry=True,
            # Which labeler produced these numbers, and the axis accuracy is read on. No
            # worker_id: `complete_job` nulls `claimed_by` when it releases the lease, so
            # by the time a job is scorable the machine that ran it is no longer on the
            # row, and a fabricated one would be worse than none.
            telemetry_context={"labels": label_shard["name"], "job_id": job.job_id},
        )
        if metrics is not None:
            scored.append(label_shard["name"])
    return scored


def fill_achieved(rows: pd.DataFrame, metrics: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Write one shard's scored metrics into its ``achieved_*`` columns.

    ``metrics`` is indexed by ``(approach_name, query, precision_guarantee,
    recall_guarantee)`` and ``rows`` carries the last three as columns, so the join is on
    those three. A row with no matching metric keeps its ``None`` - that is what an
    unscorable query looks like in the CSV, and it must stay distinguishable from a
    scored zero.
    """
    if metrics is None or metrics.empty:
        return rows
    for idx, m in metrics.iterrows():
        _, query, prec, rec = idx
        mask = (
            (rows["query"] == query)
            & (rows["precision_guarantee"] == prec)
            & (rows["recall_guarantee"] == rec)
        )
        rows.loc[mask, "achieved_precision"] = m.get("precision")
        rows.loc[mask, "achieved_recall"] = m.get("recall")
        rows.loc[mask, "achieved_f1"] = m.get("f1_score")
    return rows


def add_query_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """Stamp each row's query *shape* statistics, read off the pinned query set.

    Deliberately not part of :func:`fill_achieved`, though both widen the same frame:
    these describe the query, not the scoring. Joining them through the metrics frame
    would give a row no complexity bucket whenever its label shard never landed - which
    is exactly the case the runtime figures care about, since ``--figures breakdown`` and
    every ``*_runtime`` metric are drawable on a task nobody has scored. Keyed on the
    query alone for the same reason: the guarantee pair does not enter.

    Only statistics a benchmark actually declares become columns, and only those in
    ``QUERY_STAT_COLUMNS``. :func:`evaluate` can afford to pass ``additional_info``
    through verbatim because a metrics CSV covers one benchmark; a merged sweep CSV spans
    six, so a stray key on one of them would become a column that is empty everywhere
    else.

    An existing non-null value wins, so a sweep that stamped these at write time is not
    overwritten by a later re-merge against a regenerated query set.
    """
    if rows.empty or not {"benchmark", "split", "query"} <= set(rows.columns):
        return rows

    out = rows.copy()
    for (benchmark, split), group in rows.groupby(["benchmark", "split"], dropna=False):
        stats = query_stats_for(str(benchmark), str(split))
        if not stats:
            continue
        unknown = {key for one in stats.values() for key in one} - set(QUERY_STAT_COLUMNS)
        if unknown:
            logger.info(
                "Ignoring query statistics %s declared by %s: not in QUERY_STAT_COLUMNS.",
                sorted(unknown),
                benchmark,
            )
        for column in QUERY_STAT_COLUMNS:
            values = {q: one[column] for q, one in stats.items() if column in one}
            if not values:
                continue
            if column not in out.columns:
                out[column] = pd.NA
            existing = out.loc[group.index, column]
            out.loc[group.index, column] = existing.where(
                existing.notna(), group["query"].map(values)
            )
    return out


def merge_scored_rows(job_output_dirs: List[str], kind: str) -> List[pd.DataFrame]:
    """Every job's ``rows.parquet``, with its ``achieved_*`` columns filled in.

    ``record_telemetry=False`` throughout: :func:`score_point_job` already reported these
    same rows to the monitor and ``query_metrics`` is append-only, so a second emitter
    would double every accuracy row on the dashboard. One emitter, the early one; the
    merge keeps writing the files.

    Query shape statistics are stamped here too (:func:`add_query_stats`), before any
    scoring: an already-finished task picks them up by being re-merged, which reads the
    shards already on disk and re-runs no query.
    """
    labels_by_benchmark = {}
    for shard in load_shards(job_output_dirs):
        if shard.get("kind") == "label":
            labels_by_benchmark.setdefault(shard["benchmark"], []).append(shard)

    out: List[pd.DataFrame] = []
    for d in job_output_dirs:
        rows_path = Path(d) / "rows.parquet"
        if not rows_path.is_file():
            continue
        rows = add_query_stats(pd.read_parquet(rows_path))
        for point_shard in load_shards([d]):
            if point_shard.get("kind") != kind:
                continue
            for label_shard in labels_by_benchmark.get(point_shard["benchmark"], []):
                rows = fill_achieved(
                    rows,
                    score_shard(
                        point_shard,
                        label_shard,
                        debug_root=Path(d) / "debug_outputs",
                        record_telemetry=False,
                    ),
                )
        out.append(rows)
    return out
