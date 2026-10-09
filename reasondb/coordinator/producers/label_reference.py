"""Experiment: what does optimizing against ground truth buy over optimizing against a model?

One axis -- which reference the optimizer tunes to -- with both arms scored by the *same*
gold labeller:

===========  ==================================  ====================
arm          optimizes against                   scored against
===========  ==================================  ====================
``model``    the best model's own verdicts       gold **and** silver
``human``    per-tuple ground truth              gold **and** silver
===========  ==================================  ====================

A model-optimized plan can always escalate tuples to a gold operator the loss scores as
perfect, so it reports ``guarantee_met=True`` against a reference that is itself a model
(see ``guarantee_metrics`` in ``evaluation/evaluation.py``). Whether that guarantee survives contact with ground truth is what this measures,
and the two numbers live in different columns of the same row: ``guarantee_met`` /
``achieved_*_lower`` are what the optimizer believed, ``precision`` / ``recall`` are what
``evaluate()`` measured against the labels it was handed.

Scoring both arms against **silver as well** is the control, and it is nearly free
because both label jobs run either way: the model-optimized arm should look fine against
its own reference and worse against ground truth. Without that panel the gold numbers
have nothing to be worse *than*.

**Why this wraps ``run_benchmark`` and not the sweep engine.** The benchmarks that carry
per-tuple ground truth are all fixed query sets (``benchmarks/curated.py``), and
``parameter_sweep`` -- along with every wrapper in ``experiments.py`` -- resolves names
through ``RANDOM_BENCHMARKS``, which excludes them by design. ``run_benchmark`` resolves
through ``ALL_BENCHMARKS`` and already carries the ``has_ground_truth`` guard that makes
``--human-labels`` refuse a benchmark nobody labelled.

**The hazard this producer is shaped around.** ``run_benchmark._merge_one_benchmark``
keys its ``predictions`` dict by ``shard["name"]``, the executor name. Two arms of
``optim_global`` therefore collide -- ``.update()`` per (query, guarantee), last writer
wins, no error, and the resulting CSV looks like a normal single-arm run. Three things
follow: the shard records ``human_labels`` (added in ``run_benchmark.run_job``), the job
ids carry an arm suffix so the arms also get separate output dirs and results caches, and
``merge`` below is this module's own rather than the engine's.

``--precompute`` is rejected here: record with ``--producer run_benchmark``, whose
recording serves both arms. Precomputed work is keyed by (operator, expression, base
tables); both arms build the same ``get_default_configurator`` operator set, and the
label operator is profiled but never executed, so it makes no model calls to record.

    python scripts/run_coordinator.py --local --producer label_reference --task-id ref01 \\
      --benchmarks artwork_curated --precision-guarantees 0.5 0.7 0.9 \\
      --recall-guarantees 0.5 0.7 0.9 --device cuda:0

A note on ``ecommerce_curated``, which is in the default benchmark set: its labels are
derived from the SemBench product catalog rather than annotated per predicate
(``LABEL_PROVENANCE`` in ``benchmarks/curated.py``). This experiment deliberately treats
them as human labels -- they are non-model ground truth, which is the property the axis
turns on -- and the distinction belongs in the paper text rather than in a facet here.
"""

import argparse
import copy
import functools
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from reasondb.coordinator.models import Job
from reasondb.coordinator.producers import run_benchmark as engine
from reasondb.coordinator.producers.experiments import reject_pinned
from reasondb.coordinator.producers.shards import load_answer, query_stats_for
from reasondb.evaluation.evaluation import evaluate
from reasondb.utils.benchmark_args import (
    DEFAULT_BENCHMARKS as COORDINATOR_DEFAULT_BENCHMARKS,
)

logger = logging.getLogger(__name__)

PRODUCER_NAME = "label_reference"

#: The curated benchmarks -- the only ones carrying per-tuple ground truth, hence the
#: only ones on which this axis means anything. Named rather than derived so adding a
#: benchmark to `curated.py` is a deliberate act here too.
DEFAULT_BENCHMARKS = [
    "artwork_curated",
    "email_curated",
    "rotowire_curated",
    "ecommerce_curated",
    "movie_huge_curated",
]

#: Both label sets: gold is the measurement, silver the control. Not configurable --
#: dropping either one removes half of what the experiment compares.
LABEL_SETS = ["silver", "gold"]

#: Default approach, applied when the caller names none. `run_benchmark`'s own default is
#: all five executors, which includes `lotus` -- and `lotus` cannot be run with human
#: labels at all, so a bare `--producer label_reference` would fail at enumeration on a
#: flag the caller never passed. `tuning` pins `optim_global` for the same shape of
#: reason. An explicitly requested `lotus` is still rejected, by the engine's own guard.
DEFAULT_EXECUTORS = ["optim_global"]

#: arm -> (spec value for human_labels, job-id suffix). Both arms are spelled out in the
#: id, including the default one: `_axis_suffix` in parameter_sweep makes the same call,
#: so a job id says what it is without the reader knowing what the defaults were.
ARMS = {"model": (False, "-refmodel"), "human": (True, "-refhuman")}

#: Column added to the merged CSVs naming the arm. `optimized_against`, not
#: `human_labels`: the rows already carry a `labels` notion (which labeller *scored*
#: them), and a boolean named for one of its two values reads badly next to it.
ARM_COLUMN = "optimized_against"


def enumerate_jobs(
    task_id: str, output_root: Path, args: argparse.Namespace
) -> List[Job]:
    """Both arms' approach jobs, plus one shared set of label jobs.

    The engine is called once per arm on a copy of the namespace -- the ``experiments.py``
    pattern of editing a copy rather than the caller's object, so a wrapper cannot leave
    its own defaults on what the rest of the run reads.

    Label jobs are deduplicated across the two calls. They are identical (a label pass
    optimizes nothing, so ``human_labels`` never reaches one) and they are the most
    expensive jobs in the task -- a full pass of the best operators over every query --
    so emitting each twice would roughly double the task for no extra information.
    """
    reject_pinned(
        args, PRODUCER_NAME, "human_labels", "--human-labels",
        swept_by=PRODUCER_NAME,
        because="runs both arms of that flag and compares them",
    )
    assert getattr(args, "precompute", None) is None, (
        "--precompute is not supported by the label_reference producer. Record with "
        "--producer run_benchmark: precomputed work is keyed by (operator, expression, "
        "base tables), both arms build the same operator set, and the label operator "
        "makes no model calls -- so one recording replays for both arms."
    )

    # "Left at the coordinator's default" counts as "not specified", the same test
    # `resolve_precompute_simulate` makes: argparse always supplies `--benchmarks`, so a
    # falsiness check alone would never select this producer's curated defaults.
    selected = getattr(args, "benchmarks", None)
    if not selected or set(selected) == set(COORDINATOR_DEFAULT_BENCHMARKS):
        args = copy.copy(args)
        args.benchmarks = list(DEFAULT_BENCHMARKS)
    if not getattr(args, "select_executors", None):
        args = copy.copy(args)
        args.select_executors = list(DEFAULT_EXECUTORS)

    jobs: List[Job] = []
    seen_label_jobs = set()
    for arm, (human_labels, suffix) in ARMS.items():
        arm_args = copy.copy(args)
        arm_args.human_labels = human_labels
        arm_args.labels = list(LABEL_SETS)

        for job in engine.enumerate_jobs(
            task_id, output_root, arm_args, producer_name=PRODUCER_NAME
        ):
            if job.spec.get("kind") != "approach":
                if job.job_id in seen_label_jobs:
                    continue
                seen_label_jobs.add(job.job_id)
                jobs.append(job)
                continue
            # Distinct id -> distinct output_dir, which is also what keeps the two arms'
            # results caches apart: collect_results_all_guarantees keys its cache on
            # out_dir, so identical ids would have the second arm replay the first's
            # answers and the experiment would compare a run against itself.
            job.job_id += suffix
            job.output_dir = str(Path(output_root) / f"job_{job.job_id}")
            job.spec["arm"] = arm
            jobs.append(job)
    return jobs


# The engine executes both arms identically -- the arm only changes which configurator
# `run_job` builds, off `spec["human_labels"]`, which the engine already reads. score_job
# likewise needs no help: it scores one approach job against whatever label shards exist,
# so each arm is reported separately, and the `job_id` it puts in telemetry_context
# carries the suffix. `human_labels` is already a monitor dimension and the `job_spec`
# event already carries it, so the dashboard groups by arm with nothing added here.
run_job = functools.partial(engine.run_job, producer_name=PRODUCER_NAME)
score_job = engine.score_job


def _evaluate_arms(
    benchmark_name: str,
    split: str,
    shards: List[dict],
    out_dir: Path,
) -> List[Path]:
    """Score every (executor, arm) against every label set that finished.

    The engine's own ``_merge_one_benchmark`` cannot be reused: it groups predictions by
    executor name only, so the two arms would be folded into one before ``evaluate()``
    ever saw them.
    """
    labels: Dict[str, dict] = {}
    # (executor name, arm) -> merged predictions / costs across that arm's guarantee jobs
    predictions: Dict[tuple, Dict[str, dict]] = {}
    costs: Dict[tuple, Dict[str, dict]] = {}

    # As in the engine's own merge: a shard's `results` is a manifest, so the answers
    # come out of the files it names (`shards.load_answer`).
    for shard in shards:
        if shard["kind"] == "label":
            labels[shard["name"]] = {q: load_answer(shard, q) for q in shard["results"]}
            continue
        arm = "human" if shard.get("human_labels") else "model"
        key = (shard["name"], arm)
        for query, per_guarantee in shard["results"].items():
            predictions.setdefault(key, {}).setdefault(query, {}).update(
                {gk: load_answer(shard, query, gk) for gk in per_guarantee}
            )
        for query, per_guarantee in shard["costs"].items():
            costs.setdefault(key, {}).setdefault(query, {}).update(per_guarantee)

    missing = [name for name in LABEL_SETS if name not in labels]
    if missing:
        logger.warning(
            "label_reference merge: no %s label shard for %s; those metrics are skipped.",
            "/".join(missing), benchmark_name,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = query_stats_for(benchmark_name, split)
    written: List[Path] = []

    for label_name in LABEL_SETS:
        if label_name not in labels:
            continue
        frames = []
        for (exec_name, arm), arm_predictions in sorted(predictions.items()):
            # record_telemetry=False: `score_job` already reported each of these rows the
            # moment its labels landed, and query_metrics is append-only -- see
            # evaluate()'s docstring.
            frame = evaluate(
                benchmark_name,
                exec_name,
                arm_predictions,
                labels[label_name],
                costs[(exec_name, arm)],
                record_telemetry=False,
                query_stats=stats,
            )
            frame[ARM_COLUMN] = arm
            frames.append(frame)
        if not frames:
            continue
        path = out_dir / f"{PRODUCER_NAME}_{label_name}_metrics.csv"
        pd.concat(frames).to_csv(path, index=True)
        written.append(path)
    return written


def merge(task_id: str, job_output_dirs: List[str]) -> List[Path]:
    """One pair of CSVs per benchmark: gold-scored and silver-scored, both arms in each.

    Filenames end in ``_metrics.csv`` so they stay inside the
    ``<output-dir>/<benchmark>/<split>/*metrics.csv`` glob every ``scripts/plot_*.py``
    already uses, and are prefixed with the producer name so two experiments merged into
    one directory stay distinguishable.
    """
    shards = engine._load_shards(job_output_dirs)
    if not shards:
        return []

    by_benchmark: Dict[tuple, List[dict]] = {}
    for shard in shards:
        by_benchmark.setdefault((shard["benchmark"], shard["split"]), []).append(shard)

    task_root = Path(job_output_dirs[0]).parent
    written: List[Path] = []
    for (benchmark_name, split), benchmark_shards in by_benchmark.items():
        out_dir = task_root / "merged" / benchmark_name / split
        written.extend(
            _evaluate_arms(benchmark_name, split, benchmark_shards, out_dir)
        )
    return written
