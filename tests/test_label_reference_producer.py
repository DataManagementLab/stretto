"""The `label_reference` producer: two arms of one axis, scored by one labeller.

The experiment compares optimizing against ground truth with optimizing against the best
model's verdicts. Both arms run the same executor at the same guarantees, which is
exactly what makes it fragile: `run_benchmark._merge_one_benchmark` keys predictions by
executor name, so without the arm being carried through the shard the two would overwrite
each other per (query, guarantee) -- silently, producing a CSV that looks like a normal
single-arm run. `test_merge_keeps_the_two_arms_apart` pins that.
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from conftest import cached_answer
import pytest

from reasondb.coordinator.producers import get_producer
from reasondb.coordinator.producers import label_reference, run_benchmark
from reasondb.executor import CostSummary
from reasondb.query_plan.physical_operator import ProfilingCost

GUARANTEE = (0.9, 0.9)


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        benchmarks=["artwork_curated"],
        split="dev",
        skip_executors=[],
        select_executors=["optim_global"],
        precision_guarantees=[0.9],
        recall_guarantees=[0.9],
        all_guarantee_combinations=False,
        use_indexes=False,
        cost_type="runtime",
        press_name="expected_attention",
        precompute=None,
        simulate=None,
        clean_result_cache=False,
        debug_query=None,
        sample_sizes=None,
        human_labels=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _enumerate(tmp_path, **overrides):
    return label_reference.enumerate_jobs("ref01", tmp_path, _args(**overrides))


# ── enumeration ──────────────────────────────────────────────────────────────


def test_both_arms_are_enumerated_for_every_guarantee(tmp_path):
    jobs = _enumerate(tmp_path, precision_guarantees=[0.5, 0.9], recall_guarantees=[0.5, 0.9])
    approaches = [j for j in jobs if j.spec["kind"] == "approach"]
    assert len(approaches) == 4  # 2 arms x 2 guarantees
    assert sorted(j.spec["arm"] for j in approaches) == ["human", "human", "model", "model"]
    for job in approaches:
        assert job.spec["human_labels"] == (job.spec["arm"] == "human")


def test_the_arms_get_distinct_ids_and_output_dirs(tmp_path):
    """Distinct ids are not cosmetic: `collect_results_all_guarantees` keys its results
    cache on the job's out_dir, so two arms sharing one would have the second replay the
    first's answers -- the experiment comparing a run against itself."""
    approaches = [j for j in _enumerate(tmp_path) if j.spec["kind"] == "approach"]
    assert len({j.job_id for j in approaches}) == len(approaches)
    assert len({j.output_dir for j in approaches}) == len(approaches)
    assert {j.job_id.rsplit("-", 1)[1] for j in approaches} == {"refmodel", "refhuman"}


def test_label_jobs_are_shared_not_duplicated(tmp_path):
    """Both arms are scored against the same passes. A label pass optimizes nothing, so
    the arm cannot reach it -- and it is the most expensive job in the task."""
    labels = [j for j in _enumerate(tmp_path) if j.spec["kind"] == "label"]
    assert sorted(j.spec["name"] for j in labels) == ["gold", "silver"]
    assert len({j.job_id for j in labels}) == len(labels)
    assert all("human_labels" not in j.spec for j in labels)


def test_both_label_sets_are_requested_regardless_of_the_labels_flag(tmp_path):
    """Gold is the measurement and silver the control; dropping either removes half the
    comparison, so `--labels` does not narrow it."""
    labels = [j for j in _enumerate(tmp_path, labels=["gold"]) if j.spec["kind"] == "label"]
    assert sorted(j.spec["name"] for j in labels) == ["gold", "silver"]


def test_explicit_human_labels_is_rejected(tmp_path):
    """Picking the experiment by name and picking it by flag can never disagree."""
    with pytest.raises(AssertionError, match="--human-labels is fixed by the label_reference"):
        _enumerate(tmp_path, human_labels=True)


def test_precompute_is_rejected_with_a_pointer_to_run_benchmark(tmp_path):
    with pytest.raises(AssertionError, match="--producer run_benchmark"):
        _enumerate(tmp_path, precompute={"artwork_curated": Path("x.json")})


def test_lotus_is_still_refused_when_asked_for(tmp_path):
    """Inherited from run_benchmark's guard: Lotus tunes to its own silver operator, so
    it has no human-labels arm to compare."""
    with pytest.raises(AssertionError, match="lotus"):
        _enumerate(tmp_path, select_executors=["optim_global", "lotus"])


def test_the_default_approach_does_not_include_lotus(tmp_path):
    """A bare `--producer label_reference` must run.

    `run_benchmark`'s default is all five executors, `lotus` among them, and `lotus` with
    human labels is refused -- so inheriting that default would fail at enumeration on a
    flag the caller never passed.
    """
    approaches = [j for j in _enumerate(tmp_path, select_executors=[]) if j.spec["kind"] == "approach"]
    assert {j.spec["name"] for j in approaches} == {"optim_global"}
    assert len(approaches) == 2  # one per arm


def test_the_callers_namespace_is_not_mutated(tmp_path):
    """Wrappers edit a copy: leaving defaults on the caller's object would have the rest
    of the run read choices this producer made for its own enumeration."""
    args = _args(select_executors=[])
    label_reference.enumerate_jobs("ref01", tmp_path, args)
    assert args.select_executors == []
    assert args.human_labels is False


def test_a_benchmark_without_ground_truth_is_refused(tmp_path):
    """Also inherited: without labels the human arm is a silent no-op identical to the
    model arm, and the experiment would report a difference of zero as a finding."""
    with pytest.raises(AssertionError, match="needs per-tuple ground truth"):
        _enumerate(tmp_path, benchmarks=["ecommerce_random"])


def test_jobs_are_stamped_with_this_producer(tmp_path):
    """`coordinator.merge` and `coordinator.scoring` resolve the producer off
    `job.producer`; jobs claiming to be plain run_benchmark ones would be merged by the
    engine and lose the arm column."""
    assert {j.producer for j in _enumerate(tmp_path)} == {"label_reference"}
    assert get_producer("label_reference").merge is label_reference.merge


def test_run_benchmark_enumeration_is_unchanged_without_the_parameter(tmp_path):
    """The producer_name parameter must be invisible to the existing producer."""
    jobs = run_benchmark.enumerate_jobs("t1", tmp_path, _args())
    assert {j.producer for j in jobs} == {"run_benchmark"}
    assert all(j.job_id.startswith("t1-run_benchmark-") for j in jobs)


# ── merge ────────────────────────────────────────────────────────────────────


def _cost() -> CostSummary:
    return CostSummary(
        execution_cost=ProfilingCost(0.0, 0.0, 0.0),
        tuning_cost=ProfilingCost(0.0, 0.0, 0.0),
    )


def _write_shard(out_dir: Path, **shard) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "shard.pkl", "wb") as f:
        pickle.dump(shard, f)


def _frame(values) -> pd.DataFrame:
    return pd.DataFrame({"answer": values})


def _answer(out_dir: Path, values) -> str:
    """A cached answer plus the manifest entry pointing at it, relative to the task dir.

    A shard names where each answer is rather than carrying it - see
    ``shards.write_shard`` - so a hand-built shard needs the file on disk too.
    """
    cached_answer(out_dir / "cache", "q", _frame(values))
    return f"{out_dir.name}/cache/q.sig.pkl"


def test_merge_keeps_the_two_arms_apart(tmp_path):
    """The two arms' predictions stay separate through the merge.

    Two shards, same executor name, same query, same guarantee, different arm. Keyed by
    name alone -- which is what the engine's merge does -- the second silently replaces
    the first and the CSV shows one arm's numbers under both labels.
    """
    task_root = tmp_path / "ref01"
    query = "q"
    for arm, human, answers in (("model", False, ["a", "b"]), ("human", True, ["a"])):
        _write_shard(
            task_root / f"job_{arm}",
            benchmark="artwork_curated", split="dev", kind="approach",
            name="optim_global", human_labels=human,
            results={query: {GUARANTEE: _answer(task_root / f"job_{arm}", answers)}},
            costs={query: {GUARANTEE: _cost()}},
            pipeline_tracks={},
        )
    for label_name in ("silver", "gold"):
        _write_shard(
            task_root / f"job_label_{label_name}",
            benchmark="artwork_curated", split="dev", kind="label",
            name=label_name,
            results={query: _answer(task_root / f"job_label_{label_name}", ["a"])},
            costs={query: _cost()}, pipeline_tracks={},
        )

    written = label_reference.merge(
        "ref01", [str(p) for p in sorted(task_root.iterdir())]
    )
    assert {p.name for p in written} == {
        "label_reference_gold_metrics.csv",
        "label_reference_silver_metrics.csv",
    }

    gold = pd.read_csv(next(p for p in written if "gold" in p.name))
    assert sorted(gold[label_reference.ARM_COLUMN].unique()) == ["human", "model"]
    assert len(gold) == 2, "one row per arm; a collision would leave one"

    # The arms' predictions really are different, i.e. they were not merged before
    # scoring: the model arm returned a false positive, the human arm did not.
    by_arm = gold.set_index(label_reference.ARM_COLUMN)["precision"].to_dict()
    assert by_arm["human"] == 1.0
    assert by_arm["model"] == 0.5


def test_merge_skips_a_label_set_that_has_not_finished(tmp_path):
    """An approach job can finish long before the silver pass it is compared against;
    the gold half must still be written rather than the whole merge failing."""
    task_root = tmp_path / "ref01"
    _write_shard(
        task_root / "job_model",
        benchmark="artwork_curated", split="dev", kind="approach",
        name="optim_global", human_labels=False,
        results={"q": {GUARANTEE: _answer(task_root / "job_model", ["a"])}},
        costs={"q": {GUARANTEE: _cost()}}, pipeline_tracks={},
    )
    _write_shard(
        task_root / "job_label_gold",
        benchmark="artwork_curated", split="dev", kind="label",
        name="gold", results={"q": _answer(task_root / "job_label_gold", ["a"])},
        costs={"q": _cost()}, pipeline_tracks={},
    )
    written = label_reference.merge(
        "ref01", [str(p) for p in sorted(task_root.iterdir())]
    )
    assert [p.name for p in written] == ["label_reference_gold_metrics.csv"]


def test_merge_returns_nothing_when_no_shards_exist(tmp_path):
    assert label_reference.merge("ref01", [str(tmp_path)]) == []
