"""Scoring reads a set of rows, so a shard carries hashes of them rather than the rows.

The reduction exists because ``movie_random_huge``'s self-join queries cross-join 10 000
reviews into 10^8 candidate pairs: a shard holding those frames would overflow pickle's
memo (one entry per distinct object, ~22 a row, and the index is four bytes) with
``memo id too large for LONG_BINGET``. These tests pin the two things that must hold for
that trade to be free - the metrics are unchanged, and the shard's size does not track
the answer's - plus the compatibility path that keeps frame-based shards scorable.
"""

import io
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reasondb.coordinator.models import Job
from reasondb.coordinator.producers import shards as shards_mod
from reasondb.evaluation.metrics.metrics_manager import (
    MetricsManager,
    normalize_for_comparison,
)
from reasondb.evaluation.row_signature import RowSignature
from reasondb.utils.answer_normalization import postprocess_dataframe

_FLOAT_INT_RE = re.compile(r"^(-?\d+)\.0+$")


def _reference_row_set(df: pd.DataFrame) -> set:
    """What "the answer" means, written out in plain Python.

    Deliberately not a call into the module under test: it is the definition the hashes
    are supposed to stand for - each row a frozen multiset of its normalized cells, the
    whole answer a set of those.
    """
    processed = postprocess_dataframe(df)
    rows = set()
    for _, row in processed.iterrows():
        cells = []
        for value in row:
            cell = str(value).strip().lower()
            cells.append(_FLOAT_INT_RE.sub(r"\1", cell))
        rows.add(tuple(sorted(cells)))
    return rows


def _reference_metrics(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict:
    preds, gold = _reference_row_set(predictions), _reference_row_set(labels)
    tp = len(preds & gold)
    precision = tp / len(preds) if preds else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


def _memo_entries(obj) -> int:
    """How many objects one ``pickle.dump`` would memoize.

    This is the quantity that overflowed: the C pickler writes a memo index into four
    bytes, so a dump of more than 2**32 distinct objects raises. The pure-Python pickler
    exposes the same memo, and counting it is a far cheaper test than building a frame
    large enough to actually overflow one.
    """

    class _Sink(io.RawIOBase):
        def write(self, b):
            return len(b)

    pickler = pickle._Pickler(_Sink(), protocol=5)
    pickler.dump(obj)
    return len(pickler.memo)


def _answers():
    """(predictions, labels) pairs covering the shapes the metrics have to survive."""
    gold = pd.DataFrame({"id": [1, 2, 3, 4], "verdict": ["Yes", "no", "yes", "NO"]})
    return {
        "exact": (gold.copy(), gold),
        "half": (gold.head(2).copy(), gold),
        "disjoint": (
            pd.DataFrame({"id": [7, 8], "verdict": ["yes", "no"]}),
            gold,
        ),
        "over-predicting": (
            pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "verdict": ["yes"] * 6}),
            gold,
        ),
        "duplicate rows": (
            pd.concat([gold, gold], ignore_index=True),
            gold,
        ),
        "column order swapped": (
            gold[["verdict", "id"]].copy(),
            gold,
        ),
        "float-ish integers": (
            pd.DataFrame({"id": [1.0, 2.00, 3.0, 4.0], "verdict": ["yes", "no", "yes", "no"]}),
            gold,
        ),
        "empty predictions": (gold.iloc[0:0].copy(), gold),
    }


@pytest.mark.parametrize("case", sorted(_answers()))
def test_signatures_score_exactly_as_the_row_sets_they_stand_for(case):
    predictions, labels = _answers()[case]
    expected = _reference_metrics(predictions, labels)

    from reasondb.evaluation.evaluation import _score_signatures

    got = _score_signatures(
        "q", RowSignature.of(predictions), RowSignature.of(labels)
    )
    assert got == pytest.approx(expected)
    # And the in-memory path callers outside the coordinator still use agrees with it.
    assert MetricsManager(
        postprocess_dataframe(predictions), postprocess_dataframe(labels)
    ).evaluate_all() == pytest.approx(expected)


def test_the_raw_cardinality_survives_the_reduction():
    """``predicted_output_cardinality`` counts rows, not distinct rows.

    The hashes are deduplicated - that is what ``|preds|`` means in the confusion counts
    - so the column the metrics CSV reports has to be carried separately or a fan-out
    join would report its answer as far smaller than it was.
    """
    frame = pd.DataFrame({"a": ["x"] * 10, "b": ["y"] * 10})
    signature = RowSignature.of(frame)
    assert signature.n_rows == 10
    assert signature.n_unique == 1


def test_the_same_instant_at_two_offsets_still_matches():
    """Why datetimes are kept out of the comparison at all.

    The same instant comes back rendered against different UTC offsets across runs
    (Europe/Paris LMT "+00:09" against CET "+01:00"), equal as instants and different as
    strings - so an unrelated timestamp column would otherwise break the match for rows
    that were filtered correctly.
    """
    labels = pd.DataFrame({"id": ["1"], "seen": ["1500-01-01 01:00:00+01:00"]})
    predictions = pd.DataFrame({"id": ["1"], "seen": ["1500-01-01 00:09:00+00:09"]})

    from reasondb.evaluation.evaluation import _score_signatures

    scored = _score_signatures(
        "q", RowSignature.of(predictions), RowSignature.of(labels)
    )
    assert scored["precision"] == 1.0 and scored["recall"] == 1.0


def test_a_datetime_column_present_on_one_side_only_no_longer_narrows_the_frame():
    """Excluding a datetime column must not change how *wide* the frame is.

    Whether a column looks like a datetime is read off its own values, so a timestamp
    column that came back all-null on one side is recognized on the other side only.
    Dropping it would leave the two frames at different widths and the comparison would
    run over a prefix of their columns - matching rows that agree on the prefix and
    disagree past it. Neutralizing the column instead keeps the widths equal, so the disagreement stays where it is (these two frames do not match: one
    knows when, the other does not) rather than being scored on a truncated answer.
    """
    labels = pd.DataFrame(
        {
            "id": ["1", "2"],
            "seen": ["1500-01-01 01:00:00+01:00", "1500-01-02 01:00:00+01:00"],
        }
    )
    predictions = pd.DataFrame({"id": ["1", "2"], "seen": [None, None]})

    pred_signature, gt_signature = RowSignature.of(predictions), RowSignature.of(labels)
    assert pred_signature.n_columns == gt_signature.n_columns == 2
    assert len(normalize_for_comparison(labels).columns) == 2


def test_an_all_datetime_frame_still_compares_on_its_values():
    """Neutralizing every column would make any two non-empty answers identical."""
    labels = pd.DataFrame({"seen": ["1500-01-01 01:00:00+01:00"]})
    other = pd.DataFrame({"seen": ["1600-01-01 01:00:00+01:00"]})
    assert not np.array_equal(
        RowSignature.of(labels).hashes, RowSignature.of(other).hashes
    )


def test_column_order_is_not_part_of_the_answer():
    frame = pd.DataFrame({"a": ["x"], "b": ["y"]})
    assert np.array_equal(
        RowSignature.of(frame).hashes, RowSignature.of(frame[["b", "a"]]).hashes
    )
    # ...and the normalized frame it is derived from is positional, not named.
    assert list(normalize_for_comparison(frame).columns) == [0, 1]


def test_a_shard_no_longer_grows_with_the_answer_it_describes(tmp_path):
    """Pickle memoizes per distinct object, ~22 of them a row.

    A cross-join answer is tens of millions of 22-column rows of text, enough to overflow
    the four-byte memo index. A shard therefore carries one uint64 per distinct
    row - a numpy buffer, memoized once whatever its length - plus a fixed-size sample,
    so the memo cost of an answer stops depending on how big the answer is.
    """
    def shard_for(n_rows: int) -> dict:
        frame = pd.DataFrame(
            {f"c{c}": [f"r{r}-c{c}-payload" for r in range(n_rows)] for c in range(22)}
        )
        return {"q1": {(0.7, 0.7): RowSignature.of(frame)}}

    # Both past SAMPLE_ROWS, so the only part that could still grow is the hash buffer.
    small, large = _memo_entries(shard_for(200)), _memo_entries(shard_for(20_000))
    assert large == small, (
        f"memo grew with the answer ({small} -> {large}); a big enough one overflows it"
    )
    # For contrast: the frames themselves cost an entry per cell.
    raw = _memo_entries(
        {"q1": {(0.7, 0.7): pd.DataFrame({f"c{c}": [f"r{r}-c{c}" for r in range(5000)]
                                          for c in range(22)})}}
    )
    assert raw > 5000 * 22


def _job(tmp_path, job_id="j1") -> Job:
    """A job whose output dir sits under a task dir, which is what a manifest is relative
    to (`shards.manifest_base`)."""
    return Job(
        job_id=job_id, task_id="t", producer="parameter_sweep", benchmark="bench_a",
        split="dev", spec={"kind": "step"}, required_capabilities=[],
        output_dir=str(tmp_path / "task" / f"job_{job_id}"),
    )


def _cache(job: Job, name: str, signature: RowSignature) -> Path:
    """Write one cached answer where the executor would have, and return its path."""
    cache_dir = Path(job.output_dir) / "cache_point"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.sig.pkl"
    with open(path, "wb") as f:
        pickle.dump(signature, f)
    return path


def test_a_shard_points_at_the_cached_answers_rather_than_copying_them(tmp_path):
    """One copy of the hashes on disk, and one query's loaded at a time.

    Hashes do not compress - a saturating self-join answer is ~8 bytes a row where the
    parquet answer frame is under one - so a shard that embedded them would double what
    the run costs on disk, and would have to be unpickled whole to score a single query.
    """
    job = _job(tmp_path)
    signature = RowSignature.of(pd.DataFrame({"id": [1, 2], "v": ["yes", "no"]}))
    path = _cache(job, "q1", signature)

    shard_path = shards_mod.write_shard(
        job,
        {
            "kind": "step", "name": "point", "label_set": "silver",
            "results": {"q1": {(0.7, 0.7): path}},
            "costs": {},
        },
    )
    written = pickle.loads(shard_path.read_bytes())
    entry = written["results"]["q1"][(0.7, 0.7)]
    assert not isinstance(entry, RowSignature)
    # Relative to the *task* directory, so the tree can be moved or read from elsewhere.
    assert not Path(entry).is_absolute()
    assert Path(entry).parts[0] == f"job_{job.job_id}"

    loaded = shards_mod.load_shards([job.output_dir])[0]
    assert np.array_equal(
        shards_mod.load_answer(loaded, "q1", (0.7, 0.7)).hashes, signature.hashes
    )


def test_a_manifest_may_not_point_outside_its_own_task(tmp_path):
    """A path that will not relativize is a wiring bug, not something to absolutize.

    An absolute path baked into a shard is unreadable from any other mount, and the
    failure would surface as a missing answer hours later in the merge.
    """
    job = _job(tmp_path)
    Path(job.output_dir).mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="not under the task directory"):
        shards_mod.write_shard(
            job,
            {"kind": "label", "name": "silver",
             "results": {"q1": tmp_path / "elsewhere" / "q1.sig.pkl"}},
        )


def test_a_shard_scores_the_answers_its_manifest_names(tmp_path):
    """End to end through `score_shard`: manifest -> files -> metrics."""
    point, label = _job(tmp_path, "point"), _job(tmp_path, "label")
    predictions = RowSignature.of(
        pd.DataFrame({"id": [1, 2, 3], "v": ["yes", "no", "yes"]})
    )
    labels = RowSignature.of(pd.DataFrame({"id": [1, 2], "v": ["yes", "no"]}))

    shards_mod.write_shard(
        point,
        {"kind": "step", "name": "p",
         "results": {"q1": {(0.7, 0.7): _cache(point, "q1", predictions)}},
         "costs": {"q1": {(0.7, 0.7): _cost()}}},
    )
    shards_mod.write_shard(
        label,
        {"kind": "label", "name": "silver",
         "results": {"q1": _cache(label, "q1", labels)}},
    )

    point_shard = shards_mod.load_shards([point.output_dir])[0]
    label_shard = shards_mod.load_shards([label.output_dir])[0]
    metrics = shards_mod.score_shard(
        point_shard, label_shard, debug_root=tmp_path / "debug", record_telemetry=False
    )
    assert list(metrics["precision"]) == [pytest.approx(2 / 3)]
    assert list(metrics["recall"]) == [1.0]
    # The cardinality columns count rows, not distinct rows, and survive the reduction.
    assert list(metrics["predicted_output_cardinality"]) == [3]
    assert list(metrics["true_output_cardinality"]) == [2]


def _cost():
    from reasondb.executor import CostSummary
    from reasondb.query_plan.physical_operator import ProfilingCost

    return CostSummary(
        execution_cost=ProfilingCost(runtime=1.0, monetary_cost=0.0),
        tuning_cost=ProfilingCost(runtime=0.5, monetary_cost=0.0),
        component_times={"end_to_end": 2.0, "execution": 1.0},
    )


# ── the executor's result cache ──────────────────────────────────────────────


class _NoopComponent:
    def set_database(self, database):
        pass


def _executor(name="test-executor"):
    from reasondb.executor import Executor

    return Executor(
        database=_NoopComponent(), reasoner=_NoopComponent(),
        optimizer=_NoopComponent(), configurator=_NoopComponent(), name=name,
    )


def _cache_one(executor, cache_dir, query, answer):
    executor.cache_result(
        results_cache_dir=cache_dir, query_str=query, guarantees=(),
        results=answer, costs=_cost(), tuned_pipelines=["[]"],
        logger=executor.logger,
    )


def test_a_reducing_run_caches_signatures_and_never_a_frame(tmp_path):
    """The retry cache stores what scoring reads, not the answer it was reduced from.

    Reading the frame back with `read_parquet` inflates under a megabyte on disk to a
    gigabyte in pandas for a saturating self-join, when all anything downstream needs is
    the hashes.
    """
    from reasondb.query_plan.query import Query

    executor, cache_dir = _executor(), tmp_path / "cache"
    query = "Extract [title] from {r.text}"
    signature = RowSignature.of(pd.DataFrame({"title": ["a", "b"]}))
    _cache_one(executor, cache_dir, query, signature)

    assert [p.name.split("_", 1)[1] for p in cache_dir.glob("*.sig2.pkl")]
    assert not list(cache_dir.glob("*.parquet"))

    result = executor.execute_benchmark(
        [Query(query)], results_cache_dir=cache_dir, reduce_results=True
    )
    assert isinstance(result.results[query], RowSignature)
    assert np.array_equal(result.results[query].hashes, signature.hashes)
    # And the shard is told where it is, rather than being handed a second copy.
    assert result.result_paths[query].suffix == ".pkl"
    assert result.result_paths[query].is_file()


def test_a_cache_hit_scores_exactly_as_the_run_that_filled_it(tmp_path):
    """The key property of the reduction: identical metrics.

    Nothing is recomputed on a hit - the signature that comes back is the one the fresh
    run wrote - so this is a round-trip check on the pickling, not on the arithmetic.
    """
    from reasondb.evaluation.evaluation import evaluate
    from reasondb.query_plan.query import Query

    executor, cache_dir = _executor(), tmp_path / "cache"
    query = "Extract [title] from {r.text}"
    fresh = RowSignature.of(pd.DataFrame({"title": ["a", "b", "c"]}))
    labels = {query: RowSignature.of(pd.DataFrame({"title": ["a", "b"]}))}
    _cache_one(executor, cache_dir, query, fresh)

    from_cache = executor.execute_benchmark(
        [Query(query)], results_cache_dir=cache_dir, reduce_results=True
    ).results[query]

    scored = [
        evaluate("bench", "approach", {query: answer}, labels, {query: _cost()},
                 debug_root=tmp_path / f"dbg{i}", record_telemetry=False)
        for i, answer in enumerate((fresh, from_cache))
    ]
    for column in ("precision", "recall", "f1_score",
                   "predicted_output_cardinality", "true_output_cardinality"):
        assert list(scored[0][column]) == list(scored[1][column])
    assert list(scored[0]["precision"]) == [pytest.approx(2 / 3)]


def test_a_frame_cache_is_a_miss_for_a_reducing_caller(tmp_path):
    """The two forms get different extensions rather than a version field inside one.

    A directory written in the other mode is simply a miss - never a half-read entry
    handing a frame to code expecting hashes.
    """
    executor, cache_dir = _executor(), tmp_path / "cache"
    query = "Extract [title] from {r.text}"
    _cache_one(executor, cache_dir, query, pd.DataFrame({"title": ["a"]}))
    assert list(cache_dir.glob("*.parquet"))

    hit = executor.check_cached_results(
        results_cache_dir=cache_dir, query_str=query, guarantees=(),
        force_running_queries=[], logger=executor.logger, reduced=True,
    )
    assert hit is None


# --- which hash produced the numbers -------------------------------------------------


def test_two_schemes_are_refused_rather_than_scored():
    """The same row hashes differently under each, so a mixed pair scores ~0.

    Indistinguishable from a genuinely bad run, and nothing can be salvaged - hashes
    cannot be converted, only recomputed. So this raises where the `pandas_version`
    check beside it warns.
    """
    from reasondb.evaluation.evaluation import _score_signatures

    frame = pd.DataFrame({"a": ["x", "y"]})
    pandas_side, sql_side = RowSignature.of(frame), RowSignature.of(frame)
    sql_side.scheme = "sql-md5add-v1"

    assert _score_signatures("q", pandas_side, pandas_side)["precision"] == 1.0
    with pytest.raises(ValueError, match="not comparable"):
        _score_signatures("q", sql_side, pandas_side)


def test_a_signature_pickled_before_the_field_existed_says_pandas():
    """`scheme` is a dataclass field, so its default is also a class attribute.

    Unpickling restores `__dict__` without calling `__init__`, so a signature pickled
    without the field simply falls through to the class attribute - which is the right answer, since it
    really was hashed by the pandas scheme. It then fails loudly against a new one
    rather than being compared as if the two agreed.
    """
    signature = RowSignature.of(pd.DataFrame({"a": ["x"]}))
    del signature.__dict__["scheme"]
    restored = pickle.loads(pickle.dumps(signature))
    assert restored.scheme == "pandas-v1"
