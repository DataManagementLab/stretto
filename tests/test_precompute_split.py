"""``coordinator.producers.precompute_split``: one precompute job per modality.

What ``--split-both-capability-datasets`` has to get right, and what each test here pins:

- a dataset that needs one KV server is *unchanged* by the flag, so it can be left on for
  an invocation mixing ecommerce with the text-only datasets;
- each half asks for its own capability alone, which is the whole point - a
  ``--capability text`` and a ``--capability image`` node can then record one dataset side
  by side, rather than only a ``both`` node matching the job;
- the halves never write the mapped path (``SimulateStore.save`` rewrites its whole file
  per query with no locking, so two writers race), and are merged back into it only when
  *every* half is present and nothing conflicts.
"""

import json
import types

import pytest

from reasondb.coordinator.producers import precompute_split
from reasondb.coordinator.scheduler import worker_can_run
from reasondb.utils import precompute_modalities


def _benchmark_class(text=False, image=False, audio=False):
    """Only what ``capabilities_for_benchmark`` reads: the modalities the columns declare."""
    database = types.SimpleNamespace(
        external_tables=[
            types.SimpleNamespace(
                name="t",
                text_columns=["t.body"] if text else [],
                image_columns=["t.pic"] if image else [],
                audio_columns=["t.clip"] if audio else [],
            )
        ]
    )
    return types.SimpleNamespace(
        name=lambda: "fake_bench",
        load=lambda split: types.SimpleNamespace(database=database),
        load_without_queries=lambda split: types.SimpleNamespace(database=database),
    )


def _store(**buckets):
    base = {
        "text_qa": {},
        "vision": {},
        "precomputed_ops": [],
        "operator_configs": {},
        "filter_stats": {},
    }
    base.update(buckets)
    return base


def _write(path, data):
    path.write_text(json.dumps(data))
    return path


# ── plan_parts ───────────────────────────────────────────────────────────────


def test_flag_off_is_one_unsplit_part(tmp_path):
    (part,) = precompute_split.plan_parts(
        _benchmark_class(text=True, image=True), "dev", tmp_path / "ecomm.json", False
    )
    assert part.modality is None
    assert part.skip_modalities == []
    assert part.job_id_suffix == ""
    # The mapped file itself, so the job is identical to an unsplit one.
    assert part.store_path == str(tmp_path / "ecomm.json")
    assert set(part.required_capabilities) == {"embedding", "text_kv", "image_kv"}


def test_a_single_modality_dataset_is_unsplit_even_with_the_flag_on(tmp_path):
    """Nothing to split: the "other" half would ask for no KV capability and record
    nothing. This is what makes the flag safe on a --precompute naming ecommerce *and*
    the text-only datasets."""
    (part,) = precompute_split.plan_parts(
        _benchmark_class(text=True), "dev", tmp_path / "movie.json", True
    )
    assert part.modality is None
    assert part.n_parts == 1
    assert part.store_path == str(tmp_path / "movie.json")


def test_a_mixed_dataset_splits_into_one_job_per_modality(tmp_path):
    parts = precompute_split.plan_parts(
        _benchmark_class(text=True, image=True), "dev", tmp_path / "ecomm.json", True
    )
    assert [p.modality for p in parts] == ["text", "image"]
    assert [p.n_parts for p in parts] == [2, 2]
    assert [p.job_id_suffix for p in parts] == ["-text", "-image"]
    # Siblings of the mapped file, never the mapped file itself: two writers on one path
    # is a lost-update race, since a store rewrites its whole file after every query.
    assert [p.store_path for p in parts] == [
        str(tmp_path / "ecomm.text.json"),
        str(tmp_path / "ecomm.image.json"),
    ]
    assert all(p.base_path == str(tmp_path / "ecomm.json") for p in parts)


def test_each_half_asks_for_its_own_capability_alone(tmp_path):
    """The point of the flag. Unsplit, ecommerce's recording matches only a worker
    holding all four KV servers, and its text half queues behind its image half."""
    text_part, image_part = precompute_split.plan_parts(
        _benchmark_class(text=True, image=True), "dev", tmp_path / "ecomm.json", True
    )
    assert set(text_part.required_capabilities) == {"embedding", "text_kv"}
    assert set(image_part.required_capabilities) == {"embedding", "image_kv"}
    assert worker_can_run("text", text_part.required_capabilities, job_is_simulate=False)
    assert not worker_can_run("image", text_part.required_capabilities, job_is_simulate=False)
    assert worker_can_run("image", image_part.required_capabilities, job_is_simulate=False)
    assert not worker_can_run("text", image_part.required_capabilities, job_is_simulate=False)


def test_a_half_skips_every_modality_it_is_not_recording(tmp_path):
    """Not only the dataset's *other* modalities: PRECOMPUTE_SKIP_MODALITIES gates the
    operator toolbox, which carries every modality's operators regardless of which ones
    the benchmark has columns for."""
    text_part, image_part = precompute_split.plan_parts(
        _benchmark_class(text=True, image=True), "dev", tmp_path / "ecomm.json", True
    )
    assert set(text_part.skip_modalities) == {"image", "audio"}
    assert set(image_part.skip_modalities) == {"text", "audio"}


def test_three_modalities_split_three_ways(tmp_path):
    parts = precompute_split.plan_parts(
        _benchmark_class(text=True, image=True, audio=True), "dev", tmp_path / "x.json", True
    )
    assert [p.modality for p in parts] == ["text", "image", "audio"]
    assert all(p.n_parts == 3 for p in parts)


def test_spec_keys_are_written_for_the_unsplit_case_too(tmp_path):
    """The spec schema must not branch: an ordinary precompute job carries
    ``precompute_modality: None``, which the run-configuration panel reads as a fixed
    axis ("all"), rather than no key at all, which reads as no such axis."""
    (unsplit,) = precompute_split.plan_parts(
        _benchmark_class(text=True), "dev", tmp_path / "movie.json", True
    )
    keys = precompute_split.spec_keys(unsplit)
    assert keys["precompute_modality"] is None
    assert keys["precompute_skip_modalities"] == []
    assert keys["precompute_split_parts"] == 1
    assert keys["precompute_base_path"] == keys["precompute_path"]


# ── skipping_modalities ──────────────────────────────────────────────────────


def test_skipping_modalities_applies_and_restores(monkeypatch):
    """Assigned as a module attribute, because the env var is parsed once at import and
    a worker claims its jobs long after that. Restored because one worker process runs
    many jobs and a leaked skip set would make the next one record nothing."""
    monkeypatch.setattr(precompute_modalities, "PRECOMPUTE_SKIP_MODALITIES", frozenset())
    spec = {"precompute_modality": "text", "precompute_skip_modalities": ["image", "audio"]}
    with precompute_split.skipping_modalities(spec):
        assert precompute_modalities.PRECOMPUTE_SKIP_MODALITIES == {"image", "audio"}
    assert precompute_modalities.PRECOMPUTE_SKIP_MODALITIES == frozenset()


def test_skipping_modalities_restores_after_a_failure(monkeypatch):
    monkeypatch.setattr(precompute_modalities, "PRECOMPUTE_SKIP_MODALITIES", frozenset())
    spec = {"precompute_modality": "text", "precompute_skip_modalities": ["image"]}
    with pytest.raises(RuntimeError):
        with precompute_split.skipping_modalities(spec):
            raise RuntimeError("job died")
    assert precompute_modalities.PRECOMPUTE_SKIP_MODALITIES == frozenset()


def test_an_unsplit_job_keeps_whatever_the_environment_set(monkeypatch):
    """The env var is still the hand-run escape hatch documented on --precompute; an
    ordinary job must not silently clear it."""
    monkeypatch.setattr(
        precompute_modalities, "PRECOMPUTE_SKIP_MODALITIES", frozenset({"image"})
    )
    with precompute_split.skipping_modalities({"precompute_modality": None}):
        assert precompute_modalities.PRECOMPUTE_SKIP_MODALITIES == {"image"}


# ── seed_store ───────────────────────────────────────────────────────────────


def test_seed_carries_markers_configs_and_stats_but_not_responses(tmp_path):
    """Metadata only, deliberately. A precompute pass never *reads* a recorded response
    (backends look up the simulate store and only record into the precompute one), so
    copying the base's records into every half would put three copies on disk and make
    each half's after-every-query rewrite as expensive as the base's. The three buckets
    it does carry are the ones a half cannot work without: without the markers a top-up
    re-records everything, without the pins it re-derives phrasing."""
    base = _write(
        tmp_path / "ecomm.json",
        _store(
            text_qa={"model-a": {"h1": {"question": "q", "response": "yes"}}},
            precomputed_ops=["TextQaFilter|expr|_T0"],
            operator_configs={"TextQaFilter|expr|_T0": {"question_template": "q"}},
            filter_stats={"ecommerce_random": {"dev": {"matrix": [[1]]}}},
        ),
    )
    half = tmp_path / "ecomm.text.json"

    precompute_split.seed_store(
        {"precompute_path": str(half), "precompute_base_path": str(base)}
    )

    seeded = json.loads(half.read_text())
    assert seeded["precomputed_ops"] == ["TextQaFilter|expr|_T0"]
    assert seeded["operator_configs"] == {"TextQaFilter|expr|_T0": {"question_template": "q"}}
    assert seeded["filter_stats"] == {"ecommerce_random": {"dev": {"matrix": [[1]]}}}
    assert seeded["text_qa"] == {}


def test_seed_keeps_a_half_own_pin_and_records(tmp_path):
    """A half resumed after a crash already recorded responses under its own pin; the
    base's pin must not overwrite it, or those recordings become unreachable. The merge
    reports the disagreement instead."""
    base = _write(
        tmp_path / "ecomm.json",
        _store(operator_configs={"TextQaFilter|expr|_T0": {"question_template": "base"}}),
    )
    half = _write(
        tmp_path / "ecomm.text.json",
        _store(
            text_qa={"model-a": {"h1": {"question": "half", "response": "yes"}}},
            operator_configs={"TextQaFilter|expr|_T0": {"question_template": "half"}},
        ),
    )

    precompute_split.seed_store(
        {"precompute_path": str(half), "precompute_base_path": str(base)}
    )

    seeded = json.loads(half.read_text())
    assert seeded["operator_configs"]["TextQaFilter|expr|_T0"] == {"question_template": "half"}
    assert seeded["text_qa"]["model-a"]["h1"]["question"] == "half"


def test_seeding_is_a_no_op_for_an_unsplit_job_and_a_missing_base(tmp_path):
    half = tmp_path / "only.json"
    precompute_split.seed_store(
        {"precompute_path": str(half), "precompute_base_path": str(half)}
    )
    precompute_split.seed_store(
        {"precompute_path": str(half), "precompute_base_path": str(tmp_path / "nope.json")}
    )
    assert not half.exists()


# ── manifests and the merge ──────────────────────────────────────────────────


def _job(output_dir):
    return types.SimpleNamespace(benchmark="ecommerce_random", split="dev", output_dir=str(output_dir))


def _half(tmp_path, modality, n_parts=2, **buckets):
    """One finished half: its store beside the base, its manifest in its job directory."""
    store = _write(tmp_path / f"ecomm.{modality}.json", _store(**buckets))
    job_dir = tmp_path / f"job_{modality}"
    job_dir.mkdir()
    precompute_split.write_manifest(
        job_dir,
        _job(job_dir),
        {
            "precompute_path": str(store),
            "precompute_modality": modality,
            "precompute_base_path": str(tmp_path / "ecomm.json"),
            "precompute_split_parts": n_parts,
        },
    )
    return str(job_dir)


def test_an_unsplit_job_writes_no_manifest(tmp_path):
    """The "a directory with no sidecar is skipped" convention: an ordinary precompute
    task's merge stays a no-op glob."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    assert (
        precompute_split.write_manifest(
            job_dir, _job(job_dir), {"precompute_modality": None}
        )
        is None
    )
    assert precompute_split.merge_completed_splits([str(job_dir)]) == []


def test_both_halves_are_merged_into_the_mapped_file(tmp_path):
    base = _write(
        tmp_path / "ecomm.json",
        _store(operator_configs={"shared|expr|_T0": {"question_template": "q"}}),
    )
    text_dir = _half(
        tmp_path,
        "text",
        text_qa={"model-a": {"h1": {"question": "q1", "response": "yes"}}},
        precomputed_ops=["TextQaFilter|expr|_T0"],
    )
    image_dir = _half(
        tmp_path,
        "image",
        vision={"model-b": {"h2": {"question": "q2", "response": "no"}}},
        precomputed_ops=["ImageQaFilter|expr|_T0"],
    )

    written = precompute_split.merge_completed_splits([text_dir, image_dir])

    assert written == [base]
    merged = json.loads(base.read_text())
    assert merged["text_qa"] == {"model-a": {"h1": {"question": "q1", "response": "yes"}}}
    assert merged["vision"] == {"model-b": {"h2": {"question": "q2", "response": "no"}}}
    assert merged["precomputed_ops"] == ["ImageQaFilter|expr|_T0", "TextQaFilter|expr|_T0"]
    # The base's own pins survive: they are what the phase-0 filter-stats job wrote and
    # what both halves were seeded from.
    assert merged["operator_configs"]["shared|expr|_T0"] == {"question_template": "q"}


def test_a_missing_half_is_not_merged(tmp_path):
    """merge_task only passes *done* jobs' directories, so a group short of a half means
    a half that failed or is still writing. Writing the base then would put a store that
    looks complete at the path --simulate reads."""
    base = _write(tmp_path / "ecomm.json", _store())
    text_dir = _half(tmp_path, "text", precomputed_ops=["TextQaFilter|expr|_T0"])

    assert precompute_split.merge_completed_splits([text_dir]) == []
    assert json.loads(base.read_text())["precomputed_ops"] == []


def test_a_conflicting_pin_leaves_the_base_untouched(tmp_path):
    """Two halves that derived different question phrasing for one operator: merging past
    that would leave a store whose pins disagree with the responses recorded under them,
    which surfaces as a --simulate cache miss hours into a later sweep."""
    base = _write(tmp_path / "ecomm.json", _store())
    text_dir = _half(
        tmp_path, "text", operator_configs={"Shared|expr|_T0": {"question_template": "a"}}
    )
    image_dir = _half(
        tmp_path, "image", operator_configs={"Shared|expr|_T0": {"question_template": "b"}}
    )

    assert precompute_split.merge_completed_splits([text_dir, image_dir]) == []
    assert json.loads(base.read_text()) == _store()
    # The halves are intact, so scripts/merge_precompute.py --force can still finish it.
    assert (tmp_path / "ecomm.text.json").is_file()
    assert (tmp_path / "ecomm.image.json").is_file()


def test_a_re_run_half_does_not_stand_in_for_the_missing_one(tmp_path):
    """Two manifests for the same modality (a retried half writes its own output
    directory) must not satisfy a two-part group."""
    _write(tmp_path / "ecomm.json", _store())
    first = _half(tmp_path, "text")
    retry_dir = tmp_path / "job_text_retry"
    retry_dir.mkdir()
    precompute_split.write_manifest(
        retry_dir,
        _job(retry_dir),
        {
            "precompute_path": str(tmp_path / "ecomm.text.json"),
            "precompute_modality": "text",
            "precompute_base_path": str(tmp_path / "ecomm.json"),
            "precompute_split_parts": 2,
        },
    )

    assert precompute_split.merge_completed_splits([first, str(retry_dir)]) == []


def test_the_merge_leaves_no_temporary_file_behind(tmp_path):
    """Written aside and renamed, because the base path is what --simulate reads and a
    merge interrupted mid-dump would otherwise truncate it."""
    _write(tmp_path / "ecomm.json", _store())
    dirs = [_half(tmp_path, "text"), _half(tmp_path, "image")]

    precompute_split.merge_completed_splits(dirs)

    assert not list(tmp_path.glob("*.merge-tmp"))


def test_a_merge_without_an_existing_base_still_writes_one(tmp_path):
    """The mapped file need not exist: a first recording of a dataset has only halves."""
    dirs = [
        _half(tmp_path, "text", precomputed_ops=["TextQaFilter|expr|_T0"]),
        _half(tmp_path, "image", precomputed_ops=["ImageQaFilter|expr|_T0"]),
    ]

    (written,) = precompute_split.merge_completed_splits(dirs)

    assert written == tmp_path / "ecomm.json"
    assert len(json.loads(written.read_text())["precomputed_ops"]) == 2


# ── the CLI guard ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        # A producer that has no precompute path at all.
        ["--task-id", "t1", "--producer", "baselines", "--split-both-capability-datasets"],
        # The right producer, but a replaying run: nothing to split.
        [
            "--task-id",
            "t1",
            "--producer",
            "parameter_sweep",
            "--split-both-capability-datasets",
        ],
    ],
)
def test_the_flag_is_rejected_where_it_would_do_nothing(argv, monkeypatch):
    """Rejected rather than ignored, for the same reason ``_PRECOMPUTE_PRODUCERS`` and
    ``--precompute-states`` are: a flag that reads as configuration and quietly changes
    nothing looks like success until the run it was supposed to shape turns out not to
    have been shaped."""
    run_coordinator = pytest.importorskip("scripts.run_coordinator")
    monkeypatch.setattr("sys.argv", ["run_coordinator.py", *argv])
    with pytest.raises(AssertionError, match="split-both-capability-datasets"):
        run_coordinator.main()
