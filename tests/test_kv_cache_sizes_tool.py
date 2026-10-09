"""``kv_cache_sizes.py`` as a standalone tool: it runs from a copy, and it deletes safely.

The file is copied to machines whose checkout cannot be updated, so three things are
pinned here that nothing else would catch:

- the facts it mirrors from the package (the spec grid, the generator tags, the sweep
  benchmarks and where their caches live) still agree with the package;
- a copy of the file runs with no site-packages at all (``python -I -S``), where only the
  standard library exists - so a stray package import is a test failure here, not a
  surprise on a server whose installed ``reasondb`` is an older version;
- the only files it creates or removes are the recording (and its lock and temporary
  file) and, for ``delete``, the named level directories. Audited with a Python audit
  hook on every command.

And the deletion itself: every refusal happens before anything is removed, and what is
left afterwards is exactly what a replay needs.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from reasondb.evaluation import kv_cache_sizes as kvs

SOURCE = Path(kvs.__file__)
PRESS = kvs.DEFAULT_PRESS
M8 = "meta-llama/Llama-3.1-8B-Instruct"
M70 = "meta-llama/Llama-3.1-70B-Instruct"
TEXT_LEVELS = {M8: (0.0, 0.5, 0.8), M70: (0.3, 0.6, 0.8)}
BENCH = "movie_random_huge"


# ── The mirrored facts ───────────────────────────────────────────────────────


def test_the_spec_grid_mirrors_the_operator_spec_tables():
    from reasondb.interface.default_operator_toolbox import IMAGE_SPECS, TEXT_SPECS

    expected = {}
    for modality, specs in (("text", TEXT_SPECS), ("image", IMAGE_SPECS)):
        for spec in specs:
            if not spec.vanilla:
                expected.setdefault(spec.model, (modality, set()))[1].add(spec.effective_cr)
    assert {m: (mod, set(r)) for m, (mod, r) in kvs.SPEC_GRID.items()} == expected
    # Sorted, since the spec slots the package plans from list their ratios that way.
    assert all(list(r) == sorted(r) for _mod, r in kvs.SPEC_GRID.values())


def test_the_method_tags_mirror_the_model_registry():
    from reasondb.config.model_registry import ModelRegistry

    registry = ModelRegistry.get().method_config(None)
    for level in kvs.spec_levels():
        assert registry[level.method] == (level.model, level.ratio), level


def test_the_benchmark_list_mirrors_the_registry():
    from reasondb.evaluation.benchmark_registry import RANDOM_BENCHMARKS

    assert set(kvs.RANDOM_BENCHMARKS) == {cls.name() for cls in RANDOM_BENCHMARKS.values()}


@pytest.mark.parametrize("benchmark", kvs.RANDOM_BENCHMARKS)
def test_the_cache_layout_mirrors_the_database(benchmark, monkeypatch, tmp_path):
    from reasondb.evaluation.benchmark_registry import RANDOM_BENCHMARKS
    from reasondb.utils.cache import CACHE_DIR

    try:
        database = RANDOM_BENCHMARKS[benchmark].load_without_queries("dev").database
    except Exception as error:  # a dataset absent from this machine
        pytest.skip(f"{benchmark} cannot be loaded here: {type(error).__name__}")
    assert database.cache_dir == kvs.benchmark_dir(CACHE_DIR, benchmark, "dev")


# ── A fixture cache ──────────────────────────────────────────────────────────


def _level_dir(cache_root, model, ratio, benchmark=BENCH):
    return (
        kvs.benchmark_dir(cache_root, benchmark, "dev")
        / "kv-text-qa-cache" / model / PRESS / f"comp{kvs.to_compression_tag(ratio)}"
    )


@pytest.fixture
def cache_root(tmp_path):
    root = tmp_path / "shared" / "cache"
    for model, ratios in TEXT_LEVELS.items():
        for i, ratio in enumerate(ratios):
            directory = _level_dir(root, model, ratio)
            directory.mkdir(parents=True)
            for item in range(3):
                (directory / f"cache_entry_{item}.pt").write_bytes(b"x" * (100 * (i + 1)))
    bench = kvs.benchmark_dir(root, BENCH, "dev")
    (bench / "embedding_cache").mkdir()
    (bench / "embedding_cache" / "vectors.bin").write_bytes(b"keep")
    (bench / "reviews.db").write_bytes(b"keep")
    return root


@pytest.fixture
def store(tmp_path):
    return _store(tmp_path, [(m, r) for m, rs in TEXT_LEVELS.items() for r in rs])


def _store(tmp_path, levels, name="store.json"):
    text_qa = {f"{m}-cr{r}": {"k": {}} for m, r in levels}
    text_qa[f"{M70}-cr0.0-vanilla"] = {"k": {}}
    path = tmp_path / name
    path.write_text(json.dumps({"text_qa": text_qa, "vision": {}}))
    return path


@pytest.fixture
def recording(tmp_path, cache_root):
    path = tmp_path / "home" / "sizes.json"
    root = kvs.benchmark_dir(cache_root, BENCH, "dev")
    kvs.record(BENCH, "dev", PRESS, kvs.scan_models(root, PRESS), root, path)
    return path


def _levels(*tags):
    return kvs.resolve_levels(tags)


def _delete(cache_root, recording, store, *tags, **kwargs):
    kwargs.setdefault("recording_sha256", kvs.file_sha256(recording))
    return kvs.delete_levels(
        cache_root, BENCH, "dev", PRESS, _levels(*tags),
        recording=recording, store=store, **kwargs,
    )


def _all_present(cache_root):
    return all(
        _level_dir(cache_root, m, r).is_dir() for m, rs in TEXT_LEVELS.items() for r in rs
    )


# ── Refusals, all before anything is removed ─────────────────────────────────


def test_a_deletion_needs_the_digest_of_the_saved_recording(cache_root, recording, store):
    with pytest.raises(kvs.PruneRefused, match="recording-sha256"):
        _delete(cache_root, recording, store, "kv8B05", recording_sha256="0" * 64)
    assert _all_present(cache_root)


def test_a_deletion_never_records_so_an_unrecorded_level_is_refused(tmp_path, cache_root, store):
    """The recording must hold the level *before* the run, so the copy saved elsewhere
    is guaranteed to hold it too."""
    recording = tmp_path / "home" / "sizes.json"
    root = kvs.benchmark_dir(cache_root, BENCH, "dev")
    scans = kvs.scan_models(root, PRESS)
    scans[M8][0.5] = kvs.LevelScan(1, 1, {})  # recorded, but smaller than the disk
    kvs.record(BENCH, "dev", PRESS, scans, root, recording)

    with pytest.raises(kvs.PruneRefused, match="kv8B05: the recording holds less"):
        _delete(cache_root, recording, store, "kv8B05")
    assert _all_present(cache_root)


def test_a_recording_missing_part_of_the_grid_is_refused(tmp_path, cache_root, store):
    recording = tmp_path / "home" / "sizes.json"
    root = kvs.benchmark_dir(cache_root, BENCH, "dev")
    kvs.record(BENCH, "dev", PRESS, {M8: kvs.scan_models(root, PRESS)[M8]}, root, recording)

    with pytest.raises(kvs.PruneRefused, match="does not cover every level"):
        _delete(cache_root, recording, store, "kv8B05")
    assert _all_present(cache_root)


def test_a_level_the_store_cannot_replay_is_refused(tmp_path, cache_root, recording):
    """All or nothing: kv8B05 was fine, and it stays."""
    store = _store(tmp_path, [(M8, 0.5)])

    with pytest.raises(kvs.PruneRefused, match="kv8B00: the store has no responses"):
        _delete(cache_root, recording, store, "kv8B00", "kv8B05")
    assert _all_present(cache_root)


def test_a_deletion_without_a_store_is_refused(cache_root, recording):
    with pytest.raises(kvs.PruneRefused, match="no --store"):
        _delete(cache_root, recording, None, "kv8B05")
    assert _all_present(cache_root)


def test_an_unreadable_recording_is_refused(tmp_path, cache_root, store):
    recording = tmp_path / "sizes.json"
    recording.write_text("{ not json")

    with pytest.raises(kvs.UnreadableRecording):
        _delete(cache_root, recording, store, "kv8B05")
    assert _all_present(cache_root)


def test_a_wrong_cache_root_is_refused(tmp_path, recording, store):
    with pytest.raises(kvs.PruneRefused, match="does not exist"):
        _delete(tmp_path / "elsewhere", recording, store, "kv8B05")


def test_a_level_no_sweep_asks_for_cannot_be_named():
    with pytest.raises(ValueError, match="kv8B03: not a level any sweep can ask for"):
        _levels("kv8B03")


def test_a_dry_run_checks_everything_but_the_digest_and_writes_nothing(
    cache_root, recording, store
):
    before = kvs.file_sha256(recording)

    _statuses, targets = _delete(
        cache_root, recording, store, "kv8B05", "kv70B08",
        recording_sha256=None, dry_run=True,
    )

    assert [level.method for level, _d, _b in targets] == ["kv8B05", "kv70B08"]
    assert sum(size for _l, _d, size in targets) == 3 * 200 + 3 * 300
    assert _all_present(cache_root)
    assert kvs.file_sha256(recording) == before


# ── A deletion ───────────────────────────────────────────────────────────────


def test_only_the_named_level_directories_go(cache_root, recording, store):
    _statuses, targets = _delete(cache_root, recording, store, "kv8B05", "kv70B08")

    assert not _level_dir(cache_root, M8, 0.5).exists()
    assert not _level_dir(cache_root, M70, 0.8).exists()
    for model, ratios in TEXT_LEVELS.items():
        for ratio in ratios:
            if (model, ratio) not in {(M8, 0.5), (M70, 0.8)}:
                assert _level_dir(cache_root, model, ratio).is_dir()
    bench = kvs.benchmark_dir(cache_root, BENCH, "dev")
    assert (bench / "embedding_cache" / "vectors.bin").read_bytes() == b"keep"
    assert (bench / "reviews.db").read_bytes() == b"keep"


def test_a_deletion_leaves_the_recording_byte_for_byte(cache_root, recording, store):
    before = recording.read_bytes()
    _delete(cache_root, recording, store, "kv8B05")
    assert recording.read_bytes() == before


def test_the_status_afterwards_lists_what_to_regenerate(cache_root, recording, store):
    statuses, _targets = _delete(cache_root, recording, store, "kv8B05", "kv70B08")

    assert {s.level.method for s in statuses if s.deleted} == {"kv8B05", "kv70B08"}
    assert all(s.recording_holds_disk for s in statuses if s.materialized)


def test_a_level_already_gone_is_not_a_target(cache_root, recording, store):
    _delete(cache_root, recording, store, "kv8B05")
    _statuses, targets = _delete(cache_root, recording, store, "kv8B05")
    assert targets == []


def test_a_replay_plans_every_level_from_the_recording_afterwards(
    cache_root, recording, store, monkeypatch
):
    """Through the package, the way a sweep reads it."""
    import argparse
    from types import SimpleNamespace

    from reasondb.evaluation import parameter_sweep as psweep

    monkeypatch.setenv(kvs.SIZES_PATH_ENV, str(recording))
    bench = SimpleNamespace(
        name=lambda: BENCH,
        database=SimpleNamespace(cache_dir=kvs.benchmark_dir(cache_root, BENCH, "dev")),
    )
    slots = [
        psweep.ModelSlot("text_small", "text", M8, False),
        psweep.ModelSlot("text_large", "text", M70, True),
    ]
    live = psweep.resolve_level_scans(
        bench, argparse.Namespace(split="dev", press_name=PRESS, simulate=None), slots
    )
    _delete(cache_root, recording, store, "kv8B05", "kv70B08")

    replayed = psweep.resolve_level_scans(
        bench,
        argparse.Namespace(split="dev", press_name=PRESS, simulate=[Path("store.json")]),
        slots,
    )
    assert replayed == live


def test_a_real_run_afterwards_does_not_erase_the_recording(cache_root, recording, store, monkeypatch):
    import argparse
    from types import SimpleNamespace

    from reasondb.evaluation import parameter_sweep as psweep

    monkeypatch.setenv(kvs.SIZES_PATH_ENV, str(recording))
    _delete(cache_root, recording, store, "kv8B05")
    bench = SimpleNamespace(
        name=lambda: BENCH,
        database=SimpleNamespace(cache_dir=kvs.benchmark_dir(cache_root, BENCH, "dev")),
    )
    psweep.resolve_level_scans(
        bench,
        argparse.Namespace(split="dev", press_name=PRESS, simulate=None),
        [psweep.ModelSlot("text_small", "text", M8, False)],
    )
    assert kvs.load_recorded(BENCH, "dev", PRESS, recording)[M8][0.5].bytes == 3 * 200


# ── merge ────────────────────────────────────────────────────────────────────


def test_merge_only_ever_grows_and_keeps_where_numbers_came_from(tmp_path, cache_root, recording):
    target = tmp_path / "repo" / "kv_cache_sizes.json"
    kvs.record(BENCH, "dev", PRESS, {M8: {0.5: kvs.LevelScan(10_000, 99, {})}}, Path("/other"), target)
    kvs.record("rotowire_random", "dev", PRESS, {M8: {0.8: kvs.LevelScan(5, 1, {})}}, Path("/r"), target)

    assert kvs.merge_recordings(recording, target)

    merged = kvs.load_recorded(BENCH, "dev", PRESS, target)
    assert merged[M8][0.5] == kvs.LevelScan(10_000, 99, {})  # larger, kept
    assert merged[M8][0.0].bytes == 3 * 100  # new, taken
    assert kvs.load_recorded("rotowire_random", "dev", PRESS, target)  # untouched, kept
    sources = json.loads(target.read_text())["benchmarks"][BENCH]["dev"][PRESS]["sources"]
    assert sources[M70]["cache_root"] == str(kvs.benchmark_dir(recording.parents[1] / "shared" / "cache", BENCH, "dev"))
    assert not kvs.merge_recordings(recording, target)  # a second merge changes nothing


# ── The copied file, run in isolation ────────────────────────────────────────

AUDIT = r"""
import os, runpy, sys
log = open(os.environ["AUDIT_LOG"], "a")
WRITE = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
def hook(event, args):
    if event == "open" and args[0] is not None and not isinstance(args[0], int):
        mode, flags = args[1], args[2]
        if (mode and any(c in mode for c in "wax+")) or (flags & WRITE):
            log.write(f"write {os.fspath(args[0])}\n")
    elif event in ("os.remove", "os.unlink", "os.rmdir", "os.mkdir", "os.rename", "os.replace", "shutil.rmtree"):
        log.write(f"{event} {os.fspath(args[0])}\n")
    log.flush()
sys.addaudithook(hook)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
"""


def _run_copy(tmp_path, *args):
    """Run a copy of the file with the standard library and nothing else."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    copy = home / "kv_cache_sizes.py"
    shutil.copyfile(SOURCE, copy)
    audit = tmp_path / "audit.log"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "TMPDIR": str(home),
        "AUDIT_LOG": str(audit),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", AUDIT, str(copy), *map(str, args)],
        capture_output=True, text=True, env=env, cwd=home,
    )
    writes = audit.read_text().splitlines() if audit.exists() else []
    audit.unlink(missing_ok=True)
    return result, writes


def test_the_copy_runs_on_the_standard_library_alone(tmp_path):
    result, _writes = _run_copy(tmp_path, "--help")
    assert result.returncode == 0, result.stderr
    probe = subprocess.run(
        [sys.executable, "-I", "-S", "-c", "import reasondb"], capture_output=True, cwd=tmp_path
    )
    assert probe.returncode != 0, "without site-packages the package must be unreachable"


def test_the_whole_procedure_through_the_copy_writes_only_where_it_should(
    tmp_path, cache_root, store
):
    """record -> status -> dry run -> delete -> merge, audited for every write."""
    sizes = tmp_path / "home" / "sizes.json"
    common = ["--cache-root", cache_root, "--recording", sizes, "--benchmarks", BENCH]

    result, writes = _run_copy(tmp_path, "record", *common)
    assert result.returncode == 0, result.stderr
    # The recording, its lock and its temporary file - and an existence check on the
    # directory they live in (``mkdir(exist_ok=True)``), which creates nothing here.
    assert writes
    for event in writes:
        name, path = event.split(" ", 1)
        assert path.startswith(str(sizes)) or (name == "os.mkdir" and path == str(sizes.parent)), event
    digest = kvs.file_sha256(sizes)
    assert f"sha256 {digest}" in result.stdout

    result, writes = _run_copy(tmp_path, "status", *common, "--store", f"{BENCH}={store}")
    assert result.returncode == 0, result.stderr
    assert writes == []
    assert "kv8B05" in result.stdout and "on disk" in result.stdout

    dry = ["delete", *common, "--store", f"{BENCH}={store}", "--levels", "kv8B05", "kv70B08"]
    result, writes = _run_copy(tmp_path, *dry, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert writes == []
    assert f"--recording-sha256 {digest}" in result.stdout
    assert _all_present(cache_root)

    result, writes = _run_copy(tmp_path, *dry)  # no digest
    assert result.returncode == 1 and "REFUSED" in result.stderr
    assert writes == [] and _all_present(cache_root)

    result, writes = _run_copy(tmp_path, *dry, "--recording-sha256", digest)
    assert result.returncode == 0, result.stderr
    deleted = {str(_level_dir(cache_root, M8, 0.5)), str(_level_dir(cache_root, M70, 0.8))}
    # ``shutil.rmtree`` removes a tree through directory file descriptors, so the entries
    # inside it are audited by bare name. Every absolute path touched is one of the two
    # levels, and every bare name is a file that lived in one of them.
    absolute = {w.split(" ", 1)[1] for w in writes if w.split(" ", 1)[1].startswith("/")}
    relative = {w.split(" ", 1)[1] for w in writes if not w.split(" ", 1)[1].startswith("/")}
    assert absolute == deleted, writes
    assert relative <= {f"cache_entry_{i}.pt" for i in range(3)}, writes
    assert "to regenerate: --kv-methods" in result.stdout

    target = tmp_path / "repo" / "kv_cache_sizes.json"
    target.parent.mkdir()
    result, writes = _run_copy(tmp_path, "merge", sizes, "--into", target)
    assert result.returncode == 0, result.stderr
    for event in writes:
        name, path = event.split(" ", 1)
        assert path.startswith(str(target)) or (name == "os.mkdir" and path == str(target.parent)), event
    assert kvs.load_recorded(BENCH, "dev", PRESS, target)[M8][0.5].bytes == 3 * 200


def test_status_without_a_cache_root_says_so(tmp_path):
    result, _writes = _run_copy(tmp_path, "status")
    assert result.returncode != 0
    assert "--cache-root" in result.stderr
