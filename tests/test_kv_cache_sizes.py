"""Recorded KV cache sizes: a replay plans from the recording, not from the disk.

A ``--simulate`` sweep never reads a KV cache, but it still needs to know which levels
exist and what they cost - which the storage walk answers.
``reasondb.evaluation.kv_cache_sizes`` records that walk. What these tests
pin is the part that has to be exactly right for a replay to be trusted: the recording
rebuilds the same footprint the disk would have given, in either serving mode; a replay
reads it and nothing else; and a machine without caches never records itself as the
truth about a benchmark.

The autouse fixture in ``conftest.py`` already points the recording at a per-test file.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    from reasondb.evaluation import kv_cache_sizes
    from reasondb.evaluation import parameter_sweep as psweep
except ImportError:  # pragma: no cover - the guard the sibling storage tests use
    pytest.skip("parameter_sweep deps not installed", allow_module_level=True)

from reasondb.interface.default_operator_toolbox import TEXT_MODEL_8B, TEXT_MODEL_70B

LevelScan = kv_cache_sizes.LevelScan
PRESS = "expected_attention"


def _write(path, nbytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * nbytes)


def _level_dir(root, model, cr_tag):
    return root / "bench_dev" / "kv-text-qa-cache" / model / PRESS / f"comp{cr_tag}"


# ── The scan and what a serving mode makes of it ─────────────────────────────


def test_a_scan_keeps_each_relative_index_apart(tmp_path):
    base = _level_dir(tmp_path, TEXT_MODEL_8B, "0")
    _write(base / "cache_entry_a.pt", 1000)
    _write(base / "cache_entry_b.pt", 1000)
    _write(base / "indices" / "comp0_5" / "idx_a.pt", 30)
    _write(base / "indices" / "comp0_5" / "_meta.json", 20)
    _write(base / "indices" / "comp0_8" / "idx_a.pt", 10)

    scan = psweep.scan_level(tmp_path, 0.0, PRESS, TEXT_MODEL_8B)

    assert scan == LevelScan(2000, 2, {0.5: 50, 0.8: 10})


def test_a_serving_mode_decides_which_indices_a_footprint_carries(tmp_path):
    """The reason the raw scan is what gets recorded: the same disk costs two different
    amounts depending on whether the run reaches for the indices."""
    scan = LevelScan(2000, 2, {0.5: 50, 0.8: 10})

    assert scan.size(None) == (2000, 2)
    assert scan.size(frozenset({0.5})) == (2050, 2)
    assert scan.size(frozenset({0.5, 0.8})) == (2060, 2)


@pytest.mark.parametrize(
    "allowed", [None, frozenset(), frozenset({0.5}), frozenset({0.5, 0.8, 0.9})]
)
def test_measure_level_is_still_the_scan_under_one_mode(tmp_path, allowed):
    base = _level_dir(tmp_path, TEXT_MODEL_8B, "0")
    _write(base / "cache_entry_a.pt", 1000)
    _write(base / "indices" / "comp0_5" / "idx_a.pt", 30)
    _write(base / "indices" / "comp0_8" / "deeper" / "idx_b.pt", 7)

    assert psweep.measure_level(
        tmp_path, 0.0, PRESS, TEXT_MODEL_8B, allowed
    ) == psweep.scan_level(tmp_path, 0.0, PRESS, TEXT_MODEL_8B).size(allowed)


# ── The recording ────────────────────────────────────────────────────────────


def test_a_recording_round_trips_exactly(tmp_path):
    scans = {TEXT_MODEL_8B: {0.0: LevelScan(2000, 2, {0.5: 50}), 0.5: LevelScan(0, 0, {})}}

    assert kv_cache_sizes.record("bench", "dev", PRESS, scans, tmp_path)
    assert kv_cache_sizes.load_recorded("bench", "dev", PRESS) == scans


def test_recording_the_same_numbers_again_leaves_the_file_alone(tmp_path):
    """A live walk records on every job, so an unchanged walk must not rewrite the file."""
    scans = {TEXT_MODEL_8B: {0.0: LevelScan(2000, 2, {})}}
    kv_cache_sizes.record("bench", "dev", PRESS, scans, tmp_path)

    assert not kv_cache_sizes.record("bench", "dev", PRESS, scans, tmp_path)


def test_recording_one_model_keeps_the_others(tmp_path):
    kv_cache_sizes.record(
        "bench", "dev", PRESS, {TEXT_MODEL_8B: {0.0: LevelScan(1, 1, {})}}, tmp_path
    )
    kv_cache_sizes.record(
        "bench", "dev", PRESS, {TEXT_MODEL_70B: {0.3: LevelScan(2, 1, {})}}, tmp_path
    )

    assert set(kv_cache_sizes.load_recorded("bench", "dev", PRESS)) == {
        TEXT_MODEL_8B, TEXT_MODEL_70B,
    }


def test_a_recording_says_where_its_numbers_came_from(tmp_path):
    kv_cache_sizes.record(
        "bench", "dev", PRESS, {TEXT_MODEL_8B: {0.0: LevelScan(1, 1, {})}}, tmp_path
    )

    payload = json.loads(kv_cache_sizes.sizes_path().read_text())
    source = payload["benchmarks"]["bench"]["dev"][PRESS]["sources"][TEXT_MODEL_8B]
    assert source["cache_root"] == str(tmp_path)
    assert set(source) == {"host", "cache_root", "measured"}


def test_a_recording_in_another_format_is_ignored_rather_than_misread(tmp_path):
    kv_cache_sizes.sizes_path().write_text(json.dumps({"format": 0, "benchmarks": {}}))

    assert kv_cache_sizes.load_recorded("bench", "dev", PRESS) == {}


def test_nothing_recorded_is_an_empty_answer(tmp_path):
    assert kv_cache_sizes.load_recorded("bench", "dev", PRESS) == {}


# ── resolve_level_scans: who is asked, and when ─────────────────────────────


SLOTS = [
    psweep.ModelSlot("text_small", "text", TEXT_MODEL_8B, large=False),
    psweep.ModelSlot("text_large", "text", TEXT_MODEL_70B, large=True),
]


def _benchmark(cache_root):
    return SimpleNamespace(
        name=lambda: "bench", database=SimpleNamespace(cache_dir=cache_root)
    )


def _args(simulate):
    return argparse.Namespace(split="dev", press_name=PRESS, simulate=simulate)


def _materialize(root):
    """Every text level on disk, with sizes that differ per level."""
    for slot in SLOTS:
        for i, cr in enumerate(psweep.slot_effective_ratios(slot)):
            tag = psweep.to_compression_tag(cr)
            _write(root / "kv-text-qa-cache" / slot.model / PRESS / f"comp{tag}" / "cache_entry_a.pt", 100 + i)


def test_a_live_walk_is_recorded(tmp_path):
    root = tmp_path / "cache"
    _materialize(root)

    scans = psweep.resolve_level_scans(_benchmark(root), _args(None), SLOTS)

    recorded = kv_cache_sizes.load_recorded("bench", "dev", PRESS)
    assert {slot.model: scans[slot.key] for slot in SLOTS} == recorded


def test_a_replay_reads_the_recording_and_not_the_disk(tmp_path):
    """The whole point: the caches can be gone."""
    root = tmp_path / "cache"
    _materialize(root)
    live = psweep.resolve_level_scans(_benchmark(root), _args(None), SLOTS)

    replayed = psweep.resolve_level_scans(
        _benchmark(tmp_path / "nowhere"), _args([Path("store.json")]), SLOTS
    )

    assert replayed == live


def test_a_replay_prefers_the_recording_even_where_the_disk_differs(tmp_path):
    """Coordinator and workers decide from the same flag, so they must read the same
    numbers - a replay that consulted whichever disk it happened to land on could plan
    different states on two machines."""
    kv_cache_sizes.record(
        "bench", "dev", PRESS,
        {
            slot.model: {cr: LevelScan(7, 1, {}) for cr in psweep.slot_effective_ratios(slot)}
            for slot in SLOTS
        },
        tmp_path,
    )
    root = tmp_path / "cache"
    _materialize(root)

    replayed = psweep.resolve_level_scans(_benchmark(root), _args([Path("s.json")]), SLOTS)

    assert all(scan.bytes == 7 for levels in replayed.values() for scan in levels.values())


def test_a_replay_walks_the_disk_when_the_recording_has_a_hole(tmp_path, caplog):
    """A level missing from the file would silently vanish from every plan, so a partial
    recording is not planned from."""
    import logging

    kv_cache_sizes.record(
        "bench", "dev", PRESS, {TEXT_MODEL_8B: {0.0: LevelScan(7, 1, {})}}, tmp_path
    )
    root = tmp_path / "cache"
    _materialize(root)

    with caplog.at_level(logging.WARNING):
        replayed = psweep.resolve_level_scans(
            _benchmark(root), _args([Path("s.json")]), SLOTS
        )

    assert replayed["text_small"][0.0].bytes == 100
    assert any("no complete recording" in r.message for r in caplog.records)


def test_a_walk_that_finds_nothing_is_not_recorded(tmp_path):
    """"No caches here" is a fact about this machine. Recording it would make a later
    replay skip a benchmark whose caches simply live elsewhere."""
    psweep.resolve_level_scans(_benchmark(tmp_path / "empty"), _args(None), SLOTS)

    assert not kv_cache_sizes.sizes_path().exists()


def test_a_real_run_walks_the_disk_even_with_a_recording(tmp_path):
    """Only a replay may trust the file. A run that reads the caches measures them."""
    kv_cache_sizes.record(
        "bench", "dev", PRESS,
        {
            slot.model: {cr: LevelScan(7, 1, {}) for cr in psweep.slot_effective_ratios(slot)}
            for slot in SLOTS
        },
        tmp_path,
    )
    root = tmp_path / "cache"
    _materialize(root)

    live = psweep.resolve_level_scans(_benchmark(root), _args(None), SLOTS)

    assert live["text_small"][0.0].bytes == 100


def test_spec_slots_cover_every_materializable_model():
    """What the standalone script records, so no --text-*-model choice finds a hole."""
    from reasondb.interface.default_operator_toolbox import IMAGE_SPECS, TEXT_SPECS

    expected = {s.model for s in (*TEXT_SPECS, *IMAGE_SPECS) if not s.vanilla}
    slots = psweep.spec_slots()

    assert {slot.model for slot in slots} == expected
    assert all(slot.key == slot.model for slot in slots)


# ── scripts/measure_kv_cache_sizes.py ─────────────────────────────────────────


def _run_script(monkeypatch, *argv):
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "measure_kv_cache_sizes.py"
    spec = importlib.util.spec_from_file_location("measure_kv_cache_sizes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["measure_kv_cache_sizes.py", *argv])
    module.main()


def _fake_registry(monkeypatch, cache_roots):
    """Benchmarks whose databases name *cache_roots*, without loading real datasets."""
    import reasondb.evaluation.benchmark_registry as registry

    classes = {}
    for name, root in cache_roots.items():
        database = SimpleNamespace(cache_dir=root)
        classes[name] = type(
            name,
            (),
            {
                "name": staticmethod(lambda n=name: n),
                "load_without_queries": staticmethod(
                    lambda split, db=database: SimpleNamespace(database=db)
                ),
            },
        )
    monkeypatch.setattr(registry, "ALL_BENCHMARKS", classes)
    monkeypatch.setattr(registry, "RANDOM_BENCHMARKS", classes)


def test_the_script_records_every_spec_model_without_running_anything(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    _materialize(root)
    _fake_registry(monkeypatch, {"bench": root})

    _run_script(monkeypatch)

    recorded = kv_cache_sizes.load_recorded("bench", "dev", PRESS)
    # Every model the spec tables define, the ones with nothing on disk included - the
    # recording has to be able to say "absent", or a replay would read it as a hole.
    assert set(recorded) == {slot.model for slot in psweep.spec_slots()}
    assert recorded[TEXT_MODEL_8B][0.0].bytes == 100
    assert all(scan.bytes == 0 for scan in recorded["llava-hf/llava-next-72b-hf"].values())
    # And a replay can plan from it with the caches gone.
    assert psweep.recording_covers(recorded, SLOTS)


def test_the_script_skips_a_benchmark_it_has_no_caches_for(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    _materialize(root)
    _fake_registry(monkeypatch, {"bench": root, "elsewhere": tmp_path / "missing"})

    _run_script(monkeypatch)

    assert kv_cache_sizes.load_recorded("bench", "dev", PRESS)
    assert kv_cache_sizes.load_recorded("elsewhere", "dev", PRESS) == {}


def test_a_dry_run_records_nothing(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    _materialize(root)
    _fake_registry(monkeypatch, {"bench": root})

    _run_script(monkeypatch, "--dry-run")

    assert not kv_cache_sizes.sizes_path().exists()


# ── record_generated_levels: what the generator scripts call ────────────────


def _registered(monkeypatch, cache_root, name="bench"):
    """One registered benchmark whose database names *cache_root*."""
    import reasondb.evaluation.benchmark_registry as registry

    database = SimpleNamespace(cache_dir=cache_root)
    cls = type(
        name,
        (),
        {
            "name": staticmethod(lambda: name),
            "load_without_queries": staticmethod(
                lambda split: SimpleNamespace(database=database)
            ),
        },
    )
    monkeypatch.setattr(registry, "ALL_BENCHMARKS", {name: cls})


def test_generating_one_model_records_the_whole_benchmark(tmp_path, monkeypatch):
    """Every spec model, absent levels as zero - or a text-only benchmark, whose image
    models no generator ever writes, would never have a complete recording and every
    replay of it would fall back to a disk that is not there."""
    root = tmp_path / "cache"
    _materialize(root)
    _registered(monkeypatch, root)

    assert psweep.record_generated_levels(
        "bench", "dev", PRESS, [TEXT_MODEL_8B], written_under=root / "kv-text-qa-cache"
    )

    recorded = kv_cache_sizes.load_recorded("bench", "dev", PRESS)
    assert set(recorded) == {slot.model for slot in psweep.spec_slots()}
    assert recorded[TEXT_MODEL_8B][0.0].bytes == 100
    assert all(scan.bytes == 0 for scan in recorded["llava-hf/llava-next-72b-hf"].values())
    # Which is what lets a replay with every slot of the sweep trust it.
    image_slots = [
        psweep.ModelSlot("image_small", "image", "llava-hf/llama3-llava-next-8b-hf", False),
        psweep.ModelSlot("image_large", "image", "llava-hf/llava-next-72b-hf", True),
    ]
    assert psweep.recording_covers(recorded, [*SLOTS, *image_slots])


def test_generating_one_model_refreshes_the_others_from_disk(tmp_path, monkeypatch):
    """The disk is the truth at the moment a generator finishes, for every model on it."""
    root = tmp_path / "cache"
    _materialize(root)
    _registered(monkeypatch, root)
    kv_cache_sizes.record(
        "bench", "dev", PRESS, {TEXT_MODEL_70B: {0.3: LevelScan(7, 1, {})}}, root
    )

    psweep.record_generated_levels("bench", "dev", PRESS, [TEXT_MODEL_8B])

    recorded = kv_cache_sizes.load_recorded("bench", "dev", PRESS)
    assert recorded[TEXT_MODEL_70B][0.3].bytes == 100


def test_caches_written_outside_the_sweeps_root_are_not_recorded(tmp_path, monkeypatch):
    """A scan of the sweep's root would describe some other tree than the one written."""
    root = tmp_path / "cache"
    _materialize(root)
    _registered(monkeypatch, root)

    assert not psweep.record_generated_levels(
        "bench", "dev", PRESS, [TEXT_MODEL_8B], written_under=tmp_path / "elsewhere"
    )
    assert not kv_cache_sizes.sizes_path().exists()


def test_an_unregistered_benchmark_is_not_recorded(tmp_path, monkeypatch):
    _registered(monkeypatch, tmp_path / "cache")

    assert not psweep.record_generated_levels("nobody", "dev", PRESS, [TEXT_MODEL_8B])


def test_a_model_no_sweep_can_ask_for_is_not_recorded(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    _materialize(root)
    _registered(monkeypatch, root)

    assert not psweep.record_generated_levels("bench", "dev", PRESS, ["Qwen/Qwen2-72B"])


def test_a_generation_that_left_nothing_on_disk_is_not_recorded(tmp_path, monkeypatch):
    _registered(monkeypatch, tmp_path / "empty")

    assert not psweep.record_generated_levels("bench", "dev", PRESS, [TEXT_MODEL_8B])
    assert not kv_cache_sizes.sizes_path().exists()


def test_a_failure_while_recording_never_fails_the_generation(tmp_path, monkeypatch):
    """It runs at the end of hours of GPU work."""
    root = tmp_path / "cache"
    _materialize(root)
    _registered(monkeypatch, root)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(kv_cache_sizes, "record", broken)

    assert not psweep.record_generated_levels("bench", "dev", PRESS, [TEXT_MODEL_8B])


@pytest.mark.parametrize(
    "script",
    [
        "generate_kv_cache.py",
        "generate_kv_cache_image.py",
        "generate_kv_caches_indices.py",
        "generate_kv_caches_image_indices.py",
    ],
)
def test_every_generator_records_what_it_wrote(script):
    """Read off the source rather than run: each generator loads a model onto a GPU. What
    is pinned is that none of them can be edited into not recording without this noticing,
    and that the two with a --tmp mode skip the throwaway caches."""
    source = (Path(__file__).resolve().parents[1] / "scripts" / script).read_text()

    assert "record_generated_levels" in source
    assert source.count("record_sizes(") >= 2  # the helper, and at least one call
    if "tmp_root" in source:
        assert "if tmp_root is None:\n" in source
