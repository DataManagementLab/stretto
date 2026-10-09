"""One pass through the real CLI parser, registry, and merge, in-process.

Every other coordinator test calls a producer function directly with a
hand-built ``argparse.Namespace``. That leaves a gap nothing covers: whether the
flags ``scripts/run_coordinator.py`` actually defines line up with the keys each
producer's ``enumerate_jobs`` reads off ``args``, and whether ``--producer``
resolves through the registry to something whose jobs run and merge. A namespace
built by hand cannot catch a renamed flag or a missing default; this can.

It is also the single-machine path: ``run_coordinator.py --local``.

Kept in-process (no subprocess, no Flask, no SQLite, no GPU, no servers) by
faking the two functions that would otherwise need a model: the per-approach
executor factory and the labelling pass.
"""

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from conftest import fake_slot_map

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import scripts.run_coordinator as rc
    from reasondb.coordinator.producers import parameter_sweep as srp
except ImportError:  # pragma: no cover - deps not installed
    pytest.skip("coordinator deps not installed", allow_module_level=True)


FAKE_QUERIES = ["q0", "q1"]


_LOCAL_STATES = [({"text_small": [0.0]}, 1000), ({"text_small": [0.5]}, 400)]


def _fake_benchmark():
    return types.SimpleNamespace(
        name=lambda: "movie_random",
        has_ground_truth=False,
        database=object(),
        queries=list(FAKE_QUERIES),
        query_count=lambda debug_query=None: len(FAKE_QUERIES),
    )


@pytest.fixture
def local_env(monkeypatch, tmp_path):
    """Everything that would need a GPU, a model server, or a KV cache on disk."""
    benchmark = _fake_benchmark()
    monkeypatch.setattr(
        srp, "BENCHMARKS",
        {
            "movie_random": types.SimpleNamespace(
                name=lambda: "movie_random",
                load=lambda split: benchmark,
                load_without_queries=lambda split: benchmark,
                count_queries=lambda split, debug_query=None: len(FAKE_QUERIES),
            )
        },
    )
    # The phase-0 job needs a real benchmark class and a gold model; it has its own
    # tests. Here the point is the CLI -> registry -> run -> merge path.
    monkeypatch.setattr(
        srp, "enumerate_filter_stats_jobs", lambda task_id, out, args, producer: []
    )
    # prepare_sweep would `du` real cache directories; hand it a two-step plan.
    monkeypatch.setattr(
        srp.psweep, "prepare_sweep",
        lambda b, a: srp.psweep.SweepPrep(
            _LOCAL_STATES,
            {"text_small": {0.0: 1000, 0.5: 400}},
            {"text_small": {0.0: 100, 0.5: 100}},
            [],
            fake_slot_map(_LOCAL_STATES),
            {"text_small": [0.0, 0.5]},
        ),
    )
    monkeypatch.setattr(srp, "collect_labels", lambda *a, **k: {})

    ran = []

    def fake_run_state(**kwargs):
        ran.append(kwargs)
        prec, rec = kwargs["guarantees"][0]
        rows = [
            {
                "benchmark": "movie_random",
                "split": "dev",
                "step": kwargs["step_idx"],
                "tune_parameters": kwargs["tune_parameters"],
                "sample_size": kwargs["sample_size"],
                "query": q,
                "precision_guarantee": prec,
                "recall_guarantee": rec,
                "execution_runtime_s": 1.0,
                "storage_gb": kwargs["footprint_bytes"] / (1024 ** 3),
            }
            for q in FAKE_QUERIES
        ]
        # (rows, predictions, costs) - the last two are what scoring reads off the shard.
        return rows, {}, {}

    monkeypatch.setattr(srp.psweep, "run_state", fake_run_state)
    return types.SimpleNamespace(ran=ran, tmp_path=tmp_path)


def _parse(*argv):
    """Parse *and resolve*, the way ``main()`` does.

    ``--precompute``/``--simulate`` reach a producer as a ``{benchmark: path}`` mapping,
    not as the raw strings argparse collects; resolving is where the two are reconciled
    with ``--benchmarks``. A test that skipped it would exercise a shape no real run has.
    """
    args = rc.build_parser().parse_args(list(argv))
    rc.resolve_precompute_simulate(args, rc.ALL_BENCHMARKS, rc.DEFAULT_BENCHMARKS)
    return args


def _run(tmp_path, *extra_argv):
    args = _parse(
        "--task-id", "t_local",
        "--local",
        "--producer", "parameter_sweep",
        "--benchmarks", "movie_random",
        "--device", "cpu",
        "--precision-guarantees", "0.7",
        "--recall-guarantees", "0.7",
        "--output-dir", str(tmp_path / "out"),
        *extra_argv,
    )
    rc.run_local(args)
    return args


def test_local_run_enumerates_runs_and_merges(local_env):
    """parser -> enumerate_jobs -> run_job -> merge, through the real registry."""
    tmp_path = local_env.tmp_path
    _run(tmp_path)

    # Two storage steps ran, each over both queries.
    assert len(local_env.ran) == 2
    assert {k["step_idx"] for k in local_env.ran} == {0, 1}

    merged = tmp_path / "out" / "merged" / "movie_random" / "dev" / "parameter_sweep.csv"
    assert merged.is_file(), "merge did not write where the plot scripts glob"
    df = pd.read_csv(merged)
    assert len(df) == 4  # 2 steps x 2 queries
    assert sorted(df["step"].unique()) == [0, 1]


def test_local_run_defaults_leave_the_optimizer_axes_alone(local_env):
    """Without axis flags, the optimizer axes stay at their defaults."""
    _run(local_env.tmp_path)

    assert all(k["tune_parameters"] is True for k in local_env.ran)
    assert all(k["sample_size"] is None for k in local_env.ran)


def test_local_run_crosses_the_optimizer_axes(local_env):
    """The four-axis sweep, end to end."""
    tmp_path = local_env.tmp_path
    _run(
        tmp_path,
        "--tune-parameters", "true", "false",
        "--sample-sizes", "30", "150",
    )

    # 2 steps x 2 tune values x 2 sample sizes.
    assert len(local_env.ran) == 8
    assert {(k["tune_parameters"], k["sample_size"]) for k in local_env.ran} == {
        (True, 30), (True, 150), (False, 30), (False, 150),
    }

    df = pd.read_csv(
        tmp_path / "out" / "merged" / "movie_random" / "dev" / "parameter_sweep.csv"
    )
    assert set(df["tune_parameters"]) == {True, False}
    assert set(df["sample_size"]) == {30, 150}


def test_local_precompute_run_writes_the_file_the_mapping_named(local_env, monkeypatch):
    """`--precompute bench=path.json` records straight into that path.

    There is no merge step: one job per dataset owns one file, so the store the later
    `--simulate` reads is the one the operator asked for rather than a copy of it.
    """
    tmp_path = local_env.tmp_path
    calls = []

    def fake_run_precompute(benchmark, executor, output_path, **kwargs):
        calls.append(kwargs)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            '{"text_qa": {}, "vision": {}, "precomputed_ops": [], '
            '"operator_configs": {}}'
        )
        return types.SimpleNamespace(counts=lambda: {"text_qa": 0})

    monkeypatch.setattr(srp, "run_precompute", fake_run_precompute)
    monkeypatch.setattr(srp.psweep, "build_precompute_configurator", lambda *a, **k: object())
    monkeypatch.setattr(srp, "build_reasoner", lambda c: object())
    monkeypatch.setattr(srp, "build_approach_executor", lambda *a, **k: object())

    args = _parse(
        "--task-id", "t_pre", "--local", "--producer", "parameter_sweep",
        "--device", "cpu",
        # No --benchmarks: the mapping is the selection.
        "--precompute", f"movie_random={tmp_path / 'responses.json'}",
        "--output-dir", str(tmp_path / "pre"),
    )
    assert args.benchmarks == ["movie_random"]
    rc.run_local(args)

    # One pass over the whole benchmark - not split by query range, so the store's
    # own (operator, expression) reuse applies across every query.
    assert len(calls) == 1
    assert "query_slice" not in calls[0]
    assert (tmp_path / "responses.json").is_file()
    assert not (tmp_path / "pre" / "merged").exists()


def test_local_run_attributes_telemetry_to_the_job_that_produced_it(local_env, monkeypatch):
    """``--local`` has no forwarder to tag batches, so it relies entirely on the
    emit-time stamp; without it, telemetry would carry no ``job_id`` and the
    dashboard could not tell one job's queries from the next's."""
    import time as _time

    from reasondb.monitor import collector as monitor

    tmp_path = local_env.tmp_path
    collector = monitor.Collector(jsonl_path=tmp_path / "telemetry.jsonl").install()
    try:
        # Emit from inside the job, where a real producer's query_end comes from.
        inner = srp.psweep.run_state

        def emitting_run_state(**kwargs):
            monitor.record_query_end(
                query=f"q-step{kwargs['step_idx']}",
                query_index=0,
                executor="e",
                cached=False,
            )
            return inner(**kwargs)

        monkeypatch.setattr(srp.psweep, "run_state", emitting_run_state)
        _run(tmp_path)

        deadline = _time.time() + 2.0
        while collector.snapshot_run()["queue_depth"] > 0 and _time.time() < deadline:
            _time.sleep(0.01)
        ends = [
            e["data"]
            for e in collector.events_since(since=0, limit=1000)["events"]
            if e["type"] == "query_end"
        ]
    finally:
        collector.close()
        monitor.set_current_job(None)

    # Each step's event names its own job, and every event is attributed.
    assert len(ends) == 2
    assert all(d.get("job_id") for d in ends)
    by_query = {d["query"]: d["job_id"] for d in ends}
    assert by_query["q-step0"] != by_query["q-step1"]
    assert all("movie_random" in job for job in by_query.values())
