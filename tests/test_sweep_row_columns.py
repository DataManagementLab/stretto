"""What one sweep point records, as a schema contract on ``run_state``'s rows.

Everything downstream reads these columns by name: the merge concatenates them, the
plotting registry maps them to axes, and a figure drawn from a CSV that lacks a column
degrades rather than raising. So the columns are an interface, and the ones ``kvop01``
reads - what a state's caches cost per item, how many tuples the dataset has, which
modalities it uses, and how many LLM operators the optimizer could choose between - are
worth pinning where they are written rather than only where they are drawn.

The executor is stubbed. Building a real one needs model servers, and what is under test
is the row, not the run.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    from reasondb.evaluation import parameter_sweep as psweep
except ImportError:  # pragma: no cover - the guard the sibling storage tests use
    pytest.skip("parameter_sweep deps not installed", allow_module_level=True)

from reasondb.database.indentifier import InPlaceColumn
from reasondb.interface.default_operator_toolbox import TEXT_MODEL_8B, TEXT_MODEL_70B

QUERY = "how many reviews are positive"


class _FakeCost:
    component_times = {"execution": 12.0, "end_to_end": 20.0}
    execution_cost = SimpleNamespace(runtime=12.0)
    tuning_cost = SimpleNamespace(runtime=3.0)
    total_cost = SimpleNamespace(runtime=15.0, monetary_cost=0.25)


class _FakeExecutor:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_benchmark(self, queries, *args, **kwargs):
        return SimpleNamespace(
            costs={QUERY: _FakeCost()},
            result_paths={QUERY: "/dev/null"},
        )


@pytest.fixture
def benchmark(tmp_path):
    reviews = tmp_path / "reviews.csv"
    reviews.write_text("reviewtext\na\nb\nc\nd\n")
    database = SimpleNamespace(
        cache_dir=tmp_path / "cache",
        external_tables=[
            SimpleNamespace(
                name="reviews",
                path=reviews,
                file_type="csv",
                text_columns=[InPlaceColumn("reviews.reviewtext")],
                image_columns=[],
                audio_columns=[],
            )
        ],
    )
    return SimpleNamespace(
        database=database,
        name=lambda: "movie_random",
        queries=[SimpleNamespace(query=QUERY)],
    )


@pytest.fixture
def args(tmp_path):
    return argparse.Namespace(
        split="dev",
        use_indexes=False,
        human_labels=False,
        cost_type="runtime",
        device="cpu",
        output_dir=tmp_path / "out",
        debug_query=None,
        state_plan="kv_operator",
        sweep_to_gold=False,
        press_name="expected_attention",
        text_small_model=TEXT_MODEL_8B,
        text_large_model=TEXT_MODEL_70B,
        image_small_model="llava-hf/llama3-llava-next-8b-hf",
        image_large_model="llava-hf/llava-next-72b-hf",
    )


def _run(monkeypatch, benchmark, args, state, footprint, step_idx, entries):
    monkeypatch.setattr(psweep, "build_reasoner", lambda configurator: None)
    monkeypatch.setattr(
        psweep, "build_approach_executor", lambda *a, **k: _FakeExecutor()
    )
    slots = psweep.build_slots(args)
    slot_by_key = {s.key: s for s in slots if s.key in state}
    storage = {"text_small": {0.5: footprint}, "text_large": {0.8: footprint}}
    rows, _paths, _costs = psweep.run_state(
        benchmark=benchmark,
        step_idx=step_idx,
        state=state,
        footprint_bytes=footprint,
        slot_by_key=slot_by_key,
        storage_table=storage,
        entry_table=entries,
        guarantees=[(0.7, 0.7)],
        args=args,
        approach="optim_global",
        tune_parameters=True,
        sample_size=100,
        output_dir=Path(args.output_dir),
    )
    return rows


def test_a_compressed_state_reports_its_cache_per_item(monkeypatch, benchmark, args):
    """One state, one cache, so the per-item cost is that operator's own - which is the
    number kvop01 is built to report."""
    (row,) = _run(
        monkeypatch,
        benchmark,
        args,
        state={"text_small": [0.5], "text_large": []},
        footprint=8000,
        step_idx=1,
        entries={"text_small": {0.5: 4}, "text_large": {0.8: 4}},
    )

    assert row["storage_bytes"] == 8000
    assert row["cache_entries"] == 4
    assert row["storage_bytes_per_entry"] == 2000
    assert row["text_small_cache_entries"] == 4
    assert row["text_large_cache_entries"] is None


def test_a_state_caching_nothing_has_no_per_item_cost(monkeypatch, benchmark, args):
    """None rather than 0: an operator that costs nothing per item and no operator at all
    are different facts, and 0 reads as the first."""
    (row,) = _run(
        monkeypatch,
        benchmark,
        args,
        state={"text_small": [], "text_large": []},
        footprint=0,
        step_idx=0,
        entries={"text_small": {0.5: 4}, "text_large": {0.8: 4}},
    )

    assert row["cache_entries"] == 0
    assert row["storage_bytes_per_entry"] is None


def test_the_row_describes_the_dataset_as_well_as_the_sweep_point(
    monkeypatch, benchmark, args
):
    """A footprint is not comparable between benchmarks until it is divided by the rows
    behind it, so the row records them."""
    (row,) = _run(
        monkeypatch,
        benchmark,
        args,
        state={"text_small": [0.5], "text_large": []},
        footprint=8000,
        step_idx=1,
        entries={"text_small": {0.5: 4}, "text_large": {0.8: 4}},
    )

    assert row["num_tuples"] == 4
    assert json.loads(row["num_tuples_per_table"]) == {"reviews": 4}
    assert row["modality"] == "text"


def test_both_kv_operator_steps_report_two_llm_operators(monkeypatch, benchmark, args):
    """Two by two different routes - both vanilla operators at step 0, the gold one plus
    the retained level afterwards - which the retained ratios alone cannot say."""
    entries = {"text_small": {0.5: 4}, "text_large": {0.8: 4}}
    (vanilla,) = _run(
        monkeypatch, benchmark, args,
        state={"text_small": [], "text_large": []}, footprint=0, step_idx=0,
        entries=entries,
    )
    (compressed,) = _run(
        monkeypatch, benchmark, args,
        state={"text_small": [0.5], "text_large": []}, footprint=8000, step_idx=1,
        entries=entries,
    )

    assert vanilla["n_llm_operators"] == 2
    assert compressed["n_llm_operators"] == 2
    assert vanilla["state_plan"] == "kv_operator"
