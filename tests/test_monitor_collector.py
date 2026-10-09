"""The monitor's hot path must be a true no-op when disabled, and its ring buffer must
stay bounded under a long run instead of growing without limit.

`record_*` is called from `PhysicalOperator.run_outside_db` (every operator, every
query) and from every KV backend client (every inference call), so an accidental import
error, a stray lock, or an unbounded buffer there would show up as overhead in every
benchmark run, not just as a monitor bug. These tests pin the two properties that make
that impossible: disabled calls touch nothing, and the live buffer evicts.
"""

import json
import time

import pytest

from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector
from reasondb.monitor.events import EVENT_TYPES


@pytest.fixture(autouse=True)
def clean_sink():
    """The sink is a process-global registry (mirrors SimulateStore); don't leak it."""
    yield
    if monitor.get_collector() is not None:
        monitor.get_collector().close()


def test_disabled_record_calls_are_noop():
    assert monitor.get_collector() is None
    assert monitor.is_enabled() is False
    # None of these may raise or have any observable effect with no sink installed.
    monitor.record_run_start(script="x")
    monitor.record_phase("tuning", 1.23)
    monitor.record_operator_run(operator="op", seconds=0.1, n_input_rows=1)
    monitor.record_kv_inference({"schema": 1})
    monitor.record_error("somewhere", "boom")


def test_install_and_uninstall_toggle_is_enabled(tmp_path):
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        assert monitor.is_enabled() is True
        assert monitor.get_collector() is c
    finally:
        c.close()
    assert monitor.is_enabled() is False


def test_double_install_asserts(tmp_path):
    c1 = Collector(jsonl_path=tmp_path / "a.jsonl").install()
    try:
        with pytest.raises(AssertionError):
            Collector(jsonl_path=tmp_path / "b.jsonl").install()
    finally:
        c1.close()


def test_events_recorded_and_ordered_by_sequence(tmp_path):
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        for i in range(5):
            monitor.record_phase(f"phase-{i}", 0.01)
        _drain(c)
        payload = c.events_since(since=0, limit=100)
        seqs = [e["seq"] for e in payload["events"]]
        assert seqs == sorted(seqs)
        assert len(seqs) == len(set(seqs))
        assert payload["events"][0]["data"]["name"] == "phase-0"
    finally:
        c.close()


def test_ring_buffer_evicts_beyond_maxlen(tmp_path):
    c = Collector(jsonl_path=tmp_path / "t.jsonl", ring_size=10).install()
    try:
        for i in range(37):
            monitor.record_phase(f"phase-{i}", 0.0)
        _drain(c)
        payload = c.events_since(since=0, limit=1000)
        assert len(payload["events"]) == 10
        assert payload["events"][-1]["data"]["name"] == "phase-36"
        assert payload["gap"] is True
    finally:
        c.close()


def test_queue_overflow_drops_and_counts_instead_of_raising(tmp_path):
    c = Collector(jsonl_path=tmp_path / "t.jsonl", max_queue=3).install()
    try:
        # The drain thread is running but may not keep up instantly; push far more than
        # the cap so at least one put has to observe the queue as full.
        for i in range(2000):
            monitor.record_phase(f"p{i}", 0.0)
        deadline = time.time() + 2.0
        while c.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
            time.sleep(0.01)
        run = c.snapshot_run()
        assert run["dropped_events"] >= 0  # never raised getting here
    finally:
        c.close()


def test_unknown_event_type_is_dropped_not_raised(tmp_path, caplog):
    """The drain thread must survive a malformed event rather than dying silently."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        c.put("not_a_real_kind", {})
        monitor.record_phase("after", 0.0)
        _drain(c)
        payload = c.events_since(since=0, limit=100)
        assert [e["type"] for e in payload["events"]] == ["phase"]
    finally:
        c.close()


def test_close_flushes_pending_events_before_exit(tmp_path):
    path = tmp_path / "t.jsonl"
    c = Collector(jsonl_path=path).install()
    monitor.record_phase("final", 0.5)
    c.close(timeout=5.0)
    lines = path.read_text().strip().splitlines()
    kinds = [json.loads(line)["type"] for line in lines]
    assert "phase" in kinds


def test_snapshot_aggregates_operator_totals(tmp_path):
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_operator_run(
            operator="TextQaFilter", seconds=1.0, n_input_rows=10,
            runtime=1.0, monetary_cost=0.02, fake_cost=0.0,
        )
        monitor.record_operator_run(
            operator="TextQaFilter", seconds=2.0, n_input_rows=10,
            runtime=2.0, monetary_cost=0.02, fake_cost=0.0,
        )
        _drain(c)
        agg = c.snapshot_aggregates()
        (row,) = [o for o in agg["operators"] if o["operator"] == "TextQaFilter"]
        assert row["calls"] == 2
        assert row["seconds"] == pytest.approx(3.0)
        assert row["input_rows"] == 20
    finally:
        c.close()


def test_operator_buckets_split_by_class_model_variant_and_phase(tmp_path):
    """The tuples-per-operator and per-CR charts need (operation_class, model, ratio)
    buckets, distinct from the flat `operators` aggregate keyed on the full identifier
    string - and split by phase, because `PhysicalOperator.profile` runs the very same
    operator over a *sample*, so pooling the two makes a tuple count meaningless."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        for seconds in (1.0, 2.0):
            monitor.record_operator_run(
                operator="TextQaFilter-LLMTextQABackend-m-cr0.5",
                operation_class="TextQaFilter",
                model_name="m",
                effective_compression_ratio=0.5,
                materialized_compression_ratio=0.5,
                vanilla=False,
                phase="execution",
                seconds=seconds, n_input_rows=10,
            )
        monitor.record_operator_run(
            operator="TextQaFilter-LLMTextQABackend-m-cr0.5",
            operation_class="TextQaFilter",
            model_name="m",
            effective_compression_ratio=0.5,
            vanilla=False,
            phase="profiling",
            seconds=0.4, n_input_rows=3,
        )
        monitor.record_operator_run(
            operator="TextQaFilter-LLMTextQABackend-m-cr0.0-vanilla",
            operation_class="TextQaFilter",
            model_name="m",
            effective_compression_ratio=0.0,
            materialized_compression_ratio=0.0,
            vanilla=True,
            phase="execution",
            seconds=3.0, n_input_rows=10,
        )
        monitor.record_operator_run(
            operator="TraditionalFilter",
            operation_class="TraditionalFilter",
            phase="execution",
            seconds=0.1, n_input_rows=10,
        )
        _drain(c)
        agg = c.snapshot_aggregates()
        buckets = {
            (e["operation_class"], e["cr_label"], e["phase"]): e
            for e in agg["operator_buckets"]
        }

        run = buckets[("TextQaFilter", "cr0.5", "execution")]
        assert run["calls"] == 2
        assert run["seconds"] == pytest.approx(3.0)
        assert run["input_rows"] == 20
        # The sample the profiler put through the same operator is its own bucket.
        assert buckets[("TextQaFilter", "cr0.5", "profiling")]["input_rows"] == 3
        assert buckets[("TextQaFilter", "vanilla", "execution")]["seconds"] == pytest.approx(3.0)
        assert buckets[("TextQaFilter", "vanilla", "execution")]["vanilla"] is True
        assert buckets[("TraditionalFilter", "n/a", "execution")]["seconds"] == pytest.approx(0.1)
    finally:
        c.close()


def test_run_state_keeps_per_worker_progress_and_a_cumulative_roll_up():
    """Progress counters are per worker: per-job ones reset, cumulative ones never do.

    A single counter bumped by every worker's query_end, divided by a total taken from
    whichever query_start arrived last, reports more queries done than exist as soon as
    two workers run. A per-iteration counter reset on each executor_start restarts the
    bar instead of advancing it."""
    from reasondb.monitor.collector import _RunState
    from reasondb.monitor.events import make_event

    state = _RunState()

    def fire(event_type, **data):
        state.apply(make_event(1, event_type, 0.0, data))

    # Two workers, two jobs of 2 queries each, interleaved as they really arrive.
    for worker, job in (("w-a", "j1"), ("w-b", "j2")):
        fire("executor_start", worker_id=worker, job_id=job, executor="optim_global")
    for i in range(2):
        for worker, job in (("w-a", "j1"), ("w-b", "j2")):
            fire("query_start", worker_id=worker, job_id=job, query=f"q{i}",
                 query_index=i, n_queries=2)
            fire("query_end", worker_id=worker, job_id=job, query=f"q{i}", cached=False)

    payload = state.to_json()
    assert payload["queries_done"] == 4  # 2 workers x 2 queries
    assert payload["n_queries"] == 4  # summed over both jobs in flight, not one job's
    assert payload["queries_done"] <= payload["n_queries"]
    per_worker = {w["worker_id"]: w for w in payload["workers"]}
    assert per_worker["w-a"]["queries_done_in_job"] == 2
    assert per_worker["w-b"]["queries_finished"] == 2

    # w-a moves on to a third job: its per-job counter resets, its cumulative one does
    # not, and the roll-up still never exceeds the work in flight.
    fire("executor_start", worker_id="w-a", job_id="j3", executor="lotus")
    fire("query_start", worker_id="w-a", job_id="j3", query="q0", query_index=0, n_queries=5)
    payload = state.to_json()
    per_worker = {w["worker_id"]: w for w in payload["workers"]}
    assert per_worker["w-a"]["queries_done_in_job"] == 0
    assert per_worker["w-a"]["queries_finished"] == 2
    assert payload["queries_done"] == 4


def test_run_plan_is_folded_into_the_run_state():
    """A sweep script announces its full size once; the dashboard needs it verbatim to
    draw a bar over every iteration instead of restarting on each."""
    from reasondb.monitor.collector import _RunState
    from reasondb.monitor.events import make_event

    state = _RunState()
    state.apply(
        make_event(1, "run_plan", 0.0, {"total_queries": 120, "total_iterations": 6})
    )
    assert state.to_json()["plan"] == {"total_queries": 120, "total_iterations": 6}


def test_query_times_are_slim_and_the_plan_lives_in_the_detail_record(tmp_path):
    """`query_times` is polled by every open dashboard and holds up to a thousand rows,
    so it must not carry the tuned pipeline (a full JSON plan per row). The plan, and
    the per-operator tuple counts, go to the detail record the Query tab fetches for
    one query at a time."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_executor_start(
            executor="optim_global", precision_guarantee=0.7, recall_guarantee=0.9,
            worker_id="w-a", job_id="t1-run_benchmark-movie-optim_global-p0.7-r0.9",
        )
        monitor.record_query_start(
            query="Extract [title] from {r.text}", query_index=3, n_queries=4,
            worker_id="w-a", job_id="t1-run_benchmark-movie-optim_global-p0.7-r0.9",
        )
        monitor.record_operator_run(
            operator="TextQaFilter-cr0.5", operation_class="TextQaFilter",
            phase="execution", seconds=1.0, n_input_rows=50,
            worker_id="w-a", job_id="t1-run_benchmark-movie-optim_global-p0.7-r0.9",
        )
        monitor.record_query_end(
            executor="optim_global",
            benchmark="movie_random",
            query="Extract [title] from {r.text}",
            query_index=3,
            cached=False,
            component_times={"end_to_end": 3.0, "execution": 1.2, "tuning": 1.0, "profiling": 0.4},
            tuned_pipeline=["[]", '[{"operator": "TextQaFilter-cr0.5"}]'],
            worker_id="w-a",
            job_id="t1-run_benchmark-movie-optim_global-p0.7-r0.9",
        )
        _drain(c)

        (row,) = c.snapshot_aggregates()["query_times"]
        assert row["query"] == "Extract [title] from {r.text}"
        assert row["worker_id"] == "w-a"
        assert row["job_id"] == "t1-run_benchmark-movie-optim_global-p0.7-r0.9"
        assert "tuned_pipeline" not in row
        # Guarantees only ever appear on executor_start; the aggregate stamps them onto
        # the query so the dashboard can group and filter by target.
        assert (row["precision"], row["recall"]) == (0.7, 0.9)
        # Non-overlapping, and adding up: tuning(1.0) - profiling(0.4) = optimization.
        assert row["phase_components"]["time_optimization"] == pytest.approx(0.6)
        assert sum(
            row["phase_components"][k]
            for k in row["phase_components"] if k != "time_end_to_end"
        ) == pytest.approx(3.0)

        index = c.snapshot_queries()["queries"]
        assert [q["query"] for q in index] == ["Extract [title] from {r.text}"]
        detail = c.snapshot_query_detail("Extract [title] from {r.text}")
        (run,) = detail["runs"]
        assert run["tuned_pipeline"] == ["[]", '[{"operator": "TextQaFilter-cr0.5"}]']
        assert run["run_key"] == "t1-run_benchmark-movie-optim_global-p0.7-r0.9"
        (op,) = run["operators"]
        assert (op["operator"], op["input_rows"], op["phase"]) == (
            "TextQaFilter-cr0.5", 50, "execution",
        )
    finally:
        c.close()


def test_query_times_tags_absent_for_a_plain_single_process_run():
    """A run.py script that never touches the coordinator must not fabricate
    worker_id/job_id - absent (None), not a stray placeholder string."""
    from reasondb.monitor.collector import _Aggregates
    from reasondb.monitor.events import make_event

    agg = _Aggregates()
    agg.apply(make_event(1, "query_end", 0.0, {"executor": "optim_global", "query": "q"}))
    (row,) = agg.query_times
    assert row["worker_id"] is None
    assert row["job_id"] is None
    # The run key still has to distinguish configurations, so it falls back to the only
    # things a lone process varies between iterations.
    assert row["run_key"] == "optim_global|None|None"


def test_a_jobs_label_pass_does_not_overwrite_its_own_sweep_record():
    """One job emits two records per query, and the Query tab must keep both.

    A storage-sweep step job runs its sweep queries and *then* a labelling pass over the
    same queries (``evaluation.parameter_sweep.run_state`` resolves labels after the
    executor loop). Both carry the same ``job_id``, so a run key that was only the job id
    would collide in ``query_details``, and the label pass (which lands second) would
    overwrite the sweep point the job exists to measure.
    """
    from reasondb.monitor.collector import _Aggregates
    from reasondb.monitor.events import make_event

    job = "sweep01-parameter_sweep-rotowire_random-s0-p0.7-r0.7-tunetrue-nNone"
    agg = _Aggregates()
    for role, executor, cached, pipeline in (
        ("sweep", "storage_step0_tunetrue_nNone", False, ['[{"operator": "TextQaFilter-cr0.5"}]']),
        ("label", "silver", True, ['[{"operator": "TextQaFilter-vanilla"}]']),
    ):
        agg.apply(make_event(1, "query_end", 0.0, {
            "executor": executor, "role": role, "job_id": job, "query": "q",
            "cached": cached, "tuned_pipeline": pipeline,
            "component_times": {"end_to_end": 1.0},
        }))

    runs = agg.query_details["q"]["runs"]
    assert sorted(runs) == [job, f"{job}|label"]
    assert runs[job]["cached"] is False
    assert runs[job]["tuned_pipeline"] == ['[{"operator": "TextQaFilter-cr0.5"}]']
    assert runs[f"{job}|label"]["role"] == "label"


def test_phase_aggregate_separates_self_time_from_nested_spans(tmp_path):
    """`measure()` spans nest - end_to_end contains tuning contains profiling - so
    summing raw span seconds reports roughly double the wall clock actually spent."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_phase("profiling", 4.0, parent="tuning")
        monitor.record_phase("tuning", 10.0, parent="end_to_end")
        monitor.record_phase("execution", 5.0, parent="end_to_end")
        monitor.record_phase("end_to_end", 20.0, parent=None)
        _drain(c)
        phases = c.snapshot_aggregates()["phases"]

        assert phases["end_to_end"]["seconds"] == pytest.approx(20.0)
        assert phases["end_to_end"]["self_seconds"] == pytest.approx(5.0)  # 20 - 10 - 5
        assert phases["tuning"]["self_seconds"] == pytest.approx(6.0)  # 10 - 4
        assert phases["profiling"]["self_seconds"] == pytest.approx(4.0)
        # The whole point: self time partitions the top-level span.
        assert sum(p["self_seconds"] for p in phases.values()) == pytest.approx(20.0)
    finally:
        c.close()


def test_all_event_types_are_constructible_kinds():
    """Every kind the collector accepts must be a real, spellable string."""
    for kind in EVENT_TYPES:
        assert isinstance(kind, str) and kind


def _drain(collector: Collector, timeout: float = 2.0) -> None:
    """Wait for the background thread to catch up before asserting on state."""
    deadline = time.time() + timeout
    while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
        time.sleep(0.01)


def test_search_space_is_recorded_per_configuration_and_step(tmp_path):
    """The Search space tab needs what the optimizer could have chosen, not just what it
    did: a tuned pipeline keeps only the pick, so without this there is no way to tell a
    good choice from a search space that never contained the good option."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_executor_start(
            executor="optim_global", precision_guarantee=0.7, recall_guarantee=0.7,
            worker_id="w-a", job_id="job-s0",
        )
        monitor.record_query_start(query="q1", query_index=0, worker_id="w-a", job_id="job-s0")
        monitor.record_search_space(
            query="q1",
            step_index=0,
            logical_type="LogicalFilter",
            logical_expression="{r.text} is positive",
            inputs=["reviews"],
            output="out",
            n_candidates=2,
            candidates=[
                {"operator": "TextQaFilter-cr0.8", "quality": 6.5, "gold": False},
                {"operator": "TextQaFilter-vanilla", "quality": 8.0, "gold": True,
                 "vanilla": True},
            ],
            worker_id="w-a",
            job_id="job-s0",
        )
        _drain(c)
        (step,) = c.snapshot_search_space()["steps"]
        assert step["logical_type"] == "LogicalFilter"
        assert step["job_id"] == "job-s0"
        assert (step["precision"], step["recall"]) == (0.7, 0.7)
        assert [c_["operator"] for c_ in step["candidates"]] == [
            "TextQaFilter-cr0.8", "TextQaFilter-vanilla",
        ]
        (gold,) = [c_ for c_ in step["candidates"] if c_["gold"]]
        assert gold["operator"] == "TextQaFilter-vanilla"
    finally:
        c.close()


def test_search_space_deduplicates_repeated_configuration_of_one_step(tmp_path):
    """The same step is reconfigured for every guarantee pair and every repeat of a
    query; the candidate set belongs to the configuration, not to the attempt."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_executor_start(executor="optim_global", worker_id="w", job_id="j")
        for _ in range(3):
            monitor.record_search_space(
                query="q1", step_index=0, logical_type="LogicalFilter",
                candidates=[{"operator": "op", "quality": 1.0, "gold": True}],
                worker_id="w", job_id="j",
            )
        # A different step of the same query is its own record.
        monitor.record_search_space(
            query="q1", step_index=1, logical_type="LogicalExtract",
            candidates=[{"operator": "op2", "quality": 2.0, "gold": True}],
            worker_id="w", job_id="j",
        )
        _drain(c)
        steps = c.snapshot_search_space()["steps"]
        assert len(steps) == 2
        assert sorted(s["step_index"] for s in steps) == [0, 1]
    finally:
        c.close()


def test_query_metrics_carry_achieved_accuracy_and_target_ratios(tmp_path):
    """Accuracy only exists after evaluate() scores predictions against labels, so it
    arrives long after the query ran - as its own record, stamped with the guarantee it
    was scored against, and carrying the achieved/target ratio plot_meets_target uses."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_executor_start(
            executor="storage_step0", precision_guarantee=0.7, recall_guarantee=0.7,
            worker_id="w", job_id="j0",
        )
        monitor.record_query_metrics(
            benchmark="movie_random",
            executor="storage_step0",
            query="q1",
            precision_guarantee=0.7,
            recall_guarantee=0.7,
            precision=0.84,
            recall=0.63,
            f1_score=0.72,
            worker_id="w",
            job_id="j0",
        )
        _drain(c)
        (row,) = c.snapshot_aggregates()["query_metrics"]
        assert row["query"] == "q1"
        assert row["job_id"] == "j0"
        assert row["precision_met"] == pytest.approx(1.2)  # 0.84 / 0.7 -> guarantee held
        assert row["recall_met"] == pytest.approx(0.9)  # 0.63 / 0.7 -> missed
        assert (row["precision_target"], row["recall_target"]) == (0.7, 0.7)
        assert (row["precision_achieved"], row["recall_achieved"]) == (0.84, 0.63)
        # The achieved values must not land under the dimension names: `precision` and
        # `recall` mean "the guarantee this ran under" on every record kind, and the
        # facet bar derives one shared set of chips from all of them. A continuous
        # measurement here would turn the "Recall target" filter into one chip per query.
        assert (row["precision"], row["recall"]) == (0.7, 0.7)
    finally:
        c.close()


def test_query_metrics_without_a_target_have_no_ratio(tmp_path):
    """A no-guarantee baseline has nothing to divide by; the ratio must be absent rather
    than infinite or silently zero, so it simply doesn't plot on that axis."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_query_metrics(
            executor="gpt", query="q1", precision=0.9, recall=0.8,
            precision_guarantee=None, recall_guarantee=None,
        )
        # A zero target would divide by zero.
        monitor.record_query_metrics(
            executor="gpt", query="q2", precision=0.9, recall=0.8,
            precision_guarantee=0.0, recall_guarantee=0.0,
        )
        _drain(c)
        for row in c.snapshot_aggregates()["query_metrics"]:
            assert row["precision_met"] is None
            assert row["recall_met"] is None
            assert row["precision_achieved"] == pytest.approx(0.9)
    finally:
        c.close()


def test_query_metrics_tolerate_nan_from_pandas(tmp_path):
    """Metrics come out of a DataFrame row, so a NaN is a real possibility; it must
    normalize to None rather than poisoning a box plot's domain."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_query_metrics(
            executor="e", query="q", precision=float("nan"), recall=0.5,
            precision_guarantee=0.7, recall_guarantee=0.7,
        )
        _drain(c)
        (row,) = c.snapshot_aggregates()["query_metrics"]
        assert row["precision_achieved"] is None
        assert row["precision_met"] is None
        assert row["recall_met"] == pytest.approx(0.5 / 0.7)
    finally:
        c.close()


def test_one_operator_at_several_plan_steps_stays_several_rows(tmp_path):
    """A plan can run the *same* physical operator - same class, same backend, same
    ratio - at several positions with different prompts. `get_operation_identifier()`
    is identical for all of them, so without the step tag the per-query fold would report
    their summed time and tuples as one operator's. Split on `step_expression`; a call
    without one still folds into a single row.

    The run-wide `operator_buckets` deliberately do *not* split: across a sweep the
    distinction is noise and one bucket per prompt per configuration is not.
    """
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_query_start(query="q", query_index=0, n_queries=1, worker_id="w")
        for expression, seconds, rows in (
            ("{t.text} mentions a director", 1.0, 30),
            ("{t.text} praises an actor", 2.0, 20),
            ("{t.text} names the movie", 4.0, 10),
        ):
            monitor.record_operator_run(
                operator="TextQaExtract-cr0.8", operation_class="TextQaExtract",
                phase="execution", seconds=seconds, n_input_rows=rows, step_expression=expression, worker_id="w",
            )
        monitor.record_query_end(query="q", cached=False, component_times={}, worker_id="w")
        _drain(c)

        (run,) = c.snapshot_query_detail("q")["runs"]
        ops = sorted(run["operators"], key=lambda o: o["seconds"])
        assert [o["seconds"] for o in ops] == [1.0, 2.0, 4.0]
        assert [o["input_rows"] for o in ops] == [30, 20, 10]
        assert [o["step_expression"] for o in ops] == [
            "{t.text} mentions a director",
            "{t.text} praises an actor",
            "{t.text} names the movie",
        ]
        # Same three calls, pooled, in the sweep-wide aggregate.
        (bucket,) = [
            b for b in c.snapshot_aggregates()["operator_buckets"]
            if b["operation_class"] == "TextQaExtract"
        ]
        assert (bucket["calls"], bucket["seconds"], bucket["input_rows"]) == (3, 7.0, 60)
    finally:
        c.close()


def test_untagged_operator_calls_still_fold_together(tmp_path):
    """Calls without a `step_expression` fold into one row rather than each call
    becoming its own bar."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_query_start(query="q", query_index=0, n_queries=1, worker_id="w")
        for _ in range(3):
            monitor.record_operator_run(
                operator="TextQaExtract-cr0.8", operation_class="TextQaExtract",
                phase="execution", seconds=1.0, n_input_rows=10,
                worker_id="w",
            )
        monitor.record_query_end(query="q", cached=False, component_times={}, worker_id="w")
        _drain(c)

        (run,) = c.snapshot_query_detail("q")["runs"]
        (op,) = run["operators"]
        assert (op["calls"], op["seconds"], op["step_expression"]) == (3, 3.0, None)
    finally:
        c.close()


def test_query_metrics_carry_the_label_set_that_scored_them(tmp_path):
    """A benchmark with ground truth is scored twice - against silver (a full pass of
    the best model) and against gold (the ground-truth files) - and both land in this
    one append-only list. Without the label set on the record the two are
    indistinguishable rows for the same (query, guarantee), and the dashboard averages
    a silver precision together with a gold one."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        for labels, precision in (("silver", 0.84), ("gold", 0.91)):
            monitor.record_query_metrics(
                benchmark="artwork_random", executor="optim_global", query="q1",
                precision_guarantee=0.7, recall_guarantee=0.7,
                precision=precision, recall=0.7, labels=labels, job_id="j0",
            )
        _drain(c)
        rows = c.snapshot_aggregates()["query_metrics"]
        assert {r["labels"]: r["precision_achieved"] for r in rows} == {
            "silver": pytest.approx(0.84),
            "gold": pytest.approx(0.91),
        }
    finally:
        c.close()


def test_query_metrics_without_a_label_set_still_record(tmp_path):
    """Recordings without the field, such as those of the plain run_benchmark
    script, emit no label set; those rows must carry None rather than being dropped or
    defaulted to a label pass they were not scored against."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    try:
        monitor.record_query_metrics(
            benchmark="movie_random", executor="lotus", query="q1",
            precision_guarantee=0.7, recall_guarantee=0.7, precision=0.5, recall=0.5,
        )
        _drain(c)
        (row,) = c.snapshot_aggregates()["query_metrics"]
        assert row["labels"] is None
        assert row["precision_achieved"] == pytest.approx(0.5)
    finally:
        c.close()


def test_metrics_scored_by_the_coordinator_are_not_tagged_as_a_labelling_pass():
    """Accuracy rows must survive the accuracy panel's default "Sweep only" filter.

    `_apply_query_metrics` stamps `role` from the emitting worker's context. If scoring
    ran inside the job, that context would just have been set to "label" by the labelling
    pass that produced the labels, so every accuracy row would be tagged `role="label"`
    and hidden by the guarantee-satisfaction panel, which excludes labelling passes by
    default.

    `coordinator.producers.parameter_sweep.score_job` therefore emits from the
    coordinator, with a job_id and deliberately no worker_id, so a worker's label context
    cannot leak onto the row.
    """
    from reasondb.monitor.collector import _Aggregates
    from reasondb.monitor.events import make_event

    agg = _Aggregates()
    # A worker finishes its labelling pass, which must not leak into another's context.
    agg.apply(make_event(1, "executor_start", 0.0, {
        "executor": "silver", "role": "label", "worker_id": "w-a",
    }))
    agg.apply(make_event(2, "query_end", 1.0, {
        "executor": "silver", "role": "label", "query": "q1", "worker_id": "w-a",
    }))
    # ...and the coordinator scores that worker's finished job.
    agg.apply(make_event(3, "query_metrics", 2.0, {
        "benchmark": "bench_a", "executor": "storage_step0_tunetrue_nNone", "query": "q1",
        "precision_guarantee": 0.7, "recall_guarantee": 0.7,
        "precision": 0.9, "recall": 0.8, "f1_score": 0.85,
        "labels": "silver", "job_id": "t1-parameter_sweep-bench_a-s0-p0.7-r0.7",
    }))

    (row,) = agg.query_metrics
    assert row["role"] != "label", (
        "an accuracy row tagged as a labelling pass is dropped by the panel's default"
    )
    assert row["executor"] == "storage_step0_tunetrue_nNone"
    assert row["precision_met"] == pytest.approx(0.9 / 0.7)


def test_run_tiles_count_cache_hits_per_role_over_their_own_denominator():
    """The "From cache" tile must describe sweep passes, and divide by its own counter.

    Counting every cached query_end, labelling passes included, against a "Queries done"
    taken from the job queue would report sweeps as largely cached even when no measured
    query was: each job replays a labelling pass it has already cached. Cache hits are
    therefore counted per role, over a denominator of the same role.
    """
    from reasondb.monitor.collector import _RunState
    from reasondb.monitor.events import make_event

    state = _RunState()
    # Two sweep queries, both computed; three labelling replays, all cached.
    for i, (role, cached) in enumerate(
        [("sweep", False), ("sweep", False)] + [("label", True)] * 3
    ):
        state.apply(make_event(i + 1, "query_end", 0.0, {
            "executor": "storage_step0_tunetrue_nNone", "role": role,
            "query": f"q{i}", "cached": cached, "worker_id": "w-a",
        }))

    run = state.to_json()
    assert (run["queries_cached_sweep"], run["queries_done_sweep"]) == (0, 2)
    # The all-roles totals stay, so the labelling replays are visible rather than lost.
    assert (run["queries_cached"], run["queries_done"]) == (3, 5)


def test_an_untagged_query_counts_as_a_sweep_pass():
    """Matching `applyRoleMode`: a record that is not *known* to be labelling is never
    treated as labelling. A plain single-process run tags no roles at all, and its tile
    must still say something."""
    from reasondb.monitor.collector import _RunState
    from reasondb.monitor.events import make_event

    state = _RunState()
    state.apply(make_event(1, "query_end", 0.0, {"executor": "optim_global", "query": "q"}))

    run = state.to_json()
    assert (run["queries_done_sweep"], run["queries_done"]) == (1, 1)
