"""Query shape statistics have to survive all the way to a scored row.

``num_semops`` and friends are declared on every ``QueryShape`` and copied onto the
``Query`` it instantiates, and they are the only axis that describes query difficulty:
everything else on a ``query_metrics`` row describes how the query was
*run*, not how hard it is.

Two links in the chain are pinned here, and a break in either is silent on its own:

- ``Query.to_json`` keeps ``additional_info``, so it survives a round trip through a
  pinned ``queries.json``. Every coordinator job loads from that file, so otherwise the
  statistics would exist only in the process that drew the query set.
- ``evaluate()`` attaches them, although it runs in the coordinator's scorer - a process
  holding query strings and pickled predictions, which never built a benchmark.
"""

import pandas as pd

from reasondb.evaluation.row_signature import RowSignature
import pytest

from reasondb.executor import CostSummary
from reasondb.optimizer.profiler import ProfilingCost

from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.query_plan.logical_plan import LogicalFilter, LogicalPlan
from reasondb.query_plan.query import Queries, Query


def _cost():
    cost = CostSummary(ProfilingCost(1.0, 0.0), ProfilingCost(0.5, 0.0))
    cost.component_times = {"end_to_end": 2.0, "execution": 1.0}
    return cost


def _query_with_plan(text="q", **info):
    plan = LogicalPlan(
        [
            LogicalFilter(
                inputs=[VirtualTableIdentifier("rows")],
                output=VirtualTableIdentifier("output"),
                expression="{rows.text} matches p",
                explanation="",
                labels=None,
            )
        ]
    )
    return Query(text, _ground_truth_logical_plan=plan, additional_info=info or None)


def test_additional_info_round_trips_through_queries_json(tmp_path):
    original = _query_with_plan(num_semops=3, num_sem_filter=2, num_sem_extract=1)

    Queries(original).dump(tmp_path)
    (loaded,) = Queries.load(tmp_path).queries

    assert loaded.additional_info == {
        "num_semops": 3,
        "num_sem_filter": 2,
        "num_sem_extract": 1,
    }


def test_a_file_pinned_before_the_field_existed_still_loads(tmp_path):
    """A ``queries.json`` without an ``additional_info`` key still loads."""
    import json

    payload = _query_with_plan().to_json()
    del payload["additional_info"]
    (tmp_path / "queries.json").write_text(json.dumps([payload]))

    (loaded,) = Queries.load(tmp_path).queries
    assert loaded.additional_info == {}


def test_evaluate_puts_the_statistics_on_the_row_and_the_telemetry(tmp_path, monkeypatch):
    """Both destinations, from one argument: the returned DataFrame (which becomes the
    metrics CSV, hence plot_benchmark's --facets) and the monitor row (hence the
    dashboard's group-by)."""
    from reasondb.evaluation import evaluation

    recorded = []
    monkeypatch.setattr(evaluation._monitor, "is_enabled", lambda: True)
    monkeypatch.setattr(
        evaluation._monitor, "record_query_metrics", lambda **kw: recorded.append(kw)
    )

    df = RowSignature.of(pd.DataFrame({"a": ["x"]}))
    metrics_df = evaluation.evaluate(
        benchmark_name="fake",
        approach_name="optim_global",
        all_predictions={"q": df},
        all_labels={"q": df},
        all_costs={"q": _cost()},
        debug_root=tmp_path,
        query_stats={"q": {"num_semops": 3, "num_sem_filter": 2}},
    )

    assert metrics_df["num_semops"].tolist() == [3]
    assert metrics_df["num_sem_filter"].tolist() == [2]
    assert len(recorded) == 1
    assert recorded[0]["num_semops"] == 3
    assert recorded[0]["num_sem_filter"] == 2


def test_evaluate_without_statistics_is_unchanged(tmp_path):
    """A benchmark whose shapes declare no additional_info simply gets no columns."""
    from reasondb.evaluation import evaluation

    df = RowSignature.of(pd.DataFrame({"a": ["x"]}))
    metrics_df = evaluation.evaluate(
        benchmark_name="fake",
        approach_name="optim_global",
        all_predictions={"q": df},
        all_labels={"q": df},
        all_costs={"q": _cost()},
        debug_root=tmp_path,
        record_telemetry=False,
    )

    assert "num_semops" not in metrics_df.columns


def test_query_stats_reads_the_pinned_set_without_a_database(tmp_path, monkeypatch):
    """``Benchmark.query_stats`` is what the coordinator's scorer calls, in a process
    that has no database - so it must go through ``queries.json`` and nothing else."""
    from reasondb.evaluation.benchmark import RandomBenchmark

    class Fake(RandomBenchmark):
        @classmethod
        def name(cls):
            return "fake_random"

        @classmethod
        def dir(cls):
            return tmp_path / "fake_random"

        @property
        def has_ground_truth(self):
            return False

        @staticmethod
        def urls():
            return {}

        @staticmethod
        def download(split):
            raise AssertionError("not used")

        @classmethod
        def _load_database(cls, split):
            raise AssertionError("query_stats must not build a database")

        @classmethod
        def _get_query_shapes(cls):
            raise AssertionError("query_stats must not generate queries")

        @classmethod
        def _get_operator_options(cls):
            return []

    assert Fake.query_stats("dev") == {}, "unpinned set is empty, not an error"

    Queries(
        _query_with_plan("with stats", num_semops=4),
        _query_with_plan("without stats"),
    ).dump(Fake.benchmark_dir() / "dev")

    assert Fake.query_stats("dev") == {"with stats": {"num_semops": 4}}


@pytest.mark.parametrize(
    "name",
    ["num_semops", "num_sem_filter", "num_sem_extract", "num_sem_join", "num_tradops"],
)
def test_every_statistic_is_labelled_for_the_dashboard(name):
    from reasondb.monitor.dimensions import DIMENSION_LABELS

    assert name in DIMENSION_LABELS


def test_declared_statistics_match_the_shapes_that_declare_them():
    """A shape whose num_semops disagrees with its own placeholders lands in the wrong
    bucket of every group-by - which is worse than having no statistic at all."""
    from reasondb.evaluation.benchmark_registry import RANDOM_BENCHMARKS
    from reasondb.query_plan.logical_plan import LogicalExtract, LogicalJoin

    semantic = (LogicalFilter, LogicalExtract, LogicalJoin)
    mismatches = []
    for name, cls in sorted(RANDOM_BENCHMARKS.items()):
        for key, shapes in cls.get_query_shapes().items():
            for i, shape in enumerate(shapes):
                declared = shape.additional_info.get("num_semops")
                if declared is None:
                    continue
                counts = shape.get_required_operators_per_type()
                actual = sum(
                    n for t, n in counts.items() if issubclass(t, semantic)
                )
                if declared != actual:
                    mismatches.append(f"{name}/{key or '-'}#{i}: {declared} != {actual}")

    assert not mismatches, mismatches
