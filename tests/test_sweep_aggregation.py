"""The arithmetic behind the sweep figures, tested without drawing anything.

``reasondb.evaluation.sweep_frames`` is deliberately free of matplotlib and seaborn so
that the decisions it encodes - how a phase breakdown is derived from nested spans, and
what a cross-dataset "Overall" number means - can be pinned here rather than inspected
by eye in a PDF.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from reasondb.evaluation import sweep_frames as sf
from reasondb.monitor.phases import derive_phase_components


# ---------------------------------------------------------------------------
# geomean_shares
# ---------------------------------------------------------------------------


def test_segments_sum_to_the_geometric_mean_of_the_totals():
    """The bar's height is the geomean of the totals AND its segments sum to it.

    Both halves at once is the whole point of the share-based split: taking a geometric
    mean per phase independently satisfies neither.
    """
    per_dataset = np.array([[10.0, 5.0, 1.0], [40.0, 20.0, 4.0], [90.0, 45.0, 9.0]])
    segments = sf.geomean_shares(per_dataset)

    totals = per_dataset.sum(axis=1)
    expected_total = float(np.exp(np.mean(np.log(totals))))
    assert segments.sum() == pytest.approx(expected_total)


def test_a_uniform_ratio_between_two_arms_survives_the_geometric_mean():
    """If arm A costs 3x arm B on *every* dataset, the pooled bars are in ratio 3.

    This is the property a sum does not have - a sum reports the ratio on the largest
    dataset - and it is the reason the pooled panel is a geometric mean at all.
    """
    b = np.array([[1.0, 2.0], [100.0, 300.0], [7.0, 1.0]])
    a = b * 3.0
    assert sf.geomean_shares(a).sum() / sf.geomean_shares(b).sum() == pytest.approx(3.0)


def test_scaling_commutes():
    """geomean_shares(c * X) == c * geomean_shares(X), so seconds may become hours first."""
    values = np.array([[3.0, 1.0], [30.0, 20.0], [7.0, 0.5]])
    scaled = sf.geomean_shares(values * (1 / 3600))
    assert scaled == pytest.approx(sf.geomean_shares(values) / 3600)


def test_identity_for_a_single_dataset():
    values = np.array([[6.0, 3.0, 1.0]])
    assert sf.geomean_shares(values) == pytest.approx(values[0])


def test_identity_when_every_dataset_is_equal():
    row = [4.0, 2.0, 2.0]
    assert sf.geomean_shares(np.array([row, row, row])) == pytest.approx(row)


def test_a_zero_total_dataset_is_excluded_rather_than_zeroing_everything():
    """log(0) is -inf: one benchmark with no timings must not collapse the pooled bar."""
    with_zero = np.array([[10.0, 10.0], [0.0, 0.0], [40.0, 40.0]])
    without = np.array([[10.0, 10.0], [40.0, 40.0]])
    assert sf.geomean_shares(with_zero) == pytest.approx(sf.geomean_shares(without))
    assert sf.geomean_shares(with_zero).sum() > 0


def test_all_zero_input_is_all_zero_not_nan():
    assert sf.geomean_shares(np.zeros((3, 4))) == pytest.approx(np.zeros(4))


def test_a_phase_that_is_zero_everywhere_stays_zero():
    """``time_reasoning`` is structurally zero on sweep data; it must not gain mass."""
    values = np.array([[5.0, 0.0, 2.0], [50.0, 0.0, 20.0]])
    segments = sf.geomean_shares(values)
    assert segments[1] == 0.0
    assert segments.sum() > 0


def test_shares_are_normalized_when_a_row_does_not_sum_to_its_own_total():
    values = np.array([[1.0, 1.0], [2.0, 2.0]])
    segments = sf.geomean_shares(values)
    assert segments[0] == pytest.approx(segments[1])


# ---------------------------------------------------------------------------
# component_times
# ---------------------------------------------------------------------------


def _component_json(**kwargs) -> str:
    return json.dumps(kwargs)


def test_explode_component_times_partitions_end_to_end():
    """The six phase columns add up to end_to_end - the nesting has been undone."""
    df = pd.DataFrame(
        {
            "component_times": [
                _component_json(
                    configuring=6.0, tuning=64.0, profiling=60.0, execution=50.0,
                    end_to_end=130.0,
                ),
                _component_json(configuring=1.0, tuning=2.0, execution=3.0, end_to_end=9.0),
            ]
        }
    )
    out = sf.explode_component_times(df)
    assert (out[sf.PHASE_COLUMNS].sum(axis=1) - out["time_end_to_end"]).abs().max() == 0.0
    # tuning - profiling, not tuning: profiling is a span nested inside it.
    assert out["time_optimization"].iloc[0] == pytest.approx(4.0)
    assert out["time_other"].iloc[0] == pytest.approx(130.0 - (6 + 60 + 4 + 50))


def test_explode_component_times_ignores_optimizer_solve():
    """``optimizer_solve`` nests inside tuning; counting it again would double-charge."""
    df = pd.DataFrame(
        {
            "component_times": [
                _component_json(
                    tuning=10.0, profiling=4.0, optimizer_solve=5.0, execution=1.0,
                    end_to_end=12.0,
                )
            ]
        }
    )
    out = sf.explode_component_times(df)
    assert out["time_optimization"].iloc[0] == pytest.approx(6.0)
    assert out[sf.PHASE_COLUMNS].sum(axis=1).iloc[0] == pytest.approx(12.0)


@pytest.mark.parametrize("cell", [None, float("nan"), "", "{}", "not json", "[1, 2]"])
def test_explode_component_times_tolerates_junk(cell):
    out = sf.explode_component_times(pd.DataFrame({"component_times": [cell]}))
    assert out[sf.PHASE_COLUMNS].to_numpy().sum() == 0.0


def test_explode_component_times_accepts_an_already_parsed_dict():
    """The .parquet twin of every merged CSV round-trips the column as a dict."""
    parsed = pd.DataFrame({"component_times": [{"execution": 3.0, "end_to_end": 3.0}]})
    out = sf.explode_component_times(parsed)
    assert out["time_execution"].iloc[0] == pytest.approx(3.0)


def test_explode_matches_derive_phase_components_row_for_row():
    """This module must stay a wrapper, not a second implementation of the split."""
    raw = {"configuring": 2.0, "tuning": 9.0, "profiling": 4.0, "execution": 5.0, "end_to_end": 20.0}
    out = sf.explode_component_times(pd.DataFrame({"component_times": [json.dumps(raw)]}))
    expected = derive_phase_components(raw)
    for column in sf.PHASE_COLUMNS + ["time_end_to_end"]:
        assert out[column].iloc[0] == pytest.approx(expected[column])


def test_missing_component_times_column_yields_zeros_rather_than_raising():
    out = sf.explode_component_times(pd.DataFrame({"benchmark": ["movie_random"]}))
    assert out[sf.PHASE_COLUMNS].to_numpy().sum() == 0.0


# ---------------------------------------------------------------------------
# Operator counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("[0.5, 0.8]", [0.5, 0.8]),
        ("[]", []),
        (None, []),
        (float("nan"), []),
        ("", []),
        ([0.0, 0.5], [0.0, 0.5]),
        ("garbage", []),
    ],
)
def test_parse_crs(value, expected):
    assert sf.parse_crs(value) == expected


def test_add_operator_count_counts_levels_and_ignores_absent_slots():
    df = pd.DataFrame(
        {
            "benchmark": ["movie_random", "ecommerce_random_large"],
            "text_small_crs": ["[0.5, 0.8]", "[0.8]"],
            "text_large_crs": ["[0.8]", "[0.8]"],
            "image_small_crs": [None, "[0.5, 0.8]"],
            "image_large_crs": [None, "[0.8]"],
        }
    )
    out = sf.add_operator_count(df)
    assert list(out["n_cached_levels"]) == [3, 5]
    # One vanilla operator per modality the benchmark has: movie is text-only, so it
    # gains one; ecommerce is both, so it gains two.
    assert list(out["n_operators"]) == [4, 7]


def test_operator_count_survives_a_state_that_materializes_nothing():
    """The walk's terminal state caches nothing but still holds one operator per modality."""
    df = pd.DataFrame(
        {
            "benchmark": ["movie_random", "movie_random"],
            "text_small_crs": ["[0.8]", None],
            "text_large_crs": ["[0.8]", None],
            "image_small_crs": [None, None],
            "image_large_crs": [None, None],
        }
    )
    out = sf.add_operator_count(df)
    assert list(out["n_cached_levels"]) == [2, 0]
    assert list(out["n_operators"]) == [3, 1]


# ---------------------------------------------------------------------------
# The ablation's guarantee-blind arm
# ---------------------------------------------------------------------------


def _ablation_frame(blind_settings) -> pd.DataFrame:
    rows = []
    for setting in ("p:0.5_r:0.5", "p:0.7_r:0.7", "p:0.9_r:0.9"):
        for approach in ("optim_global",):
            rows.append(
                dict(benchmark="movie_random", split="dev", approach=approach,
                     guarantee_setting=setting, value=1.0)
            )
    for setting in blind_settings:
        rows.append(
            dict(benchmark="movie_random", split="dev", approach="no_optim",
                 guarantee_setting=setting, value=2.0)
        )
    return pd.DataFrame(rows)


def test_fan_out_completes_the_guarantee_axis():
    """COLLAPSE_GUARANTEE_AXIS runs the blind arm once; every facet still needs it."""
    out = sf.fan_out_guarantee_blind_rows(_ablation_frame(["p:0.5_r:0.5"]))
    blind = out[out["approach"] == "no_optim"]
    assert sorted(blind["guarantee_setting"]) == [
        "p:0.5_r:0.5", "p:0.7_r:0.7", "p:0.9_r:0.9",
    ]
    assert set(blind["value"]) == {2.0}


def test_fan_out_is_a_noop_when_the_arm_is_already_complete():
    """Flipping COLLAPSE_GUARANTEE_AXIS off must not double the arm."""
    complete = _ablation_frame(["p:0.5_r:0.5", "p:0.7_r:0.7", "p:0.9_r:0.9"])
    out = sf.fan_out_guarantee_blind_rows(complete)
    pd.testing.assert_frame_equal(out, complete)


def test_fan_out_is_a_noop_without_a_blind_arm():
    frame = _ablation_frame([])
    pd.testing.assert_frame_equal(sf.fan_out_guarantee_blind_rows(frame), frame)


def test_fan_out_does_not_invent_settings_another_benchmark_ran():
    """A task whose benchmarks used different guarantee grids must not be cross-filled."""
    frame = pd.concat(
        [
            _ablation_frame(["p:0.5_r:0.5"]),
            pd.DataFrame(
                [
                    dict(benchmark="artwork_random_medium", split="dev",
                         approach="optim_global", guarantee_setting="p:0.6_r:0.6", value=1.0),
                    dict(benchmark="artwork_random_medium", split="dev",
                         approach="no_optim", guarantee_setting="p:0.6_r:0.6", value=2.0),
                ]
            ),
        ],
        ignore_index=True,
    )
    out = sf.fan_out_guarantee_blind_rows(frame)
    movie_blind = out[(out["benchmark"] == "movie_random") & (out["approach"] == "no_optim")]
    assert "p:0.6_r:0.6" not in set(movie_blind["guarantee_setting"])


# ---------------------------------------------------------------------------
# An arm borrowed from another task
# ---------------------------------------------------------------------------


def _sweep_frame(queries=("q1", "q2"), benchmark="movie_random") -> pd.DataFrame:
    """A sample-size sweep: one approach, two sample sizes, three targets."""
    rows = [
        dict(benchmark=benchmark, split="dev", approach="optim_global", query=query,
             guarantee_setting=setting, arm=str(size), total_runtime_s=10.0,
             achieved_precision=0.9, achieved_recall=0.9, achieved_f1=0.9)
        for query in queries
        for setting in ("p:0.5_r:0.5", "p:0.7_r:0.7", "p:0.9_r:0.9")
        for size in (10, 100)
    ]
    return pd.DataFrame(rows)


def _reference_frame(queries=("q1", "q2"), settings=("p:0.5_r:0.5",),
                     benchmark="movie_random", perfect=True) -> pd.DataFrame:
    score = 1.0 if perfect else 0.8
    rows = [
        dict(benchmark=benchmark, split="dev", approach="no_optim", query=query,
             guarantee_setting=setting, arm="ignored", total_runtime_s=30.0,
             achieved_precision=score, achieved_recall=score, achieved_f1=score)
        for query in queries
        for setting in settings
    ]
    # An arm of the reference task that is not the one being borrowed.
    rows += [
        dict(benchmark=benchmark, split="dev", approach="optim_global", query=query,
             guarantee_setting="p:0.5_r:0.5", arm="ignored", total_runtime_s=20.0,
             achieved_precision=0.9, achieved_recall=0.9, achieved_f1=0.9)
        for query in queries
    ]
    return pd.DataFrame(rows)


def test_a_borrowed_arm_is_read_at_every_target_the_sweep_ran():
    """The arm never reads a guarantee, so one measurement stands at all three."""
    out = sf.reference_arm_rows(
        _sweep_frame(), _reference_frame(), sf.ReferenceArm()
    )
    assert sorted(out["guarantee_setting"].unique()) == [
        "p:0.5_r:0.5", "p:0.7_r:0.7", "p:0.9_r:0.9",
    ]
    assert set(out["arm"]) == {"No optimization"}
    assert set(out["approach"]) == {"no_optim"}
    # Two queries at three settings, and the reference's own optim_global rows left behind.
    assert len(out) == 6


def test_a_benchmark_whose_query_sets_differ_is_dropped_rather_than_compared():
    """Per-benchmark totals over different query sets have no ratio between them."""
    with pytest.raises(SystemExit):
        sf.reference_arm_rows(
            _sweep_frame(queries=("q1", "q2")),
            _reference_frame(queries=("q1", "q3")),
            sf.ReferenceArm(),
        )


def test_one_benchmark_may_drop_out_without_taking_the_others_with_it():
    sweep = pd.concat(
        [_sweep_frame(benchmark="movie_random"),
         _sweep_frame(benchmark="artwork_random_medium")],
        ignore_index=True,
    )
    reference = pd.concat(
        [_reference_frame(benchmark="movie_random"),
         _reference_frame(benchmark="artwork_random_medium", queries=("q1", "q9"))],
        ignore_index=True,
    )
    out = sf.reference_arm_rows(sweep, reference, sf.ReferenceArm())
    assert set(out["benchmark"]) == {"movie_random"}


def test_a_reference_run_per_target_keeps_its_own_rows():
    """Only a *blind* arm is replicated; one that read the targets is taken as measured."""
    reference = _reference_frame(settings=("p:0.5_r:0.5", "p:0.9_r:0.9"))
    reference.loc[reference["guarantee_setting"] == "p:0.9_r:0.9", "total_runtime_s"] = 40.0
    out = sf.reference_arm_rows(_sweep_frame(), reference, sf.ReferenceArm())
    per_setting = out.groupby("guarantee_setting")["total_runtime_s"].max().to_dict()
    assert per_setting == {"p:0.5_r:0.5": 30.0, "p:0.9_r:0.9": 40.0}
    assert "p:0.7_r:0.7" not in per_setting


def test_a_benchmark_the_sweep_never_ran_is_left_out():
    out = sf.reference_arm_rows(
        _sweep_frame(benchmark="movie_random"),
        pd.concat(
            [_reference_frame(benchmark="movie_random"),
             _reference_frame(benchmark="email_random")],
            ignore_index=True,
        ),
        sf.ReferenceArm(),
    )
    assert set(out["benchmark"]) == {"movie_random"}


def test_an_approach_the_reference_does_not_carry_is_an_error():
    with pytest.raises(SystemExit):
        sf.reference_arm_rows(
            _sweep_frame(), _reference_frame(), sf.ReferenceArm(approach="lotus")
        )


def test_the_tick_breaks_the_label_at_its_spaces():
    assert sf.ReferenceArm().tick == "No\noptimization"
    assert sf.ReferenceArm(label="Gold").tick == "Gold"


def test_a_perfect_score_on_every_query_is_recognized_as_a_tautology():
    """The silver labels are this arm's own plan, so its accuracy measures nothing."""
    perfect = _reference_frame()
    assert sf.scores_by_construction(perfect[perfect["approach"] == "no_optim"])
    scored = _reference_frame(perfect=False)
    assert not sf.scores_by_construction(scored[scored["approach"] == "no_optim"])


def test_an_unscored_frame_is_not_mistaken_for_a_perfect_one():
    """A task whose scoring pass never ran has NaNs, which are not 1.0."""
    rows = _reference_frame()
    for column in ("achieved_precision", "achieved_recall", "achieved_f1"):
        rows[column] = float("nan")
    assert not sf.scores_by_construction(rows)


def test_only_the_accuracy_metrics_are_declared_accuracy():
    accuracy = {k for k, m in sf.METRICS.items() if m.accuracy}
    assert accuracy == {"f1", "mean_precision", "mean_recall"}


# ---------------------------------------------------------------------------
# Frame-level aggregation
# ---------------------------------------------------------------------------


def _totals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["movie", "movie", "artwork", "artwork"],
            "guarantee_setting": ["p:0.7_r:0.7"] * 4,
            "arm": ["a", "b", "a", "b"],
            "time_execution": [1.0, 2.0, 100.0, 200.0],
            "time_profiling": [1.0, 1.0, 100.0, 100.0],
            "time_optimization": [0.0, 0.0, 0.0, 0.0],
            "time_configuring": [0.0, 0.0, 0.0, 0.0],
            "time_reasoning": [0.0, 0.0, 0.0, 0.0],
            "time_other": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_with_overall_appends_one_row_per_group_and_arm():
    out = sf.with_overall(_totals_frame(), "arm")
    pooled = out[out["dataset"] == sf.OVERALL]
    assert len(pooled) == 2
    # Arm b costs exactly 1.5x arm a on both datasets (3/2 and 300/200), so the pooled
    # bars must be in that ratio too.
    heights = pooled.set_index("arm")[sf.PHASE_COLUMNS].sum(axis=1)
    assert heights["b"] / heights["a"] == pytest.approx(1.5)


def test_with_overall_is_skipped_for_a_single_dataset():
    single = _totals_frame()
    single = single[single["dataset"] == "movie"]
    out = sf.with_overall(single, "arm")
    assert sf.OVERALL not in set(out["dataset"])


# ---------------------------------------------------------------------------
# The other pooling: a plain sum


def test_sum_overall_adds_the_datasets_phase_by_phase():
    """The fleet's bill: every dataset's hours, added up."""
    pooled = sf.sum_overall(_totals_frame(), "arm").set_index("arm")
    assert pooled.loc["a", "time_execution"] == pytest.approx(101.0)
    assert pooled.loc["b", "time_execution"] == pytest.approx(202.0)
    assert pooled.loc["a", "time_profiling"] == pytest.approx(101.0)
    # An arm that costs 1.5x on *both* datasets is 1.5x either way; the two rules only
    # part company when the per-dataset ratios differ (below).
    heights = pooled[sf.PHASE_COLUMNS].sum(axis=1)
    assert heights["b"] / heights["a"] == pytest.approx(1.5)


def test_the_two_poolings_part_company_when_the_ratio_is_not_the_same_everywhere():
    """Which is the whole reason both are drawn rather than one being chosen.

    Arm b is 2x on the small dataset and 1.01x on the large one. The geometric mean
    reports a ratio between the two; the sum reports the large dataset's, because that
    is where the machine time actually went.
    """
    totals = pd.DataFrame(
        {
            "dataset": ["movie", "movie", "artwork", "artwork"],
            "guarantee_setting": ["p:0.7_r:0.7"] * 4,
            "arm": ["a", "b", "a", "b"],
            "time_execution": [1.0, 2.0, 100.0, 101.0],
            "time_profiling": [0.0] * 4,
            "time_optimization": [0.0] * 4,
            "time_configuring": [0.0] * 4,
            "time_reasoning": [0.0] * 4,
            "time_other": [0.0] * 4,
        }
    )
    geo = sf.geomean_overall(totals, "arm").set_index("arm")["time_execution"]
    summed = sf.sum_overall(totals, "arm").set_index("arm")["time_execution"]
    assert geo["b"] / geo["a"] == pytest.approx(math.sqrt(2.0 * 1.01))
    assert summed["b"] / summed["a"] == pytest.approx(103.0 / 101.0)


def test_sum_overall_segments_still_sum_to_the_bar():
    pooled = sf.sum_overall(_totals_frame(), "arm")
    totals = _totals_frame()
    for arm in ("a", "b"):
        expected = totals[totals["arm"] == arm][sf.PHASE_COLUMNS].to_numpy().sum()
        assert pooled[pooled["arm"] == arm][sf.PHASE_COLUMNS].to_numpy().sum() == (
            pytest.approx(expected)
        )


def test_pooled_overall_dispatches_on_how():
    frame = _totals_frame()
    assert sf.pooled_overall(frame, "arm", how="geomean").equals(
        sf.geomean_overall(frame, "arm")
    )
    assert sf.pooled_overall(frame, "arm", how="sum").equals(sf.sum_overall(frame, "arm"))
    with pytest.raises(ValueError, match="unknown pooling"):
        sf.pooled_overall(frame, "arm", how="median")


def test_with_overall_takes_the_pooling_it_is_given():
    out = sf.with_overall(_totals_frame(), "arm", how="sum")
    pooled = out[out["dataset"] == sf.OVERALL].set_index("arm")
    assert pooled.loc["a", "time_execution"] == pytest.approx(101.0)


def test_a_runtime_answers_the_pooling_flag_and_an_accuracy_does_not():
    """`--pooling sum` is a choice between two ways of combining an extensive quantity.

    A sum of five datasets' F1 is not a number anyone has a use for, so the accuracy
    metrics stay on the arithmetic mean under either rule - and say so on their pooled
    panel, which is what stops one figure being read as the other.
    """
    runtime = sf.METRICS["total_runtime"]
    accuracy = sf.METRICS["f1"]
    assert runtime.pool_rule("geomean") == "geomean"
    assert runtime.pool_rule("sum") == "sum"
    assert runtime.varies_with_pooling
    assert accuracy.pool_rule("sum") == "mean"
    assert not accuracy.varies_with_pooling


def test_pool_metric_sums_a_runtime_and_still_averages_an_accuracy():
    totals = pd.DataFrame(
        {
            "dataset": ["movie", "artwork"],
            "guarantee_setting": ["p:0.7_r:0.7"] * 2,
            "arm": ["a", "a"],
            "total_runtime_s": [2.0, 8.0],
            "achieved_f1": [0.4, 0.8],
        }
    )
    summed = sf.pool_metric(totals, "arm", sf.METRICS["total_runtime"], how="sum")
    assert summed["total_runtime_s"].iloc[0] == pytest.approx(10.0)
    pooled_f1 = sf.pool_metric(totals, "arm", sf.METRICS["f1"], how="sum")
    assert pooled_f1["achieved_f1"].iloc[0] == pytest.approx(0.6)


def test_drop_all_zero_phases_keeps_only_what_is_visible():
    kept = sf.drop_all_zero_phases(_totals_frame())
    assert kept == ["time_execution", "time_profiling"]


def test_per_dataset_totals_sums_over_queries():
    df = pd.DataFrame(
        {
            "dataset": ["movie"] * 3,
            "guarantee_setting": ["p:0.7_r:0.7"] * 3,
            "arm": ["a"] * 3,
            "time_execution": [1.0, 2.0, 3.0],
        }
    )
    out = sf.per_dataset_totals(df, "arm", value_cols=["time_execution"])
    assert out["time_execution"].iloc[0] == pytest.approx(6.0)


def test_per_dataset_totals_applies_the_scale():
    df = pd.DataFrame(
        {
            "dataset": ["movie"],
            "guarantee_setting": ["p:0.7_r:0.7"],
            "arm": ["a"],
            "time_execution": [3600.0],
        }
    )
    out = sf.per_dataset_totals(df, "arm", value_cols=["time_execution"], scale=1 / 3600)
    assert out["time_execution"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Per-metric aggregation
# ---------------------------------------------------------------------------


def _metric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["movie", "movie", "artwork", "artwork"],
            "guarantee_setting": ["p:0.7_r:0.7"] * 4,
            "arm": ["a", "a", "a", "a"],
            "total_runtime_s": [3600.0, 7200.0, 36000.0, 72000.0],
            "achieved_f1": [0.8, 0.6, 0.9, 0.5],
        }
    )


def test_a_cost_metric_sums_over_queries_and_scales_to_hours():
    totals = sf.metric_totals(_metric_frame(), "arm", sf.METRICS["total_runtime"])
    by_dataset = totals.set_index("dataset")["total_runtime_s"]
    assert by_dataset["movie"] == pytest.approx(3.0)
    assert by_dataset["artwork"] == pytest.approx(30.0)


def test_an_accuracy_metric_averages_over_queries():
    totals = sf.metric_totals(_metric_frame(), "arm", sf.METRICS["f1"])
    by_dataset = totals.set_index("dataset")["achieved_f1"]
    assert by_dataset["movie"] == pytest.approx(0.7)
    assert by_dataset["artwork"] == pytest.approx(0.7)


def test_a_cost_metric_pools_by_geometric_mean():
    """3h and 30h pool to sqrt(90) ~ 9.49, not to 16.5 - artwork must not dominate."""
    metric = sf.METRICS["total_runtime"]
    totals = sf.metric_totals(_metric_frame(), "arm", metric)
    pooled = sf.with_overall_metric(totals, "arm", metric)
    value = pooled.loc[pooled["dataset"] == sf.OVERALL, "total_runtime_s"].iloc[0]
    assert value == pytest.approx(np.sqrt(3.0 * 30.0))


def test_an_accuracy_metric_pools_by_arithmetic_mean():
    """A geometric mean of two ratios bounded by 1 answers no question anyone asks."""
    metric = sf.METRICS["f1"]
    totals = sf.metric_totals(_metric_frame(), "arm", metric)
    pooled = sf.with_overall_metric(totals, "arm", metric)
    value = pooled.loc[pooled["dataset"] == sf.OVERALL, "achieved_f1"].iloc[0]
    assert value == pytest.approx(0.7)


def test_metric_totals_is_empty_when_the_column_is_absent_or_all_nan():
    frame = _metric_frame().drop(columns=["achieved_f1"])
    assert sf.metric_totals(frame, "arm", sf.METRICS["f1"]).empty
    frame = _metric_frame()
    frame["achieved_f1"] = float("nan")
    assert sf.metric_totals(frame, "arm", sf.METRICS["f1"]).empty


# ---------------------------------------------------------------------------
# Query complexity as an axis
# ---------------------------------------------------------------------------


def test_a_per_query_metric_averages_and_stays_in_seconds():
    """A bucketed axis holds different numbers of queries, so the sum is not the answer.

    Seconds rather than hours because one query is ~1e-2 hours: scaling would give an
    axis of zeros.
    """
    totals = sf.metric_totals(_metric_frame(), "arm", sf.METRICS["mean_total_runtime"])
    by_dataset = totals.set_index("dataset")["total_runtime_s"]
    assert by_dataset["movie"] == pytest.approx(5400.0)
    assert by_dataset["artwork"] == pytest.approx(54000.0)


def test_a_per_query_metric_still_pools_by_geometric_mean():
    """Only the within-dataset aggregation moved; artwork still must not dominate."""
    metric = sf.METRICS["mean_total_runtime"]
    totals = sf.metric_totals(_metric_frame(), "arm", metric)
    pooled = sf.with_overall_metric(totals, "arm", metric)
    value = pooled.loc[pooled["dataset"] == sf.OVERALL, "total_runtime_s"].iloc[0]
    assert value == pytest.approx(np.sqrt(5400.0 * 54000.0))


#: Rows a comparison accepts, where the shared frame below is not one of its experiments.
#:
#: ``ablation_arm`` and ``reorder_only_arm`` both refuse combinations that are not arms of
#: their own producer, and with ``reorder_only``'s arms being (`no_optim`, off) and
#: (`no_optim_reorder`, on) no single pair of rows is an arm of both. One frame per
#: comparison is the honest way to say that; what the test pins is the shape of what
#: ``build`` returns, not which rows it was given.
_COMPARISON_ROWS = {
    "reorder_only_arm": {
        "approach": ["no_optim_reorder", "no_optim"],
        "step": [0, 0],
        "reorder": [True, False],
    },
}


@pytest.mark.parametrize("name", sorted(sf.COMPARISONS))
def test_every_comparison_builds_a_string_arm_for_every_row(name):
    """``build`` feeds a column named "arm" that every consumer coerces with astype(str).

    A numeric Series survives the groupby and then fails to match the string arm_order,
    which draws empty bars rather than raising.
    """
    frame = pd.DataFrame(
        {
            "approach": ["optim_global", "no_optim"],
            "step": [0, 1],
            "sample_size": [100, pd.NA],
            "adaptive_sampling": [False, True],
            # One arm each of the ablation, and of every comparison that keys on a single
            # column. `_COMPARISON_ROWS` overrides these for the ones that need their own.
            "reorder": [True, False],
            "num_semops": [3, pd.NA],
            "dataset": ["movie", "movie"],
        }
    )
    for column, values in _COMPARISON_ROWS.get(name, {}).items():
        frame[column] = values
    arm = sf.COMPARISONS[name].build(frame)
    assert len(arm) == len(frame)
    assert list(arm.index) == list(frame.index)
    assert all(isinstance(v, str) for v in arm), f"{name} built a non-string arm"


def test_the_complexity_arm_is_an_integer_and_orders_numerically():
    """3.0 must tick as "3", and a bucketless row as "?" sorted last."""
    frame = pd.DataFrame({"num_semops": [3.0, 10.0, 2.0, None]})
    arm = sf.COMPARISONS["num_semops"].build(frame)
    assert list(arm) == ["3", "10", "2", "?"]
    ordered = sf.COMPARISONS["num_semops"].order(pd.DataFrame({"arm": arm}))
    assert ordered == ["2", "3", "10", "?"]


def test_the_complexity_axis_defaults_when_a_csv_predates_it(tmp_path):
    """A task merged without the column renders as one "?" bucket, not a KeyError."""
    csv = tmp_path / "baselines.csv"
    pd.DataFrame(
        {
            "benchmark": ["movie"],
            "split": ["dev"],
            "precision_guarantee": [0.7],
            "recall_guarantee": [0.7],
            "query": ["q"],
            "total_runtime_s": [1.0],
        }
    ).to_csv(csv, index=False)
    loaded = sf.load_sweep([csv])
    assert "num_semops" in loaded.columns
    assert list(sf.COMPARISONS["num_semops"].build(loaded)) == ["?"]


def _facet_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "benchmark": ["movie", "artwork"] * 2,
            "dataset": ["movie", "artwork"] * 2,
            "guarantee_setting": ["p:0.7_r:0.7"] * 4,
            "arm": ["optim_global"] * 4,
            "num_semops": [2, 2, 3, 3],
            "time_execution": [10.0, 1000.0, 20.0, 2000.0],
        }
    )


def test_facet_panels_keeps_the_benchmark_and_labels_the_bucket():
    out = sf.facet_panels(_facet_frame(), "num_semops")
    assert sorted(out["dataset"].unique()) == ["2 sem. ops", "3 sem. ops"]
    # The benchmark has to survive: the aggregation one level up pools it away, and a
    # sum across benchmarks inside a panel would report whichever one is slowest.
    assert sorted(out["benchmark"].unique()) == ["artwork", "movie"]


def test_facet_panels_drops_rows_with_no_bucket(caplog):
    frame = _facet_frame()
    frame.loc[0, "num_semops"] = None
    with caplog.at_level("WARNING"):
        out = sf.facet_panels(frame, "num_semops")
    assert len(out) == 3
    assert "dropped from the panels" in caplog.text


@pytest.mark.parametrize("mutate", [lambda f: f.drop(columns=["num_semops"]),
                                    lambda f: f.assign(num_semops=None)])
def test_facet_panels_says_re_merge_when_the_column_is_absent_or_empty(mutate):
    with pytest.raises(SystemExit, match="re-run merge"):
        sf.facet_panels(mutate(_facet_frame()), "num_semops")


def test_pooling_the_benchmarks_away_inside_a_panel_is_a_geometric_mean():
    """Benchmarks inside a panel pool by geometric mean, not by a sum across them.

    10s on movie and 1000s on artwork pool to sqrt(10*1000) ~ 100, not to 1010.
    """
    totals = sf.per_dataset_totals(
        sf.facet_panels(_facet_frame(), "num_semops"),
        "arm",
        value_cols=["time_execution"],
        group_cols=("benchmark", "dataset", "guarantee_setting"),
    )
    pooled = sf.geomean_overall(
        totals,
        "arm",
        ["time_execution"],
        group_cols=["dataset", "guarantee_setting"],
        dataset_col="benchmark",
        label="pooled",
    )
    assert len(pooled) == 2, "one row per (panel, guarantee, arm)"
    by_panel = pooled.set_index("dataset")["time_execution"]
    assert by_panel["2 sem. ops"] == pytest.approx(np.sqrt(10.0 * 1000.0))
    assert by_panel["3 sem. ops"] == pytest.approx(np.sqrt(20.0 * 2000.0))


def test_panels_sort_by_their_leading_number():
    """Lexicographic order would put "10 sem. ops" before "2 sem. ops"."""
    frame = pd.DataFrame({"dataset": ["10 sem. ops", "2 sem. ops", "3 sem. ops"]})
    assert sf.dataset_col_order(frame) == ["2 sem. ops", "3 sem. ops", "10 sem. ops"]
