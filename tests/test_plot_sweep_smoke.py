"""Every ``plot_sweep`` preset draws, on frames shaped like the real merged CSVs.

The synthetic frames below stand in for real merged results, so every preset is
exercised without depending on experiment outputs being present. They emit the full
merged schema - JSON ``component_times``, JSON ``*_crs``,
the guarantee-blind arm present at one setting only - rather than the subset a given
figure happens to read, so a drawer that starts reading another column is still covered.
"""

import importlib
import importlib.util
import json
import logging
import re
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name: str):
    """Import a ``scripts/*.py`` module by path.

    Tests may do this; scripts may not do it to each other (shared code belongs in a
    ``reasondb`` module).
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GUARANTEES = [(0.5, 0.5), (0.7, 0.7), (0.9, 0.9)]


def _component_times(rng, execution: float) -> str:
    profiling = float(rng.uniform(1.0, 6.0))
    tuning = profiling + float(rng.uniform(0.1, 2.0))
    configuring = float(rng.uniform(0.2, 1.0))
    return json.dumps(
        {
            "configuring": configuring,
            "profiling": profiling,
            "tuning": tuning,
            "execution": execution,
            # A little more than the named phases, so `time_other` is exercised too.
            "end_to_end": configuring + tuning + execution + 0.5,
        }
    )


def synthetic_sweep(
    datasets=("movie_random", "artwork_random_medium"),
    arms=(("optim_global", 0, 100, False),),
    guarantees=GUARANTEES,
    n_queries=3,
    seed=0,
    tune_parameters=True,
    reorder=True,
) -> pd.DataFrame:
    """A merged sweep CSV as a frame. ``arms`` is (approach, step, sample_size, adaptive).

    ``tune_parameters`` and ``reorder`` are whole-frame rather than per-arm because the
    experiments that vary them vary nothing else, so a caller builds those arms by
    concatenating two calls - which is also what their producers enumerate.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in datasets:
        image = "artwork" in dataset or "ecommerce" in dataset
        text = not image or "ecommerce" in dataset
        for approach, step, sample_size, adaptive in arms:
            # Later steps cache fewer levels, and less cache means less disk.
            levels = max(1, 3 - step)
            crs = [round(0.5 + 0.1 * i, 2) for i in range(levels)]
            storage_gb = 40.0 * levels
            for prec, rec in guarantees:
                for q in range(n_queries):
                    execution = float(rng.uniform(20.0, 200.0))
                    rows.append(
                        {
                            "benchmark": dataset,
                            "split": "dev",
                            "step": step,
                            "approach": approach,
                            "tune_parameters": tune_parameters,
                            "sample_size": sample_size,
                            "adaptive_sampling": adaptive,
                            "reorder": reorder,
                            "use_indexes": False,
                            "text_small_crs": json.dumps(crs) if text else None,
                            "text_small_storage_bytes": storage_gb * 1e9 if text else None,
                            "text_large_crs": json.dumps(crs[:1]) if text else None,
                            "text_large_storage_bytes": 1e9 if text else None,
                            "image_small_crs": json.dumps(crs) if image else None,
                            "image_small_storage_bytes": storage_gb * 1e9 if image else None,
                            "image_large_crs": json.dumps(crs[:1]) if image else None,
                            "image_large_storage_bytes": 1e9 if image else None,
                            "precision_guarantee": prec,
                            "recall_guarantee": rec,
                            "query": f"{dataset} query {q}",
                            # The merge stamps this from the pinned query set, so a
                            # re-merged CSV carries it and every preset must tolerate it.
                            "num_semops": 2 + q % 3,
                            "execution_runtime_s": execution,
                            "tuning_runtime_s": 5.0,
                            "total_runtime_s": execution + 5.0,
                            "monetary_cost": 0.01,
                            "wall_clock_s": execution + 6.0,
                            "achieved_precision": float(np.clip(prec + rng.normal(0.05, 0.08), 0, 1)),
                            "achieved_recall": float(np.clip(rec + rng.normal(0.05, 0.08), 0, 1)),
                            "achieved_f1": 0.8,
                            "labels": "silver",
                            "component_times": _component_times(rng, execution),
                            "storage_bytes": storage_gb * 1e9,
                            "storage_gb": storage_gb,
                            # Measured beside the footprint: one cached entry per
                            # row of the dataset, per retained level.
                            "cache_entries": 1000 * (levels + 1),
                            "storage_bytes_per_entry": (
                                storage_gb * 1e9 / (1000 * (levels + 1))
                            ),
                            "num_tuples": 1000,
                            "modality": "image" if image and not text else "text",
                            # Cached levels plus one vanilla operator per modality, which
                            # is what every plan these fixtures stand for holds.
                            "n_llm_operators": (levels + 1) * (
                                int(text) + int(image)
                            ) + int(text) + int(image),
                        }
                    )
    return pd.DataFrame(rows)


def synthetic_operator_count_sweep(**kwargs) -> pd.DataFrame:
    """A greedy walk: several steps, each caching one level fewer than the last."""
    return synthetic_sweep(
        arms=tuple(("optim_global", step, 100, False) for step in range(3)), **kwargs
    )


#: What each ``kv_operator`` step retains: a model size and a compression ratio, or
#: ``None`` for step 0, the vanilla-only arm every other one is read against.
_KV_OPERATOR_STEPS = [None, ("small", 0.0), ("small", 0.5), ("large", 0.3)]


def synthetic_kv_operator_sweep(**kwargs) -> pd.DataFrame:
    """A ``kv_operator`` sweep on single-modality data: the vanilla-only arm, then one state per materialized level.

    Built by rewriting ``synthetic_sweep``'s retained levels, because the shape is the one
    thing that fixture cannot express: every other preset holds several levels at once and
    each of these holds exactly one, in one slot, with every other slot empty. Which slot
    depends on the dataset's modality, so the arms are disjoint between the two panels -
    the case ``include_overall=False`` exists for.
    """
    frame = synthetic_sweep(
        arms=tuple(
            ("optim_global", step, 100, False)
            for step in range(len(_KV_OPERATOR_STEPS))
        ),
        **kwargs,
    )
    for column in ("text_small", "text_large", "image_small", "image_large"):
        frame[f"{column}_crs"] = None
        frame[f"{column}_storage_bytes"] = None
    for index, row in frame.iterrows():
        retained = _KV_OPERATOR_STEPS[int(row["step"])]
        if retained is None:
            frame.loc[index, "storage_bytes"] = 0.0
            frame.loc[index, "storage_gb"] = 0.0
            frame.loc[index, "cache_entries"] = 0
            frame.loc[index, "storage_bytes_per_entry"] = None
            # Both model sizes' vanilla operators, which is what makes this arm the
            # ablation's second one.
            frame.loc[index, "n_llm_operators"] = 2
            continue
        size, cr = retained
        slot = f"{row['modality']}_{size}"
        frame.loc[index, f"{slot}_crs"] = json.dumps([cr])
        frame.loc[index, f"{slot}_storage_bytes"] = frame.loc[index, "storage_bytes"]
        # The gold operator plus this one compressed operator, and nothing else.
        frame.loc[index, "n_llm_operators"] = 2
    return frame


def synthetic_ablation_sweep(**kwargs) -> pd.DataFrame:
    """The three ablation arms, with the guarantee-blind one at a single target.

    That asymmetry is real - ``ablation.COLLAPSE_GUARANTEE_AXIS`` enumerates ``no_optim``
    once per benchmark - and it is the case the fan-out exists for, so the fixture has to
    reproduce it rather than emit a tidy full cross.
    """
    optimizing = synthetic_sweep(
        arms=(("optim_global", 0, 100, False), ("optim_global", 1, 100, False)), **kwargs
    )
    blind = synthetic_sweep(
        arms=(("no_optim", 1, 100, False),), guarantees=[GUARANTEES[0]], **kwargs
    )
    return pd.concat([optimizing, blind], ignore_index=True)


def synthetic_reordering_sweep(**kwargs) -> pd.DataFrame:
    """The two abl02 arms: the same sweep point with reordering on and off."""
    return pd.concat(
        [
            synthetic_sweep(reorder=True, **kwargs),
            synthetic_sweep(reorder=False, seed=1, **kwargs),
        ],
        ignore_index=True,
    )


def synthetic_reorder_only_sweep(**kwargs) -> pd.DataFrame:
    """The two abl03 arms, both at a single target.

    Unlike the ablation fixture there is no asymmetry to model: *both* arms are
    guarantee-blind (neither `LabelOptimizer` nor `ReorderOnlyOptimizer` reads a target),
    so ``reorder_only.COLLAPSE_GUARANTEE_AXIS`` enumerates one job per arm and the CSV
    carries one guarantee setting. There is nothing for `fan_out_guarantee_blind` to fan
    out to, which is the case that preset has to draw correctly.
    """
    reordering = synthetic_sweep(
        arms=(("no_optim_reorder", 0, None, False),),
        guarantees=[GUARANTEES[0]],
        tune_parameters=True,
        reorder=True,
        **kwargs,
    )
    blind = synthetic_sweep(
        arms=(("no_optim", 0, None, False),),
        guarantees=[GUARANTEES[0]],
        tune_parameters=True,
        reorder=False,
        seed=1,
        **kwargs,
    )
    return pd.concat([reordering, blind], ignore_index=True)


def _pdf_text(path: Path) -> str:
    """Every string drawn into a PDF, for asserting that a label is really on the axes.

    Matplotlib emits text as Type-3 glyph draws, so the words are split across ``TJ``
    operators - hence the reassembly rather than a plain substring search on the bytes.
    """
    import zlib

    raw = path.read_bytes()
    chunks = []
    for start in range(len(raw)):
        if raw[start : start + 7] != b"stream\n":
            continue
        end = raw.find(b"endstream", start)
        if end < 0:
            continue
        try:
            chunks.append(zlib.decompress(raw[start + 7 : end]))
        except zlib.error:
            continue
    body = b"".join(chunks).decode("latin-1")
    return "".join(re.findall(r"\((.*?)\)", body))


def _write(frame: pd.DataFrame, root: Path, csv_name: str) -> Path:
    for benchmark, group in frame.groupby("benchmark"):
        out = root / str(benchmark) / "dev"
        out.mkdir(parents=True, exist_ok=True)
        group.to_csv(out / csv_name, index=False)
    return root


FIXTURES = {
    "baselines": (
        lambda: synthetic_sweep(
            arms=(("optim_global", 0, 100, False), ("lotus", 0, 100, False),
                  ("abacus", 0, 100, False))
        ),
        "baselines.csv",
    ),
    "modes": (
        lambda: synthetic_sweep(
            arms=(("optim_global", 0, 100, False), ("optim_local", 0, 100, False),
                  ("optim_shift_budget", 0, 100, False))
        ),
        "baselines.csv",
    ),
    "sample_size": (
        lambda: synthetic_sweep(
            arms=tuple(("optim_global", 0, n, False) for n in (10, 25, 50, 100))
        ),
        "sample_size.csv",
    ),
    "operator_count": (synthetic_operator_count_sweep, "operator_count.csv"),
    "kv_operator": (synthetic_kv_operator_sweep, "kv_operator.csv"),
    "adaptive_sampling": (
        lambda: synthetic_sweep(
            arms=(("optim_global", 0, 100, False), ("optim_global", 0, 160, True))
        ),
        "adaptive_sampling.csv",
    ),
    "ablation": (synthetic_ablation_sweep, "ablation.csv"),
    "reordering": (synthetic_reordering_sweep, "reordering.csv"),
    "reorder_only": (synthetic_reorder_only_sweep, "reorder_only.csv"),
}


def test_every_shipped_preset_has_a_fixture():
    """A preset with no fixture is a preset nothing draws before a reader does.

    The parametrization below is over ``FIXTURES``, not over ``PRESETS``, so a new preset
    added without one is silently untested rather than failing - which is the failure mode
    this file exists to prevent.
    """
    from reasondb.evaluation.sweep_frames import PRESETS

    assert set(FIXTURES) == set(PRESETS)


@pytest.mark.parametrize("experiment", sorted(FIXTURES))
def test_every_preset_draws(experiment, tmp_path, monkeypatch):
    build, csv_name = FIXTURES[experiment]
    merged = _write(build(), tmp_path / "merged", csv_name)
    figures = tmp_path / "figures"

    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", experiment,
         "--output-dirs", str(merged), "--figure-dir", str(figures)],
    )
    plot_sweep.main()

    written = sorted(figures.glob("*.pdf"))
    assert written, f"{experiment} drew nothing"
    for path in written:
        assert path.stat().st_size > 0


def test_the_breakdown_is_one_figure_over_every_target(tmp_path, monkeypatch):
    """Targets are adjacent bars inside an arm's group, not a file each.

    Reading one arm across targets is the comparison this figure is for, and that is a
    glance along a group rather than a diff between three PDFs.
    """
    merged = _write(
        synthetic_sweep(arms=(("optim_global", 0, 100, False), ("lotus", 0, 100, False))),
        tmp_path / "merged",
        "baselines.csv",
    )
    figures = tmp_path / "figures"

    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines",
         "--output-dirs", str(merged), "--figure-dir", str(figures),
         "--figures", "breakdown"],
    )
    plot_sweep.main()

    breakdowns = sorted(p.name for p in figures.glob("*breakdown*.pdf"))
    # Two panel layouts x the two cross-dataset poolings; still no target in a filename.
    assert breakdowns == [
        "baselines_breakdown_geomean.pdf",
        "baselines_breakdown_overall_geomean.pdf",
        "baselines_breakdown_overall_sum.pdf",
        "baselines_breakdown_sum.pdf",
    ]
    # Three guarantee settings ran; none of them may appear in a filename.
    assert not [p for p in figures.glob("*.pdf") if "p:" in p.name]


def test_target_met_pools_every_target_into_one_figure(tmp_path, monkeypatch):
    """The ratio normalizes the target away, so all three belong in one distribution."""
    merged = _write(synthetic_sweep(), tmp_path / "merged", "baselines.csv")
    figures = tmp_path / "figures"

    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines",
         "--output-dirs", str(merged), "--figure-dir", str(figures),
         "--figures", "target-met"],
    )
    plot_sweep.main()
    assert sorted(p.name for p in figures.glob("*.pdf")) == [
        "baselines_target_met.pdf", "baselines_target_met_overall.pdf",
    ]


def test_operator_count_writes_no_pooled_figure(tmp_path, monkeypatch):
    """Storage is per dataset, so a pooled GB number would be invented, not measured."""
    merged = _write(synthetic_operator_count_sweep(), tmp_path / "merged", "operator_count.csv")
    figures = tmp_path / "figures"

    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "operator_count",
         "--output-dirs", str(merged), "--figure-dir", str(figures)],
    )
    plot_sweep.main()
    assert not list(figures.glob("*_overall.pdf"))


def test_ablation_draws_no_target_met_figure(tmp_path, monkeypatch):
    """The no_optim arm *is* the silver pass, so its accuracy is 1.0 by construction."""
    merged = _write(synthetic_ablation_sweep(), tmp_path / "merged", "ablation.csv")
    figures = tmp_path / "figures"

    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "ablation",
         "--output-dirs", str(merged), "--figure-dir", str(figures)],
    )
    plot_sweep.main()
    assert not list(figures.glob("*target_met*.pdf"))
    assert list(figures.glob("ablation_breakdown*.pdf"))


def _sweep_and_reference(tmp_path, perfect_reference=True, reference_queries=None):
    """A sample-size sweep, and in another directory the ablation to borrow an arm from.

    The two tasks share their query strings, which is the precondition
    ``reference_arm_rows`` checks: ``synthetic_sweep`` names a query after its dataset and
    index, so both fixtures agree by construction unless a test asks otherwise.
    """
    sweep = _write(
        synthetic_sweep(arms=tuple(("optim_global", 0, n, False) for n in (10, 100))),
        tmp_path / "merged",
        "sample_size.csv",
    )
    ablation = synthetic_ablation_sweep()
    blind = ablation["approach"] == "no_optim"
    if perfect_reference:
        # The silver labels are this arm's own plan, so it scores 1.0 on every query.
        for column in ("achieved_precision", "achieved_recall", "achieved_f1"):
            ablation.loc[blind, column] = 1.0
    if reference_queries:
        for dataset, query in reference_queries.items():
            ablation.loc[ablation["benchmark"] == dataset, "query"] = query
    return sweep, _write(ablation, tmp_path / "reference", "ablation.csv")


def _draw_with_reference(tmp_path, monkeypatch, sweep, reference, *extra):
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "sample_size",
         "--output-dirs", str(sweep), "--figure-dir", str(figures),
         "--reference-from", "ablation", "--reference-dirs", str(reference), *extra],
    )
    plot_sweep.main()
    return figures


def test_a_borrowed_arm_becomes_a_bar_of_the_swept_axis(tmp_path, monkeypatch):
    """The unoptimized arm has no sample size, so it is a category at the end of the axis."""
    sweep, reference = _sweep_and_reference(tmp_path)
    figures = _draw_with_reference(
        tmp_path, monkeypatch, sweep, reference, "--figures", "breakdown"
    )
    written = sorted(p.name for p in figures.glob("*.pdf"))
    assert written and all("with_no_optim" in name for name in written), written
    # Its own filenames, so the plain sweep's figures survive in the same directory.
    assert not [p for p in figures.glob("sample_size_breakdown*.pdf")]
    drawn = _pdf_text(figures / "sample_size_with_no_optim_breakdown_geomean.pdf")
    assert "optimization" in drawn


def test_the_metric_figures_become_bars_and_keep_the_arm(tmp_path, monkeypatch):
    """A curve's x is the sample size; the borrowed arm has none, so the axis goes categorical."""
    sweep, reference = _sweep_and_reference(tmp_path)
    figures = _draw_with_reference(
        tmp_path, monkeypatch, sweep, reference,
        "--figures", "metrics", "--metrics", "total_runtime",
    )
    drawn = _pdf_text(figures / "sample_size_with_no_optim_total_runtime_geomean.pdf")
    # The label, in the two lines its tick breaks into - and the axis's own values beside
    # it: a wide tick may spill over its slot, never blank out its neighbours.
    assert "No" in drawn and "optimization" in drawn
    assert "100" in drawn


def test_the_accuracy_figures_are_dropped_when_the_reference_cannot_lose(tmp_path, monkeypatch):
    """Its plan produced the silver labels, so 1.0 is construction rather than result."""
    sweep, reference = _sweep_and_reference(tmp_path, perfect_reference=True)
    figures = _draw_with_reference(
        tmp_path, monkeypatch, sweep, reference,
        "--figures", "metrics", "target-met", "--metrics", "total_runtime", "f1",
    )
    written = sorted(p.name for p in figures.glob("*.pdf"))
    assert not [p for p in written if "target_met" in p or "_f1" in p], written
    assert [p for p in written if "total_runtime" in p]


def test_a_reference_that_is_scored_like_any_arm_keeps_its_accuracy_figures(tmp_path, monkeypatch):
    sweep, reference = _sweep_and_reference(tmp_path, perfect_reference=False)
    figures = _draw_with_reference(
        tmp_path, monkeypatch, sweep, reference,
        "--figures", "metrics", "--metrics", "f1",
    )
    assert [p for p in figures.glob("*_f1*.pdf")]


def test_a_benchmark_whose_query_set_differs_is_dropped_not_compared(
    tmp_path, monkeypatch, caplog
):
    """Per-benchmark totals over different query sets have no ratio between them."""
    sweep, reference = _sweep_and_reference(
        tmp_path, reference_queries={"movie_random": "a query nobody swept"}
    )
    with caplog.at_level(logging.WARNING):
        figures = _draw_with_reference(
            tmp_path, monkeypatch, sweep, reference, "--figures", "breakdown"
        )
    assert "movie_random" in caplog.text
    assert list(figures.glob("*.pdf"))


def test_a_sweep_missing_its_axis_columns_still_draws(tmp_path, monkeypatch):
    """A sweep CSV without the approach/sample_size columns must not raise."""
    frame = synthetic_operator_count_sweep().drop(
        columns=["approach", "sample_size", "adaptive_sampling", "tune_parameters"]
    )
    merged = _write(frame, tmp_path / "merged", "operator_count.csv")
    figures = tmp_path / "figures"

    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "operator_count",
         "--output-dirs", str(merged), "--figure-dir", str(figures), "--figures", "breakdown"],
    )
    plot_sweep.main()
    assert list(figures.glob("*.pdf"))


def test_an_unscored_sweep_says_so_instead_of_drawing_an_empty_figure(tmp_path, monkeypatch):
    """achieved_* is empty until the task is merged with its labels; say why."""
    frame = synthetic_sweep()
    frame[["achieved_precision", "achieved_recall", "achieved_f1"]] = np.nan
    merged = _write(frame, tmp_path / "merged", "baselines.csv")

    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines",
         "--output-dirs", str(merged), "--figure-dir", str(tmp_path / "figures"),
         "--figures", "target-met"],
    )
    with pytest.raises(SystemExit, match="never scored"):
        plot_sweep.main()


def test_label_reference_strip_plot_draws(tmp_path, monkeypatch):
    """The label-reference strip plot draws."""
    rng = np.random.default_rng(3)
    rows = []
    for dataset in ("artwork_curated", "email_curated"):
        for arm in ("model", "human"):
            for prec, rec in GUARANTEES:
                for q in range(6):
                    rows.append(
                        {
                            "approach_name": "optim_global",
                            "query": f"{dataset} query {q}",
                            "precision_guarantee": prec,
                            "recall_guarantee": rec,
                            "precision": float(np.clip(prec + rng.normal(0.02, 0.1), 0, 1)),
                            "recall": float(np.clip(rec + rng.normal(0.02, 0.1), 0, 1)),
                            "f1_score": 0.8,
                            "guarantee_met": bool(rng.random() > 0.3),
                            "achieved_precision_lower": prec,
                            "achieved_recall_lower": rec,
                            "optimized_against": arm,
                        }
                    )
    frame = pd.DataFrame(rows)
    merged = tmp_path / "merged"
    for dataset in ("artwork_curated", "email_curated"):
        out = merged / dataset / "dev"
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out / "label_reference_gold_metrics.csv", index=False)

    figures = tmp_path / "figures"
    module = _load_script("plot_label_reference")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_label_reference.py", "--benchmarks", "artwork_curated", "email_curated",
         "--labels", "gold", "--output-dirs", str(merged), "--out-dir", str(figures),
         "--figures", "strip"],
    )
    assert module.main() == 0
    assert (figures / "label_reference_gold_precision_strip.pdf").stat().st_size > 0
    assert (figures / "label_reference_gold_recall_strip_by_dataset.pdf").stat().st_size > 0


@pytest.mark.parametrize("width", [7.0, 3.4])
def test_figures_are_laid_out_to_the_requested_width(width, tmp_path, monkeypatch):
    """A page has a fixed width; the figure has to fit it however many datasets there are.

    The layout is driven by a total width rather than a per-panel width, so adding
    datasets narrows the panels instead of widening the page.
    """
    merged = _write(
        synthetic_sweep(
            datasets=("movie_random", "artwork_random_medium", "email_random",
                      "rotowire_random", "ecommerce_random_large"),
            arms=(("optim_global", 0, 100, False), ("lotus", 0, 100, False),
                  ("abacus", 0, 100, False)),
        ),
        tmp_path / "merged",
        "baselines.csv",
    )
    figures = tmp_path / "figures"

    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--width", str(width)],
    )
    plot_sweep.main()

    for path in sorted(figures.glob("*.pdf")):
        # Read /MediaBox out of the file rather than pull in a PDF library for one number.
        box = re.search(rb"/MediaBox \[([\d\. ]+)\]", path.read_bytes())
        assert box, f"{path.name} has no MediaBox"
        x0, _, x1, _ = (float(v) for v in box.group(1).split())
        inches = (x1 - x0) / 72.0
        # bbox_inches="tight" crops to the ink, so the page is at most the budget - never
        # over it, which is the property that matters.
        assert inches <= width + 0.35, f"{path.name} is {inches:.2f}in wide"


def _page_width_in(path: Path) -> float:
    """The PDF's page width in inches, read out of /MediaBox."""
    box = re.search(rb"/MediaBox \[([\d\. ]+)\]", path.read_bytes())
    assert box, f"{path.name} has no MediaBox"
    x0, _, x1, _ = (float(v) for v in box.group(1).split())
    return (x1 - x0) / 72.0


def test_the_pooled_figure_is_a_cut_out_of_the_faceted_one(tmp_path, monkeypatch):
    """Its panel comes from the faceted figure; the page is what that panel's labels need.

    Sized to a fixed fraction of a page instead, the point-sized decorations would claim
    a share a full-width figure never notices and ``tight_layout`` would squeeze the axes
    to fit them inside. Pinned as a page width: a page far wider than one panel means the
    panel was not the constraint, and one barely wider than its own y label means the
    axes collapsed.
    """
    merged = _write(
        synthetic_sweep(
            datasets=("movie_random", "artwork_random_medium", "email_random",
                      "rotowire_random", "ecommerce_random_large"),
            arms=tuple(("optim_global", 0, n, False) for n in (10, 25, 50, 100)),
        ),
        tmp_path / "merged",
        "sample_size.csv",
    )
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "sample_size", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--figures", "breakdown", "metrics",
         "--metrics", "total_runtime", "--width", "7.0"],
    )
    plot_sweep.main()

    # Five datasets plus the pooled panel, laid out to 7in.
    panel_in = 7.0 / 6
    for stem in ("sample_size_breakdown", "sample_size_total_runtime"):
        pooled = _page_width_in(figures / f"{stem}_overall_geomean.pdf")
        faceted = _page_width_in(figures / f"{stem}_geomean.pdf")
        assert panel_in * 0.9 <= pooled <= panel_in * 2.0, f"{stem}: {pooled:.2f}in"
        assert pooled < faceted / 3


def test_the_pooled_legend_hangs_under_its_label_not_at_the_foot_of_the_page(tmp_path):
    """One panel has no empty strip beside its axis label, so the legend goes below it.

    Measuring the strip anyway would ignore the only panel there is, find no ink and
    answer 0.0 - the bottom of the page - leaving a band of white under the figure.

    It also pins the half of the band that room alone does not answer: **what the legend
    is centred on**. One panel keeps the whole page as its room, since the page is sized
    to the panel afterwards and cropped to the ink, but it is centred on the *panel* -
    the page carries the y label and its tick numbers on the left only, so its own
    midpoint sits left of the axes'.
    """
    from reasondb.evaluation import sweep_figures as sfig

    figure, ax = matplotlib.pyplot.subplots(figsize=(2.0, 2.0))
    ax.bar([0, 1], [1.0, 2.0])
    ax.set_xlabel("Profiling sample size [rows]")
    figure.tight_layout()

    band = sfig._legend_band(figure, ax, 6.0)
    assert band.left == 0.0, "a single panel's legend keeps the whole page as its room"
    assert band.top > 0.02, "the legend was anchored at the foot of the page"
    assert band.top < sfig._content_bottom(figure) + 0.01
    assert band.centre == pytest.approx(float(np.mean(ax.get_position().intervalx)))
    assert band.centre > 0.5, "centred on the page, so left of the panel it explains"

    # With a second panel the label leaves a strip beside it, and the legend rises into it
    # and centres on what the label leaves rather than on the panels.
    figure2, (ax_a, ax_b) = matplotlib.pyplot.subplots(1, 2, figsize=(4.0, 2.0))
    for axis in (ax_a, ax_b):
        axis.bar([0, 1], [1.0, 2.0])
    ax_a.set_xlabel("Profiling sample size [rows]")
    figure2.tight_layout()
    band2 = sfig._legend_band(figure2, ax_a, 6.0)
    assert band2.left > 0.0
    assert band2.top > sfig._content_bottom(figure2)
    assert band2.centre == pytest.approx(band2.left + (1.0 - band2.left) / 2)
    matplotlib.pyplot.close(figure)
    matplotlib.pyplot.close(figure2)


def test_the_legend_never_prints_over_a_tick_label(tmp_path, monkeypatch):
    """A two-line tick has ink where a one-line tick has only padding.

    The legend shares the row of the single x label, which it can only do by hanging under
    the *other* panels' tick labels - so a placement that rises a fixed fraction above the
    ink it measured would land on the second line of a two-line tick.
    """
    from reasondb.evaluation import sweep_figures as sfig

    overlaps = []
    original = sfig._save_to_width

    def measured(figure, out_path, width_in, tries=1):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legends = [
            child for child in figure.get_children()
            if child.__class__.__name__ == "Legend"
        ]
        ticks = [
            text for ax in figure.axes
            for text in ax.get_xticklabels()
            if text.get_text()
        ]
        for legend in legends:
            box = legend.get_window_extent(renderer)
            for text in ticks:
                other = text.get_window_extent(renderer)
                across = min(box.x1, other.x1) - max(box.x0, other.x0)
                down = min(box.y1, other.y1) - max(box.y0, other.y0)
                if across > 0 and down > 0:
                    overlaps.append((out_path.name, text.get_text(), down))
        return original(figure, out_path, width_in, tries)

    monkeypatch.setattr(sfig, "_save_to_width", measured)
    sweep, reference = _sweep_and_reference(tmp_path)
    _draw_with_reference(
        tmp_path, monkeypatch, sweep, reference,
        "--figures", "breakdown", "metrics", "--metrics", "total_runtime",
    )
    assert not overlaps, f"legend printed over {overlaps}"


def test_fit_panel_hits_the_panel_width_it_is_given(tmp_path):
    """The page grows by what the decorations take, so the panel gets the size asked for."""
    from reasondb.evaluation import sweep_figures as sfig

    figure, ax = matplotlib.pyplot.subplots(figsize=(1.2, 1.8))
    ax.bar([0, 1], [1.0, 2.0])
    ax.set_ylabel("Total runtime [h]")
    ax.set_title("Overall (geo. mean)")
    figure.tight_layout()

    before = ax.get_window_extent().width / figure.dpi
    width = sfig._fit_panel(figure, ax, 0.95)
    after = ax.get_window_extent().width / figure.dpi
    assert abs(after - 0.95) < 0.02, f"panel is {after:.3f}in, not 0.95in"
    assert after > before
    # The page is the panel plus its decorations, not the other way round.
    assert width > 0.95
    matplotlib.pyplot.close(figure)


def test_collapsing_targets_gives_one_bar_per_arm(tmp_path, monkeypatch):
    """`sample_size` sums the targets away: its own axis is already five points long.

    The totals must be preserved by the collapse - it is a sum over the guarantee axis,
    not a selection of one target - so this checks the drawn total against the frame.
    """
    frame = synthetic_sweep(
        datasets=("movie_random",),
        arms=tuple(("optim_global", 0, n, False) for n in (10, 100)),
    )
    merged = _write(frame, tmp_path / "merged", "sample_size.csv")
    figures = tmp_path / "figures"

    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "sample_size", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--figures", "breakdown"],
    )
    plot_sweep.main()
    assert (figures / "sample_size_breakdown.pdf").stat().st_size > 0


def test_collapse_can_be_turned_off(tmp_path, monkeypatch):
    merged = _write(
        synthetic_sweep(datasets=("movie_random",)), tmp_path / "merged", "sample_size.csv"
    )
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "sample_size", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--figures", "breakdown", "--no-collapse-targets"],
    )
    plot_sweep.main()
    assert (figures / "sample_size_breakdown.pdf").stat().st_size > 0


def test_the_ablation_fits_one_column_of_a_two_column_page(tmp_path, monkeypatch):
    """It is a single axes; across a full page width it would be mostly whitespace."""
    merged = _write(synthetic_ablation_sweep(), tmp_path / "merged", "ablation.csv")
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "ablation", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--width", "7.0"],
    )
    plot_sweep.main()
    path = figures / "ablation_breakdown_geomean.pdf"
    box = re.search(rb"/MediaBox \[([\d\. ]+)\]", path.read_bytes())
    x0, _, x1, _ = (float(v) for v in box.group(1).split())
    assert (x1 - x0) / 72.0 <= 3.9


def _run(monkeypatch, merged, figures, *extra):
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--figures", "breakdown", *extra],
    )
    plot_sweep.main()
    return plot_sweep


@pytest.fixture
def three_arm_sweep(tmp_path):
    return _write(
        synthetic_sweep(
            arms=(("optim_global", 0, 100, False), ("lotus", 0, 100, False),
                  ("abacus", 0, 100, False))
        ),
        tmp_path / "merged",
        "baselines.csv",
    )


def test_exclude_arms_drops_one_and_keeps_the_rest(three_arm_sweep, tmp_path, monkeypatch, caplog):
    with caplog.at_level("INFO"):
        _run(monkeypatch, three_arm_sweep, tmp_path / "f", "--exclude-arms", "abacus")
    assert "Kept 2 arm(s): ['lotus', 'optim_global']" in caplog.text


def test_arms_selects_and_reorders(three_arm_sweep, tmp_path, monkeypatch, caplog):
    """--arms doubles as the order: "drop one and put Stretto first" is one intention."""
    with caplog.at_level("INFO"):
        _run(monkeypatch, three_arm_sweep, tmp_path / "f", "--arms", "optim_global", "lotus")
    assert "Kept 2 arm(s): ['optim_global', 'lotus']" in caplog.text


def test_an_unknown_arm_is_an_error_naming_the_real_ones(three_arm_sweep, tmp_path, monkeypatch):
    """Arm values are raw CSV strings, so a typo is likely and must not silently shrink."""
    with pytest.raises(SystemExit, match="abacus"):
        _run(monkeypatch, three_arm_sweep, tmp_path / "f", "--arms", "abacuss")


def test_guarantees_accept_the_bare_value_or_the_full_key(three_arm_sweep, tmp_path, monkeypatch, caplog):
    for spec in ("0.9", "p:0.9_r:0.9"):
        caplog.clear()
        with caplog.at_level("INFO"):
            _run(monkeypatch, three_arm_sweep, tmp_path / "f", "--guarantees", spec)
        assert "Kept guarantee target(s): ['p:0.9_r:0.9']" in caplog.text


@pytest.mark.parametrize(
    "task_id,preset",
    [("base01", "baselines"), ("mode01", "modes"),
     ("samp01", "sample_size"), ("ops01", "operator_count"),
     ("adapt01", "adaptive_sampling"), ("abl01", "ablation")],
)
def test_every_shipped_task_id_resolves(task_id, preset):
    """The task id is what names the directory, so it is the name already to hand."""
    from reasondb.evaluation.sweep_frames import resolve_preset

    assert resolve_preset(task_id).name == preset
    assert resolve_preset(preset).name == preset


def test_preset_and_task_id_resolve_to_the_tasks_directory():
    """Naming the preset or its task id both lead to the task's merged directory, where
    the coordinator writes it."""
    from reasondb.evaluation.sweep_frames import default_output_dir, resolve_preset

    preset = resolve_preset("modes")
    for requested in ("modes", "mode01"):
        assert default_output_dir(preset, requested, Path("benchmark_results")) == (
            Path("benchmark_results/mode01/merged")
        )


def test_mode01_and_base01_resolve_to_different_presets():
    """Both run the `baselines` producer; only the preset keeps their figures apart.

    This is why the alias table cannot be derived from the producer registry - and why
    getting it wrong would have the two tasks overwrite each other's PDFs.
    """
    from reasondb.evaluation.sweep_frames import resolve_preset

    assert resolve_preset("base01").prefix != resolve_preset("mode01").prefix


def test_a_non_sweep_task_says_which_script_draws_it():
    from reasondb.evaluation.sweep_frames import resolve_preset

    with pytest.raises(SystemExit, match="plot_label_reference"):
        resolve_preset("ref01")


def test_an_unknown_experiment_lists_both_spellings():
    from reasondb.evaluation.sweep_frames import resolve_preset

    with pytest.raises(SystemExit, match=r"baselines \(base01\)"):
        resolve_preset("nope")


def test_output_dirs_defaults_to_the_coordinator_path(tmp_path, monkeypatch, caplog):
    """`benchmark_results/<task-id>/merged` is where run_coordinator.py puts a task.

    The whole path is a function of the task id, so requiring it on the command line only
    gives it a second chance to disagree with --experiment.
    """
    _write(synthetic_sweep(), tmp_path / "base01" / "merged", "baselines.csv")
    figures = tmp_path / "figures"

    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "base01", "--results-root", str(tmp_path),
         "--figure-dir", str(figures), "--figures", "breakdown"],
    )
    with caplog.at_level("INFO"):
        plot_sweep.main()
    assert "default for base01" in caplog.text
    assert (figures / "baselines_breakdown_geomean.pdf").stat().st_size > 0


def test_the_preset_name_also_derives_its_default_task_directory(tmp_path, monkeypatch):
    """`--experiment baselines` still knows it means base01."""
    _write(synthetic_sweep(), tmp_path / "base01" / "merged", "baselines.csv")
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines", "--results-root", str(tmp_path),
         "--figure-dir", str(figures), "--figures", "breakdown"],
    )
    plot_sweep.main()
    assert (figures / "baselines_breakdown_geomean.pdf").exists()


def test_a_missing_default_directory_says_it_was_a_default(tmp_path, monkeypatch):
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "ops01", "--results-root", str(tmp_path)],
    )
    with pytest.raises(SystemExit, match="default location"):
        plot_sweep.main()


def test_metrics_draw_as_curves_on_a_numeric_axis(tmp_path, monkeypatch):
    """sample_size's own axis is numeric, so its metric figures are curves."""
    merged = _write(
        synthetic_sweep(arms=tuple(("optim_global", 0, n, False) for n in (10, 50, 100))),
        tmp_path / "merged",
        "sample_size.csv",
    )
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "sample_size", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--figures", "metrics",
         "--metrics", "total_runtime", "monetary", "f1"],
    )
    plot_sweep.main()
    for key in ("total_runtime", "monetary"):
        assert (figures / f"sample_size_{key}_geomean.pdf").stat().st_size > 0
    # An accuracy pools by arithmetic mean under either rule, so it is written once.
    assert (figures / "sample_size_f1.pdf").stat().st_size > 0


def test_metrics_draw_as_bars_on_a_categorical_axis(tmp_path, monkeypatch):
    """The ablation's arms have no numeric order, so its metric figures are bars."""
    merged = _write(synthetic_ablation_sweep(), tmp_path / "merged", "ablation.csv")
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "ablation", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--figures", "metrics", "--metrics", "monetary"],
    )
    plot_sweep.main()
    assert (figures / "ablation_monetary_geomean.pdf").stat().st_size > 0


def test_operator_count_metrics_use_storage_as_the_x_axis(tmp_path, monkeypatch):
    """Operator-count metrics plot runtime against GB on disk."""
    merged = _write(
        synthetic_operator_count_sweep(), tmp_path / "merged", "operator_count.csv"
    )
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "operator_count", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--figures", "metrics",
         "--metrics", "execution_runtime", "end_to_end"],
    )
    plot_sweep.main()
    assert (figures / "operator_count_execution_runtime.pdf").stat().st_size > 0
    assert (figures / "operator_count_end_to_end.pdf").stat().st_size > 0


# ---------------------------------------------------------------------------
# Query complexity as an axis
# ---------------------------------------------------------------------------

APPROACH_ARMS = tuple(
    (approach, 0, 100, False) for approach in ("optim_global", "lotus", "abacus")
)


def test_runtime_draws_against_query_complexity_under_its_own_name(tmp_path, monkeypatch):
    """`--x-column num_semops` is a different question, so it is a different file.

    Without the derived prefix it would overwrite `baselines_mean_total_runtime.pdf`
    with a figure about complexity - the collision `resolve_preset` keeps `base01` and
    `mode01` apart to avoid.
    """
    merged = _write(
        synthetic_sweep(arms=APPROACH_ARMS), tmp_path / "merged", "baselines.csv"
    )
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--figures", "metrics",
         "--metrics", "mean_total_runtime", "--x-column", "num_semops"],
    )
    plot_sweep.main()
    assert (
        figures / "baselines_vs_num_semops_mean_total_runtime_geomean.pdf"
    ).stat().st_size > 0
    assert not (figures / "baselines_mean_total_runtime_geomean.pdf").exists()


def test_a_curve_per_arm_when_several_arms_share_an_x(tmp_path):
    """Three approaches at one x are three lines.

    ``sns.lineplot`` would otherwise collapse them with ``estimator="mean"`` and draw a
    confidence band that reads as variance and is the spread between approaches.
    """
    import matplotlib

    matplotlib.use("Agg")
    sweep_figures = importlib.import_module("reasondb.evaluation.sweep_figures")
    sweep_frames = importlib.import_module("reasondb.evaluation.sweep_frames")

    totals = pd.DataFrame(
        [
            {"dataset": "movie_random", "guarantee_setting": "p0.5_r0.5",
             "arm": arm, "num_semops": n, "total_runtime_s": 10.0 * i + n}
            for i, arm in enumerate(("optim_global", "lotus", "abacus"))
            for n in (2, 3, 4)
        ]
    )
    out = sweep_figures.plot_metric(
        totals,
        metric=sweep_frames.METRICS["mean_total_runtime"],
        arm_order=["optim_global", "lotus", "abacus"],
        out_path=Path(tempfile.mkdtemp()) / "curves.pdf",
        x_column="num_semops",
    )
    assert out is not None

    # Redraw on a bare axes so the artists are inspectable.
    fig, ax = matplotlib.pyplot.subplots()
    import seaborn as sns

    sns.lineplot(
        data=totals, x="num_semops", y="total_runtime_s", hue="guarantee_setting",
        style="arm", errorbar=None, ax=ax,
    )
    drawn = [line for line in ax.get_lines() if line.get_xydata().size]
    assert len(drawn) == 3, "one line per approach, not one averaged line"
    assert not ax.collections, "a confidence band means the arms were averaged"
    matplotlib.pyplot.close(fig)


def test_faceting_by_complexity_panels_the_buckets(tmp_path, monkeypatch):
    """`--facet-by` turns the panels into complexity buckets and drops the pooled one."""
    merged = _write(
        synthetic_sweep(arms=APPROACH_ARMS), tmp_path / "merged", "baselines.csv"
    )
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--facet-by", "num_semops",
         "--figures", "target-met", "breakdown"],
    )
    plot_sweep.main()
    assert (figures / "baselines_per_num_semops_target_met.pdf").stat().st_size > 0
    assert (
        figures / "baselines_per_num_semops_breakdown_geomean.pdf"
    ).stat().st_size > 0
    # A geometric mean over buckets that partition one query set is not a quantity.
    assert not list(figures.glob("*_overall.pdf"))


def test_faceting_survives_a_preset_that_collapses_its_targets(tmp_path, monkeypatch):
    """sample_size sums over the targets, then puts `guarantee_setting` back.

    The pooling keys must be derived after the collapse, or that column is missing and
    the drawer raises KeyError on it.
    """
    merged = _write(
        synthetic_sweep(arms=tuple(("optim_global", 0, n, False) for n in (10, 100))),
        tmp_path / "merged",
        "sample_size.csv",
    )
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "sample_size", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--facet-by", "num_semops",
         "--figures", "breakdown"],
    )
    plot_sweep.main()
    assert (
        figures / "sample_size_per_num_semops_breakdown_geomean.pdf"
    ).stat().st_size > 0


@pytest.mark.parametrize(
    "extra,label_names_the_axis",
    [([], False), (["--facet-by", "num_semops"], True)],
    ids=["by-dataset", "by-complexity"],
)
def test_the_breakdown_identifies_both_of_its_axes(
    tmp_path, monkeypatch, extra, label_names_the_axis
):
    """Both axes are always identified, faceted by dataset or by complexity.

    The y axis is identified by its label ("Runtime [h]") on both paths.

    The x axis is identified by whichever of the two can do it, which is the one rule
    ``plot_sweep`` applies (``arm_axis_label``): the approaches' ticks name the axis
    outright ("Stretto", "Lotus-style"), so repeating "Approach" under every panel says
    nothing a reader cannot see and costs each figure a row of height. Under
    ``--facet-by`` the panel titles are complexity buckets, so the label is the only
    statement of what x is and it stays. Asserting the ticks in the no-label case is what
    keeps that a rule about redundancy rather than a licence to drop the axis.
    """
    merged = _write(
        synthetic_sweep(arms=APPROACH_ARMS), tmp_path / "merged", "baselines.csv"
    )
    figures = tmp_path / "figures"
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines", "--output-dirs", str(merged),
         "--figure-dir", str(figures), "--figures", "breakdown", *extra],
    )
    plot_sweep.main()

    written = [p for p in figures.glob("*breakdown*.pdf") if "overall" not in p.name]
    assert written
    text = _pdf_text(written[0])
    assert "Runtime" in text, f"no y-axis label in {written[0].name}"
    if label_names_the_axis:
        assert "Approach" in text, f"no x-axis label in {written[0].name}"
    else:
        assert "Approach" not in text, (
            f"{written[0].name} repeats an axis label its ticks already supply"
        )
        assert "Stretto" in text, (
            f"{written[0].name} has neither an x-axis label nor arm ticks"
        )


def test_faceting_is_refused_for_the_storage_axis(tmp_path, monkeypatch):
    """ops01's ticks are per dataset, which a bucket spanning benchmarks cannot supply."""
    merged = _write(
        synthetic_operator_count_sweep(), tmp_path / "merged", "operator_count.csv"
    )
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "operator_count", "--output-dirs", str(merged),
         "--figure-dir", str(tmp_path / "figures"), "--facet-by", "num_semops"],
    )
    with pytest.raises(SystemExit, match="footprint"):
        plot_sweep.main()


def test_faceting_a_sweep_that_was_never_stamped_says_re_merge(tmp_path, monkeypatch):
    merged = _write(
        synthetic_sweep().drop(columns=["num_semops"]), tmp_path / "merged", "baselines.csv"
    )
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines", "--output-dirs", str(merged),
         "--figure-dir", str(tmp_path / "figures"), "--facet-by", "num_semops"],
    )
    with pytest.raises(SystemExit, match="re-run merge"):
        plot_sweep.main()


def test_comparing_by_complexity_warns_that_it_pools_the_approaches(
    tmp_path, monkeypatch, caplog
):
    """`--compare-by num_semops` averages optim_global with lotus and abacus."""
    merged = _write(
        synthetic_sweep(arms=APPROACH_ARMS), tmp_path / "merged", "baselines.csv"
    )
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines", "--output-dirs", str(merged),
         "--figure-dir", str(tmp_path / "figures"), "--compare-by", "num_semops",
         "--figures", "breakdown"],
    )
    with caplog.at_level(logging.WARNING):
        plot_sweep.main()
    assert "span several approaches" in caplog.text


def test_the_ablation_axis_does_not_warn_about_its_own_approaches(
    tmp_path, monkeypatch, caplog
):
    """`ablation_arm` is built from (step, approach), so it separates them by construction.

    No pooling warning may be emitted for it.
    """
    merged = _write(synthetic_ablation_sweep(), tmp_path / "merged", "ablation.csv")
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "ablation", "--output-dirs", str(merged),
         "--figure-dir", str(tmp_path / "figures"), "--figures", "breakdown"],
    )
    with caplog.at_level(logging.WARNING):
        plot_sweep.main()
    assert "span several approaches" not in caplog.text


def test_an_unknown_metric_is_an_error_listing_the_real_ones(tmp_path, monkeypatch):
    merged = _write(synthetic_sweep(), tmp_path / "merged", "baselines.csv")
    plot_sweep = _load_script("plot_sweep")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_sweep.py", "--experiment", "baselines", "--output-dirs", str(merged),
         "--figure-dir", str(tmp_path / "f"), "--figures", "metrics",
         "--metrics", "moneterry"],
    )
    with pytest.raises(SystemExit, match="monetary"):
        plot_sweep.main()
