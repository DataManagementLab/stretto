# Single-operator KV cache results

The results behind Figure 6 (KV profile quality vs. runtime): the per-method metrics of the
single-operator experiment, which `scripts/plot_quality_runtime.py` reads.

## What goes here

The `filter_stats/` and `extract_stats/` folders that `scripts/run_benchmark_single_operator.py`
and `scripts/run_benchmark_single_operator_image.py` write under their `--output-dir`, unchanged:

```
artifacts/single_operator/
  filter_stats/<benchmark>/<press>/<split>/.../<method>_vs_<reference>_silver_metrics.csv
  extract_stats/<benchmark>/<press>/<split>/.../<method>_vs_<reference>_silver_metrics.csv
```

Only the `*_vs_*_metrics.csv` files are read; debug outputs and logs can be left out.

## Redrawing Figure 6

```sh
python scripts/plot_quality_runtime.py --results-dir artifacts/single_operator \
  --dataset email artwork --model llama llava
```
