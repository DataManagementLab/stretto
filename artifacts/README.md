# Paper artifacts

The scored results behind the paper's figures. These are the merged CSVs the plotting
scripts read; nothing here needs a GPU, a model server or a precompute store to redraw.

## What is here

One directory per experiment, in the layout the plotting scripts expect
(`<task>/merged/<benchmark>/<split>/<producer>.csv`), so this folder can be passed
straight to `--results-root`.

| task | producer | CSV | compares |
| --- | --- | --- | --- |
| `base01` | `baselines` | `baselines.csv` | Stretto against Lotus and Abacus |
| `samp01` | `sample_size` | `sample_size.csv` | cost and accuracy against profiling budget |
| `ops01` | `operator_count` | `operator_count.csv` | cost and accuracy against operator set and storage |
| `abl01` | `ablation` | `ablation.csv` | Stretto, minus compression, minus the optimizer |
| `mode01` | `baselines` | `baselines.csv` | global against per-step optimization, over the full operator set at n=150 |
| `ref01` | `label_reference` | `label_reference_{gold,silver}_metrics.csv` | model-derived against human labels |

The first five cover five random benchmarks each — `artwork_random_medium`,
`rotowire_random`, `movie_random_huge`, `email_random`, `ecommerce_random_large`. `ref01`
covers the five curated ones, which are the benchmarks carrying per-tuple ground truth.

`single_operator/` holds the single-operator KV cache results behind Figure 6 (see its
README).

`kvops/kvop01/` holds the KV-operator sweep behind the footprint-vs-speedup figure
(`kvop01/merged/<benchmark>/dev/kv_operator.csv`): the uncompressed default suite plus
one KV-compressed operator per state, against the suite without it.

## Redrawing the figures

```sh
# One task. Every figure family, both pooling rules, all dataset panels.
python scripts/plot_sweep.py --experiment base01 --results-root artifacts

# ref01 has its own script
python scripts/plot_label_reference.py --output-dirs artifacts/ref01/merged --figures strip

# The KV-operator sweep (reads artifacts/kvops by default)
python scripts/plot_kvop.py kvop01 --target-ratio 0.7 0.9 --compact --metric total --aggregate geomean
```

Figures land in the task's own directory unless `--figure-dir` (or `--out-dir`, for
`plot_label_reference.py`) says otherwise. See "The paper's figures" in the README at the
repository root for the exact command behind each figure.
