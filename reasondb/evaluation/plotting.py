"""Shared plotting helpers for benchmark figures.

Shared by the plotting scripts (``scripts/plot_benchmark.py`` and
``scripts/plot_sweep.py``): one label map, dataset ordering, style, and
axis/legend relabelling logic.
"""

from typing import Dict, Optional

import seaborn as sns
from matplotlib import pyplot as plt

# Preferred left-to-right ordering of dataset facets. Datasets not listed here
# fall through to alphabetical after these.
DATASET_ORDER = [
    "Overall",
    "artwork_random_medium",
    "rotowire_random",
    "email_random",
    "movie_random",
    "ecommerce_random_large",
]

# Human-readable names for machine identifiers that show up as axis labels,
# legend entries, tick labels, or facet titles ("<facet> = <value>").
LABEL_MAP: Dict[str, str] = {
    # approaches
    "true_output_cardinality": "Cardinality",
    "kv09": "KV-0.9",
    "kv00": "KV-0.0",
    "gpt": "GPT",
    "abacus": "Abacus\n-style",
    "lotus": "Lotus\n-style",
    "optim_local": "Local",
    "optim_shift_budget": "Stretto\n-independent",
    "optim_global": "Stretto",
    "no_optim": "No optimization",
    "no_optim_reorder": "Reordering only",
    "optim_combo": "Stretto Full",
    "optim_no_guarantee": "Stretto (no guarantees)",
    "approach_name": "Approach",
    # cost columns
    "total_cost_fake_cost": "Total Cost (Fake)",
    "execution_cost_fake_cost": "Execution Cost (Fake)",
    "total_cost_runtime": "Total Cost (Runtime)",
    "execution_cost_runtime": "Runtime [h]",
    "total_cost_monetary": "Total Cost (Monetary)",
    "execution_cost_monetary": "Execution Cost (Monetary)",
    "precision_met": "Precision Met",
    "recall_met": "Recall Met",
    # facet titles
    "dataset = Overall": "Overall",
    "no_facet = Overall": "Overall",
    "dataset = ecommerce_random_large": "Ecommerce",
    "dataset = artwork_random": "Artwork (small)",
    "dataset = artwork_random_medium": "Artwork",
    "dataset = rotowire_random": "Rotowire",
    "dataset = movie_random": "Movie",
    "dataset = movie_random_huge": "Movie",
    "dataset = email_random": "Enron Email",
    "dataset = ecommerce": "Ecommerce",
    # speedups
    "speedup_execution_cost_fake_cost": "Speedup Execution Cost (Fake)",
    "speedup_total_cost_fake_cost": "Speedup Total Cost (Fake)",
    "speedup_execution_cost_runtime": "Speedup Execution Cost (Runtime)",
    "speedup_total_cost_runtime": "Speedup Total Cost (Runtime)",
    "speedup_execution_cost_monetary": "Speedup Execution Cost (Monetary)",
    "speedup_total_cost_monetary": "Speedup Total Cost (Monetary)",
    # guarantee settings
    "p:0.5_r:0.5": "Prec=0.5/Rec=0.5",
    "p:0.7_r:0.7": "Prec=0.7/Rec=0.7",
    "p:0.8_r:0.8": "Prec=0.8/Rec=0.8",
    "p:0.9_r:0.9": "Prec=0.9/Rec=0.9",
    # facet counts
    "num_semops = 2": "2 Semantic Ops",
    "num_semops = 3": "3 Semantic Ops",
    "num_semops = 4": "4 Semantic Ops",
    "target_met": "Target Met",
    # storage / runtime axes
    "storage_gb": "Storage [GB]",
    "execution_runtime_s": "Execution time [s]",
    "wall_clock_s": "End-to-end runtime [s]",
    "total_runtime_s": "Total runtime [s]",
    # What one cached item costs, which is the storage axis a reader can carry to a
    # dataset nobody has measured. MB rather than bytes: a text item runs to a few of
    # them and an image item to tens, and neither reads as a nine-digit number.
    "storage_mb_per_entry": "Cache per item [MB]",
    "cache_entries": "Cached items",
    "tokens_per_item": "Tokens per item",
    "num_tuples": "Tuples",
    "modality": "Modality",
    "kv_cr": "Compression ratio",
    "kv_model_size": "Proxy model size",
    "kv_modality": "Operator modality",
    "n_llm_operators": "LLM operators",
    # kv_operator facet titles.
    "modality = text": "Text",
    "modality = image": "Image",
    "modality = multimodal": "Multi-modal",
}


def apply_default_style(font_size: int = 16) -> None:
    """Apply the shared matplotlib/seaborn paper style used across plots."""
    import scienceplots  # noqa: F401  (registers the "science" style)

    plt.rcParams.update({"font.size": font_size})
    plt.style.use(["science", "no-latex", "grid", "high-contrast"])


def fix_labels(g: sns.FacetGrid, name_map: Optional[Dict[str, str]] = None) -> None:
    """Relabel axes, legends, ticks, and facet titles via ``name_map``.

    Facet titles of the form ``"<lhs> = <rhs>"`` are relabelled component-wise
    when no whole-string mapping exists.
    """
    name_map = LABEL_MAP if name_map is None else name_map
    for ax in g.axes.flat:
        ax.set_xlabel(name_map.get(ax.get_xlabel(), ax.get_xlabel()))
        ax.set_ylabel(name_map.get(ax.get_ylabel(), ax.get_ylabel()))

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_text(name_map.get(text.get_text(), text.get_text()))

        labels = [label.get_text() for label in ax.get_xticklabels()]
        labels = [name_map.get(lab, lab) for lab in labels]
        ax.set_xticklabels(labels, rotation=25, ha="right", rotation_mode="anchor")

        old = ax.get_title()
        new = name_map.get(old, old)
        if new == old and "=" in old:
            lhs = old.split("=")[0].strip()
            rhs = old.split("=")[1].strip()
            new = f"{name_map.get(lhs, lhs)} = {name_map.get(rhs, rhs)}"
        ax.set_title(new)


def relabel_legend(g: sns.FacetGrid, name_map: Optional[Dict[str, str]] = None) -> None:
    """Relabel a FacetGrid-level legend (``g.add_legend``) via ``name_map``."""
    name_map = LABEL_MAP if name_map is None else name_map
    if g.legend is not None:
        for text in g.legend.get_texts():
            text.set_text(name_map.get(text.get_text(), text.get_text()))
