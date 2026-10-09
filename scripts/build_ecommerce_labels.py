"""Derive per-tuple ecommerce ground truth from the SemBench product catalog.

Unlike the artwork/email/rotowire label files, nobody annotated these predicates by
looking at the images. The labels come from the Myntra catalog attributes SemBench ships
(``SemBench/ecomm/1/fashion-dataset/styles.csv``), which a human wrote when listing the
product. That is still *non-model* ground truth -- which is the property ``--human-labels``
needs, and the reason a model-generated label set would be circular -- but it is an
annotation of the item, not of the question being asked. Provenance is recorded in the
header comment of the emitted file and in ``benchmarks/curated.py``; read an
``ecommerce_curated`` number with that in mind.

Only predicates that map to a catalog attribute *without* changing what is being asked
are emitted. Four do:

    is_tshirt_or_top  articleType in {Tshirts, Tops}
    is_menswear       gender in {Men, Boys}
    carries_objects   subCategory in {Bags, Wallets}
    is_footwear       masterCategory == "Footwear"

``is_tshirt_or_top`` covers the union of ``Tshirts`` (142 of the 1000 products, polos
included) and ``Tops`` (48, women's and girls' knitwear). The catalog separates the two,
but that is a distinction a lister makes and a product photo does not carry, so the
question names the union ("a t-shirt, a polo shirt or a top") rather than Myntra's
taxonomy. ``Shirts`` (76, button-up) and ``Kurtas`` (32) are excluded on both sides.

``is_footwear`` is deliberately the easy one and serves as a control. It is
``masterCategory``, the coarsest level of the taxonomy, and coincides exactly with
``subCategory in {Shoes, Sandal, Flip Flops}`` on these products (220 of 1000), so there
is no boundary for a lister and a vision model to disagree about: it is the one curated
predicate on which the two references are expected to agree.

Three image predicates in the operator pool have no catalog counterpart at all (contains
a human, more than one item, only the model's legs -- the catalog describes items, not
image composition), and the ten ``description`` predicates ask what the *text mentions*,
which the catalog cannot answer: it records that an item IS discounted, not that its
description says so. Labelling those from attributes would mark correct answers wrong.

``baseColour`` is deliberately not emitted either. ``postprocess_string`` folds case and
whitespace only, so the catalog's 40-term palette ("Navy Blue", "Grey Melange", "Off
White", "Burgundy") can never match what a vision model would answer for a product photo
("blue", "grey", "white", "red"). That extract would measure vocabulary alignment with
Myntra, not colour perception.

One file serves both product tables: ``ecommerce_products.csv`` is exactly the first 65
rows of ``ecommerce_products_large.csv``, so ``_index_products`` means the same product in
both. The script asserts that rather than trusting it.

    python scripts/build_ecommerce_labels.py [--check]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
STYLES = REPO_ROOT / "SemBench/ecomm/1/fashion-dataset/styles.csv"
FILES = REPO_ROOT / "reasondb/evaluation/benchmarks/files"
OUT = REPO_ROOT / "reasondb/evaluation/ground_truth/ecommerce/ecommerce_products.csv"

#: predicate column -> (catalog attribute, accepted values). Kept as data so the test
#: suite can assert the emitted file still matches the catalog it was derived from.
DERIVATIONS = {
    "is_tshirt_or_top": ("articleType", {"Tshirts", "Tops"}),
    "is_menswear": ("gender", {"Men", "Boys"}),
    "carries_objects": ("subCategory", {"Bags", "Wallets"}),
    "is_footwear": ("masterCategory", {"Footwear"}),
}


def derive(products: pd.DataFrame, styles: pd.DataFrame) -> pd.DataFrame:
    """Return ``_index_products`` plus one 0/1 column per derivable predicate."""
    joined = products.join(styles.set_index("id"), on="id", rsuffix="_catalog")

    missing = int(joined["articleType"].isna().sum())
    assert not missing, (
        f"{missing} product(s) have no row in styles.csv, so they cannot be labelled. "
        "Every row of the table must be covered or lookup_labels raises on the first gap."
    )

    out = pd.DataFrame({"_index_products": range(len(products))})
    for column, (attribute, accepted) in DERIVATIONS.items():
        unknown = accepted - set(joined[attribute].unique())
        assert not unknown, (
            f"{column}: {sorted(unknown)} do not occur in styles.csv column "
            f"{attribute!r}, so the mapping is stale. Present: "
            f"{sorted(joined[attribute].unique())[:12]}..."
        )
        out[column] = joined[attribute].isin(accepted).astype(int).values
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without writing.")
    args = parser.parse_args()

    if not STYLES.exists():
        print(
            f"{STYLES} not found. Download the ecommerce dataset (see README.md) into "
            "<project-root>/SemBench/ecomm/1/fashion-dataset/.",
            file=sys.stderr,
        )
        return 1

    styles = pd.read_csv(STYLES, on_bad_lines="skip")
    small = pd.read_csv(FILES / "ecommerce_products.csv")
    large = pd.read_csv(FILES / "ecommerce_products_large.csv")

    # The load-bearing assumption: one label file, keyed by row ordinal, for two tables.
    # If the small table is ever resampled independently, every one of its labels silently
    # describes a different product -- so this is checked, not assumed.
    assert small["id"].tolist() == large["id"].tolist()[: len(small)], (
        "ecommerce_products.csv is no longer a prefix of ecommerce_products_large.csv, so "
        "one label file cannot key both. Emit one per table, or restore the prefix."
    )

    out = derive(large, styles)

    if args.check:
        print(f"OK: {len(out)} rows, columns {list(out.columns)}")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} rows to {OUT.relative_to(REPO_ROOT)}")
    for column, (attribute, accepted) in DERIVATIONS.items():
        positives = int(out[column].sum())
        print(
            f"  {column:16s} {positives:4d}/{len(out)} positive ({positives / len(out):.0%})"
            f"  <- {attribute} in {sorted(accepted)}"
        )
    print(f"  (first {len(small)} rows also key ecommerce_products.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
