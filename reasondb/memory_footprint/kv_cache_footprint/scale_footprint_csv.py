"""Project the measured 1k-item KV cache footprint tables to larger corpora.

Each data column header carries its own item count, e.g. "Rotowire (728)". The KV
cache footprint is linear in the number of items, so every cell is scaled by
target_items / source_items and the header is rewritten to the target count.

Run from anywhere:  python scale_footprint_csv.py
"""

import csv
import re
from math import floor, log10
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
TARGETS = {"10k": 10_000, "100k": 100_000, "1M": 1_000_000}
HEADER_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<items>[\d\s,]+)\)$")


def sig(value: float, digits: int = 3) -> str:
    """Round to `digits` significant figures, printed in plain decimal notation."""
    if value == 0:
        return "0"
    rounded = round(value, -int(floor(log10(abs(value)))) + (digits - 1))
    if rounded == int(rounded) and abs(rounded) >= 1:
        return str(int(rounded))
    return f"{rounded:g}"


def project(src: Path, target_items: int, dst: Path) -> None:
    rows = list(csv.reader(src.open()))
    header, body = rows[0], rows[1:]

    # Columns without an "(n items)" header are labels (Model, Ratio, ...).
    factors: dict[int, float] = {}
    out_header = list(header)
    for i, col in enumerate(header):
        m = HEADER_RE.match(col)
        if not m:
            continue
        items = int(m.group("items").replace(",", "").replace(" ", ""))
        factors[i] = target_items / items
        out_header[i] = f"{m.group('name')} ({target_items})"

    out_rows = [
        [sig(float(cell) * factors[i]) if i in factors else cell for i, cell in enumerate(row)]
        for row in body
    ]

    with dst.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(out_header)
        writer.writerows(out_rows)
    print(f"wrote {dst}")


def main() -> None:
    for kind in ("text", "image"):
        src = SRC_DIR / f"kv_cache_footprint_{kind}_1k.csv"
        for suffix, items in TARGETS.items():
            project(src, items, SRC_DIR / f"kv_cache_footprint_{kind}_{suffix}.csv")


if __name__ == "__main__":
    main()
