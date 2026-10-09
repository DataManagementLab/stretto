"""Derive per-tuple movie-review ground truth from the SemBench Rotten Tomatoes dump.

Like ``build_ecommerce_labels.py`` and unlike the artwork/email/rotowire files, nobody
annotated these predicates by reading the reviews. The labels come from ``scoreSentiment``
in ``SemBench/movie.zip``'s ``rotten_tomatoes_movie_reviews.csv`` -- Rotten Tomatoes'
record of whether the critic's own score was fresh or rotten, which the critic assigned
when they published the review. That is *non-model* ground truth, which is the property
``--human-labels`` needs and the reason a model-generated set would be circular, but it is
an annotation of the **score** rather than of the question. See the caveat below.

Three columns are emitted, all from that one signal, because it is the only one in the
dump that covers every row:

    is_positive   scoreSentiment == "POSITIVE"   (6576/10000)
    is_negative   scoreSentiment == "NEGATIVE"   (3424/10000)
    sentiment     "positive" / "negative"        (the extract's answer)

``reviewState`` is not a fourth signal: fresh/rotten agrees with POSITIVE/NEGATIVE on all
10000 rows, so it is the same bit under another name.

**Why nothing from the movie catalog is used.** ``rotten_tomatoes_movies.csv`` is the
obvious place to look for a second, independent predicate, and every candidate in it fails
one of two tests -- the same two ``build_ecommerce_labels.py`` applies.

*It must not change the question.* The operator pool asks what the review text *mentions*
("Extract the movie [title] mentioned in ...", "Extract the [director]'s name ... or
'none' if not mentioned"), and the catalog records what the movie *is*. Measured over
these 10000 reviews, the text contains the full title in 24% of them and the director's
surname in 16%. Labelling those from the catalog would mark the other 76% and 84% wrong
for correctly answering "not mentioned" -- and the same objection retires genre, MPAA
rating, runtime and language, none of which a single review is obliged to state.

*It must cover every tuple.* ``lookup_labels`` raises ``MissingLabelsError`` on the first
row it cannot find, so a column with gaps cannot label a step at all. Nothing else in the
dump is complete: ``genre`` is missing on 1.3% of these rows, ``originalLanguage`` on
1.7%, ``rating`` on 33%, and ``originalScore`` parses to a number on only 68% (the rest
are blank, letter grades or free text). That is also what rules out deriving "is a rave
review" / "is a scathing review" from the score: a review with no score recorded is not
thereby known to be neither.

**The caveat to read a number with.** ``scoreSentiment`` is the sign of the critic's
score, not a judgement of the prose, so it splits at fresh/rotten and nowhere else: a
lukewarm 3/5 is ``is_positive=1``. The queries in ``benchmarks/curated.py`` are worded as
"expresses a favorable overall verdict" rather than the operator pool's "is clearly
positive" for exactly that reason -- "clearly" asks about intensity, which this label does
not record, and an operator answering it correctly on a mild review would be scored wrong.
The cost of the rewording is that these expressions are not the pool's, so a
``movie_random_huge`` precompute store does not cover them; record this benchmark's own.

    python scripts/build_movie_labels.py [--check]
"""

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DUMP = REPO_ROOT / "SemBench/movie.zip"
#: The member of that zip holding the reviews, read straight out of it -- it is 410MB
#: extracted and nothing else here needs it on disk.
DUMP_MEMBER = "rotten_tomatoes_movie_reviews.csv"
REVIEWS = REPO_ROOT / "reasondb/evaluation/benchmarks/files/reviews_10000.csv"
OUT = REPO_ROOT / "reasondb/evaluation/ground_truth/movie/movie_reviews_huge.csv"

#: The only column of the dump these labels rest on, and the two values it takes.
SENTIMENT_COLUMN = "scoreSentiment"
POSITIVE, NEGATIVE = "POSITIVE", "NEGATIVE"


def read_dump_reviews(path: Path = DUMP) -> pd.DataFrame:
    """``reviewId``-keyed sentiment from the zip, without extracting it."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(DUMP_MEMBER) as handle:
            frame = pd.read_csv(
                handle,
                usecols=["reviewId", SENTIMENT_COLUMN, "reviewState"],
                on_bad_lines="skip",
            )
    return frame.drop_duplicates("reviewId").set_index("reviewId")


def derive(reviews: pd.DataFrame, dump: pd.DataFrame) -> pd.DataFrame:
    """Return ``_index_reviews`` plus the three label columns, in table order.

    *reviews* is ``reviews_10000.csv``; *dump* is :func:`read_dump_reviews`'s frame. The
    labels are taken from *dump* rather than from the sentiment column ``reviews_10000.csv``
    happens to carry, so this is a derivation from SemBench and not a copy of a column
    somebody could have edited -- and the two are asserted equal below, which is what makes
    that claim checkable rather than merely stated.
    """
    missing = ~reviews["reviewid"].isin(dump.index)
    assert not missing.any(), (
        f"{int(missing.sum())} of {len(reviews)} reviews have no row in {DUMP_MEMBER}, "
        "so they cannot be labelled. Every row of the table must be covered or "
        "lookup_labels raises on the first gap."
    )

    aligned = dump.loc[reviews["reviewid"]]
    sentiment = aligned[SENTIMENT_COLUMN].values

    unexpected = set(pd.unique(sentiment)) - {POSITIVE, NEGATIVE}
    assert not unexpected, (
        f"{SENTIMENT_COLUMN} takes unexpected value(s) {sorted(unexpected)}; the mapping "
        "to is_positive/is_negative is no longer exhaustive."
    )

    # The local CSV is a sample *of* the dump, so its own columns must agree with what we
    # just read. A mismatch means one of the two files was regenerated independently and
    # the row ordinals these labels are keyed by no longer describe the same reviews.
    assert (reviews["scoresentiment"].values == sentiment).all(), (
        "reviews_10000.csv's scoresentiment disagrees with the SemBench dump; the sample "
        "and the dump are out of sync."
    )
    # fresh/rotten carries no information POSITIVE/NEGATIVE does not, which is why only
    # one of them becomes a label. Asserted rather than assumed: were they to diverge,
    # there would be a second signal here and this file would be understating the data.
    assert (
        (aligned["reviewState"].values == "fresh") == (sentiment == POSITIVE)
    ).all(), (
        "reviewState no longer agrees with scoreSentiment on every row; they are two "
        "signals now, and a second predicate could be derived from the other one."
    )

    is_positive = (sentiment == POSITIVE).astype(int)
    return pd.DataFrame(
        {
            "_index_reviews": range(len(reviews)),
            "is_positive": is_positive,
            "is_negative": 1 - is_positive,
            "sentiment": pd.Series(sentiment).str.lower().values,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without writing.")
    args = parser.parse_args()

    if not DUMP.exists():
        print(
            f"{DUMP} not found. Download the SemBench movie dataset (see README.md) to "
            "<project-root>/SemBench/movie.zip.",
            file=sys.stderr,
        )
        return 1

    reviews = pd.read_csv(REVIEWS)
    out = derive(reviews, read_dump_reviews())

    if args.check:
        print(f"OK: {len(out)} rows, columns {list(out.columns)}")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} rows to {OUT.relative_to(REPO_ROOT)}")
    for column in ("is_positive", "is_negative"):
        positives = int(out[column].sum())
        print(
            f"  {column:12s} {positives:5d}/{len(out)} positive "
            f"({positives / len(out):.0%})  <- {SENTIMENT_COLUMN}"
        )
    print(f"  {'sentiment':12s} {out['sentiment'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
