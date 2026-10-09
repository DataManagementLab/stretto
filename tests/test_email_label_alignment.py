"""The enron labels join on a row ordinal, so whatever fixes the order is the key.

`LabelsDefinition` keys ground truth by `_index_<table>`, the DuckDB rowid of the loaded
table -- i.e. an email's *position* in it. Nothing about that key is self-describing: a
label lands on whichever email sits at that ordinal, so an order that shifts by one row
scores hundreds of emails against their neighbour's annotation and says nothing.

The table is therefore built from `EMAIL_LABELS` itself, in its `_index_emails` order,
rather than from a directory listing, which would agree with it only by convention.

Alignment holds by construction, so what is left to test is that the construction holds
(the loader follows the manifest, and refuses a corpus that has drifted from it) and that
the manifest is right about the corpus at all -- which only reading the emails can show.
"""

from pathlib import Path

import pandas as pd
import pytest

from reasondb.evaluation.benchmarks.email import EnronEmail

EMAILS_DIR = Path("palimpzest/testdata/enron-eval")
LABELS = Path("reasondb/evaluation/ground_truth/emails/enron-eval.csv")


def _fake_corpus(tmp_path, names, manifest_names=None):
    """A directory of emails plus a manifest, in deliberately different orders."""
    for name in names:
        (tmp_path / name).write_text("body")
    manifest = pd.DataFrame(
        {
            "_index_emails": range(len(manifest_names or names)),
            "filename": list(manifest_names or names),
        }
    )
    path = tmp_path / "labels.csv"
    manifest.to_csv(path, index=False)
    return path


def test_the_table_follows_the_manifest_rather_than_the_directory(tmp_path, monkeypatch):
    """The manifest order is not the sorted order here, so only one of the two can pass.

    Also checks that `emails.csv`, which `load_email_table` writes into the directory it
    reads, is not picked up as an extra row on a second call."""
    manifest = _fake_corpus(tmp_path, ["a.txt", "b.txt", "c.txt"], ["c.txt", "a.txt", "b.txt"])
    monkeypatch.setattr("reasondb.evaluation.benchmarks.email.EMAIL_LABELS", manifest)

    first = pd.read_csv(EnronEmail.load_email_table(tmp_path))
    second = pd.read_csv(EnronEmail.load_email_table(tmp_path))

    assert [Path(p).name for p in first["text_path"]] == ["c.txt", "a.txt", "b.txt"]
    assert second.equals(first), (
        "The table must not depend on how often it has been generated: a row ordinal is "
        "the ground truth's join key."
    )


@pytest.mark.parametrize(
    "on_disk, annotated",
    [
        (["a.txt", "b.txt"], ["a.txt", "b.txt", "c.txt"]),  # an email went missing
        (["a.txt", "b.txt", "c.txt"], ["a.txt", "b.txt"]),  # an unannotated email
    ],
)
def test_a_corpus_that_drifted_from_the_manifest_raises(
    tmp_path, monkeypatch, on_disk, annotated
):
    """Not "use what both have": a corpus one email short of its annotations has
    invalidated the ordinal key, and every way of continuing scores emails against
    somebody else's label."""
    manifest = _fake_corpus(tmp_path, on_disk, annotated)
    monkeypatch.setattr("reasondb.evaluation.benchmarks.email.EMAIL_LABELS", manifest)

    with pytest.raises(ValueError, match="does not hold exactly"):
        EnronEmail.load_email_table(tmp_path)


def test_a_manifest_with_a_gap_in_its_ordinals_raises(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("body")
    (tmp_path / "b.txt").write_text("body")
    path = tmp_path / "labels.csv"
    pd.DataFrame({"_index_emails": [0, 2], "filename": ["a.txt", "b.txt"]}).to_csv(
        path, index=False
    )
    monkeypatch.setattr("reasondb.evaluation.benchmarks.email.EMAIL_LABELS", path)

    with pytest.raises(ValueError, match="contiguously"):
        EnronEmail.load_email_table(tmp_path)


@pytest.mark.skipif(
    not EMAILS_DIR.is_dir(), reason="palimpzest testdata not checked out"
)
def test_the_manifest_covers_the_real_corpus_exactly():
    labels = pd.read_csv(LABELS)
    assert set(labels["filename"]) == {p.name for p in EMAILS_DIR.glob("*.txt")}
    assert labels["filename"].is_unique


@pytest.mark.skipif(
    not EMAILS_DIR.is_dir(), reason="palimpzest testdata not checked out"
)
def test_the_annotated_sender_occurs_in_its_own_email():
    """The checks above are structural; this one reads the emails.

    A permuted key passes nothing here (matches at chance level), so this is
    what distinguishes "the manifest is self-consistent" from "it is right about the
    corpus".
    """
    labels = pd.read_csv(LABELS)

    found = sum(
        str(sender).lower()
        in (EMAILS_DIR / name).read_text(errors="ignore").lower()
        for name, sender in zip(labels["filename"], labels["sender"])
    )
    assert found == len(labels), f"{len(labels) - found} senders are not in their email"
