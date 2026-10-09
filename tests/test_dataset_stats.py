"""Dataset-shaped columns: tuple counts, modality, and the pinned token statistics.

``reasondb.evaluation.dataset_stats`` exists because a footprint in gigabytes is not a
comparable number until it is divided by the rows behind it, so a result row records
how many rows a benchmark has. What these pin is the part that is load-bearing and easy to
get quietly wrong: the counts are read from the table *files*, so they work on a benchmark
that was loaded without preparing its database, and the token statistics degrade to empty
rather than raising when nothing has been pinned.
"""

import json

import pytest

try:
    from reasondb.evaluation import dataset_stats
except ImportError:  # pragma: no cover - same guard the sibling storage tests use
    pytest.skip("evaluation deps not installed", allow_module_level=True)

from types import SimpleNamespace

from reasondb.database.indentifier import InPlaceColumn, RemoteColumn


def _table(name, path, text=(), image=(), audio=()):
    return SimpleNamespace(
        name=name,
        path=path,
        file_type="csv",
        text_columns=list(text),
        image_columns=list(image),
        audio_columns=list(audio),
    )


def _csv(tmp_path, name, header, rows):
    path = tmp_path / f"{name}.csv"
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def test_row_counts_come_from_the_table_files(tmp_path):
    """No prepared database: ``Database.prepare`` is async and downloads remote media, and
    neither an enumeration nor a result row may trigger it."""
    reviews = _csv(tmp_path, "reviews", "reviewtext", ["a", "b", "c"])
    database = SimpleNamespace(
        external_tables=[_table("reviews", reviews, text=[InPlaceColumn("reviews.reviewtext")])]
    )

    assert dataset_stats.table_row_counts(database) == {"reviews": 3}
    assert dataset_stats.total_rows(database) == 3


def test_several_tables_are_summed_and_also_reported_apart(tmp_path):
    """The sum is what a single-table benchmark's operators run over. For a join it is
    neither input nor output cardinality, which is why the per-table counts travel too."""
    column = [InPlaceColumn("left.text")]
    left = _csv(tmp_path, "left", "text", ["a", "b"])
    right = _csv(tmp_path, "right", "text", ["c"])
    database = SimpleNamespace(
        external_tables=[
            _table("left", left, text=column),
            _table("right", right, text=[InPlaceColumn("right.text")]),
        ]
    )

    assert dataset_stats.table_row_counts(database) == {"left": 2, "right": 1}
    assert dataset_stats.total_rows(database) == 3


def test_only_the_cached_tables_count_toward_the_tuple_total(tmp_path):
    """Rotowire in miniature: five tables, one of them carrying the text a KV cache is
    built over. Dividing a footprint by all of their rows reports an operator far cheaper
    per item than it is, which is the whole reason this number is recorded."""
    reports = _csv(tmp_path, "reports", "report", ["a", "b"])
    players = _csv(tmp_path, "players", "name", ["p", "q", "r", "s"])
    database = SimpleNamespace(
        external_tables=[
            _table("reports", reports, text=[InPlaceColumn("reports.report")]),
            _table("players", players),
        ]
    )

    assert dataset_stats.cached_table_names(database) == ["reports"]
    assert dataset_stats.total_rows(database) == 2
    # And every table is still reported, for anyone who wants the other reading.
    assert dataset_stats.table_row_counts(database) == {"reports": 2, "players": 4}


class _Tokenizer:
    """Whitespace tokens, which is all these tests need of a vocabulary."""

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [text.split() for text in texts]}


def test_an_in_place_text_column_is_tokenized_from_the_cell(tmp_path):
    path = _csv(tmp_path, "t", "body", ["one two three", "four"])
    database = SimpleNamespace(
        external_tables=[_table("t", path, text=[InPlaceColumn("t.body")])]
    )

    lengths = dataset_stats.column_token_lengths(database, _Tokenizer())

    assert lengths == {"t.body": [3, 1]}


def test_a_remote_text_column_is_tokenized_from_the_file_it_names(tmp_path):
    """The cell holds a path and ``TableMetadata.setup`` replaces it with that file's
    contents before anything caches it - so tokenizing the cell would measure path
    lengths. Enron is the benchmark this is true of, and its items are ~1000 tokens where
    their paths are ~30."""
    body = tmp_path / "mail.txt"
    body.write_text("a much longer message than its own file name")
    path = _csv(tmp_path, "emails", "text_path", [str(body)])
    database = SimpleNamespace(
        external_tables=[
            _table("emails", path, text=[RemoteColumn("emails.text_path", "emails.text")])
        ]
    )

    lengths = dataset_stats.column_token_lengths(database, _Tokenizer())

    assert lengths == {"emails.text_path": [9]}


def test_an_unreadable_table_is_reported_rather_than_raised_on(tmp_path, caplog):
    """A descriptive column must never stop a benchmark that otherwise runs."""
    import logging

    database = SimpleNamespace(external_tables=[_table("gone", tmp_path / "missing.csv")])
    with caplog.at_level(logging.WARNING):
        assert dataset_stats.table_row_counts(database) == {}
    assert dataset_stats.total_rows(database) is None
    assert any("could not count" in r.message.lower() for r in caplog.records)


def test_modality_reads_the_same_declarations_the_scheduler_does(tmp_path):
    """So a benchmark cannot be described here as one thing and scheduled as another."""
    text = _table("t", tmp_path / "t.csv", text=[InPlaceColumn("t.body")])
    image = _table("i", tmp_path / "i.csv", image=[RemoteColumn("i.url", "i.path")])

    assert dataset_stats.modality(SimpleNamespace(external_tables=[text])) == "text"
    assert dataset_stats.modality(SimpleNamespace(external_tables=[image])) == "image"
    assert (
        dataset_stats.modality(SimpleNamespace(external_tables=[text, image]))
        == "multimodal"
    )
    assert dataset_stats.modality(SimpleNamespace(external_tables=[])) == "none"


def test_token_summary_reports_the_tail_as_well_as_the_mean():
    """The mean sets the footprint and the tail sets the batch a server can hold, so a
    column with a heavy tail is expensive in a way its mean does not show."""
    summary = dataset_stats.token_summary([1, 2, 3, 100])

    assert summary["n_items"] == 4
    assert summary["median"] == 2.5
    assert summary["max"] == 100
    assert summary["p95"] == 100


def test_token_summary_of_nothing_is_a_count_and_no_statistics():
    assert dataset_stats.token_summary([]) == {"n_items": 0}


def test_tokens_per_item_weights_the_columns_by_their_items():
    """A benchmark whose two columns differ in length must not be described by whichever
    was declared first."""
    stats = {
        "bench": {
            "text_columns": {
                "t.short": {"n_items": 1, "mean": 10.0, "p95": 12.0},
                "t.long": {"n_items": 3, "mean": 100.0, "p95": 300.0},
            }
        }
    }

    headline = dataset_stats.tokens_per_item(stats, "bench")

    assert headline["tokens_per_item"] == pytest.approx((10.0 + 3 * 100.0) / 4)
    assert headline["tokens_per_item_p95"] == 300.0


def test_image_columns_contribute_no_tokens():
    """What a vision model caches per image is a function of the model and the resolution
    it tiles to, so counting it here would put a model constant in a dataset column."""
    stats = {"bench": {"text_columns": {}, "image_columns": {"t.img": {"n_items": 50}}}}

    assert dataset_stats.tokens_per_item(stats, "bench") == {}


def test_missing_token_statistics_are_empty_rather_than_fatal(tmp_path):
    """Every figure they feed draws without them, one dimension poorer."""
    assert dataset_stats.load_token_stats(tmp_path / "nothing.json") == {}


def test_pinned_token_statistics_round_trip(tmp_path):
    path = tmp_path / "pinned.json"
    path.write_text(
        json.dumps(
            {
                "generated": "2026-09-17",
                "benchmarks": {"bench": {"num_tuples": 7, "modality": "text"}},
            }
        )
    )

    stats = dataset_stats.load_token_stats(path)

    assert stats["bench"]["num_tuples"] == 7
