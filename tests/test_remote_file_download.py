"""Fetching an image benchmark's remote files pauses for downloads, not for cache hits.

``download_remote_files`` rate-limits itself against the remote host. The table is
re-prepared for every query (``reset_db_before_each_query`` clears ``prepared``), so a
pause per *row* is paid again on each one even when every file is already on disk: 0.1s x
1000 images is 100 seconds a query, ~17 minutes over a benchmark's ten single-filter
queries, with no work done. Pausing per *download* keeps the rate limit where it belongs.
"""

import time
import types
from pathlib import Path

import pytest

from reasondb.database.external_table import ExternalTable


class _NullLogger:
    def debug(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


def _table(tmp_path, urls, monkeypatch):
    """An ExternalTable stub with just the state download_remote_files reads."""
    table = ExternalTable.__new__(ExternalTable)
    table.remote_files_dir = tmp_path / "remote"
    table.remote_files_dir.mkdir(parents=True, exist_ok=True)
    table.path = tmp_path / "rows.csv"
    table.image_columns = [
        types.SimpleNamespace(
            url=True,
            orig_identifier=types.SimpleNamespace(column_name="image_url"),
        )
    ]
    table.audio_columns = []
    table._connection = types.SimpleNamespace(
        execute=lambda sql: types.SimpleNamespace(fetchall=lambda: [(u,) for u in urls])
    )
    return table


def test_a_fully_cached_table_does_not_sleep(tmp_path, monkeypatch):
    urls = [f"http://host/img{i}.png" for i in range(50)]
    for url in urls:
        (tmp_path / "remote" / url.split("/")[-1]).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "remote" / url.split("/")[-1]).write_bytes(b"x")
    table = _table(tmp_path, urls, monkeypatch)

    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    import asyncio

    asyncio.run(table.download_remote_files(_NullLogger()))

    assert slept == [], "a cache hit must not pause"


def test_each_real_download_still_pauses(tmp_path, monkeypatch):
    """The rate limit is why the pause exists; it must survive for actual fetches."""
    urls = [f"http://host/img{i}.png" for i in range(3)]
    table = _table(tmp_path, urls, monkeypatch)

    fetched = []

    def fake_download(url, logger):
        fetched.append(url)
        return True  # pretend the network was used

    monkeypatch.setattr(table, "download_file", fake_download)
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    import asyncio

    asyncio.run(table.download_remote_files(_NullLogger()))

    assert fetched == urls
    assert slept == [0.1, 0.1, 0.1]


def test_a_partly_cached_table_pauses_only_for_the_misses(tmp_path, monkeypatch):
    urls = [f"http://host/img{i}.png" for i in range(4)]
    table = _table(tmp_path, urls, monkeypatch)

    monkeypatch.setattr(
        table, "download_file", lambda url, logger: url.endswith(("1.png", "3.png"))
    )
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    import asyncio

    asyncio.run(table.download_remote_files(_NullLogger()))

    assert slept == [0.1, 0.1]


def test_download_file_reports_whether_it_used_the_network(tmp_path):
    """The pause decision rests on this return value."""
    table = _table(tmp_path, [], None)
    cached = table.remote_files_dir / "already.png"
    cached.write_bytes(b"x")

    assert table.download_file("http://host/already.png", _NullLogger()) is False
