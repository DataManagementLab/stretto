import asyncio
import platform
from pathlib import Path
import re
import hashlib
import shutil
import time
import uuid
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import duckdb
import requests
from tqdm import tqdm

from reasondb.database.indentifier import InPlaceColumn, RemoteColumn
from reasondb.database.metadata import TableMetadata
from reasondb.reasoning.embeddings import TextEmbeddingModel
from reasondb.reasoning.llm import LargeLanguageModel
from reasondb.utils.logging import FileLogger

if TYPE_CHECKING:
    from reasondb.database.database import Database

BUFFER_SIZE = 1024 * 1024  # 1 MB


class ExternalTable:
    def __init__(
        self,
        name: str,
        path: Path,
        file_type: Literal["csv", "parquet", "json"],
        image_columns: Sequence[RemoteColumn],
        audio_columns: Sequence[RemoteColumn],
        text_columns: Sequence[Union[RemoteColumn, InPlaceColumn]],
        database: "Database",
    ):
        self.name = name
        self.path = path
        self.file_type = file_type
        self.cache_dir = database.cache_dir
        self.database = database
        self._data_hash = hashlib.sha256()
        self._hostname = platform.node()
        self.prepared = False
        self.image_columns: Sequence[RemoteColumn] = image_columns
        self.audio_columns: Sequence[RemoteColumn] = audio_columns
        self.text_columns: Sequence[Union[RemoteColumn, InPlaceColumn]] = text_columns
        self.backup_path: Optional[Path] = None

    def reset(self):
        cache_path = self.get_cache_path()
        if self.backup_path is not None:
            # Restore the query-independent prepared state (e.g. computed
            # embeddings) instead of wiping and recomputing it from scratch.
            shutil.copy2(self.backup_path, cache_path)
            self._copy_wal(self.backup_path, cache_path)
        else:
            self._delete(cache_path)
        self.prepared = False

    def snapshot(self):
        """Snapshot the cache file once it holds the generic, query-independent
        prepared state (e.g. computed embeddings), so later resets can restore
        it instead of recomputing that state from scratch.

        Idempotent: only the first call (per run) after `prepared` becomes True
        actually takes the snapshot; later calls no-op.
        """
        if self.backup_path is not None or not self.prepared:
            return
        cache_path = self.get_cache_path()
        if not cache_path.exists():
            return
        backup_path = cache_path.parent / (
            f".backup_{cache_path.stem}_{uuid.uuid4().hex}{cache_path.suffix}"
        )
        shutil.copy2(cache_path, backup_path)
        self._copy_wal(cache_path, backup_path)
        self.backup_path = backup_path

    def delete_backup(self):
        """Remove the per-run backup snapshot, if one was taken."""
        if self.backup_path is not None:
            self._delete(self.backup_path)
            self.backup_path = None

    @staticmethod
    def _wal_path(db_path: Path) -> Path:
        return db_path.parent / (db_path.name + ".wal")

    def _delete(self, db_path: Path):
        if db_path.exists():
            db_path.unlink()
        wal_path = self._wal_path(db_path)
        if wal_path.exists():
            wal_path.unlink()

    def _copy_wal(self, src: Path, dst: Path):
        src_wal = self._wal_path(src)
        dst_wal = self._wal_path(dst)
        if src_wal.exists():
            shutil.copy2(src_wal, dst_wal)
        elif dst_wal.exists():
            dst_wal.unlink()

    async def prepare(
        self,
        embedding_model: TextEmbeddingModel,
        llm: LargeLanguageModel,
        logger: FileLogger,
    ) -> Path:
        if not self.prepared:
            already_exists = self.setup_cache_path()
            self.table_metadata = TableMetadata(self, embedding_model, llm)

            await self.download_remote_files(logger)
            if not already_exists:
                self.setup_metadata()
            self._connection.close()
            self.prepared = True

        return self.cache_path

    async def download_remote_files(self, logger: FileLogger):
        for remote_col in self.image_columns:
            if remote_col.url:
                urls = self._connection.execute(
                    f"SELECT {remote_col.orig_identifier.column_name} FROM '{self.path}';"
                ).fetchall()
                for url in urls:
                    # Pause only when the network was actually used; the table is
                    # re-prepared per query, so cache hits must not incur the delay.
                    if self.download_file(url[0], logger):
                        time.sleep(0.1)  # to avoid overwhelming the remote host
        for remote_col in self.audio_columns:
            if remote_col.url:
                urls = self._connection.execute(
                    f"SELECT {remote_col.orig_identifier.column_name} FROM '{self.path}';"
                ).fetchall()
                loop = asyncio.get_event_loop()
                coroutines = [
                    loop.run_in_executor(None, self.download_file, *(url[0], logger))
                    for url in urls
                ]
                await asyncio.gather(
                    *coroutines,
                    return_exceptions=True,
                )

    def download_file(self, url: str, logger: FileLogger) -> bool:
        """Fetch *url* into the remote-file cache unless it is already there.

        :returns: whether the network was actually used, so a caller rate-limiting
            itself against the remote host can skip the wait for a cache hit.
        """
        # check if file already exists and has the correct hash
        file_name = url.split("/")[-1]
        file_path = Path(self.remote_files_dir / file_name)
        try:
            exists = file_path.exists()
        except OSError:
            file_ending = file_name.split(".")[-1]
            file_name_shortened = ".".join(file_name.split(".")[:-1])[:50]
            file_name = f"{file_name_shortened}.{file_ending}"
            file_path = Path(self.remote_files_dir / file_name)
            exists = file_path.exists()

        if not exists:
            logger.debug(__name__, f"Downloading {url} to {file_path}")
            r = requests.get(
                url,
                stream=True,
                headers={
                    "User-Agent": "Stretto/0.1 (multi-modal dataset creation; mailto:matthias.urban@cs.tu-darmstadt.de)"
                },
            )
            if r.status_code != 200:
                logger.error(
                    __name__,
                    f"Failed to download {url}. Status code: {r.status_code}",
                )
            with open(file_path, "wb") as f:
                try:
                    total_length = int(r.headers.get("content-length"))  # type: ignore
                except TypeError:
                    total_length = 0
                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=total_length / 1024,
                    unit="KB",
                    desc=f"Downloading {url}",
                ):
                    if chunk:
                        f.write(chunk)
            logger.debug(__name__, f"Downloaded {url} to {file_name}")
            return True
        return False

    def get_cache_path(self):
        self._data_hash = hashlib.sha256()
        with open(self.path, "rb") as f:
            while file_content := f.read(BUFFER_SIZE):
                if not file_content:
                    break
                self._data_hash.update(file_content)

        cache_filename = str(self.path.absolute())
        cache_filename = re.sub(r"[^a-zA-Z0-9]", "", cache_filename)
        cache_filename = "_".join(
            [
                self.name,
                cache_filename[:50],
                self._data_hash.hexdigest(),
                self._hostname,
            ]
        )
        cache_path = self.cache_dir / f"{cache_filename}.db"
        return cache_path

    def setup_cache_path(self):
        self.cache_path = self.get_cache_path()
        self.remote_files_dir = self.cache_dir / f"{self.name}_files"
        already_exists = self.cache_path.exists()
        self.remote_files_dir.mkdir(exist_ok=True, parents=True)
        self._connection = duckdb.connect(self.cache_path)
        self._connection.execute("INSTALL vss;")
        self._connection.execute("LOAD vss;")
        self._connection.execute("SET hnsw_enable_experimental_persistence=true;")
        return already_exists

    def setup_metadata(self):
        self.table_metadata.setup(self._connection)
