import asyncio
import shutil
from pathlib import Path
import re
import hashlib
import time
from typing import TYPE_CHECKING, Literal, Sequence, Union

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
        self._hash = hashlib.sha256()
        self.prepared = False
        self.image_columns: Sequence[RemoteColumn] = image_columns
        self.audio_columns: Sequence[RemoteColumn] = audio_columns
        self.text_columns: Sequence[Union[RemoteColumn, InPlaceColumn]] = text_columns

    def reset(self):
        cache_path = self.get_cache_path()
        if cache_path.exists():
            cache_path.unlink()
            wal_path = cache_path.parent / (cache_path.name + ".wal")
            if wal_path.exists():
                wal_path.unlink()
        self.prepared = False

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
                loop = asyncio.get_event_loop()
                for url in urls:
                    self.download_file(url[0], logger)
                    time.sleep(0.1)  # to avoid overwhelming the server
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

    def download_file(self, url: str, logger: FileLogger):
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
                    "User-Agent": "Reasondb/0.1 (multi-modal dataset creation; mailto:matthias.urban@cs.tu-darmstadt.de)"
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

    def get_cache_path(self):
        self._hash = hashlib.sha256()
        with open(self.path, "rb") as f:
            while file_content := f.read(BUFFER_SIZE):
                if not file_content:
                    break
                self._hash.update(file_content)

        cache_filename = str(self.path.absolute())
        cache_filename = re.sub(r"[^a-zA-Z0-9]", "", cache_filename)
        cache_filename = "_".join(
            [self.name, cache_filename[:50], self._hash.hexdigest()]
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
