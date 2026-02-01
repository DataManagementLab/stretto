import asyncio
from pathlib import Path
from typing import Literal, Optional, Sequence, Union
from reasondb.database.indentifier import InPlaceColumn, RemoteColumn
from reasondb.executor import Executor
from reasondb.interface.config import Config
from reasondb.interface.df import TableInterface
from reasondb.interface.nl import NlQuery


class RaccoonDB:
    def __init__(self, db_name, config: Optional[Config] = None):
        self.db_name = db_name
        self.config = config or Config(db_name)
        if config is not None:
            assert db_name == config.identifier_for_caching
        self._executor: Optional[Executor] = None

    @property
    def executor(self) -> Executor:
        assert (
            self._executor is not None
        ), "Executor is not initialized. Use 'with RaccoonDB(...) as rc:'."
        return self._executor

    def __enter__(self):
        self._executor = self.config.construct_executor()
        asyncio.run(self.executor.setup())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor is not None:
            self.executor.shutdown()

    def prepare(self):
        asyncio.run(self.executor.prepare())

    @property
    def database(self):
        assert (
            self.executor is not None
        ), "Executor is not initialized. Use 'with RaccoonDB(...) as rc:'."
        return self.executor.database

    def add_table(
        self,
        path: Path,
        table_name: str,
        image_columns: Sequence[RemoteColumn] = (),
        audio_columns: Sequence[RemoteColumn] = (),
        text_columns: Sequence[Union[InPlaceColumn, RemoteColumn]] = (),
        file_type: Literal["csv", "parquet", "json"] = "csv",
    ) -> TableInterface:
        self.database.add_table(
            path=path,
            name=table_name,
            image_columns=image_columns,
            audio_columns=audio_columns,
            text_columns=text_columns,
            file_type=file_type,
        )
        return TableInterface(self, table_name)

    def nl_query(self, query) -> NlQuery:
        return NlQuery(self, query)
