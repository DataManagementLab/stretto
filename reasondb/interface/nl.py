import asyncio
from typing import TYPE_CHECKING
from reasondb.interface.base_query_interface import QueryInterface
from reasondb.optimizer.guarantees import Guarantee
from reasondb.query_plan.query import Query
from reasondb.interface.df import TableInterface

if TYPE_CHECKING:
    from reasondb.interface.connect import RaccoonDB


class NlQuery(QueryInterface):
    def __init__(self, connection: "RaccoonDB", query: str):
        self.query = query
        self.connection = connection

    def execute(self, name, *guarantees: Guarantee) -> "TableInterface":
        query = Query(self.query)
        logger = self.connection.executor.logger / f"nl_query_{name}"
        result, _ = asyncio.run(
            self.connection.executor.execute_query(query, logger, *guarantees)
        )
        self.connection.executor.database.register_query_result(name, result)
        self.connection.executor.clean_up()
        return TableInterface(self.connection, name)
