from typing import TYPE_CHECKING
from reasondb.interface.base_query_interface import QueryInterface
from reasondb.optimizer.guarantees import Guarantee

if TYPE_CHECKING:
    from reasondb.interface.df import TableInterface
    from reasondb.interface.connect import RaccoonDB


class SqlExtendedQuery(QueryInterface):
    def __init__(self, connection: "RaccoonDB", query: str):
        self.query = query
        self.connection = connection

    def execute(self, name, *guarantees: Guarantee) -> "TableInterface":
        raise NotImplementedError
