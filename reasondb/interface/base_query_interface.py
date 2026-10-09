from abc import ABC
from typing import TYPE_CHECKING

from reasondb.optimizer.guarantees import Guarantee

if TYPE_CHECKING:
    from reasondb.interface.df import DataFrameInterface


# Abstract base for query interfaces; DataFrame queries are executed via df.py.
class QueryInterface(ABC):
    def execute(self, name, *guarantees: Guarantee) -> "DataFrameInterface":
        raise NotImplementedError("Execute method not implemented")
