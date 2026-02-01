from abc import ABC
from typing import TYPE_CHECKING

from reasondb.optimizer.guarantees import Guarantee

if TYPE_CHECKING:
    from reasondb.interface.df import DataFrameInterface


# NO need anymore as execute implemented in df.py file
class QueryInterface(ABC):
    def execute(self, name, *guarantees: Guarantee) -> "DataFrameInterface":
        raise NotImplementedError("Execute method not implemented")
