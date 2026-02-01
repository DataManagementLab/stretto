from abc import ABC, abstractmethod
import datetime
from hashlib import sha256
import hashlib
import duckdb.typing
from collections.abc import Sequence
import enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Type,
    Union,
)
import numpy as np
import duckdb
import re
from dateutil import parser
import pytz

if TYPE_CHECKING:
    from reasondb.query_plan.logical_plan import LogicalPlanStep

# Detect trailing IANA timezone names like "America/Los_Angeles"
TZ_PATTERN = re.compile(r"(.*)\s([A-Za-z_]+/[A-Za-z_]+)$")
DEFAULT_TZ = pytz.UTC


class BaseIdentifier(ABC):
    def __init__(self, name: str):
        self._name = name.lower().replace(" ", "_").replace("-", "_")
        assert re.match(
            r"^([a-z_][a-z0-9_]*|<left_table>|<right_table>|<input_table>)(\.[a-z_][a-z0-9_]*)?$",
            self.name,
        )

    def __str__(self) -> str:
        return self.name

    def to_json(self):
        return self.name

    @property
    def name(self):
        return self._name

    @staticmethod
    def from_json(json_identifier, virtual: bool):
        if virtual:
            if "." in json_identifier:
                return VirtualColumnIdentifier(json_identifier)
            return VirtualTableIdentifier(json_identifier)
        else:
            if "." in json_identifier:
                return ConcreteColumnIdentifier(json_identifier)
            return ConcreteTableIdentifier(json_identifier)

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:
        return repr(self.name)

    def __lt__(self, other):
        return self.name < other.name


class BaseTableIdentifier(BaseIdentifier):
    def __init__(self, name: str):
        assert "." not in name
        BaseIdentifier.__init__(self, name)

    @property
    def table_name(self):
        return self.name


class BaseColumnIdentifier(BaseIdentifier):
    def __init__(self, name: str):
        assert name.count(".") == 1
        BaseIdentifier.__init__(self, name)

    @property
    def alias(self):
        return self.name.split(".")[1]


class BaseColumn(BaseColumnIdentifier):
    @property
    @abstractmethod
    def data_type(self) -> "DataType":
        pass


class DataType(enum.Enum):
    INT = "INT"
    STRING = "STRING"
    TEXT = "TEXT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
    DATE = "DATE"
    TIME = "TIME"
    DATETIME = "DATETIME"
    DATETIME_WITH_TZ = "DATETIME_WITH_TIME_ZONE"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    EMBEDDING = "EMBEDDING"

    def convert(self, value):
        value = str(value).strip()
        if self == DataType.INT:
            match = re.search(r"[\-\+]?\d+", str(value))
            return int(match.group(0)) if match else 0
        elif self == DataType.STRING:
            return str(value)
        elif self == DataType.TEXT:
            return str(value)
        elif self == DataType.FLOAT:
            # can also be in scientific notation
            match = re.search(r"[\-\+]?\d*\.?\d+(?:[eE][\-\+]?\d+)?", str(value))
            return float(match.group(0)) if match else 0.0
        elif self == DataType.BOOL:
            return value in (
                True,
                "True",
                "true",
                1,
                "1",
                "yes",
                "YES",
                "Yes",
                "Y",
                "y",
                "t",
                "T",
                "TRUE",
            )
        elif self == DataType.DATE:
            if isinstance(value, datetime.date):
                return value
            try:
                return datetime.date.fromisoformat(value)
            except ValueError:
                return datetime.date.fromisoformat("1970-01-01")
        elif self == DataType.TIME:
            if isinstance(value, datetime.time):
                return value
            return datetime.time.fromisoformat(value)
        elif self == DataType.DATETIME:
            if isinstance(value, datetime.datetime):
                if value.tzinfo is not None:
                    return value.astimezone(DEFAULT_TZ).replace(tzinfo=None)  # type: ignore
                return value
            return self.parse_datetime_best_guess(value, timezone=True)
        elif self == DataType.DATETIME_WITH_TZ:
            if isinstance(value, datetime.datetime):
                if value.tzinfo is None:
                    return DEFAULT_TZ.localize(value)
                return value
            return self.parse_datetime_best_guess(value, timezone=False)
        elif self == DataType.IMAGE:
            return str(value)
        elif self == DataType.AUDIO:
            return str(value)
        raise ValueError(f"Unknown data type {self}")

    @staticmethod
    def parse_datetime_best_guess(
        value, timezone: bool, default_tz=DEFAULT_TZ
    ) -> datetime.datetime:
        """
        Best-guess parser for date/time strings.
        Supports ISO formats, human-readable formats,
        and IANA timezone names (e.g. America/Los_Angeles).
        """
        value = value.strip()

        # Explicit handling for IANA timezone names
        match = TZ_PATTERN.match(value)
        if match and timezone:
            date_part, tz_part = match.groups()
            tz = pytz.timezone(tz_part)
            dt = parser.parse(date_part, fuzzy=True)
            return tz.localize(dt)

        # Fallback: dateutil handles ISO, offsets, etc.
        try:
            dt = parser.parse(value, fuzzy=True)
        except (ValueError, OverflowError):
            dt = datetime.datetime(1970, 1, 1)

        if timezone and dt.tzinfo is None:
            return default_tz.localize(dt)

        if not timezone and dt.tzinfo is not None:
            return dt.astimezone(default_tz).replace(tzinfo=None)

        return dt

    @staticmethod
    def from_duckdb(
        dtype: str, is_image: bool, is_text: bool, is_audio: bool = False
    ) -> "DataType":
        if is_image:
            return DataType.IMAGE
        if is_audio:
            return DataType.AUDIO
        if is_text:
            return DataType.TEXT
        elif dtype in {
            "BIGINT",
            "HUGEINT",
            "INTEGER",
            "SMALLINT",
            "TINYINT",
            "UBIGINT",
            "UHUGEINT",
            "UINTEGER",
            "USMALLINT",
            "UTINYINT",
        }:
            return DataType.INT
        elif dtype == "VARCHAR":
            return DataType.STRING
        elif dtype in {"DOUBLE", "FLOAT"}:
            return DataType.FLOAT
        elif dtype == "BOOLEAN":
            return DataType.BOOL
        elif dtype in {"DATE"}:
            return DataType.DATE
        elif dtype in {"TIME"}:
            return DataType.TIME
        elif dtype in {"TIMESTAMP", "DATETIME"}:
            return DataType.DATETIME
        elif dtype in {"TIMESTAMP WITH TIME ZONE", "TIMESTAMPTZ", "TIMESTAMP_TZ"}:
            return DataType.DATETIME_WITH_TZ
        elif dtype.startswith("FLOAT["):
            return DataType.EMBEDDING
        raise ValueError(f"Unknown data type {dtype}")

    def to_duckdb_type(self):
        duckdb_str = self.to_duckdb()
        if duckdb_str != "TIMESTAMPTZ":
            return getattr(duckdb.typing, self.to_duckdb())
        else:
            return duckdb.typing.TIMESTAMP_TZ

    def to_duckdb(self):
        if self == DataType.INT:
            return "BIGINT"
        elif self == DataType.STRING:
            return "VARCHAR"
        elif self == DataType.TEXT:
            return "VARCHAR"
        elif self == DataType.FLOAT:
            return "DOUBLE"
        elif self == DataType.BOOL:
            return "BOOLEAN"
        elif self == DataType.DATE:
            return "DATE"
        elif self == DataType.TIME:
            return "TIME"
        elif self == DataType.DATETIME:
            return "TIMESTAMP"
        elif self == DataType.DATETIME_WITH_TZ:
            return "TIMESTAMPTZ"
        elif self == DataType.IMAGE:
            return "VARCHAR"
        elif self == DataType.AUDIO:
            return "VARCHAR"
        raise ValueError(f"Unknown data type {self}")

    @staticmethod
    def from_pandas(
        dtype: Union[np.dtype, Type],
        is_image: bool,
        is_text: bool,
        is_audio: bool,
        example_values: Any,
    ) -> "DataType":
        if is_image:
            return DataType.IMAGE
        if is_audio:
            return DataType.AUDIO
        if is_text:
            return DataType.TEXT
        if dtype in (np.int64, int):
            return DataType.INT
        elif dtype in (np.float64, float):
            return DataType.FLOAT
        elif dtype in (np.bool, bool):
            return DataType.BOOL
        elif dtype in (np.datetime64, datetime.datetime) and all(
            isinstance(v, (datetime.datetime, np.datetime64)) for v in example_values
        ):
            if any(
                isinstance(v, datetime.datetime) and v.tzinfo is not None
                for v in example_values
            ):
                return DataType.DATETIME_WITH_TZ
            else:
                return DataType.DATETIME
            return DataType.DATETIME
        elif dtype in (datetime.date,):
            return DataType.DATE
        elif dtype in (datetime.time,):
            return DataType.TIME
        elif dtype in (np.dtype("O"), str):
            return DataType.STRING
        raise ValueError(f"Unknown data type {dtype}")

    def to_pandas(self):
        if self == DataType.INT:
            return np.int64
        elif self == DataType.STRING:
            return str
        elif self == DataType.TEXT:
            return str
        elif self == DataType.FLOAT:
            return np.float64
        elif self == DataType.BOOL:
            return np.bool_
        elif self == DataType.DATE:
            return datetime.date
        elif self == DataType.TIME:
            return datetime.time
        elif self == DataType.DATETIME:
            return datetime.datetime
        elif self == DataType.DATETIME_WITH_TZ:
            return datetime.datetime
        elif self == DataType.IMAGE:
            return str
        elif self == DataType.AUDIO:
            return str
        raise ValueError(f"Unknown data type {self}")


class VirtualTableIdentifier(BaseTableIdentifier):
    def __init__(self, name: str):
        assert "." not in name
        assert not name.startswith("_")
        BaseTableIdentifier.__init__(self, name)


class VirtualColumnIdentifier(BaseColumnIdentifier):
    def __init__(self, name: str):
        assert not name.startswith("_")
        assert name.count(".") == 1
        BaseColumnIdentifier.__init__(self, name)

    @property
    def column_name(self):
        return self.name.split(".")[1]

    @property
    def table_name(self):
        return self.name.split(".")[0]

    @property
    def table_identifier(self):
        return VirtualTableIdentifier(self.table_name)


class StarIdentifier:
    def __init__(
        self,
    ):
        self._table_identifer: Optional[ConcreteTableIdentifier] = None

    @property
    def column_name(self):
        return "*"

    def __str__(self):
        return "*"

    def __repr__(self):
        return "*"

    @property
    def no_alias(self):
        return "*"

    @property
    def alias(self):
        return "*"


class VirtualColumn(VirtualColumnIdentifier, BaseColumn):
    def __init__(self, name: str, data_type: DataType):
        VirtualColumnIdentifier.__init__(self, name)
        assert isinstance(data_type, DataType)
        self._data_type = data_type

    @property
    def data_type(self):
        return self._data_type


class ConcreteTableIdentifier(BaseTableIdentifier):
    def __init__(self, name: str):
        assert "." not in name
        BaseTableIdentifier.__init__(self, name)

    @property
    def column_name(self):
        return self.name.split(".")[1]

    @property
    def table_name(self):
        return self.name.split(".")[0]

    def check_exists(
        self,
        database: duckdb.DuckDBPyConnection,
        table_renamings: Dict["ConcreteTableIdentifier", "ConcreteTableIdentifier"],
    ):
        tables = database.execute("SHOW TABLES").fetchall()
        assert tables is not None, "No tables found"
        renamed = table_renamings.get(self, self)
        if renamed not in [ConcreteTableIdentifier(table[0]) for table in tables]:
            raise ValueError(f"Table {self.table_name} not found")

    def to_hidden_table_identifier(self):
        return HiddenTableIdentifier(self.name)


class ConcreteColumnIdentifier(BaseColumnIdentifier):
    def __init__(self, name: str):
        assert name.count(".") == 1
        BaseColumnIdentifier.__init__(self, name)

    def rename_table(self, new_table_name: str):
        new_name = f"{new_table_name}.{self.name.split('.')[1]}"
        return ConcreteColumnIdentifier(new_name)

    @property
    def shortened(self) -> str:
        return self.no_alias

    @property
    def table_name(self) -> Optional[str]:
        return self.name.split(".")[0]

    @property
    def table_identifier(self) -> Optional[ConcreteTableIdentifier]:
        if self.table_name is None:
            return None
        else:
            return ConcreteTableIdentifier(self.table_name)

    @property
    def no_alias(self):
        return self.name

    @property
    def alias(self):
        return self.name.split(".")[1]


class RealColumnIdentifier(ConcreteColumnIdentifier):
    @property
    def column_name(self):
        return self.name.split(".")[1]

    @property
    def table_name(self) -> str:
        return self.name.split(".")[0]

    def rename_table(self, new_table_name: str):
        new_name = f"{new_table_name}.{self.name.split('.')[1]}"
        return RealColumnIdentifier(new_name)

    @property
    def table_identifier(self) -> ConcreteTableIdentifier:
        return ConcreteTableIdentifier(self.table_name)


class ConcreteColumn(ConcreteColumnIdentifier, BaseColumn):
    def __init__(self, name: str, data_type: DataType, alias: Optional[str] = None):
        ConcreteColumnIdentifier.__init__(self, name)
        assert isinstance(data_type, DataType)
        self._data_type = data_type
        self._alias = alias

    def rename_table(self, new_table_name: str):
        new_name = f"{new_table_name}.{self.name.split('.')[1]}"
        return ConcreteColumn(new_name, self.data_type, self.alias)

    def __str__(self) -> str:
        if self._alias and self._alias != super().alias:
            return f"{super().no_alias} AS {self._alias}"
        return self.no_alias

    @property
    def no_alias(self) -> str:
        return super().no_alias

    @property
    def alias(self) -> str:
        if self._alias is None:
            return super().alias
        return self._alias

    @property
    def data_type(self) -> DataType:
        return self._data_type

    def check_exists(
        self,
        connection: duckdb.DuckDBPyConnection,
        table_renamings: Dict[ConcreteTableIdentifier, ConcreteTableIdentifier] = {},
    ):
        assert self.table_name is not None
        renamed_table_name = self.table_name
        if self.table_identifier is not None:
            renamed_table_name = table_renamings.get(
                self.table_identifier, self.table_identifier
            ).name
        cols = connection.execute(f"DESCRIBE {renamed_table_name}").fetchall()
        assert cols is not None, f"Table {renamed_table_name} not found"
        if self not in [
            ConcreteColumnIdentifier(f"{self.table_name}.{col[0]}") for col in cols
        ]:
            raise ValueError(f"Column {self.name} not found in table {self.table_name}")


class RealColumn(ConcreteColumn, RealColumnIdentifier):
    @property
    def column_name(self):
        return self.name.split(".")[1]

    @property
    def table_name(self) -> str:
        return self.name.split(".")[0]

    @property
    def table_identifier(self) -> ConcreteTableIdentifier:
        return ConcreteTableIdentifier(self.table_name)

    def rename_table(self, new_table_name: str):
        new_name = f"{new_table_name}.{self.name.split('.')[1]}"
        return RealColumn(new_name, self.data_type, self.alias)


class UDFColumn(ConcreteColumn):
    def __init__(
        self,
        base_column: ConcreteColumn,
        udf_name: str,
        data_type: DataType,
        alias: str,
        is_expensive: bool,
        is_potentially_flawed: bool,
    ):
        self._base_column = base_column
        self._udf_name = udf_name
        self._alias = alias
        self._data_type = data_type
        self._is_expensive = is_expensive
        self._is_potentially_flawed = is_potentially_flawed
        self._overwrite_fraction = 0.0
        self._logical_plan_step = None

    def rename_table(self, new_table_name: str):
        new_base_column = self._base_column.rename_table(new_table_name)
        return UDFColumn(
            new_base_column,
            self._udf_name,
            self._data_type,
            self.alias,
            self._is_expensive,
            self._is_potentially_flawed,
        )

    def get_certain_keep_flag(self, gold_mixing=False) -> str:
        flag = "TRUE"
        if gold_mixing:
            flag = f"(__random__ >= {self._overwrite_fraction})"
        return flag

    @property
    def get_certain_keep_flag_alias(self) -> str:
        assert self._logical_plan_step is not None
        return f"_flag_{self._logical_plan_step.identifier}"

    def set_overwrite_fraction(self, overwrite_fraction: float):
        self._overwrite_fraction = overwrite_fraction

    def set_logical_plan_step(self, logical_plan_step: "LogicalPlanStep"):
        self._logical_plan_step = logical_plan_step

    @property
    def name(self):
        return str(self)

    @property
    def udf_name(self):
        return self._udf_name

    def __str__(self) -> str:
        return f"{self._udf_name}({self._base_column.no_alias}) AS {self.alias}"

    @property
    def no_alias(self) -> str:
        return f"{self._udf_name}({self._base_column.no_alias})"

    @property
    def table_name(self) -> Optional[str]:
        if self._base_column.table_name is None:
            return None
        return self._base_column.table_name

    def check_exists(
        self,
        connection: duckdb.DuckDBPyConnection,
        table_renamings: Dict[ConcreteTableIdentifier, ConcreteTableIdentifier] = {},
    ):
        self._base_column.check_exists(connection, table_renamings)

    @property
    def is_expensive(self):
        return self._is_expensive

    @property
    def is_potentially_flawed(self):
        return self._is_potentially_flawed

    @property
    def data_type(self) -> DataType:
        return self._data_type

    @property
    def is_supplementary_column(self):
        return False

    @property
    def column_type(self):
        return HiddenColumnType.VALUE_COLUMN

    @property
    def parent_hidden_column_identifiers(self):
        return []


class SimilarityColumn(ConcreteColumn):
    def __init__(
        self,
        base_column: ConcreteColumn,
        description_embedding: np.ndarray,
    ):
        self._base_column = base_column  # embedding column
        self._description_embedding = description_embedding
        hash = hashlib.sha256(description_embedding.tobytes()).hexdigest()
        self._alias = base_column.alias + f"_distance_{hash}"

    def rename_table(self, new_table_name: str):
        new_base_column = self._base_column.rename_table(new_table_name)
        return SimilarityColumn(new_base_column, self._description_embedding)

    @property
    def name(self):
        return str(self)

    def __str__(self) -> str:
        return f"{self.no_alias} AS {self.alias}"

    @property
    def alias(self) -> str:
        # compute using sha256 hash
        if self._alias is None:
            self._alias = sha256(
                self._description_embedding.tobytes() + self._base_column.name.encode()
            ).hexdigest()
        return self._alias

    @property
    def no_alias(self) -> str:
        return f"(1 - array_cosine_distance({self._base_column.no_alias}, CAST({self._description_embedding.tolist()} AS FLOAT[{len(self._description_embedding)}])))"

    @property
    def shortened(self) -> str:
        return f"(1 - array_cosine_distance({self._base_column.no_alias}, CAST([<embedding>] AS FLOAT[{len(self._description_embedding)}])))"

    @property
    def table_name(self) -> Optional[str]:
        if self._base_column.table_name is None:
            return None
        return self._base_column.table_name

    def check_exists(
        self,
        connection: duckdb.DuckDBPyConnection,
        table_renamings: Dict[ConcreteTableIdentifier, ConcreteTableIdentifier] = {},
    ):
        self._base_column.check_exists(connection, table_renamings)


class AggregateColumn(ConcreteColumn):
    def __init__(
        self,
        base_column: Union[ConcreteColumn, StarIdentifier],
        aggregate_function: str,
        data_type: DataType,
        alias: str,
    ):
        self._base_column = base_column
        self._aggregate_function = aggregate_function
        self._alias = alias
        self._data_type = data_type

    @property
    def name(self):
        return str(self)

    def rename_table(self, new_table_name: str):
        if isinstance(self._base_column, StarIdentifier):
            return AggregateColumn(
                self._base_column,
                self._aggregate_function,
                self._data_type,
                self.alias,
            )
        else:
            new_base_column = self._base_column.rename_table(new_table_name)
            return AggregateColumn(
                new_base_column,
                self._aggregate_function,
                self._data_type,
                self.alias,
            )

    def __str__(self) -> str:
        return (
            f"{self._aggregate_function}({self._base_column.no_alias}) AS {self.alias}"
        )

    @property
    def no_alias(self) -> str:
        return f"{self._aggregate_function}({self._base_column.no_alias})"

    @property
    def table_name(self) -> Optional[str]:
        if isinstance(self._base_column, StarIdentifier):
            return None
        else:
            return self._base_column.table_name

    def check_exists(
        self,
        connection: duckdb.DuckDBPyConnection,
        table_renamings: Dict[ConcreteTableIdentifier, ConcreteTableIdentifier] = {},
    ):
        if not isinstance(self._base_column, StarIdentifier):
            self._base_column.check_exists(connection, table_renamings)


class HiddenTableIdentifier(ConcreteTableIdentifier):
    def __init__(self, name: str):
        assert "." not in name
        ConcreteTableIdentifier.__init__(self, name)


class HiddenColumnIdentifier(RealColumnIdentifier):
    def __init__(self, name: str):
        assert "._" in name
        ConcreteColumnIdentifier.__init__(self, name)

    def rename_table(self, new_table_name: str):
        new_name = f"{new_table_name}.{self.name.split('.')[1]}"
        return HiddenColumnIdentifier(new_name)


class HiddenColumnType(enum.IntEnum):
    VALUE_COLUMN = 0
    FILTER_COLUMN = 1
    THRESHOLD_COLUMN = 2


class HiddenColumn(HiddenColumnIdentifier, RealColumn):
    def __init__(
        self,
        name: str,
        data_type: DataType,
        column_type: HiddenColumnType,
        is_expensive: bool,
        is_supplementary_column: bool,
        is_potentially_flawed: bool,
        parent_hidden_column_identifiers: Sequence[HiddenColumnIdentifier] = (),
    ):
        HiddenColumnIdentifier.__init__(self, name)
        ConcreteColumn.__init__(self, name, data_type)
        self._ctype = column_type
        self._is_expensive = is_expensive
        self._is_supplementary_column = is_supplementary_column
        self._is_potentially_flawed = is_potentially_flawed
        self._parent_hidden_column_identifiers = parent_hidden_column_identifiers

    @property
    def column_type(self):
        return self._ctype

    @property
    def is_expensive(self):
        return self._is_expensive

    @property
    def is_supplementary_column(self):
        return self._is_supplementary_column

    @property
    def is_potentially_flawed(self):
        return self._is_potentially_flawed

    @property
    def parent_hidden_column_identifiers(self):
        return self._parent_hidden_column_identifiers

    def rename_table(self, new_table_name: str):
        new_name = f"{new_table_name}.{self.name.split('.')[1]}"
        return HiddenColumn(
            new_name,
            self.data_type,
            self.column_type,
            self.is_expensive,
            self.is_supplementary_column,
            self.is_potentially_flawed,
            self.parent_hidden_column_identifiers,
        )


class DataTypes:
    TRADITIONAL = frozenset(
        [
            DataType.INT,
            DataType.STRING,
            DataType.FLOAT,
            DataType.BOOL,
            DataType.DATE,
            DataType.TIME,
            DataType.DATETIME,
            DataType.DATETIME_WITH_TZ,
        ]
    )
    MULTI_MODAL = frozenset([DataType.TEXT, DataType.IMAGE, DataType.AUDIO])
    TEXT = frozenset([DataType.TEXT])
    IMAGE = frozenset([DataType.IMAGE])
    AUDIO = frozenset([DataType.AUDIO])
    STRING_BASED = frozenset([DataType.STRING, DataType.TEXT])
    NUMBER = frozenset([DataType.INT, DataType.FLOAT])
    DATE = frozenset([DataType.DATE])
    ALL = TRADITIONAL | MULTI_MODAL
    NO_IMAGES = ALL - IMAGE


class MultiModalColumn:
    pass


class RemoteColumn(MultiModalColumn):
    def __init__(self, orig_name, new_name, url: bool = False):
        self.orig_identifier = RealColumnIdentifier(orig_name)
        self.new_identifier = RealColumnIdentifier(new_name)
        assert (
            self.orig_identifier.table_name == self.new_identifier.table_name
        ), "Remote column must have the same table name as the original column."
        self.url = url

    @property
    def table_name(self):
        return self.orig_identifier.table_name


class InPlaceColumn(MultiModalColumn):
    def __init__(self, column_name):
        self.identifier = RealColumnIdentifier(column_name)

    @property
    def table_name(self):
        return self.identifier.table_name

    @property
    def column_name(self):
        return self.identifier.column_name


class IndexColumn:
    def __init__(
        self,
        orig_table: str,
        renamed_table: Optional[str] = None,
        materialized_table: Optional[str] = None,
        origin: Optional["IndexColumn"] = None,
    ):
        self.orig_table = orig_table
        if renamed_table is None:
            renamed_table = orig_table
        self.renamed_table = renamed_table

        if materialized_table is None:
            materialized_table = renamed_table
        self.materialized_table = materialized_table

        self.origin = self
        if origin is not None:
            self.origin = origin

    def rename(
        self, table_renamings: Dict[ConcreteTableIdentifier, ConcreteTableIdentifier]
    ) -> Sequence["IndexColumn"]:
        renamed_table_map = {}
        materialized_table_map = {}
        for new_name, old_name in table_renamings.items():
            suffix = new_name.name.split("_")[-1]
            if old_name.name.lower() == self.renamed_table.lower():
                renamed_table_map[suffix] = new_name.name
            if old_name.name.lower() == self.materialized_table.lower():
                materialized_table_map[suffix] = new_name.name

        if not renamed_table_map:
            return [self]

        result = []
        for suffix in renamed_table_map:
            result.append(
                IndexColumn(
                    orig_table=self.orig_table,
                    renamed_table=renamed_table_map[suffix],
                    materialized_table=materialized_table_map[suffix],
                )
            )
        return result

    def __str__(self):
        raise NotImplementedError()

    @property
    def project_str(self):
        return f"{self.materialized_table}._index_{self.orig_table} AS _index_{self.renamed_table}"

    @property
    def project_no_alias(self):
        return f"{self.materialized_table}._index_{self.orig_table}"

    @property
    def renamed_no_alias(self):
        return f"{self.materialized_table}._index_{self.renamed_table}"

    @property
    def col_name(self):
        return f"_index_{self.renamed_table}"

    @property
    def table_identifier(self):
        return ConcreteTableIdentifier(self.materialized_table)

    @property
    def orig_col_name(self):
        return f"_index_{self.orig_table}"

    def __eq__(self, other):
        if not isinstance(other, IndexColumn):
            return False
        return (
            self.orig_table == other.orig_table
            and self.renamed_table == other.renamed_table
        )

    def __hash__(self):
        return hash((self.orig_table, self.renamed_table))

    def __lt__(self, other):
        if not isinstance(other, IndexColumn):
            return NotImplemented
        return (self.orig_table, self.renamed_table) < (
            other.orig_table,
            other.renamed_table,
        )
