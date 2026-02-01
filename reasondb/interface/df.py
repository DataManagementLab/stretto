import asyncio
import re
from typing import TYPE_CHECKING, Optional, Sequence
from reasondb.database.indentifier import (
    ConcreteTableIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.table import ConcreteTable
from reasondb.database.virtual_table import RootTable
from reasondb.interface.base_query_interface import QueryInterface

from reasondb.optimizer.guarantees import Guarantee

from reasondb.query_plan.logical_plan import LogicalPlanStep, LogicalRename
from reasondb.query_plan.logical_plan import (
    LogicalFilter,
    LogicalAggregate,
    LogicalExtract,
    LogicalGroupBy,
    LogicalJoin,
    LogicalPlan,
    LogicalProject,
    LogicalSorting,
    LogicalLimit,
    LogicalTransform,
    # LogicalOffset,
)

import uuid
import random


if TYPE_CHECKING:
    from reasondb.interface.connect import RaccoonDB

rng = random.Random(43)


class DataFrameInterface(QueryInterface):
    def __init__(
        self,
        connection: "RaccoonDB",
        table_name: str,
        parents: Sequence["DataFrameInterface"],  # parents as links for a linked list
        step: LogicalPlanStep,
        # scope:list = []
    ):
        self.connection = connection
        self.table_name = table_name
        self.parents = parents
        self.step = step

    def get_uuid(self) -> uuid.UUID:
        return uuid.UUID(int=rng.getrandbits(128), version=4)

    def make_fully_qualified(
        self,
        cond_exp: str,
        table_names: Sequence[str],
        prefixes: Optional[Sequence[str]] = None,
        expand_delimiter: str = r" = ",
    ) -> str:
        cond_exp = cond_exp.strip()
        if prefixes is None:
            assert len(table_names) == 1

            cond_exp = re.sub(
                pattern=r"\{([A-Za-z_]\w*)\}",
                repl=rf"{{{table_names[0]}.\1}}",
                string=cond_exp,
                flags=re.IGNORECASE,
            )

        else:
            assert len(prefixes) == len(table_names)
            assert len(prefixes) > 1

            # added set of lines for the condition

            # case 1: "join on {sender}"  -> fully-qualify on both sides using expand_delimiter
            repl = expand_delimiter.join([rf"{{{t}.\1}}" for t in table_names])
            cond_exp = re.sub(
                pattern=r"\{([A-Za-z_]\w*)\}",
                repl=repl,
                string=cond_exp,
                flags=re.IGNORECASE,
            )

            # case 2: allow {left.col} / {right.col} placeholders
            for t, p in zip(table_names, prefixes):
                cond_exp = re.sub(
                    pattern=rf"\{{{p}\.([A-Za-z_]\w*)\}}",
                    repl=rf"{{{t}.\1}}",
                    string=cond_exp,
                    flags=re.IGNORECASE,
                )
        return cond_exp

    def filter(self, cond_exp: str) -> "DataFrameInterface":
        if self.parents == []:
            table_in = VirtualTableIdentifier(self.table_name)
            # table_in = VirtualTableIdentifier("emails")     #table_in = RaccoonDB.database.
        else:
            table_in = self.step.output

        table_out = VirtualTableIdentifier(f"{self.table_name}-{str(self.get_uuid())}")

        cond_exp = self.make_fully_qualified(cond_exp, [table_in.name])
        new_step = LogicalFilter(  # self.Step
            explanation=f"Filter the data by: {cond_exp}",
            inputs=[table_in],
            output=table_out,
            expression=cond_exp,
        )

        return DataFrameInterface(
            connection=self.connection,
            table_name=self.table_name,
            parents=[self],
            step=new_step,
        )

    def project(self, cond_exp: str) -> "DataFrameInterface":
        if self.parents == []:
            table_in = VirtualTableIdentifier(self.table_name)
            # table_in = VirtualTableIdentifier("emails")
        else:
            table_in = self.step.output

        table_out = VirtualTableIdentifier(f"{self.table_name}-{str(self.get_uuid())}")

        cond_exp = self.make_fully_qualified(cond_exp, [table_in.name])
        new_step = LogicalProject(
            explanation=f"Project the data by: {cond_exp}",
            inputs=[table_in],
            output=table_out,
            expression=cond_exp,
        )

        return DataFrameInterface(
            connection=self.connection,
            table_name=self.table_name,
            parents=[self],
            step=new_step,
        )

    def join(self, other: "DataFrameInterface", cond_exp: str) -> "DataFrameInterface":
        # cond expr uses left and right as default identifier?
        if self.parents == []:
            table_in_left = VirtualTableIdentifier(self.table_name)
        else:
            table_in_left = self.step.output

        if other.parents == []:
            table_in_right = VirtualTableIdentifier(other.table_name)
        else:
            table_in_right = other.step.output

        # new added lines
        left_id = table_in_left.name
        right_id = table_in_right.name

        table_out = VirtualTableIdentifier(
            f"{self.table_name}-{other.table_name}-{str(self.get_uuid())}"
        )

        # new added line
        # cond = re.sub(r"\{[^}]+\.(\w+)\}", r"{\1}", cond_exp)
        cond = cond_exp.strip()

        # added set of lines for the condition

        # case 1: "join on {sender}"  -> fully-qualify on both sides
        m = re.fullmatch(r"join on\s*\{([A-Za-z_]\w*)\}", cond, re.IGNORECASE)
        if m:
            col = m.group(1)
            cond = f"join on {{{left_id}.{col}}} = {{{right_id}.{col}}}"

        # case 2: allow {left.col} / {right.col} placeholders
        cond = re.sub(
            r"\{left\.([A-Za-z_]\w*)\}", rf"{{{left_id}.\1}}", cond, flags=re.IGNORECASE
        )
        cond = re.sub(
            r"\{right\.([A-Za-z_]\w*)\}",
            rf"{{{right_id}.\1}}",
            cond,
            flags=re.IGNORECASE,
        )

        # replaced cond_exp with cond in the logicalJoin below

        new_step = LogicalJoin(  # self.Step
            explanation=f"Joins two tables on: {cond}",
            inputs=[table_in_left, table_in_right],
            output=table_out,
            # expression="{a.sender} = {b.recipient}"
            expression=cond,
        )

        return DataFrameInterface(
            connection=self.connection,
            table_name=self.table_name,  # maintains the tablename of the left table/self -> should make nor difference
            parents=[self, other],
            step=new_step,
        )

    def groupby(self, cond_exp: str) -> "DataFrameInterface":
        if self.parents == []:
            table_in = VirtualTableIdentifier(self.table_name)
            # table_in = VirtualTableIdentifier("emails")
        else:
            table_in = self.step.output

        table_out = VirtualTableIdentifier(f"{self.table_name}-{str(self.get_uuid())}")

        cond_exp = self.make_fully_qualified(cond_exp, [table_in.name])
        new_step = LogicalGroupBy(
            explanation=f"Group the data by: {cond_exp}",
            inputs=[table_in],
            output=table_out,
            expression=cond_exp,
        )

        return DataFrameInterface(
            connection=self.connection,
            table_name=self.table_name,
            parents=[self],
            step=new_step,
        )

    def aggregate(self, cond_exp: str) -> "DataFrameInterface":
        if self.parents == []:
            raise ValueError("Aggregate must follow a groupby operation")

        table_in = self.step.output
        # table_out = VirtualTableIdentifier(table_in.name + "_aggregated")
        table_out = VirtualTableIdentifier(f"{self.table_name}-{str(self.get_uuid())}")

        new_step = LogicalAggregate(
            explanation=f"Aggregate the grouped data: {cond_exp}",
            inputs=[table_in],
            output=table_out,
            expression=cond_exp,
        )

        cond_exp = self.make_fully_qualified(cond_exp, [table_in.name])
        return DataFrameInterface(
            connection=self.connection,
            table_name=self.table_name,
            parents=[self],
            step=new_step,
        )

    #  raise ValueError("Aggregate must follow a groupby operator")
    def orderby(self, cond_exp: str) -> "DataFrameInterface":
        if self.parents == []:
            table_in = VirtualTableIdentifier(self.table_name)
            # table_in = VirtualTableIdentifier("emails")
        else:
            table_in = self.step.output

        table_out = VirtualTableIdentifier(f"{self.table_name}-{str(self.get_uuid())}")

        cond_exp = self.make_fully_qualified(cond_exp, [table_in.name])
        new_step = LogicalSorting(
            explanation=f"Order the data by: {cond_exp}",
            inputs=[table_in],
            output=table_out,
            expression=cond_exp,
        )

        return DataFrameInterface(
            connection=self.connection,
            table_name=self.table_name,
            parents=[self],
            step=new_step,
        )

    def limit(self, limit_exp: str) -> "DataFrameInterface":
        if self.parents == []:
            table_in = VirtualTableIdentifier(self.table_name)
            # table_in = VirtualTableIdentifier("emails")
        else:
            table_in = self.step.output

        table_out = VirtualTableIdentifier(f"{self.table_name}-{str(self.get_uuid())}")

        new_step = LogicalLimit(
            explanation=f"Limit the data to: {limit_exp}",
            inputs=[table_in],
            output=table_out,
            expression=limit_exp,  # func needs int, but logical needs str
        )

        return DataFrameInterface(
            connection=self.connection,
            table_name=self.table_name,
            parents=[self],
            step=new_step,
        )

    def offset(self, offset: int) -> "DataFrameInterface":
        # skipping the first x amount of rows?
        # logical offset -->> not implemented
        raise NotImplementedError("Offset method not implemented")

    def extract(self, cond_exp: str) -> "DataFrameInterface":
        if self.parents == []:
            table_in = VirtualTableIdentifier(self.table_name)
            # table_in = VirtualTableIdentifier("emails")
        else:
            table_in = self.step.output

        table_out = VirtualTableIdentifier(f"{self.table_name}-{str(self.get_uuid())}")

        cond_exp = self.make_fully_qualified(cond_exp, [table_in.name])
        new_step = LogicalExtract(
            explanation=f"Extract from the data: {cond_exp}",
            inputs=[table_in],
            output=table_out,
            expression=cond_exp,
        )

        return DataFrameInterface(
            connection=self.connection,
            table_name=self.table_name,
            parents=[self],
            step=new_step,
        )

    def rename(self, rename_exp: str) -> "DataFrameInterface":
        if self.parents == []:
            table_in = VirtualTableIdentifier(self.table_name)
        else:
            table_in = self.step.output

        table_out = VirtualTableIdentifier(f"{self.table_name}-{str(self.get_uuid())}")

        rename_exp = self.make_fully_qualified(rename_exp, [table_in.name])
        new_step = LogicalRename(
            explanation=f"Rename the columns: {rename_exp}",
            inputs=[table_in],
            output=table_out,
            expression=rename_exp,
        )

        return DataFrameInterface(
            connection=self.connection,
            table_name=self.table_name,
            parents=[self],
            step=new_step,
        )

    def transform(self, cond_exp: str) -> "DataFrameInterface":
        if self.parents == []:
            table_in = VirtualTableIdentifier(self.table_name)
            # table_in = VirtualTableIdentifier("emails")
        else:
            table_in = self.step.output

        table_out = VirtualTableIdentifier(f"{self.table_name}-{str(self.get_uuid())}")

        cond_exp = self.make_fully_qualified(cond_exp, [table_in.name])
        new_step = LogicalTransform(
            explanation=f"Transform the data like: {cond_exp}",
            inputs=[table_in],
            output=table_out,
            expression=cond_exp,
        )

        return DataFrameInterface(
            connection=self.connection,
            table_name=self.table_name,
            parents=[self],
            step=new_step,
        )

    def _flatten_plan_(self):
        # recursivly traverses the parent network to construct a flat list of operations

        if self.parents == []:
            return []

        if isinstance(self.step, LogicalJoin):  # self.Step is LogicalJoin:
            flat_list_0 = self.parents[0]._flatten_plan_()
            flat_list_1 = self.parents[1]._flatten_plan_()
            flat_list = flat_list_0 + flat_list_1
        else:
            flat_list = self.parents[0]._flatten_plan_()

        flat_list.append(self.step)

        return flat_list

    def execute(self, name, *guarantees: Guarantee) -> "TableInterface":

        # flatten the nexted linked list of parents into a flat list in correct order
        plan_steps = self._flatten_plan_()

        logical_plan = LogicalPlan(plan_steps)

        logger = self.connection.executor.logger / f"nl_query_{name}"
        result, _ = asyncio.run(
            self.connection.executor.execute_logical_plan(
                logical_plan,
                guarantees=guarantees,
                logger=logger,
            )
        )
        self.connection.executor.database.register_query_result(name, result)
        self.connection.executor.clean_up()
        return TableInterface(self.connection, name)


class TableInterface(DataFrameInterface):
    def __init__(self, connection: "RaccoonDB", name: str):
        self.connection = connection
        self.table_name = name
        self.root_table = RootTable(
            self.connection.database,
            ConcreteTable(ConcreteTableIdentifier(name), self.connection.database),
        )
        # ---
        self.parents: Sequence["DataFrameInterface"] = []

    #  def pprint(self):
    #  return self.root_table.pprint()
    def pprint(self):
        self.connection.prepare()
        #  name = getattr(self, "table_name", "unknown")
        print(f"Benchmark {self.table_name}:")
        self.root_table.pprint()
        print("*" * 80)

    def to_df(self):
        self.connection.prepare()
        #  name = getattr(self, "table_name", "unknown")
        return self.root_table.to_df()
