from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Type
from reasondb.database.database import Database
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.query_plan.capabilities import Capability
from reasondb.query_plan.logical_plan import (
    LogicalOperatorToolbox,
    LogicalPlan,
    LogicalPlanStep,
)
from reasondb.query_plan.query import Query
from reasondb.reasoning.few_shot_database import FewShotDatabase
from reasondb.reasoning.llm import LargeLanguageModel
from reasondb.utils.logging import FileLogger


class Reasoner(ABC):
    def __init__(
        self,
        llm: LargeLanguageModel,
        configurator: PlanConfigurator,
        logical_operators: LogicalOperatorToolbox,
        few_shot_database: FewShotDatabase,
    ):
        self._database: Optional[Database] = None
        self.llm = llm
        self.configurator = configurator
        self.logical_operators = logical_operators
        self.few_shot_database = few_shot_database

    async def run(self, query: Query, logger: FileLogger) -> LogicalPlan:
        return await self._run(query, logger)

    async def prepare(self):
        await self.llm.prepare()

    async def wind_down(self):
        await self.llm.close()

    @abstractmethod
    async def _run(self, query: Query, logger: FileLogger) -> LogicalPlan:
        raise NotImplementedError

    @property
    def database(self) -> Database:
        assert self._database is not None
        return self._database

    def set_database(self, database: Database):
        self._database = database

    def get_capabilities(self) -> Dict[Type[LogicalPlanStep], Sequence[Capability]]:
        return {
            logical_step: self.configurator.get_capabilities(logical_step)
            for logical_step in self.logical_operators
        }


ONE_SHOT_REASONER_PROMPT = """
Translate the following query into a multi-modal query plan: {{query}}.
A multi-modal query plan consists of a sequence of logical operators that transform the input data into the desired output.
The operators are capable of analyzing multi-modal data (e.g. looking at images, reading and analyzing text like a human) and can be combined in various ways.
For instance, a filter operator can be used to select rows where a text is of positive sentiment or an extract operator can be used to extract the the number of people depicted in an image.
Each operator requires the user to specify an expression that defines a condition (e.g. for filters, joins, etc.), a transformation (e.g. for extractions, transformations, etc.), etc.
These expressions can contain references to the data columns using curly braces (e.g. {table_name.column_name}), some use square brackets for output columns (e.g. [output_column]), etc.
When specifiying an expression, make sure to keep all necessary detail from the input question (such as example values) and to be as precise as possible.
The expressions should be free from ambiguity (e.g. specify exactly what should be returned).

!IMPORTANT!
Sometimes it is possible that there are multiple ways of getting some information. For instance you want to know the topic of a song and have
the song's text, the single cover and the lyrics available. While all of those could give you some idea of a song's topic, title and cover
are often misleading. Only with the lyrics you can be sure to understand the topic of a song. Thus, in these cases, you must recognize that
there are multiple possibilities and choose the column that most likely contains the information you need, may it be in text, images, traditional column
or any other modality. Do not take computational complexity of getting info from images or texts into account.
Moreover, assume that analyzing images or texts is very reliable. The most important factor is that selected data source contains the required information for all items in the dataset!
So extract information from text or images, if it is more likely to contain the information you need. Also take into
account that you only get a data sample as input, so reason about whether any potential patterns you see there is likely to hold for all values in the data.
For instance, when you see in the sample that the song titles in the sample are in english, it does not necessarily mean that all song titles are in english.
I will get angry if in your reasoning you argue about the efficiency of certain operations or you say that based on your sample using a certain columns is likely sufficient.
!IMPORTANT!

This is the available data: {{data}}.

These are the available_operators: {{logical_operators}}.

Here are some examples: {{examples}}.

Use the following output format:
```json
{{output_format}}
```"""

ONE_SHOT_REASONER_FIX_PROMPT = """
    You made a mistake: {{mistake}}.
    Try again! Output the full query plan.
"""
