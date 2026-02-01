import asyncio
from collections import defaultdict
from hashlib import sha256
from typing import List
import pandas as pd
import numpy as np

from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.query_plan.logical_plan import LogicalOperatorToolbox, LogicalPlan
from reasondb.query_plan.query import Query
from reasondb.reasoning.exceptions import Mistake
from reasondb.reasoning.few_shot_database import FewShotDatabase
from reasondb.reasoning.llm import LargeLanguageModel
from reasondb.reasoning.reasoners.self_correction import SelfCorrectionReasoner
from reasondb.utils.logging import FileLogger


class SelfConsistencyReasoner(SelfCorrectionReasoner):
    def __init__(
        self,
        llm: LargeLanguageModel,
        configurator: PlanConfigurator,
        logical_operators: LogicalOperatorToolbox,
        few_shot_database: FewShotDatabase,
        num_rollouts: int = 3,
    ):
        super().__init__(llm, configurator, logical_operators, few_shot_database)
        self.num_rollouts = num_rollouts

    async def prepare(self):
        await super().prepare()
        self.test_lock = asyncio.Lock()

    async def _run(self, query: Query, logger: FileLogger) -> LogicalPlan:
        plans = await asyncio.gather(
            *[
                self._run_single_trace(
                    run_id=i, query=query, logger=logger / f"rollout-{i}"
                )
                for i in range(self.num_rollouts)
            ]
        )
        return await self.pick_most_consistent_plan(plans, logger=logger)

    async def pick_most_consistent_plan(
        self, plans: List[LogicalPlan], logger: FileLogger
    ) -> LogicalPlan:
        collect = []
        result = defaultdict(list)
        for logical_plan in plans:
            async with self.test_lock:
                test_output, get_test_len = await self.configurator.get_test_output(
                    logical_plan, logger=logger / "pick-plan"
                )
                collect.append((logical_plan, test_output, get_test_len))
        for logical_plan, test_output, get_test_len in collect:
            hash = self.hash(test_output, get_test_len())
            result[hash].append(logical_plan)
        most_common_hash = max(result, key=lambda x: len(result[x]))
        most_consistent_plans = result[most_common_hash]
        logger.info(
            __name__,
            f"Most consistent plan appeared {len(most_consistent_plans)}/{len(plans)} times",
        )
        return most_consistent_plans[0]

    def hash(self, data: pd.DataFrame, length) -> str:
        data = data.map(str)
        flatten = data.values.flatten()
        row_coords = np.repeat(np.arange(data.shape[0]), data.shape[1])
        col_coords = np.tile(np.arange(data.shape[1]), data.shape[0])
        sort = sorted(zip(flatten, row_coords, col_coords))
        row_order = pd.Index(pd.unique(pd.Series([x[1] for x in sort])))
        col_order = pd.Index(pd.unique(pd.Series([x[2] for x in sort])))
        reshuffled = data.iloc[row_order, col_order]
        concatenated = "--".join(reshuffled.values.flatten()) + f"--{length}"
        hash = sha256(concatenated.encode()).hexdigest()
        return hash

    async def _run_single_trace(
        self, run_id: int, query: Query, logger: FileLogger
    ) -> LogicalPlan:
        assert self.database is not None
        prompt = await self.generate_prompt(
            query, logger=logger / "generate-prompt", temperature=0.8, seed=run_id
        )
        success = False
        logical_plan = None

        while not success:
            response = await self.llm.invoke(prompt, logger)
            try:
                logical_plan = self.logical_operators.parse(response)
                logical_plan.validate(self.database)
                async with self.test_lock:
                    await self.configurator.test_logical_plan(
                        query, logical_plan, logger=logger / "test-plan"
                    )
                success = True
            except Mistake as mistake:
                prompt = self.generate_fix_prompt(mistake, response, prompt)

        if logical_plan is None:
            raise Mistake("Could not generate a logical plan")
        return logical_plan

