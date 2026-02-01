from reasondb.query_plan.logical_plan import (
    LogicalPlan,
)
from reasondb.query_plan.query import Query
from reasondb.reasoning.exceptions import Mistake
from reasondb.reasoning.llm import Message, Prompt, PromptTemplate
from reasondb.reasoning.reasoner import (
    ONE_SHOT_REASONER_FIX_PROMPT,
    ONE_SHOT_REASONER_PROMPT,
    Reasoner,
)
from reasondb.utils.logging import FileLogger


class SelfCorrectionReasoner(Reasoner):
    async def _run(self, query: Query, logger: FileLogger) -> LogicalPlan:
        assert self.database is not None
        prompt = await self.generate_prompt(query, logger=logger / "generate-prompt")
        success = False
        logical_plan = None

        while not success:
            response = await self.llm.invoke(prompt, logger)
            try:
                logical_plan = self.logical_operators.parse(response)
                logical_plan.validate(self.database)
                await self.configurator.test_logical_plan(
                    query, logical_plan, logger=logger / "test-plan"
                )
                success = True
            except Mistake as mistake:
                prompt = self.generate_fix_prompt(mistake, response, prompt)

        if logical_plan is None:
            raise Mistake("Could not generate a logical plan")
        return logical_plan

    async def generate_prompt(
        self, query: Query, logger: FileLogger, temperature=0.0, seed=0
    ) -> Prompt:
        prompt = PromptTemplate(
            [
                Message(
                    role="system",
                    text="You are a smart and helpful assistant that translates user queries into multi-modal query plans",
                ),
                Message(
                    role="user",
                    text=ONE_SHOT_REASONER_PROMPT,
                ),
            ],
            temperature=temperature,
            seed=seed,
        ).fill(
            query=query.query,
            data=await self.database.for_prompt(query, logger=logger),
            logical_operators=self.logical_operators.for_prompt(
                self.get_capabilities()
            ),
            examples=self.few_shot_database.retrieve(query).few_shot_for_prompt(),
            output_format=self.logical_operators.get_output_format(),
        )
        return prompt

    def generate_fix_prompt(
        self, mistake: Mistake, response: str, prompt: Prompt
    ) -> Prompt:
        prompt = PromptTemplate(
            prompt.messages
            + [
                Message(role="assistant", text=response),
                Message(
                    role="user",
                    text=ONE_SHOT_REASONER_FIX_PROMPT,
                ),
            ],
            temperature=0.0,
        ).fill(
            mistake=mistake,
        )
        return prompt
