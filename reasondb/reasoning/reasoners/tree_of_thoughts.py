import asyncio
from enum import Enum
import json
from typing import List, Sequence, Tuple

from reasondb.database.intermediate_state import IntermediateState
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.query_plan.logical_plan import LogicalOperatorToolbox, LogicalPlan
from reasondb.query_plan.query import Query
from reasondb.reasoning.exceptions import Mistake, ReasoningDeadEnd
from reasondb.reasoning.few_shot_database import FewShotDatabase
from reasondb.reasoning.llm import LargeLanguageModel, Message, Prompt, PromptTemplate
from reasondb.reasoning.reasoner import Reasoner
from reasondb.utils.logging import FileLogger
from reasondb.utils.parsing import get_json_from_response


class ThoughtGenerationStrategy(Enum):
    SAMPLE = 1
    PROPOSE = 2


class StateEvaluationStrategy(Enum):
    VALUE = 1
    VOTE = 2


class TreeOfThoughtsReasoner(Reasoner):
    def __init__(
        self,
        llm: LargeLanguageModel,
        configurator: PlanConfigurator,
        logical_operators: LogicalOperatorToolbox,
        few_shot_database: FewShotDatabase,
        fanout: int = 3,
        thought_generation_strategy: ThoughtGenerationStrategy = ThoughtGenerationStrategy.PROPOSE,
        state_evaluation_strategy: StateEvaluationStrategy = StateEvaluationStrategy.VALUE,
    ):
        super().__init__(llm, configurator, logical_operators, few_shot_database)
        self.fanout = fanout
        self.thought_generation_strategy = thought_generation_strategy
        self.state_evaluation_strategy = state_evaluation_strategy
        self.utility_threshold = 0

    async def prepare(self):
        await super().prepare()
        self.test_lock = asyncio.Lock()

    async def _run(self, query: Query, logger: FileLogger) -> LogicalPlan:
        system_prompt = self.get_system_prompt()
        frontier: List[
            Tuple[LogicalPlan, IntermediateState, Prompt, Prompt, FileLogger]
        ] = list()
        frontier.append(
            (
                LogicalPlan([]),
                IntermediateState(self.database, None),
                system_prompt,
                system_prompt,
                logger,
            )
        )

        while len(frontier):
            (
                current_plan_prefix,
                current_intermediate_state,
                generate_prompt,
                evaluate_prompt,
                node_logger,
            ) = frontier.pop()
            (
                finished,
                next_plan_prefixes,
                generate_prompts,
                child_loggers,
            ) = await self.generate_next_steps(
                query=query,
                intermediate_state=current_intermediate_state,
                plan_prefix=current_plan_prefix,
                prompt_prefix=generate_prompt,
                logger=node_logger,
            )
            if finished:
                return current_plan_prefix

            (
                utilities,
                intermediate_states,
                evaluate_prompts,
            ) = await self.evaluate_plans(
                query=query,
                plan_prefixes=next_plan_prefixes,
                intermediate_state=current_intermediate_state,
                prompt_prefix=evaluate_prompt,
                loggers=child_loggers,
            )
            sorted_by_utility = sorted(
                zip(
                    next_plan_prefixes[::-1],
                    intermediate_states[::-1],
                    utilities[::-1],
                    generate_prompts[::-1],
                    evaluate_prompts[::-1],
                    child_loggers[::-1],
                ),
                key=lambda x: x[2],
            )
            for (
                next_plan_prefix,
                next_intermediate_state,
                utility,
                generate_prompt,
                evaluate_prompt,
                child_logger,
            ) in sorted_by_utility:
                if utility > self.utility_threshold:
                    frontier.append(
                        (
                            next_plan_prefix,
                            next_intermediate_state,
                            generate_prompt,
                            evaluate_prompt,
                            child_logger,
                        )
                    )
        raise Mistake("Could not generate a logical plan")

    def get_system_prompt(self):
        system_prompt = Prompt(
            [
                Message(
                    role="assistant",
                    text="You are a smart and helpful assistant that translates user queries into multi-modal query plans",
                )
            ]
        )
        return system_prompt

    async def generate_next_steps(
        self,
        query: Query,
        plan_prefix: LogicalPlan,
        prompt_prefix: Prompt,
        intermediate_state: IntermediateState,
        logger: FileLogger,
    ) -> Tuple[bool, List[LogicalPlan], List[Prompt], List[FileLogger]]:
        assert self.database is not None
        if self.thought_generation_strategy == ThoughtGenerationStrategy.SAMPLE:
            return await self.sample_next_steps(
                query=query,
                plan_prefix=plan_prefix,
                prompt_prefix=prompt_prefix,
                intermediate_state=intermediate_state,
                logger=logger,
            )
        elif self.thought_generation_strategy == ThoughtGenerationStrategy.PROPOSE:
            return await self.propose_next_steps(
                query=query,
                plan_prefix=plan_prefix,
                prompt_prefix=prompt_prefix,
                intermediate_state=intermediate_state,
                logger=logger,
            )
        else:
            raise ValueError("Invalid thought generation strategy")

    async def propose_next_steps(
        self,
        query: Query,
        plan_prefix: LogicalPlan,
        prompt_prefix: Prompt,
        intermediate_state: IntermediateState,
        logger: FileLogger,
    ) -> Tuple[bool, List[LogicalPlan], List[Prompt], List[FileLogger]]:
        prompt = await self.get_propose_prompt(
            query=query,
            intermediate_state=intermediate_state,
            plan_prefix=plan_prefix,
            prompt_prefix=prompt_prefix,
            logger=logger,
            temperature=0.0,
        )
        response = await self.llm.invoke(prompt, logger)
        finished, options = self.logical_operators.parse_options(response)
        options = [plan_prefix + option for option in options]
        prompts = [prompt + Message(role="assistant", text="...") for _ in options]
        loggers = [logger / f"option-{i}" for i in range(len(options))]
        return finished, options, prompts, loggers

    async def sample_next_steps(
        self,
        query: Query,
        plan_prefix: LogicalPlan,
        prompt_prefix: Prompt,
        intermediate_state: IntermediateState,
        logger: FileLogger,
    ) -> Tuple[bool, List[LogicalPlan], List[Prompt], List[FileLogger]]:
        raise NotImplementedError

    async def evaluate_plans(
        self,
        query: Query,
        intermediate_state: IntermediateState,
        plan_prefixes: List[LogicalPlan],
        prompt_prefix: Prompt,
        loggers: List[FileLogger],
    ) -> Tuple[List[float], List[IntermediateState], List[Prompt]]:
        assert self.database is not None
        if self.state_evaluation_strategy == StateEvaluationStrategy.VALUE:
            return await self.evaluate_plans_value(
                query=query,
                plan_prefixes=plan_prefixes,
                prompt_prefix=prompt_prefix,
                intermediate_state=intermediate_state,
                loggers=loggers,
            )
        elif self.state_evaluation_strategy == StateEvaluationStrategy.VOTE:
            return await self.evaluate_plans_vote(
                query=query,
                plan_prefixes=plan_prefixes,
                prompt_prefix=prompt_prefix,
                intermediate_state=intermediate_state,
                loggers=loggers,
            )
        else:
            raise ValueError("Invalid state evaluation strategy")

    async def evaluate_plans_value(
        self,
        query: Query,
        plan_prefixes: List[LogicalPlan],
        prompt_prefix: Prompt,
        intermediate_state: IntermediateState,
        loggers: List[FileLogger],
    ) -> Tuple[List[float], List[IntermediateState], List[Prompt]]:
        scores = []
        intermediate_states = []
        prompts = []
        for plan_prefix, logger in zip(plan_prefixes, loggers):
            try:
                new_intermediate_state = (
                    await self.configurator.test_logical_plan_prefix(
                        query=query, plan_prefix=plan_prefix, logger=logger
                    )
                )
            except (Mistake, ReasoningDeadEnd):
                scores.append(0.0)
                intermediate_states.append(intermediate_state)
                prompts.append(prompt_prefix)
                continue
            prompt = await self.get_value_prompt(
                query=query,
                plan_prefix=plan_prefix,
                new_intermediate_state=new_intermediate_state,
                prompt_prefix=prompt_prefix,
                logger=logger,
                temperature=0.0,
            )
            response = await self.llm.invoke(prompt, logger)
            score = float(
                json.loads(get_json_from_response(response, start_char="{"))["rating"]
            )
            scores.append(score)
            intermediate_states.append(new_intermediate_state)
            prompts.append(prompt + Message(role="assistant", text=response))
        return scores, intermediate_states, prompts

    async def evaluate_plans_vote(
        self,
        query: Query,
        plan_prefixes: Sequence[LogicalPlan],
        prompt_prefix: Prompt,
        intermediate_state: IntermediateState,
        loggers: List[FileLogger],
    ) -> Tuple[List[float], List[IntermediateState], List[Prompt]]:
        raise NotImplementedError

    async def get_propose_prompt(
        self,
        query: Query,
        plan_prefix: LogicalPlan,
        prompt_prefix: Prompt,
        intermediate_state: IntermediateState,
        logger: FileLogger,
        temperature=0.0,
        seed=0,
    ) -> Prompt:
        if len(plan_prefix) == 0:
            template_text = PROPOSE_PROMPT_FIRST_STEP
        else:
            template_text = PROPOSE_PROMPT_FOLLOW_UP_STEP
        template = PromptTemplate(
            [
                *prompt_prefix.messages,
                Message(
                    role="user",
                    text=template_text,
                ),
            ],
            temperature=temperature,
            seed=seed,
        )

        if len(plan_prefix) > 0:
            prompt = template.fill(
                data=await intermediate_state.for_prompt(query, logger=logger),
                picked_step=plan_prefix[-1],
                fanout=self.fanout,
            )
        else:
            output_format = self.logical_operators.get_output_format_options(
                step_id=len(plan_prefix), fanout=self.fanout
            )
            prompt = template.fill(
                query=query.query,
                data=await intermediate_state.for_prompt(query, logger=logger),
                logical_operators=self.logical_operators.for_prompt(
                    self.get_capabilities()
                ),
                examples=self.few_shot_database.retrieve(query).few_shot_for_prompt(),
                output_format=output_format,
                fanout=self.fanout,
            )
        return prompt

    async def get_value_prompt(
        self,
        query: Query,
        plan_prefix: LogicalPlan,
        prompt_prefix: Prompt,
        new_intermediate_state: IntermediateState,
        logger: FileLogger,
        temperature=0.0,
        seed=0,
    ) -> Prompt:
        if len(plan_prefix) == 1:
            value_prompt = VALUE_PROMPT_FIRST_STEP
        else:
            value_prompt = VALUE_PROMPT_FOLLOW_UP_STEP
        template = PromptTemplate(
            [
                *prompt_prefix.messages,
                Message(
                    role="user",
                    text=value_prompt,
                ),
            ],
            temperature=temperature,
            seed=seed,
        )
        if len(plan_prefix) == 1:
            prompt = template.fill(
                query=query.query,
                logical_operators=self.logical_operators.for_prompt(
                    self.get_capabilities()
                ),
                examples=self.few_shot_database.retrieve(query).few_shot_for_prompt(),
                proposed_step=plan_prefix[-1],
                data_after_step=await new_intermediate_state.for_prompt(
                    query, logger=logger
                ),
                init_data=await self.database.for_prompt(query, logger=logger),
            )
        else:
            prompt = template.fill(
                proposed_step=plan_prefix[-1],
                data_after_step=await new_intermediate_state.for_prompt(
                    query, logger=logger
                ),
            )
        return prompt


PROPOSE_PROMPT_FIRST_STEP = """
Translate the following query into a multi-modal query plan: {{query}}.
A multi-modal query plan consists of a sequence of logical operators that transform the input data into the desired output.
The operators are capable of analyzing multi-modal data (e.g. looking at images, reading and analyzing text like a human) and can be combined in various ways.
For instance, a filter operator can be used to select rows where a text is of positive sentiment or an extract operator can be used to extract the the number of people depicted in an image.
Each operator requires the user to specify an expression that defines a condition (e.g. for filters, joins, etc.), a transformation (e.g. for extractions, transformations, etc.), etc.
These expressions can contain references to the data columns using curly braces (e.g. {table_name.column_name}), some use square brackets for output columns (e.g. [output_column]), etc.
When specifiying an expression, make sure to keep all necessary detail from the input question (such as example values) and to be as precise as possible.
If the input question is ambiguous, try to make a reasonable assumption and resolve the ambiguity in your expressions.

Importantly, reason about which information can be better extracted from structured data (strings and numbers) and which information
is better extracted from multi-modal data (images and texts).

This is the available data: {{data}}.

These are the available_operators: {{logical_operators}}.

Here are some examples: {{examples}}.

Importantly, I want you to translate the query step by step. Thus, in this step, only provide the first step of the plan.
Please propose several possible first steps, if applicable. Please propose up to {{fanout}} possible first steps.

Use the following output format:
```json
{{output_format}}
```"""

PROPOSE_PROMPT_FOLLOW_UP_STEP = """
I pick: ```json
{{picked_step}}
```

This is the available data after executing the picked steps above: {{data}}.

Now, only provide the next step of the plan.

Please propose several possible next steps, if applicable. Please propose up to {{fanout}} possible next steps.
Use the same output format as before. If you think that the plan is complete, set finished to true and leave the options_for_step_x empty.
"""

VALUE_PROMPT_FIRST_STEP = """
Help me translate the following query into a multi-modal query plan: {{query}}.
A multi-modal query plan consists of a sequence of logical operators that transform the input data into the desired output.
The operators are capable of analyzing multi-modal data (e.g. looking at images, reading and analyzing text like a human) and can be combined in various ways.
For instance, a filter operator can be used to select rows where a text is of positive sentiment or an extract operator can be used to extract the the number of people depicted in an image.
Each operator requires the user to specify an expression that defines a condition (e.g. for filters, joins, etc.), a transformation (e.g. for extractions, transformations, etc.), etc.
These expressions can contain references to the data columns using curly braces (e.g. {table_name.column_name}), some use square brackets for output columns (e.g. [output_column]), etc.
When specifiying an expression, make sure to keep all necessary detail from the input question (such as example values) and to be as precise as possible.
If the input question is ambiguous, try to make a reasonable assumption and resolve the ambiguity in your expressions.

This is the available data: {{init_data}}.

These are the available_operators: {{logical_operators}}.

Here are some examples: {{examples}}.

I already have an idea of how I could start the multi-modal query plan. Can you tell me whether my proposed first step makes sense and is the best in this situation.
Please rate the following propsed first step with a value between 0 and 10:
- A value of 0 means that you are absolutely certain that the proposed first step is incorrect and cannot be part of the plan.
- A value of 3 means if the proposed step likely does not work in this situation.
- A value of 5 means if you are uncertain if the proposed first step is the best in this situation.
- A value of 8 means if the  proposed step is likely the best in this situation.
- A value of 10 means that you are absolutely certain that the proposed first step is correct and the best in this situation.
Please also use the values 1, 2, 4, 6, 7, 9 to express your uncertainty.

This is the proposed first step: {{proposed_step}}.

And this is how the data looks like after executing the first step: {{data_after_step}}.

Use the following output format:
```json
{
    "explanation": "Explain your reasoning for your rating",
    "rating": "Your rating"
}
```"""

VALUE_PROMPT_FOLLOW_UP_STEP = """
Thanks, I think I'll give it a shot. What about the next step. Again, I have a proposal and would like to know how good you think it is.

This is the proposed next step: {{proposed_step}}.

And this is how the data looks like after executing the next step: {{data_after_step}}.

Use the same rating schema and output format as before.
```"""
