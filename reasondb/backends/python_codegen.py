from abc import abstractmethod
import json
import pandas as pd
import re
import importlib
from typing import Callable, List, Tuple, Type
from reasondb.backends.backend import Backend
from reasondb.database.indentifier import DataType, VirtualColumnIdentifier
from reasondb.reasoning.llm import (
    LargeLanguageModel,
    Message,
    Prompt,
    PromptTemplate,
)
from reasondb.utils.logging import FileLogger
from reasondb.reasoning.exceptions import Mistake
from reasondb.utils.parsing import get_json_from_response

IMPORTS = ["pandas", "datetime", "numpy", "re", "math"]
IMPORT_REGEX = r"(from \w+ |)import (\w+)( as \w+|)"


class PythonCodegenBackend(Backend):
    def __init__(self):
        pass

    @abstractmethod
    async def run(
        self,
        task_description: str,
        input_column: VirtualColumnIdentifier,
        output_column: VirtualColumnIdentifier,
        data_sample: pd.DataFrame,
        logger: FileLogger,
    ) -> Tuple[Callable, DataType]:
        raise NotImplementedError

    @abstractmethod
    async def prepare(self):
        pass

    @abstractmethod
    async def wind_down(self):
        pass


class LLMPythonCodegenBackend(PythonCodegenBackend):
    def __init__(self, llm: LargeLanguageModel):
        self.llm = llm

    async def prepare(self):
        await self.llm.prepare()

    async def wind_down(self):
        await self.llm.close()

    async def run(
        self,
        task_description: str,
        input_column: VirtualColumnIdentifier,
        output_column: VirtualColumnIdentifier,
        data_sample: pd.DataFrame,
        logger: FileLogger,
    ):
        data_sample = data_sample.head(5)
        chat_thread = []
        i = 0
        while True:
            try:
                func, dtype, func_str = await self.get_func(
                    task_description=task_description,
                    data_sample=data_sample,
                    chat_thread=chat_thread,
                    input_column=input_column,
                    output_column=output_column,
                    logger=logger,
                )

                result = data_sample[input_column.column_name].apply(func)

                if result.isnull().all():
                    raise ValueError(
                        "The function returned only None/NaN values. Please ensure the function handles all cases."
                    )
                result = result.astype(dtype)

                reason_db_dtype = DataType.from_pandas(
                    dtype,
                    is_image=False,
                    is_text=False,
                    is_audio=False,
                    example_values=result.head(5).tolist(),
                )

                def augmented_func(x):
                    try:
                        return func(x)
                    except Exception:
                        return 0

                return augmented_func, reason_db_dtype
            except Exception as e:
                if i >= 5:
                    raise Mistake("Python tool failed. Use another tool!")
                chat_thread = await self.handle_errors(
                    chat_thread=chat_thread,
                    error=e,
                    request=task_description,
                    logger=logger,
                )
                i += 1

    def get_operation_identifier(self) -> str:
        return f"LLMPythonCodegenBackend-{self.llm.model_id}"

    async def get_func(
        self,
        task_description: str,
        data_sample: pd.DataFrame,
        chat_thread: List[Message],
        input_column: VirtualColumnIdentifier,
        output_column: VirtualColumnIdentifier,
        logger: FileLogger,
    ) -> Tuple[Callable, Type, str]:
        modules = ", ".join(IMPORTS)
        params = dict(
            explanation=task_description,
            data=str(data_sample[input_column.column_name]),
            dtype=str(type(data_sample[input_column.column_name].iloc[0])),
            first_element=str(data_sample[input_column.column_name].iloc[0]),
            data_len=len(data_sample),
            modules=modules,
            column=input_column.column_name,
            new_column=output_column.column_name,
        )

        if len(chat_thread) == 0:  # Start of conversation
            chat_thread.append(
                Message(
                    role="user",
                    text="{{explanation}}:\n"
                    "```py\n>>> print({{column}}[:{{data_len}}])\n{{data}}\n\n"
                    ">>> print({{column}}[0])\n{{first_element}}\n\n"
                    ">>> print(type({{column}}[0]))\n{{dtype}}\n```\n\n"
                    "It is a pandas Series object. Please call the 'apply' method with a lambda expression, "
                    "and make sure to always call astype() in the same line. Assign the result to a variable called '{{new_column}}'. "
                    "Template to use: `{{new_column}} = {{column}}.apply(lambda x: <code>).astype(<dtype>)`. You can use {{modules}}.",
                )
            )
            prompt = PromptTemplate(messages=list(chat_thread)).fill(**params)
        else:
            prompt = Prompt(messages=list(chat_thread))
        chat_thread.clear()  
        chat_thread.extend(prompt.messages)
        result = await self.llm.invoke(prompt, logger=logger)
        chat_thread.append(Message(role="assistant", text=result))
        match = re.search(
            rf"{output_column.column_name} = (\w+\[\"|\w+\['|){input_column.column_name}(\"\]|'\]|)\.apply\((.*)\)\.astype\((.*)\)",
            result,
        )
        if match is None:
            raise ValueError(
                f"Use correct template: `{output_column.column_name} = {input_column.column_name}.apply(lambda x: <code>).astype(<dtype>)`"
            )

        code, dtype = match[3], match[4]
        dtype_python = {
            "int64": "int",
            "float64": "float",
            "object": "str",
            "bool": "bool",
        }.get(dtype.strip("'\"".lower()), dtype)
        code = re.sub(
            r"^\s*lambda\s+([^:]*):\s+(.*)$",
            rf"lambda \g<1>: {dtype_python}(\g<2>)",
            code,
        )
        functions = self.parse_function_definitions(result)

        function_str = "\n".join(functions)
        function_str = f"{function_str}{input_column}.apply({code}).astype({dtype})"
        loc = self.manage_imports(result, functions)

        func = eval(code, loc)  # get function handler
        dtype = eval(dtype_python, loc)
        return func, dtype, function_str

    def parse_function_definitions(self, result):
        functions = list()
        for m in re.finditer(r"( *)def (\w+)\(.*\):.*(\n\1    .*)+", result):
            indent = len(m[1])
            func = "\n".join(line[indent:] for line in m[0].split("\n"))
            if not re.search(IMPORT_REGEX, func):
                functions.append(func + "\n")
        return functions

    def manage_imports(self, result, functions):
        if "```" in result:
            result = result.split("```")[1]
        loc = {m: importlib.import_module(m) for m in IMPORTS}
        for from_stmt, module, alias in re.findall(IMPORT_REGEX, result):
            from_stmt = [x for x in from_stmt[5:].strip().split(".") if x]
            alias = alias[4:].strip() or module
            module = from_stmt + [module]

            target = loc[module[0]]
            for m in module[1:]:
                target = getattr(target, m)

            loc[alias] = target
            for f in functions:
                exec(f, loc)
        return loc

    async def handle_errors(
        self,
        chat_thread: List[Message],
        error: Exception,
        request: str,
        logger: FileLogger,
    ):
        error_str = f"{type(error).__name__}({error})"
        code = re.search(
            r"\w+ = (\w+\[\"|\w+\['|)\w+(\"\]|'\]|)\.apply\((.*)\)\.astype\((.*)\)",
            chat_thread[-1].text,
        )
        code = code[0] if code is not None else "<could not parse code with template>"
        libraries = ", ".join(IMPORTS)
        prompt = PromptTemplate(
            [
                *chat_thread,
                Message(
                    role="user",
                    text="Something went wrong executing `{{code}}`. This is the error I got: {{error}}. "
                    "Before fixing the code, please answer these four questions:\n"
                    "1. What is the reason for the error?\n"
                    "2. Is there another way to '{{request}}', potentially using another library (from {{libraries}})?\n"
                    "3. Can this be fixed? Or is there something wrong in my request? Answer 'Yes' if it can be fixed, and 'No' otherwise.\n"
                    "4. If it can be fixed, how can it be fixed? If it cannot be fixed, please explain the error and why it cannot be implemented using python.\n"
                    "5. Finally, please fix the code, but make sure you adhere to the template!\n\n"
                    'Reply in JSON as follows:\nThe output should be in this format: {"1": "reason", "2": "another way", "3": "yes/no", "4": "fix_idea", "5": "fixed_code"}',
                ),
            ]
        ).fill(error=error_str, code=code, libraries=libraries, request=request)

        logger.info(__name__, str(prompt))
        result = await self.llm.invoke(prompt, logger=logger)
        logger.info(__name__, result)

        result = get_json_from_response(result, start_char="{")
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            try:
                result = json.loads(result.split('"5": ')[0].strip(" \n,") + "}")
            except json.JSONDecodeError as e:
                raise Mistake(f"Invalid JSON response. Please try again: {e}")
        can_be_fixed = "yes" in result["3"].lower()
        explanation = result["1"]
        fix_idea = result["4"]
        if not can_be_fixed:
            raise Mistake(explanation + "\nError: " + str(error))

        prompt = PromptTemplate(
            [
                *chat_thread,
                Message(
                    role="user",
                    text="Something went wrong executing `{{code}}`. This is the error I got: {{error}}. "
                    "{{explanation}} Please fix it, but make sure you adhere to the template! This is how you could do it: {{fix_idea}}",
                ),
            ]
        ).fill(error=error_str, code=code, explanation=explanation, fix_idea=fix_idea)
        return prompt.messages
