import asyncio
import openai
import psutil
import base64
import json
import time
import os
import re
from pathlib import Path
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
from openai import NOT_GIVEN, AsyncOpenAI

# from openai.types.chat import ChatCompletionMessageParam
import hashlib

from reasondb.utils.cache import CACHE_DIR
from reasondb.utils.logging import FileLogger


LLM_CACHE_DIR = CACHE_DIR / Path("llm_cache")


@dataclass
class LLMCharacteristics:
    rpm: int
    tpm: int
    out_len: int
    in_len: int
    in_cost: float  # per million tokens
    out_cost: float  # per million tokens


class LargeLanguageModel(ABC):
    def __init__(
        self,
        model_id,
        characteristics: LLMCharacteristics,
        cache_enabled: bool,
        num_concurrent: int = 20,
    ):
        self.characteristics = characteristics
        self.model_id = model_id
        self.cache_enabled = cache_enabled
        self.num_concurrent = num_concurrent
        self.semaphore = None
        self.cache_semaphore = None

    async def prepare(self):
        self.semaphore = asyncio.Semaphore(self.num_concurrent)
        self.cache_semaphore = asyncio.Semaphore(self.num_concurrent)

    async def close(self):
        pass

    def print_open_files(self):
        proc = psutil.Process()
        print("*** Open Files ***")
        for i, f in enumerate(proc.open_files()):
            print(f"{i}: {f.path}")
        print("*** End Open Files ***")

    async def invoke_with_runtime_and_cost(
        self,
        prompt: "Prompt",
        logger: FileLogger,
        stop: List[str] = [],
    ) -> Tuple[str, float, float]:
        logger.debug(__name__, f"Invoking LLM with prompt: {prompt}")
        response = None
        cached = None
        if self.cache_enabled:
            assert self.cache_semaphore is not None, "Prepare LLM first"
            async with self.cache_semaphore:
                cached = self.get_cached(prompt)
        if cached is None:
            logger.debug(__name__, "Invoking LLM")
            assert self.semaphore is not None, "Prepare LLM first"
            assert self.cache_semaphore is not None, "Prepare LLM first"
            async with self.semaphore:
                response, runtime, cost = await self._invoke(prompt, stop=stop)
            async with self.cache_semaphore:
                # self.print_open_files()
                self.cache(prompt, response, runtime, cost)
        else:
            response, runtime, cost = cached
            logger.debug(__name__, "Using cached response")
        logger.debug(__name__, f"LLM response: {response}")
        return response, runtime, cost

    async def invoke(
        self,
        prompt: "Prompt",
        logger: FileLogger,
        stop: List[str] = [],
    ) -> str:
        return (await self.invoke_with_runtime_and_cost(prompt, logger, stop))[0]

    def cache(self, prompt: "Prompt", response: str, runtime: float, cost: float):
        hash = prompt.sha256()
        LLM_CACHE_DIR.mkdir(exist_ok=True)
        path = LLM_CACHE_DIR / hash
        with open(path, "w") as f:
            json.dump(
                {
                    "prompt": str(prompt),
                    "response": response,
                    "temperature": prompt.temperature,
                    "seed": prompt.seed,
                    "runtime": runtime,
                    "cost": cost,
                },
                f,
            )

    def get_cached(self, prompt: "Prompt") -> Optional[Tuple[str, float, float]]:
        hash = prompt.sha256()
        path = LLM_CACHE_DIR / hash
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        if (
            data["prompt"] == str(prompt)
            and data["temperature"] == prompt.temperature
            and data["seed"] == prompt.seed
            and "runtime" in data
        ):
            return data["response"], data["runtime"], data["cost"]

    @abstractmethod
    async def _invoke(
        self, prompt: "Prompt", stop: List[str] = []
    ) -> Tuple[str, float, float]:
        raise NotImplementedError


class Message:
    def __init__(
        self,
        text: str,
        role: Literal["system", "user", "assistant"],
        image: Optional[Path] = None,
    ):
        self._text = textwrap.dedent(text).strip()
        self._role: Literal["system", "user", "assistant"] = role
        self._image = image

    @property
    def only_text(self) -> str:
        return self._text

    @property
    def text(self) -> str:
        return self._text

    @property
    def role(self) -> Literal["system", "user", "assistant"]:
        return self._role

    @property
    def image(self) -> Optional[Path]:
        return self._image

    def __str__(self):
        string = f"{self.role.capitalize()}: {self.only_text}"
        if self._image is not None:
            string += f" [Image: {self._image}]"
        return string

    def to_tuple(self):
        return (self.role, self.only_text, self.image)


class PromptTemplate:
    def __init__(
        self,
        messages: List["Message"],
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        self._messages = messages
        self.temperature = temperature
        self.keys = set(
            re.findall(
                r"{{(.*?)}}", "\n".join(message.only_text for message in self._messages)
            )
        )
        self.seed = seed
        if self.temperature > 0 and self.seed is None:
            raise ValueError("Seed must be provided when temperature > 0")

    def fill(self, **kwargs) -> "Prompt":
        messages = []
        if kwargs.keys() != self.keys:
            missing_keys = self.keys - kwargs.keys()
            extra_keys = kwargs.keys() - self.keys
            raise ValueError(
                f"Keys do not match. Missing: {missing_keys}, Extra: {extra_keys}"
            )
        for msg in self._messages:
            role, text, image = msg.to_tuple()
            for key, value in kwargs.items():
                text = text.replace("{{" + key + "}}", str(value))
            messages.append(Message(text=text, role=role, image=image))  # type: ignore
        return Prompt(messages=messages, temperature=self.temperature, seed=self.seed)


class Prompt:
    def __init__(
        self,
        messages: List["Message"],
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        self._messages = messages
        self.temperature = temperature
        self.seed = seed

        if self.temperature > 0 and self.seed is None:
            raise ValueError("Seed must be provided when temperature > 0")

        for message in self._messages:
            assert message.role in ["system", "user", "assistant"]

    def __add__(self, other: Message) -> "Prompt":
        return Prompt(
            messages=self._messages + [other],
            temperature=self.temperature,
            seed=self.seed,
        )

    def encode_image(self, image_path: Path):
        # need to resize it in case it is too large
        import io
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = 250000000

        with Image.open(image_path) as image:
            max_size = (1024, 1024)

            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                ratio = max(image.size[0] / max_size[0], image.size[1] / max_size[1])
                new_size = (
                    int(image.size[0] / ratio),
                    int(image.size[1] / ratio),
                )
                image.thumbnail(new_size)
            buffered = io.BytesIO()
            image = image.convert("RGB")
            image.save(buffered, format="JPEG")
            encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return encoded_string

    @property
    def prompt_messages(self):  # -> List[ChatCompletionMessageParam]:
        # result: List[ChatCompletionMessageParam] = []
        result = []
        for message in self._messages:
            if message.image is not None:
                result.append(
                    {
                        "role": message.role,
                        "content": [
                            {"type": "text", "text": message.only_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self.encode_image(message.image)}"
                                },
                            },
                        ],
                    }
                )

            else:
                result.append({"role": message.role, "content": message.only_text})  # type: ignore
        return result

    @property
    def messages(self) -> List["Message"]:
        return self._messages

    def __str__(self) -> str:
        return "\n".join(str(message) for message in self._messages)

    def sha256(self) -> str:
        return hashlib.sha256(
            f"{self}-{self.temperature}-{self.seed}".encode()
        ).hexdigest()


class OpenAILLM(LargeLanguageModel):
    def __init__(self, model_id, characteristics, cache_enabled: bool):
        super().__init__(model_id, characteristics, cache_enabled)
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = None

    async def prepare(self):
        await super().prepare()
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def close(self):
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def _invoke(
        self, prompt: Prompt, stop: List[str] = []
    ) -> Tuple[str, float, float]:
        time_start = time.time()
        assert self.client is not None, "Prepare LLM first"
        try:
            response = await self.client.chat.completions.create(
                messages=prompt.prompt_messages,
                model=self.model_id,
                seed=prompt.seed,
                temperature=prompt.temperature,
                stop=stop or NOT_GIVEN,
            )
        except openai.PermissionDeniedError as e:
            raise RuntimeError(
                "OpenAI Permission Error: Please check if your API key has access to the specified model."
            ) from e
        assert response.choices[0].message.content is not None
        time_end = time.time()
        runtime = time_end - time_start

        assert response.usage is not None
        in_usage = response.usage.prompt_tokens
        out_usage = response.usage.completion_tokens
        in_cost = (in_usage / 1_000_000) * self.characteristics.in_cost
        out_cost = (out_usage / 1_000_000) * self.characteristics.out_cost
        total_cost = in_cost + out_cost
        return response.choices[0].message.content, runtime, total_cost


class GPT4oMini(OpenAILLM):
    def __init__(self):
        super().__init__(
            model_id="gpt-4o-mini-2024-07-18",
            characteristics=LLMCharacteristics(
                rpm=30_000,
                tpm=150_000_000,
                out_len=4096,
                in_len=16_385,
                in_cost=0.15,
                out_cost=0.6,
            ),
            cache_enabled=True,
        )


class GPT4o(OpenAILLM):
    def __init__(self):
        super().__init__(
            model_id="gpt-4o-2024-11-20",
            characteristics=LLMCharacteristics(
                rpm=10_000,
                tpm=30_000_000,
                out_len=4096,
                in_len=16_385,
                in_cost=2.50,
                out_cost=10.0,
            ),
            cache_enabled=True,
        )
