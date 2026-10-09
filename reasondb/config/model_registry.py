from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


MODEL_SPECS = {
    # Llama family
    "llama8B": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "port": 5010,
        "method_prefix": "kv8B",
        "family": "llama",
        "size_class": "small",
    },
    "llama70B": {
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "port": 5012,
        "method_prefix": "kv70B",
        "family": "llama",
        "size_class": "large",
    },
    # Mistral family
    # Dense text Mistral (MistralForCausalLM, flat decoder) — the small-text analog of
    # Qwen2.5-7B. NOTE: the multimodal Ministral-3 (Mistral3ForConditionalGeneration)
    # does NOT work through the text server, whose decoder access assumes a flat
    # `model.model.layers`; that checkpoint belongs to the vision entry (mistralVL8B).
    "mistral8B": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "port": 5020,
        "method_prefix": "kvMistral8B",
        "family": "mistral",
        "size_class": "small",
    },
    # Dense text Mistral Small 3 (MistralForCausalLM, flat decoder) — the large-text
    # model. NOTE: use the 2501 (3.0) checkpoint, which is text-only; Mistral-Small-3.1
    # (2503) and the Ministral-3 line are multimodal Mistral3 and do NOT run through the
    # text server (its decoder access assumes a flat `model.model.layers`).
    "mistralSmall24B": {
        "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
        "port": 5022,
        "method_prefix": "kvMistralSmall24B",
        "family": "mistral",
        "size_class": "large",
    },
    # Qwen family
    "qwen7B": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "port": 5030,
        "method_prefix": "kvQwen7B",
        "family": "qwen",
        "size_class": "small",
    },
    "qwen72B": {
        "model_name": "Qwen/Qwen2.5-72B-Instruct",
        "port": 5032,
        "method_prefix": "kvQwen72B",
        "family": "qwen",
        "size_class": "large",
    },
    # Vision-language (VL) models — served by kv_cache_image_qa_server.py, not the
    # text server. LLaVA uses ports 5008/5009; the other VL models are even-spaced
    # from 5040 like the text families above.
    "llava8B": {
        "model_name": "llava-hf/llama3-llava-next-8b-hf",
        "port": 5009,
        "method_prefix": "kvLlava8B",
        "family": "llava",
        "size_class": "small",
        "modality": "vision",
    },
    "llava72B": {
        "model_name": "llava-hf/llava-next-72b-hf",
        "port": 5008,
        "method_prefix": "kvLlava72B",
        "family": "llava",
        "size_class": "large",
        "modality": "vision",
    },
    "qwenVL8B": {
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "port": 5040,
        "method_prefix": "kvQwenVL8B",
        "family": "qwen",
        "size_class": "small",
        "modality": "vision",
    },
    "qwenVL32B": {
        "model_name": "Qwen/Qwen3-VL-32B-Instruct",
        "port": 5042,
        "method_prefix": "kvQwenVL32B",
        "family": "qwen",
        "size_class": "large",
        "modality": "vision",
    },
    # Mistral 3 VL family: Mistral3ForConditionalGeneration (Pixtral tower + Mistral
    # decoder). Image tokens get ordinary sequential 1D positions in the decoder, so
    # KV compression, key rerotation, and index reconstruction work unchanged.
    # NOTE: mistralVL8B is the SAME checkpoint as the text entry mistral8B — Ministral 3
    # is natively multimodal; the text server just never feeds it images. It is
    # registered twice (once per modality) with distinct ports/method prefixes.
    "mistralVL8B": {
        "model_name": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "port": 5044,
        "method_prefix": "kvMistralVL8B",
        "family": "mistral",
        "size_class": "small",
        "modality": "vision",
    },
    # 3.1 rather than 3.2: the 3.2 repo ships only mistral-common tokenizer files
    # (tekken.json, no HF processor configs), so AutoProcessor cannot load it; 3.1 is
    # the same Mistral3 vision architecture with complete HF tokenizer+processor files.
    "mistralVL24B": {
        "model_name": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "port": 5046,
        "method_prefix": "kvMistralVL24B",
        "family": "mistral",
        "size_class": "large",
        "modality": "vision",
    },
}

COMPRESSION_RATIOS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

COST_SCHEDULE = {
    0.99: 0.01,
    0.9: 0.1,
    0.8: 0.3,
    0.7: 0.375,
    0.6: 0.45,
    0.5: 0.6,
    0.4: 0.75,
    0.3: 0.85,
    0.2: 0.87,
    0.0: 0.9,
}

COST_OFFSET_LARGE = 0.5


def _cr_to_tag(cr: float) -> str:
    if cr == 0.99:
        return "099"
    if cr == 0.0:
        return "00"
    s = f"{cr:.1f}".replace("0.", "0")
    return s


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    port: int
    method_prefix: str
    family: str
    size_class: str
    modality: str = "text"  # "text" | "vision"


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None

    def __init__(self) -> None:
        self._specs: Dict[str, ModelSpec] = {}
        # Keyed by (model_name, modality); modality None holds the first-registered
        # spec for that name (text entries come first, so text wins for checkpoints
        # registered under both modalities, e.g. Ministral-3-8B).
        self._by_model_name: Dict[Tuple[str, Optional[str]], ModelSpec] = {}
        self._by_prefix: Dict[str, ModelSpec] = {}
        for key, entry in MODEL_SPECS.items():
            spec = ModelSpec(key=key, **entry)
            self._specs[key] = spec
            self._by_model_name.setdefault((spec.model_name, spec.modality), spec)
            self._by_model_name.setdefault((spec.model_name, None), spec)
            self._by_prefix[spec.method_prefix] = spec

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _filtered(self, modality: Optional[str]) -> List[ModelSpec]:
        if modality is None:
            return list(self._specs.values())
        return [s for s in self._specs.values() if s.modality == modality]

    def port_map(self, modality: Optional[str] = "text") -> Dict[str, int]:
        """Model name → port. Defaults to text models;
        pass modality="vision" for VL servers or None for everything."""
        return {s.model_name: s.port for s in self._filtered(modality)}

    def all_model_names(self, modality: Optional[str] = "text") -> List[str]:
        """Defaults to text models — these feed the text QA server's --model-name
        default/choices. Pass modality="vision" or None to widen."""
        return [s.model_name for s in self._filtered(modality)]

    def all_keys(self, modality: Optional[str] = None) -> List[str]:
        return [s.key for s in self._filtered(modality)]

    def spec_by_key(self, key: str) -> ModelSpec:
        return self._specs[key]

    def spec_by_model_name(
        self, name: str, modality: Optional[str] = None
    ) -> Optional[ModelSpec]:
        """A checkpoint can be registered under several modalities. modality=None
        returns the first-registered spec (text wins for dual-registered names);
        pass "text"/"vision" to disambiguate."""
        return self._by_model_name.get((name, modality))

    def spec_by_prefix(self, prefix: str) -> Optional[ModelSpec]:
        return self._by_prefix.get(prefix)

    def method_config(self, modality: Optional[str] = "text") -> Dict[str, Tuple[str, float]]:
        """Method tag → (model name, compression ratio). Defaults to text models so the
        text-side consumers (cache generation, benchmark configurators) see only text
        methods; pass modality="vision" or None to include VL methods."""
        result: Dict[str, Tuple[str, float]] = {}
        for spec in self._filtered(modality):
            for cr in COMPRESSION_RATIOS:
                tag = _cr_to_tag(cr)
                method_name = f"{spec.method_prefix}{tag}"
                result[method_name] = (spec.model_name, cr)
        return result

    def cost_for(self, cr: float, size_class: str) -> float:
        base = COST_SCHEDULE.get(cr, 0.5)
        if size_class == "large":
            base += COST_OFFSET_LARGE
        return base
