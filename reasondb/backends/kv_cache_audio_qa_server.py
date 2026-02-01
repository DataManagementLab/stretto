import argparse
import json
import asyncio
import logging
from tqdm import tqdm
from flask import Flask, request
from flask_restful import Resource, Api
from typing import List

from reasondb.backends.kv_cache_base import KVCachingBackendBase
from reasondb.backends.audio_model import PORT_KV_AUDIO
import librosa
import torch
from kvpress import ExpectedAttentionPress, KeyRerotationPress

from transformers import DynamicCache, pipeline  # type: ignore
import os
import contextlib

from transformers import AutoProcessor

from reasondb.memory_footprint.memory_report import compute_memory_footprints


MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
PRESS = {
    "expected_attention": lambda compression_ratio: ExpectedAttentionPress(
        compression_ratio=compression_ratio
    ),
}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)


class KvAudioQaModelWrapper(KVCachingBackendBase):
    def __init__(
        self,
        model_name,
        device_id,
        compression_rations=(0.0, 0.9),
        batch_sizes=None,
        press_name="expected_attention",
    ):
        self.device_id = device_id
        self.compression_ratios = compression_rations
        batch_sizes = batch_sizes or (None,) * len(compression_rations)
        self.compression_ratio_to_batch_size = {
            cr: bs for cr, bs in zip(compression_rations, batch_sizes)
        }
        self.model_name = model_name
        self.press_name = press_name
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Will be initialized in init()
        self.init()

    def compute_audio_qa_response(
        self,
        column_name: str,
        audio_paths: List[str],
        question: str,
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
    ):
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
        responses, log_odds = asyncio.run(
            self._run_kv_cache_multimodal(
                column_name=column_name,
                audio_paths=audio_paths,
                question=question,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
            )
        )
        return {"answers": responses, "log_odds": log_odds}

    def hash_path(self, path: str) -> str:
        """Generate a sha256hash for a given path."""
        import hashlib

        return hashlib.sha256(path.encode()).hexdigest()

    async def prepare_caches(
        self,
        column_name: str,
        audio_paths: List[str],
        cache_dir: str,
        compression_ratio: float,
    ):
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"

        # Check if caches exist and generate if not (current method: checked one by one)
        # Store mapping of row indices to cache files
        save_dir = f"{cache_dir}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        errors = {}
        cache_filenames = []

        for i, audio_path in tqdm(
            enumerate(audio_paths),
            total=len(audio_paths),
            desc=f"Preparing caches for CR {compression_ratio}",
        ):
            hash_name = self.hash_path(audio_path)
            cache_filename = f"{save_dir}/cache_entry_{hash_name}.pt"
            cache_filenames.append(cache_filename.split("/")[-1])

            if os.path.exists(cache_filename):
                continue

            try:
                audio = self.convert_audio(audio_path)
                await self._generate_cache_for_audio(
                    audio, cache_filename, compression_ratio
                )
            except Exception as e:
                logger.warning(
                    f"Error processing audio {i} {audio_path} with hash {hash_name}: {str(e)}",
                )
                errors[cache_filename] = str(e)
        compute_memory_footprints(cache_dir, column_name, cache_filenames, model_name=self.model_name)

        with open(
            f"{cache_dir}/comp{self.to_compression_tag(compression_ratio)}/ERRORS.json",
            "w",
        ) as f:
            json.dump(errors, f, indent=4)

    def to_compression_tag(self, compression_ratio: float) -> str:
        """Convert compression ratio to a string tag for directory naming."""
        return (
            str(compression_ratio).replace(".", "_")
            if compression_ratio != 0.0
            else "0"
        )

    def init(self):
        """Initialize the QwenAudio model pipeline and compression settings."""
        logger.info("Setting up KV Cache Filter with QwenAudio model...")

        # Set up device
        self.device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"

        # Initialize the pipeline
        args = [{"attn_implementation": "flash_attention_2"}, {}]
        self.pipe = None
        for x in args:
            try:
                self.pipe = pipeline(
                    "kv-press-text-generation",  # type: ignore
                    model=self.model_name,
                    device=self.device,
                    torch_dtype=torch.bfloat16,
                    model_kwargs=x,  # type: ignore
                )
                break
            except Exception as e:
                logger.warning(
                    f"Error initializing model with args {x}: {str(e)}",
                )
        assert self.pipe is not None, "Failed to initialize the model pipeline."
        self.pipe.model.eval()

        # QwenAudio architecture is incompatible with flash attention, need to set a pipeline with sdpa for cache generation
        args_cache_generation = [{"attn_implementation": "sdpa"}, {}]
        self.pipe_cache_generation = None
        for x in args_cache_generation:
            try:
                self.pipe_cache_generation = pipeline(
                    "kv-press-text-generation",  # type: ignore
                    model=self.model_name,
                    device=self.device,
                    torch_dtype=torch.bfloat16,
                    model_kwargs=x,  # type: ignore
                )
                break
            except Exception as e:
                logger.warning(
                    f"Error initializing model with args {x}: {str(e)}",
                )
        assert (
            self.pipe_cache_generation is not None
        ), "Failed to initialize the model pipeline for cache generation."
        self.pipe_cache_generation.model.eval()

        # Set up compression press
        if self.press_name not in PRESS:
            raise ValueError(
                f"Unknown press_name '{self.press_name}'. Available options: {list(PRESS.keys())}"
            )
        self.presses = {
            cr: KeyRerotationPress(PRESS[self.press_name](compression_ratio=cr))
            for cr in self.compression_ratios
        }

        logger.info(
            f"Using press {self.press_name} with compression_ratios={self.compression_ratios}",
        )
        logger.info(
            f"Model {self.model_name} loaded on {next(self.pipe.model.parameters()).device}",
        )

        # Patch the model's forward method to support cache class and remove cache_position
        self.pipe.model._supports_cache_class = True

        # Store the original unbound forward method to avoid recursive patching
        import types

        original_forward_func = (
            self.pipe.model.forward.__func__
            if hasattr(self.pipe.model.forward, "__func__")
            else self.pipe.model.forward
        )

        def patched_forward(model_self, *args, **kwargs):
            kwargs.pop("cache_position", None)
            # Call the original function properly
            if isinstance(original_forward_func, types.FunctionType):
                return original_forward_func(model_self, *args, **kwargs)
            else:
                return original_forward_func(*args, **kwargs)

        self.pipe.model.forward = types.MethodType(patched_forward, self.pipe.model)

    # Extract audio from file path
    def convert_audio(self, audio_path):
        audio, _ = librosa.load(
            audio_path, sr=self.processor.feature_extractor.sampling_rate
        )
        return audio

    async def _run_kv_cache_multimodal(
        self,
        column_name: str,
        audio_paths: List[str],
        question: str,
        compression_ratio: float,
        cache_dir: str,
        boolean_question: bool,
    ):
        """Run KV cache-based multimodal inference on the data."""
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."

        # Check if caches exist and generate if not (current method: checked one by one)
        # Store mapping of row indices to cache files
        save_dir = f"{cache_dir}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        recorded_errors = {}
        with open(
            f"{save_dir}/ERRORS.json",
            "r",
        ) as f:
            recorded_errors = json.load(f)

        cache_files = []
        audio_paths_with_caches = []
        audio_paths_without_caches = []
        for i, audio_path in tqdm(
            enumerate(audio_paths),
            total=len(audio_paths),
            desc=f"Preparing caches for CR {compression_ratio}",
        ):
            cache_name = self.hash_path(audio_path)
            cache_filename = f"{save_dir}/cache_entry_{cache_name}.pt"

            if not os.path.exists(cache_filename):
                assert (
                    cache_filename in recorded_errors
                ), f"Cache file {cache_filename} missing but no recorded error found."
                logger.warning(
                    f"Skipping audio {i} {audio_path} due to previous error: {recorded_errors[cache_filename]}",
                )
                audio_paths_without_caches.append(audio_path)
                continue

            cache_files.append(cache_filename)
            audio_paths_with_caches.append(audio_path)

        if not cache_files:
            logger.warning("No valid caches or audio found")
            return []

        # Set up multimodal processing parameters
        # context = "Answer the following question based on the audio with 'yes' or 'no'. Do not add any other comments."
        if boolean_question:
            context = "Answer the following question based on the audio with '1' or '0'. Do not add any other comments."
        else:
            context = "Answer the following question based on the audio. Do not add any other comments."
        question = context + " " + question
        answer_prefix = "Answer: "
        batch_size = self.compression_ratio_to_batch_size[compression_ratio]
        batch_size = self._get_max_batch_size(
            column_name=column_name,
            batch_size=batch_size,
            compression_ratio=compression_ratio,
            device_id=self.device_id,
            file_paths=cache_files,
            cache_dir=cache_dir,
        )
        max_new_tokens = 4 if boolean_question else 64

        # Initialize processor
        processor = self.processor
        answers = []
        log_odds = []

        # Process images in batches
        for batch_start in tqdm(
            range(0, len(cache_files), batch_size),
            desc=f"Processing batches for CR {compression_ratio}",
        ):
            batch_input_ids = []
            batch_attention_masks = []
            caches = []
            context_lengths = []

            batch_audios = cache_files[
                batch_start : min(batch_start + batch_size, len(cache_files))
            ]
            batch_size_actual = len(batch_audios)

            # Generate or load caches for each audio in the batch
            for i in range(batch_size_actual):
                # Load the pre-generated cache
                cache = torch.load(
                    batch_audios[i], map_location=self.device, weights_only=False
                )
                caches.append(cache)
                context_lengths.append(cache.get_seq_length())

            if not caches:
                continue

            max_context_len = max(context_lengths)

            # Prepare inputs for each audio in the batch
            for i, ctx_len in enumerate(context_lengths):
                padded_context_ids = torch.full(
                    (1, ctx_len),
                    self.pipe.tokenizer.pad_token_id + 1,
                    device=self.device,
                )
                pad_len = max_context_len - ctx_len
                padding_ids = torch.full(
                    (1, pad_len), self.pipe.tokenizer.pad_token_id, device=self.device
                )
                padded_context = torch.cat([padding_ids, padded_context_ids], dim=1)

                # Create the prompt structure
                separator = "\n" + "#" * ctx_len if ctx_len > 0 else "\n" + "#"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio"},
                            {"type": "text", "text": context + separator},
                        ],
                    }
                ]

                # Apply chat template
                prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                _, question_suffix = prompt.split(separator)

                question_text = question + question_suffix + answer_prefix
                question_ids = self.pipe.tokenizer.encode(
                    question_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)

                input_ids = torch.cat([padded_context, question_ids], dim=1)

                # Create attention masks
                context_mask = torch.ones_like(padded_context_ids)
                padding_mask = torch.zeros_like(padding_ids)
                question_mask = torch.ones_like(question_ids)
                attention_mask = torch.cat(
                    [padding_mask, context_mask, question_mask], dim=1
                )

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)

            # Batch the inputs
            batched_inputs = torch.cat(batch_input_ids, dim=0)
            batched_attention_mask = torch.cat(batch_attention_masks, dim=0)

            # Batch the caches
            batched_cache = []
            for layers in zip(*caches):
                max_seq_len = max(k.shape[2] for k, _ in layers)
                keys_padded = []
                values_padded = []

                for k, v in layers:
                    seq_len = k.shape[2]
                    pad_len = max_seq_len - seq_len
                    k_padded = (
                        torch.nn.functional.pad(k, (0, 0, pad_len, 0))
                        if pad_len > 0
                        else k
                    )
                    v_padded = (
                        torch.nn.functional.pad(v, (0, 0, pad_len, 0))
                        if pad_len > 0
                        else v
                    )
                    k_padded = k_padded.contiguous()
                    v_padded = v_padded.contiguous()
                    keys_padded.append(k_padded)
                    values_padded.append(v_padded)

                keys_cat = torch.cat(keys_padded, dim=0)
                values_cat = torch.cat(values_padded, dim=0)
                batched_cache.append((keys_cat, values_cat))

            padded_cache = DynamicCache()
            for layer_idx, (keys, values) in enumerate(batched_cache):
                padded_cache.update(keys, values, layer_idx)

            # Move everything to device
            batched_inputs = batched_inputs.to(self.device)
            batched_attention_mask = batched_attention_mask.to(self.device)
            padded_cache.key_cache = [k.to(self.device) for k in padded_cache.key_cache]
            padded_cache.value_cache = [
                v.to(self.device) for v in padded_cache.value_cache
            ]

            # Generate responses
            with torch.no_grad():
                generated = self.pipe.model.generate(
                    input_ids=batched_inputs,
                    attention_mask=batched_attention_mask,
                    past_key_values=padded_cache,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            logits = generated.scores[0]  # type: ignore
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            id0 = self.pipe.tokenizer.convert_tokens_to_ids("0")
            id1 = self.pipe.tokenizer.convert_tokens_to_ids("1")
            log_probs_0 = log_probs[:, id0]
            log_probs_1 = log_probs[:, id1]
            log_odds_1_vs_0 = log_probs_1 - log_probs_0

            # Decode the generated tokens
            decoded = self.pipe.tokenizer.batch_decode(
                generated.sequences[:, batched_inputs.shape[1] :],
                skip_special_tokens=True,
            )

            answers.extend(decoded)
            log_odds.extend(log_odds_1_vs_0.cpu().tolist())

            # Cleanup batch resources
            del caches, batched_inputs, batched_attention_mask, padded_cache
            torch.cuda.empty_cache()

        # Process answers and create mask
        # mask = []
        # transformed_keep_answer = "1" if keep_answer == "yes" else "0"
        # for data_id, answer in zip(data_ids, answers):
        #     predicted_answer = transformed_keep_answer in answer.lower()
        #     # logger.info(f"Row {data_id} answer: {answer} -> {predicted_answer}")
        #     mask.append((data_id, predicted_answer))

        result_answers = {
            aud_path: "Not sure" for aud_path in audio_paths_without_caches
        }
        result_answers.update(
            {
                aud_path: answer
                for aud_path, answer in zip(audio_paths_with_caches, answers)
            }
        )

        result_log_odds = {img_path: 0.0 for img_path in audio_paths_without_caches}
        result_log_odds.update(
            {
                img_path: log_odd
                for img_path, log_odd in zip(audio_paths_with_caches, log_odds)
            }
        )

        return result_answers, result_log_odds

    async def _generate_cache_for_audio(self, audio, cache_filename, compression_ratio):
        """Generate and save multimodal KV cache for a single audio."""
        assert (
            self.pipe_cache_generation is not None
        ), "Model pipeline for cache generation is not initialized."

        try:
            # Standard context and answer format
            context = " "
            answer_prefix = "Answer: "

            # Generate cache
            inputs = self.pipe_cache_generation.preprocess(
                context=context,
                questions=[""],
                answer_prefix=answer_prefix,
                max_context_length=128000,
                audio=audio,
            )

            cache = DynamicCache()
            press = self.presses[compression_ratio]
            with torch.inference_mode():
                with (
                    press(self.pipe_cache_generation.model)
                    if press is not None
                    else contextlib.nullcontext()
                ):
                    _ = self.pipe_cache_generation._forward(
                        inputs, press=press, cache=cache
                    )

            # Save cache to disk
            cache.key_cache = [k.detach().cpu() for k in cache.key_cache]
            cache.value_cache = [v.detach().cpu() for v in cache.value_cache]

            logger.info(f"Saving cache to: {cache_filename}")

            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
            torch.save(cache, cache_filename)

            if not os.path.exists(cache_filename):
                logger.error(f"Cache file was not actually created: {cache_filename}")

            # Cleanup
            del cache
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error generating cache for item {cache_filename}: {str(e)}")
            raise


model_wrapper = None


class Status(Resource):
    def get(self):
        assert model_wrapper is not None
        return {
            "status": "alive",
            "model_name": model_wrapper.model_name,
            "compression_ratios": list(model_wrapper.compression_ratios),
        }, 200


class PrepareCaches(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "audio_paths": [
                "/path/to/audio1.wav",
                "/path/to/audio2.wav",
            ],
            "cache_dir": "/path/to/cache/dir"
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        aud_paths = data["audio_paths"]
        cache_dir = data["cache_dir"]
        compression_ratio = data["compression_ratio"]
        assert model_wrapper is not None
        asyncio.run(
            model_wrapper.prepare_caches(
                column_name=column_name,
                audio_paths=aud_paths,
                cache_dir=cache_dir,
                compression_ratio=compression_ratio,
            )
        )
        return {"status": "cache_ready"}, 200


class AudioQA(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "audio_paths": [
                "/path/to/audio1.wav",
                "/path/to/audio2.wav",
            ],
            "question": "Which animal is making the sound?",
            "compression_ratio": 0.5,
            "boolean": true,
            "cache_dir": "/path/to/cache/dir",

        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        aud_paths = data["audio_paths"]
        question = data["question"]
        compression_ratio = data["compression_ratio"]
        boolean_question = data["boolean"]
        cache_dir = data["cache_dir"]
        assert model_wrapper is not None
        responses = model_wrapper.compute_audio_qa_response(
            column_name=column_name,
            audio_paths=aud_paths,
            question=question,
            compression_ratio=compression_ratio,
            boolean_question=boolean_question,
            cache_dir=cache_dir,
        )
        return responses, 200


api.add_resource(Status, "/status")
api.add_resource(AudioQA, "/audio_qa")
api.add_resource(PrepareCaches, "/prepare_caches")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID for GPU to use",
    )
    args = parser.parse_args()
    device_id = args.device_id
    model_wrapper = KvAudioQaModelWrapper(MODEL_NAME, device_id)
    app.run(host="127.0.0.1", port=PORT_KV_AUDIO, debug=False)
