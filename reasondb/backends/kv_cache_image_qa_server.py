import argparse
import json
import asyncio
import logging
from tqdm import tqdm
from flask import Flask, request
from flask_restful import Resource, Api
from typing import Dict, List, Tuple

from reasondb.backends.kv_cache_base import KVCachingBackendBase
from reasondb.backends.vision_model import PORT_KV_VISION
from PIL import Image, ImageFile
import torch
from kvpress import ExpectedAttentionPress, KeyRerotationPress

from transformers import DynamicCache, pipeline  # type: ignore
import os
import contextlib

import base64
from io import BytesIO
from transformers import AutoProcessor

from reasondb.memory_footprint.memory_report import compute_memory_footprints
import numpy as np

MODEL_NAME = "llava-hf/llava-next-72b-hf"
IMAGE_MAX_PIXELS = 1400 * 1400
MODEL_TAG = {
    "llava-hf/llava-next-72b-hf": "llava-70B",
    "llava-hf/llama3-llava-next-8b-hf": "llava-8B",
}
PRESS = {
    "expected_attention": lambda compression_ratio: ExpectedAttentionPress(
        compression_ratio=compression_ratio
    ),
}
# Enable loading of truncated images
# Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)


class KvImageQaModelWrapper(KVCachingBackendBase):
    def __init__(
        self,
        model_name,
        device_id,
        compression_ratios=(0.0, 0.5, 0.8, 0.9, 0.99),
        batch_sizes=None,
        press_name="expected_attention",
        gold_vanilla=False,
    ):
        self.device_id = device_id
        self.compression_ratios = compression_ratios
        batch_sizes = batch_sizes or (None,) * len(compression_ratios)
        self.compression_ratio_to_batch_size = {
            cr: bs for cr, bs in zip(compression_ratios, batch_sizes)
        }
        self.model_name = model_name
        self.press_name = press_name
        self.gold_vanilla = gold_vanilla

        self.model_tag = MODEL_TAG.get(model_name, "unknown_model")

        # Will be initialized in init()
        self.init()

    def compute_image_qa_response(
        self,
        column_name: str,
        image_paths: List[str],
        question: str,
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
    ) -> Dict[str, Dict]:
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
        responses, log_odds = asyncio.run(
            self._run_kv_cache_multimodal(
                column_name=column_name,
                image_paths=image_paths,
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
        image_paths: List[str],
        cache_dir: str,
        compression_ratio: float,
    ):
        # Skip cache preparation if gold_vanilla is enabled for CR=0 and 72b model
        if self.gold_vanilla and compression_ratio == 0.0 and "72b" in self.model_name:
            logger.info(
                "Skipping cache preparation for gold-vanilla mode (CR=0, 72b model)"
            )
            return

        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"

        # Check if caches exist and generate if not (current method: checked one by one)
        # Store mapping of row indices to cache files
        save_dir = f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        errors = {}
        cache_filenames = []

        for i, image_path in tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc=f"Preparing caches for CR {compression_ratio}",
        ):
            hash_name = self.hash_path(image_path)
            cache_filename = f"{save_dir}/cache_entry_{hash_name}.pt"
            cache_filenames.append(cache_filename.split("/")[-1])

            if os.path.exists(cache_filename):
                continue

            try:
                image = self.deserialize_image(image_path)
                await self._generate_cache_for_image(
                    image, cache_filename, compression_ratio
                )
            except Exception as e:
                logger.warning(
                    f"Error processing image {i} {image_path} with hash {hash_name}: {str(e)}",
                )
                errors[cache_filename] = str(e)
        compute_memory_footprints(cache_dir, column_name, cache_filenames, model_name=self.model_name)

        with open(
            f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}/ERRORS.json",
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
        """Initialize the llava-next model pipeline and compression settings."""
        logger.info("Setting up KV Cache Filter with llava-next model...")

        # Set up device
        self.device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"

        # Set PIL image size limit
        Image.MAX_IMAGE_PIXELS = 250000000

        # Initialize the pipeline
        args = [{"attn_implementation": "flash_attention_2"}, {}]
        self.pipe = None
        for x in args:
            try:
                self.pipe = pipeline(
                    "kv-press-text-generation",  # type: ignore
                    model=self.model_name,
                    # device=self.device,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    model_kwargs=x,  # type: ignore
                )
                break
            except Exception as e:
                logger.warning(
                    f"Error initializing model with args {x}: {str(e)}", exc_info=True
                )
        assert self.pipe is not None, "Failed to initialize the model pipeline."
        self.pipe.model.eval()

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

    # Convert DataFrame images to PIL Images
    @staticmethod
    def deserialize_image(b64_or_path):
        """Loads an image and downsizes it if it's too large."""
        try:
            if isinstance(b64_or_path, str) and b64_or_path.startswith("data:image"):
                # Handle base64 encoded images
                image_data = base64.b64decode(b64_or_path.split(",")[1])
                image = Image.open(BytesIO(image_data))
                image.load()  # Force load to catch truncation errors early
                image = image.convert("RGB")
            elif isinstance(b64_or_path, str):
                # Handle file paths
                image = Image.open(b64_or_path)
                image.load()  # Force load to catch truncation errors early
                image = image.convert("RGB")
            else:
                # Assume it's already a PIL Image or bytes
                if hasattr(b64_or_path, "convert"):
                    image = b64_or_path.convert("RGB")
                else:
                    image = Image.open(BytesIO(b64_or_path))
                    image.load()  # Force load to catch truncation errors early
                    image = image.convert("RGB")

            # Downsize image if too large
            if int(np.prod(image.size)) > IMAGE_MAX_PIXELS:
                ratio = np.sqrt(IMAGE_MAX_PIXELS / np.prod(image.size))
                image.thumbnail(
                    (int(image.size[0] * ratio), int(image.size[1] * ratio))
                )

            return image
        except Exception as e:
            logger.error(f"Error deserializing image {b64_or_path}: {str(e)}")
            raise

    async def _run_vanilla_inference(
        self,
        image_paths: List[str],
        question: str,
        boolean_question: bool,
    ):
        """Run vanilla inference without KV caching."""
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."

        # Set up multimodal processing parameters
        if boolean_question:
            context = "Answer the following question based on the image with '1' or '0'. Do not add any other comments."
        else:
            context = "Answer the following question based on the image. Do not add any other comments."

        full_question = context + " " + question
        max_new_tokens = 4 if boolean_question else 64

        # Initialize processor
        processor = AutoProcessor.from_pretrained(self.model_name)
        answers = []
        log_odds = []

        # Process images one by one (could be batched for efficiency)
        for image_path in tqdm(image_paths, desc="Processing images (vanilla)"):
            try:
                # Load and deserialize image
                image = self.deserialize_image(image_path)

                # Create prompt structure
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": full_question},
                        ],
                    }
                ]

                # Apply chat template and prepare inputs
                prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )

                inputs = processor(images=image, text=prompt, return_tensors="pt")

                # Move inputs to device
                first_device = next(self.pipe.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

                # Generate response
                with torch.no_grad():
                    generated = self.pipe.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.pipe.tokenizer.eos_token_id,
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
                    generated.sequences[:, inputs["input_ids"].shape[1] :],  # type: ignore
                    skip_special_tokens=True,
                )

                answers.extend(decoded)
                log_odds.extend(log_odds_1_vs_0.cpu().tolist())
                # Decode the generated tokens

                # Cleanup
                del inputs, generated
                torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Error processing image {image_path}: {str(e)}")
                answers.append("Not sure")

        # Create result dictionary
        result_answers = {
            img_path: answer for img_path, answer in zip(image_paths, answers)
        }
        result_log_odds = {
            img_path: log_odd for img_path, log_odd in zip(image_paths, log_odds)
        }

        return result_answers, result_log_odds

    async def _run_kv_cache_multimodal(
        self,
        column_name: str,
        image_paths: List[str],
        question: str,
        compression_ratio: float,
        cache_dir: str,
        boolean_question: bool,
    ) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Run KV cache-based multimodal inference on the data."""

        # Use vanilla inference if gold_vanilla is enabled for CR=0 and 72b model
        if self.gold_vanilla and compression_ratio == 0.0 and "72b" in self.model_name:
            logger.info("Using gold-vanilla mode for inference (CR=0, 72b model)")
            return await self._run_vanilla_inference(
                image_paths=image_paths,
                question=question,
                boolean_question=boolean_question,
            )

        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."

        # Check if caches exist and generate if not (current method: checked one by one)
        # Store mapping of row indices to cache files
        save_dir = f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        recorded_errors = {}
        with open(
            f"{save_dir}/ERRORS.json",
            "r",
        ) as f:
            recorded_errors = json.load(f)

        cache_files = []
        image_paths_with_caches = []
        image_paths_without_caches = []
        for i, image_path in tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc=f"Preparing caches for CR {compression_ratio}",
        ):
            cache_name = self.hash_path(image_path)
            cache_filename = f"{save_dir}/cache_entry_{cache_name}.pt"

            if not os.path.exists(cache_filename):
                assert (
                    cache_filename in recorded_errors
                ), f"Cache file {cache_filename} missing but no recorded error found."
                logger.warning(
                    f"Skipping image {i} {image_path} due to previous error: {recorded_errors[cache_filename]}",
                )
                image_paths_without_caches.append(image_path)
                continue

            cache_files.append(cache_filename)
            image_paths_with_caches.append(image_path)

        if not cache_files:
            logger.warning("No valid caches or images found")
            return {}, {}

        # Set up multimodal processing parameters
        # context = "Answer the following question based on the image with 'yes' or 'no'. Do not add any other comments."
        if boolean_question:
            context = "Answer the following question based on the image with '1' or '0'. Do not add any other comments."
        else:
            context = "Answer the following question based on the image. Do not add any other comments."
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
        processor = AutoProcessor.from_pretrained(self.model_name)
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

            batch_images = cache_files[
                batch_start : min(batch_start + batch_size, len(cache_files))
            ]
            batch_size_actual = len(batch_images)

            # Generate or load caches for each image in the batch
            for i in range(batch_size_actual):
                # Load the pre-generated cache to CPU first to avoid GPU memory accumulation
                cache = torch.load(
                    batch_images[i], map_location="cpu", weights_only=False
                )
                caches.append(cache)
                context_lengths.append(cache.get_seq_length())

            if not caches:
                continue

            max_context_len = max(context_lengths)

            # Prepare inputs for each image in the batch
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
                            {"type": "image"},
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

            # Move inputs to the device of the embedding layer (first layer)
            first_device = next(self.pipe.model.parameters()).device
            batched_inputs = batched_inputs.to(first_device)
            batched_attention_mask = batched_attention_mask.to(first_device)

            # Move each cache layer to the same device as the corresponding model layer
            for layer_idx in range(len(padded_cache.key_cache)):
                layer_device = self.pipe.model.language_model.layers[
                    layer_idx
                ].self_attn.q_proj.weight.device
                padded_cache.key_cache[layer_idx] = padded_cache.key_cache[
                    layer_idx
                ].to(layer_device)
                padded_cache.value_cache[layer_idx] = padded_cache.value_cache[
                    layer_idx
                ].to(layer_device)

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
                generated.sequences[:, batched_inputs.shape[1] :],  # type: ignore
                skip_special_tokens=True,
            )

            answers.extend(decoded)
            log_odds.extend(log_odds_1_vs_0.cpu().tolist())

            # Cleanup batch resources
            del (
                caches,
                batched_inputs,
                batched_attention_mask,
                padded_cache,
                generated,
                logits,
                log_probs,
            )
            del batch_input_ids, batch_attention_masks, batched_cache
            torch.cuda.empty_cache()

        # Process answers and create mask
        # mask = []
        # transformed_keep_answer = "1" if keep_answer == "yes" else "0"
        # for data_id, answer in zip(data_ids, answers):
        #     predicted_answer = transformed_keep_answer in answer.lower()
        #     # logger.info(f"Row {data_id} answer: {answer} -> {predicted_answer}")
        #     mask.append((data_id, predicted_answer))

        result_answers = {
            img_path: "Not sure" for img_path in image_paths_without_caches
        }
        result_answers.update(
            {
                img_path: answer
                for img_path, answer in zip(image_paths_with_caches, answers)
            }
        )
        result_log_odds = {img_path: 0.0 for img_path in image_paths_without_caches}
        result_log_odds.update(
            {
                img_path: log_odd
                for img_path, log_odd in zip(image_paths_with_caches, log_odds)
            }
        )

        return result_answers, result_log_odds

    async def _generate_cache_for_image(self, image, cache_filename, compression_ratio):
        """Generate and save multimodal KV cache for a single image."""
        assert self.pipe is not None, "Model pipeline is not initialized."

        try:
            # Standard context and answer format
            context = " "
            answer_prefix = "Answer: "

            # Generate cache
            inputs = self.pipe.preprocess(
                context=context,
                questions=[""],
                answer_prefix=answer_prefix,
                max_context_length=128000,
                image=image,
            )

            cache = DynamicCache()
            press = self.presses[compression_ratio]
            with torch.inference_mode():
                with (
                    press(self.pipe.model)
                    if press is not None
                    else contextlib.nullcontext()
                ):
                    _ = self.pipe._forward(inputs, press=press, cache=cache)

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
            "image_paths": [
                "/path/to/image1.jpg",
                "/path/to/image2.jpg",
            ],
            "cache_dir": "/path/to/cache/dir"
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        img_paths = data["image_paths"]
        cache_dir = data["cache_dir"]
        compression_ratio = data["compression_ratio"]
        assert model_wrapper is not None
        asyncio.run(
            model_wrapper.prepare_caches(
                column_name=column_name,
                image_paths=img_paths,
                cache_dir=cache_dir,
                compression_ratio=compression_ratio,
            )
        )
        return {"status": "cache_ready"}, 200


class ImageQA(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "image_paths": [
                "/path/to/image1.jpg",
                "/path/to/image2.jpg",
            ],
            "question": "How many cats are there?",
            "compression_ratio": 0.5,
            "boolean": true,
            "cache_dir": "/path/to/cache/dir",

        }
        """
        data = request.get_json(force=True)
        img_paths = data["image_paths"]
        column_name = data["column_name"]
        question = data["question"]
        compression_ratio = data["compression_ratio"]
        boolean_question = data["boolean"]
        cache_dir = data["cache_dir"]
        assert model_wrapper is not None
        responses = model_wrapper.compute_image_qa_response(
            column_name=column_name,
            image_paths=img_paths,
            question=question,
            compression_ratio=compression_ratio,
            boolean_question=boolean_question,
            cache_dir=cache_dir,
        )
        return responses, 200


api.add_resource(Status, "/status")
api.add_resource(ImageQA, "/image_qa")
api.add_resource(PrepareCaches, "/prepare_caches")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        choices=sorted(PORT_KV_VISION.keys()),
        help="Name of the model to use",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID for GPU to use",
    )
    parser.add_argument(
        "--gold-vanilla",
        action="store_true",
        help="Use vanilla inference (no KV caching) for CR=0 with 72b models",
    )
    args = parser.parse_args()
    device_id = args.device_id
    gold_vanilla = args.gold_vanilla
    model_wrapper = KvImageQaModelWrapper(
        args.model_name,
        device_id,
        compression_ratios=(0.0, 0.5, 0.8, 0.9, 0.99),
        gold_vanilla=gold_vanilla,
    )
    app.run(host="127.0.0.1", port=PORT_KV_VISION.get(args.model_name), debug=False)
