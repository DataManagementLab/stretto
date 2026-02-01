import argparse
import json
import asyncio
import logging
from tqdm import tqdm
from flask import Flask, request
from flask_restful import Resource, Api
from typing import List


from reasondb.backends.kv_cache_base import KVCachingBackendBase
from reasondb.backends.text_qa import PORT_KV_TEXT_QA
import torch
from kvpress import ExpectedAttentionPress, KeyRerotationPress, KVzipPress

from transformers import DynamicCache, pipeline  # type: ignore
import os
import contextlib

from reasondb.memory_footprint.memory_report import compute_memory_footprints


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

PRESS = {
    "expected_attention": lambda compression_ratio: ExpectedAttentionPress(
        compression_ratio=compression_ratio
    ),
    "kvzip": lambda compression_ratio: KVzipPress(compression_ratio=compression_ratio),
}


class KvTextQaModelWrapper(KVCachingBackendBase):
    def __init__(
        self,
        model_name,
        device_id: int,
        compression_ratios=(0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9),
        # compression_ratios=(0.5, 0.6, 0.8, 0.9),
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
        self.init()

    def compute_text_qa_response(
        self,
        column_name: str,
        texts: List[str],
        questions: List[str],
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
    ):
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
        return asyncio.run(
            self._run_kv_cache_text(
                column_name=column_name,
                texts=texts,
                all_questions=questions,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
            )
        )

    def hash_text(self, text: str) -> str:
        """Generate a sha256hash for a given path."""
        import hashlib

        return hashlib.sha256(text.encode()).hexdigest()

    async def prepare_caches(
        self,
        column_name: str,
        texts: List[str],
        cache_dir: str,
        compression_ratio: float,
    ):
        # Skip cache preparation if gold_vanilla is enabled for CR=0 and 70B model
        if self.gold_vanilla and compression_ratio == 0.0 and "70B" in self.model_name:
            logger.info(
                "Skipping cache preparation for gold-vanilla mode (CR=0, 70B model)"
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

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc=f"Preparing caches {compression_ratio}",
        ):
            hash_name = self.hash_text(text)
            cache_filename = f"{save_dir}/cache_entry_{hash_name}.pt"
            cache_filenames.append(cache_filename.split("/")[-1])

            if os.path.exists(cache_filename):
                continue

            try:
                await self._generate_cache_for_text(
                    text, cache_filename, compression_ratio
                )
            except Exception as e:
                logger.warning(
                    f"Error processing text {i} with hash {hash_name}: {str(e)}",
                )
                errors[cache_filename] = str(e)

        compute_memory_footprints(cache_dir, column_name, cache_filenames, model_name=self.model_name)
        with open(
            f"{save_dir}/ERRORS.json",
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
        """Initialize the text model pipeline and compression settings."""
        logger.info("Setting up KV Cache Text Filter...")

        # Set up device
        self.device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"

        # Initialize the pipeline
        # args = [{"attn_implementation": "sdpa"}, {}]
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
                    f"Error initializing model with args {x}: {str(type(e))}({str(e)})",
                    exc_info=True,
                )

        assert self.pipe is not None, "Failed to initialize the model pipeline."
        self.pipe.model.eval()
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer.pad_token = self.pipe.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Set up compression press
        if self.press_name not in PRESS:
            raise ValueError(
                f"Unknown press_name '{self.press_name}'. Available options: {list(PRESS.keys())}"
            )
        self.presses = {
            cr: KeyRerotationPress(PRESS[self.press_name](compression_ratio=cr))
            if cr > 0.0
            else None
            for cr in self.compression_ratios
        }

        logger.info(
            f"Using press {self.press_name} with compression_ratios {self.compression_ratios}",
        )
        logger.info(
            f"Model {self.model_name} loaded on {next(self.pipe.model.parameters()).device}",
        )

    async def _run_vanilla_inference(
        self,
        texts: List[str],
        all_questions: List[str],
        boolean_question: bool,
    ):
        """Run vanilla inference without KV caching."""
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."

        answer_prefix = "Answer: "
        max_new_tokens = 4 if boolean_question else 64

        answers = []

        # Process texts one by one (could be batched for efficiency)
        for text, question in tqdm(
            zip(texts, all_questions),
            total=len(texts),
            desc="Processing texts (vanilla)",
        ):
            try:
                # Determine question suffix based on chat template
                if self.tokenizer.chat_template is None:
                    question_suffix = "\n"
                else:
                    template_context = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": "Example of context\n###"}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    _, question_suffix = template_context.split("\n###")

                # Create full prompt
                full_prompt = text + "\n" + question + question_suffix + answer_prefix

                # Tokenize input
                inputs = self.tokenizer(full_prompt, return_tensors="pt")

                # Move inputs to device
                first_device = next(self.pipe.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

                # Generate response
                with torch.no_grad():
                    generated = self.pipe.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode the generated tokens
                decoded = self.tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )

                answers.append(decoded)

                # Cleanup
                del inputs, generated
                torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(
                    f"Error processing text with question '{question}': {str(e)}"
                )
                answers.append("Not sure")

        return {"answers": answers}

    async def _run_kv_cache_text(
        self,
        column_name: str,
        texts: List[str],
        all_questions: List[str],
        compression_ratio: float,
        cache_dir: str,
        boolean_question: bool,
    ):
        """Run KV cache-based text inference on the data."""

        # Use vanilla inference if gold_vanilla is enabled for CR=0 and 70B model
        if self.gold_vanilla and compression_ratio == 0.0 and "70B" in self.model_name:
            logger.info("Using gold-vanilla mode for inference (CR=0, 70B model)")
            return await self._run_vanilla_inference(
                texts=texts,
                all_questions=all_questions,
                boolean_question=boolean_question,
            )

        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."

        # Check if caches exist and generate if not
        save_dir = f"{cache_dir}/{self.model_name}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        cache_files = []
        hashed_texts = []

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc=f"Preparing caches for cr {compression_ratio}",
        ):
            hashed_text = self.hash_text(text)
            cache_filename = f"{save_dir}/cache_entry_{hashed_text}.pt"

            assert os.path.exists(
                cache_filename
            ), f"Cache file does not exist for text at index {i}: {cache_filename}"
            cache_files.append(cache_filename)
            hashed_texts.append(hashed_text)

        if not cache_files:
            logger.warning("No valid caches or texts found")
            return []

        # Set up text processing parameters
        if boolean_question:
            context = "Answer the following question based on the context with '1' or '0'. Do not add any other comments."
        else:
            context = "Answer the following question based on the context. Do not add any other comments."
        all_questions = [context + " " + q for q in all_questions]
        answer_prefix = "Answer: "
        batch_size = self.compression_ratio_to_batch_size[compression_ratio]
        batch_size = self._get_max_batch_size(
            column_name=column_name,
            batch_size=batch_size,
            compression_ratio=compression_ratio,
            device_id=self.device_id,
            cache_dir=cache_dir,
            file_paths=cache_files,
        )
        max_new_tokens = 4 if boolean_question else 64

        answers = []
        log_odds = []

        # Process texts in batches
        for batch_start in tqdm(
            range(0, len(cache_files), batch_size),
            desc=f"Processing batches for CR {compression_ratio}",
        ):
            batch_input_ids = []
            batch_attention_masks = []
            caches = []
            context_lengths = []
            questions = []

            batch_cache_files = cache_files[
                batch_start : min(batch_start + batch_size, len(cache_files))
            ]
            batch_questions = all_questions[
                batch_start : min(batch_start + batch_size, len(cache_files))
            ]
            batch_size_actual = len(batch_cache_files)

            # Generate or load caches for each text in the batch
            for i in range(batch_size_actual):
                # Load the pre-generated cache to CPU first to avoid GPU memory accumulation
                cache = torch.load(
                    batch_cache_files[i], map_location="cpu", weights_only=False
                )
                caches.append(cache)
                context_lengths.append(cache.get_seq_length())
                questions.append(batch_questions[i])

            if not caches:
                continue

            max_context_len = max(context_lengths)

            question_ids_list = []
            context_ids_list = []
            padded_context_ids_mask_list = []

            for i, (ctx_len, q) in enumerate(zip(context_lengths, questions)):
                # Pad context to align with KV cache
                padded_context_ids = torch.full(
                    (1, ctx_len),
                    self.tokenizer.pad_token_id + 1,
                    device=self.device,
                )
                pad_len = max_context_len - ctx_len
                padding_ids = torch.full(
                    (1, pad_len), self.tokenizer.pad_token_id, device=self.device
                )
                padded_context = torch.cat([padding_ids, padded_context_ids], dim=1)
                padding_mask = torch.zeros_like(padding_ids)
                padded_context_mask = torch.ones_like(padded_context_ids)
                padded_context_ids_mask = torch.cat(
                    [padding_mask, padded_context_mask], dim=1
                )

                # Determine question suffix
                if self.tokenizer.chat_template is None:
                    question_suffix = "\n"
                else:
                    separator = "\n" + "#" * ctx_len if ctx_len > 0 else "\n#"
                    template_context = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": "Example of context" + separator}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    _, question_suffix = template_context.split(separator)

                # Tokenize question (variable-length)
                complete_question = q + question_suffix + answer_prefix
                question_ids = self.tokenizer.encode(
                    complete_question,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)

                # Store for later padding
                context_ids_list.append(padded_context)
                question_ids_list.append(question_ids)
                padded_context_ids_mask_list.append(padded_context_ids_mask)

            # Compute max question length across batch
            max_question_len = max(q.shape[1] for q in question_ids_list)

            # Pad questions and build batch tensors
            batch_input_ids = []
            batch_attention_masks = []

            for padded_context, question_ids, padded_context_ids_mask in zip(
                context_ids_list, question_ids_list, padded_context_ids_mask_list
            ):
                q_len = question_ids.shape[1]
                q_pad_len = max_question_len - q_len

                # Pad question to match longest in batch
                if q_pad_len > 0:
                    q_padding = torch.full(
                        (1, q_pad_len),
                        self.tokenizer.pad_token_id,
                        device=self.device,
                    )
                    padded_question = torch.cat([q_padding, question_ids], dim=1)
                    padded_question_mask = torch.cat(
                        [torch.zeros_like(q_padding), torch.ones_like(question_ids)],
                        dim=1,
                    )
                else:
                    padded_question = question_ids
                    padded_question_mask = torch.ones_like(question_ids)

                # Concatenate full input
                input_ids = torch.cat([padded_context, padded_question], dim=1)

                # Build attention mask: padding + context + padding + question
                attention_mask = torch.cat(
                    [padded_context_ids_mask, padded_question_mask], dim=1
                )

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)

            # Stack batched tensors
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
                layer_device = self.pipe.model.model.layers[
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
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            logits = generated.scores[0]  # type: ignore
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            id0 = self.tokenizer.convert_tokens_to_ids("0")
            id1 = self.tokenizer.convert_tokens_to_ids("1")
            log_probs_0 = log_probs[:, id0]
            log_probs_1 = log_probs[:, id1]
            log_odds_1_vs_0 = log_probs_1 - log_probs_0

            # Decode the generated tokens
            decoded = self.tokenizer.batch_decode(
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
            del (
                batch_input_ids,
                batch_attention_masks,
                context_ids_list,
                question_ids_list,
                padded_context_ids_mask_list,
                batched_cache,
            )
            torch.cuda.empty_cache()

        # Process answers and create mask
        # mask = []
        # for data_id, answer in zip(data_ids, answers):
        #     predicted_answer = keep_answer.strip().lower() in answer.lower()
        #     # logger.info(f"Row {data_id} answer: {answer} -> {predicted_answer}")
        #     mask.append((data_id, predicted_answer))

        return {"answers": answers, "log_odds": log_odds}

    async def _generate_cache_for_text(
        self, text_content, cache_filename, compression_ratio
    ):
        """Generate and save text KV cache for a single text."""
        assert (
            self.pipe is not None
        ), "Model pipeline for cache generation is not initialized."

        try:
            context = text_content[
                : min(128000, len(text_content))
            ]  # Limit context length
            answer_prefix = "Answer: "

            # Generate cache using preprocess method
            inputs = self.pipe.preprocess(
                context=context,
                questions=[""],
                answer_prefix=answer_prefix,
                max_context_length=128000,
            )

            context_ids = inputs["context_ids"]
            cache = DynamicCache()

            press = self.presses[compression_ratio]
            with torch.inference_mode():
                with (
                    press(self.pipe.model)
                    if press is not None
                    else contextlib.nullcontext()
                ):
                    # Run the model without the lm head for pre-filling
                    self.pipe.model.model(
                        input_ids=context_ids,
                        past_key_values=cache,
                        use_cache=True,
                        output_attentions=self.pipe.output_attentions(press),
                    )

            # Move each cache layer to the same device as the corresponding model layer
            # This is necessary when using device_map="auto" which can distribute layers across GPUs
            # for layer_idx in range(len(cache.key_cache)):
            #    layer_device = self.pipe.model.model.layers[layer_idx].self_attn.q_proj.weight.device
            #    cache.key_cache[layer_idx] = cache.key_cache[layer_idx].to(layer_device)
            #    cache.value_cache[layer_idx] = cache.value_cache[layer_idx].to(layer_device)

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
            logger.error(f"Error generating cache for text {cache_filename}: {str(e)}")
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
            "texts": [
                "text1",
                "text2",
            ],
            "cache_dir": "/path/to/cache/dir"
            "compression_ratio": 0.5
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        texts = data["texts"]
        cache_dir = data["cache_dir"]
        compresion_ratio = data["compression_ratio"]
        assert model_wrapper is not None
        asyncio.run(
            model_wrapper.prepare_caches(
                column_name=column_name,
                texts=texts,
                cache_dir=cache_dir,
                compression_ratio=compresion_ratio,
            )
        )
        return {"status": "cache_ready"}, 200


class TextQA(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "texts": [
                "text 1",
                "text 2",
            ],
            "questions": [
                "How many cats are there?"
                "How many dogs are there?"
            ],
            "compression_ratio": 0.5,
            "boolean": true,
            "cache_dir": "/path/to/cache/dir"
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        texts = data["texts"]
        questions = data["questions"]
        compression_ratio = data["compression_ratio"]
        boolean_question = data["boolean"]
        cache_dir = data["cache_dir"]
        assert model_wrapper is not None
        assert (
            len(texts) == len(questions)
        ), f"Number of texts {len(texts)} must match number of questions {len(questions)}"
        responses = model_wrapper.compute_text_qa_response(
            column_name=column_name,
            texts=texts,
            questions=questions,
            compression_ratio=compression_ratio,
            boolean_question=boolean_question,
            cache_dir=cache_dir,
        )
        return responses, 200


api.add_resource(Status, "/status")
api.add_resource(TextQA, "/text_qa")
api.add_resource(PrepareCaches, "/prepare_caches")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID for GPU to use",
    )
    parser.add_argument(
        "--gold-vanilla",
        action="store_true",
        help="Use vanilla inference (no KV caching) for CR=0 with 70B models",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        choices=sorted(PORT_KV_TEXT_QA.keys()),
        default=MODEL_NAME,
        help="Name of the model to use",
    )
    args = parser.parse_args()
    device_id = args.device_id
    gold_vanilla = args.gold_vanilla
    model_name = args.model_name
    model_wrapper = KvTextQaModelWrapper(
        model_name,
        device_id,
        batch_sizes=[None, None, None, None, None, None, None],
        compression_ratios=[
            0.0,
            0.5,
            0.8,
            0.9,
            0.6,
            0.4,
            0.3,
        ], 
        gold_vanilla=gold_vanilla,
    )
    app.run(host="127.0.0.1", port=PORT_KV_TEXT_QA.get(model_name), debug=False)
