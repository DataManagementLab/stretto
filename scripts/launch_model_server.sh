#!/bin/bash
# Launch a KV cache QA server for a model from the registry. Text models start the
# text QA server, vision (VL) models start the image QA server.
# Usage: ./scripts/launch_model_server.sh <model_key> [gpu_ids]
#   gpu_ids: one GPU id or a comma-separated list (default 0). The server loads with
#            device_map="auto", so a list shards the model across all listed GPUs.
# Example: ./scripts/launch_model_server.sh llama8B 0
#          ./scripts/launch_model_server.sh mistral8B 1
#          ./scripts/launch_model_server.sh qwenVL32B 0,1,2   # shard 32B over 3 GPUs
#
# Available model keys (from reasondb/config/model_registry.py):
#   llama8B      - meta-llama/Llama-3.1-8B-Instruct            (port 5010, text)
#   llama70B     - meta-llama/Llama-3.1-70B-Instruct           (port 5012, text)
#   mistral8B    - mistralai/Mistral-7B-Instruct-v0.3          (port 5020, text)
#   mistralSmall24B - mistralai/Mistral-Small-24B-Instruct-2501 (port 5022, text)
#   qwen7B       - Qwen/Qwen2.5-7B-Instruct                    (port 5030, text)
#   qwen72B      - Qwen/Qwen2.5-72B-Instruct                   (port 5032, text)
#   llava8B      - llava-hf/llama3-llava-next-8b-hf            (port 5009, vision)
#   llava72B     - llava-hf/llava-next-72b-hf                  (port 5008, vision)
#   qwenVL8B     - Qwen/Qwen3-VL-8B-Instruct                   (port 5040, vision)
#   qwenVL32B    - Qwen/Qwen3-VL-32B-Instruct                  (port 5042, vision)
#   mistralVL8B  - mistralai/Ministral-3-8B-Instruct-2512-BF16 (port 5044, vision)
#   mistralVL24B - mistralai/Mistral-Small-3.1-24B-Instruct-2503 (port 5046, vision)

set -euo pipefail

# Let the CUDA caching allocator reuse freed blocks across shape changes, which limits
# memory fragmentation during long-running joins.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_KEY=${1:?"Usage: $0 <model_key> [gpu_ids]"}
GPU_IDS=${2:-0}

eval "$(python -c "
from reasondb.config.model_registry import ModelRegistry
spec = ModelRegistry.get().spec_by_key('${MODEL_KEY}')
print(f'MODEL_NAME=\"{spec.model_name}\"')
print(f'MODEL_PORT={spec.port}')
print(f'MODEL_MODALITY=\"{spec.modality}\"')
")"

if [ "${MODEL_MODALITY}" = "vision" ]; then
    SERVER_SCRIPT=reasondb/backends/kv_cache_image_qa_server.py
else
    SERVER_SCRIPT=reasondb/backends/kv_cache_text_qa_server.py
fi

echo "Launching ${MODEL_MODALITY} server for ${MODEL_KEY}: ${MODEL_NAME} on GPU(s) ${GPU_IDS}, port ${MODEL_PORT}"

# --device-id is relative to CUDA_VISIBLE_DEVICES, so it stays 0; with multiple
# GPUs visible, device_map="auto" shards the model across all of them.
CUDA_VISIBLE_DEVICES=${GPU_IDS} python "${SERVER_SCRIPT}" \
    --model-name "${MODEL_NAME}" --device-id 0
