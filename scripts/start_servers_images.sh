# Whether the KV cache servers reconstruct a compressed cache on the fly from a
# less-compressed baseline + relative indices, instead of reading a cache materialized
# at exactly the requested compression ratio. Must match how the benchmark is run:
#
#   bash scripts/start_servers_images.sh                 # physical cache per ratio
#     -> python scripts/run_coordinator.py --local --producer run_benchmark ...
#   USE_INDICES=1 bash scripts/start_servers_images.sh   # baseline + relative indices
#     -> python scripts/run_coordinator.py --local --producer run_benchmark ... --use-indexes
#
# Mismatched: with indices on but --use-indexes off, the servers never generate the
# missing physical caches and the client fails setup with "no usable cache or relative
# index"; with indices off but --use-indexes on, the servers re-prefill each ratio.
USE_INDICES=${USE_INDICES:-0}

INDEX_FLAG=""
if [ "$USE_INDICES" != "0" ]; then
  INDEX_FLAG="--use-relative-indices"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# RAM budget (GB) for KV caches that an -in-memory operator pins at prepare(). The default
# operator suite uses no -in-memory operators, so 0 serves every operator from disk. A
# server without a budget refuses -in-memory operators at setup(). Pinned caches are never
# evicted, so size the budget to hold the whole column's cache set.
export KV_CACHE_PIN_GB=${KV_CACHE_PIN_GB:-0}


# Per-host GPU overrides (defaults: the GPU layout of our experiment machine). See
# reasondb/coordinator/capabilities.py for how a worker sources these per-host.
IMAGE_8B_GPUS=${IMAGE_8B_GPUS:-3}
IMAGE_70B_GPUS=${IMAGE_70B_GPUS:-0,1,2}
EMBED_GPUS=${EMBED_GPUS:-3}

CUDA_VISIBLE_DEVICES=$IMAGE_8B_GPUS python reasondb/backends/kv_cache_image_qa_server.py --model-name llava-hf/llama3-llava-next-8b-hf $INDEX_FLAG &
KV8B_IMAGE_QA_SERVER_PID=$!
CUDA_VISIBLE_DEVICES=$IMAGE_70B_GPUS python reasondb/backends/kv_cache_image_qa_server.py --model-name llava-hf/llava-next-72b-hf $INDEX_FLAG &
KV70B_IMAGE_QA_SERVER_PID=$!

CUDA_VISIBLE_DEVICES=$EMBED_GPUS python reasondb/backends/image_similarity_server.py &
CUDA_VISIBLE_DEVICES=$EMBED_GPUS python reasondb/backends/text_embed_server.py

