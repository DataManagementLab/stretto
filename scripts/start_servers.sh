#NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# CUDA_VISIBLE_DEVICES=0,1,2 python reasondb/backends/kv_cache_image_qa_server.py --model-name llava-hf/llama3-llava-next-8b-hf &
# KV8B_IMAGE_QA_SERVER_PID=$!
# CUDA_VISIBLE_DEVICES=0,1,2 python reasondb/backends/kv_cache_image_qa_server.py --model-name llava-hf/llava-next-72b-hf &
# KV70B_IMAGE_QA_SERVER_PID=$!
CUDA_VISIBLE_DEVICES=0,1,2 python reasondb/backends/kv_cache_text_qa_server.py --model-name meta-llama/Llama-3.1-8B-Instruct &
KV8B_TEXT_QA_SERVER_PID=$!
CUDA_VISIBLE_DEVICES=0,1,2 python reasondb/backends/kv_cache_text_qa_server.py --model-name meta-llama/Llama-3.1-70B-Instruct &
KV70B_TEXT_QA_SERVER_PID=$!

# if [ "$NUM_GPU" -gt 2 ]; then
#   python reasondb/backends/kv_cache_audio_qa_server.py --device-id 2 &
#   KV_AUDIO_QA_SERVER_PID=$!
# else
#   python reasondb/backends/kv_cache_audio_qa_server.py --device-id 1 &
#   KV_AUDIO_QA_SERVER_PID=$!
# fi

# CUDA_VISIBLE_DEVICES=3 python reasondb/backends/image_similarity_server.py &
# KV_IMAGE_SIMILARITY_SERVER_PID=$!

echo "kill $KV8B_TEXT_QA_SERVER_PID; kill $KV70B_TEXT_QA_SERVER_PID;" >scripts/kill_servers_rotowire.sh
# bash scripts/kill_servers.sh
