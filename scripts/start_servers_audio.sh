#!/bin/bash
# Starts the audio KV cache server plus the two always-required embedding servers.
# Unlike the text/image servers, the audio server does not support relative indices
# (--use-relative-indices), so USE_INDICES has no effect here.
#
# Used by reasondb/coordinator/capabilities.py for workers started with
# --capability audio.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Per-host GPU overrides (default: device 0). See
# reasondb/coordinator/capabilities.py for how a worker sources these per-host.
AUDIO_DEVICE_ID=${AUDIO_DEVICE_ID:-0}
EMBED_GPUS=${EMBED_GPUS:-0}

python reasondb/backends/kv_cache_audio_qa_server.py --device-id $AUDIO_DEVICE_ID &

CUDA_VISIBLE_DEVICES=$EMBED_GPUS python reasondb/backends/image_similarity_server.py &
CUDA_VISIBLE_DEVICES=$EMBED_GPUS python reasondb/backends/text_embed_server.py
