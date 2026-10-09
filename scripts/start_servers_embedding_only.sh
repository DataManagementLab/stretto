#!/bin/bash
# Starts only the two embedding/similarity servers (image_similarity_server.py,
# text_embed_server.py) - nothing else. This is all a `--simulate` worker ever needs:
# SimulateStore replaces the real KV text/image/audio servers entirely, but the
# embedding backends (ImageSimilarityBackend/TextSimilarityBackend) recompute
# embeddings per query regardless of --simulate, so they're required either way. See
# reasondb/backends/image_similarity.py's ImageSimilarityBackend.assert_ready.
#
# Used by reasondb/coordinator/capabilities.py for workers started with
# --capability embedding-only or --capability simulate.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EMBED_GPUS=${EMBED_GPUS:-0}

CUDA_VISIBLE_DEVICES=$EMBED_GPUS python reasondb/backends/image_similarity_server.py &
CUDA_VISIBLE_DEVICES=$EMBED_GPUS python reasondb/backends/text_embed_server.py
