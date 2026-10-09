import logging
import argparse
from flask import Flask, request
from flask_restful import Resource, Api
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from reasondb.backends.text_embeddings import PORT_TEXT_SIM

# SOTA model for short text embeddings (MTEB Leaderboard favorite)
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32
DATALOADER_NUM_WORKERS = 4

# How many (text1, text2) pairs to score in one gather. The pair count is the join
# fan-out, so it reaches tens of millions on the large benchmarks: gathering all of
# them at once materializes two [n_pairs, embed_dim] float tensors (98 GiB at 68M
# pairs), while the unique-embedding table it gathers from is a few hundred MB.
PAIR_CHUNK_SIZE = 1 << 16
# Smallest batch worth retrying at before giving up on the GPU entirely.
MIN_BATCH_SIZE = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)


class TextEmbeddingModelWrapper:
    def __init__(self, model_name, device_id):
        self.model_name = model_name
        self.device_id = device_id
        self.device = torch.device(
            f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        )
        self.init_model()

    def init_model(self):
        logger.info(f"Loading Text Embedding model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Get embedding dimension dynamically
        self.embed_dim = self.model.config.hidden_size
        logger.info(f"Model loaded. Dimension: {self.embed_dim}")

    def _embed_one_batch(self, texts: List[str]):
        # Tokenize
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. BGE uses the [CLS] token (index 0)
            embeddings = model_output[0][:, 0]
            # Normalize embeddings (standard practice for BGE/Cosine Similarity)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def compute_batch_embeddings(self, texts: List[str], batch_size: int = BATCH_SIZE):
        """Embed `texts` in fixed-size batches, returning one [len(texts), dim] tensor.

        Batched rather than one forward pass over everything: the tokenizer pads to the
        longest text in whatever it is handed, so a single call over thousands of texts
        allocates a [n_texts, 512] attention workspace sized by the worst-case text.
        Texts are ordered by length so each batch pads to its own longest member, and
        the results are scattered back into input order.
        """
        if not texts:
            return torch.empty((0, self.embed_dim), device=self.device)

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.empty((len(texts), self.embed_dim), device=self.device)

        start = 0
        while start < len(order):
            size = min(batch_size, len(order) - start)
            while True:
                chunk = order[start : start + size]
                try:
                    out[chunk] = self._embed_one_batch([texts[i] for i in chunk])
                    break
                except torch.OutOfMemoryError:
                    if size <= MIN_BATCH_SIZE:
                        raise
                    size = max(MIN_BATCH_SIZE, size // 2)
                    torch.cuda.empty_cache()
                    logger.warning(
                        "OOM while embedding; retrying at batch size %d", size
                    )
            start += size

        return out

    def get_single_embedding(self, text: str):
        """Helper for single query strings."""
        return self._embed_one_batch([text])[0].cpu().numpy()


model_wrapper = None


class Status(Resource):
    def get(self):
        assert model_wrapper is not None
        return {
            "status": "alive",
            "model_name": model_wrapper.model_name,
            "embed_dim": model_wrapper.embed_dim,
            "device": str(model_wrapper.device),
        }, 200


def _score_pairs(embeddings, idx1: List[int], idx2: List[int]) -> List[float]:
    """Cosine similarity of `embeddings[idx1[k]]` against `embeddings[idx2[k]]`.

    Scored in chunks of `PAIR_CHUNK_SIZE`: the pair count is the join fan-out and runs
    to 10^8 on the large benchmarks, so gathering every pair at once would allocate
    two tensors of `n_pairs * embed_dim` floats regardless of how few texts are behind
    them.
    """
    device = embeddings.device
    similarities: List[float] = []
    for start in range(0, len(idx1), PAIR_CHUNK_SIZE):
        stop = start + PAIR_CHUNK_SIZE
        chunk1 = torch.tensor(idx1[start:stop], dtype=torch.long, device=device)
        chunk2 = torch.tensor(idx2[start:stop], dtype=torch.long, device=device)
        scores = (embeddings[chunk1] * embeddings[chunk2]).sum(1)
        similarities.extend(scores.cpu().tolist())
    return similarities


class TextSimilarity(Resource):
    def post(self):
        """
        Expects JSON, either as texts:
        {
            "texts1": ["Text", ...],
            "texts2": ["Text", ...]
        }
        or, preferably, as indices into a deduplicated text list:
        {
            "texts": ["Text", ...],
            "idx1": [0, ...],
            "idx2": [1, ...]
        }
        The second form is what `TextSimilarityBackend` sends. A pair list is the join
        fan-out and repeats the same handful of extracted values millions of times, so
        spelling both sides out as strings makes the request body — and the dict this
        would rebuild from it — some tens of times larger than the texts in it.
        """
        data = request.get_json(force=True)

        assert model_wrapper is not None
        if "texts" in data:
            texts_unique = data["texts"]
            idx1 = data.get("idx1", [])
            idx2 = data.get("idx2", [])
        else:
            texts1 = data.get("texts1", [])
            texts2 = data.get("texts2", [])
            # dict.fromkeys rather than set(): stable order, so a repeated request
            # embeds the same texts in the same batches.
            texts_unique = list(dict.fromkeys(texts1 + texts2))
            texts_map = {t: i for i, t in enumerate(texts_unique)}
            idx1 = [texts_map[t] for t in texts1]
            idx2 = [texts_map[t] for t in texts2]

        logger.info(
            "text_sim: %d pairs over %d distinct texts", len(idx1), len(texts_unique)
        )
        embeddings = None
        try:
            embeddings = model_wrapper.compute_batch_embeddings(texts_unique)
            similarities = _score_pairs(embeddings, idx1, idx2)
        finally:
            # This GPU is shared with the KV cache servers, so hand the blocks back
            # rather than holding a request's peak until the next one needs it.
            del embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"similarities": similarities}, 200


api.add_resource(Status, "/status")
api.add_resource(TextSimilarity, "/text_sim")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    model_wrapper = TextEmbeddingModelWrapper(MODEL_NAME, args.device_id)
    app.run(host="127.0.0.1", port=PORT_TEXT_SIM, debug=False)
