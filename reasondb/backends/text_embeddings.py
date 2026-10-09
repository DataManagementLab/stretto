import numpy as np
import requests
from datetime import datetime
from typing import Dict, List, Tuple

from reasondb.backends.backend import Backend
from reasondb.utils.logging import FileLogger


BATCH_SIZE = 100
PORT_TEXT_SIM = 5324


class TextSimilarityBackend(Backend):
    def __init__(self, model_name: str):
        self.check_model_name = model_name
        self.embed_dim = 0

    def setup(self, logger: FileLogger):
        result = requests.get(f"http://localhost:{PORT_TEXT_SIM}/status")
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self.check_model_name

        self.embed_dim = json_response["embed_dim"]
        logger.info(
            __name__,
            f"Text similarity model {self.check_model_name} is ready (Dim: {self.embed_dim})",
        )

    async def wind_down(self):
        pass

    def shutdown(self, logger: FileLogger):
        pass

    async def run(
        self,
        texts1: List[str],
        texts2: List[str],
    ) -> Tuple[List[float], float, float]:
        """Generates an embedding for a search query.

        Scores each distinct (text1, text2) pair once and fans the score back out over
        the input positions. Callers index the result positionally against their rows,
        and the same extracted values recur constantly - `ExtractAndMatch*` above a join
        sees each pair of extracted values once per candidate row, so the duplicate
        factor is the join fan-out. Same contract as `KvTextQABackend.run`.

        The deduplication runs over integer codes in numpy rather than over `(str, str)`
        keys in a dict: a self-join on the larger benchmarks reaches 10^8 pairs, where a
        dict keyed on tuples costs tens of GB of host memory before anything is sent,
        and the request body carries only the distinct texts plus two index arrays.
        """
        assert len(texts1) == len(texts2)
        if not texts1:
            return [], 0.0, 0.0

        # Distinct texts first (a few thousand extracted values behind millions of
        # pairs), then the pairs as codes into that list.
        text_to_code: Dict[str, int] = {}
        codes1 = self._encode(texts1, text_to_code)
        codes2 = self._encode(texts2, text_to_code)

        # One int64 key per pair, so np.unique dedups pairs without materializing them.
        n_texts = len(text_to_code)
        pair_keys = codes1.astype(np.int64) * n_texts + codes2
        unique_keys, inverse = np.unique(pair_keys, return_inverse=True)
        del pair_keys, codes1, codes2

        start_time = datetime.now()
        result = requests.post(
            f"http://localhost:{PORT_TEXT_SIM}/text_sim",
            json={
                "texts": list(text_to_code),
                "idx1": (unique_keys // n_texts).tolist(),
                "idx2": (unique_keys % n_texts).tolist(),
            },
        )
        assert result.status_code == 200
        result_json = result.json()

        unique_similarities = np.asarray(result_json["similarities"], dtype=np.float64)
        similarities: List[float] = unique_similarities[inverse].tolist()
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()

        return similarities, runtime, 0.0

    @staticmethod
    def _encode(texts: List[str], text_to_code: Dict[str, int]) -> np.ndarray:
        """Map `texts` to int codes in a shared table, as an array rather than a list.

        `np.fromiter` with an explicit count keeps this to 4 bytes per pair; the list a
        comprehension would build costs ~8 bytes of pointer plus a boxed int per entry
        for anything above 256.
        """

        def code(text: str) -> int:
            existing = text_to_code.get(text)
            if existing is None:
                existing = len(text_to_code)
                text_to_code[text] = existing
            return existing

        return np.fromiter(
            (code(t) for t in texts), dtype=np.int32, count=len(texts)
        )

    def get_operation_identifier(self) -> str:
        return f"TextSimilarityBackend-{self.check_model_name}"

    def setup_index(self, table, column):
        pass
