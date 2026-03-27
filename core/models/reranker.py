import httpx
import json
import math
from config.config import VLLM_API_URL, VLLM_RERANK_API_PORT, VLLM_RERANK_MODEL_NAME


class VLLMClient:
    def __init__(
        self,
        server_url: str = f"http://{VLLM_API_URL}:{VLLM_RERANK_API_PORT}",
        model_name: str = VLLM_RERANK_MODEL_NAME,
    ):
        self.server_url = server_url
        self.model_name = model_name

    @staticmethod
    def _sigmoid(x: float) -> float:
        # Numerically stable sigmoid for large positive/negative logits.
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _normalize_score(self, score: float) -> float:
        """
        Return a score in [0, 1].

        - If backend already returns probabilities, keep them.
        - If backend returns logits (common in llama.cpp rerank adapters),
          transform with sigmoid.
        """
        value = float(score)
        if 0.0 <= value <= 1.0:
            return value
        return self._sigmoid(value)

    async def rerank(
        self, query: str, documents: list[str], top_n: int = None
    ) -> list[dict]:
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{self.server_url}/v1/rerank", json=payload)

            if response.status_code != 200:
                error_body = await response.aread()
                raise Exception(
                    f"VLLM Rerank API error: {error_body.decode()}")

            response_json = response.json()

            if isinstance(response_json, dict):
                if "results" in response_json:
                    results = sorted(
                        response_json["results"], key=lambda x: x["index"])
                    return [self._normalize_score(r["relevance_score"]) for r in results]
                else:
                    raise Exception(
                        f"Unexpected response format: {response_json}")
            else:
                raise Exception(
                    f"Invalid response type: {type(response_json)}")
