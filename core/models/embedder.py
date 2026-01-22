import httpx
import json
from configs.config import VLLM_API_URL, VLLM_EMBED_API_PORT, VLLM_EMBED_MODEL_NAME


class VLLMClient:
    def __init__(
        self,
        server_url: str = f"http://{VLLM_API_URL}:{VLLM_EMBED_API_PORT}",
        model_name: str = VLLM_EMBED_MODEL_NAME,
    ):
        self.server_url = server_url
        self.model_name = model_name

    async def generate(self, text: str) -> list[float]:
        payload = {
            "model": self.model_name,
            "input": text,
        }

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{self.server_url}/v1/embeddings", json=payload
            )

            if response.status_code != 200:
                error_body = await response.aread()
                raise Exception(f"VLLM API error: {error_body.decode()}")

            response_json = response.json()

            if isinstance(response_json, dict):
                if "data" in response_json and len(response_json["data"]) > 0:
                    embedding = response_json["data"][0]["embedding"]
                    return embedding
                else:
                    raise Exception(f"Unexpected response format: {response_json}")
            else:
                raise Exception(f"Invalid response type: {type(response_json)}")

    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch"""
        payload = {
            "model": self.model_name,
            "input": texts,  # vLLM supports batch inputs
        }

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{self.server_url}/v1/embeddings",
                json=payload,
            )

            if response.status_code != 200:
                error_body = await response.aread()
                raise Exception(f"VLLM API error: {error_body.decode()}")

            response_json = response.json()

            if isinstance(response_json, dict):
                if "data" in response_json:
                    # Extract all embeddings, sorted by index
                    embeddings = sorted(response_json["data"], key=lambda x: x["index"])
                    return [item["embedding"] for item in embeddings]
                else:
                    raise Exception(f"Unexpected response format: {response_json}")
            else:
                raise Exception(f"Invalid response type: {type(response_json)}")

    async def late_chunking_embed(
        self,
        text: str,
        chunk_spans: list[tuple[int, int]],
        task: str = "retrieval.passage",
    ) -> list[list[float]]:
        payload = {
            "model": self.model_name,
            "input": text,
            "task": task,
            "late_chunking": True,
            "spans": chunk_spans,
        }

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{self.server_url}/v1/embeddings", json=payload
            )

            if response.status_code != 200:
                error_body = await response.aread()
                raise Exception(f"VLLM API error: {error_body.decode()}")

            response_json = response.json()

            if isinstance(response_json, dict):
                if "data" in response_json:
                    embeddings = sorted(
                        response_json["data"], key=lambda x: x.get("index", 0)
                    )
                    return [item["embedding"] for item in embeddings]
                else:
                    raise Exception(f"Unexpected response format: {response_json}")
            else:
                raise Exception(f"Invalid response type: {type(response_json)}")
