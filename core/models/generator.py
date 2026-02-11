import httpx
import json
from config.config import VLLM_API_URL, VLLM_GEN_API_PORT, VLLM_GEN_MODEL_NAME


class VLLMClient:
    async def stream(self, messages, tools=None):
        payload = {"model": VLLM_GEN_MODEL_NAME, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"http://{VLLM_API_URL}:{VLLM_GEN_API_PORT}/v1/chat/completions",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise Exception(f"VLLM API error: {error_body.decode()}")

                async for line in response.aiter_lines():
                    if not line or not line.strip():
                        continue

                    # vLLM/OpenAI streaming uses SSE format with "data: " prefix
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix

                    # Skip the [DONE] marker
                    if line.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(line)

                        # In streaming mode, content is in delta, not message
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})

                            reasoning = delta.get("reasoning", "")
                            if reasoning:
                                yield ("thinking", reasoning)

                            content = delta.get("content", "")
                            if content:
                                yield ("content", content)
                    except json.JSONDecodeError as e:
                        continue

    async def generate(self, messages, tools=None, tool_choice=None, temperature=0.7):
        payload = {
            "model": VLLM_GEN_MODEL_NAME,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"http://{VLLM_API_URL}:{VLLM_GEN_API_PORT}/v1/chat/completions",
                json=payload,
            )
            if response.status_code != 200:
                error_body = await response.aread()
                raise Exception(f"VLLM API error: {error_body.decode()}")

            resp_json = response.json()
            # Try common response shapes
            if isinstance(resp_json, dict):
                if "message" in resp_json and isinstance(resp_json["message"], dict):
                    return resp_json["message"]
                if "choices" in resp_json and len(resp_json["choices"]) > 0:
                    choice = resp_json["choices"][0]
                    if (
                        isinstance(choice, dict)
                        and "message" in choice
                        and isinstance(choice["message"], dict)
                    ):
                        # Return the full message object (may contain tool_calls)
                        return choice["message"]
                    if "text" in choice:
                        return choice.get("text", "")
            # Fallback to returning raw text
            return response.text
