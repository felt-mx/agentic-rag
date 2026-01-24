from fastapi import APIRouter
from pydantic import BaseModel
from core.models.generator import VLLMClient
from retrieval.pipeline import RetrievalPipeline
from retrieval.answer.prompt_builder import (
    build_prompt,
    build_reformulation_prompt,
    build_retry_prompt,
    build_relevance_check_prompt,
)

chat_router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    text: str


@chat_router.post("")
async def chat(request: ChatRequest):
    try:
        retrieval_pipeline = RetrievalPipeline()
        vllm_client = VLLMClient()
        count = 0
        input_texts = []

        reformulation_prompt = build_reformulation_prompt(request.text)
        reformulated_response = await vllm_client.generate(
            reformulation_prompt, tools=None, tool_choice=None
        )

        input_texts = [request.text]

        print(reformulated_response)

        reformulated_text = reformulated_response.get("content", "").strip()

        results = await retrieval_pipeline.retrieve(
            reformulated_text,
            top_k=5,  # Default 5 if not specified
            retrieval_k=20,  # Default 20 if not specified
            rerank_method="weighted",  # Default "weighted" if not specified
        )

        while True:
            # First check: Are there any results?
            if not results and count < 3:
                top_result = None
                count += 1

                input_texts.append(reformulated_text)
                retry_prompt = build_retry_prompt(input_texts)

                reformulated_response = await vllm_client.generate(
                    retry_prompt, tools=None, tool_choice=None
                )
                reformulated_text = reformulated_response.get("content", "").strip()

                print(f"Retry {count}: {reformulated_text}")
                results = await retrieval_pipeline.retrieve(
                    reformulated_text,
                    top_k=5,  # Default 5 if not specified
                    retrieval_k=20,  # Default 20 if not specified
                    rerank_method="weighted",  # Default "weighted" if not specified
                )
            elif results and count < 3:
                # Second check: Are the results actually relevant and sufficient?
                relevance_prompt = build_relevance_check_prompt(
                    request.text, results
                )
                relevance_response = await vllm_client.generate(
                    relevance_prompt, tools=None, tool_choice=None
                )
                relevance_verdict = relevance_response.get("content", "").strip().upper()
                
                print(f"Relevance check: {relevance_verdict}")
                
                if "INSUFFICIENT" in relevance_verdict:
                    # Results exist but are not relevant enough, retry with reformulation
                    count += 1
                    input_texts.append(reformulated_text)
                    retry_prompt = build_retry_prompt(input_texts)

                    reformulated_response = await vllm_client.generate(
                        retry_prompt, tools=None, tool_choice=None
                    )
                    reformulated_text = reformulated_response.get("content", "").strip()

                    print(f"Retry {count} (insufficient results): {reformulated_text}")
                    results = await retrieval_pipeline.retrieve(
                        reformulated_text,
                        top_k=5,  # Default 5 if not specified
                        retrieval_k=20,  # Default 20 if not specified
                        rerank_method="weighted",  # Default "weighted" if not specified
                    )
                else:
                    # Results are sufficient, exit the loop
                    break
            else:  # Either max retries reached or results found and relevant
                break

        if not results:
            results = "No relevant information found."
        else:
            top_result = results[0]

        prompt = build_prompt(results, request.text, None)

        response = await vllm_client.generate(prompt, tools=None, tool_choice=None)

        return {
            "data": {
                "message": response,
                "score": top_result["score"] if top_result else None,
                "metadata": {
                    "source": (
                        top_result["metadata"]["file_name"] if top_result else None
                    ),
                    "page": (
                        top_result["metadata"]["section_title"] if top_result else None
                    ),
                },
            }
        }
    except Exception as e:
        return {"error": str(e)}
