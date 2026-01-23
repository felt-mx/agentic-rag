from fastapi import APIRouter
from pydantic import BaseModel
from core.models.generator import VLLMClient
from retrieval.pipeline import RetrievalPipeline
from retrieval.answer.prompt_builder import build_prompt, build_reformulation_prompt, build_retry_prompt

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

        reformulated_text = reformulated_response.message.content.strip()

        results = await retrieval_pipeline.retrieve(
            reformulated_text,
            top_k=5,  # Default 5 if not specified
            retrieval_k=20,  # Default 20 if not specified
            rerank_method="weighted",  # Default "weighted" if not specified
        )

        while True:
            if not results and count < 3:
                top_result = None
                answer = "No relevant information found."
                count += 1

                input_texts.append(reformulated_text)
                retry_prompt = build_retry_prompt(
                    reformulated_text)

                reformulated_response = await vllm_client.generate(
                    retry_prompt, tools=None, tool_choice=None
                )
                reformulated_text = reformulated_response.message.content.strip()
                results = await retrieval_pipeline.retrieve(
                    reformulated_text,
                    top_k=5,  # Default 5 if not specified
                    retrieval_k=20,  # Default 20 if not specified
                    rerank_method="weighted",  # Default "weighted" if not specified
                )
            else:  # Either results found or max retries reached
                top_result = results[0]  # Get the top result (highest score)
                answer = top_result["answer"]
                break

        prompt = build_prompt(answer, request.text, None)

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
