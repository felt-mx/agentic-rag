from fastapi import APIRouter
from pydantic import BaseModel
from core.models.generator import VLLMClient
from retrieval.pipeline import RetrievalPipeline
from retrieval.answer.prompt_builder import build_prompt

chat_router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    text: str


@chat_router.post("")
async def chat(request: ChatRequest):
    try:
        retrieval_pipeline = RetrievalPipeline()
        vllm_client = VLLMClient()

        results = await retrieval_pipeline.retrieve(
            request.text,
            top_k=5,  # Default 5 if not specified
            retrieval_k=20,  # Default 20 if not specified
            rerank_method="weighted",  # Default "weighted" if not specified
        )

        if not results:
            answer = "No relevant information found."
            top_result = None
        else:
            top_result = results[0]  # Get the top result (highest score)
            answer = top_result["answer"]

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
