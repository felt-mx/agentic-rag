from fastapi import APIRouter
from pydantic import BaseModel
from core.models.generator import VLLMClient
from retrieval.pipeline import RetrievalPipeline
from retrieval.state import AgentState
from retrieval.query.dispatcher import dispatch
from retrieval.query.corpus_summary import get_corpus_summary
from retrieval.answer.prompt_builder import (
    build_prompt,
    build_critique_prompt,
    build_clarifying_question_prompt,
)
from config.config import MILVUS_DATABASE

chat_router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    text: str
    database: str = None


@chat_router.post("")
async def chat(request: ChatRequest):
    try:
        database = request.database or MILVUS_DATABASE
        retrieval_pipeline = RetrievalPipeline(database=database)
        vllm_client = VLLMClient()
        corpus_summary = get_corpus_summary()

        state = AgentState(original_query=request.text)
        results = []

        # ------------------------------------------------------------------
        # Agentic dispatch loop
        # ------------------------------------------------------------------
        while state.retries_remaining >= 0:
            # 1. Dispatch: choose strategy + processed queries
            state = await dispatch(request.text, vllm_client, state, corpus_summary)
            print(
                f"[dispatch] strategy={state.current_strategy} "
                f"queries={state.processed_queries} "
                f"reason={state.reasoning}"
            )

            # 2. Execute the chosen strategy worker
            if state.current_strategy == "Expansion":
                results = await retrieval_pipeline.retrieve_with_expansion(
                    original_query=request.text,
                    top_k=5,
                    retrieval_k=20,
                )
            elif state.current_strategy == "Decomposition":
                results = await retrieval_pipeline.retrieve_with_decomposition(
                    original_query=request.text,
                    queries=state.processed_queries,
                    top_k=5,
                    retrieval_k=20,
                )
            else:
                # Hybrid: decompose then expand each sub-query
                results = await retrieval_pipeline.retrieve_hybrid(
                    original_query=request.text,
                    sub_queries=state.processed_queries,
                    top_k=5,
                    retrieval_k=20,
                )

            print(f"[retrieval] got {len(results)} results")

            # 3. Sufficiency check
            if results:
                critique_prompt = build_critique_prompt(request.text, results)
                critique_response = await vllm_client.generate(
                    critique_prompt, tools=None, tool_choice=None
                )
                critique_text = critique_response.get("content", "").strip()
                first_line = critique_text.splitlines()[0].strip().upper()
                print(f"[sufficiency] {critique_text}")

                if "INSUFFICIENT" not in first_line:
                    state.best_results = results
                    break
                else:
                    lines = critique_text.splitlines()
                    critique_sentence = lines[1].strip() if len(
                        lines) > 1 else critique_text
                    state.critique_log.append(critique_sentence)
                    if results and not state.best_results:
                        state.best_results = results
            else:
                state.critique_log.append("Retrieval returned no results.")

            state.retries_remaining -= 1

        # ------------------------------------------------------------------
        # Exit strategy
        # ------------------------------------------------------------------
        final_results = results if results else state.best_results
        disclaimer = ""

        if not final_results:
            # Nothing useful at all — return a clarifying question in the
            # standard response shape so the frontend needs no changes.
            clarify_prompt = build_clarifying_question_prompt(
                request.text, state.critique_log
            )
            clarify_response = await vllm_client.generate(
                clarify_prompt, tools=None, tool_choice=None
            )
            clarifying_text = clarify_response.get("content", "").strip()
            return {
                "data": {
                    "message": {"content": clarifying_text},
                    "score": None,
                    "metadata": {"source": None, "page": None},
                    "all_results": [],
                }
            }

        if state.retries_remaining < 0 and results != state.best_results:
            disclaimer = (
                "> **Note:** I could not find a definitive answer after multiple attempts. "
                "Here is the best available information:\n\n"
            )

        # ------------------------------------------------------------------
        # Answer generation (non-streaming)
        # ------------------------------------------------------------------
        prompt = build_prompt(final_results, request.text, None)
        response = await vllm_client.generate(
            prompt, tools=None, tool_choice=None, enable_thinking=True
        )

        top_result = final_results[0] if final_results else None
        RAW_SCORE_MIN = 0.01
        scores_and_metadata = [
            {
                "score": r.get("score"),
                "metadata": {
                    "source": r.get("metadata", {}).get("file_name"),
                    "page": r.get("metadata", {}).get("section_title"),
                },
            }
            for r in (final_results if isinstance(final_results, list) else [])
            if r.get("score", 0) >= RAW_SCORE_MIN
        ]

        answer_content = disclaimer + (response.get("content", "") or "")

        return {
            "data": {
                "message": {**response, "content": answer_content},
                "score": top_result["score"] if top_result and top_result["score"] >= RAW_SCORE_MIN else None,
                "metadata": {
                    "source": top_result["metadata"]["file_name"] if top_result and top_result["score"] >= RAW_SCORE_MIN else None,
                    "page": top_result["metadata"]["section_title"] if top_result and top_result["score"] >= RAW_SCORE_MIN else None,
                },
                "all_results": scores_and_metadata,
            }
        }

    except Exception as e:
        return {"error": str(e)}
