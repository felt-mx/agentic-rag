import base64
from fastapi import APIRouter, File, Form, UploadFile
from core.models.generator import VLLMClient
from retrieval.pipeline import RetrievalPipeline
from retrieval.state import AgentState
from retrieval.query.dispatcher import dispatch
from retrieval.query.corpus_summary import get_corpus_summary
from retrieval.answer.prompt_builder import (
    build_prompt,
    build_critique_prompt,
    build_clarifying_question_prompt,
    describe_image,
    build_augmented_query,
)
from config.config import MILVUS_DATABASE

chat_router = APIRouter(prefix="/chat")


@chat_router.post("")
async def chat(
    text: str = Form(...),
    database: str = Form(None),
    thinking: bool = Form(False),
    files: list[UploadFile] = File(default=None),
):
    try:
        database = database or MILVUS_DATABASE
        retrieval_pipeline = RetrievalPipeline(database=database)
        vllm_client = VLLMClient()
        corpus_summary = get_corpus_summary()

        image_descriptions = []
        if files:
            for file in files:
                file_bytes = await file.read()
                mime = file.content_type or "image/jpeg"
                b64 = base64.b64encode(file_bytes).decode("utf-8")
                data_uri = f"data:{mime};base64,{b64}"
                desc = await describe_image(data_uri, vllm_client)
                if desc:
                    image_descriptions.append(desc)

        effective_query = build_augmented_query(text, image_descriptions)
        state = AgentState(original_query=text, image_descriptions=image_descriptions)
        results = []

        # ------------------------------------------------------------------
        # Agentic dispatch loop
        # ------------------------------------------------------------------
        while state.retries_remaining >= 0:
            # 1. Dispatch: choose strategy + processed queries
            state = await dispatch(effective_query, vllm_client, state, corpus_summary)
            print(
                f"[dispatch] strategy={state.current_strategy} "
                f"queries={state.processed_queries} "
                f"reason={state.reasoning}"
            )

            # Record these queries so the dispatcher can avoid repeating them.
            state.tried_queries.append(list(state.processed_queries))

            # 2. Execute the chosen strategy worker
            if state.current_strategy == "Expansion":
                results = await retrieval_pipeline.retrieve_with_expansion(
                    original_query=effective_query,
                    top_k=5,
                    retrieval_k=20,
                )
            elif state.current_strategy == "Decomposition":
                results = await retrieval_pipeline.retrieve_with_decomposition(
                    original_query=effective_query,
                    queries=state.processed_queries,
                    top_k=5,
                    retrieval_k=20,
                )
            else:
                # Hybrid: decompose then expand each sub-query
                results = await retrieval_pipeline.retrieve_hybrid(
                    original_query=effective_query,
                    sub_queries=state.processed_queries,
                    top_k=5,
                    retrieval_k=20,
                )

            print(f"[retrieval] got {len(results)} results")

            # 3. Merge new results into the accumulated pool (dedup by text,
            #    keep highest raw score when the same chunk appears twice).
            acc_map = {r.get("answer", "")[:500]: r for r in state.accumulated_results}
            for r in results:
                key = r.get("answer", "")[:500]
                if key not in acc_map or r.get("score", 0) > acc_map[key].get(
                    "score", 0
                ):
                    acc_map[key] = r
            state.accumulated_results = sorted(
                acc_map.values(), key=lambda x: x.get("score", 0), reverse=True
            )

            # 4. Sufficiency check — evaluate the *accumulated* pool so we
            #    don't discard useful chunks found in earlier iterations.
            if state.accumulated_results:
                critique_prompt = build_critique_prompt(
                    effective_query, state.accumulated_results
                )
                critique_response = await vllm_client.generate(
                    critique_prompt, tools=None, tool_choice=None
                )
                critique_text = critique_response.get("content", "").strip()
                first_line = critique_text.splitlines()[0].strip().upper()
                print(f"[sufficiency] {critique_text}")

                if "INSUFFICIENT" not in first_line:
                    state.best_results = state.accumulated_results
                    break
                else:
                    lines = critique_text.splitlines()
                    critique_sentence = (
                        lines[1].strip() if len(lines) > 1 else critique_text
                    )
                    state.critique_log.append(critique_sentence)
                    if not state.best_results:
                        state.best_results = state.accumulated_results
            else:
                state.critique_log.append("Retrieval returned no results.")

            state.retries_remaining -= 1

        # ------------------------------------------------------------------
        # Exit strategy
        # ------------------------------------------------------------------
        final_results = state.accumulated_results or state.best_results
        disclaimer = ""

        if not final_results:
            # Nothing useful at all — return a clarifying question in the
            # standard response shape so the frontend needs no changes.
            clarify_prompt = build_clarifying_question_prompt(
                effective_query, state.critique_log
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
        prompt = build_prompt(final_results, effective_query, None)
        response = await vllm_client.generate(
            prompt, tools=None, tool_choice=None, enable_thinking=thinking
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
                "score": (
                    top_result["score"]
                    if top_result and top_result["score"] >= RAW_SCORE_MIN
                    else None
                ),
                "metadata": {
                    "source": (
                        top_result["metadata"]["file_name"]
                        if top_result and top_result["score"] >= RAW_SCORE_MIN
                        else None
                    ),
                    "page": (
                        top_result["metadata"]["section_title"]
                        if top_result and top_result["score"] >= RAW_SCORE_MIN
                        else None
                    ),
                },
                "all_results": scores_and_metadata,
            }
        }

    except Exception as e:
        return {"error": str(e)}
