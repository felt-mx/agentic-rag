import base64
import socketio
from .server import get_sio
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

sio = get_sio()


@sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")
    await sio.emit("status", {"message": "Connected to chat server"}, room=sid)


@sio.event
async def disconnect(sid):
    print(f"Client {sid} disconnected")


@sio.event
async def chat_stream(sid, data):
    try:
        user_text = data.get("text", "")
        if not user_text:
            await sio.emit("error", {"message": "No text provided"}, room=sid)
            return

        enable_thinking = bool(data.get("thinking", False))

        retrieval_pipeline = RetrievalPipeline()
        vllm_client = VLLMClient()
        corpus_summary = get_corpus_summary()

        image_descriptions = []
        raw_files = data.get("files") or []
        for file_entry in raw_files:
            data_uri = None
            if isinstance(file_entry, str) and file_entry.startswith("data:"):
                # Already a full data URI
                data_uri = file_entry
            elif isinstance(file_entry, dict):
                # { name, type, data } object sent by the frontend
                mime = file_entry.get("type", "image/png")
                b64 = file_entry.get("data", "")
                if b64:
                    data_uri = f"data:{mime};base64,{b64}"
            if data_uri:
                desc = await describe_image(data_uri, vllm_client)
                if desc:
                    image_descriptions.append(desc)

        effective_query = build_augmented_query(user_text, image_descriptions)
        state = AgentState(
            original_query=user_text, image_descriptions=image_descriptions
        )
        results = []

        # ------------------------------------------------------------------
        # Agentic dispatch loop
        # ------------------------------------------------------------------
        while state.retries_remaining >= 0:
            # 1. Dispatch: choose strategy + processed queries
            state = await dispatch(effective_query, vllm_client, state, corpus_summary)
            await sio.emit(
                "status",
                {
                    "message": (
                        f"Strategy: {state.current_strategy} | "
                        f"Queries: {state.processed_queries}"
                    )
                },
                room=sid,
            )
            print(
                f"[dispatch] strategy={state.current_strategy} "
                f"queries={state.processed_queries} "
                f"reason={state.reasoning}"
            )

            # 2. Execute the chosen strategy worker
            if state.current_strategy == "Expansion":
                # LLM-based query expansion then parallel retrieval
                results = await retrieval_pipeline.retrieve_with_expansion(
                    original_query=effective_query,
                    top_k=5,
                    retrieval_k=20,
                )
            elif state.current_strategy == "Decomposition":
                # Parallel retrieval for each decomposed sub-query
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

            # 3. Sufficiency check
            if results:
                critique_prompt = build_critique_prompt(effective_query, results)
                critique_response = await vllm_client.generate(
                    critique_prompt, tools=None, tool_choice=None
                )
                critique_text = critique_response.get("content", "").strip()
                first_line = critique_text.splitlines()[0].strip().upper()
                print(f"[sufficiency] {critique_text}")

                if "INSUFFICIENT" not in first_line:
                    # Results are good — proceed to answer generation
                    state.best_results = results
                    break
                else:
                    # Extract the critique sentence (line 2 if present)
                    lines = critique_text.splitlines()
                    critique_sentence = (
                        lines[1].strip() if len(lines) > 1 else critique_text
                    )
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
            # Nothing useful at all — generate a clarifying question and deliver
            # it as the answer so the frontend needs no changes.
            clarify_prompt = build_clarifying_question_prompt(
                effective_query, state.critique_log
            )
            clarify_response = await vllm_client.generate(
                clarify_prompt, tools=None, tool_choice=None
            )
            clarifying_text = clarify_response.get("content", "").strip()
            await sio.emit("stream_content", {"content": clarifying_text}, room=sid)
            complete_response = {
                "data": {
                    "message": {"content": clarifying_text},
                    "score": None,
                    "metadata": {"source": None, "page": None},
                    "all_results": [],
                }
            }
            await sio.emit("stream_content", complete_response, room=sid)
            return

        if state.retries_remaining < 0 and results != state.best_results:
            disclaimer = (
                "> **Note:** I could not find a definitive answer after multiple attempts. "
                "Here is the best available information:\n\n"
            )

        # ------------------------------------------------------------------
        # Answer generation (streaming)
        # ------------------------------------------------------------------
        prompt = build_prompt(final_results, effective_query, None)
        full_content = disclaimer

        async for chunk_type, content_chunk in vllm_client.stream(
            prompt, tools=None, enable_thinking=enable_thinking
        ):
            if chunk_type == "thinking":
                await sio.emit("stream_thinking", {"content": content_chunk}, room=sid)
            elif chunk_type == "content":
                if content_chunk:
                    full_content += content_chunk
                    await sio.emit(
                        "stream_content", {"content": content_chunk}, room=sid
                    )

        # Build complete response data
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

        complete_response = {
            "data": {
                "message": {"content": full_content},
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

        await sio.emit("stream_complete", complete_response, room=sid)

    except Exception as e:
        await sio.emit("error", {"message": str(e)}, room=sid)
