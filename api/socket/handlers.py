import socketio
from .server import get_sio
from core.models.generator import VLLMClient
from retrieval.pipeline import RetrievalPipeline
from retrieval.answer.prompt_builder import (
    build_prompt,
    build_reformulation_prompt,
    build_retry_prompt,
    build_relevance_check_prompt,
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

        await sio.emit(
            "stream_start", {"message": "Processing your request..."}, room=sid
        )

        retrieval_pipeline = RetrievalPipeline()
        vllm_client = VLLMClient()
        count = 0
        input_texts = []

        # Reformulate the query
        reformulation_prompt = build_reformulation_prompt(user_text)
        reformulated_response = await vllm_client.generate(
            reformulation_prompt, tools=None, tool_choice=None, temperature=0.1
        )

        input_texts = [user_text]
        reformulated_text = reformulated_response.get("content", "").strip()

        print(reformulated_response)

        await sio.emit(
            "processing",
            {"step": "query_reformulation", "reformulated_query": reformulated_text},
            room=sid,
        )

        # Retrieve relevant documents
        results = await retrieval_pipeline.retrieve(
            reformulated_text,
            top_k=5,
            retrieval_k=20,
            rerank_method="weighted",
        )

        # Retry logic for better results
        while True:
            if not results and count < 3:
                top_result = None
                count += 1

                input_texts.append(reformulated_text)
                retry_prompt = build_retry_prompt(input_texts)

                reformulated_response = await vllm_client.generate(
                    retry_prompt, tools=None, tool_choice=None
                )
                reformulated_text = reformulated_response.get(
                    "content", "").strip()

                print(f"Retry {count}: {reformulated_text}")

                await sio.emit(
                    "processing",
                    {
                        "step": "retry_reformulation",
                        "attempt": count,
                        "reformulated_query": reformulated_text,
                    },
                    room=sid,
                )

                results = await retrieval_pipeline.retrieve(
                    reformulated_text,
                    top_k=5,
                    retrieval_k=20,
                    rerank_method="weighted",
                )
            elif results and count < 3:
                # Check relevance
                relevance_prompt = build_relevance_check_prompt(
                    user_text, results)
                relevance_response = await vllm_client.generate(
                    relevance_prompt, tools=None, tool_choice=None
                )
                relevance_verdict = (
                    relevance_response.get("content", "").strip().upper()
                )

                print(f"Relevance check: {relevance_verdict}")

                await sio.emit(
                    "processing",
                    {"step": "relevance_check", "verdict": relevance_verdict},
                    room=sid,
                )

                if "INSUFFICIENT" in relevance_verdict:
                    count += 1
                    input_texts.append(reformulated_text)
                    retry_prompt = build_retry_prompt(input_texts)

                    reformulated_response = await vllm_client.generate(
                        retry_prompt, tools=None, tool_choice=None
                    )
                    reformulated_text = reformulated_response.get(
                        "content", "").strip()

                    print(
                        f"Retry {count} (insufficient results): {reformulated_text}")

                    await sio.emit(
                        "processing",
                        {
                            "step": "insufficient_retry",
                            "attempt": count,
                            "reformulated_query": reformulated_text,
                        },
                        room=sid,
                    )

                    results = await retrieval_pipeline.retrieve(
                        reformulated_text,
                        top_k=5,
                        retrieval_k=20,
                        rerank_method="weighted",
                    )
                else:
                    break
            else:
                break

        if not results:
            results = "No relevant information found."
            top_result = None
        else:
            top_result = results[0]

        # Build final prompt and start streaming
        prompt = build_prompt(results, user_text, None)

        await sio.emit(
            "stream_content_start", {"message": "Generating response..."}, room=sid
        )

        # Stream the response content
        full_content = ""
        async for chunk_type, content_chunk in vllm_client.stream(prompt, tools=None, enable_thinking=True):
            if chunk_type == "thinking":
                await sio.emit("stream_thinking", {"content": content_chunk}, room=sid)
            elif chunk_type == "content":
                if content_chunk:
                    full_content += content_chunk
                    await sio.emit(
                        "stream_content", {"content": content_chunk}, room=sid
                    )

        # Build complete response data
        scores_and_metadata = []
        if results and isinstance(results, list):
            for result in results:
                scores_and_metadata.append(
                    {
                        "score": result.get("score"),
                        "metadata": {
                            "source": result.get("metadata", {}).get("file_name"),
                            "page": result.get("metadata", {}).get("section_title"),
                        },
                    }
                )

        complete_response = {
            "data": {
                "message": {"content": full_content},
                "score": top_result["score"] if top_result else None,
                "metadata": {
                    "source": (
                        top_result["metadata"]["file_name"] if top_result else None
                    ),
                    "page": (
                        top_result["metadata"]["section_title"] if top_result else None
                    ),
                },
                "all_results": scores_and_metadata,
            }
        }

        # Send complete response
        await sio.emit("stream_complete", complete_response, room=sid)

    except Exception as e:
        await sio.emit("error", {"message": str(e)}, room=sid)
