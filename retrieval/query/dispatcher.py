from __future__ import annotations

import json

from retrieval.state import AgentState

# ---------------------------------------------------------------------------
# Tool schema handed to the generator for structured output
# ---------------------------------------------------------------------------

DISPATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "dispatch_strategy",
        "description": (
            "Analyse the user query and choose the best retrieval strategy, "
            "then emit the processed queries to use for retrieval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["Expansion", "Decomposition", "Hybrid"],
                    "description": (
                        "Expansion: query is short, vague, uses acronyms, or lacks detail. "
                        "Set processed_queries to [original_query] only — do NOT generate expansion variants here; "
                        "the retrieval layer will handle domain-aware expansion internally. "
                        "Decomposition: query has multiple distinct subjects or 'and/but' clauses — "
                        "split into one query per sub-topic. "
                        "Hybrid: needs both — split into sub-topics AND each sub-topic also needs expansion; "
                        "provide the sub-topic queries and the retrieval layer expands each."
                    ),
                },
                "processed_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "For Expansion: 2-3 semantically equivalent, enriched rewrites of the query. "
                        "For Decomposition: one query per distinct sub-topic. "
                        "For Hybrid: decomposed sub-queries with precise technical terms."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining why this strategy was chosen.",
                },
            },
            "required": ["strategy", "processed_queries", "reasoning"],
        },
    },
}


async def dispatch(
    user_text: str,
    generator,
    state: AgentState,
    corpus_summary: str = "",
) -> AgentState:
    """
    Calls the generator with the dispatch tool to decide which retrieval
    strategy to use and what processed queries to issue.

    On retry iterations the critique_log in *state* is included so the LLM
    can learn from previous failures and choose a different approach.
    """
    from retrieval.answer.prompt_builder import build_dispatcher_prompt

    prompt = build_dispatcher_prompt(
        user_text, state.critique_log, corpus_summary, state.tried_queries)

    try:
        response = await generator.generate(
            prompt,
            tools=[DISPATCH_TOOL],
            tool_choice={"type": "function", "function": {
                "name": "dispatch_strategy"}},
            temperature=0.1,
        )

        tool_calls = response.get("tool_calls")
        if tool_calls:
            args = json.loads(tool_calls[0]["function"]["arguments"])
            state.current_strategy = args.get("strategy", "Expansion")
            state.processed_queries = args.get(
                "processed_queries") or [user_text]
            state.reasoning = args.get("reasoning", "")
        else:
            # Generator didn't return a tool call — safe fallback
            state.current_strategy = "Expansion"
            state.processed_queries = [user_text]
            state.reasoning = "Fallback: dispatcher returned no tool call."

    except Exception as e:
        print(f"[dispatcher] Warning: {e}")
        state.current_strategy = "Expansion"
        state.processed_queries = [user_text]
        state.reasoning = f"Error during dispatch: {e}"

    return state
