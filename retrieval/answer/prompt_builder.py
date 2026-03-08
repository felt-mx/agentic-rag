from retrieval.answer.system_prompt import (
    get_retry_prompt,
    get_system_prompt,
    get_reformatted_prompt,
    get_relevance_check_prompt,
    get_image_description_prompt,
    get_dispatcher_prompt,
    get_expansion_prompt,
    get_critique_prompt,
    get_clarifying_question_prompt,
)


def build_prompt(answer_context, user_text, images=None):
    prompt = []

    # Add system prompt
    system_content = get_system_prompt(answer_context)
    prompt.append({"role": "system", "content": system_content})

    # Add user message (required by vLLM)
    prompt.append({"role": "user", "content": user_text})

    return prompt


def build_reformulation_prompt(user_text):
    prompt = []

    # Add reformulation prompt
    reformatted_content = get_reformatted_prompt()
    prompt.append({"role": "system", "content": reformatted_content})

    # Add user message (required by vLLM)
    prompt.append({"role": "user", "content": user_text})

    return prompt


def build_retry_prompt(user_text: list[str]) -> str:
    prompt = []

    retry_content = get_retry_prompt(user_text)
    prompt.append({"role": "system", "content": retry_content})

    # Add user message (required by vLLM)
    prompt.append({"role": "user", "content": user_text[0]})

    return prompt


def build_relevance_check_prompt(user_query: str, retrieved_results: list) -> list:
    prompt = []

    relevance_content = get_relevance_check_prompt(retrieved_results)
    prompt.append({"role": "system", "content": relevance_content})

    # Add user message (required by vLLM)
    prompt.append({"role": "user", "content": user_query})

    return prompt


def build_image_description_prompt(image_data: str) -> list:
    prompt = []

    # Add system prompt
    system_content = get_image_description_prompt()
    prompt.append({"role": "system", "content": system_content})

    # Add user message with image
    prompt.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please analyze this image and describe EVERYTHING that you see.",
                },
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        }
    )

    return prompt


# ---------------------------------------------------------------------------
# New builders for the agentic dispatch / strategy loop
# ---------------------------------------------------------------------------

def build_dispatcher_prompt(
    user_text: str,
    critique_log: list,
    corpus_summary: str = "",
    tried_queries: list = None,
) -> list:
    system_content = get_dispatcher_prompt(
        corpus_summary, critique_log, tried_queries or []
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_text},
    ]


def build_expansion_prompt(query_text: str, corpus_summary: str = "") -> list:
    return [
        {"role": "system", "content": get_expansion_prompt(corpus_summary)},
        {"role": "user", "content": query_text},
    ]


def build_critique_prompt(user_query: str, retrieved_results: list) -> list:
    system_content = get_critique_prompt(retrieved_results)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query},
    ]


def build_clarifying_question_prompt(user_query: str, critique_log: list) -> list:
    system_content = get_clarifying_question_prompt(critique_log)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query},
    ]
