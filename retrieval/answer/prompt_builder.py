from retrieval.answer.system_prompt import (
    get_retry_prompt,
    get_system_prompt,
    get_reformatted_prompt,
    get_relevance_check_prompt,
    get_image_description_prompt,
    get_chat_image_analysis_prompt,
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


def build_chat_image_analysis_prompt(image_data: str) -> list:
    """Prompt for the chat flow: asks the model to *interpret* the image content
    rather than just transcribe it, producing concept-level descriptions suitable
    for RAG query augmentation.
    """
    return [
        {"role": "system", "content": get_chat_image_analysis_prompt()},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is this an image of? Describe the subject using clinical or domain-specific concepts only, in one sentence.",
                },
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        },
    ]


async def describe_image(image_data: str, vllm_client) -> str:
    """Call the vision-capable LLM to produce an interpretive description of
    *image_data* suitable for RAG query augmentation.

    Uses ``build_chat_image_analysis_prompt`` which instructs the model to
    interpret meaning (e.g. 'pre-diabetes') rather than only transcribe raw
    values.  *image_data* must be a base64 data URI.
    Returns the description string, or an empty string on failure.
    """
    prompt = build_chat_image_analysis_prompt(image_data)
    try:
        response = await vllm_client.generate(
            prompt, tools=None, tool_choice=None, enable_thinking=False
        )
        return response.get("content", "").strip()
    except Exception:
        return ""


def build_augmented_query(user_text: str, image_descriptions: list) -> str:
    """Merge image descriptions with the user query into a single text string.

    Returns *user_text* unchanged when the list is empty so that text-only
    requests are completely unaffected.
    """
    if not image_descriptions:
        return user_text
    image_blocks = "\n\n".join(
        f"[Image {i + 1}]\n{desc}" for i, desc in enumerate(image_descriptions)
    )
    return f"{image_blocks}\n\n[User Question]\n{user_text}"


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
