from retrieval.answer.system_prompt import (
    get_retry_prompt,
    get_system_prompt,
    get_reformatted_prompt,
    get_relevance_check_prompt,
    get_image_description_prompt,
)


def build_prompt(answer_context, user_text, images=None):
    prompt = []

    # Add system prompt
    system_content = get_system_prompt(answer_context, user_text)
    prompt.append({"role": "system", "content": system_content})

    return prompt


def build_reformulation_prompt(user_text):
    prompt = []

    # Add reformulation prompt
    reformatted_content = get_reformatted_prompt(user_text)
    prompt.append({"role": "system", "content": reformatted_content})

    return prompt


def build_retry_prompt(user_text: list[str]) -> str:
    prompt = []

    retry_content = get_retry_prompt(user_text)
    prompt.append({"role": "system", "content": retry_content})

    return prompt


def build_relevance_check_prompt(user_query: str, retrieved_results: list) -> list:
    prompt = []

    relevance_content = get_relevance_check_prompt(user_query, retrieved_results)
    prompt.append({"role": "system", "content": relevance_content})

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
