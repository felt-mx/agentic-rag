from retrieval.answer.system_prompt import get_retry_prompt, get_system_prompt, get_reformatted_prompt


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
