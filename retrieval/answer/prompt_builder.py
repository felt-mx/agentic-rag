from retrieval.answer.system_prompt import get_system_prompt


def build_prompt(answer_context, user_text, images=None):
    prompt = []

    # Add system prompt
    system_content = get_system_prompt(answer_context, user_text)
    prompt.append({"role": "system", "content": system_content})

    return prompt
