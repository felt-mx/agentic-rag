def get_system_prompt(answer_context: str, user_text: str) -> str:
    return f"""
        The following question is asked by the user:
        {user_text}

        The following context is extracted from various documents to help you answer the user's question.
        Context: {answer_context}

        Strictly answer only if the context contains the specific fact. If the answer is not present, you MUST respond with [I don't know]. Do not use outside knowledge.
        Do NOT make assumptions and ensure your answer is directly supported by the provided context.
        When answering the user's question based on the provided answer, answer with confidence and clarity.
        Basic greetings are allowed, but other than that, you must not generate any information that is not present in the context.
        NEVER include sources in your answer. E.g. You can find more information at...
        Your final output MUST be of markdown format.

        If no relevant information is found, kindly let the user know that the question is out of your knowledge base.
        """


def get_reformatted_prompt(user_text: str) -> str:
    return f"""
        Please reformulate the following user query to be more specific and detailed,
        so that it can be answered accurately using the provided context.
        User query: {user_text}

        You are a smart AI agent designed to answer user's questions. Reformat the user's question accordingly.
        When reformatting, try to use the same wording as the original query as much as possible. Only beautify the query.
        
        Return the reformulated query only, without any additional text.
        """


def get_retry_prompt(user_text: list[str]) -> str:
    return f"""
        The previous reformulated query did not return any relevant information.
        Please try to reformulate the following user query again, making it more specific and detailed,
        so that it can be answered accurately using the provided context.

        The user's original query was: {user_text[0]}.

        The previous reformulated queries were:
        {', '.join(user_text[1:])}

        You are a smart AI agent designed to answer user's questions. Try to ask in a different way by using different wording than the previous reformulated queries.
        However, ensure the semantics and intent of the original query is preserved.
        Return the reformulated query only, without any additional text.
        """


def get_relevance_check_prompt(user_query: str, retrieved_results: list) -> str:
    return f"""
        You are a relevance judge for a smart AI agent RAG system.
        
        User's Original Query: {user_query}
        
        Retrieved Results:
        {retrieved_results}
        
        Your task is to determine if the retrieved results are CLEARLY OFF-TOPIC or COMPLETELY UNRELATED to the user's query.
        
        IMPORTANT: Be conservative. Only mark as INSUFFICIENT if the results are OBVIOUSLY wrong or unrelated.
        If there is ANY possibility the results could help answer the query, mark as SUFFICIENT.
        
        Criteria for INSUFFICIENT:
        1. The content is clearly about a completely different topic
        2. There is no useful information whatsoever for the query
        3. A human would immediately see this is the wrong information
        
        Respond with ONLY one word:
        - "SUFFICIENT" if results have ANY relevance or useful information (default to this)
        - "INSUFFICIENT" only if results are CLEARLY and OBVIOUSLY wrong
        
        When in doubt, respond SUFFICIENT.
        Do not provide any explanation, just the single word verdict.
        """


def get_image_description_prompt() -> str:
    return """
        You are an advanced OCR and image description AI model to convert images into text for RAG applications.
        Your task is to look at the image provided and provide a caption that is as detailed and descriptive as possible.
        You MUST extract any text present in the image and include it in the description.
        Your description can include things like objects, shapes, colors, texts, layout, or any other relevant information present in the image that may help with the retrieval process.
        Your description will be in ONE giant paragraph format without any bullet points or new lines.
        """
