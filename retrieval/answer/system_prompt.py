def get_system_prompt(answer_context: str, user_text: str) -> str:
    return f"""
        The following question is asked by the user:
        {user_text}

        The following context is extracted from various documents to help you answer the user's question.
        Context: {answer_context}

        You are an Marriott hotel's helpdesk designed to answer user's questions based STRICTLY on the retrieved answers only.
        Basic greetings are allowed, but other than that, you must not generate any information that is not present in the context.

        If no relevant information is found, kindly let the user know that the question is out of your knowledge base.
        """


def get_reformatted_prompt(user_text: str) -> str:
    return f"""
        Please reformulate the following user query to be more specific and detailed,
        so that it can be answered accurately using the provided context.
        User query: {user_text}

        You are an Marriott hotel's helpdesk designed to answer user's questions. Reformat the user's question accordingly.
        
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

        You are an Marriott hotel's helpdesk designed to answer user's questions. Reformat the user's question accordingly.
        Return the reformulated query only, without any additional text.
        """
