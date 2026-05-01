def build_rag_prompt(question: str, context: str) -> str:
    """
    Build a RAG prompt using retrieved context and the user's question.
    """

    return f"""
You are an AI research assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""


def build_concise_prompt(question: str, context: str) -> str:
    """
    Build a shorter prompt for concise answers.
    """

    return f"""
Answer the question briefly using the context.

Context:
{context}

Question:
{question}

Short answer:
"""