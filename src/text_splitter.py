from typing import List


def split_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50
) -> List[str]:
    """
    Split a long document into smaller overlapping text chunks.

    Why this is needed:
    Large language models and embedding models cannot always process
    very long documents at once. In a RAG system, we split documents
    into smaller chunks, embed each chunk, and later retrieve the most
    relevant chunks for a user's question.

    Args:
        text: The full document text.
        chunk_size: The maximum number of characters in each chunk.
        overlap: The number of characters shared between two nearby chunks.

    Returns:
        A list of text chunks.
    """

    # This list will store all generated chunks.
    chunks = []

    # Start reading the text from the beginning.
    start = 0

    # Continue splitting until we reach the end of the document.
    while start < len(text):
        # The end position of the current chunk.
        end = start + chunk_size

        # Extract one chunk from the full text.
        chunk = text[start:end]

        # Add the chunk to the list.
        chunks.append(chunk)

        # Move the start position forward.
        # We subtract overlap so that adjacent chunks share some text.
        # This helps preserve context across chunk boundaries.
        start += chunk_size - overlap

    return chunks