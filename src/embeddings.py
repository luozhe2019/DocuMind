from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


# Load the embedding model (only once)
# This model converts sentences into dense vector representations
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert a list of text strings into vector embeddings.

    Why we do this:
    Computers cannot understand text directly.
    So we convert text into numerical vectors (embeddings)
    that capture semantic meaning.

    Example:
    "cat" and "dog" will have similar vectors.

    Args:
        texts (List[str]):
            A list of input text strings.

    Returns:
        np.ndarray:
            A 2D array where each row is the embedding of a text.
            Shape: (num_texts, embedding_dimension)
    """

    # Step 1: Encode the text into embeddings
    # The model outputs a list of vectors (one per text)
    embeddings = model.encode(texts)

    # Step 2: Convert to NumPy array (required by FAISS later)
    # Also ensure type is float32 for compatibility and performance
    embeddings = np.array(embeddings, dtype="float32")

    return embeddings