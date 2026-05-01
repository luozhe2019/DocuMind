import faiss
import numpy as np
from typing import List, Tuple


class VectorStore:
    """
    A simple FAISS-based vector store.

    This class stores document chunk embeddings and allows us to
    search for the most semantically similar chunks to a user query.
    """

    def __init__(self, embedding_dim: int):
        """
        Create a FAISS index.

        Args:
            embedding_dim:
                The size of each embedding vector.
        """

        # IndexFlatL2 uses L2 distance to compare vectors.
        # Smaller distance means more similar.
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Store the original text chunks so we can return them later.
        self.chunks: List[str] = []

    def add_documents(self, chunks: List[str], embeddings: np.ndarray) -> None:
        """
        Add document chunks and their embeddings into the vector store.

        Args:
            chunks:
                The text chunks from the document.
            embeddings:
                The vector embeddings for each chunk.
        """

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Search for the most relevant chunks.

        Args:
            query_embedding:
                The embedding vector of the user's question.
            top_k:
                Number of top results to return.

        Returns:
            A list of tuples:
            (matched_text_chunk, distance_score)
        """

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(distance)))

        return results