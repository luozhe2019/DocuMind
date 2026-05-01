from src.document_loader import load_document
from src.text_splitter import split_text
from src.embeddings import embed_texts
from src.vector_store import VectorStore
from src.prompts import build_rag_prompt

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np


class RAGPipeline:
    """
    End-to-end RAG pipeline:
    document → chunk → embeddings → vector search → LLM answer
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Initialize the LLM and tokenizer
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )

        self.vector_store = None

    def build_index(self, file_path: str):
        """
        Load document and build vector index
        """

        # 1. Load document
        text = load_document(file_path)

        # 2. Split into chunks
        chunks = split_text(text)

        # 3. Convert chunks into embeddings
        embeddings = embed_texts(chunks)

        # 4. Build vector store
        dim = embeddings.shape[1]
        self.vector_store = VectorStore(dim)
        self.vector_store.add_documents(chunks, embeddings)

    def ask(self, question: str) -> str:
        """
        Ask a question using RAG
        """

        # 1. Convert question to embedding
        query_embedding = embed_texts([question])

        # 2. Retrieve top relevant chunks
        results = self.vector_store.search(query_embedding, top_k=3)

        # Extract only text
        context = "\n".join([chunk for chunk, _ in results])

        # 3. Build prompt
        prompt = build_rag_prompt(question, context)

        # 4. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 5. Generate answer
        outputs = self.model.generate(**inputs, max_new_tokens=150)

        # 6. Decode output
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        return answer