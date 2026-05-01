# Import the main RAG pipeline class
from src.rag_pipeline import RAGPipeline


# Entry point of the program
if __name__ == "__main__":

    # Initialize the RAG system (loads model and prepares pipeline)
    rag = RAGPipeline()

    # Build the vector index from a document
    # This step:
    # 1. Loads the document
    # 2. Splits it into chunks
    # 3. Converts chunks into embeddings
    # 4. Stores them in a vector database (FAISS)
    rag.build_index("data/sample.txt")

    # Start an interactive loop for user questions
    while True:

        # Get user input
        question = input("\nAsk a question (type 'exit' to quit): ")

        # Exit condition
        if question.lower() == "exit":
            break

        # Pass the question through the RAG pipeline:
        # 1. Convert question to embedding
        # 2. Retrieve relevant document chunks
        # 3. Build prompt with context
        # 4. Generate answer using LLM
        answer = rag.ask(question)

        # Print the generated answer
        print("\nAnswer:")
        print(answer)