\# AI Research Copilot



**A Retrieval-Augmented Generation (RAG) System for Document Question Answering**



\---



\## Overview



This project implements a Retrieval-Augmented Generation (RAG) pipeline that enables users to query documents and receive context-aware responses from a local Large Language Model (LLM).



The system improves answer accuracy by retrieving relevant document context before generation, reducing hallucination compared to standard LLM usage.



\---



\## Key Features



\* Support for multiple document formats (`.txt`, `.md`, `.pdf`)

\* Context-preserving text chunking with overlap

\* Semantic embeddings using transformer-based models

\* Efficient similarity search using FAISS

\* Local LLM inference (Qwen)

\* Interactive command-line interface

\* Prompt engineering for controlled response generation



\---



\## System Architecture



```text

User Query

&#x20;   ↓

Query Embedding

&#x20;   ↓

Vector Search (FAISS)

&#x20;   ↓

Relevant Context Retrieval

&#x20;   ↓

Prompt Construction

&#x20;   ↓

LLM Generation

&#x20;   ↓

Final Answer

```



\---



\## Project Structure



```text

ai-research-copilot/

├── data/

│   └── sample.txt

├── src/

│   ├── document\_loader.py

│   ├── text\_splitter.py

│   ├── embeddings.py

│   ├── vector\_store.py

│   ├── prompts.py

│   └── rag\_pipeline.py

├── app.py

└── README.md

```



\---



\## Installation



\### 1. Clone the repository



```bash

git clone https://github.com/your-username/ai-research-copilot.git

cd ai-research-copilot

```



\### 2. Create environment



```bash

conda create -n llm python=3.11

conda activate llm

```



\### 3. Install dependencies



```bash

pip install torch transformers sentence-transformers faiss-cpu pypdf accelerate

```



\---



\## Usage



Run the application:



```bash

python app.py

```



Example queries:



```text

What is this document about?

Summarize the document

Explain how RAG works

```



\---



\## Technical Implementation



\### Document Processing



\* Supports `.txt`, `.md`, `.pdf`

\* Extracts raw text for downstream processing



\### Text Chunking



\* Fixed-size chunking with overlap

\* Preserves semantic continuity across chunks



\### Embedding Generation



\* Model: `all-MiniLM-L6-v2`

\* Converts text into dense vector representations



\### Vector Search



\* FAISS-based similarity search using L2 distance

\* Retrieves top-k relevant document chunks



\### RAG Pipeline



\* Query → embedding → retrieval → prompt → LLM generation

\* Ensures responses are grounded in retrieved context



\---



\## Prompt Engineering



The system enforces controlled generation using structured prompts:



```text

Use only the provided context

Do not hallucinate

Answer clearly and concisely

```



\---



\## Example



\*\*Input:\*\*



```text

What is this document about?

```



\*\*Output:\*\*



```text

The document describes large language models, their applications,

and how retrieval-augmented generation improves answer accuracy.

```



\---



\## Limitations



\* Small local model may produce less fluent responses

\* Limited context window for long documents

\* Command-line interface only



\---



\## Future Work



\* Web interface using Streamlit

\* Multi-document retrieval support

\* Model quantization for faster inference

\* API deployment using FastAPI

\* Advanced reranking techniques



\---



\## Learnings



\* End-to-end RAG system design

\* Embedding-based semantic search

\* Prompt engineering strategies

\* Integration of LLMs with retrieval systems



\---



\## Author



Zhe Luo



\---



\## License



MIT License



