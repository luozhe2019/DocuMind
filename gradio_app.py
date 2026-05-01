import traceback
import gradio as gr
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.build_index("data/sample.txt")


def answer_question(question):
    try:
        if not question or not question.strip():
            return "Please enter a question."

        return rag.ask(question)

    except Exception:
        traceback.print_exc()
        return "Something went wrong. Please check the terminal."


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Question",
        value="What is this document about?"
    ),
    outputs=gr.Textbox(label="Answer"),
    title="AI Research Copilot",
    description="Ask questions about data/sample.txt using a local RAG pipeline."
)

demo.launch()