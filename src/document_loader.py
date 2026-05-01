from pathlib import Path
from pypdf import PdfReader


def load_txt_or_md(file_path: str) -> str:
    path = Path(file_path)
    return path.read_text(encoding="utf-8")


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


def load_document(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() in [".txt", ".md"]:
        return load_txt_or_md(file_path)

    if path.suffix.lower() == ".pdf":
        return load_pdf(file_path)

    raise ValueError(f"Unsupported file type: {path.suffix}")