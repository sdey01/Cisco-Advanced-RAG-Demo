from pathlib import Path
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(
    path: Path, chunk_size: int = 1024, chunk_overlap: int = 32
) -> list[Document]:
    if not path.is_file():
        raise FileNotFoundError(f"Source PDF was not found: {path}")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size")

    try:
        pages = PyPDFLoader(str(path)).load()
    except Exception as exc:
        raise RuntimeError(f"Could not extract pages from {path.name}: {exc}") from exc
    if not pages or not any(page.page_content.strip() for page in pages):
        raise ValueError(f"No readable text was extracted from {path.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    source_name = path.name
    per_page_index: dict[int, int] = {}
    for chunk in chunks:
        page = int(chunk.metadata.get("page", 0)) + 1
        chunk_index = per_page_index.get(page, 0)
        per_page_index[page] = chunk_index + 1
        chunk.metadata.update(
            {
                "source": source_name,
                "document_name": source_name,
                "page": page,
                "chunk_id": hashlib.sha256(
                    f"{source_name}:{page}:{chunk_index}:{chunk.page_content}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
        )
    return chunks
