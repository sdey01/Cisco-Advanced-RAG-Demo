from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document


def open_vectorstore(embeddings, persist_directory: Path, collection_name: str) -> Chroma:
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def replace_documents(vectorstore: Chroma, chunks: list[Document]) -> None:
    if not chunks:
        raise ValueError("Refusing to create an empty vector index")
    existing_ids = vectorstore.get().get("ids", [])
    if existing_ids:
        vectorstore.delete(ids=existing_ids)
    vectorstore.add_documents(
        chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks]
    )
