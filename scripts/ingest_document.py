import logging

from src.config import Settings
from src.embeddings import create_embeddings
from src.ingestion.pdf_loader import load_and_split_pdf
from src.vectorstore import open_vectorstore, replace_documents


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    settings = Settings.from_env()
    chunks = load_and_split_pdf(
        settings.source_document_path,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    embeddings = create_embeddings(settings.embedding_model)
    vectorstore = open_vectorstore(
        embeddings, settings.chroma_persist_directory, settings.collection_name
    )
    replace_documents(vectorstore, chunks)
    logging.info(
        "Indexed %s chunks from %s into %s",
        len(chunks),
        settings.source_document_path.name,
        settings.chroma_persist_directory,
    )


if __name__ == "__main__":
    main()
