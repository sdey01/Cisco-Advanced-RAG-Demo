from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _path_setting(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else ROOT_DIR / value


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    llm_model: str
    embedding_model: str
    reranker_model: str
    chunk_size: int
    chunk_overlap: int
    factual_retrieval_k: int
    broad_retrieval_k: int
    factual_context_k: int
    broad_context_k: int
    query_expansion_count: int
    cache_ttl_seconds: int
    cache_similarity_threshold: float
    chroma_persist_directory: Path
    collection_name: str
    source_document_path: Path
    cache_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "gpt-5.4-mini"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            reranker_model=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1024")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "32")),
            factual_retrieval_k=int(os.getenv("FACTUAL_RETRIEVAL_K", "4")),
            broad_retrieval_k=int(os.getenv("BROAD_RETRIEVAL_K", "20")),
            factual_context_k=int(os.getenv("FACTUAL_CONTEXT_K", "4")),
            broad_context_k=int(os.getenv("BROAD_CONTEXT_K", "8")),
            query_expansion_count=3,
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
            cache_similarity_threshold=float(
                os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.90")
            ),
            chroma_persist_directory=_path_setting(
                "CHROMA_PERSIST_DIRECTORY", "./data/chroma"
            ),
            collection_name=os.getenv(
                "COLLECTION_NAME", "cisco_nexus_troubleshooting"
            ),
            source_document_path=_path_setting(
                "SOURCE_DOCUMENT_PATH",
                "./cisco-nexus-9000-series-nx-os-troubleshooting-guide-106x.pdf",
            ),
            cache_path=ROOT_DIR / "data" / "semantic_cache.json",
        )
