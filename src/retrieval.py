from langchain_core.documents import Document


def retrieve_and_deduplicate(
    vectorstore, queries: list[str], per_query_k: int
) -> list[dict]:
    best_by_id: dict[str, dict] = {}
    for query in queries:
        for document, score in vectorstore.similarity_search_with_relevance_scores(
            query, k=per_query_k
        ):
            chunk_id = document.metadata.get("chunk_id") or _fallback_id(document)
            current = best_by_id.get(chunk_id)
            if current is None or float(score) > current["retrieval_score"]:
                best_by_id[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": document.page_content,
                    "metadata": dict(document.metadata),
                    "retrieval_score": float(score),
                }
    return list(best_by_id.values())


def _fallback_id(document: Document) -> str:
    return f"{document.metadata.get('source', '')}:{document.metadata.get('page', '')}:{hash(document.page_content)}"
