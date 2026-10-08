from datetime import datetime, timedelta, timezone

from src.retrieval import retrieve_and_deduplicate
from src.semantic_cache import SemanticCache


def test_semantic_cache_hit_and_miss(tmp_path):
    cache = SemanticCache(tmp_path / "cache.json", 3600, 0.9)
    assert cache.get("first", [1.0, 0.0]) is None
    cache.set("first", [1.0, 0.0], {"answer": "grounded"})
    hit = cache.get("similar", [0.99, 0.01])
    assert hit is not None
    assert hit["response"]["answer"] == "grounded"
    assert cache.metrics()["cache_hits"] == 1


def test_semantic_cache_ignores_expired_entry(tmp_path):
    cache = SemanticCache(tmp_path / "cache.json", 3600, 0.9)
    cache.set("old", [1.0, 0.0], {"answer": "stale"})
    entry = cache._read()[0]
    entry["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    cache._write([entry])
    assert cache.get("old", [1.0, 0.0]) is None


def test_semantic_cache_tracks_average_hit_and_miss_latency(tmp_path):
    cache = SemanticCache(tmp_path / "cache.json", 3600, 0.9)
    assert cache.get("miss one", [1.0, 0.0]) is None
    cache.record_latency(False, 100)
    assert cache.get("miss two", [0.0, 1.0]) is None
    cache.record_latency(False, 300)
    cache.set("known", [1.0, 0.0], {"answer": "answer"})
    assert cache.get("known", [1.0, 0.0]) is not None
    cache.record_latency(True, 20)
    assert cache.metrics()["cache_hit_latency_ms"] == 20
    assert cache.metrics()["cache_miss_latency_ms"] == 200


def test_retrieval_deduplicates_chunks_and_keeps_best_score():
    class FakeVectorStore:
        def similarity_search_with_relevance_scores(self, query, k):
            return [(DocumentStub("same", "chunk-1"), 0.7 if query == "a" else 0.9)]

    results = retrieve_and_deduplicate(FakeVectorStore(), ["a", "b"], 2)
    assert len(results) == 1
    assert results[0]["retrieval_score"] == 0.9


class DocumentStub:
    def __init__(self, content, chunk_id):
        self.page_content = content
        self.metadata = {"chunk_id": chunk_id}
