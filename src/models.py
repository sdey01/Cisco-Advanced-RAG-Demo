from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryResult:
    query: str
    answer: str
    classification: str
    classification_confidence: float
    pipeline: str
    cache_hit: bool
    latency_ms: float
    expanded_queries: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
