from datetime import datetime, timedelta, timezone
import json
import logging
import math
from pathlib import Path


logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self, path: Path, ttl_seconds: int, similarity_threshold: float):
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_hit_latency_ms_total = 0.0
        self.cache_miss_latency_ms_total = 0.0

    def get(self, query: str, query_embedding: list[float]) -> dict | None:
        entries = self._read()
        now = datetime.now(timezone.utc)
        live_entries = []
        best: tuple[float, dict] | None = None
        for entry in entries:
            try:
                if datetime.fromisoformat(entry["expires_at"]) <= now:
                    continue
                live_entries.append(entry)
                score = _cosine_similarity(query_embedding, entry["query_embedding"])
                if score >= self.similarity_threshold and (best is None or score > best[0]):
                    best = (score, entry)
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed semantic cache entry")
        if len(live_entries) != len(entries):
            self._write(live_entries)
        if best:
            self.cache_hits += 1
            return {**best[1], "similarity": best[0]}
        self.cache_misses += 1
        return None

    def set(self, query: str, query_embedding: list[float], response: dict) -> None:
        now = datetime.now(timezone.utc)
        entry = {
            "query": query,
            "query_embedding": query_embedding,
            "response": response,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=self.ttl_seconds)).isoformat(),
        }
        entries = [
            value
            for value in self._read()
            if _not_expired(value, now)
            and value.get("query", "").casefold() != query.casefold()
        ]
        entries.append(entry)
        self._write(entries)

    def metrics(self) -> dict[str, float | int]:
        total = self.cache_hits + self.cache_misses
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / total if total else 0.0,
            "cache_hit_latency_ms": (
                self.cache_hit_latency_ms_total / self.cache_hits
                if self.cache_hits
                else 0.0
            ),
            "cache_miss_latency_ms": (
                self.cache_miss_latency_ms_total / self.cache_misses
                if self.cache_misses
                else 0.0
            ),
        }

    def record_latency(self, cache_hit: bool, latency_ms: float) -> None:
        if cache_hit:
            self.cache_hit_latency_ms_total += latency_ms
        else:
            self.cache_miss_latency_ms_total += latency_ms

    def _read(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, OSError):
            logger.exception("Could not read semantic cache; starting empty")
            return []

    def _write(self, entries: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(entries), encoding="utf-8")
        temporary.replace(self.path)


def _not_expired(entry: dict, now: datetime) -> bool:
    try:
        return datetime.fromisoformat(entry["expires_at"]) > now
    except (KeyError, TypeError, ValueError):
        return False


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else -1.0
