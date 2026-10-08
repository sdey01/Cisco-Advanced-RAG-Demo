import logging
from time import perf_counter
from uuid import uuid4

from langchain_openai import ChatOpenAI

from src.generation import AnswerGenerator
from src.models import QueryResult
from src.query_expansion import QueryExpander
from src.reranker import CrossEncoderReranker
from src.retrieval import retrieve_and_deduplicate
from src.routing import QueryClassifier
from src.semantic_cache import SemanticCache


logger = logging.getLogger(__name__)


class QueryOrchestrator:
    def __init__(self, settings, embeddings, vectorstore):
        self.settings = settings
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        model = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            timeout=45,
            max_retries=1,
        )
        self.classifier = QueryClassifier(model)
        self.expander = QueryExpander(model)
        self.reranker = CrossEncoderReranker(settings.reranker_model)
        self.generator = AnswerGenerator(model)
        self.cache = SemanticCache(
            settings.cache_path,
            settings.cache_ttl_seconds,
            settings.cache_similarity_threshold,
        )

    def ask(self, query: str) -> QueryResult:
        started = perf_counter()
        request_id = uuid4().hex
        query = query.strip()
        if not query:
            raise ValueError("Enter a question before submitting")
        query_embedding = self.embeddings.embed_query(query)
        try:
            cached = self.cache.get(query, query_embedding)
        except Exception:
            logger.exception("Semantic cache lookup failed; continuing uncached")
            cached = None
        if cached:
            response = cached["response"]
            latency_ms = (perf_counter() - started) * 1000
            self.cache.record_latency(True, latency_ms)
            logger.info(
                "request_id=%s cache_hit=true latency_ms=%.1f",
                request_id,
                latency_ms,
            )
            return QueryResult(
                query=query,
                answer=response["answer"],
                classification=response["classification"],
                classification_confidence=response["classification_confidence"],
                pipeline=response["pipeline"],
                cache_hit=True,
                latency_ms=latency_ms,
                expanded_queries=response.get("expanded_queries", []),
                retrieved_chunks=response.get("retrieved_chunks", []),
            )

        classification = self.classifier.classify(query)
        expanded = self.expander.expand(query)
        factual = classification.category == "FACTUAL_LOOKUP"
        pipeline = "Factual Lookup" if factual else "Broad / Strategic"
        per_query_k = (
            self.settings.factual_retrieval_k
            if factual
            else self.settings.broad_retrieval_k
        )
        final_k = (
            self.settings.factual_context_k
            if factual
            else self.settings.broad_context_k
        )
        candidates = retrieve_and_deduplicate(self.vectorstore, expanded, per_query_k)
        try:
            ranked = self.reranker.rerank(query, candidates, final_k)
        except Exception:
            logger.exception("Cross-encoder reranking failed; falling back to retrieval scores")
            ranked = [
                {**chunk, "rerank_score": chunk["retrieval_score"], "final_rank": rank}
                for rank, chunk in enumerate(
                    sorted(candidates, key=lambda item: item["retrieval_score"], reverse=True)[
                        :final_k
                    ],
                    start=1,
                )
            ]
        answer = self.generator.generate(query, ranked)
        result = QueryResult(
            query=query,
            answer=answer,
            classification=classification.category,
            classification_confidence=classification.confidence,
            pipeline=pipeline,
            cache_hit=False,
            latency_ms=(perf_counter() - started) * 1000,
            expanded_queries=expanded,
            retrieved_chunks=ranked,
        )
        response = {
            "answer": result.answer,
            "classification": result.classification,
            "classification_confidence": result.classification_confidence,
            "pipeline": result.pipeline,
            "expanded_queries": result.expanded_queries,
            "retrieved_chunks": result.retrieved_chunks,
            "latency_ms": result.latency_ms,
        }
        try:
            self.cache.set(query, query_embedding, response)
        except Exception:
            logger.exception("Could not write semantic cache")
        result.latency_ms = (perf_counter() - started) * 1000
        self.cache.record_latency(False, result.latency_ms)
        logger.info(
            "request_id=%s classification=%s pipeline=%s cache_hit=false "
            "retrieval_candidates=%s reranked_candidates=%s latency_ms=%.1f",
            request_id,
            result.classification,
            result.pipeline,
            len(candidates),
            len(ranked),
            result.latency_ms,
        )
        return result
