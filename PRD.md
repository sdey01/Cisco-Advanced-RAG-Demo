# Product Requirements Document (PRD)

# Cisco Nexus 9000 Troubleshooting RAG Application

**Document type:** Implementation-ready Product Requirements Document\
**Primary implementation assistant:** GitHub Copilot\
**Project type:** Client-facing Proof of Concept (POC)\
**Backend:** Python\
**Frontend:** Streamlit\
**RAG framework:** LangChain\
**Primary source document:** Cisco Nexus 9000 Series product
troubleshooting guide\
**Target LLM:** OpenAI GPT-5.4 Mini\
**Embedding model:** all-MiniLM-L6-v2\
**Vector database:** ChromaDB

------------------------------------------------------------------------

## 1. Executive Summary

Build a Python-based Retrieval-Augmented Generation (RAG) application
that allows engineers to ask technical troubleshooting and configuration
questions against a Cisco Nexus 9000 Series troubleshooting guide.

The application must demonstrate advanced RAG capabilities beyond a
basic vector-search implementation:

1.  Query expansion into three semantically useful query variations.
2.  Vector retrieval using ChromaDB.
3.  Cross-encoder/information reranking of retrieved chunks.
4.  Semantic caching with a one-hour TTL.
5.  Cache hit/miss tracking and query latency measurement.
6.  LLM-based query classification.
7.  Dynamic routing to one of two RAG pipelines:
    -   **Factual Lookup RAG:** optimized for precise technical
        facts/specifications.
    -   **Broad/Strategic RAG:** optimized for questions requiring
        information from multiple parts of the document.
8.  A Streamlit interface showing the answer, retrieved context,
    classification, cache status, latency, and conversation history.

The system is a POC and should remain simple enough to run locally on a
CPU-based machine while demonstrating production-oriented engineering
practices.

------------------------------------------------------------------------

# 2. Goals

## 2.1 Primary Goals

-   Enable engineers to ask natural-language questions about the Cisco
    Nexus 9000 troubleshooting guide.
-   Produce grounded answers based only on retrieved document context.
-   Demonstrate advanced RAG orchestration rather than simple top-k
    similarity search.
-   Dynamically select an appropriate retrieval strategy based on query
    intent.
-   Make the RAG pipeline observable to the end user.
-   Demonstrate the performance benefit of semantic caching.
-   Keep the complete stack Python-based where practical.
-   Make the codebase structured and documented enough for GitHub
    Copilot to implement incrementally.

## 2.2 Secondary Goals

-   Make retrieval behavior explainable.
-   Expose retrieved/reranked context for demonstration purposes.
-   Provide enough metrics to compare cache hits versus cache misses.
-   Maintain a modular architecture so individual RAG components can be
    replaced later.

------------------------------------------------------------------------

# 3. Non-Goals

The following are explicitly outside the initial POC scope:

-   Multi-user enterprise authentication and authorization.
-   Persistent user accounts.
-   MySQL/PostgreSQL or another relational database.
-   Distributed deployment.
-   Kubernetes.
-   Complex agentic workflows.
-   Fine-tuning an LLM.
-   Fine-tuning the embedding model.
-   Real-time Cisco device integration.
-   Executing commands on Cisco devices.
-   Automatically modifying device configuration.
-   Production-scale distributed caching.
-   Enterprise-grade document-management workflows.

The system must provide **information and recommendations grounded in
the document**, not execute troubleshooting commands against live
infrastructure.

------------------------------------------------------------------------

# 4. Target Users

## 4.1 Primary Persona: Network Engineer

A network engineer is troubleshooting a Cisco Nexus 9000 device and
wants answers such as:

-   What does a particular error code mean?
-   What does a specific device/status code indicate?
-   What are the technical specifications or configuration requirements
    for a feature?
-   What troubleshooting steps are recommended?
-   What conditions can cause a particular failure?
-   What should be checked when a particular subsystem is not
    functioning correctly?

## 4.2 Secondary Persona: Technical Demonstration Audience

Client stakeholders should be able to observe:

-   How a query is classified.
-   Which RAG pipeline was selected.
-   Whether the answer came from semantic cache.
-   How long the query took.
-   Which document passages were retrieved.
-   How advanced RAG components improve the pipeline.

------------------------------------------------------------------------

# 5. Representative User Stories

### US-01 --- Factual technical lookup

> "What does error code X indicate on a Cisco Nexus 9000?"

Expected behavior:

1.  Classify as `FACTUAL_LOOKUP`.
2.  Expand the query into three variations.
3.  Retrieve a small number of candidates.
4.  Rerank candidates.
5.  Provide a concise grounded answer.
6.  Display classification, cache status, latency, and source context.

### US-02 --- Broad troubleshooting question

> "What are the possible causes and troubleshooting steps when this
> Nexus 9000 subsystem repeatedly fails?"

Expected behavior:

1.  Classify as `BROAD_STRATEGIC`.
2.  Expand the query.
3.  Retrieve a larger candidate set.
4.  Rerank the larger set.
5.  Generate a broader answer using multiple relevant passages.
6.  Display classification, cache status, latency, and retrieved
    context.

------------------------------------------------------------------------

# 6. High-Level Architecture

``` text
                         ┌──────────────────────┐
                         │      Streamlit UI    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Query Orchestrator   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   Semantic Cache      │
                         └──────┬─────────┬─────┘
                                │         │
                         Cache Hit         │ Cache Miss
                                │         │
                                │         ▼
                                │  ┌───────────────┐
                                │  │ Query         │
                                │  │ Classifier    │
                                │  └───────┬───────┘
                                │          │
                                │    ┌─────┴─────┐
                                │    │           │
                                │    ▼           ▼
                                │ FACTUAL     BROAD
                                │ LOOKUP      STRATEGIC
                                │    │           │
                                │    ▼           ▼
                                │ Query         Query
                                │ Expansion     Expansion
                                │    │           │
                                │    ▼           ▼
                                │ Retrieval     Retrieval
                                │ small-k      large-k
                                │    │           │
                                │    ▼           ▼
                                │ Reranking     Reranking
                                │    │           │
                                │    └─────┬─────┘
                                │          ▼
                                │    Context Assembly
                                │          │
                                │          ▼
                                │      GPT-5.4 Mini
                                │          │
                                │          ▼
                                │    Answer + Sources
                                │          │
                                └──────────┬┘
                                           ▼
                                  Cache New Response
                                           │
                                           ▼
                                      Streamlit UI
```

------------------------------------------------------------------------

# 7. End-to-End Request Flow

For every user query:

1.  Accept the original query.
2.  Record request start time.
3.  Generate an embedding for semantic-cache lookup.
4.  Check the semantic cache.
5.  If a sufficiently similar, non-expired cached query exists:
    -   Return the cached response.
    -   Mark `cache_hit = true`.
    -   Record latency.
    -   Display cached result.
6.  If no cache entry exists:
    -   Mark `cache_hit = false`.
    -   Classify the query.
    -   Expand the query into three variations.
    -   Execute the appropriate retrieval pipeline.
    -   Rerank retrieved candidates.
    -   Assemble the final context.
    -   Generate the grounded answer.
    -   Store the result in semantic cache with a one-hour TTL.
    -   Record latency.
7.  Display the answer and pipeline metadata.

------------------------------------------------------------------------

# 8. Document Ingestion Pipeline

## 8.1 Source

The primary source is the Cisco Nexus 9000 Series troubleshooting guide
PDF.

The document should be treated as the authoritative knowledge source for
the POC.

## 8.2 PDF Loading

Use the LangChain PDF loader based on PyPDF:

-   `PyPDFLoader`

The ingestion code must:

1.  Load the PDF.
2.  Preserve page metadata.
3.  Preserve document/source metadata.
4.  Validate that pages were successfully extracted.
5.  Report extraction failures clearly.

## 8.3 Chunking

Use LangChain's:

-   `RecursiveCharacterTextSplitter`

Initial configuration:

``` text
chunk_size = 1024
chunk_overlap = 32
```

These values must be configurable rather than hard-coded throughout the
application.

## 8.4 Chunk Metadata

Each chunk should retain metadata including, where available:

``` text
source
page
document_name
chunk_id
```

Additional metadata may include:

``` text
ingestion_timestamp
section
```

if it can be obtained reliably.

------------------------------------------------------------------------

# 9. Embedding Layer

Use:

``` text
all-MiniLM-L6-v2
```

The embedding model must be suitable for local CPU execution.

Requirements:

-   Load locally.
-   Do not require a GPU.
-   Make the model configurable through application configuration.
-   Use the same embedding space consistently for document chunks and
    semantic-cache queries.
-   Avoid re-embedding documents on every application startup if
    persisted embeddings already exist.

------------------------------------------------------------------------

# 10. Vector Database

Use:

``` text
ChromaDB
```

through the appropriate LangChain integration.

Requirements:

-   Persist the vector store locally.
-   Store chunk embeddings and metadata.
-   Provide an ingestion/indexing process separate from normal question
    answering.
-   Avoid rebuilding the vector database every time the Streamlit
    application starts.
-   Make the persistence directory configurable.

Example conceptual configuration:

``` text
CHROMA_PERSIST_DIRECTORY=./data/chroma
COLLECTION_NAME=cisco_nexus_troubleshooting
```

------------------------------------------------------------------------

# 11. Query Expansion

Query expansion is a mandatory component.

## 11.1 Objective

Convert the original user query into exactly **three useful query
variations** designed to improve retrieval recall.

Example:

Original:

> "Why is the Nexus 9000 reporting error XYZ?"

Possible expansions:

1.  Error XYZ meaning and cause on Nexus 9000
2.  Nexus 9000 troubleshooting for XYZ error
3.  Conditions and recommended resolution for XYZ

The exact generated variations must depend on the original question.

## 11.2 Implementation

Use GPT-5.4 Mini to generate the query variations.

The query-expansion prompt must require structured output.

Recommended schema:

``` json
{
  "original_query": "...",
  "expanded_queries": [
    "...",
    "...",
    "..."
  ]
}
```

Requirements:

-   Exactly three variations.
-   Variations must preserve the original intent.
-   Avoid irrelevant broadening.
-   Avoid inventing technical terms not implied by the question.
-   Handle technical identifiers such as error codes, feature names,
    device names, and configuration keywords carefully.

------------------------------------------------------------------------

# 12. Query Classification / Triage

Before executing the RAG retrieval pipeline, classify the user query.

Use:

``` text
OpenAI GPT-5.4 Mini
```

## 12.1 Classification Categories

The initial POC has exactly two categories:

### `FACTUAL_LOOKUP`

Questions seeking:

-   A specific fact.
-   Error-code meaning.
-   Device/status code information.
-   Technical specification.
-   Specific configuration value.
-   Specific command or documented requirement.
-   Direct definition.

### `BROAD_STRATEGIC`

Questions requiring:

-   Multiple pieces of information.
-   Multiple troubleshooting considerations.
-   Causes and remediation.
-   Broader troubleshooting analysis.
-   Comparison or synthesis across multiple document sections.
-   A multi-step explanation.

## 12.2 Classification Output

Use structured output:

``` json
{
  "category": "FACTUAL_LOOKUP",
  "confidence": 0.94,
  "reason": "The question requests the meaning of a specific error code."
}
```

The allowed category values must be constrained to:

``` text
FACTUAL_LOOKUP
BROAD_STRATEGIC
```

## 12.3 Classification Fallback

If classification fails or produces invalid structured output:

1.  Log the failure.
2.  Apply a safe fallback.
3.  Default to `BROAD_STRATEGIC` when the query cannot confidently be
    treated as a narrow factual lookup.
4.  Continue processing rather than crashing the application.

The UI should indicate the final classification used.

------------------------------------------------------------------------

# 13. RAG Pipeline A --- Factual Lookup

## 13.1 Purpose

Optimize retrieval for questions requiring a precise answer from a small
amount of evidence.

## 13.2 Pipeline

``` text
Original Query
      ↓
Semantic Cache
      ↓
Classification
      ↓
Query Expansion
      ↓
Retrieve small candidate set
      ↓
Rerank
      ↓
Select top context
      ↓
GPT-5.4 Mini
      ↓
Grounded Answer
```

## 13.3 Retrieval

Initial target:

``` text
3–4 final retrieval candidates
```

Because query expansion generates three queries, the implementation may
retrieve a configurable number of candidates per expanded query and then
deduplicate before reranking.

The exact retrieval fan-out must be configurable.

## 13.4 Reranking

Reranking is mandatory.

Do **not** implement information compression as a substitute for
reranking.

The POC should use a reranker capable of scoring the relevance of
retrieved chunks against the original user query.

The reranker should produce:

``` text
chunk
relevance_score
rank
```

The final context should prioritize the highest-scoring chunks.

------------------------------------------------------------------------

# 14. RAG Pipeline B --- Broad / Strategic

## 14.1 Purpose

Handle questions requiring broader evidence across the troubleshooting
guide.

## 14.2 Pipeline

``` text
Original Query
      ↓
Semantic Cache
      ↓
Classification
      ↓
Query Expansion
      ↓
Retrieve larger candidate set
      ↓
Rerank
      ↓
Select highest-value context
      ↓
GPT-5.4 Mini
      ↓
Grounded Answer
```

## 14.3 Retrieval

Initial target:

``` text
20 candidate chunks
```

The exact number must be configurable.

The system should favor recall during initial retrieval and relevance
during reranking.

## 14.4 Reranking

All retrieved candidates should be scored and ranked before final
context assembly.

The final LLM context should contain only the highest-value evidence
after reranking.

------------------------------------------------------------------------

# 15. Retrieval Deduplication

Because three expanded queries are searched independently, the same
document chunk may appear multiple times.

The system must:

1.  Identify duplicate chunks using a stable `chunk_id`.
2.  Merge duplicate retrieval results.
3.  Preserve the strongest retrieval score.
4.  Avoid sending duplicate chunks to the reranker.
5.  Rerank the deduplicated candidate set.

------------------------------------------------------------------------

# 16. Reranking Requirements

Reranking is a core differentiating feature of this POC.

The system should distinguish:

``` text
Initial retrieval score
```

from:

``` text
Reranker relevance score
```

The UI should expose the final ranked context.

Example internal representation:

``` python
{
    "chunk_id": "...",
    "content": "...",
    "source": "...",
    "page": 123,
    "retrieval_score": 0.81,
    "rerank_score": 0.94,
    "final_rank": 1
}
```

The exact scoring implementation should be encapsulated behind a
reranker interface so it can be replaced later.

------------------------------------------------------------------------

# 17. Semantic Cache

Semantic caching is a mandatory feature.

## 17.1 Objective

If a new query is semantically similar to a recently answered query,
reuse the cached answer rather than executing the full RAG pipeline.

## 17.2 TTL

Initial TTL:

``` text
1 hour
```

The TTL must be configurable.

## 17.3 Cache Lookup

For each incoming question:

1.  Generate its embedding.
2.  Search the semantic cache.
3.  Compare against cached query embeddings.
4.  If similarity exceeds a configurable threshold and the entry has not
    expired:
    -   Return cached answer.
    -   Mark cache hit.
5.  Otherwise:
    -   Mark cache miss.
    -   Continue through normal RAG processing.

## 17.4 Cache Entry

A cache entry should contain:

``` text
query
query_embedding
answer
classification
retrieved_context
pipeline
created_at
expires_at
latency
```

The implementation should avoid storing unnecessary sensitive
information.

## 17.5 Cache Miss

After generating a new response:

1.  Create cache entry.
2.  Store the query embedding.
3.  Store the answer and required metadata.
4.  Set expiration to one hour from creation.

## 17.6 Cache Refresh

Entries should expire automatically based on TTL.

The implementation may use either:

-   TTL-aware lookup and lazy expiration, or
-   periodic cleanup.

A full cache rebuild is not required every hour; expired entries simply
must not be returned.

------------------------------------------------------------------------

# 18. Cache Metrics

Track at minimum:

``` text
cache_hits
cache_misses
cache_hit_rate
```

The application should also track:

``` text
cache_hit_latency
cache_miss_latency
```

This allows the demonstration to show the performance advantage of
caching.

------------------------------------------------------------------------

# 19. Latency Measurement

Measure end-to-end request latency.

At minimum:

``` text
request_start
request_end
total_latency_ms
```

For cache misses, optionally measure stages:

``` text
cache_lookup_ms
classification_ms
query_expansion_ms
retrieval_ms
reranking_ms
generation_ms
cache_write_ms
```

For cache hits:

``` text
cache_lookup_ms
total_latency_ms
```

The UI should prominently show total query latency.

------------------------------------------------------------------------

# 20. LLM Answer Generation

Use:

``` text
OpenAI GPT-5.4 Mini
```

The model must receive:

-   Original user query.
-   Selected reranked context.
-   Appropriate system instructions.
-   Source metadata where useful.

## 20.1 Grounding Rules

The generation prompt must instruct the model to:

1.  Answer using the supplied document context.
2.  Avoid unsupported claims.
3.  Clearly indicate when the retrieved context does not contain enough
    information.
4.  Never invent error-code meanings or configuration values.
5.  Preserve technical identifiers exactly.
6.  Prefer precise technical terminology.
7.  Cite or identify supporting source pages.

## 20.2 Answer Structure

For technical troubleshooting questions, prefer:

``` text
Direct answer

Explanation

Relevant troubleshooting/configuration details

Source references
```

The response should not be unnecessarily verbose for simple factual
questions.

------------------------------------------------------------------------

# 21. Source Attribution

Every generated answer should be traceable to retrieved document
content.

The UI must expose:

-   Source document.
-   Page number.
-   Retrieved chunk text.
-   Reranking score where appropriate.

The answer should include human-readable source references such as:

``` text
Source: Cisco Nexus 9000 Troubleshooting Guide, page 123
```

The application must never fabricate page numbers.

------------------------------------------------------------------------

# 22. Streamlit Frontend

The frontend must use:

``` text
Streamlit
```

## 22.1 Layout

Recommended layout:

``` text
┌──────────────────────────────────────────────────────────────┐
│ Cisco Nexus 9000 Troubleshooting Assistant                  │
├──────────────────┬───────────────────────────────────────────┤
│ Conversation     │ Query                                     │
│ History          │ [ Ask your question... ]                  │
│                  │                                           │
│ Q1 / A1          │ Answer                                    │
│ Q2 / A2          │ ----------------------------------------- │
│ Q3 / A3          │ Classification: FACTUAL_LOOKUP            │
│                  │ Pipeline: Factual Lookup                 │
│                  │ Cache: HIT / MISS                         │
│                  │ Latency: 1.24 s                           │
│                  │                                           │
│                  │ Retrieved Context                         │
│                  │ [expandable source chunks]                │
└──────────────────┴───────────────────────────────────────────┘
```

## 22.2 Main Query Area

Provide:

-   Text input.
-   Submit/action control.
-   Generated answer.
-   Classification.
-   Selected pipeline.
-   Cache status.
-   Latency.
-   Retrieved context.

## 22.3 Conversation History

The left sidebar should display previous questions and answers for the
current Streamlit session.

At minimum:

``` text
Question
Classification
Answer preview
```

Clicking/selecting a previous question should allow the user to inspect
the associated result if practical.

No external relational database is required for conversation history in
the initial POC.

Use Streamlit session state.

------------------------------------------------------------------------

# 23. UI Observability

For demonstration purposes, the UI must expose the following after every
query:

  Metric                  Required
  ----------------------- -------------
  Query classification    Yes
  Selected RAG pipeline   Yes
  Cache hit/miss          Yes
  Total latency           Yes
  Retrieved context       Yes
  Source/page             Yes
  Rerank scores           Recommended
  Expanded queries        Recommended

The UI should make the advanced RAG architecture visible without
overwhelming a normal end user.

Use expandable Streamlit sections for technical diagnostics.

------------------------------------------------------------------------

# 24. Configuration

All configurable values should be centralized.

Example configuration:

``` text
OPENAI_API_KEY

LLM_MODEL=gpt-5.4-mini

EMBEDDING_MODEL=all-MiniLM-L6-v2

CHUNK_SIZE=1024
CHUNK_OVERLAP=32

FACTUAL_RETRIEVAL_K=4
BROAD_RETRIEVAL_K=20

QUERY_EXPANSION_COUNT=3

CACHE_TTL_SECONDS=3600
CACHE_SIMILARITY_THRESHOLD=<configurable>

CHROMA_PERSIST_DIRECTORY=./data/chroma

SOURCE_DOCUMENT_PATH=./data/cisco_nexus_troubleshooting.pdf
```

Secrets must be loaded from `.env` or Streamlit secrets and must never
be committed to source control.

------------------------------------------------------------------------

# 25. Recommended Project Structure

``` text
cisco-nexus-rag/
│
├── app.py
├── README.md
├── PRD.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── source/
│   │   └── cisco_nexus_troubleshooting.pdf
│   └── chroma/
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pdf_loader.py
│   │
│   ├── chunking/
│   │   └── text_splitter.py
│   │
│   ├── embeddings/
│   │   └── embedding_model.py
│   │
│   ├── vectorstore/
│   │   └── chroma_store.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── query_expander.py
│   │   └── reranker.py
│   │
│   ├── routing/
│   │   └── classifier.py
│   │
│   ├── pipelines/
│   │   ├── factual_pipeline.py
│   │   └── broad_pipeline.py
│   │
│   ├── cache/
│   │   └── semantic_cache.py
│   │
│   ├── generation/
│   │   └── answer_generator.py
│   │
│   ├── orchestration/
│   │   └── query_orchestrator.py
│   │
│   └── utils/
│       ├── timing.py
│       └── logging.py
│
├── scripts/
│   └── ingest_document.py
│
└── tests/
    ├── test_classifier.py
    ├── test_query_expansion.py
    ├── test_retrieval.py
    ├── test_reranker.py
    ├── test_cache.py
    └── test_pipelines.py
```

GitHub Copilot should maintain separation of concerns according to this
structure.

------------------------------------------------------------------------

# 26. Core Interfaces

The implementation should use modular classes/functions rather than
placing the complete pipeline in `app.py`.

Conceptual interfaces:

``` python
class QueryClassifier:
    def classify(self, query: str) -> ClassificationResult:
        ...


class QueryExpander:
    def expand(self, query: str) -> list[str]:
        ...


class Retriever:
    def retrieve(
        self,
        queries: list[str],
        top_k: int
    ) -> list[RetrievedChunk]:
        ...


class Reranker:
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk]
    ) -> list[RankedChunk]:
        ...


class SemanticCache:
    def get(self, query: str) -> CacheResult | None:
        ...

    def set(self, query: str, response: CacheEntry) -> None:
        ...


class AnswerGenerator:
    def generate(
        self,
        query: str,
        context: list[RankedChunk]
    ) -> Answer:
        ...
```

The exact implementation may differ, but the responsibilities must
remain separated.

------------------------------------------------------------------------

# 27. Data Models

Use typed models where practical, preferably Pydantic/dataclasses
depending on the implementation approach.

Recommended models:

``` text
ClassificationResult
ExpandedQueryResult
RetrievedChunk
RankedChunk
CacheEntry
Answer
QueryResult
LatencyMetrics
```

A `QueryResult` should contain enough information for the UI:

``` python
{
    "query": "...",
    "answer": "...",
    "classification": "...",
    "classification_confidence": 0.94,
    "pipeline": "factual",
    "cache_hit": False,
    "latency_ms": 1240,
    "expanded_queries": [...],
    "retrieved_chunks": [...],
    "sources": [...]
}
```

------------------------------------------------------------------------

# 28. Error Handling

The application must fail gracefully.

Handle at minimum:

-   Missing PDF.
-   Invalid PDF.
-   PDF extraction failure.
-   Empty document.
-   Chroma initialization failure.
-   Missing OpenAI API key.
-   OpenAI API failure.
-   LLM timeout.
-   Invalid LLM structured output.
-   Query-expansion failure.
-   Classification failure.
-   Reranker failure.
-   Cache failure.
-   Empty retrieval results.

If retrieval produces no relevant context, do not generate an
unsupported answer.

Return a clear message such as:

> "I could not find sufficient supporting information in the Cisco Nexus
> 9000 troubleshooting guide to answer this confidently."

------------------------------------------------------------------------

# 29. Logging and Observability

Use structured application logging.

Log:

-   Request identifier.
-   Timestamp.
-   Query classification.
-   Selected pipeline.
-   Cache hit/miss.
-   Retrieval candidate count.
-   Reranked candidate count.
-   Total latency.
-   Component latency.
-   Errors.

Do not log:

-   API keys.
-   Secrets.
-   Sensitive credentials.
-   Unnecessary personal information.

------------------------------------------------------------------------

# 30. Testing Requirements

## 30.1 Unit Tests

Test:

-   PDF loading.
-   Chunking.
-   Query expansion.
-   Classification.
-   Retrieval.
-   Deduplication.
-   Reranking.
-   Cache hit.
-   Cache miss.
-   TTL expiration.
-   Latency measurement.
-   Pipeline routing.

## 30.2 Integration Tests

At minimum verify:

### Factual flow

``` text
Query
→ classifier
→ factual pipeline
→ query expansion
→ retrieval
→ reranking
→ generation
→ cache
```

### Broad flow

``` text
Query
→ classifier
→ broad pipeline
→ query expansion
→ retrieval
→ reranking
→ generation
→ cache
```

### Cache flow

``` text
First query → MISS → generate → cache

Similar query within TTL → HIT → return cached response
```

## 30.3 Regression Tests

Maintain a small fixed set of representative Cisco troubleshooting
questions.

Examples should include:

-   Error-code lookup.
-   Technical specification lookup.
-   Configuration-related lookup.
-   Narrow troubleshooting question.
-   Broad troubleshooting question.
-   Query with a technical acronym.
-   Query with a device/feature identifier.

------------------------------------------------------------------------

# 31. RAG Evaluation

The POC should include a lightweight evaluation dataset.

Create a set of representative questions with expected supporting
pages/chunks where feasible.

Evaluate:

### Retrieval

-   Recall@K.
-   Relevance of top-ranked chunks.
-   Reranking effectiveness.

### Generation

-   Groundedness.
-   Correctness.
-   Citation/source accuracy.
-   Unsupported-claim rate.

### Routing

-   Classification accuracy.
-   Appropriate pipeline selection.

### Cache

-   Cache hit rate for repeated/semantically similar queries.
-   Latency difference between hits and misses.

The evaluation framework does not need to be a sophisticated enterprise
evaluation platform for the initial POC.

------------------------------------------------------------------------

# 32. Security Requirements

-   API keys must never be hard-coded.
-   `.env` must be included in `.gitignore`.
-   Provide `.env.example` with placeholder values.
-   Do not expose API keys in Streamlit.
-   Validate user input.
-   Avoid executing user-provided commands.
-   Do not integrate with live Cisco devices.
-   Treat retrieved document text as untrusted content for
    prompt-injection purposes.

The generation prompt should explicitly instruct the model to treat
retrieved document content as reference material, not as instructions
that override system/application rules.

------------------------------------------------------------------------

# 33. Performance Requirements

The application is intended for local POC demonstration.

Performance goals:

-   Cached queries should be materially faster than cache misses.
-   Document ingestion should be performed offline rather than during
    every user query.
-   Embeddings should be reused.
-   Retrieval and reranking should be performed only when required.
-   Streamlit should remain responsive during normal usage.

No hard enterprise SLA is required for the POC.

------------------------------------------------------------------------

# 34. Dependency Principles

Use the requested ecosystem wherever practical:

-   Python
-   LangChain
-   PyPDF-based LangChain document loading
-   RecursiveCharacterTextSplitter
-   all-MiniLM-L6-v2
-   ChromaDB
-   OpenAI GPT-5.4 Mini
-   Streamlit

Avoid adding frameworks merely for architectural complexity.

Every additional dependency should have a clear purpose.

------------------------------------------------------------------------

# 35. GitHub Copilot Implementation Instructions

The PRD is intended to be consumed by GitHub Copilot.

Copilot should implement the project incrementally.

## Phase 1 --- Project Foundation

Implement:

-   Project structure.
-   Configuration.
-   `.env.example`.
-   Logging.
-   Requirements.
-   Basic Streamlit shell.

## Phase 2 --- Ingestion

Implement:

-   PDF loading.
-   Chunking.
-   Metadata.
-   Chroma persistence.
-   Ingestion script.

Verify that the document can be indexed successfully.

## Phase 3 --- Basic Retrieval

Implement:

-   Embeddings.
-   Chroma retrieval.
-   Basic retrieval tests.

## Phase 4 --- Query Expansion

Implement:

-   LLM-based expansion.
-   Structured output.
-   Exactly three variations.
-   Validation and fallback.

## Phase 5 --- Reranking

Implement:

-   Reranker abstraction.
-   Candidate deduplication.
-   Relevance scoring.
-   Ranked context selection.

## Phase 6 --- Query Classification

Implement:

-   `FACTUAL_LOOKUP`.
-   `BROAD_STRATEGIC`.
-   Structured output.
-   Fallback behavior.

## Phase 7 --- Two RAG Pipelines

Implement:

-   Factual pipeline with small retrieval target.
-   Broad pipeline with 20-candidate retrieval target.
-   Shared reusable components.

## Phase 8 --- Semantic Cache

Implement:

-   Query embeddings.
-   Semantic similarity lookup.
-   One-hour TTL.
-   Cache hit/miss tracking.
-   Cache metrics.

## Phase 9 --- Answer Generation

Implement:

-   Grounded answer generation.
-   Source attribution.
-   No-context fallback.

## Phase 10 --- Streamlit UX

Implement:

-   Question input.
-   Answer.
-   Classification.
-   Pipeline.
-   Cache status.
-   Latency.
-   Retrieved context.
-   Conversation history.
-   Expanded queries.

## Phase 11 --- Testing and Evaluation

Implement:

-   Unit tests.
-   Integration tests.
-   Representative evaluation dataset.
-   Retrieval/reranking measurements.

------------------------------------------------------------------------

# 36. Acceptance Criteria

The POC is complete when all of the following are true.

## Document

-   [ ] Cisco Nexus 9000 troubleshooting PDF can be loaded.
-   [ ] Pages are preserved as metadata.
-   [ ] Document is chunked using RecursiveCharacterTextSplitter.
-   [ ] Chunk size defaults to 1024.
-   [ ] Chunk overlap defaults to 32.
-   [ ] ChromaDB index persists locally.

## Query Processing

-   [ ] Every query is checked against semantic cache.
-   [ ] Cache uses semantic similarity rather than exact string
    matching.
-   [ ] Cache TTL defaults to one hour.
-   [ ] Cache hit/miss is displayed.
-   [ ] Query latency is displayed.
-   [ ] Cache misses result in a new RAG execution.
-   [ ] New responses are cached.

## Query Expansion

-   [ ] Original query is expanded into exactly three variations.
-   [ ] Expanded queries are used for retrieval.
-   [ ] Duplicate chunks are deduplicated.

## Classification

-   [ ] Query is classified using GPT-5.4 Mini.
-   [ ] Only `FACTUAL_LOOKUP` and `BROAD_STRATEGIC` are valid
    categories.
-   [ ] Classification is visible in the UI.
-   [ ] Classification determines the RAG pipeline.
-   [ ] Classification failures have a safe fallback.

## Factual Pipeline

-   [ ] Uses a smaller retrieval candidate set.
-   [ ] Query expansion is performed.
-   [ ] Retrieved candidates are reranked.
-   [ ] Final context is generated from highest-ranked evidence.
-   [ ] GPT-5.4 Mini generates the answer.

## Broad Pipeline

-   [ ] Uses approximately 20 retrieval candidates by default.
-   [ ] Query expansion is performed.
-   [ ] Candidates are reranked.
-   [ ] Highest-value evidence is passed to the LLM.
-   [ ] GPT-5.4 Mini generates the answer.

## Reranking

-   [ ] Reranking occurs after initial retrieval.
-   [ ] Reranking is not replaced by information compression.
-   [ ] Reranking scores are available for diagnostics.
-   [ ] Final context is ordered by relevance.

## UI

-   [ ] Streamlit application launches successfully.
-   [ ] User can submit questions.
-   [ ] Answer is displayed.
-   [ ] Classification is displayed.
-   [ ] Selected pipeline is displayed.
-   [ ] Cache hit/miss is displayed.
-   [ ] Latency is displayed.
-   [ ] Retrieved context is inspectable.
-   [ ] Source/page information is shown.
-   [ ] Conversation history appears in the left sidebar.

## Grounding

-   [ ] Answers are generated from retrieved document context.
-   [ ] Unsupported answers are explicitly identified.
-   [ ] Source pages are not fabricated.
-   [ ] Technical identifiers are preserved accurately.

------------------------------------------------------------------------

# 37. Demo Flow

The application should support a compelling client demonstration.

## Demo 1 --- Factual Lookup

Ask a specific technical question involving:

-   an error code,
-   device code,
-   technical specification,
-   configuration parameter,
-   or documented troubleshooting fact.

Show:

``` text
Classification → FACTUAL_LOOKUP
Pipeline → Factual RAG
Query expansion → 3 queries
Retrieval → small candidate set
Reranking → ranked evidence
Cache → MISS
Latency → displayed
Answer → grounded with source
```

Then ask a semantically similar question.

Show:

``` text
Classification → same/appropriate category
Cache → HIT
Latency → significantly lower
Answer → returned from semantic cache
```

## Demo 2 --- Broad Troubleshooting

Ask a question requiring several pieces of information.

Show:

``` text
Classification → BROAD_STRATEGIC
Pipeline → Broad RAG
Query expansion → 3 queries
Retrieval → approximately 20 candidates
Reranking → ranked evidence
Answer → synthesized from multiple sources
```

This demonstrates why the application does not use one fixed retrieval
strategy for every query.

------------------------------------------------------------------------

# 38. Important Architectural Principles

### Principle 1 --- Retrieval before generation

The LLM must not be treated as the primary knowledge source.

### Principle 2 --- Classification controls retrieval strategy

Different questions require different retrieval depth.

### Principle 3 --- Query expansion improves recall

Three query variations should broaden retrieval without changing the
user's intent.

### Principle 4 --- Reranking improves precision

Initial vector retrieval provides candidates; reranking determines which
candidates are most useful.

### Principle 5 --- Cache before expensive processing

Semantic cache lookup should happen before classification, expansion,
retrieval, reranking, and generation.

### Principle 6 --- Observability is part of the product

The POC should visibly demonstrate how the RAG system works.

### Principle 7 --- Modular architecture

Query classification, expansion, retrieval, reranking, caching, and
generation should be independently replaceable.

------------------------------------------------------------------------

# 39. Future Extensions

These are deliberately deferred but should be possible without major
architectural redesign:

-   More query categories.
-   Agentic troubleshooting workflows.
-   Multi-document RAG.
-   Hybrid keyword + vector search.
-   Advanced reranking models.
-   Query decomposition.
-   Parent-child retrieval.
-   Metadata filtering.
-   Conversation-aware retrieval.
-   Persistent user history.
-   Evaluation dashboards.
-   LangSmith/LangFuse-style tracing.
-   Enterprise authentication.
-   Cloud deployment.
-   Live Cisco device/tool integration.

------------------------------------------------------------------------

# 40. Final Definition of Done

The project is considered complete when a user can:

1.  Launch the Streamlit application.
2.  Ask a Cisco Nexus 9000 troubleshooting question.
3.  Have the question checked against the semantic cache.
4.  Have a cache miss classified into one of two query categories.
5.  Have the query expanded into three retrieval variations.
6.  Execute the appropriate retrieval strategy.
7.  Rerank retrieved chunks.
8.  Generate a grounded answer using GPT-5.4 Mini.
9.  View supporting document context and source pages.
10. See classification, pipeline, cache status, and latency.
11. Ask a similar question within one hour and receive a semantic-cache
    hit.
12. Ask a broad troubleshooting question and observe the broader RAG
    pipeline being selected.
13. Inspect the conversation history in the Streamlit sidebar.

The resulting application should clearly demonstrate that it is an
**advanced, routed RAG system**, rather than simply a PDF chatbot with
vector search.
