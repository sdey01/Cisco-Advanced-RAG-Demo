# Cisco Nexus 9000 Troubleshooting Assistant

Local Streamlit RAG proof of concept for the Cisco Nexus 9000 NX-OS troubleshooting guide. It uses GPT-5.4 Mini for query classification, query expansion, and grounded answers; ChromaDB and all-MiniLM-L6-v2 for retrieval; a CPU cross-encoder for reranking; and a one-hour semantic cache.

## Setup

Python 3.12 is recommended. Create and activate the environment, then install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. The provided troubleshooting PDF is selected by default. Index the document once:

```powershell
python -m scripts.ingest_document
```

Start the UI:

```powershell
streamlit run app.py
```

Models are downloaded on first use and run locally on CPU; the OpenAI API is used for classification, query expansion, and answer generation. ChromaDB persists under `data/chroma`, and semantic cache entries persist under `data/semantic_cache.json`.

## Tests

```powershell
pytest -q
```
