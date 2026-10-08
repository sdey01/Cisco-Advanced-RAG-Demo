import logging
from pathlib import Path

import streamlit as st

from src.config import Settings
from src.embeddings import create_embeddings
from src.orchestrator import QueryOrchestrator
from src.vectorstore import open_vectorstore


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = Settings.from_env()

st.set_page_config(
    page_title="Nexus Troubleshooting Assistant",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root { --ink: #192421; --muted: #65736f; --line: #dce5e1; --mint: #d8eee4; --green: #087e67; --amber: #c26b23; }
    .stApp { background: #f5f8f6; color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #edf3ef; border-right: 1px solid var(--line); }
    .block-container { padding-top: 2rem; max-width: 1180px; }
    .eyebrow { color: var(--green); font-size: .76rem; font-weight: 750; letter-spacing: .11em; text-transform: uppercase; }
    h1 { color: var(--ink); font-size: 2.2rem !important; line-height: 1.12 !important; margin: .3rem 0 .5rem !important; }
    .subhead { color: var(--muted); margin-bottom: 1.5rem; }
    div[data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 4px; padding: .8rem 1rem; }
    div[data-testid="stForm"] { background: #fff; border: 1px solid var(--line); border-radius: 4px; padding: 1rem 1rem .6rem; }
    .stButton button[kind="primary"], .stFormSubmitButton button { background: var(--green); border-color: var(--green); }
    [data-testid="stExpander"] { border-color: var(--line); background: #fff; border-radius: 4px; }
    .history-note { color: var(--muted); font-size: .86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading local CPU embedding and retrieval models...")
def get_orchestrator(
    embedding_model: str, persist_directory: str, collection_name: str
):
    embeddings = create_embeddings(embedding_model)
    vectorstore = open_vectorstore(
        embeddings, Path(persist_directory), collection_name
    )
    if vectorstore._collection.count() == 0:
        raise RuntimeError("The document index is empty. Run: python scripts/ingest_document.py")
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to the local .env file and restart.")
    return QueryOrchestrator(settings, embeddings, vectorstore)


st.markdown('<div class="eyebrow">Cisco Nexus 9000 · Troubleshooting Guide</div>', unsafe_allow_html=True)
st.title("Troubleshooting Assistant")
st.markdown(
    '<div class="subhead">Document-grounded answers with routed retrieval, reranking, and semantic caching.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Session history")
    history = st.session_state.setdefault("history", [])
    selected_history = st.radio(
        "Previous questions",
        options=list(range(len(history))),
        format_func=lambda index: (
            history[index]["query"]
            if len(history[index]["query"]) <= 48
            else history[index]["query"][:45].rstrip() + "..."
        ),
        index=None,
        label_visibility="collapsed",
        key="history_selection",
    )
    if selected_history is not None:
        st.session_state["current_result"] = history[selected_history]
    if history and st.button("Clear session history", icon=":material/delete:"):
        st.session_state["history"] = []
        st.session_state.pop("current_result", None)
        st.rerun()
    cache_metrics = st.session_state.get("cache_metrics")
    if cache_metrics:
        st.markdown("### Cache metrics")
        st.caption(
            f"Hits: {cache_metrics['cache_hits']} · Misses: {cache_metrics['cache_misses']} · "
            f"Hit rate: {cache_metrics['cache_hit_rate']:.0%}"
        )
        st.caption(
            f"Avg hit: {cache_metrics.get('cache_hit_latency_ms', 0):.0f} ms · "
            f"Avg miss: {cache_metrics.get('cache_miss_latency_ms', 0):.0f} ms"
        )
    st.markdown('<div class="history-note">History is stored only in this browser session.</div>', unsafe_allow_html=True)

with st.form("question_form", clear_on_submit=False):
    query = st.text_area(
        "Question",
        placeholder="Ask about a Nexus 9000 error, feature, configuration, or troubleshooting procedure...",
        height=90,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "Search the guide", type="primary", icon=":material/search:"
    )

if submitted:
    if not query.strip():
        st.warning("Enter a troubleshooting question to continue.")
    else:
        try:
            orchestrator = get_orchestrator(
                settings.embedding_model,
                str(settings.chroma_persist_directory),
                settings.collection_name,
            )
            with st.spinner("Classifying, retrieving, reranking, and checking the guide..."):
                result = orchestrator.ask(query)
            result_data = vars(result)
            st.session_state["current_result"] = result_data
            history = st.session_state["history"]
            history.insert(0, result_data)
            st.session_state["history"] = history[:20]
            st.session_state["cache_metrics"] = orchestrator.cache.metrics()
            st.rerun()
        except Exception as exc:
            logging.exception("Query request failed")
            st.error(str(exc))

result = st.session_state.get("current_result")
if result:
    st.markdown("### Answer")
    with st.container(border=True):
        st.markdown(result["answer"])
    st.write("")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Classification", result["classification"])
    metric_columns[1].metric("Selected pipeline", result["pipeline"])
    metric_columns[2].metric("Semantic cache", "HIT" if result["cache_hit"] else "MISS")
    metric_columns[3].metric("Total latency", f"{result['latency_ms']:.0f} ms")
    st.caption(f"Classifier confidence: {result['classification_confidence']:.0%}")
    with st.expander(f"Retrieved and reranked context ({len(result['retrieved_chunks'])})", expanded=True):
        if result["retrieved_chunks"]:
            for chunk in result["retrieved_chunks"]:
                metadata = chunk["metadata"]
                label = (
                    f"#{chunk['final_rank']} · {metadata.get('document_name', metadata.get('source', 'Guide'))} "
                    f"· page {metadata.get('page', 'unknown')} · rerank {chunk.get('rerank_score', 0):.3f} "
                    f"· retrieval {chunk.get('retrieval_score', 0):.3f}"
                )
                with st.expander(label):
                    st.write(chunk["content"])
        else:
            st.info("No supporting passages were retrieved.")
    with st.expander("Query expansion", expanded=False):
        for index, expanded_query in enumerate(result["expanded_queries"], start=1):
            st.write(f"{index}. {expanded_query}")
elif not submitted:
    st.info("Ask a question to inspect its answer, routing decision, cache result, and supporting passages.")
