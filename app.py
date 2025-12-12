"""
app.py
------
Streamlit UI for the RAG Document Q&A system.
Upload PDFs/TXTs, ask questions, see answers with sources.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.evaluator import evaluate, print_eval_report

st.set_page_config(
    page_title = "RAG Document Q&A",
    page_icon  = "📚",
    layout     = "wide",
)

st.markdown("""
<style>
.main { background: #0f1117; }
.stTextInput > div > div { background: #1a1d2e; }
.answer-box { background: #1a1d2e; border-left: 4px solid #6c63ff;
              padding: 16px; border-radius: 8px; margin: 12px 0; }
.source-chip { background: #252847; border-radius: 12px; padding: 3px 12px;
               font-size: 0.82rem; display: inline-block; margin: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Q&A")
    st.caption("Retrieval-Augmented Generation with Claude")
    st.divider()

    api_key = st.text_input("Anthropic API Key", type="password",
                             placeholder="sk-ant-...")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    uploaded = st.file_uploader(
        "Upload Documents (PDF / TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    ingest_btn = st.button("📥 Ingest Documents", type="primary",
                           use_container_width=True)

    st.divider()
    show_eval = st.checkbox("Show answer evaluation scores")
    st.caption("Model: claude-sonnet-4-20250514\n"
               "Embeddings: all-MiniLM-L6-v2\n"
               "Vector DB: ChromaDB (local)")

# ── Session state ─────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "history" not in st.session_state:
    st.session_state.history = []

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📚 RAG Document Q&A System")
st.markdown("Upload documents, then ask questions grounded in their content.")

# ── Ingest ────────────────────────────────────────────────────────────────────
if ingest_btn and uploaded:
    with tempfile.TemporaryDirectory() as tmp_dir:
        for f in uploaded:
            path = Path(tmp_dir) / f.name
            path.write_bytes(f.read())

        with st.spinner(f"Ingesting {len(uploaded)} document(s)..."):
            try:
                n = st.session_state.pipeline.ingest_documents(tmp_dir)
                st.success(f"✅ Ingested {n} chunks from {len(uploaded)} document(s)")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

# ── Stats ─────────────────────────────────────────────────────────────────────
stats = st.session_state.pipeline.get_stats()
if stats.get("chunks_stored"):
    col1, col2 = st.columns(2)
    col1.metric("Chunks in Vector Store", stats["chunks_stored"])
    col2.metric("Storage", stats["persist_dir"])

st.divider()

# ── Query ─────────────────────────────────────────────────────────────────────
st.subheader("Ask a Question")
question = st.text_input("", placeholder="What does the document say about...")

if st.button("🔍 Ask", type="primary") and question:
    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar.")
    elif stats.get("status") == "not initialized" and not stats.get("chunks_stored"):
        st.warning("Please upload and ingest documents first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                result = st.session_state.pipeline.query(question)
                st.session_state.history.append({
                    "question": question,
                    "result":   result,
                })
            except Exception as e:
                st.error(f"Query failed: {e}")

# ── Display history ───────────────────────────────────────────────────────────
for item in reversed(st.session_state.history):
    q      = item["question"]
    result = item["result"]

    st.markdown(f"**Q: {q}**")
    st.markdown(f'<div class="answer-box">{result["answer"]}</div>',
                unsafe_allow_html=True)

    if result["sources"]:
        st.markdown("**Sources:** " + " ".join(
            f'<span class="source-chip">📄 {Path(s).name}</span>'
            for s in result["sources"]
        ), unsafe_allow_html=True)

    if show_eval and api_key:
        with st.expander("📊 Evaluation Scores"):
            with st.spinner("Evaluating..."):
                try:
                    ev = evaluate(q, result["answer"], result["source_documents"])
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Relevance",    f"{ev.relevance_score}/5")
                    c2.metric("Faithfulness", f"{ev.faithfulness_score}/5")
                    c3.metric("Ctx Precision",f"{ev.context_precision}/5")
                    c4.metric("Overall",      f"{ev.overall_score}/5")
                except Exception as e:
                    st.warning(f"Evaluation error: {e}")

    st.divider()
