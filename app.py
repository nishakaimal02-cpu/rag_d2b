# app.py
# PURPOSE: Streamlit UI — chat interface with search mode toggle
# Connects ingest.py (index building) and retriever.py (search + answer)

import streamlit as st
from ingest import ingest_docs
from retriever import load_vectorstore, answer_question
import os
import hmac

# ── Password protection ───────────────────────────────────────
def check_password():
    """Block access until correct password is entered."""
    if "password_correct" not in st.session_state:
        st.text_input("Enter password", type="password", key="password")
        if st.button("Login"):
            if hmac.compare_digest(
                st.secrets["APP_PASSWORD"],
                st.session_state["password"]
            ):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("Wrong password")
        st.stop()

check_password()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


tab1, tab2 = st.tabs(["💬 Chat", "🗺️ Embedding Map"])

with tab1:
    st.title("🔬 AI Research Assistant")
    st.caption("Semantic search, keyword search, and hybrid — over real AI research papers.")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Search mode toggle — key feature today
    search_mode = st.radio(
        "Search mode",
        options=["hybrid", "semantic", "bm25"],
        format_func=lambda x: {
            "hybrid": "🔀 Hybrid (recommended)",
            "semantic": "🧠 Semantic only",
            "bm25": "🔤 Keyword only (BM25)"
        }[x],
        help="Hybrid combines semantic and keyword search. Try all three on the same question to compare."
    )
    
    st.divider()
    
    # Number of chunks to retrieve
    k_value = st.slider(
        "Chunks to retrieve (k)",
        min_value=1, max_value=6, value=3,
        help="How many document chunks to retrieve per query"
    )
    
    st.divider()
    
    # Index status
    if os.path.exists("chroma_db"):
        st.success("✅ Index loaded")
    else:
        st.warning("⚠️ No index found. Click Build Index.")
    
    # Rebuild index button
    if st.button("🔄 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Building index from docs folder..."):
            ingest_docs()
            # Clear cached vectorstore so it reloads fresh
            st.session_state.pop("vectorstore", None)
        st.success("Index built!")
    
    st.divider()
    
    # Loaded papers reference
    st.markdown("**📄 Knowledge base:**")
    if os.path.exists("docs"):
        files = [f for f in os.listdir("docs") if f.endswith((".txt", ".pdf"))]
        for f in files:
            st.markdown(f"- {f}")
    
    st.divider()
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Guard — stop if no index exists ──────────────────────────
if not os.path.exists("chroma_db"):
    st.warning("👈 Click 'Build / Rebuild Index' in the sidebar to get started.")
    st.stop()

# ── Load vectorstore (cached in session state) ────────────────
# Only loads once per session — not on every message
if "vectorstore" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.vectorstore = load_vectorstore()

# ── Chat history ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay all previous messages on page load
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show confidence warning if it was flagged
        if msg.get("confidence") == "low":
            st.warning("⚠️ Low confidence — the retrieved chunks may not fully answer this question.")
        
        # Show sources if available
        if msg.get("sources"):
            with st.expander("📎 Sources retrieved"):
                for s in msg["sources"]:
                    st.code(s)

# ── Chat input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask about transformers, GPT-4, or RAG..."):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner(f"Searching with {search_mode} mode..."):
            result = answer_question(
                prompt,
                st.session_state.vectorstore,
                search_mode=search_mode,
                k=k_value
            )
        
        # Show answer
        st.markdown(result["answer"])
        
        # Show confidence warning if low
        if result["confidence"] == "low":
            st.warning("⚠️ Low confidence — the retrieved chunks may not fully answer this question.")
        
        # Show sources with page numbers and scores
        if result["sources"]:
            with st.expander("📎 Sources retrieved"):
                for s in result["sources"]:
                    st.code(s)
    
    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"]
    })
    
    
with tab2:
    st.title("🗺️ Embedding Map")
    st.caption("Each dot is one chunk from your knowledge base. Similar content clusters together.")
    
    if st.button("Generate UMAP visualisation", use_container_width=True):
        with st.spinner("Running UMAP — this takes 30-60 seconds..."):
            from umap_viz import get_embeddings_from_chroma, reduce_with_umap, build_dataframe, plot_embeddings
            embeddings, documents, metadatas = get_embeddings_from_chroma()
            reduced = reduce_with_umap(embeddings)
            df = build_dataframe(reduced, documents, metadatas)
            fig = plot_embeddings(df)
            st.plotly_chart(fig, use_container_width=True)