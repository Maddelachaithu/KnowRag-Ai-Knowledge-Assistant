"""
KnowRAG — AI-Powered Knowledge Assistant
-----------------------------------------
Streamlit web application providing a modern, professional, conversational UI
for querying the KnowRAG knowledge base with strict source-grounding,
greetings & casual chat, slash commands, dynamic multi-document PDF/TXT uploads,
and incremental ChromaDB indexing.

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 standard output encoding on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
from src.conversation import process_user_query
from src.ingestion import process_uploaded_file
from src.llm import load_environment
from src.rag_pipeline import create_rag_pipeline
from src.vector_store import (
    clear_uploaded_documents,
    get_knowledge_base_stats,
    index_uploaded_document,
)

# Configure Streamlit page
st.set_page_config(
    page_title="KnowRAG — AI-Powered Knowledge Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Comprehensive Custom CSS for a professional, clean, modern aesthetic
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Layout Reset */
    html, body, [data-testid="stAppViewContainer"], .main, .block-container {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
        box-sizing: border-box !important;
        color: #1E293B;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        max-width: 860px !important;
    }

    /* Prevent text overflow across all markdown and chat containers */
    .stMarkdown, .stChatMessage, .stChatMessageContent, p, span, div, li {
        overflow-wrap: break-word !important;
        word-break: break-word !important;
        line-height: 1.65 !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', Consolas, Monaco, monospace !important;
        white-space: pre-wrap !important;
        word-break: break-all !important;
        max-width: 100% !important;
        overflow-x: auto !important;
        border-radius: 6px !important;
    }

    /* Hero Header Container */
    .knowrag-hero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
        color: #FFFFFF;
    }

    .hero-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
        margin-bottom: 0.85rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #E2E8F0;
        letter-spacing: 0.02em;
    }

    .hero-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(37, 99, 235, 0.25);
        border: 1px solid rgba(96, 165, 250, 0.4);
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #93C5FD;
    }

    .hero-title-container {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0.4rem;
    }

    .hero-icon-box {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 12px;
        font-size: 1.6rem;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 600;
        color: #60A5FA;
        margin: 0.15rem 0 0 0;
    }

    .hero-description {
        font-size: 0.95rem;
        color: #CBD5E1;
        margin: 0.75rem 0 0 0;
        line-height: 1.5;
    }

    /* Sidebar Custom Styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    .sidebar-brand-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.2rem;
    }

    .sidebar-brand-sub {
        font-size: 0.82rem;
        font-weight: 500;
        color: #2563EB;
        margin-bottom: 0.5rem;
    }

    .sidebar-status-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .sidebar-section-header {
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin: 1.2rem 0 0.6rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .sidebar-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.85rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    /* Stat Badges */
    .stat-row {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.6rem;
    }

    .stat-pill {
        flex: 1;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 0.5rem 0.6rem;
        text-align: center;
    }

    .stat-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E293B;
        line-height: 1.1;
    }

    .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        margin-top: 0.15rem;
    }

    /* Welcome / Empty State Screen */
    .welcome-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.75rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }

    .welcome-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .welcome-badge {
        display: inline-block;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1D4ED8;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
    }

    .welcome-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 0.35rem 0;
    }

    .welcome-desc {
        font-size: 0.95rem;
        color: #64748B;
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.55;
    }

    .suggested-section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #334155;
        margin: 1.25rem 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Source Citation Cards */
    .citation-card {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 3px solid #2563EB;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin-top: 0.75rem;
    }

    .citation-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #1E293B;
        display: flex;
        align-items: center;
        gap: 0.35rem;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .citation-chips-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }

    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 0.25rem 0.6rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: #334155;
        word-break: break-all;
    }

    .no-source-card {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 3px solid #D97706;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin-top: 0.65rem;
        font-size: 0.85rem;
        color: #92400E;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Buttons Modern Styling */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton > button:hover {
        border-color: #2563EB !important;
        color: #2563EB !important;
        background-color: #EFF6FF !important;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.12) !important;
    }

    /* Chat Input Styling */
    [data-testid="stChatInput"] {
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        font-family: inherit !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }

    /* Chat Messages styling */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 0.6rem 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """
    Initialize and cache the RAG pipeline (Vector Index + Groq LLM).
    Reuses existing ChromaDB index and Groq configuration.
    """
    try:
        return create_rag_pipeline(temperature=0.0)
    except ValueError as e:
        if "GROQ_API_KEY" in str(e):
            return None, None
        raise e


# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize pipeline
pipeline_res = get_pipeline()
vector_index = pipeline_res[0] if pipeline_res else None
groq_llm = pipeline_res[1] if pipeline_res else None

# Check API key status
has_api_key = load_environment()


# -------------------------------------------------------------
# SIDEBAR NAVIGATION & KNOWLEDGE BASE MANAGEMENT
# -------------------------------------------------------------
with st.sidebar:
    # Sidebar Brand Card
    st.markdown(
        """
        <div class="sidebar-brand-card">
            <div class="sidebar-brand-title">📚 KnowRAG</div>
            <div class="sidebar-brand-sub">AI-Powered Knowledge Assistant</div>
            <div class="sidebar-status-tag">🟢 System Active</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Quick Actions Section
    st.markdown('<div class="sidebar-section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("❓ Help", key="btn_quick_help", use_container_width=True):
            st.session_state.pending_query = "/help"
            st.rerun()
        if st.button("🧹 Clear Chat", key="btn_quick_clear", use_container_width=True):
            st.session_state.messages = []
            if "pending_query" in st.session_state:
                del st.session_state.pending_query
            st.session_state.messages.append({
                "role": "assistant",
                "content": "🧹 Chat history cleared. How can I help you?",
                "sources": [],
                "source_details": [],
            })
            st.rerun()

    with col2:
        if st.button("📄 Documents", key="btn_quick_docs", use_container_width=True):
            st.session_state.pending_query = "/show documents"
            st.rerun()
        if st.button("ℹ️ About", key="btn_quick_about", use_container_width=True):
            st.session_state.pending_query = "/about"
            st.rerun()

    # 2. Upload Documents Section
    st.markdown('<div class="sidebar-section-header">📤 Upload Documents</div>', unsafe_allow_html=True)
    st.caption("Upload PDF or TXT files to expand the KnowRAG knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="doc_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files and vector_index is not None:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            try:
                with st.status(f"Processing `{filename}`...", expanded=False) as status_box:
                    status_box.write("🔄 Processing document...")
                    status_box.write("📖 Extracting content...")
                    documents, doc_id = process_uploaded_file(
                        filename=filename,
                        file_bytes=file_bytes,
                        save_to_disk=True,
                    )
                    status_box.write("🧩 Creating chunks...")
                    status_box.write("🧠 Generating embeddings...")
                    status_box.write("💾 Updating knowledge base...")

                    index_result = index_uploaded_document(
                        documents=documents,
                        index=vector_index,
                    )
                    status_box.update(label=f"Processed `{filename}`", state="complete")

                status = index_result.get("status")

                if status == "success":
                    chunks_indexed = index_result.get("chunks_indexed", len(documents))
                    st.success(
                        f"✅ **Document uploaded successfully!**\n\n"
                        f"📄 **File:** `{filename}` ({chunks_indexed} chunks)\n\n"
                        f"The document has been indexed. You can now ask questions about it."
                    )
                elif status == "duplicate":
                    st.info(
                        f"ℹ️ **This document is already in the knowledge base.**\n\n"
                        f"📄 `{filename}`"
                    )
                else:
                    st.error(
                        f"❌ **I couldn't process this document.** Please make sure it is a valid PDF or TXT file.\n\n"
                        f"Details: {index_result.get('message')}"
                    )

            except Exception:
                st.error(
                    f"❌ **I couldn't process this document.** Please make sure it is a valid PDF or TXT file."
                )

    # 3. Knowledge Base Overview Section
    st.markdown('<div class="sidebar-section-header">📚 Knowledge Base</div>', unsafe_allow_html=True)
    try:
        kb_stats = get_knowledge_base_stats()
        doc_count = kb_stats.get("total_documents", 0)
        chunk_count = kb_stats.get("total_chunks", 0)
        uploaded_docs = kb_stats.get("uploaded_documents", [])

        st.markdown(
            f"""
            <div class="stat-row">
                <div class="stat-pill">
                    <div class="stat-value">📚 {doc_count}</div>
                    <div class="stat-label">Documents</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-value">🧩 {chunk_count}</div>
                    <div class="stat-label">Chunks</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if uploaded_docs:
            st.markdown("**Uploaded Files:**")
            for u_doc in uploaded_docs:
                st.markdown(f"• `{u_doc['filename']}` ({u_doc['chunks']} chunks)")

            if st.button("🗑️ Clear Uploaded Documents", key="btn_clear_uploads", use_container_width=True):
                deleted_chunks = clear_uploaded_documents(index=vector_index)
                st.success(f"Cleared {deleted_chunks} uploaded chunks. Baseline intact.")
                st.rerun()

    except Exception as e:
        st.caption(f"Knowledge base status unavailable: {e}")

    # 4. Sample Questions Sidebar Section
    st.markdown('<div class="sidebar-section-header">💡 Sample Questions</div>', unsafe_allow_html=True)
    sidebar_sample_questions = [
        "What services does the university library provide?",
        "What academic programs are offered?",
        "What is TechNova University?",
        "What rules should students follow on campus?",
        "What support is available to students?",
        "Give me some cybersecurity project ideas",
    ]
    for sq in sidebar_sample_questions:
        if st.button(sq, key=f"sb_btn_{sq}", use_container_width=True):
            st.session_state.pending_query = sq
            st.rerun()

    # 5. About & Architecture Guide
    st.markdown('<div class="sidebar-section-header">ℹ️ About KnowRAG</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-card">
            <div style="font-size: 0.82rem; color: #475569; line-height: 1.5; margin-bottom: 0.5rem;">
                KnowRAG is an AI-powered knowledge assistant that uses <strong>Retrieval-Augmented Generation (RAG)</strong> to provide source-grounded answers from the university knowledge base.
            </div>
            <div style="font-size: 0.75rem; font-weight: 700; color: #1E293B; margin-bottom: 0.3rem;">RAG PIPELINE:</div>
            <div style="font-size: 0.74rem; color: #64748B; line-height: 1.4;">
                📄 Documents<br>
                ↳ 📖 Text Extraction<br>
                ↳ 🧩 Chunking (512 tokens)<br>
                ↳ 🧠 BGE Embeddings (384-d)<br>
                ↳ 💾 ChromaDB Vector Store<br>
                ↳ 🔎 Similarity Retrieval<br>
                ↳ 🤖 Groq LLM<br>
                ↳ 🎯 Grounded Answer
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------
# MAIN APPLICATION HERO HEADER
# -------------------------------------------------------------
st.markdown(
    """
    <div class="knowrag-hero-header">
        <div class="hero-badge-row">
            <span class="hero-badge">🎓 TechNova University</span>
            <span class="hero-status-pill">🟢 ChromaDB Active</span>
            <span class="hero-status-pill">⚡ Groq LLM Connected</span>
            <span class="hero-status-pill">🛡️ Strict Grounding</span>
        </div>
        <div class="hero-title-container">
            <div class="hero-icon-box">📚</div>
            <div>
                <h1 class="hero-title">KnowRAG</h1>
                <div class="hero-subtitle">AI-Powered Knowledge Assistant</div>
            </div>
        </div>
        <p class="hero-description">
            Ask questions, explore your knowledge base, and get source-grounded answers.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render API Key warning if not configured
if not has_api_key:
    st.warning(
        "⚠️ **GROQ_API_KEY is not configured.** Please add your Groq API key to the `.env` file or deployment secrets."
    )


# -------------------------------------------------------------
# WELCOME SCREEN / EMPTY STATE (When no chat history exists)
# -------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-header">
                <span class="welcome-badge">🚀 University Knowledge Base</span>
                <h2 class="welcome-title">Welcome to KnowRAG</h2>
                <p class="welcome-desc">
                    Ask questions about your university knowledge base and receive accurate, source-grounded answers.
                    Select a sample question below or type your question in the chat input.
                </p>
            </div>
            <div class="suggested-section-title">✨ Recommended Questions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    starter_col1, starter_col2 = st.columns(2)
    with starter_col1:
        if st.button("📚 What services does the university library provide?", key="start_1", use_container_width=True):
            st.session_state.pending_query = "What services does the university library provide?"
            st.rerun()
        if st.button("🏫 What is TechNova University?", key="start_2", use_container_width=True):
            st.session_state.pending_query = "What is TechNova University?"
            st.rerun()
        if st.button("🛡️ What support is available to students?", key="start_3", use_container_width=True):
            st.session_state.pending_query = "What support is available to students?"
            st.rerun()

    with starter_col2:
        if st.button("🎓 What academic programs are offered?", key="start_4", use_container_width=True):
            st.session_state.pending_query = "What academic programs are offered by the university?"
            st.rerun()
        if st.button("📋 What rules should students follow on campus?", key="start_5", use_container_width=True):
            st.session_state.pending_query = "What rules should students follow on campus?"
            st.rerun()
        if st.button("💡 Give me some cybersecurity project ideas", key="start_6", use_container_width=True):
            st.session_state.pending_query = "Give me some cybersecurity project ideas"
            st.rerun()


# -------------------------------------------------------------
# CHAT CONVERSATION STREAM
# -------------------------------------------------------------
for msg in st.session_state.messages:
    avatar_icon = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
        
        sources_to_show = msg.get("source_details") or msg.get("sources", [])
        if sources_to_show:
            chips_html = "".join([f'<span class="source-chip">📄 {src}</span>' for src in sources_to_show])
            st.markdown(
                f"""
                <div class="citation-card">
                    <div class="citation-title">📄 Sources</div>
                    <div class="citation-chips-container">{chips_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif msg["role"] == "assistant" and msg.get("is_out_of_kb"):
            st.markdown(
                """
                <div class="no-source-card">
                    <span>ℹ️ No relevant information was found in the knowledge base.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -------------------------------------------------------------
# CHAT INPUT & EXECUTION
# -------------------------------------------------------------
query_input = st.chat_input("Ask a question, say hello, or type /help...")

# Check if a pending query was triggered by button click or text input
if "pending_query" in st.session_state and st.session_state.pending_query:
    active_query = st.session_state.pending_query
    del st.session_state.pending_query
else:
    active_query = query_input

# Handle query processing
if active_query:
    # 1. Append and render user message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_query)

    # 2. Process query with conversational routing
    with st.chat_message("assistant", avatar="🤖"):
        if not has_api_key and not active_query.strip().startswith("/"):
            error_msg = "⚠️ **GROQ_API_KEY is not configured.** Please add your Groq API key to the `.env` file or deployment secrets."
            st.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "source_details": [],
            })
        else:
            with st.spinner("Analyzing knowledge base & generating answer..."):
                try:
                    result = process_user_query(
                        query=active_query,
                        index=vector_index,
                        llm=groq_llm,
                    )

                    # Check for clear command action
                    if result.get("action") == "clear_chat":
                        st.session_state.messages = []
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": [],
                            "source_details": [],
                        })
                        st.rerun()

                    answer_text = result["answer"]
                    sources = result.get("sources", [])
                    source_details = result.get("source_details", sources)
                    intent = result.get("intent", "rag")

                    is_out_of_kb = (
                        intent == "rag"
                        and (
                            not sources
                            or "not available in the knowledge base" in answer_text.lower()
                        )
                    )

                    # Render response text
                    st.markdown(answer_text)

                    # Render source citations (only for grounded RAG answers with citations)
                    if source_details:
                        chips_html = "".join([f'<span class="source-chip">📄 {src}</span>' for src in source_details])
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-title">📄 Sources</div>
                                <div class="citation-chips-container">{chips_html}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif is_out_of_kb:
                        st.markdown(
                            """
                            <div class="no-source-card">
                                <span>ℹ️ No relevant information was found in the knowledge base.</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Save assistant response to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "source_details": source_details,
                        "is_out_of_kb": is_out_of_kb,
                        "intent": intent,
                    })

                except Exception as e:
                    if "GROQ_API_KEY" in str(e):
                        error_msg = "⚠️ **GROQ_API_KEY is not configured.** Please add your Groq API key to the `.env` file or deployment secrets."
                    else:
                        error_msg = f"An error occurred while processing your request: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "source_details": [],
                        "is_out_of_kb": True,
                    })
