"""
KnowRAG — AI-Powered Knowledge Assistant
-----------------------------------------
Streamlit web application providing a conversational user interface
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

# Custom styling for a clean, professional aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.1rem;
    }
    .main-subtitle {
        font-size: 1.15rem;
        font-weight: 500;
        color: #3B82F6;
        margin-bottom: 0.4rem;
    }
    .main-description {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .source-tag {
        display: inline-block;
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 2px 4px 2px 0;
        font-size: 0.85rem;
        color: #334155;
    }
    .no-source-notice {
        color: #94A3B8;
        font-style: italic;
        font-size: 0.88rem;
    }
    .command-badge {
        display: inline-block;
        background-color: #E2E8F0;
        color: #0F172A;
        font-weight: 600;
        padding: 1px 6px;
        border-radius: 4px;
        font-family: monospace;
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


# Initialize session state for conversation history & upload cache
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize pipeline
pipeline_res = get_pipeline()
vector_index = pipeline_res[0] if pipeline_res else None
groq_llm = pipeline_res[1] if pipeline_res else None

# Check API key status
has_api_key = load_environment()

# Sidebar information & controls
with st.sidebar:
    st.title("📚 KnowRAG")
    st.markdown("**AI-Powered Knowledge Assistant**")
    st.markdown("---")

    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("❓ Help", key="btn_quick_help", use_container_width=True):
            st.session_state.pending_query = "/help"
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
        if st.button("ℹ️ About", key="btn_quick_about", use_container_width=True):
            st.session_state.pending_query = "/about"

    st.markdown("---")

    # Upload Documents Section
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="doc_uploader",
    )

    if uploaded_files and vector_index is not None:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            try:
                # Process document (extract text, validate, generate hash ID)
                documents, doc_id = process_uploaded_file(
                    filename=filename,
                    file_bytes=file_bytes,
                    save_to_disk=True,
                )

                # Incrementally index into ChromaDB
                index_result = index_uploaded_document(
                    documents=documents,
                    index=vector_index,
                )

                status = index_result.get("status")

                if status == "success":
                    st.success(
                        f"✅ **Document uploaded successfully!**\n\n"
                        f"📄 **File:** `{filename}`\n\n"
                        f"The document has been processed and added to the knowledge base. "
                        f"You can now ask questions about it."
                    )
                elif status == "duplicate":
                    st.info(f"ℹ️ **This document is already in the knowledge base.**\n\n📄 `{filename}`")
                else:
                    st.error(
                        f"❌ **I couldn't process this document.** Please make sure it is a valid PDF or TXT file.\n\n"
                        f"Details: {index_result.get('message')}"
                    )

            except Exception:
                st.error(
                    f"❌ **I couldn't process this document.** Please make sure it is a valid PDF or TXT file."
                )

    st.markdown("---")

    # Knowledge Base Overview Section
    st.markdown("### 📚 Knowledge Base")
    try:
        kb_stats = get_knowledge_base_stats()
        st.markdown(f"**Documents:** {kb_stats['total_documents']}")
        st.markdown(f"**Chunks:** {kb_stats['total_chunks']}")

        if kb_stats["uploaded_documents"]:
            st.markdown("**Uploaded Documents:**")
            for u_doc in kb_stats["uploaded_documents"]:
                st.markdown(f"✅ `{u_doc['filename']}` ({u_doc['chunks']} chunks)")

            if st.button("🗑️ Clear Uploaded Documents", use_container_width=True):
                deleted_chunks = clear_uploaded_documents(index=vector_index)
                st.success(f"Cleared {deleted_chunks} uploaded chunks. Baseline knowledge base intact.")
                st.rerun()
    except Exception as e:
        st.caption(f"Knowledge base status unavailable: {e}")

    st.markdown("---")

    # Sample Questions Section
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What services does the university library provide?",
        "What academic programs are offered by the university?",
        "What rules should students follow on campus?",
        "What support is available to students?",
        "What is TechNova University?",
        "Give me some cybersecurity project ideas",
    ]
    for sq in sample_questions:
        if st.button(sq, key=f"btn_{sq}", use_container_width=True):
            st.session_state.pending_query = sq

# Main Application Header
st.markdown('<div class="main-title">KnowRAG</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">AI-Powered Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-description">Ask questions about TechNova University, brainstorm ideas, or type <span class="command-badge">/help</span> for commands.</div>',
    unsafe_allow_html=True,
)

# Render API Key warning if not set
if not has_api_key:
    st.warning("⚠️ **GROQ_API_KEY is not configured.** Please add your Groq API key to the `.env` file or deployment secrets.")

# Render existing conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        sources_to_show = msg.get("source_details") or msg.get("sources", [])
        if sources_to_show:
            st.markdown("**Sources:**")
            for source in sources_to_show:
                st.markdown(f"📄 `{source}`")
        elif msg["role"] == "assistant" and msg.get("is_out_of_kb"):
            st.markdown(
                '<div class="no-source-notice">No relevant information was found in the knowledge base.</div>',
                unsafe_allow_html=True,
            )

# Determine user input (from chat_input or sample question button)
query_input = st.chat_input("Ask a question, say hello, or type /help...")

if "pending_query" in st.session_state and st.session_state.pending_query:
    active_query = st.session_state.pending_query
    del st.session_state.pending_query
else:
    active_query = query_input

# Handle query processing
if active_query:
    # Append user question to history
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Process query with conversational routing
    with st.chat_message("assistant"):
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
            with st.spinner("Processing..."):
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

                    # Render answer
                    st.markdown(answer_text)

                    # Render sources (only for RAG queries that retrieved sources)
                    if source_details:
                        st.markdown("**Sources:**")
                        for source in source_details:
                            st.markdown(f"📄 `{source}`")
                    elif is_out_of_kb:
                        st.markdown(
                            '<div class="no-source-notice">No relevant information was found in the knowledge base.</div>',
                            unsafe_allow_html=True,
                        )

                    # Append assistant response to session history
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
