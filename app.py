"""
KnowRAG — AI-Powered Knowledge Assistant
-----------------------------------------
Streamlit web application providing a conversational user interface
for querying the KnowRAG university knowledge base with strict source-grounding.

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.rag_pipeline import ask_question, create_rag_pipeline

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
    return create_rag_pipeline(temperature=0.0)


# Initialize session state for conversation history safely
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar information & controls
with st.sidebar:
    st.title("📚 KnowRAG")
    st.markdown("**Knowledge Assistant**")
    st.markdown("---")
    st.markdown("### 🔍 About")
    st.markdown(
        "KnowRAG delivers accurate, source-grounded answers to your questions "
        "using documents stored in the university knowledge base."
    )
    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What services does the university library provide?",
        "What academic programs are offered by the university?",
        "What rules should students follow on campus?",
        "What support is available to students?",
        "What is TechNova University?",
        "What is the university's policy on underwater basket weaving?",
    ]
    for sq in sample_questions:
        if st.button(sq, key=f"btn_{sq}", use_container_width=True):
            st.session_state.pending_query = sq

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        if "pending_query" in st.session_state:
            del st.session_state.pending_query
        st.rerun()

# Main Application Header
st.markdown('<div class="main-title">KnowRAG</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">AI-Powered Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-description">Ask questions about the information available in the KnowRAG knowledge base.</div>',
    unsafe_allow_html=True,
)

# Render existing conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("**Sources:**")
            for source in msg["sources"]:
                st.markdown(f"📄 `{source}`")
        elif msg["role"] == "assistant" and msg.get("is_out_of_kb"):
            st.markdown('<div class="no-source-notice">No relevant information was found in the knowledge base.</div>', unsafe_allow_html=True)

# Determine user input (from chat_input or sample question button)
query_input = st.chat_input("Ask a question about TechNova University...")

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

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                vector_index, groq_llm = get_pipeline()
                result = ask_question(
                    question=active_query,
                    index=vector_index,
                    llm=groq_llm,
                )

                answer_text = result["answer"]
                sources = result.get("sources", [])
                is_out_of_kb = (
                    not sources
                    or "not available in the knowledge base" in answer_text.lower()
                )

                # Render answer
                st.markdown(answer_text)

                # Render sources
                if sources:
                    st.markdown("**Sources:**")
                    for source in sources:
                        st.markdown(f"📄 `{source}`")
                elif is_out_of_kb:
                    st.markdown('<div class="no-source-notice">No relevant information was found in the knowledge base.</div>', unsafe_allow_html=True)

                # Append assistant response to session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                    "is_out_of_kb": is_out_of_kb,
                })

            except Exception as e:
                error_msg = f"An error occurred while processing your request: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "is_out_of_kb": True,
                })
