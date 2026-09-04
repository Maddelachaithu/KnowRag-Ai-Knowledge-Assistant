# KnowRAG — AI-Powered Knowledge Assistant

KnowRAG is an end-to-end Retrieval-Augmented Generation (RAG) knowledge assistant designed to answer natural-language questions using a local document knowledge base, dynamically uploaded user documents (.pdf and .txt), and intelligent conversational intent routing. It combines dense semantic retrieval with high-speed LLM generation, enforcing strict factual grounding, page-aware source attribution, conversational greetings, slash commands, and safe refusal of out-of-scope questions.

---

## Features

- **Conversational Greetings & Polite Dialogue**: Recognizes standard greetings (`hello`, `hi`, `hey`, `good morning`, `good evening`, etc.) and casual dialogue (`thanks`, `bye`, `who are you?`, `ok`) with friendly, natural responses without invoking unnecessary vector search.
- **Slash Commands**: Built-in command handling for `/help`, `/show documents`, `/clear`, and `/about`.
- **Brainstorming & Ideation Routing**: Directs open-ended queries (e.g., *"give me cybersecurity project ideas"*) to the conversational LLM while keeping factual document queries strictly grounded.
- **Dynamic Multi-Document Upload**: Upload multiple PDF and TXT documents directly through the Streamlit sidebar.
- **Page-Aware Document Ingestion**: Ingests TXT documents and extracts text page-by-page from PDFs using `pypdf`, tracking page numbers in metadata.
- **Deterministic Duplicate Protection**: Uses SHA-256 content and filename hashing to prevent duplicate chunks in ChromaDB.
- **Incremental ChromaDB Indexing**: Seamlessly embeds and inserts newly uploaded document chunks without deleting or rebuilding existing vectors.
- **Text Chunking**: Splits documents into contextual chunks (chunk size: 512, overlap: 64) with SentenceSplitter, preserving document and page metadata.
- **Semantic Embeddings**: Generates 384-dimensional dense vector embeddings using Hugging Face Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`).
- **Semantic Retrieval**: Retrieves top-k semantically relevant chunks based on cosine similarity across default and uploaded documents.
- **Fast LLM Generation**: Powered by Groq's high-speed inference engine (default: `openai/gpt-oss-20b`).
- **Strict Grounding**: System prompts designed to prevent hallucinations and adhere strictly to retrieved context.
- **Page-Level Source Attribution**: Transparently cites exact source document filenames and page numbers (e.g., `📄 transportation_guidelines.pdf — Page 1`).
- **Out-of-Knowledge-Base Refusal**: Safely rejects unsupported or out-of-domain questions with zero hallucination.
- **Knowledge Base Management UI**: Visual metrics for total documents and chunks, list of uploaded documents, quick action shortcuts, and safe selective clearing.
- **Automated Test Suites**: Includes conversational test suite (`test_conversational_suite.py` - 17/17 PASS), upload verification suite (`test_upload_feature.py` - 6/6 PASS), and baseline regression suite (`evaluate_rag.py` - 6/6 PASS).

---

## Conversational Flow & Architecture

```text
User Input
    ↓
Normalize Input
    ↓
Check Slash Commands (/help, /show documents, /clear, /about)
    ↓
Check Greetings (hello, hi, good morning, etc.)
    ↓
Check Casual Conversation (thanks, bye, who are you?, ok)
    ↓
Check Brainstorming / Ideation Inquiries
    ├── If Brainstorming → Conversational Groq LLM (Direct generation, no RAG context)
    └── If Document Query → RAG Pipeline:
            ↓
          Semantic Similarity Retrieval (ChromaDB top-k)
            ↓
          Retrieved Chunks + Page Metadata
            ↓
          Strictly Grounded Prompt Construction
            ↓
          Groq LLM Generation
            ↓
          Grounded Answer + Page-Level Source Attribution
```

---

## Available Commands

| Command | Description | Example Output |
| :--- | :--- | :--- |
| `/help` | Display guidance on querying, uploading documents, and available commands | Lists all actions and slash commands |
| `/show documents` | Dynamically display all default and user-uploaded documents in ChromaDB | Shows list of available files & chunk counts |
| `/clear` | Clear the chat history in the current session | `"🧹 Chat history cleared. How can I help you?"` |
| `/about` | Display information about KnowRAG's architecture and RAG pipeline | Overview of chunking, embedding, vector store, and LLM |

---

## Technology Stack

- **Language**: Python 3.10+
- **RAG Framework**: [LlamaIndex](https://www.llamaindex.ai/) (`llama-index-core`)
- **PDF Extraction**: [pypdf](https://pypdf.readthedocs.io/) (`pypdf>=4.0.0`)
- **Embedding Model**: [Sentence Transformers](https://www.sbert.net/) (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (`chromadb`, `llama-index-vector-stores-chroma`)
- **LLM Provider**: [Groq](https://groq.com/) (`llama-index-llms-groq`)
- **User Interface**: [Streamlit](https://streamlit.io/)
- **Environment Management**: `python-dotenv`

---

## Project Structure

```text
KnowRAG/
├── app.py                            # Streamlit web application with conversational UI & uploader
├── data/                             # Source knowledge-base documents
│   ├── academic_programs.txt         # Baseline university document
│   ├── campus_rules.txt              # Baseline university document
│   ├── library_services.txt          # Baseline university document
│   ├── student_services.txt          # Baseline university document
│   ├── university_overview.txt       # Baseline university document
│   ├── pdfs/                         # Source PDFs for reference
│   └── uploads/                      # Staging directory for user-uploaded files
├── src/                              # Core RAG backend modules
│   ├── conversation.py               # Intent classification, greetings, slash commands, & router
│   ├── ingestion.py                  # Multi-format document loading, PDF extraction, & hashing
│   ├── chunking.py                   # Sentence-aware text splitting preserving metadata
│   ├── embeddings.py                 # SentenceTransformer embedding model initialization
│   ├── vector_store.py               # ChromaDB persistent store, incremental indexing & stats
│   ├── retrieval.py                  # Top-k semantic search & retriever caching
│   ├── llm.py                        # Secure Groq LLM initialization & connectivity
│   └── rag_pipeline.py               # Grounded RAG orchestrator with page-aware citations
├── tests/                            # Automated testing and evaluation
│   ├── __init__.py
│   ├── test_conversational_suite.py  # 17-point comprehensive conversational & functional test suite
│   ├── evaluate_rag.py               # Baseline 6-question automated regression suite
│   └── test_upload_feature.py        # Dynamic upload, deduplication, and page retrieval suite
├── chroma_db/                        # Persistent ChromaDB collection directory
├── requirements.txt                  # Project dependencies
├── README.md                         # Project documentation
├── .gitignore                        # Git exclusion rules (keeps .env and uploads secure)
└── .env                              # Local environment variables (DO NOT COMMIT)
```

---

## Setup & Installation

### 1. Clone the Repository & Create Virtual Environment (Windows PowerShell)

```powershell
# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Required: Groq API Key
GROQ_API_KEY=your_actual_groq_api_key_here

# Optional: Model override (default is openai/gpt-oss-20b)
GROQ_MODEL=openai/gpt-oss-20b
```

> **Security Note:** The `.env` file contains secret credentials and must **never** be committed to version control. It is excluded by `.gitignore`.

---

## Running the Application

Launch the Streamlit web interface:

```powershell
python -m streamlit run app.py
```

The application will be accessible at: `http://localhost:8501`

---

## Running Automated Evaluation & Tests

### 1. Comprehensive Conversational & Functional Test Suite (17 Tests)

```powershell
python tests\test_conversational_suite.py
```

| Test # | Objective | Result |
| :---: | :--- | :---: |
| **01** | Greeting Detection | **PASS** |
| **02** | `"hello"` Response | **PASS** |
| **03** | `"hi"` Response | **PASS** |
| **04** | `/help` Command | **PASS** |
| **05** | `/show documents` Command (Dynamic List) | **PASS** |
| **06** | `/clear` Command Handling | **PASS** |
| **07** | `/about` Command | **PASS** |
| **08** | `"thanks"` Response | **PASS** |
| **09** | `"bye"` Response | **PASS** |
| **10** | Brainstorming / General Question Routing | **PASS** |
| **11** | Existing RAG Question Grounding | **PASS** |
| **12** | Out-of-Knowledge-Base Safe Refusal | **PASS** |
| **13** | PDF Upload & Query | **PASS** |
| **14** | TXT Upload & Query | **PASS** |
| **15** | Duplicate Upload Protection | **PASS** |
| **16** | Source Citation Verification | **PASS** |
| **17** | Missing `GROQ_API_KEY` Graceful Handling | **PASS** |

**Result**: **17/17 PASS (100%)**

### 2. Baseline Regression Suite

```powershell
python tests\evaluate_rag.py
```

**Result**: **6/6 PASS (100%)**

### 3. Dynamic Upload Feature Test Suite

```powershell
python tests\test_upload_feature.py
```

**Result**: **6/6 PASS (100%)**

---

## Security Principles

- **Zero Credential Exposure**: API keys are dynamically loaded via environment variables and never hardcoded or printed.
- **Git Protection**: `.env` and `data/uploads/*` are explicitly ignored in `.gitignore`.
- **UI Sanitization**: Streamlit displays friendly alerts if credentials are missing without leaking stack traces or keys.
- **Closed Knowledge Boundaries**: Grounding prompts strictly restrict the LLM to retrieved context for factual questions.
