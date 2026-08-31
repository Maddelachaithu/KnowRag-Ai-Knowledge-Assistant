# KnowRAG — AI-Powered Knowledge Assistant

KnowRAG is an end-to-end Retrieval-Augmented Generation (RAG) knowledge assistant designed to answer natural-language questions using a local document knowledge base. It combines dense semantic retrieval with high-speed LLM generation, enforcing strict factual grounding and safe refusal of out-of-scope questions.

---

## Features

- **Document Ingestion**: Recursively loads source text documents with metadata preservation.
- **Text Chunking**: Splits documents into contextual chunks with configurable size and overlap.
- **Semantic Embeddings**: Generates 384-dimensional dense vector embeddings using Hugging Face Sentence Transformers.
- **Persistent Vector Storage**: Stores and indexes chunk embeddings in ChromaDB with duplicate protection.
- **Semantic Retrieval**: Retrieves top-k semantically relevant chunks based on cosine similarity.
- **Fast LLM Generation**: Powered by Groq's high-speed inference engine.
- **Strict Grounding**: System prompts designed to prevent hallucinations and adhere strictly to retrieved context.
- **Source Attribution**: Transparently cites exact source document filenames for every grounded answer.
- **Out-of-Knowledge-Base Refusal**: Safely rejects unsupported or out-of-domain questions with zero hallucination.
- **Streamlit Web UI**: Clean, interactive chat interface with persistent conversation history.
- **Automated Evaluation Suite**: Headless regression test suite verifying grounding and refusal accuracy (6/6 tests pass).

---

## Architecture

```text
Source Documents (data/*.txt)
          ↓
  Document Ingestion (src/ingestion.py)
          ↓
    Text Chunking (src/chunking.py)
          ↓
  Embedding Generation (src/embeddings.py)
          ↓
Persistent ChromaDB Vector Store (src/vector_store.py)
          ↓
  Semantic Retrieval (src/retrieval.py)
          ↓
   Retrieved Context Chunks + Metadata
          ↓
  Grounded Groq LLM Generation (src/llm.py & src/rag_pipeline.py)
          ↓
  Grounded Answer + Verified Source Files
          ↓
Interactive Streamlit UI (app.py)
```

---

## Technology Stack

- **Language**: Python 3.10+
- **RAG Framework**: [LlamaIndex](https://www.llamaindex.ai/) (`llama-index-core`)
- **Embedding Model**: [Sentence Transformers](https://www.sbert.net/) (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (`chromadb`, `llama-index-vector-stores-chroma`)
- **LLM Provider**: [Groq](https://groq.com/) (`llama-index-llms-groq`)
- **User Interface**: [Streamlit](https://streamlit.io/)
- **Environment Management**: `python-dotenv`

---

## Project Structure

```text
KnowRAG/
├── app.py                # Streamlit web application & conversational UI
├── data/                 # Source knowledge-base documents (.txt)
│   ├── academic_programs.txt
│   ├── campus_rules.txt
│   ├── library_services.txt
│   ├── student_services.txt
│   └── university_overview.txt
├── src/                  # Core RAG backend modules
│   ├── ingestion.py      # Document loading and metadata extraction
│   ├── chunking.py       # Sentence-aware text splitting
│   ├── embeddings.py     # SentenceTransformer embedding model initialization
│   ├── vector_store.py   # ChromaDB vector store creation and indexing
│   ├── retrieval.py      # Top-k semantic search & retriever caching
│   ├── llm.py            # Secure Groq LLM initialization & connectivity
│   └── rag_pipeline.py   # Grounded RAG orchestrator & question answering
├── tests/                # Automated testing and evaluation
│   ├── __init__.py
│   └── evaluate_rag.py   # Headless 6-question automated regression suite
├── chroma_db/            # Persistent ChromaDB collection directory
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── .gitignore            # Git exclusion rules (keeps .env secure)
└── .env                  # Local environment variables (DO NOT COMMIT)
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
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=groq/compound-mini
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

## Running Automated Evaluation

Execute the automated RAG evaluation and regression suite:

```powershell
python tests\evaluate_rag.py
```

### Evaluation Test Cases & Results

| Test # | Question | Ground Truth Source | Result |
| :---: | :--- | :--- | :---: |
| **1** | *"What services does the university library provide?"* | `library_services.txt` | **PASS** |
| **2** | *"What academic programs are offered by the university?"* | `academic_programs.txt` | **PASS** |
| **3** | *"What rules should students follow on campus?"* | `campus_rules.txt` | **PASS** |
| **4** | *"What support is available to students?"* | `student_services.txt`, `library_services.txt` | **PASS** |
| **5** | *"What is TechNova University?"* | `university_overview.txt`, `academic_programs.txt` | **PASS** |
| **6** | *"What is the university's policy on underwater basket weaving?"* | *None (Out-of-KB Refusal)* | **PASS** |

**Pass Rate**: **6/6 PASS (100%)**

---

## Example Questions

- *"What services does the university library provide?"*
- *"What academic programs are offered by the university?"*
- *"What rules should students follow on campus?"*
- *"What support is available to students?"*
- *"What is TechNova University?"*

**Out-of-Scope Handling:** If asked about topics outside the knowledge base (e.g., *"What is the university's policy on underwater basket weaving?"*), the assistant safely responds:
> *"The requested information is not available in the knowledge base."*

---

## Security Principles

- **Zero Credential Exposure**: API keys are dynamically loaded via environment variables and never hardcoded or printed.
- **Git Protection**: `.env` is explicitly ignored in `.gitignore`.
- **UI Sanitization**: Streamlit displays only generated responses and verified source names without leaking internal exceptions or keys.
- **Closed Knowledge Boundaries**: No unauthorized web searches or external network calls are performed during retrieval.

---

## Limitations

- **Dataset Scale**: Currently configured for a curated university knowledge base.
- **Retrieval Dependency**: Answers are strictly constrained by the quality and relevance of retrieved chunks.
- **Groq API Dependency**: Active internet connection and valid Groq API key are required for LLM generation.

---

## Future Enhancements

- Dynamic document upload via Streamlit UI (PDF, DOCX).
- Cross-encoder reranking for improved retrieval precision on large corpora.
- Context-aware multi-turn conversational retrieval.
- Interactive chunk previews and similarity score visualization.
- Production containerization (Docker) and cloud deployment.
