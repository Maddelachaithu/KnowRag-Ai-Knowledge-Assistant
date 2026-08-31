"""
KnowRAG - Document Retrieval Module
-----------------------------------
This module handles semantic retrieval from the persistent ChromaDB vector store.
Given a natural language query, it embeds the query using the KnowRAG embedding model
(sentence-transformers/all-MiniLM-L6-v2) and retrieves the top-k most semantically
relevant chunks and their metadata from the vector database.

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 standard output encoding on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from src.vector_store import build_vector_index


# Default retrieval settings
DEFAULT_TOP_K = 3

# Cache index instance to avoid re-initializing on repeated queries
_GLOBAL_INDEX: Optional[VectorStoreIndex] = None


def get_or_load_index() -> VectorStoreIndex:
    """
    Retrieve the globally cached VectorStoreIndex or load it from ChromaDB.

    Returns:
        VectorStoreIndex: The initialized LlamaIndex vector store index.
    """
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is None:
        index, _ = build_vector_index(force_rebuild=False)
        _GLOBAL_INDEX = index
    return _GLOBAL_INDEX


def retrieve_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    index: Optional[VectorStoreIndex] = None,
) -> List[NodeWithScore]:
    """
    Perform semantic search against the ChromaDB vector store for a given query.

    Args:
        query (str): The natural language question or search query.
        top_k (int): Number of most relevant chunks to retrieve. Default is 3.
        index (Optional[VectorStoreIndex]): Pre-loaded index instance.
            If None, loads the index from the persistent ChromaDB store.

    Returns:
        List[NodeWithScore]: List of retrieved chunk nodes with similarity scores and metadata.
    """
    active_index = index if index is not None else get_or_load_index()

    # Create retriever with specified top_k
    retriever = active_index.as_retriever(similarity_top_k=top_k)

    # Execute retrieval query
    retrieved_nodes = retriever.retrieve(query)
    return retrieved_nodes


if __name__ == "__main__":
    print("=" * 70)
    print("KnowRAG: Semantic Document Retrieval Test")
    print(f"Default similarity_top_k: {DEFAULT_TOP_K}")
    print("=" * 70)

    # Pre-load index once for all queries
    vector_index = get_or_load_index()

    # Test questions and their expected source document
    test_cases = [
        {
            "question": "What services does the university library provide?",
            "expected_source": "library_services.txt",
        },
        {
            "question": "What academic programs are offered by the university?",
            "expected_source": "academic_programs.txt",
        },
        {
            "question": "What rules should students follow on campus?",
            "expected_source": "campus_rules.txt",
        },
        {
            "question": "What support is available to students?",
            "expected_source": "student_services.txt",
        },
        {
            "question": "What is TechNova University?",
            "expected_source": "university_overview.txt",
        },
    ]

    all_passed = True

    for q_idx, test_case in enumerate(test_cases, start=1):
        question = test_case["question"]
        expected_source = test_case["expected_source"]

        print(f"\nQUESTION {q_idx}:")
        print(f"\"{question}\"")
        print(f"Expected Primary Source: {expected_source}")
        print("-" * 70)

        results = retrieve_documents(query=question, top_k=DEFAULT_TOP_K, index=vector_index)

        print(f"Retrieved Results ({len(results)} found):")

        found_expected = False
        for rank, item in enumerate(results, start=1):
            source_file = item.node.metadata.get("file_name", "Unknown File")
            score = f"{item.score:.4f}" if item.score is not None else "N/A (Score not provided)"
            
            # Format text preview
            text_snippet = item.node.get_content().replace("\r", " ").replace("\n", " ").strip()
            if len(text_snippet) > 160:
                text_snippet = text_snippet[:160] + "..."

            if source_file == expected_source:
                found_expected = True

            print(f"\n{rank}. Source: {source_file}")
            print(f"   Similarity/Score: {score}")
            print(f"   Text: {text_snippet}")

        # Verification check for this test case
        match_status = "PASSED" if found_expected else "FAILED"
        if not found_expected:
            all_passed = False

        print(f"\nVerification: Expected source in retrieved results -> [{match_status}]")
        print("=" * 70)

    print("\n" + "=" * 70)
    if all_passed:
        print("All 5 test questions PASSED retrieval verification successfully!")
    else:
        print("Some test questions did not retrieve expected documents.")
    print("=" * 70)
