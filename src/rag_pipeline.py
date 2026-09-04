"""
KnowRAG - End-to-End RAG Pipeline Module
----------------------------------------
This module orchestrates the complete Retrieval-Augmented Generation (RAG)
pipeline for KnowRAG with strict factual source-grounding.

Pipeline Flow:
1. Receive natural language question.
2. Retrieve top-k semantically relevant chunks from persistent ChromaDB vector store.
3. Construct a strictly grounded context prompt with document & page annotations.
4. Generate an accurate, grounded answer using the Groq LLM.
5. Parse and return structured output (answer, sources, source_details, retrieved_context).

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from llama_index.llms.groq import Groq

from src.retrieval import DEFAULT_TOP_K, get_or_load_index, retrieve_documents
from src.llm import create_llm


# Strictly grounded RAG System Prompt
RAG_SYSTEM_PROMPT = (
    "You are KnowRAG, a strict and faithful AI Knowledge Assistant for TechNova University.\n"
    "Your objective is to answer the user's question with absolute factual accuracy based ONLY on the provided retrieved context.\n\n"
    "STRICT GROUNDING INSTRUCTIONS:\n"
    "1. Answer ONLY using the facts explicitly stated in the RETRIEVED CONTEXT below.\n"
    "2. Do NOT use outside knowledge, external assumptions, or invented facts.\n"
    "3. Do NOT add programs, degrees, departments, services, rules, or names that do not appear in the context.\n"
    "4. Do NOT confuse or merge distinct facilities/services unless the context directly equates them.\n"
    "5. If the retrieved context does NOT contain direct information to answer the question, output EXACTLY:\n"
    "   The requested information is not available in the knowledge base.\n"
    "6. At the end of your response, on a new line, explicitly list the exact source document(s) from the context that directly provided the facts for your answer in this format:\n"
    "   SOURCES_USED: filename1.txt, filename2.txt\n"
    "   (If the requested information is not available in the knowledge base, write: SOURCES_USED: None)"
)

# Grounded prompt template
RAG_PROMPT_TEMPLATE = """{system_prompt}

---------------------
RETRIEVED CONTEXT:
{context_str}
---------------------

QUESTION: {question}

ANSWER:"""


def create_rag_pipeline(
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> Tuple[VectorStoreIndex, Groq]:
    """
    Initialize and load the core components of the RAG pipeline.

    Args:
        model (Optional[str]): The model identifier to use with Groq.
        temperature (float): Sampling temperature for generation. Default is 0.0 for deterministic grounding.
        **kwargs: Additional keyword arguments forwarded to create_llm.

    Returns:
        Tuple[VectorStoreIndex, Groq]: The loaded vector index and initialized LLM instance.
    """
    index = get_or_load_index()
    llm = create_llm(model=model, temperature=temperature, **kwargs)
    return index, llm


def format_context_string(retrieved_nodes: List[NodeWithScore]) -> str:
    """
    Format retrieved document nodes into a clean context string with source and page annotations.

    Args:
        retrieved_nodes (List[NodeWithScore]): The list of retrieved chunk nodes.

    Returns:
        str: Formatted context string for prompt insertion.
    """
    if not retrieved_nodes:
        return "No relevant context found."

    context_parts = []
    for idx, node_with_score in enumerate(retrieved_nodes, start=1):
        node = node_with_score.node
        source_name = node.metadata.get("file_name") or node.metadata.get("source") or "Unknown Source"
        page_number = node.metadata.get("page_number")

        if page_number:
            header = f"[Document: {source_name} — Page {page_number}]"
        else:
            header = f"[Document: {source_name}]"

        text = node.get_content().strip()
        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def parse_llm_response(
    raw_response: str,
    retrieved_nodes: List[NodeWithScore],
) -> Tuple[str, List[str]]:
    """
    Parse the LLM response to separate the grounded answer text from the sources used.

    Args:
        raw_response (str): The raw response text from the LLM.
        retrieved_nodes (List[NodeWithScore]): The retrieved nodes to validate source filenames.

    Returns:
        Tuple[str, List[str]]: Cleaned answer string and list of verified source filenames.
    """
    # Available valid filenames from retrieved context
    valid_sources = {
        node_with_score.node.metadata.get("file_name") or node_with_score.node.metadata.get("source")
        for node_with_score in retrieved_nodes
        if node_with_score.node.metadata.get("file_name") or node_with_score.node.metadata.get("source")
    }

    raw_text = raw_response.strip()
    sources: List[str] = []

    # Check for SOURCES_USED tag
    sources_pattern = r"(?i)SOURCES_USED:\s*(.*)$"
    match = re.search(sources_pattern, raw_text)

    if match:
        answer_text = raw_text[: match.start()].strip()
        sources_str = match.group(1).strip()
        if "none" not in sources_str.lower():
            # Extract mentioned valid source filenames
            for fn in valid_sources:
                if fn and fn.lower() in sources_str.lower():
                    if fn not in sources:
                        sources.append(fn)
    else:
        answer_text = raw_text

    # Check if answer is a rejection / not found
    is_not_available = (
        "not available in the knowledge base" in answer_text.lower()
        or "information is not available" in answer_text.lower()
        or "not mentioned in the provided context" in answer_text.lower()
    )

    if is_not_available:
        sources = []
    elif not sources:
        # Fallback: if SOURCES_USED was omitted by LLM but answer was grounded,
        # assign primary retrieved source
        if retrieved_nodes:
            primary_src = (
                retrieved_nodes[0].node.metadata.get("file_name")
                or retrieved_nodes[0].node.metadata.get("source")
            )
            if primary_src:
                sources = [primary_src]

    return answer_text, sources


def format_source_details(
    sources: List[str],
    retrieved_nodes: List[NodeWithScore],
) -> List[str]:
    """
    Generate rich source citation labels with page numbers when available.

    Args:
        sources (List[str]): List of cited source filenames.
        retrieved_nodes (List[NodeWithScore]): Retrieved chunk nodes.

    Returns:
        List[str]: Formatted source detail strings (e.g. 'hostel_rules.pdf — Page 2').
    """
    details: List[str] = []
    seen = set()

    for src in sources:
        pages_found = []
        for node_ws in retrieved_nodes:
            node_src = node_ws.node.metadata.get("file_name") or node_ws.node.metadata.get("source")
            if node_src == src:
                page_num = node_ws.node.metadata.get("page_number")
                if page_num and page_num not in pages_found:
                    pages_found.append(page_num)

        if pages_found:
            pages_str = ", ".join(str(p) for p in sorted(pages_found))
            label = f"{src} — Page {pages_str}"
        else:
            label = src

        if label not in seen:
            seen.add(label)
            details.append(label)

    return details


def ask_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    index: Optional[VectorStoreIndex] = None,
    llm: Optional[Groq] = None,
) -> Dict[str, Any]:
    """
    End-to-end question answering function using the strictly grounded RAG pipeline.

    Args:
        question (str): The natural language user query.
        top_k (int): Number of chunks to retrieve. Default is 3.
        index (Optional[VectorStoreIndex]): Pre-loaded index (loads default if None).
        llm (Optional[Groq]): Pre-initialized LLM (initializes default if None).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'question': Original question text.
            - 'answer': Generated answer string.
            - 'sources': List of unique verified source filenames.
            - 'source_details': List of formatted source labels with page numbers.
            - 'retrieved_context': List of retrieved node dicts (text, score, file_name, metadata).
    """
    # 1. Ensure pipeline components are loaded
    active_index = index if index is not None else get_or_load_index()
    active_llm = llm if llm is not None else create_llm()

    # 2. Semantic retrieval of top-k chunks
    retrieved_nodes = retrieve_documents(query=question, top_k=top_k, index=active_index)

    # 3. Format context string with document and page annotations
    context_str = format_context_string(retrieved_nodes)

    # 4. Build structured prompt
    prompt = RAG_PROMPT_TEMPLATE.format(
        system_prompt=RAG_SYSTEM_PROMPT,
        context_str=context_str,
        question=question,
    )

    # 5. Generate answer using Groq LLM
    response = active_llm.complete(prompt)
    answer_text, sources = parse_llm_response(response.text, retrieved_nodes)
    source_details = format_source_details(sources, retrieved_nodes)

    # 6. Package retrieved context details for inspection and grounding verification
    retrieved_context_data = [
        {
            "text": node_with_score.node.get_content().strip(),
            "score": round(float(node_with_score.score), 4) if node_with_score.score is not None else None,
            "file_name": node_with_score.node.metadata.get("file_name") or node_with_score.node.metadata.get("source", "Unknown"),
            "page_number": node_with_score.node.metadata.get("page_number"),
            "metadata": node_with_score.node.metadata,
        }
        for node_with_score in retrieved_nodes
    ]

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "source_details": source_details,
        "retrieved_context": retrieved_context_data,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("KnowRAG — Grounding Audit & RAG Pipeline Test")
    print(f"Retrieval similarity_top_k: {DEFAULT_TOP_K}")
    print("=" * 70)

    # Pre-initialize index and LLM once for efficiency
    vector_index, groq_llm = create_rag_pipeline(temperature=0.0)

    # 6 Test questions (5 in-knowledge-base, 1 out-of-knowledge-base)
    test_questions = [
        "What services does the university library provide?",
        "What academic programs are offered by the university?",
        "What rules should students follow on campus?",
        "What support is available to students?",
        "What is TechNova University?",
        "What is the university's policy on underwater basket weaving?",
    ]

    for i, q in enumerate(test_questions, start=1):
        print(f"\n{'=' * 70}")
        print(f"TEST {i}:")
        print(f"Question: {q}")
        print("-" * 70)

        result = ask_question(question=q, top_k=DEFAULT_TOP_K, index=vector_index, llm=groq_llm)

        print("Retrieved Context Chunks:")
        for idx, chunk in enumerate(result["retrieved_context"], 1):
            fn = chunk["file_name"]
            page = chunk.get("page_number")
            page_info = f", page={page}" if page else ""
            score = chunk["score"]
            snippet = chunk["text"].replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            print(f"  [{idx}] ({fn}{page_info}, score={score}): {snippet}")

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        if result["source_details"]:
            for src in result["source_details"]:
                print(f"- {src}")
        else:
            print("- None (Information unavailable in knowledge base)")
        print("=" * 70)
