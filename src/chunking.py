"""
KnowRAG - Document Chunking Module
----------------------------------
This module handles the chunking stage of the KnowRAG RAG pipeline.
It splits loaded LlamaIndex Document objects into smaller text chunks (nodes)
using SentenceSplitter while preserving document metadata.

Configuration:
- chunk_size = 512
- chunk_overlap = 64

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import sys
from pathlib import Path
from typing import List

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

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document

from src.ingestion import load_documents


# Default chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def create_chunks(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[BaseNode]:
    """
    Split LlamaIndex Document objects into smaller text chunks (nodes).

    Args:
        documents (List[Document]): The list of LlamaIndex Document objects to split.
        chunk_size (int): Maximum size of each text chunk (in tokens). Default is 512.
        chunk_overlap (int): Overlap between adjacent chunks (in tokens). Default is 64.

    Returns:
        List[BaseNode]: A list of text chunk nodes with preserved metadata.
    """
    # Initialize SentenceSplitter with configured chunk size and overlap
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Generate nodes from documents (metadata is preserved automatically)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes


if __name__ == "__main__":
    print("=" * 60)
    print("KnowRAG: Document Chunking Stage")
    print(f"Configuration: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print("=" * 60)

    # Step 1: Load documents using the ingestion module
    documents = load_documents()
    print(f"\nLoaded {len(documents)} source document(s). Creating chunks...\n")

    # Step 2: Create chunks
    chunks = create_chunks(documents)

    # Step 3: Print summary & detailed chunk information
    print(f"Total Chunks Created: {len(chunks)}")
    print("=" * 60)

    for index, chunk in enumerate(chunks, start=1):
        source_file = chunk.metadata.get("file_name", "Unknown Source")
        char_count = len(chunk.get_content())
        
        # Create a single-line clean preview of the text
        raw_text = chunk.get_content().replace("\r", " ").replace("\n", " ").strip()
        preview = raw_text[:120] + "..." if len(raw_text) > 120 else raw_text

        print(f"\nChunk {index}")
        print("-" * 40)
        print(f"Source File     : {source_file}")
        print(f"Character Count : {char_count}")
        print(f"Text Preview    : {preview}")

    print("\n" + "=" * 60)
    print("Document chunking stage completed successfully.")
    print("=" * 60)
