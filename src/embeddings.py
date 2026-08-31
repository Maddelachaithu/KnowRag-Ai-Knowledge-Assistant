"""
KnowRAG - Embedding Generation Module
-------------------------------------
This module handles the embedding generation stage of the KnowRAG RAG pipeline.
It initializes Hugging Face embedding models via LlamaIndex and generates
dense vector embeddings from text chunks.

Model: sentence-transformers/all-MiniLM-L6-v2 (Embedding Dimension: 384)

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

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import BaseNode

from src.ingestion import load_documents
from src.chunking import create_chunks


# Model configuration
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIM = 384


def create_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbedding:
    """
    Initialize and return the configured Hugging Face embedding model.

    Args:
        model_name (str): The Hugging Face repository identifier for the model.
            Default is 'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
        HuggingFaceEmbedding: The configured LlamaIndex embedding model.
    """
    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        model_kwargs={"local_files_only": True},
    )
    return embed_model


if __name__ == "__main__":
    print("=" * 60)
    print("KnowRAG: Embedding Generation Stage")
    print(f"Model: {DEFAULT_EMBEDDING_MODEL}")
    print(f"Expected Vector Dimension: {EXPECTED_EMBEDDING_DIM}")
    print("=" * 60)

    # Step 1: Load documents and create chunks
    documents = load_documents()
    chunks = create_chunks(documents)
    print(f"\n[1] Ingested {len(documents)} document(s) and created {len(chunks)} chunk(s).")

    # Step 2: Initialize embedding model
    print("\n[2] Initializing embedding model...")
    embed_model = create_embedding_model()
    print("    Embedding model initialized successfully.")

    # Step 3: Test embedding generation on chunks
    print("\n[3] Generating embedding vectors...")
    
    # Generate embedding for the first chunk to test
    sample_chunk = chunks[0]
    sample_text = sample_chunk.get_content()
    sample_file = sample_chunk.metadata.get("file_name", "Unknown")
    
    embedding_vector = embed_model.get_text_embedding(sample_text)
    vector_dim = len(embedding_vector)

    # Step 4: Verification checks
    is_numeric = all(isinstance(val, (int, float)) for val in embedding_vector)
    dim_matches = (vector_dim == EXPECTED_EMBEDDING_DIM)

    print("\n" + "=" * 60)
    print("Embedding Verification Summary")
    print("=" * 60)
    print(f"Sample Chunk Source     : {sample_file}")
    print(f"Chunks Processed Total  : {len(chunks)}")
    print(f"Embedding Vector Dim    : {vector_dim} (Expected: {EXPECTED_EMBEDDING_DIM})")
    print(f"Dimension Check Passed  : {'YES' if dim_matches else 'NO'}")
    print(f"Numeric Vector Check    : {'YES' if is_numeric else 'NO'}")
    
    # Preview first 6 dimensions of the vector
    preview_dims = [round(float(x), 6) for x in embedding_vector[:6]]
    print(f"Vector Sample (first 6) : {preview_dims} ...")
    print("=" * 60)

    if dim_matches and is_numeric:
        print("\nEmbedding stage completed and verified successfully.")
    else:
        print("\nEmbedding stage verification FAILED.")
