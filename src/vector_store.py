"""
KnowRAG - ChromaDB Vector Store Module
---------------------------------------
This module handles the vector storage and indexing stage of the KnowRAG RAG pipeline.
It stores embedded document chunks into a persistent ChromaDB database using LlamaIndex's
ChromaVectorStore integration.

Vector Database: ChromaDB (Persistent)
Collection Name: knowrag_documents
Persistence Dir: chroma_db/

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

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

import chromadb
from chromadb.api.models.Collection import Collection
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.ingestion import load_documents
from src.chunking import create_chunks
from src.embeddings import create_embedding_model


# Default ChromaDB configuration
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION_NAME = "knowrag_documents"


def create_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_dir: Optional[Path | str] = None,
) -> Tuple[ChromaVectorStore, Collection]:
    """
    Initialize a persistent ChromaDB client, create or retrieve the collection,
    and return the configured LlamaIndex ChromaVectorStore.

    Args:
        collection_name (str): Name of the ChromaDB collection.
            Defaults to 'knowrag_documents'.
        persist_dir (Optional[Path | str]): Directory where ChromaDB data is persisted.
            Defaults to the 'chroma_db' folder at project root.

    Returns:
        Tuple[ChromaVectorStore, Collection]: The vector store and raw Chroma collection.
    """
    target_dir = Path(persist_dir) if persist_dir else DEFAULT_PERSIST_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Initialize persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path=str(target_dir))

    # Retrieve existing collection or create a new one
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

    # Wrap collection with LlamaIndex ChromaVectorStore
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return vector_store, chroma_collection


def build_vector_index(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_dir: Optional[Path | str] = None,
    force_rebuild: bool = False,
) -> Tuple[VectorStoreIndex, Collection]:
    """
    Load source documents, generate chunks, initialize embedding model,
    and build or load the persistent VectorStoreIndex using ChromaDB.

    Avoids creating accidental duplicate records on repeated executions.

    Args:
        collection_name (str): Name of the ChromaDB collection.
        persist_dir (Optional[Path | str]): Persistence directory.
        force_rebuild (bool): If True, clears existing collection and re-indexes.

    Returns:
        Tuple[VectorStoreIndex, Collection]: The built/loaded index and Chroma collection.
    """
    target_dir = Path(persist_dir) if persist_dir else DEFAULT_PERSIST_DIR

    # Step 1: Create vector store and storage context
    vector_store, chroma_collection = create_vector_store(
        collection_name=collection_name, persist_dir=target_dir
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Step 2: Initialize embedding model
    embed_model = create_embedding_model()

    # Step 3: Check if collection already has existing records
    existing_count = chroma_collection.count()
    if existing_count > 0 and not force_rebuild:
        print(f"Collection '{collection_name}' already contains {existing_count} records.")
        print("Loading existing vector store index without duplicating records...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        return index, chroma_collection

    # If force_rebuild and records exist, clear them before re-indexing
    if existing_count > 0 and force_rebuild:
        print(f"Force rebuild requested: clearing {existing_count} existing records...")
        existing_data = chroma_collection.get()
        if existing_data and existing_data.get("ids"):
            chroma_collection.delete(ids=existing_data["ids"])

    # Step 4: Ingest documents and create chunks
    print("Ingesting source documents and creating chunks...")
    documents = load_documents()
    chunks = create_chunks(documents)

    # Step 5: Build VectorStoreIndex (embeddings are generated and stored in ChromaDB)
    print(f"Building vector index for {len(chunks)} chunks...")
    index = VectorStoreIndex(
        nodes=chunks,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    return index, chroma_collection


if __name__ == "__main__":
    print("=" * 60)
    print("KnowRAG: ChromaDB Vector Store Stage")
    print(f"Target Persistence Dir : {DEFAULT_PERSIST_DIR}")
    print(f"Target Collection Name : {DEFAULT_COLLECTION_NAME}")
    print("=" * 60)

    # Build or load vector index
    index, collection = build_vector_index()

    # Verification and summary
    total_records = collection.count()

    print("\n" + "=" * 60)
    print("Vector Store Verification Summary")
    print("=" * 60)
    print(f"Collection Name        : {collection.name}")
    print(f"Stored Vectors/Records : {total_records}")
    print(f"Persistence Directory  : {DEFAULT_PERSIST_DIR}")
    print(f"Records > 0 Check      : {'PASSED' if total_records > 0 else 'FAILED'}")
    print(f"Expected Count Check   : {'PASSED (5/5)' if total_records == 5 else f'COUNT: {total_records}'}")
    print("=" * 60)

    # Inspect stored items metadata
    sample_records = collection.get(limit=5)
    if sample_records and sample_records.get("metadatas"):
        print("\nStored Chunks Metadata Sample:")
        for idx, meta in enumerate(sample_records["metadatas"], start=1):
            file_name = meta.get("file_name", "Unknown") if meta else "Unknown"
            print(f"  [{idx}] Source File: {file_name}")

    if total_records == 5:
        print("\nChromaDB vector store stage completed and verified successfully.")
    else:
        print(f"\nChromaDB vector store stage completed with {total_records} records.")
