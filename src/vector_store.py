"""
KnowRAG - ChromaDB Vector Store Module
---------------------------------------
This module handles the vector storage, indexing, and incremental document additions
for the KnowRAG RAG pipeline. It stores embedded document chunks into a persistent
ChromaDB database using LlamaIndex's ChromaVectorStore integration.

Features:
- Persistent ChromaDB vector storage (default: knowrag_documents collection)
- Incremental document indexing (adds only new document chunks without rebuilds)
- Deterministic duplicate document protection based on document IDs and hashes
- Safe knowledge base statistics and selective uploaded-document clearing

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

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

import chromadb
from chromadb.api.models.Collection import Collection
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.ingestion import UPLOAD_DIR, load_documents
from src.chunking import create_chunks
from src.embeddings import create_embedding_model


# Default ChromaDB configuration
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION_NAME = "knowrag_documents"

DEFAULT_DOCUMENT_NAMES = {
    "academic_programs.txt",
    "campus_rules.txt",
    "library_services.txt",
    "student_services.txt",
    "university_overview.txt",
}


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
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        return index, chroma_collection

    # If force_rebuild and records exist, clear them before re-indexing
    if existing_count > 0 and force_rebuild:
        existing_data = chroma_collection.get()
        if existing_data and existing_data.get("ids"):
            chroma_collection.delete(ids=existing_data["ids"])

    # Step 4: Ingest documents and create chunks
    documents = load_documents()
    chunks = create_chunks(documents)

    # Step 5: Build VectorStoreIndex (embeddings are generated and stored in ChromaDB)
    index = VectorStoreIndex(
        nodes=chunks,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    return index, chroma_collection


def is_document_indexed(
    document_id: str,
    collection: Optional[Collection] = None,
    filename: Optional[str] = None,
) -> bool:
    """
    Check whether a document with the given document_id or doc_hash already exists in ChromaDB.

    Args:
        document_id (str): Unique deterministic document hash identifier.
        collection (Optional[Collection]): Chroma collection to query.
        filename (Optional[str]): Optional filename for fallback verification.

    Returns:
        bool: True if already present in collection metadata, False otherwise.
    """
    if not document_id:
        return False

    target_collection = collection
    if target_collection is None:
        _, target_collection = create_vector_store()

    try:
        # Check by doc_hash metadata field
        results_hash = target_collection.get(
            where={"doc_hash": document_id},
            limit=1,
        )
        if results_hash and results_hash.get("ids") and len(results_hash["ids"]) > 0:
            return True

        # Check by document_id metadata field
        results_doc = target_collection.get(
            where={"document_id": document_id},
            limit=1,
        )
        if results_doc and results_doc.get("ids") and len(results_doc["ids"]) > 0:
            return True

        return False
    except Exception:
        return False


def index_uploaded_document(
    documents: List[Document],
    index: Optional[VectorStoreIndex] = None,
    collection: Optional[Collection] = None,
) -> Dict[str, Any]:
    """
    Incrementally chunk, embed, and index a user-uploaded document into ChromaDB.
    Enforces duplicate document protection.

    Args:
        documents (List[Document]): Document objects representing the uploaded file.
        index (Optional[VectorStoreIndex]): Active VectorStoreIndex instance.
        collection (Optional[Collection]): Active ChromaDB Collection instance.

    Returns:
        Dict[str, Any]: Indexing status result containing status, message, and chunk count.
    """
    if not documents:
        return {
            "status": "error",
            "message": "No document content provided to index.",
            "chunks_created": 0,
            "filename": "",
            "document_id": "",
        }

    first_doc = documents[0]
    filename = first_doc.metadata.get("file_name", "unknown")
    doc_id = first_doc.metadata.get("doc_hash") or first_doc.metadata.get("document_id", "")

    # Ensure index and collection are initialized
    active_index = index
    active_collection = collection

    if active_index is None:
        active_index, active_collection = build_vector_index(force_rebuild=False)
    elif active_collection is None:
        # Retrieve underlying collection reference if available
        if hasattr(active_index, "vector_store") and hasattr(active_index.vector_store, "_collection"):
            active_collection = active_index.vector_store._collection
        else:
            _, active_collection = create_vector_store()

    # 1. Duplicate check: prevent duplicate chunks for identical files
    if is_document_indexed(document_id=doc_id, collection=active_collection, filename=filename):
        return {
            "status": "duplicate",
            "message": "⚠️ This document is already indexed.",
            "chunks_created": 0,
            "filename": filename,
            "document_id": doc_id,
        }

    # 2. Chunking using the existing KnowRAG chunking strategy
    chunks = create_chunks(documents)

    if not chunks:
        return {
            "status": "error",
            "message": f"Failed to generate text chunks from '{filename}'.",
            "chunks_created": 0,
            "filename": filename,
            "document_id": doc_id,
        }

    # 3. Incremental insertion into vector store (embeddings computed & saved to ChromaDB)
    active_index.insert_nodes(chunks)

    return {
        "status": "success",
        "message": "✅ Indexed successfully",
        "chunks_created": len(chunks),
        "filename": filename,
        "document_id": doc_id,
    }


def get_knowledge_base_stats(
    collection: Optional[Collection] = None,
) -> Dict[str, Any]:
    """
    Retrieve current statistics about default and uploaded documents in ChromaDB.

    Args:
        collection (Optional[Collection]): Chroma collection to inspect.

    Returns:
        Dict[str, Any]: Document and chunk metrics for UI display.
    """
    target_collection = collection
    if target_collection is None:
        _, target_collection = create_vector_store()

    total_chunks = target_collection.count()
    all_data = target_collection.get(include=["metadatas"])
    metadatas = all_data.get("metadatas", []) if all_data else []

    default_files_found = set()
    uploaded_docs: Dict[str, Dict[str, Any]] = {}

    for meta in metadatas:
        if not meta:
            continue
        fn = meta.get("file_name") or meta.get("source") or "unknown"
        is_uploaded = meta.get("is_uploaded", False)
        doc_id = meta.get("doc_hash") or meta.get("document_id", fn)
        file_type = meta.get("file_type", "txt")

        if is_uploaded or fn not in DEFAULT_DOCUMENT_NAMES:
            if doc_id not in uploaded_docs:
                uploaded_docs[doc_id] = {
                    "filename": fn,
                    "file_type": file_type,
                    "document_id": doc_id,
                    "chunks": 0,
                }
            uploaded_docs[doc_id]["chunks"] += 1
        else:
            default_files_found.add(fn)

    # Ensure baseline count matches known default files
    default_count = len(default_files_found) if default_files_found else len(DEFAULT_DOCUMENT_NAMES)
    total_docs = default_count + len(uploaded_docs)

    return {
        "total_chunks": total_chunks,
        "total_documents": total_docs,
        "default_documents_count": default_count,
        "uploaded_documents": list(uploaded_docs.values()),
    }


def clear_uploaded_documents(
    collection: Optional[Collection] = None,
    index: Optional[VectorStoreIndex] = None,
) -> int:
    """
    Remove only user-uploaded documents from ChromaDB and the upload storage directory.
    Guarantees that the 5 default knowledge base documents are NEVER deleted.

    Args:
        collection (Optional[Collection]): ChromaDB collection instance.
        index (Optional[VectorStoreIndex]): VectorStoreIndex instance.

    Returns:
        int: Number of deleted chunk records.
    """
    target_collection = collection
    if target_collection is None:
        if index is not None and hasattr(index, "vector_store") and hasattr(index.vector_store, "_collection"):
            target_collection = index.vector_store._collection
        else:
            _, target_collection = create_vector_store()

    all_data = target_collection.get(include=["metadatas"])
    ids_to_delete = []

    if all_data and all_data.get("ids") and all_data.get("metadatas"):
        for record_id, meta in zip(all_data["ids"], all_data["metadatas"]):
            if not meta:
                continue
            fn = meta.get("file_name") or meta.get("source") or ""
            is_uploaded = meta.get("is_uploaded", False)

            # Mark for deletion ONLY if it is an uploaded document
            if is_uploaded or (fn and fn not in DEFAULT_DOCUMENT_NAMES):
                ids_to_delete.append(record_id)

    deleted_count = len(ids_to_delete)
    if ids_to_delete:
        target_collection.delete(ids=ids_to_delete)

    # Clean up uploaded files in data/uploads/
    try:
        if UPLOAD_DIR.exists():
            for item in UPLOAD_DIR.iterdir():
                if item.is_file() and item.name != ".gitkeep":
                    item.unlink()
    except Exception:
        pass

    return deleted_count


if __name__ == "__main__":
    print("=" * 60)
    print("KnowRAG: ChromaDB Vector Store Stage")
    print(f"Target Persistence Dir : {DEFAULT_PERSIST_DIR}")
    print(f"Target Collection Name : {DEFAULT_COLLECTION_NAME}")
    print("=" * 60)

    # Build or load vector index
    index, collection = build_vector_index()

    stats = get_knowledge_base_stats(collection)

    print("\n" + "=" * 60)
    print("Vector Store Verification Summary")
    print("=" * 60)
    print(f"Collection Name        : {collection.name}")
    print(f"Total Stored Chunks    : {stats['total_chunks']}")
    print(f"Total Documents        : {stats['total_documents']}")
    print(f"Uploaded Documents     : {len(stats['uploaded_documents'])}")
    print(f"Persistence Directory  : {DEFAULT_PERSIST_DIR}")
    print("=" * 60)
