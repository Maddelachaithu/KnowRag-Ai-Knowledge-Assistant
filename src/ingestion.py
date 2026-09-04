"""
KnowRAG - Document Ingestion Module
-----------------------------------
This module handles loading both the default university knowledge base documents
and dynamically uploaded user documents (.txt and .pdf) with metadata preservation,
deterministic document hashing, and page-level metadata tracking.

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import hashlib
import io
from pathlib import Path
from typing import List, Optional, Tuple

import pypdf
from llama_index.core import Document, SimpleDirectoryReader


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"


def compute_file_hash(file_bytes: bytes, filename: str = "") -> str:
    """
    Compute a deterministic SHA-256 identifier based on filename and content bytes.

    Args:
        file_bytes (bytes): Binary content of the file.
        filename (str): Name of the file.

    Returns:
        str: 32-character hexadecimal hash string.
    """
    hasher = hashlib.sha256()
    hasher.update(filename.lower().encode("utf-8"))
    hasher.update(b"::")
    hasher.update(file_bytes)
    return hasher.hexdigest()[:32]


def load_documents(data_dir: Optional[Path | str] = None) -> List[Document]:
    """
    Load default source documents from the data directory.
    Excludes the uploads directory to keep default knowledge base separate.

    Args:
        data_dir (Optional[Path | str]): Custom data directory path if provided.

    Returns:
        List[Document]: List of LlamaIndex Document objects for default files.
    """
    target_dir = Path(data_dir) if data_dir else DATA_DIR

    if not target_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {target_dir}")

    # Load default txt documents (non-recursive to avoid uploads folder)
    reader = SimpleDirectoryReader(
        input_dir=str(target_dir),
        recursive=False,
        required_exts=[".txt"],
    )

    documents = reader.load_data()

    # Ensure consistent metadata structure for default documents
    for doc in documents:
        fn = doc.metadata.get("file_name", "unknown.txt")
        doc.metadata["source"] = fn
        doc.metadata["file_type"] = "txt"
        doc.metadata["is_uploaded"] = False
        doc.metadata["doc_hash"] = f"default_{fn}"
        doc.metadata["document_id"] = f"default_{fn}"

    return documents


def extract_text_from_txt(file_bytes: bytes, filename: str, doc_id: str) -> List[Document]:
    """
    Extract text from a plain text (.txt) file and build a LlamaIndex Document.

    Args:
        file_bytes (bytes): Binary content of the text file.
        filename (str): Name of the source file.
        doc_id (str): Deterministic document identifier.

    Returns:
        List[Document]: List containing a single LlamaIndex Document with metadata.
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="replace")

    clean_text = text.strip()
    if not clean_text:
        raise ValueError(f"The text file '{filename}' contains no readable content.")

    doc = Document(
        id_=doc_id,
        text=clean_text,
        metadata={
            "source": filename,
            "file_name": filename,
            "document_id": doc_id,
            "doc_hash": doc_id,
            "file_type": "txt",
            "is_uploaded": True,
        },
    )
    return [doc]


def extract_text_from_pdf(file_bytes: bytes, filename: str, doc_id: str) -> List[Document]:
    """
    Extract text page-by-page from a PDF (.pdf) file and build LlamaIndex Documents.

    Args:
        file_bytes (bytes): Binary content of the PDF file.
        filename (str): Name of the source file.
        doc_id (str): Deterministic document identifier.

    Returns:
        List[Document]: List of LlamaIndex Document objects (one per non-empty page).
    """
    try:
        pdf_stream = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_stream)
    except Exception as e:
        raise ValueError(f"Failed to read PDF '{filename}': {e}")

    documents: List[Document] = []
    total_text = ""

    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        clean_page_text = page_text.strip()
        if clean_page_text:
            total_text += clean_page_text
            doc = Document(
                id_=f"{doc_id}_p{page_idx}",
                text=clean_page_text,
                metadata={
                    "source": filename,
                    "file_name": filename,
                    "document_id": doc_id,
                    "doc_hash": doc_id,
                    "file_type": "pdf",
                    "page_number": page_idx,
                    "is_uploaded": True,
                },
            )
            documents.append(doc)

    if not total_text.strip() or not documents:
        raise ValueError(
            f"The PDF file '{filename}' contains no extractable text. "
            "Please ensure the document contains selectable text, not scanned images."
        )

    return documents


def save_uploaded_file(filename: str, file_bytes: bytes, upload_dir: Path = UPLOAD_DIR) -> Path:
    """
    Save uploaded file bytes to the upload directory.

    Args:
        filename (str): Original filename.
        file_bytes (bytes): Binary file data.
        upload_dir (Path): Directory where uploads are stored.

    Returns:
        Path: Path to saved file.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename
    file_path.write_bytes(file_bytes)
    return file_path


def process_uploaded_file(
    filename: str,
    file_bytes: bytes,
    save_to_disk: bool = True,
) -> Tuple[List[Document], str]:
    """
    Process an uploaded file (.txt or .pdf), validate content, and return LlamaIndex Documents.

    Args:
        filename (str): Name of the uploaded file.
        file_bytes (bytes): Raw bytes of the uploaded file.
        save_to_disk (bool): Whether to persist file in data/uploads/.

    Returns:
        Tuple[List[Document], str]: Extracted Document objects and the document hash ID.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError(f"File '{filename}' is empty.")

    doc_id = compute_file_hash(file_bytes=file_bytes, filename=filename)
    lower_name = filename.lower()

    if lower_name.endswith(".txt"):
        documents = extract_text_from_txt(file_bytes=file_bytes, filename=filename, doc_id=doc_id)
    elif lower_name.endswith(".pdf"):
        documents = extract_text_from_pdf(file_bytes=file_bytes, filename=filename, doc_id=doc_id)
    else:
        raise ValueError(
            f"Unsupported file format for '{filename}'. Only .txt and .pdf files are supported."
        )

    if save_to_disk:
        try:
            save_uploaded_file(filename=filename, file_bytes=file_bytes)
        except Exception:
            pass

    return documents, doc_id


if __name__ == "__main__":
    documents = load_documents()

    print(f"Number of default documents loaded: {len(documents)}")
    print()

    for index, document in enumerate(documents, start=1):
        print(f"Document {index}")
        print("-" * 40)
        print(f"File: {document.metadata.get('file_name')}")
        print(f"Characters: {len(document.text)}")
        print(f"Metadata: {document.metadata}")
        print()