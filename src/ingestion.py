from pathlib import Path

from llama_index.core import SimpleDirectoryReader


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_documents():
    """Load all supported documents from the data directory."""

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}"
        )

    reader = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        recursive=True,
        required_exts=[".txt"],
    )

    documents = reader.load_data()

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Number of documents loaded: {len(documents)}")
    print()

    for index, document in enumerate(documents, start=1):
        print(f"Document {index}")
        print("-" * 40)
        print(f"File: {document.metadata.get('file_name')}")
        print(f"Characters: {len(document.text)}")
        print()