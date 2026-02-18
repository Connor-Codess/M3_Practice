"""ChromaDB client for vector storage and retrieval."""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


COLLECTION_NAME = "m3_rag_historical"


class ChromaClient:
    """Client for interacting with ChromaDB vector store."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
    ):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Directory for persistent storage.
                If None, uses in-memory storage.
            collection_name: Name of the collection to use.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        if persist_directory:
            persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

        self._collection = None

    @property
    def collection(self):
        """Get or create the collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "M3 Challenge winning submissions 2011-2024",
                    "hnsw:space": "cosine",
                },
            )
        return self._collection

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """
        Add documents to the collection.

        Args:
            ids: Unique identifiers for each document.
            documents: Text content of each document.
            embeddings: Pre-computed embeddings for each document.
            metadatas: Metadata dictionaries for each document.
        """
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def upsert_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """
        Upsert documents (update if exists, insert if not).

        Args:
            ids: Unique identifiers for each document.
            documents: Text content of each document.
            embeddings: Pre-computed embeddings for each document.
            metadatas: Metadata dictionaries for each document.
        """
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> dict:
        """
        Query the collection by embedding similarity.

        Args:
            query_embedding: Query vector.
            n_results: Number of results to return.
            where: Metadata filter conditions.
            where_document: Document content filter conditions.

        Returns:
            Dictionary with ids, documents, metadatas, distances.
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )

    def query_text(
        self,
        query_text: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Query using both text and embedding.

        Args:
            query_text: Original query text (for logging/debugging).
            query_embedding: Query vector.
            n_results: Number of results to return.
            where: Metadata filter conditions.

        Returns:
            Dictionary with ids, documents, metadatas, distances, and query.
        """
        results = self.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
        )
        results["query"] = query_text
        return results

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def get_by_id(self, doc_id: str) -> Optional[dict]:
        """Get a document by its ID."""
        results = self.collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "document": results["documents"][0],
                "metadata": results["metadatas"][0],
            }
        return None

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        self._collection = None

    def reset(self) -> None:
        """Reset the collection (delete and recreate)."""
        try:
            self.delete_collection()
        except ValueError:
            pass  # Collection doesn't exist
        self._collection = None  # Will be recreated on next access


def get_chroma_client(
    persist_directory: Optional[Path] = None,
    use_default_path: bool = True,
) -> ChromaClient:
    """
    Create a ChromaDB client with sensible defaults.

    Args:
        persist_directory: Custom persistence path.
        use_default_path: If True and persist_directory is None,
            use 'chroma_db' in project root.

    Returns:
        Configured ChromaClient instance.
    """
    if persist_directory is None and use_default_path:
        persist_directory = Path("chroma_db")

    return ChromaClient(persist_directory=persist_directory)
