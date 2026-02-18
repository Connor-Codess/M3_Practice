"""Ingest chunks into the ChromaDB vector store."""

from pathlib import Path
from typing import Optional

from ..models import Chunk, SourceDocument
from ..data_processing import extract_document, chunk_document
from ..embeddings import EmbeddingClient
from .chroma_client import ChromaClient


def ingest_chunks(
    chunks: list[Chunk],
    chroma_client: ChromaClient,
    embedding_client: EmbeddingClient,
    batch_size: int = 50,
    show_progress: bool = True,
) -> int:
    """
    Ingest a list of chunks into ChromaDB.

    Args:
        chunks: List of Chunk objects to ingest.
        chroma_client: ChromaDB client.
        embedding_client: Embedding client for generating vectors.
        batch_size: Number of chunks to process per batch.
        show_progress: Whether to print progress.

    Returns:
        Number of chunks ingested.
    """
    if not chunks:
        return 0

    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1

        if show_progress:
            print(f"Processing batch {batch_num}/{total_batches}...")

        # Prepare texts for embedding
        texts = [chunk.to_embedding_text() for chunk in batch]

        # Generate embeddings
        embeddings = embedding_client.embed_batch(texts, show_progress=False)

        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in batch]
        documents = [chunk.text for chunk in batch]
        metadatas = [chunk.to_metadata() for chunk in batch]

        # Upsert to ChromaDB
        chroma_client.upsert_documents(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        if show_progress:
            print(f"  Ingested {len(batch)} chunks")

    return len(chunks)


def ingest_documents(
    source_dir: Path,
    chroma_client: ChromaClient,
    embedding_client: EmbeddingClient,
    batch_size: int = 50,
    show_progress: bool = True,
) -> dict:
    """
    Ingest all documents from a source directory.

    Args:
        source_dir: Directory containing source JSON files.
        chroma_client: ChromaDB client.
        embedding_client: Embedding client.
        batch_size: Chunks per batch for embedding.
        show_progress: Whether to print progress.

    Returns:
        Dictionary with ingestion statistics.
    """
    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "chunks_created": 0,
        "chunks_deduplicated": 0,
        "chunks_ingested": 0,
    }

    json_files = sorted(source_dir.glob("*.json"))

    if show_progress:
        print(f"Found {len(json_files)} JSON files in {source_dir}")

    all_chunks = []

    # Extract and chunk all documents
    for filepath in json_files:
        try:
            if show_progress:
                print(f"Processing {filepath.name}...")

            doc = extract_document(filepath)
            chunks = chunk_document(doc)

            all_chunks.extend(chunks)
            stats["files_processed"] += 1
            stats["chunks_created"] += len(chunks)

            if show_progress:
                print(f"  Created {len(chunks)} chunks")

        except Exception as e:
            print(f"  Error processing {filepath.name}: {e}")
            stats["files_failed"] += 1

    # Deduplicate chunks by ID (keep first occurrence)
    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.id not in seen_ids:
            seen_ids.add(chunk.id)
            unique_chunks.append(chunk)

    stats["chunks_deduplicated"] = len(all_chunks) - len(unique_chunks)

    if show_progress and stats["chunks_deduplicated"] > 0:
        print(f"\nRemoved {stats['chunks_deduplicated']} duplicate chunks")

    # Ingest unique chunks
    if unique_chunks:
        if show_progress:
            print(f"Ingesting {len(unique_chunks)} unique chunks...")

        ingested = ingest_chunks(
            chunks=unique_chunks,
            chroma_client=chroma_client,
            embedding_client=embedding_client,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        stats["chunks_ingested"] = ingested

    return stats


def ingest_single_file(
    filepath: Path,
    chroma_client: ChromaClient,
    embedding_client: EmbeddingClient,
    show_progress: bool = True,
) -> int:
    """
    Ingest a single source file.

    Args:
        filepath: Path to JSON file.
        chroma_client: ChromaDB client.
        embedding_client: Embedding client.
        show_progress: Whether to print progress.

    Returns:
        Number of chunks ingested.
    """
    doc = extract_document(filepath)
    chunks = chunk_document(doc)

    return ingest_chunks(
        chunks=chunks,
        chroma_client=chroma_client,
        embedding_client=embedding_client,
        show_progress=show_progress,
    )
