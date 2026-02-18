#!/usr/bin/env python3
"""
Ingest M3 Competition data into ChromaDB.

Usage:
    python scripts/ingest_data.py

This script:
1. Loads all JSON files from 'M3 Compilation 2026/'
2. Normalizes schemas and extracts content
3. Creates semantic chunks
4. Generates embeddings using OpenAI
5. Stores everything in ChromaDB

Prerequisites:
- Set OPENAI_API_KEY in keyholder.env
- Install dependencies: pip install -r requirements.txt
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.embeddings import get_embedding_client
from src.vector_store import get_chroma_client, ingest_documents


def main():
    # Load environment variables
    env_path = project_root / "keyholder.env"
    load_dotenv(env_path)

    print("M3 RAG Data Ingestion")
    print("=" * 50)

    # Setup paths
    source_dir = project_root / "M3 Compilation 2026"
    chroma_path = project_root / "chroma_db"

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)

    print(f"Source directory: {source_dir}")
    print(f"ChromaDB path: {chroma_path}")
    print()

    # Initialize clients
    print("Initializing embedding client...")
    try:
        embedding_client = get_embedding_client(env_path)
        print("  Embedding client ready (text-embedding-3-large)")
    except Exception as e:
        print(f"Error initializing embedding client: {e}")
        sys.exit(1)

    print("Initializing ChromaDB client...")
    chroma_client = get_chroma_client(persist_directory=chroma_path)
    print(f"  ChromaDB ready (collection: {chroma_client.collection_name})")
    print()

    # Check existing data
    existing_count = chroma_client.count()
    if existing_count > 0:
        print(f"Warning: Collection already contains {existing_count} documents.")
        response = input("Reset collection and re-ingest? [y/N]: ")
        if response.lower() == "y":
            print("Resetting collection...")
            chroma_client.reset()
        else:
            print("Aborting.")
            sys.exit(0)

    # Run ingestion
    print("Starting ingestion...")
    print("-" * 50)

    stats = ingest_documents(
        source_dir=source_dir,
        chroma_client=chroma_client,
        embedding_client=embedding_client,
        batch_size=50,
        show_progress=True,
    )

    # Print summary
    print()
    print("=" * 50)
    print("Ingestion Complete")
    print("=" * 50)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files failed: {stats['files_failed']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Chunks ingested: {stats['chunks_ingested']}")
    print()
    print(f"Total documents in collection: {chroma_client.count()}")


if __name__ == "__main__":
    main()
