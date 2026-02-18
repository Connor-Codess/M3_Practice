#!/usr/bin/env python3
"""
Test RAG query functionality.

Usage:
    python scripts/test_queries.py

This script tests the query engine with various query types
to verify retrieval quality.
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.retrieval import get_query_engine


def print_results(response, max_results=3):
    """Pretty print query results."""
    print(f"Query: {response.query}")
    print(f"Results found: {response.total_found}")
    print("-" * 40)

    for i, result in enumerate(response.results[:max_results], 1):
        print(f"\n[{i}] {result.source} (score: {result.relevance_score:.3f})")
        print(f"    Model: {result.model_type or 'N/A'}")

        # Truncate long text
        assumption_preview = result.assumption[:200]
        if len(result.assumption) > 200:
            assumption_preview += "..."
        print(f"    {assumption_preview}")

        if result.justification:
            justification_preview = result.justification[:150]
            if len(result.justification) > 150:
                justification_preview += "..."
            print(f"    Justification: {justification_preview}")

    print()


def main():
    # Load environment
    env_path = project_root / "keyholder.env"
    load_dotenv(env_path)

    chroma_path = project_root / "chroma_db"

    print("M3 RAG Query Tests")
    print("=" * 50)

    # Initialize query engine
    print("Initializing query engine...")
    try:
        engine = get_query_engine(chroma_path=chroma_path, env_path=env_path)
        doc_count = engine.chroma.count()
        print(f"  Connected to collection with {doc_count} documents")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if doc_count == 0:
        print("Error: No documents in collection. Run ingest_data.py first.")
        sys.exit(1)

    print()
    print("=" * 50)
    print("Running Test Queries")
    print("=" * 50)

    # Test 1: General topic search
    print("\n[Test 1] General topic search: 'housing supply growth prediction'")
    response = engine.search("housing supply growth prediction", n_results=5)
    print_results(response)

    # Test 2: Assumption search with justifications
    print("\n[Test 2] Assumption search: 'population growth assumptions'")
    response = engine.search_assumptions(
        "population growth assumptions",
        n_results=5,
        with_justification=True,
    )
    print_results(response)

    # Test 3: Model search
    print("\n[Test 3] Model search: 'logistic growth model'")
    response = engine.search_models("logistic growth model", n_results=5)
    print_results(response)

    # Test 4: Search with year filter
    print("\n[Test 4] Filtered search: 'homeless' (2020-2024 only)")
    response = engine.search(
        "homeless prediction",
        n_results=5,
        year_min=2020,
        year_max=2024,
    )
    print_results(response)

    # Test 5: Champion-only search
    print("\n[Test 5] Champion-only search: 'differential equations'")
    response = engine.search(
        "differential equations",
        n_results=5,
        placement=1,  # Champions only
    )
    print_results(response)

    # Test 6: Similar problems
    print("\n[Test 6] Similar problems: 'drought water management'")
    response = engine.get_similar_problems(
        "drought affecting water reservoirs and agricultural economy",
        n_results=5,
    )
    print_results(response)

    # Print summary
    print("=" * 50)
    print("All tests completed successfully!")
    print()

    # Interactive mode
    print("Enter queries to test (or 'quit' to exit):")
    while True:
        try:
            query = input("\nQuery> ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue

            response = engine.search(query, n_results=5)
            print_results(response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
