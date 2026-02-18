"""Query engine for RAG retrieval with hybrid search capabilities."""

from pathlib import Path
from typing import Optional

from ..models import AssumptionResult, RAGResponse, ChunkType
from ..embeddings import EmbeddingClient, get_embedding_client
from ..vector_store import ChromaClient, get_chroma_client


class QueryEngine:
    """
    Query engine for retrieving relevant content from the M3 RAG system.

    Supports:
    - Semantic search by topic/query
    - Hybrid search with metadata filtering
    - Specialized assumption retrieval for the Historian agent
    """

    def __init__(
        self,
        chroma_client: ChromaClient,
        embedding_client: EmbeddingClient,
    ):
        """
        Initialize the query engine.

        Args:
            chroma_client: ChromaDB client for vector search.
            embedding_client: Client for generating query embeddings.
        """
        self.chroma = chroma_client
        self.embeddings = embedding_client

    def search(
        self,
        query: str,
        n_results: int = 10,
        section_type: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        placement: Optional[int] = None,
        question: Optional[str] = None,
    ) -> RAGResponse:
        """
        Search for relevant content with optional filters.

        Args:
            query: Search query text.
            n_results: Maximum number of results.
            section_type: Filter by section type (assumption, model, etc.).
            year_min: Minimum year (inclusive).
            year_max: Maximum year (inclusive).
            placement: Filter by placement (1=Champion, 2=Runner-up, 3=Third).
            question: Filter by question number (Q1, Q2, Q3).

        Returns:
            RAGResponse with matching results.
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_text(query)

        # Build metadata filter
        where = self._build_where_clause(
            section_type=section_type,
            year_min=year_min,
            year_max=year_max,
            placement=placement,
            question=question,
        )

        # Execute search
        results = self.chroma.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where if where else None,
        )

        # Convert to AssumptionResult objects
        assumption_results = self._convert_results(results)

        return RAGResponse(
            query=query,
            results=assumption_results,
            total_found=len(assumption_results),
            search_params={
                "section_type": section_type,
                "year_range": (year_min, year_max) if year_min or year_max else None,
                "placement": placement,
                "question": question,
            },
        )

    def search_assumptions(
        self,
        query: str,
        n_results: int = 15,
        with_justification: bool = True,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> RAGResponse:
        """
        Search specifically for assumptions (optimized for Historian agent).

        Args:
            query: Problem context or topic to search for.
            n_results: Maximum number of results.
            with_justification: Only return assumptions with justifications.
            year_min: Minimum year (inclusive).
            year_max: Maximum year (inclusive).

        Returns:
            RAGResponse with assumption results formatted for LLM context.
        """
        # Build filter conditions
        conditions = [{"section_type": "assumption"}]

        if with_justification:
            conditions.append({"has_justification": True})

        if year_min is not None:
            conditions.append({"year": {"$gte": year_min}})

        if year_max is not None:
            conditions.append({"year": {"$lte": year_max}})

        # ChromaDB requires $and for multiple conditions
        where = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        query_embedding = self.embeddings.embed_text(query)

        results = self.chroma.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
        )

        assumption_results = self._convert_results(results)

        return RAGResponse(
            query=query,
            results=assumption_results,
            total_found=len(assumption_results),
            search_params={
                "section_type": "assumption",
                "with_justification": with_justification,
                "year_range": (year_min, year_max),
            },
        )

    def search_models(
        self,
        query: str,
        n_results: int = 10,
        model_type: Optional[str] = None,
    ) -> RAGResponse:
        """
        Search for model development content.

        Args:
            query: Search query describing the modeling problem.
            n_results: Maximum number of results.
            model_type: Filter by specific model type.

        Returns:
            RAGResponse with model-related results.
        """
        where = {"section_type": "model"}

        if model_type:
            where["model_type"] = model_type

        query_embedding = self.embeddings.embed_text(query)

        results = self.chroma.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
        )

        return RAGResponse(
            query=query,
            results=self._convert_results(results),
            total_found=len(results["ids"][0]) if results["ids"] else 0,
            search_params={"section_type": "model", "model_type": model_type},
        )

    def search_by_topic(
        self,
        topic: str,
        n_results: int = 10,
    ) -> RAGResponse:
        """
        Search across all content types by topic.

        Args:
            topic: Topic to search for (e.g., "housing", "drought").
            n_results: Maximum number of results.

        Returns:
            RAGResponse with mixed content types.
        """
        return self.search(query=topic, n_results=n_results)

    def get_similar_problems(
        self,
        problem_description: str,
        n_results: int = 5,
    ) -> RAGResponse:
        """
        Find similar problems from past competitions.

        Searches across all content types to find relevant past work.

        Args:
            problem_description: Description of the current problem.
            n_results: Maximum number of results.

        Returns:
            RAGResponse with similar past problem content.
        """
        return self.search(query=problem_description, n_results=n_results)

    def _build_where_clause(
        self,
        section_type: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        placement: Optional[int] = None,
        question: Optional[str] = None,
    ) -> dict:
        """Build ChromaDB where clause from filter parameters."""
        conditions = []

        if section_type:
            conditions.append({"section_type": section_type})

        if placement is not None:
            conditions.append({"placement": placement})

        if question:
            conditions.append({"question": question})

        # Year range handling
        if year_min is not None and year_max is not None:
            conditions.append({"year": {"$gte": year_min}})
            conditions.append({"year": {"$lte": year_max}})
        elif year_min is not None:
            conditions.append({"year": {"$gte": year_min}})
        elif year_max is not None:
            conditions.append({"year": {"$lte": year_max}})

        if not conditions:
            return {}

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def _convert_results(self, results: dict) -> list[AssumptionResult]:
        """Convert ChromaDB results to AssumptionResult objects."""
        if not results["ids"] or not results["ids"][0]:
            return []

        assumption_results = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
            # Parse assumption and justification from document text
            assumption_text = doc
            justification = None

            if "JUSTIFICATION:" in doc:
                parts = doc.split("JUSTIFICATION:", 1)
                assumption_text = parts[0].replace("ASSUMPTION:", "").strip()
                justification = parts[1].strip()
            elif "ASSUMPTION:" in doc:
                assumption_text = doc.replace("ASSUMPTION:", "").strip()

            # Build source citation
            year = meta.get("year", "")
            placement = meta.get("placement", 1)
            question = meta.get("question", "")

            placement_names = {1: "Champion", 2: "Runner-up", 3: "Third"}
            placement_name = placement_names.get(placement, f"Place {placement}")

            source = f"{year} {placement_name}"
            if question:
                source += f" {question}"

            # Parse topic tags
            topic_tags = meta.get("topic_tags", "")
            if isinstance(topic_tags, str):
                topic_tags = [t.strip() for t in topic_tags.split(",") if t.strip()]

            # Relevance score (convert distance to similarity)
            relevance_score = 1 - distance  # Cosine distance to similarity

            assumption_results.append(
                AssumptionResult(
                    assumption=assumption_text,
                    justification=justification,
                    source=source,
                    relevance_score=relevance_score,
                    model_type=meta.get("model_type"),
                    topic_tags=topic_tags,
                    year=year,
                )
            )

        return assumption_results


def get_query_engine(
    chroma_path: Optional[Path] = None,
    env_path: Optional[Path] = None,
) -> QueryEngine:
    """
    Create a query engine with default configuration.

    Args:
        chroma_path: Path to ChromaDB storage.
        env_path: Path to environment file with API keys.

    Returns:
        Configured QueryEngine instance.
    """
    chroma_client = get_chroma_client(persist_directory=chroma_path)
    embedding_client = get_embedding_client(env_path=env_path)

    return QueryEngine(
        chroma_client=chroma_client,
        embedding_client=embedding_client,
    )
