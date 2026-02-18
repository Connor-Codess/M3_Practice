"""Result models for RAG queries."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AssumptionResult:
    """A single assumption retrieved from the RAG system."""
    assumption: str
    justification: Optional[str]
    source: str  # "2024 Champion Q1"
    relevance_score: float
    model_type: Optional[str] = None
    topic_tags: list[str] = field(default_factory=list)
    year: int = 0

    def to_citation(self) -> str:
        """Format as citation string for LLM context."""
        citation = f"[{self.source}]"
        if self.model_type:
            citation += f" ({self.model_type})"
        return citation

    def to_context_block(self) -> str:
        """Format as context block for Historian agent."""
        lines = [
            f"SOURCE: {self.source}",
            f"ASSUMPTION: {self.assumption}",
        ]
        if self.justification:
            lines.append(f"JUSTIFICATION: {self.justification}")
        if self.model_type:
            lines.append(f"MODEL TYPE: {self.model_type}")
        lines.append(f"RELEVANCE: {self.relevance_score:.2f}")
        return "\n".join(lines)


@dataclass
class RAGResponse:
    """Complete response from a RAG query."""
    query: str
    results: list[AssumptionResult]
    total_found: int
    search_params: dict = field(default_factory=dict)

    def to_context(self, max_results: int = 5) -> str:
        """Format top results as context for LLM."""
        blocks = []
        for i, result in enumerate(self.results[:max_results], 1):
            blocks.append(f"--- Result {i} ---")
            blocks.append(result.to_context_block())
        return "\n\n".join(blocks)

    def get_unique_sources(self) -> list[str]:
        """Get list of unique source citations."""
        return list(dict.fromkeys(r.source for r in self.results))
