"""Chunk models for the RAG vector store."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ChunkType(Enum):
    """Types of chunks extracted from submissions."""
    ASSUMPTION = "assumption"
    MODEL = "model"
    SENSITIVITY = "sensitivity"
    COMMENT = "comment"
    STRENGTHS_WEAKNESSES = "strengths_weaknesses"


@dataclass
class Chunk:
    """A single chunk ready for embedding and storage."""
    id: str  # Format: "2024_S1_Q1_assumption_1"
    text: str
    chunk_type: ChunkType

    # Source identifiers
    year: int
    placement: int  # 1, 2, 3
    question: Optional[str] = None  # "Q1", "Q2", "Q3" (None for comments)
    chunk_index: int = 1
    source_file: str = ""

    # Content metadata
    has_justification: bool = False
    topic_tags: list[str] = field(default_factory=list)
    model_type: Optional[str] = None
    math_methods: list[str] = field(default_factory=list)

    # Embedding (populated after embedding generation)
    embedding: Optional[list[float]] = None

    def to_metadata(self) -> dict:
        """Convert to ChromaDB metadata format."""
        return {
            "year": self.year,
            "placement": self.placement,
            "question": self.question or "",
            "section_type": self.chunk_type.value,
            "chunk_index": self.chunk_index,
            "has_justification": self.has_justification,
            "topic_tags": ",".join(self.topic_tags),
            "model_type": self.model_type or "",
            "math_methods": ",".join(self.math_methods),
            "source_file": self.source_file,
        }

    def to_embedding_text(self) -> str:
        """Format text for embedding with context."""
        placement_name = {1: "Champion", 2: "Runner-up", 3: "Third"}.get(
            self.placement, f"Place {self.placement}"
        )

        lines = [
            f"[SECTION: {self.chunk_type.value}]",
            f"[YEAR: {self.year}] [PLACEMENT: {placement_name}]",
        ]

        if self.question:
            lines.append(f"[QUESTION: {self.question}]")

        lines.append("")
        lines.append(self.text)

        if self.model_type:
            lines.append("")
            lines.append(f"[MODEL TYPE: {self.model_type}]")

        if self.math_methods:
            lines.append(f"[METHODS: {', '.join(self.math_methods)}]")

        return "\n".join(lines)
