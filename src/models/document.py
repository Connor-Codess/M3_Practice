"""Source document models representing M3 competition submissions."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Assumption:
    """A single assumption with optional justification."""
    text: str
    justification: Optional[str] = None
    index: int = 0


@dataclass
class Model:
    """Model description from a submission."""
    development: str
    execution: Optional[str] = None
    variables: list[dict] = field(default_factory=list)


@dataclass
class Question:
    """A single question (Q1/Q2/Q3) from a submission."""
    number: str  # "Q1", "Q2", "Q3"
    definition: Optional[str] = None
    assumptions: list[Assumption] = field(default_factory=list)
    model: Optional[Model] = None
    results: Optional[str] = None
    sensitivity_analysis: Optional[str] = None
    strengths_weaknesses: Optional[str] = None


@dataclass
class SourceDocument:
    """A complete M3 competition submission."""
    year: int
    placement: int  # 1=Champion, 2=Runner-up, 3=Third
    source_file: str
    comments: list[str] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)

    @property
    def placement_name(self) -> str:
        """Human-readable placement name."""
        names = {1: "Champion", 2: "Runner-up", 3: "Third Place"}
        return names.get(self.placement, f"Place {self.placement}")

    @property
    def placement_code(self) -> str:
        """Short placement code for IDs."""
        return f"S{self.placement}"
