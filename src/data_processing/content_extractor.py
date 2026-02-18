"""Extract structured content from normalized M3 submission data."""

from pathlib import Path

from ..models import SourceDocument, Question, Assumption, Model
from .schema_normalizer import normalize_source_file


def extract_document(filepath: Path) -> SourceDocument:
    """
    Extract a SourceDocument from a JSON file.

    Handles schema normalization internally and returns a fully
    structured SourceDocument object.
    """
    normalized = normalize_source_file(filepath)

    questions = []
    for q_data in normalized["questions"]:
        # Build assumption objects
        assumptions = [
            Assumption(
                text=a["text"],
                justification=a.get("justification"),
                index=a.get("index", i + 1),
            )
            for i, a in enumerate(q_data["assumptions"])
        ]

        # Build model object
        model_data = q_data["model"]
        model = Model(
            development=model_data["development"],
            execution=model_data.get("execution"),
            variables=model_data.get("variables", []),
        ) if model_data["development"] else None

        # Build question object
        question = Question(
            number=q_data["number"],
            definition=q_data.get("definition"),
            assumptions=assumptions,
            model=model,
            results=q_data.get("results"),
            sensitivity_analysis=q_data.get("sensitivity_analysis"),
            strengths_weaknesses=q_data.get("strengths_weaknesses"),
        )
        questions.append(question)

    return SourceDocument(
        year=normalized["year"],
        placement=normalized["placement"],
        source_file=normalized["source_file"],
        comments=normalized["comments"],
        questions=questions,
    )


def extract_all_documents(source_dir: Path) -> list[SourceDocument]:
    """Extract all documents from a directory of JSON files."""
    documents = []

    for filepath in sorted(source_dir.glob("*.json")):
        try:
            doc = extract_document(filepath)
            documents.append(doc)
        except Exception as e:
            print(f"Warning: Failed to extract {filepath.name}: {e}")

    return documents
