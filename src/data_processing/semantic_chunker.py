"""Create semantic chunks from M3 submission documents."""

import re
from typing import Optional

from ..models import SourceDocument, Chunk, ChunkType


# Common mathematical model patterns for auto-tagging
MODEL_PATTERNS = {
    "logistic": ["logistic", "carrying capacity", "s-curve"],
    "linear_regression": ["linear regression", "linear model", "least squares"],
    "multivariate_regression": ["multivariate", "multiple regression", "mlr"],
    "differential_equations": ["differential equation", "dv/dt", "ode"],
    "markov_chain": ["markov", "state transition", "transition matrix"],
    "arima": ["arima", "autoregressive", "time series"],
    "monte_carlo": ["monte carlo", "simulation", "random sampling"],
    "optimization": ["optimization", "minimize", "maximize", "objective function"],
    "agent_based": ["agent-based", "agent model", "rational agent"],
    "random_forest": ["random forest", "decision tree", "ensemble"],
    "granger_causality": ["granger", "causality test"],
    "var_model": ["var ", "vector autoregressive"],
}

# Topic keywords for auto-tagging
TOPIC_KEYWORDS = {
    "housing": ["housing", "homes", "residential", "units"],
    "homelessness": ["homeless", "unhoused", "shelter"],
    "population": ["population", "demographic", "growth"],
    "economics": ["economic", "gdp", "income", "price"],
    "environment": ["drought", "water", "climate", "environmental"],
    "energy": ["energy", "power", "electricity", "renewable"],
    "transportation": ["traffic", "vehicle", "transport", "mobility"],
    "health": ["health", "disease", "medical", "opioid"],
    "agriculture": ["agriculture", "farming", "crop", "food"],
}


def detect_model_type(text: str) -> Optional[str]:
    """Detect the mathematical model type from text."""
    text_lower = text.lower()
    for model_type, patterns in MODEL_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return model_type
    return None


def detect_topics(text: str) -> list[str]:
    """Detect topic tags from text."""
    text_lower = text.lower()
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                topics.append(topic)
                break
    return topics


def detect_math_methods(text: str) -> list[str]:
    """Detect mathematical methods mentioned in text."""
    methods = []
    text_lower = text.lower()

    method_patterns = [
        (r"differential equation", "differential_equations"),
        (r"regression", "regression"),
        (r"correlation", "correlation"),
        (r"simulation", "simulation"),
        (r"optimization", "optimization"),
        (r"probability", "probability"),
        (r"statistics", "statistics"),
        (r"calculus", "calculus"),
        (r"linear algebra", "linear_algebra"),
        (r"extrapolat", "extrapolation"),
        (r"interpolat", "interpolation"),
    ]

    for pattern, method in method_patterns:
        if re.search(pattern, text_lower):
            methods.append(method)

    return list(set(methods))


def create_assumption_chunk(
    doc: SourceDocument,
    question: str,
    assumption_text: str,
    justification: Optional[str],
    index: int,
) -> Chunk:
    """Create a chunk from an assumption."""
    # Combine assumption with justification for richer content
    if justification:
        text = f"ASSUMPTION: {assumption_text}\n\nJUSTIFICATION: {justification}"
    else:
        text = f"ASSUMPTION: {assumption_text}"

    chunk_id = f"{doc.year}_{doc.placement_code}_{question}_assumption_{index}"

    return Chunk(
        id=chunk_id,
        text=text,
        chunk_type=ChunkType.ASSUMPTION,
        year=doc.year,
        placement=doc.placement,
        question=question,
        chunk_index=index,
        source_file=doc.source_file,
        has_justification=justification is not None,
        topic_tags=detect_topics(text),
        model_type=detect_model_type(text),
        math_methods=detect_math_methods(text),
    )


def create_model_chunk(
    doc: SourceDocument,
    question: str,
    development: str,
    execution: Optional[str],
) -> Chunk:
    """Create a chunk from model development/execution."""
    parts = [f"MODEL DEVELOPMENT:\n{development}"]
    if execution:
        parts.append(f"\nMODEL EXECUTION:\n{execution}")

    text = "\n".join(parts)
    chunk_id = f"{doc.year}_{doc.placement_code}_{question}_model"

    return Chunk(
        id=chunk_id,
        text=text,
        chunk_type=ChunkType.MODEL,
        year=doc.year,
        placement=doc.placement,
        question=question,
        chunk_index=1,
        source_file=doc.source_file,
        has_justification=False,
        topic_tags=detect_topics(text),
        model_type=detect_model_type(text),
        math_methods=detect_math_methods(text),
    )


def create_sensitivity_chunk(
    doc: SourceDocument,
    question: str,
    sensitivity_text: str,
) -> Chunk:
    """Create a chunk from sensitivity analysis."""
    text = f"SENSITIVITY ANALYSIS:\n{sensitivity_text}"
    chunk_id = f"{doc.year}_{doc.placement_code}_{question}_sensitivity"

    return Chunk(
        id=chunk_id,
        text=text,
        chunk_type=ChunkType.SENSITIVITY,
        year=doc.year,
        placement=doc.placement,
        question=question,
        chunk_index=1,
        source_file=doc.source_file,
        has_justification=False,
        topic_tags=detect_topics(text),
        model_type=detect_model_type(text),
        math_methods=detect_math_methods(text),
    )


def create_strengths_chunk(
    doc: SourceDocument,
    question: str,
    strengths_text: str,
) -> Chunk:
    """Create a chunk from strengths/weaknesses."""
    text = f"STRENGTHS AND WEAKNESSES:\n{strengths_text}"
    chunk_id = f"{doc.year}_{doc.placement_code}_{question}_strengths"

    return Chunk(
        id=chunk_id,
        text=text,
        chunk_type=ChunkType.STRENGTHS_WEAKNESSES,
        year=doc.year,
        placement=doc.placement,
        question=question,
        chunk_index=1,
        source_file=doc.source_file,
        has_justification=False,
        topic_tags=detect_topics(text),
        model_type=detect_model_type(text),
        math_methods=detect_math_methods(text),
    )


def create_comment_chunk(
    doc: SourceDocument,
    comment_text: str,
    index: int,
) -> Chunk:
    """Create a chunk from a judge comment."""
    text = f"JUDGE COMMENT:\n{comment_text}"
    chunk_id = f"{doc.year}_{doc.placement_code}_comment_{index}"

    return Chunk(
        id=chunk_id,
        text=text,
        chunk_type=ChunkType.COMMENT,
        year=doc.year,
        placement=doc.placement,
        question=None,  # Comments are at document level
        chunk_index=index,
        source_file=doc.source_file,
        has_justification=False,
        topic_tags=detect_topics(text),
        model_type=None,
        math_methods=[],
    )


def chunk_document(doc: SourceDocument) -> list[Chunk]:
    """
    Create all chunks from a source document.

    Extracts:
    - Individual assumptions (with justifications when available)
    - Model development/execution sections
    - Sensitivity analysis sections
    - Strengths/weaknesses sections
    - Judge comments
    """
    chunks = []

    # Process each question
    for question in doc.questions:
        # Assumption chunks - one per assumption
        for assumption in question.assumptions:
            if assumption.text.strip():
                chunk = create_assumption_chunk(
                    doc=doc,
                    question=question.number,
                    assumption_text=assumption.text,
                    justification=assumption.justification,
                    index=assumption.index,
                )
                chunks.append(chunk)

        # Model chunk
        if question.model and question.model.development.strip():
            chunk = create_model_chunk(
                doc=doc,
                question=question.number,
                development=question.model.development,
                execution=question.model.execution,
            )
            chunks.append(chunk)

        # Sensitivity analysis chunk
        if question.sensitivity_analysis and question.sensitivity_analysis.strip():
            chunk = create_sensitivity_chunk(
                doc=doc,
                question=question.number,
                sensitivity_text=question.sensitivity_analysis,
            )
            chunks.append(chunk)

        # Strengths/weaknesses chunk
        if question.strengths_weaknesses and question.strengths_weaknesses.strip():
            chunk = create_strengths_chunk(
                doc=doc,
                question=question.number,
                strengths_text=question.strengths_weaknesses,
            )
            chunks.append(chunk)

    # Comment chunks - one per comment
    for i, comment in enumerate(doc.comments, 1):
        if comment.strip():
            chunk = create_comment_chunk(doc=doc, comment_text=comment, index=i)
            chunks.append(chunk)

    return chunks
