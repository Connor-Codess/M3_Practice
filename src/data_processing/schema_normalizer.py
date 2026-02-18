"""Normalize JSON schema variations across different years of M3 submissions."""

import json
import re
from pathlib import Path
from typing import Any


def extract_year_placement(filename: str) -> tuple[int, int]:
    """Extract year and placement from filename like '2024_S1.json'."""
    match = re.match(r"(\d{4})_S(\d+)\.json", filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return int(match.group(1)), int(match.group(2))


def normalize_placement(placement: Any) -> int:
    """Normalize placement to integer (1=Champion, 2=Runner-up, 3=Third)."""
    if isinstance(placement, int):
        return placement

    placement_str = str(placement).lower()

    # Handle string formats like "Champions (Summa Cum Laude Team Prize)"
    if "champion" in placement_str or "summa cum laude" in placement_str:
        return 1
    if "runner" in placement_str or "second" in placement_str:
        return 2
    if "third" in placement_str or "honorable" in placement_str:
        return 3

    # Try to extract number
    match = re.search(r"(\d+)", placement_str)
    if match:
        return int(match.group(1))

    return 1  # Default to champion if unknown


def normalize_comments(comments: Any) -> list[str]:
    """Normalize comments to list of strings."""
    if comments is None:
        return []
    if isinstance(comments, str):
        return [comments]
    if isinstance(comments, list):
        return [str(c) for c in comments]
    return []


def normalize_assumptions(assumptions: Any) -> list[dict]:
    """
    Normalize assumptions to list of {text, justification} dicts.

    Handles both formats:
    - 2024: [{Assumption: str, Justification: str}, ...]
    - 2011: [str, str, ...]
    """
    if assumptions is None:
        return []

    result = []

    if isinstance(assumptions, list):
        for i, item in enumerate(assumptions):
            if isinstance(item, dict):
                # 2024 format: {Assumption, Justification}
                text = item.get("Assumption") or item.get("assumption") or ""
                justification = item.get("Justification") or item.get("justification")
                result.append({
                    "text": str(text),
                    "justification": str(justification) if justification else None,
                    "index": i + 1,
                })
            else:
                # 2011 format: plain string
                result.append({
                    "text": str(item),
                    "justification": None,
                    "index": i + 1,
                })
    elif isinstance(assumptions, str):
        result.append({
            "text": assumptions,
            "justification": None,
            "index": 1,
        })

    return result


def normalize_model(model: Any) -> dict:
    """
    Normalize model to {development, execution, variables} dict.

    Handles both formats:
    - 2024: {Model Development, Model Execution, Variables}
    - 2011: string or [string, ...]
    """
    if model is None:
        return {"development": "", "execution": None, "variables": []}

    if isinstance(model, dict):
        development = model.get("Model Development") or model.get("development") or ""
        execution = model.get("Model Execution") or model.get("execution")
        variables = model.get("Variables") or model.get("variables") or []
        return {
            "development": str(development),
            "execution": str(execution) if execution else None,
            "variables": variables if isinstance(variables, list) else [],
        }

    if isinstance(model, list):
        return {
            "development": " ".join(str(m) for m in model),
            "execution": None,
            "variables": [],
        }

    return {
        "development": str(model),
        "execution": None,
        "variables": [],
    }


def normalize_question(question_data: dict, question_num: str) -> dict:
    """Normalize a single question's data."""
    # Handle various field name variations
    definition = (
        question_data.get("Defining the Problem")
        or question_data.get("Definition")
        or question_data.get("definition")
        or ""
    )

    assumptions = normalize_assumptions(
        question_data.get("Assumptions")
        or question_data.get("Assumption")
        or question_data.get("assumptions")
    )

    model = normalize_model(
        question_data.get("Model")
        or question_data.get("model")
    )

    results = question_data.get("Results") or question_data.get("results") or ""
    if isinstance(results, list):
        results = " ".join(str(r) for r in results)

    discussion = question_data.get("Discussion") or question_data.get("discussion") or ""

    sensitivity = (
        question_data.get("Sensitivity Analysis")
        or question_data.get("sensitivity_analysis")
        or ""
    )

    strengths = (
        question_data.get("Strengths & Weaknesses")
        or question_data.get("Strength and Weakness")
        or question_data.get("strengths_weaknesses")
        or ""
    )
    if isinstance(strengths, list):
        strengths = " ".join(str(s) for s in strengths)

    return {
        "number": question_num,
        "definition": str(definition),
        "assumptions": assumptions,
        "model": model,
        "results": str(results),
        "discussion": str(discussion),
        "sensitivity_analysis": str(sensitivity),
        "strengths_weaknesses": str(strengths),
    }


def clean_json_content(content: str) -> str:
    r"""
    Clean non-standard escape sequences from JSON content.

    Some JSON files have escaped brackets (backslash-bracket) and other
    non-standard escapes that need to be fixed before parsing.
    """
    # Fix double-backslash-quote pattern (\\") which should be escaped quote (\")
    content = content.replace('\\\\"', '\\"')

    # Remove backslashes before non-valid-JSON-escape chars
    # Valid JSON escapes: " \ / b f n r t u
    def replace_invalid_escape(match):
        char = match.group(1)
        if char in r'"\\/bfnrtu':
            return match.group(0)  # Keep as-is
        return char  # Remove backslash

    content = re.sub(r'\\(.)', replace_invalid_escape, content)

    # Fix incomplete JSON by balancing braces
    # Count unmatched braces (simple heuristic, not perfect for all edge cases)
    open_braces = content.count('{') - content.count('}')
    if open_braces > 0:
        content = content.rstrip() + '\n' + '}' * open_braces

    return content


def normalize_source_file(filepath: Path) -> dict:
    """
    Load and normalize a source JSON file to a consistent schema.

    Returns a dict with structure:
    {
        "year": int,
        "placement": int,
        "source_file": str,
        "comments": [str],
        "questions": [
            {
                "number": "Q1",
                "definition": str,
                "assumptions": [{text, justification, index}],
                "model": {development, execution, variables},
                "results": str,
                "discussion": str,
                "sensitivity_analysis": str,
                "strengths_weaknesses": str,
            },
            ...
        ]
    }
    """
    year, placement_from_file = extract_year_placement(filepath.name)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Clean non-standard escapes before parsing
    content = clean_json_content(content)
    raw_data = json.loads(content)

    # Handle team wrapper (older format)
    # If top-level key looks like "Team #123", unwrap it
    if len(raw_data) == 1:
        key = list(raw_data.keys())[0]
        if key.lower().startswith("team"):
            raw_data = raw_data[key]

    # Extract and normalize placement
    placement = normalize_placement(
        raw_data.get("Placement") or raw_data.get("placement") or placement_from_file
    )

    # Extract comments
    comments = normalize_comments(
        raw_data.get("Comments") or raw_data.get("comments")
    )

    # Extract and normalize questions
    questions = []
    for q_num in ["Q1", "Q2", "Q3"]:
        q_data = raw_data.get(q_num) or raw_data.get(q_num.lower())
        if q_data:
            questions.append(normalize_question(q_data, q_num))

    return {
        "year": year,
        "placement": placement,
        "source_file": filepath.name,
        "comments": comments,
        "questions": questions,
    }
