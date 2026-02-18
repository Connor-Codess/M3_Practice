"""Context builder that combines question and data into enriched problem context."""

from pathlib import Path
from typing import Optional

from ..agents.models import ProblemContext, DataSummary
from .data_loader import DataLoader
from .question_parser import QuestionParser, ParsedQuestion


class ContextBuilder:
    """
    Builds enriched problem context by combining parsed questions with data summaries.

    Creates ProblemContext objects that include both the problem statement
    and actual data statistics for use by agents.
    """

    def __init__(self):
        """Initialize the context builder with loader and parser."""
        self.data_loader = DataLoader()
        self.question_parser = QuestionParser()

    def build_from_files(
        self,
        problem_path: Path,
        data_paths: Optional[list[Path]] = None,
    ) -> ProblemContext:
        """
        Build enriched context from problem and data files.

        Args:
            problem_path: Path to problem file (PDF or text).
            data_paths: Optional list of paths to data files (CSV/Excel).

        Returns:
            Enriched ProblemContext with data summary.
        """
        # Parse the problem
        parsed = self.question_parser.parse_file(problem_path)

        # Load and summarize data if provided
        data_summary = None
        data_context_string = ""

        if data_paths:
            data_summary = self.data_loader.generate_summary(data_paths)
            data_context_string = data_summary.to_context_string()

        return self._build_context(
            parsed=parsed,
            data_summary=data_summary,
            data_context_string=data_context_string,
        )

    def build_from_text(
        self,
        problem_text: str,
        data_paths: Optional[list[Path]] = None,
    ) -> ProblemContext:
        """
        Build enriched context from problem text and data files.

        Args:
            problem_text: Raw problem text.
            data_paths: Optional list of paths to data files.

        Returns:
            Enriched ProblemContext with data summary.
        """
        # Parse the problem text
        parsed = self.question_parser.parse_text(problem_text)

        # Load and summarize data if provided
        data_summary = None
        data_context_string = ""

        if data_paths:
            data_summary = self.data_loader.generate_summary(data_paths)
            data_context_string = data_summary.to_context_string()

        return self._build_context(
            parsed=parsed,
            data_summary=data_summary,
            data_context_string=data_context_string,
        )

    def _build_context(
        self,
        parsed: ParsedQuestion,
        data_summary: Optional[DataSummary],
        data_context_string: str,
    ) -> ProblemContext:
        """
        Build ProblemContext from parsed question and data.

        Args:
            parsed: Parsed question object.
            data_summary: Optional data summary.
            data_context_string: Formatted data context for prompts.

        Returns:
            ProblemContext ready for agent use.
        """
        # Build initial context
        # Scope and real_problem will be filled in by Scout agent
        context = ProblemContext(
            raw_text=parsed.raw_text,
            questions=parsed.questions if parsed.questions else [],
            constraints=parsed.constraints,
            data_provided=parsed.data_references,
            scope="",  # To be filled by Scout
            real_problem="",  # To be filled by Scout
            key_variables=[],  # To be filled by Scout
            success_criteria=[],  # To be filled by Scout
            data_summary=data_summary,
            data_context_string=data_context_string,
        )

        # If we have data, extract key variables from column names
        if data_summary:
            context.key_variables = self._extract_key_variables(data_summary)

        return context

    def _extract_key_variables(self, data_summary: DataSummary) -> list[str]:
        """Extract potential key variables from column names."""
        variables = []

        for col_name, col_info in data_summary.columns.items():
            # Clean column name
            clean_name = col_name.split(":")[-1]  # Remove file prefix if present
            clean_name = clean_name.replace("_", " ").replace("-", " ").strip()

            # Skip generic columns
            skip_patterns = ["id", "index", "unnamed", "row"]
            if any(pattern in clean_name.lower() for pattern in skip_patterns):
                continue

            # Add if it seems like a meaningful variable
            if len(clean_name) > 2:
                variables.append(clean_name)

        return variables[:10]  # Limit to top 10

    def enrich_context(
        self,
        context: ProblemContext,
        data_paths: list[Path],
    ) -> ProblemContext:
        """
        Enrich an existing context with data summaries.

        Args:
            context: Existing ProblemContext.
            data_paths: Paths to data files.

        Returns:
            Enriched ProblemContext.
        """
        data_summary = self.data_loader.generate_summary(data_paths)
        data_context_string = data_summary.to_context_string()

        # Update context
        context.data_summary = data_summary
        context.data_context_string = data_context_string

        # Add any new key variables
        new_vars = self._extract_key_variables(data_summary)
        existing = set(context.key_variables)
        context.key_variables.extend(v for v in new_vars if v not in existing)

        return context


def build_context(
    problem_path: Optional[Path] = None,
    problem_text: Optional[str] = None,
    data_paths: Optional[list[Path]] = None,
) -> ProblemContext:
    """
    Convenience function to build enriched context.

    Args:
        problem_path: Path to problem file.
        problem_text: Raw problem text (if no file).
        data_paths: Optional paths to data files.

    Returns:
        Enriched ProblemContext.
    """
    builder = ContextBuilder()

    if problem_path:
        return builder.build_from_files(problem_path, data_paths)
    elif problem_text:
        return builder.build_from_text(problem_text, data_paths)
    else:
        raise ValueError("Must provide either problem_path or problem_text")
