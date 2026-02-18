"""Question store for maintaining current question context across agents."""

from pathlib import Path
from typing import Optional

from ..agents.models import ProblemContext, DataSummary


class QuestionStore:
    """
    Simple storage for the current question context.

    Agents can query this store to get information about the current
    problem being solved, including data summaries.

    This is a singleton-like class that maintains state during a pipeline run.
    """

    _instance = None
    _context: Optional[ProblemContext] = None
    _data_files: list[Path] = []

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def set_context(cls, context: ProblemContext) -> None:
        """
        Set the current problem context.

        Args:
            context: The problem context to store.
        """
        cls._context = context

    @classmethod
    def get_context(cls) -> Optional[ProblemContext]:
        """
        Get the current problem context.

        Returns:
            The stored ProblemContext or None if not set.
        """
        return cls._context

    @classmethod
    def set_data_files(cls, paths: list[Path]) -> None:
        """
        Set paths to the current data files.

        Args:
            paths: List of paths to data files.
        """
        cls._data_files = [Path(p) for p in paths]

    @classmethod
    def get_data_files(cls) -> list[Path]:
        """
        Get paths to current data files.

        Returns:
            List of data file paths.
        """
        return cls._data_files

    @classmethod
    def get_data_summary(cls) -> Optional[DataSummary]:
        """
        Get the data summary from the current context.

        Returns:
            DataSummary or None if not available.
        """
        if cls._context:
            return cls._context.data_summary
        return None

    @classmethod
    def get_data_context_string(cls) -> str:
        """
        Get the formatted data context string for prompts.

        Returns:
            Formatted data context or empty string.
        """
        if cls._context:
            return cls._context.data_context_string
        return ""

    @classmethod
    def get_questions(cls) -> list[str]:
        """
        Get the list of questions from the current context.

        Returns:
            List of question strings.
        """
        if cls._context:
            return cls._context.questions
        return []

    @classmethod
    def get_scope(cls) -> str:
        """
        Get the problem scope from the current context.

        Returns:
            Scope string or empty string.
        """
        if cls._context:
            return cls._context.scope
        return ""

    @classmethod
    def get_key_variables(cls) -> list[str]:
        """
        Get key variables from the current context.

        Returns:
            List of key variable names.
        """
        if cls._context:
            return cls._context.key_variables
        return []

    @classmethod
    def clear(cls) -> None:
        """Clear the stored context and data files."""
        cls._context = None
        cls._data_files = []

    @classmethod
    def has_context(cls) -> bool:
        """Check if a context is currently stored."""
        return cls._context is not None

    @classmethod
    def has_data(cls) -> bool:
        """Check if data is available in the context."""
        return (
            cls._context is not None
            and cls._context.data_summary is not None
        )

    def __repr__(self) -> str:
        if self._context:
            return f"QuestionStore(scope='{self._context.scope[:50]}...', has_data={self.has_data()})"
        return "QuestionStore(empty)"


# Convenience functions for direct access
def get_current_question() -> Optional[ProblemContext]:
    """Get the current question context."""
    return QuestionStore.get_context()


def get_data_summary() -> Optional[DataSummary]:
    """Get the current data summary."""
    return QuestionStore.get_data_summary()


def get_data_context() -> str:
    """Get the formatted data context string."""
    return QuestionStore.get_data_context_string()
