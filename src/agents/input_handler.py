"""Input Handler agent - parses raw problem text into structured format."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .base import BaseAgent


class ParsedInput(BaseModel):
    """Intermediate parsed structure from raw input."""

    questions: list[str] = Field(description="Individual question texts (Q1, Q2, Q3)")
    data_references: list[str] = Field(default_factory=list, description="Referenced data files or tables")
    constraints: list[str] = Field(default_factory=list, description="Explicit constraints mentioned")
    key_terms: list[str] = Field(default_factory=list, description="Important domain-specific terms")


class InputHandler(BaseAgent):
    """
    Parses raw M3 problem input into structured components.

    Handles:
    - PDF text extraction (if needed)
    - Question identification (Q1, Q2, Q3)
    - Data reference extraction
    - Constraint identification
    """

    agent_name = "input_handler"
    default_provider = "openai"
    default_model = "gpt-4o-mini"

    SYSTEM_PROMPT = """You are an expert at parsing M3 Math Modeling Challenge problems.

Your task is to carefully read the problem statement and extract:
1. The individual questions (Q1, Q2, Q3) - each should be a complete statement of what is being asked
2. Any referenced data files, tables, or datasets mentioned
3. Explicit constraints (time limits, format requirements, units, etc.)
4. Key domain-specific terms that will be important for modeling

Be thorough and precise. Each question should capture the full scope of what is being asked."""

    def run(
        self,
        problem_text: str,
        data_description: Optional[str] = None,
    ) -> ParsedInput:
        """
        Parse raw problem text into structured components.

        Args:
            problem_text: Raw text of the M3 problem.
            data_description: Optional description of accompanying data.

        Returns:
            ParsedInput with extracted components.
        """
        prompt = f"""Parse the following M3 Challenge problem and extract the structured components.

PROBLEM TEXT:
{problem_text}
"""

        if data_description:
            prompt += f"""
ACCOMPANYING DATA:
{data_description}
"""

        prompt += """
Extract:
1. questions: List each question (Q1, Q2, Q3) as a complete statement
2. data_references: List any data files, tables, or datasets mentioned
3. constraints: List any explicit constraints (time, format, units, page limits)
4. key_terms: List important domain-specific terms for modeling"""

        return self.generate_structured(
            prompt=prompt,
            output_type=ParsedInput,
            system_prompt=self.SYSTEM_PROMPT,
        )

    def parse_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Extracted text content.
        """
        try:
            import pypdf

            reader = pypdf.PdfReader(pdf_path)
            text_parts = []

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            return "\n\n".join(text_parts)

        except ImportError:
            raise ImportError("pypdf is required for PDF parsing. Install with: pip install pypdf")
