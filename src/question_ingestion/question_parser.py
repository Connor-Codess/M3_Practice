"""Question parser for M3 Challenge problem statements."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ParsedQuestion:
    """Parsed M3 Challenge problem."""

    raw_text: str
    title: Optional[str]
    background: str
    questions: list[str]
    data_references: list[str]
    constraints: list[str]


class QuestionParser:
    """
    Parses M3 Challenge problem statements from PDF or text files.

    Extracts:
    - Problem title and background
    - Individual questions (Q1, Q2, Q3)
    - Data file references
    - Constraints and requirements
    """

    # Patterns for question extraction
    QUESTION_PATTERNS = [
        r"(?:^|\n)Q(?:uestion)?\s*(\d+)[:\.\)]?\s*(.*?)(?=(?:\n\s*Q(?:uestion)?\s*\d|$))",
        r"(?:^|\n)(\d+)[:\.\)]\s*(.*?)(?=(?:\n\s*\d+[:\.\)]|$))",
        r"(?:^|\n)Part\s*([A-C1-3])[:\.\)]?\s*(.*?)(?=(?:\n\s*Part\s*[A-C1-3]|$))",
    ]

    # Patterns for data references
    DATA_PATTERNS = [
        r"(?:Table|Figure|Appendix|Data|Dataset)\s*\d+[:\.]?\s*([^\n]+)",
        r"(?:provided|attached|included)\s+(?:data|file|table)[s]?\s*[:.]?\s*([^\n]+)",
        r"([^\n]+\.(?:csv|xlsx|xls))",
    ]

    # Patterns for constraints
    CONSTRAINT_PATTERNS = [
        r"(\d+)\s*pages?\s+(?:maximum|limit)",
        r"(?:maximum|limit)\s+(?:of\s+)?(\d+)\s*pages?",
        r"(?:must|should)\s+(?:include|contain)\s+([^\n\.]+)",
        r"(?:no\s+more\s+than|at\s+most)\s+([^\n\.]+)",
    ]

    def __init__(self):
        """Initialize the parser."""
        pass

    def parse_pdf(self, pdf_path: Path) -> ParsedQuestion:
        """
        Parse a PDF file containing an M3 problem.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ParsedQuestion with extracted components.
        """
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF parsing. Install with: pip install pypdf"
            )

        reader = pypdf.PdfReader(pdf_path)
        text_parts = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        raw_text = "\n\n".join(text_parts)
        return self.parse_text(raw_text)

    def parse_text(self, text: str) -> ParsedQuestion:
        """
        Parse raw text containing an M3 problem.

        Args:
            text: Raw problem text.

        Returns:
            ParsedQuestion with extracted components.
        """
        # Clean text
        text = self._clean_text(text)

        # Extract components
        title = self._extract_title(text)
        background = self._extract_background(text)
        questions = self._extract_questions(text)
        data_refs = self._extract_data_references(text)
        constraints = self._extract_constraints(text)

        return ParsedQuestion(
            raw_text=text,
            title=title,
            background=background,
            questions=questions,
            data_references=data_refs,
            constraints=constraints,
        )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Normalize whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove page numbers and headers
        text = re.sub(r"\n\s*Page\s+\d+\s*\n", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\n\s*\d+\s*of\s*\d+\s*\n", "\n", text)

        return text.strip()

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract problem title."""
        # Look for title patterns
        patterns = [
            r"^M3\s+Challenge.*?[-:]\s*(.+?)(?:\n|$)",
            r"^(?:20\d{2}\s+)?M3\s+(?:Challenge\s+)?Problem[:\s]+(.+?)(?:\n|$)",
            r"^Problem\s*(?:\d+)?[:\s]+(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        # Fall back to first non-empty line
        lines = text.split("\n")
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) > 10:
                return line

        return None

    def _extract_background(self, text: str) -> str:
        """Extract background/context section."""
        # Look for explicit background section
        patterns = [
            r"(?:^|\n)(?:Background|Introduction|Context)[:\s]*\n(.+?)(?=\n\s*Q(?:uestion)?\s*\d|\n\s*\d+[:\.\)])",
            r"(?:^|\n)(.+?)(?=\n\s*Q(?:uestion)?\s*\d|\n\s*\d+[:\.\)])",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                background = match.group(1).strip()
                if len(background) > 100:
                    return background

        # Return first substantial paragraph
        paragraphs = text.split("\n\n")
        for para in paragraphs[:3]:
            para = para.strip()
            if len(para) > 100:
                return para

        return text[:1000] if text else ""

    def _extract_questions(self, text: str) -> list[str]:
        """Extract individual questions."""
        questions = []

        for pattern in self.QUESTION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        # Pattern with groups
                        q_num, q_text = match
                        q_text = q_text.strip()
                    else:
                        q_text = match.strip()

                    if q_text and len(q_text) > 20:
                        # Clean up the question text
                        q_text = re.sub(r"\s+", " ", q_text)
                        questions.append(q_text)

                if questions:
                    break

        # If no structured questions found, try to split by numbered items
        if not questions:
            numbered = re.findall(r"(?:^|\n)\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|$)", text, re.DOTALL)
            for num, q_text in numbered:
                q_text = q_text.strip()
                if len(q_text) > 20:
                    questions.append(q_text)

        return questions[:5]  # Limit to 5 questions

    def _extract_data_references(self, text: str) -> list[str]:
        """Extract references to data files and tables."""
        references = []

        for pattern in self.DATA_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match = match.strip()
                if match and match not in references:
                    references.append(match)

        return references

    def _extract_constraints(self, text: str) -> list[str]:
        """Extract constraints and requirements."""
        constraints = []

        for pattern in self.CONSTRAINT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                constraint = match.strip()
                if constraint:
                    constraints.append(constraint)

        # Also look for bullet-pointed requirements
        req_section = re.search(
            r"(?:Constraints|Requirements|Guidelines)[:\s]*\n((?:[*\-\u2022]\s*.+\n?)+)",
            text,
            re.IGNORECASE,
        )
        if req_section:
            bullets = re.findall(r"[*\-\u2022]\s*(.+)", req_section.group(1))
            constraints.extend(b.strip() for b in bullets if b.strip())

        return constraints

    def parse_file(self, file_path: Path) -> ParsedQuestion:
        """
        Parse a problem file (PDF or text).

        Args:
            file_path: Path to the problem file.

        Returns:
            ParsedQuestion with extracted components.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self.parse_pdf(path)
        elif suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
            return self.parse_text(text)
        else:
            # Try to read as text
            try:
                text = path.read_text(encoding="utf-8")
                return self.parse_text(text)
            except UnicodeDecodeError:
                raise ValueError(f"Unsupported file format: {suffix}")
