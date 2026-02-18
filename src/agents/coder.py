"""Coder agent - implements mathematical models as Jupyter notebooks."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .base import BaseAgent
from .models import MathModel, Implementation, ProblemContext, NotebookCell


class Coder(BaseAgent):
    """
    Implements mathematical models as executable Jupyter notebooks.

    Generates well-organized notebooks with:
    - Markdown cells explaining each section
    - Separate code cells for imports, parameters, model, visualizations
    - Clear documentation for interpretation
    """

    agent_name = "coder"
    default_provider = "openai"
    default_model = "gpt-4o"

    SYSTEM_PROMPT = """You are an expert Python programmer creating Jupyter notebooks for M3 Math Modeling solutions.

Your output must be a well-structured notebook with SEPARATE CELLS for each component.

OUTPUT FORMAT - Return a JSON array of cells, each with:
- "cell_type": either "markdown" or "code"
- "source": the cell content
- "section": a label like "imports", "parameters", "model", "visualization", "answer"

REQUIRED SECTIONS (in order):
1. Title & Overview (markdown) - Problem summary and approach
2. Imports (code) - All library imports
3. Parameters (code) - Model parameters with detailed comments
4. Data Loading (code, if applicable) - Load any provided data
5. Model Definition (code) - Core mathematical model functions
6. Model Execution (code) - Run the simulation/calculations
7. Visualization 1 (markdown + code) - First plot with explanation
8. Visualization 2+ (markdown + code) - Additional plots as needed
9. Answer to Q1 (markdown + code) - Explicit answer with supporting output
10. Answer to Q2 (markdown + code) - Explicit answer with supporting output
11. Answer to Q3+ (markdown + code) - Continue for all questions
12. Summary (markdown) - Key findings and conclusions

MARKDOWN CELL GUIDELINES:
- Use headers (##, ###) to organize content
- Explain the PURPOSE of the next code cell
- Describe how to INTERPRET the outputs/plots
- Include the mathematical reasoning behind calculations
- Write as if explaining to someone reading the notebook

CODE CELL GUIDELINES:
- Each code cell should do ONE thing well
- Add inline comments explaining the math (not the code)
- Use descriptive variable names
- Print intermediate results with clear labels
- For plots: include title, axis labels, legend

Example cell format:
[
  {"cell_type": "markdown", "source": "# M3 Solution: EV Adoption Model\\n\\nThis notebook implements...", "section": "title"},
  {"cell_type": "code", "source": "import numpy as np\\nimport matplotlib.pyplot as plt", "section": "imports"},
  ...
]

Return ONLY the JSON array, no other text."""

    def run(
        self,
        math_model: MathModel,
        problem_context: ProblemContext,
        data_path: Optional[Path] = None,
        execute: bool = True,
    ) -> Implementation:
        """
        Generate Jupyter notebook implementation.

        Args:
            math_model: Model specification from Mathematician.
            problem_context: Problem context for reference.
            data_path: Optional path to data files.
            execute: Whether to execute the generated code.

        Returns:
            Implementation with notebook cells and flat code.
        """
        equations_text = "\n".join(f"  {eq}" for eq in math_model.equations)
        variables_text = "\n".join(f"  {k}: {v}" for k, v in math_model.variables.items())
        parameters_text = "\n".join(f"  {k}: {v}" for k, v in math_model.parameters.items())

        questions_text = "\n".join(
            f"Q{i+1}: {q}" for i, q in enumerate(problem_context.questions)
        )

        prompt = f"""Create a Jupyter notebook to implement this mathematical model.

MODEL FRAMEWORK: {math_model.framework}

EQUATIONS:
{equations_text}

VARIABLES:
{variables_text}

PARAMETERS:
{parameters_text}

BOUNDARY CONDITIONS:
{chr(10).join(f'  - {bc}' for bc in math_model.boundary_conditions)}

MODEL RATIONALE:
{math_model.rationale}

PROBLEM SCOPE:
{problem_context.scope}

QUESTIONS TO ANSWER (create a dedicated section for EACH):
{questions_text}
"""

        if data_path:
            prompt += f"""
DATA LOCATION: {data_path}
Include a data loading cell that reads and previews the data.
"""

        prompt += """
CRITICAL REQUIREMENTS:
1. Create a separate markdown + code cell pair for EACH question answer
2. Each answer cell must print the complete answer with specific numbers
3. Visualizations should be in their own cells with preceding markdown explanation
4. Every code cell should have a markdown cell before it explaining what it does
5. Use plt.show() for inline display (not savefig)
6. Make the notebook self-contained and educational

Return the JSON array of notebook cells."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self._call_llm(messages=messages, max_tokens=12000)

        # Parse the notebook cells
        cells = self._parse_notebook_cells(response)

        # Convert to flat code for execution
        flat_code = self._cells_to_flat_code(cells)

        implementation = Implementation(
            code=flat_code,
            language="python",
            dependencies=self._extract_dependencies(flat_code),
            notebook_cells=cells,
        )

        if execute:
            implementation = self._execute_code(implementation)

        return implementation

    def _parse_notebook_cells(self, response: str) -> list[NotebookCell]:
        """Parse LLM response into notebook cells."""
        import re

        text = response.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        try:
            cells_data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON array from response
            match = re.search(r"\[[\s\S]*\]", text)
            if match:
                try:
                    cells_data = json.loads(match.group())
                except json.JSONDecodeError:
                    # Fallback: create a single code cell
                    return [NotebookCell(
                        cell_type="code",
                        source=text,
                        section="fallback"
                    )]
            else:
                return [NotebookCell(
                    cell_type="code",
                    source=text,
                    section="fallback"
                )]

        cells = []
        for cell_data in cells_data:
            cells.append(NotebookCell(
                cell_type=cell_data.get("cell_type", "code"),
                source=cell_data.get("source", ""),
                section=cell_data.get("section", "")
            ))

        return cells

    def _cells_to_flat_code(self, cells: list[NotebookCell]) -> str:
        """Convert notebook cells to flat Python script."""
        code_parts = []

        for cell in cells:
            if cell.cell_type == "code":
                code_parts.append(f"# === {cell.section.upper()} ===")
                code_parts.append(cell.source)
                code_parts.append("")

        return "\n".join(code_parts)

    def create_notebook_json(self, cells: list[NotebookCell]) -> dict:
        """
        Create a valid .ipynb JSON structure from cells.

        Args:
            cells: List of NotebookCell objects.

        Returns:
            Dictionary representing a valid Jupyter notebook.
        """
        notebook = {
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.9.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": []
        }

        for i, cell in enumerate(cells):
            # Split source into lines for proper notebook format
            source_lines = cell.source.split("\n")
            # Add newlines to all but the last line
            source_with_newlines = [
                line + "\n" if j < len(source_lines) - 1 else line
                for j, line in enumerate(source_lines)
            ]

            if cell.cell_type == "markdown":
                notebook["cells"].append({
                    "cell_type": "markdown",
                    "id": f"cell-{i}",
                    "metadata": {},
                    "source": source_with_newlines
                })
            else:
                notebook["cells"].append({
                    "cell_type": "code",
                    "id": f"cell-{i}",
                    "metadata": {},
                    "source": source_with_newlines,
                    "outputs": [],
                    "execution_count": None
                })

        return notebook

    def save_notebook(self, implementation: Implementation, path: Path) -> Path:
        """
        Save implementation as a Jupyter notebook.

        Args:
            implementation: Implementation with notebook cells.
            path: Output path (should end in .ipynb).

        Returns:
            Path to saved notebook.
        """
        notebook = self.create_notebook_json(implementation.notebook_cells)

        path = Path(path)
        if not path.suffix == ".ipynb":
            path = path.with_suffix(".ipynb")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)

        return path

    def _clean_code(self, code: str) -> str:
        """Remove markdown formatting from code."""
        import re

        code = code.strip()

        if code.startswith("```"):
            code = re.sub(r"^```\w*\n?", "", code)
            code = re.sub(r"\n?```$", "", code)

        return code.strip()

    def _extract_dependencies(self, code: str) -> list[str]:
        """Extract import statements to determine dependencies."""
        import re

        deps = set()

        import_pattern = r"^(?:from\s+(\w+)|import\s+(\w+))"
        for line in code.split("\n"):
            match = re.match(import_pattern, line.strip())
            if match:
                module = match.group(1) or match.group(2)
                package_map = {
                    "numpy": "numpy",
                    "np": "numpy",
                    "pandas": "pandas",
                    "pd": "pandas",
                    "matplotlib": "matplotlib",
                    "plt": "matplotlib",
                    "scipy": "scipy",
                    "sklearn": "scikit-learn",
                }
                if module in package_map:
                    deps.add(package_map[module])

        return list(deps)

    def _execute_code(self, implementation: Implementation) -> Implementation:
        """Execute Python code and capture output."""
        # Modify code for file execution (replace plt.show() with savefig)
        exec_code = implementation.code.replace(
            "plt.show()",
            "plt.savefig(f'plot_{plt.gcf().number}.png', dpi=150, bbox_inches='tight'); plt.close()"
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(exec_code)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=tempfile.gettempdir(),
            )

            implementation.execution_log = result.stdout
            if result.returncode != 0:
                implementation.success = False
                implementation.error_message = result.stderr

            import glob
            plot_files = glob.glob(f"{tempfile.gettempdir()}/*.png")
            implementation.visualizations = plot_files

            implementation.question_answers = self._extract_question_answers(
                implementation.execution_log
            )

        except subprocess.TimeoutExpired:
            implementation.success = False
            implementation.error_message = "Code execution timed out (120s limit)"

        except Exception as e:
            implementation.success = False
            implementation.error_message = str(e)

        finally:
            Path(script_path).unlink(missing_ok=True)

        return implementation

    def _extract_question_answers(self, execution_log: str) -> dict[str, str]:
        """Extract explicit question answers from execution output."""
        import re

        answers = {}

        pattern = r"ANSWER TO (Q\d+):\s*\n=+\s*\n(.*?)(?=\n=+\s*\nANSWER TO Q\d+:|$)"
        matches = re.findall(pattern, execution_log, re.DOTALL | re.IGNORECASE)

        for question_num, answer_text in matches:
            answer_text = answer_text.strip()
            answer_text = re.sub(r"\n=+\s*$", "", answer_text).strip()
            answers[question_num.upper()] = answer_text

        if not answers:
            simple_pattern = r"(?:Answer to |Response to )?(Q\d+)[:\s]+(.+?)(?=(?:Answer to |Response to )?Q\d+[:\s]|$)"
            matches = re.findall(simple_pattern, execution_log, re.DOTALL | re.IGNORECASE)
            for question_num, answer_text in matches:
                if len(answer_text.strip()) > 50:
                    answers[question_num.upper()] = answer_text.strip()

        return answers

    def debug_code(
        self,
        implementation: Implementation,
        error_context: str,
    ) -> Implementation:
        """Debug and fix code based on error."""
        prompt = f"""Debug and fix this Python code.

ORIGINAL CODE:
```python
{implementation.code}
```

ERROR:
{error_context}

EXECUTION LOG:
{implementation.execution_log}

Fix the code to resolve the error. Return the fixed code as a JSON array of notebook cells."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self._call_llm(messages=messages, max_tokens=12000)
        cells = self._parse_notebook_cells(response)
        flat_code = self._cells_to_flat_code(cells)

        new_implementation = Implementation(
            code=flat_code,
            language="python",
            dependencies=self._extract_dependencies(flat_code),
            notebook_cells=cells,
        )

        return self._execute_code(new_implementation)

    def refine_code(
        self,
        implementation: Implementation,
        feedback: str,
    ) -> Implementation:
        """Refine code based on feedback."""
        prompt = f"""Improve this Jupyter notebook based on the feedback.

CURRENT CODE:
```python
{implementation.code}
```

FEEDBACK:
{feedback}

Improve the code to address the feedback. Return the improved code as a JSON array of notebook cells."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self._call_llm(messages=messages, max_tokens=12000)
        cells = self._parse_notebook_cells(response)
        flat_code = self._cells_to_flat_code(cells)

        new_implementation = Implementation(
            code=flat_code,
            language="python",
            dependencies=self._extract_dependencies(flat_code),
            notebook_cells=cells,
        )

        return self._execute_code(new_implementation)
