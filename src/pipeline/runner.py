"""Runner script for executing M3 Challenge pipeline."""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..agents.models import AgentState
from .workflow import M3Workflow, create_workflow


class M3Runner:
    """
    High-level runner for M3 Challenge solution generation.

    Provides:
    - Easy initialization with sensible defaults
    - Progress tracking and logging
    - Result formatting and export
    """

    def __init__(
        self,
        env_path: Optional[Path] = None,
        chroma_path: Optional[Path] = None,
        verbose: bool = True,
    ):
        """
        Initialize the runner.

        Args:
            env_path: Path to environment file with API keys.
            chroma_path: Path to ChromaDB for RAG.
            verbose: Whether to print progress updates.
        """
        self.verbose = verbose
        self.workflow = create_workflow(
            env_path=env_path,
            chroma_path=chroma_path,
        )

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def run(
        self,
        problem_text: str,
        output_dir: Optional[Path] = None,
    ) -> AgentState:
        """
        Run the full M3 solution pipeline.

        Args:
            problem_text: Raw problem text or path to problem file.
            output_dir: Optional directory to save outputs.

        Returns:
            Final AgentState with all results.
        """
        # Check if problem_text is a file path (only if it's short enough to be a path)
        if len(problem_text) < 500 and "\n" not in problem_text:
            problem_path = Path(problem_text)
            if problem_path.exists():
                self._log(f"Loading problem from: {problem_path}")
                problem_text = problem_path.read_text()

        self._log("Starting M3 Challenge solution pipeline")
        self._log(f"Problem length: {len(problem_text)} characters")

        # Run workflow
        self._log("Phase 1: Input Handling...")
        self._log("Phase 2: Problem Analysis (Scout)...")
        self._log("Phase 3: Assumption Generation (Historian + RAG)...")
        self._log("Phase 4: Model Development (Mathematician)...")
        self._log("Phase 5: Implementation (Coder)...")
        self._log("Phase 6: Sensitivity Analysis (Stress-Tester)...")
        self._log("Phase 7: Evaluation (Judges)...")
        self._log("Phase 8: Orchestration...")

        try:
            final_state = self.workflow.run(problem_text)
        except Exception as e:
            self._log(f"Pipeline error: {e}")
            raise

        # Handle LangGraph returning a dict instead of AgentState
        if isinstance(final_state, dict):
            final_state = AgentState.model_validate(final_state)

        # Log results
        decision = final_state.orchestrator_decision
        self._log(f"Pipeline complete!")
        self._log(f"Final score: {decision.consensus_score:.1f}/10")
        self._log(f"Iterations: {decision.iteration_count}")

        # Save outputs if directory provided
        if output_dir:
            self._save_outputs(final_state, output_dir)

        return final_state

    def run_from_pdf(
        self,
        pdf_path: Path,
        output_dir: Optional[Path] = None,
    ) -> AgentState:
        """
        Run pipeline from a PDF problem file.

        Args:
            pdf_path: Path to problem PDF.
            output_dir: Optional directory to save outputs.

        Returns:
            Final AgentState with all results.
        """
        self._log(f"Extracting text from PDF: {pdf_path}")

        # Use input handler to parse PDF
        problem_text = self.workflow.input_handler.parse_pdf(pdf_path)

        return self.run(problem_text, output_dir)

    def run_with_data(
        self,
        problem_path: Path,
        data_files: list[Path],
        output_dir: Optional[Path] = None,
    ) -> AgentState:
        """
        Run pipeline with problem and associated data files.

        This method processes data files to extract statistics and metadata,
        then passes enriched context to all agents.

        Args:
            problem_path: Path to problem file (PDF or text).
            data_files: List of paths to data files (CSV, Excel).
            output_dir: Optional directory to save outputs.

        Returns:
            Final AgentState with all results.
        """
        from ..question_ingestion import ContextBuilder, QuestionStore

        self._log(f"Loading problem from: {problem_path}")
        self._log(f"Loading {len(data_files)} data file(s)...")

        # Build enriched context
        builder = ContextBuilder()
        context = builder.build_from_files(
            problem_path=problem_path,
            data_paths=data_files,
        )

        # Store in QuestionStore for agent access
        QuestionStore.set_context(context)
        QuestionStore.set_data_files(data_files)

        # Log data summary
        if context.data_summary:
            summary = context.data_summary
            self._log(f"  Total rows: {summary.total_rows}")
            self._log(f"  Columns: {len(summary.columns)}")
            if summary.date_range:
                self._log(f"  Date range: {summary.date_range[0]} to {summary.date_range[1]}")
            if summary.geographic_scope:
                self._log(f"  Geographic scope: {summary.geographic_scope}")

        self._log("Starting M3 Challenge solution pipeline with data context")
        self._log(f"Problem length: {len(context.raw_text)} characters")

        # Run workflow with enriched context
        self._log("Phase 1: Input Handling...")
        self._log("Phase 2: Problem Analysis (Scout)...")
        self._log("Phase 3: Assumption Generation (Historian + RAG)...")
        self._log("Phase 4: Model Development (Mathematician)...")
        self._log("Phase 5: Implementation (Coder)...")
        self._log("Phase 6: Sensitivity Analysis (Stress-Tester)...")
        self._log("Phase 7: Evaluation (Judges)...")
        self._log("Phase 8: Orchestration...")

        try:
            final_state = self.workflow.run_with_context(context)
        except Exception as e:
            self._log(f"Pipeline error: {e}")
            QuestionStore.clear()
            raise

        # Handle LangGraph returning a dict instead of AgentState
        if isinstance(final_state, dict):
            final_state = AgentState.model_validate(final_state)

        # Log results
        decision = final_state.orchestrator_decision
        self._log(f"Pipeline complete!")
        self._log(f"Final score: {decision.consensus_score:.1f}/10")
        self._log(f"Iterations: {decision.iteration_count}")

        # Save outputs if directory provided
        if output_dir:
            self._save_outputs(final_state, output_dir)
            # Also save data summary
            if context.data_summary:
                self._save_data_summary(context.data_summary, output_dir)

        # Clear store after run
        QuestionStore.clear()

        return final_state

    def _save_data_summary(self, data_summary, output_dir: Path):
        """Save data summary to output directory."""
        summary_text = data_summary.to_context_string()
        summary_path = output_dir / "data_summary.txt"
        summary_path.write_text(summary_text)
        self._log(f"  Saved: {summary_path.name}")

    def _save_outputs(self, state: AgentState, output_dir: Path):
        """Save pipeline outputs to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Saving outputs to: {output_dir}")

        # Save Jupyter notebook (primary output)
        if state.implementation.notebook_cells:
            from ..agents.coder import Coder
            coder = Coder.__new__(Coder)  # Create instance without init for method access
            notebook_path = coder.save_notebook(
                state.implementation,
                output_dir / "solution.ipynb"
            )
            self._log(f"  Saved: {notebook_path.name}")

        # Save flat Python code (backup/reference)
        code_path = output_dir / "solution.py"
        code_path.write_text(state.implementation.code)
        self._log(f"  Saved: {code_path.name}")

        # Save summary report
        report = self.workflow.orchestrator.synthesize_final_report(
            state=state,
            decision=state.orchestrator_decision,
        )
        report_path = output_dir / "summary.txt"
        report_path.write_text(report)
        self._log(f"  Saved: {report_path.name}")

        # Save assumptions
        assumptions_text = "\n\n".join(
            f"[{a.category}] {a.assumption}\nJustification: {a.justification}"
            for a in state.assumptions.assumptions
        )
        assumptions_path = output_dir / "assumptions.txt"
        assumptions_path.write_text(assumptions_text)
        self._log(f"  Saved: {assumptions_path.name}")

        # Save model specification
        model_text = f"""Framework: {state.math_model.framework}

Equations:
{chr(10).join(state.math_model.equations)}

Variables:
{chr(10).join(f'{k}: {v}' for k, v in state.math_model.variables.items())}

Parameters:
{chr(10).join(f'{k}: {v}' for k, v in state.math_model.parameters.items())}

Rationale:
{state.math_model.rationale}
"""
        model_path = output_dir / "model.txt"
        model_path.write_text(model_text)
        self._log(f"  Saved: {model_path.name}")

        # Save sensitivity report
        sensitivity_text = f"""Stability: {state.sensitivity_report.stability_assessment}
Robustness Score: {state.sensitivity_report.robustness_score:.2f}

Critical Parameters:
{chr(10).join(f'- {p}' for p in state.sensitivity_report.critical_parameters)}

Strengths:
{chr(10).join(f'- {s}' for s in state.sensitivity_report.strengths)}

Weaknesses:
{chr(10).join(f'- {w}' for w in state.sensitivity_report.weaknesses)}

Recommendations:
{chr(10).join(f'- {r}' for r in state.sensitivity_report.recommendations)}
"""
        sensitivity_path = output_dir / "sensitivity.txt"
        sensitivity_path.write_text(sensitivity_text)
        self._log(f"  Saved: {sensitivity_path.name}")

        # Save explicit question answers
        answers_text = self._format_question_answers(state)
        answers_path = output_dir / "answers.txt"
        answers_path.write_text(answers_text)
        self._log(f"  Saved: {answers_path.name}")

    def _format_question_answers(self, state: AgentState) -> str:
        """Format question answers for output file."""
        lines = [
            "M3 SOLUTION - ANSWERS TO QUESTIONS",
            "=" * 50,
            "",
        ]

        questions = state.problem_context.questions
        answers = state.implementation.question_answers

        for i, question in enumerate(questions, 1):
            q_key = f"Q{i}"
            lines.append(f"QUESTION {i}:")
            lines.append("-" * 40)
            lines.append(question[:500] + "..." if len(question) > 500 else question)
            lines.append("")
            lines.append(f"ANSWER:")
            lines.append("-" * 40)

            if q_key in answers:
                lines.append(answers[q_key])
            else:
                # Fallback: note that answer wasn't explicitly captured
                lines.append("[Answer not explicitly captured from code output]")
                lines.append("")
                lines.append("Note: The solution code may address this question,")
                lines.append("but no explicit answer section was generated.")
                lines.append("Check solution.py and execution_log for details.")

            lines.append("")
            lines.append("=" * 50)
            lines.append("")

        # Summary of answer coverage
        answered = sum(1 for i in range(1, len(questions) + 1) if f"Q{i}" in answers)
        lines.append(f"ANSWER COVERAGE: {answered}/{len(questions)} questions explicitly answered")

        if answered < len(questions):
            missing = [f"Q{i}" for i in range(1, len(questions) + 1) if f"Q{i}" not in answers]
            lines.append(f"Missing explicit answers for: {', '.join(missing)}")

        return "\n".join(lines)

    def get_summary(self, state: AgentState) -> str:
        """Get formatted summary of pipeline results."""
        return self.workflow.orchestrator.synthesize_final_report(
            state=state,
            decision=state.orchestrator_decision,
        )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run M3 Challenge solution pipeline"
    )
    parser.add_argument(
        "problem",
        type=str,
        help="Problem text or path to problem file (txt or pdf)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory for results",
    )
    parser.add_argument(
        "-d", "--data",
        type=Path,
        nargs="+",
        help="Data files to include (CSV, Excel)",
    )
    parser.add_argument(
        "-e", "--env",
        type=Path,
        default=Path("keyholder.env"),
        help="Path to environment file",
    )
    parser.add_argument(
        "-c", "--chroma",
        type=Path,
        default=Path("chroma_db"),
        help="Path to ChromaDB",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Create runner
    runner = M3Runner(
        env_path=args.env,
        chroma_path=args.chroma,
        verbose=not args.quiet,
    )

    # Run pipeline
    problem_path = Path(args.problem)

    if args.data:
        # Run with data files
        state = runner.run_with_data(
            problem_path=problem_path,
            data_files=args.data,
            output_dir=args.output,
        )
    elif problem_path.suffix == ".pdf":
        state = runner.run_from_pdf(problem_path, args.output)
    else:
        state = runner.run(args.problem, args.output)

    # Print summary
    print("\n" + "=" * 50)
    print(runner.get_summary(state))


if __name__ == "__main__":
    main()
