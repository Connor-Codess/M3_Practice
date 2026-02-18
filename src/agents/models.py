"""Pydantic models for agent communication and data flow."""

from typing import Any, Optional
from pydantic import BaseModel, Field, AliasChoices


class ColumnInfo(BaseModel):
    """Statistics and metadata for a single data column."""

    name: str = Field(description="Column name from the data file")
    dtype: str = Field(description="Data type: numeric, categorical, datetime, or text")
    unique_count: int = Field(default=0, description="Number of unique values")
    missing_pct: float = Field(default=0.0, description="Percentage of missing values (0-100)")
    sample_values: list[Any] = Field(default_factory=list, description="First 5 non-null values")
    stats: Optional[dict] = Field(
        default=None,
        description="For numeric columns: mean, std, min, max, median, q25, q75"
    )


class DataSummary(BaseModel):
    """Summary statistics about loaded data files."""

    files: list[str] = Field(default_factory=list, description="Names of loaded data files")
    total_rows: int = Field(default=0, description="Total number of rows across all files")
    columns: dict[str, ColumnInfo] = Field(
        default_factory=dict,
        description="Column name to ColumnInfo mapping"
    )
    missing_data_pct: float = Field(default=0.0, description="Overall percentage of missing data")
    date_range: Optional[tuple[str, str]] = Field(
        default=None,
        description="For time series: (start_date, end_date) as ISO strings"
    )
    geographic_scope: Optional[str] = Field(
        default=None,
        description="Geographic coverage if location data detected"
    )
    data_quality_notes: list[str] = Field(
        default_factory=list,
        description="Warnings about data quality issues"
    )

    def to_context_string(self) -> str:
        """Generate a text summary suitable for LLM prompts."""
        parts = [f"DATA SUMMARY ({len(self.files)} files, {self.total_rows} rows total)"]

        if self.date_range:
            parts.append(f"Time Period: {self.date_range[0]} to {self.date_range[1]}")

        if self.geographic_scope:
            parts.append(f"Geographic Scope: {self.geographic_scope}")

        parts.append("\nColumns:")
        for col_name, info in self.columns.items():
            col_str = f"  - {col_name} ({info.dtype})"
            if info.stats:
                if "mean" in info.stats:
                    col_str += f": mean={info.stats['mean']:.2f}, std={info.stats['std']:.2f}"
            if info.missing_pct > 0:
                col_str += f" [{info.missing_pct:.1f}% missing]"
            parts.append(col_str)

        if self.data_quality_notes:
            parts.append("\nData Quality Notes:")
            for note in self.data_quality_notes:
                parts.append(f"  - {note}")

        return "\n".join(parts)


class ProblemContext(BaseModel):
    """Parsed problem from Input Handler and Scout agents."""

    raw_text: str = Field(description="Original problem text")
    questions: list[str] = Field(description="Q1, Q2, Q3 problem statements")
    constraints: list[str] = Field(default_factory=list, description="Time limits, units, format requirements")
    data_provided: list[str] = Field(default_factory=list, description="Referenced datasets and their descriptions")
    scope: str = Field(description="What we are solving for")
    real_problem: str = Field(description="Underlying issue vs stated problem")
    key_variables: list[str] = Field(default_factory=list, description="Important variables to model")
    success_criteria: list[str] = Field(default_factory=list, description="How solution quality will be judged")

    # Data context fields for enriched problem context
    data_summary: Optional[DataSummary] = Field(
        default=None,
        description="Statistics and metadata from loaded data files"
    )
    data_context_string: str = Field(
        default="",
        description="Formatted text summary of data for agent prompts"
    )


class JustifiedAssumption(BaseModel):
    """Single assumption with justification and historical backing."""

    assumption: str = Field(
        description="The assumption statement",
        validation_alias=AliasChoices("assumption", "statement"),
    )
    justification: str = Field(
        description="Why this assumption is reasonable",
        validation_alias=AliasChoices("justification", "reasoning", "rationale"),
    )
    historical_source: Optional[str] = Field(
        default=None,
        description="Source from RAG, e.g. '2024 Champion Q1'",
        validation_alias=AliasChoices("historical_source", "historical_reference", "source", "reference"),
    )
    relevance_score: Optional[float] = Field(default=None, description="RAG similarity score")
    category: str = Field(
        default="general",
        description="Category: data, model, scope, simplification",
        validation_alias=AliasChoices("category", "type"),
    )


class AssumptionSet(BaseModel):
    """Collection of assumptions for a solution."""

    assumptions: list[JustifiedAssumption] = Field(default_factory=list)
    problem_context_summary: str = Field(description="Brief summary of problem these assumptions address")
    total_assumptions: int = Field(default=0)

    def model_post_init(self, __context):
        self.total_assumptions = len(self.assumptions)


class MathModel(BaseModel):
    """Mathematical model specification from Mathematician agent."""

    framework: str = Field(description="Model type: Logistic Growth, Markov Chain, ODE System, etc.")
    equations: list[str] = Field(default_factory=list, description="LaTeX equations defining the model")
    variables: dict[str, str] = Field(default_factory=dict, description="Variable symbol to description mapping")
    parameters: dict[str, str] = Field(default_factory=dict, description="Parameter symbol to description mapping")
    rationale: str = Field(description="Why this model is appropriate for the problem")
    boundary_conditions: list[str] = Field(default_factory=list, description="Initial/boundary conditions")
    assumptions_used: list[str] = Field(default_factory=list, description="Which assumptions this model relies on")


class NotebookCell(BaseModel):
    """A single cell in a Jupyter notebook."""
    cell_type: str = Field(description="Either 'markdown' or 'code'")
    source: str = Field(description="Cell content")
    section: str = Field(default="", description="Section name for organization")


class Implementation(BaseModel):
    """Code implementation from Coder agent."""

    code: str = Field(description="Python code implementing the model (flat script)")
    language: str = Field(default="python", description="Programming language")
    dependencies: list[str] = Field(default_factory=list, description="Required packages")
    outputs: dict = Field(default_factory=dict, description="Computed results and values")
    visualizations: list[str] = Field(default_factory=list, description="Paths to generated plots or base64 images")
    execution_log: str = Field(default="", description="Stdout/stderr from code execution")
    success: bool = Field(default=True, description="Whether code executed successfully")
    error_message: Optional[str] = Field(default=None, description="Error message if execution failed")
    question_answers: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of question number (Q1, Q2, etc.) to explicit answer text"
    )
    notebook_cells: list[NotebookCell] = Field(
        default_factory=list,
        description="Structured notebook cells for Jupyter output"
    )


class SensitivityReport(BaseModel):
    """Sensitivity analysis from Stress-Tester agent."""

    perturbations: dict[str, dict] = Field(
        default_factory=dict,
        description="Parameter name to {change: %, impact: %} mapping"
    )
    stability_assessment: str = Field(description="Overall stability: Stable, Moderately Sensitive, Highly Sensitive")
    critical_parameters: list[str] = Field(default_factory=list, description="Parameters with highest impact")
    robustness_score: float = Field(default=0.0, description="0-1 score of model robustness")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list, description="Suggested improvements")


class JudgeVerdict(BaseModel):
    """Verdict from a single judge agent."""

    judge_id: str = Field(description="Judge identifier: accuracy, clarity, creativity")
    score: float = Field(ge=0, le=10, description="Score from 0-10")
    feedback: str = Field(description="Detailed feedback on the submission")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list, description="Specific improvement suggestions")


class OrchestratorDecision(BaseModel):
    """Decision from the Orchestrator on next steps."""

    consensus_score: float = Field(ge=0, le=10, description="Average of all judge scores")
    individual_scores: dict[str, float] = Field(default_factory=dict, description="Score by judge_id")
    should_iterate: bool = Field(description="Whether another iteration is needed")
    iteration_target: Optional[str] = Field(
        default=None,
        description="Agent to route back to: historian, mathematician, coder, stress_tester"
    )
    iteration_reason: str = Field(default="", description="Why iteration is needed")
    final_feedback: str = Field(description="Synthesized feedback from all judges")
    iteration_count: int = Field(default=0, description="Current iteration number")
    max_iterations: int = Field(default=3, description="Maximum allowed iterations")


class AgentState(BaseModel):
    """Full state passed through the agent pipeline."""

    problem_context: Optional[ProblemContext] = None
    assumptions: Optional[AssumptionSet] = None
    math_model: Optional[MathModel] = None
    implementation: Optional[Implementation] = None
    sensitivity_report: Optional[SensitivityReport] = None
    judge_verdicts: list[JudgeVerdict] = Field(default_factory=list)
    orchestrator_decision: Optional[OrchestratorDecision] = None
    iteration_history: list[str] = Field(default_factory=list, description="Log of iteration decisions")
    current_phase: str = Field(default="input", description="Current pipeline phase")
    error: Optional[str] = Field(default=None, description="Error message if pipeline failed")
