"""Clarity Judge - evaluates presentation and communication quality."""

from ..base import BaseAgent
from ..models import (
    AgentState,
    JudgeVerdict,
    ProblemContext,
    AssumptionSet,
    MathModel,
    Implementation,
    SensitivityReport,
)


class ClarityJudge(BaseAgent):
    """
    Evaluates the clarity and presentation of M3 solutions.

    Assesses:
    - Clear problem restatement
    - Well-organized structure
    - Readable code and documentation
    - Effective visualizations
    - Logical flow of arguments
    """

    agent_name = "judge_clarity"
    default_provider = "openai"
    default_model = "gpt-4o"

    SYSTEM_PROMPT = """You are an expert M3 Challenge judge evaluating CLARITY AND PRESENTATION.

You evaluate solutions on:
1. Problem Restatement (0-10): Does the solution clearly define what's being solved?
2. Assumption Clarity (0-10): Are assumptions clearly stated and justified?
3. Model Explanation (0-10): Is the mathematical approach well-explained?
4. Code Quality (0-10): Is code readable, documented, and well-structured?
5. Results Presentation (0-10): Are results clearly communicated with context?

M3 Clarity Standards:
- Judges have limited time - clarity is essential
- Each section should have clear purpose
- Visualizations should tell a story
- Technical jargon should be defined
- Transitions between sections matter

A judge should be able to understand the solution quickly. Confusion costs points."""

    def run(
        self,
        problem_context: ProblemContext,
        assumptions: AssumptionSet,
        math_model: MathModel,
        implementation: Implementation,
        sensitivity_report: SensitivityReport,
    ) -> JudgeVerdict:
        """
        Evaluate solution clarity.

        Args:
            problem_context: Problem definition.
            assumptions: Stated assumptions.
            math_model: Mathematical model.
            implementation: Code implementation.
            sensitivity_report: Sensitivity analysis.

        Returns:
            JudgeVerdict with clarity score and feedback.
        """
        # Format assumptions
        assumptions_text = "\n".join(
            f"- [{a.category}] {a.assumption}: {a.justification}"
            for a in assumptions.assumptions
        )

        prompt = f"""Evaluate this M3 solution for CLARITY AND PRESENTATION.

PROBLEM RESTATEMENT:
Scope: {problem_context.scope}
Real Problem: {problem_context.real_problem}

ASSUMPTIONS ({len(assumptions.assumptions)} total):
{assumptions_text[:2000]}

MATHEMATICAL MODEL:
Framework: {math_model.framework}
Rationale: {math_model.rationale}
Variables: {len(math_model.variables)} defined
Equations: {len(math_model.equations)} equations

CODE QUALITY:
Lines: ~{len(implementation.code.split(chr(10)))}
Dependencies: {', '.join(implementation.dependencies)}
Visualizations: {len(implementation.visualizations)} generated

SENSITIVITY ANALYSIS:
Strengths listed: {len(sensitivity_report.strengths)}
Weaknesses listed: {len(sensitivity_report.weaknesses)}
Recommendations: {len(sensitivity_report.recommendations)}

Evaluate and provide:
1. score: Overall clarity score (0-10, be precise e.g. 7.5)
2. feedback: Detailed assessment of presentation quality
3. strengths: What is communicated well
4. weaknesses: Clarity issues or confusing elements
5. suggestions: Specific improvements for clarity"""

        return self.generate_structured(
            prompt=prompt,
            output_type=JudgeVerdict,
            system_prompt=self.SYSTEM_PROMPT,
        )

    def evaluate_state(self, state: AgentState) -> JudgeVerdict:
        """Evaluate from full agent state."""
        verdict = self.run(
            problem_context=state.problem_context,
            assumptions=state.assumptions,
            math_model=state.math_model,
            implementation=state.implementation,
            sensitivity_report=state.sensitivity_report,
        )
        verdict.judge_id = "clarity"
        return verdict
