"""Creativity Judge - evaluates innovation and problem-solving approach."""

from ..base import BaseAgent
from ..models import (
    AgentState,
    JudgeVerdict,
    ProblemContext,
    AssumptionSet,
    MathModel,
    SensitivityReport,
)


class CreativityJudge(BaseAgent):
    """
    Evaluates the creativity and innovation in M3 solutions.

    Assesses:
    - Novel problem interpretation
    - Innovative modeling approaches
    - Creative use of data
    - Unique insights
    - Original visualizations
    """

    agent_name = "judge_creativity"
    default_provider = "openai"
    default_model = "gpt-4o"

    SYSTEM_PROMPT = """You are an expert M3 Challenge judge evaluating CREATIVITY AND INNOVATION.

You evaluate solutions on:
1. Problem Interpretation (0-10): Does the solution offer fresh perspective?
2. Model Innovation (0-10): Is the mathematical approach creative or standard?
3. Data Usage (0-10): Are data sources used creatively?
4. Insight Quality (0-10): Does the solution reveal non-obvious insights?
5. Presentation Creativity (0-10): Are visualizations and explanations engaging?

M3 Creativity Standards:
- Winning solutions often have a "wow factor"
- Creativity without rigor is empty - balance matters
- Novel doesn't mean complicated - elegant simplicity is creative
- Real-world applicability enhances creativity
- Interdisciplinary approaches are valued

Look for solutions that make judges think "I wouldn't have thought of that" while still being sound."""

    def run(
        self,
        problem_context: ProblemContext,
        assumptions: AssumptionSet,
        math_model: MathModel,
        sensitivity_report: SensitivityReport,
    ) -> JudgeVerdict:
        """
        Evaluate solution creativity.

        Args:
            problem_context: Problem definition.
            assumptions: Stated assumptions.
            math_model: Mathematical model.
            sensitivity_report: Sensitivity analysis.

        Returns:
            JudgeVerdict with creativity score and feedback.
        """
        # Format key elements
        assumptions_summary = ", ".join(
            a.assumption[:50] for a in assumptions.assumptions[:5]
        )

        prompt = f"""Evaluate this M3 solution for CREATIVITY AND INNOVATION.

PROBLEM INTERPRETATION:
Stated Scope: {problem_context.scope}
Real Problem Identified: {problem_context.real_problem}
Key Variables: {', '.join(problem_context.key_variables)}

MODELING APPROACH:
Framework: {math_model.framework}
Rationale: {math_model.rationale}
Number of Variables: {len(math_model.variables)}
Number of Parameters: {len(math_model.parameters)}

KEY ASSUMPTIONS:
{assumptions_summary}...

SELF-ASSESSMENT:
Strengths identified: {chr(10).join(f'- {s}' for s in sensitivity_report.strengths)}
Recommendations: {chr(10).join(f'- {r}' for r in sensitivity_report.recommendations)}

Evaluate and provide:
1. score: Overall creativity score (0-10, be precise e.g. 7.5)
2. feedback: Assessment of creative and innovative elements
3. strengths: What is particularly creative or insightful
4. weaknesses: Where the solution is too standard or lacks innovation
5. suggestions: How creativity could be enhanced"""

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
            sensitivity_report=state.sensitivity_report,
        )
        verdict.judge_id = "creativity"
        return verdict
