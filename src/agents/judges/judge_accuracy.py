"""Accuracy Judge - evaluates mathematical correctness and rigor."""

from ..base import BaseAgent
from ..models import (
    AgentState,
    JudgeVerdict,
    ProblemContext,
    MathModel,
    Implementation,
    SensitivityReport,
)


class AccuracyJudge(BaseAgent):
    """
    Evaluates the mathematical accuracy and correctness of M3 solutions.

    Assesses:
    - Mathematical rigor and correctness
    - Appropriate model selection
    - Correct implementation of equations
    - Valid numerical results
    - Proper use of assumptions
    """

    agent_name = "judge_accuracy"
    default_provider = "openai"
    default_model = "gpt-4o"

    SYSTEM_PROMPT = """You are an expert M3 Challenge judge evaluating MATHEMATICAL ACCURACY.

You evaluate solutions on:
1. Mathematical Rigor (0-10): Are equations correctly derived and justified?
2. Model Appropriateness (0-10): Is the chosen framework suitable for the problem?
3. Implementation Correctness (0-10): Does the code correctly implement the math?
4. Numerical Validity (0-10): Are results reasonable and error-free?
5. Assumption Usage (0-10): Are assumptions properly applied in the model?

M3 Accuracy Standards:
- Equations must be dimensionally consistent
- Numerical methods must be appropriate (stability, convergence)
- Results should be validated against intuition or known cases
- Errors in core calculations are heavily penalized
- Minor computational errors are forgivable if approach is sound

Be rigorous but fair. A perfect 10 is rare but achievable."""

    def run(
        self,
        problem_context: ProblemContext,
        math_model: MathModel,
        implementation: Implementation,
        sensitivity_report: SensitivityReport,
    ) -> JudgeVerdict:
        """
        Evaluate solution accuracy.

        Args:
            problem_context: Problem definition.
            math_model: Mathematical model.
            implementation: Code implementation.
            sensitivity_report: Sensitivity analysis.

        Returns:
            JudgeVerdict with accuracy score and feedback.
        """
        # Check question answer coverage
        questions = problem_context.questions
        answers = implementation.question_answers
        answered_count = sum(1 for i in range(1, len(questions) + 1) if f"Q{i}" in answers)

        answers_section = []
        for i, q in enumerate(questions, 1):
            q_key = f"Q{i}"
            if q_key in answers:
                answer_preview = answers[q_key][:300] + "..." if len(answers[q_key]) > 300 else answers[q_key]
                answers_section.append(f"Q{i}: ANSWERED\n{answer_preview}")
            else:
                answers_section.append(f"Q{i}: NOT EXPLICITLY ANSWERED")

        prompt = f"""Evaluate this M3 solution for MATHEMATICAL ACCURACY.

PROBLEM:
{problem_context.scope}

QUESTIONS TO ANSWER:
{chr(10).join(f'Q{i+1}: {q}' for i, q in enumerate(problem_context.questions))}

QUESTION COVERAGE: {answered_count}/{len(questions)} questions explicitly answered
{chr(10).join(answers_section)}

MATHEMATICAL MODEL:
Framework: {math_model.framework}
Equations: {chr(10).join(math_model.equations)}
Rationale: {math_model.rationale}

IMPLEMENTATION:
Success: {implementation.success}
Output: {implementation.execution_log[:2000] if implementation.execution_log else 'None'}
{f'Error: {implementation.error_message}' if implementation.error_message else ''}

SENSITIVITY ANALYSIS:
Stability: {sensitivity_report.stability_assessment}
Robustness: {sensitivity_report.robustness_score}

Evaluate and provide:
1. score: Overall accuracy score (0-10, be precise e.g. 7.5)
   - IMPORTANT: Penalize significantly if questions are not explicitly answered
   - Each unanswered question should reduce score by ~1-2 points
2. feedback: Detailed assessment of mathematical accuracy AND question coverage
3. strengths: What is mathematically sound
4. weaknesses: Mathematical errors, concerns, AND any unanswered questions
5. suggestions: Specific improvements for accuracy and completeness"""

        return self.generate_structured(
            prompt=prompt,
            output_type=JudgeVerdict,
            system_prompt=self.SYSTEM_PROMPT,
        )

    def evaluate_state(self, state: AgentState) -> JudgeVerdict:
        """Evaluate from full agent state."""
        verdict = self.run(
            problem_context=state.problem_context,
            math_model=state.math_model,
            implementation=state.implementation,
            sensitivity_report=state.sensitivity_report,
        )
        verdict.judge_id = "accuracy"
        return verdict
