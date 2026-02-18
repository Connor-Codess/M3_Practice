"""Orchestrator agent - coordinates the multi-agent pipeline."""

from pathlib import Path
from typing import Optional

from .base import BaseAgent
from .models import (
    AgentState,
    JudgeVerdict,
    OrchestratorDecision,
)


class Orchestrator(BaseAgent):
    """
    Coordinates the M3 multi-agent solution pipeline.

    Responsibilities:
    - Synthesize feedback from all judges
    - Decide whether to iterate or finalize
    - Route iterations to appropriate agents
    - Track iteration history
    - Make final approval decisions
    """

    agent_name = "orchestrator"
    default_provider = "openai"
    default_model = "gpt-4o"

    # Score threshold for approval
    APPROVAL_THRESHOLD = 9.0
    MAX_ITERATIONS = 3

    SYSTEM_PROMPT = """You are the orchestrator for an M3 Math Modeling Challenge solution system.

Your role is to:
1. Synthesize feedback from three judges (Accuracy, Clarity, Creativity)
2. Decide if the solution meets competition standards (score >= 9.0)
3. If iteration is needed, identify which agent should revise their work
4. Track progress across iterations

Decision Guidelines:
- Approve (should_iterate=False) if consensus score >= 9.0
- Iterate if any critical issues remain
- Target the agent whose domain had lowest scores:
  * Low accuracy → mathematician
  * Low clarity → coder
  * Low creativity → historian
  * Low assumption quality → historian
- Maximum 3 iterations to prevent infinite loops

Be decisive. Synthesize judge feedback into actionable direction."""

    def run(
        self,
        state: AgentState,
        judge_verdicts: list[JudgeVerdict],
    ) -> OrchestratorDecision:
        """
        Make orchestration decision based on judge verdicts.

        Args:
            state: Current pipeline state.
            judge_verdicts: Verdicts from all judges.

        Returns:
            OrchestratorDecision with next steps.
        """
        # Calculate scores
        scores = {v.judge_id: v.score for v in judge_verdicts}
        consensus_score = sum(scores.values()) / len(scores) if scores else 0

        # Check iteration count
        current_iteration = len(state.iteration_history)

        # Build prompt
        verdicts_text = self._format_verdicts(judge_verdicts)
        history_text = "\n".join(state.iteration_history) if state.iteration_history else "No previous iterations"

        prompt = f"""Make an orchestration decision for this M3 solution.

JUDGE VERDICTS:
{verdicts_text}

SCORES:
{chr(10).join(f'- {k}: {v:.1f}' for k, v in scores.items())}
Consensus Score: {consensus_score:.2f}

ITERATION HISTORY:
{history_text}

Current Iteration: {current_iteration} of {self.MAX_ITERATIONS}
Approval Threshold: {self.APPROVAL_THRESHOLD}

DECISION RULES:
- If consensus score >= {self.APPROVAL_THRESHOLD}: Approve (should_iterate=False)
- If current iteration >= {self.MAX_ITERATIONS}: Approve anyway, note limitations
- Otherwise: Iterate to the agent who can best address the lowest-scoring area

Provide:
1. should_iterate: Whether another iteration is needed (True/False)
2. iteration_target: Which agent to route to (historian/mathematician/coder/stress_tester) or null if approved
3. iteration_reason: Why iteration is needed (or why approved)
4. final_feedback: Synthesized feedback for the team"""

        decision = self.generate_structured(
            prompt=prompt,
            output_type=OrchestratorDecision,
            system_prompt=self.SYSTEM_PROMPT,
        )

        # Override fields with calculated values
        decision.consensus_score = consensus_score
        decision.individual_scores = scores
        decision.iteration_count = current_iteration
        decision.max_iterations = self.MAX_ITERATIONS

        # Force approval if max iterations reached
        if current_iteration >= self.MAX_ITERATIONS and decision.should_iterate:
            decision.should_iterate = False
            decision.iteration_reason = f"Maximum iterations ({self.MAX_ITERATIONS}) reached. Finalizing with current score of {consensus_score:.1f}"

        return decision

    def _format_verdicts(self, verdicts: list[JudgeVerdict]) -> str:
        """Format judge verdicts for prompt."""
        formatted = []
        for v in verdicts:
            entry = f"""[{v.judge_id.upper()}] Score: {v.score:.1f}
Feedback: {v.feedback}
Strengths: {', '.join(v.strengths[:3])}
Weaknesses: {', '.join(v.weaknesses[:3])}
Suggestions: {', '.join(v.suggestions[:2])}"""
            formatted.append(entry)

        return "\n\n".join(formatted)

    def synthesize_final_report(
        self,
        state: AgentState,
        decision: OrchestratorDecision,
    ) -> str:
        """
        Generate a final summary report.

        Args:
            state: Final pipeline state.
            decision: Final orchestrator decision.

        Returns:
            Formatted summary report.
        """
        # Format question answers summary
        questions = state.problem_context.questions
        answers = state.implementation.question_answers
        answered_count = sum(1 for i in range(1, len(questions) + 1) if f"Q{i}" in answers)

        answers_summary = []
        for i, q in enumerate(questions, 1):
            q_key = f"Q{i}"
            q_short = q[:80] + "..." if len(q) > 80 else q
            status = "ANSWERED" if q_key in answers else "NOT CAPTURED"
            answers_summary.append(f"  Q{i}: [{status}] {q_short}")

        report = f"""
M3 SOLUTION SUMMARY
==================

FINAL SCORE: {decision.consensus_score:.1f}/10

INDIVIDUAL SCORES:
{chr(10).join(f'  {k.capitalize()}: {v:.1f}' for k, v in decision.individual_scores.items())}

PROBLEM:
{state.problem_context.scope}

QUESTIONS ANSWERED: {answered_count}/{len(questions)}
{chr(10).join(answers_summary)}

MODEL FRAMEWORK:
{state.math_model.framework}

KEY ASSUMPTIONS ({len(state.assumptions.assumptions)}):
{chr(10).join(f'  - {a.assumption}' for a in state.assumptions.assumptions[:5])}

IMPLEMENTATION STATUS:
  Success: {state.implementation.success}
  Visualizations: {len(state.implementation.visualizations)}

SENSITIVITY:
  Stability: {state.sensitivity_report.stability_assessment}
  Robustness: {state.sensitivity_report.robustness_score:.2f}

ITERATIONS: {decision.iteration_count}

FINAL FEEDBACK:
{decision.final_feedback}
"""
        return report


def create_orchestrator(
    env_path: Optional[Path] = None,
    approval_threshold: float = 9.0,
    max_iterations: int = 3,
) -> Orchestrator:
    """
    Factory function to create an Orchestrator.

    Args:
        env_path: Path to environment file.
        approval_threshold: Score threshold for approval.
        max_iterations: Maximum iteration count.

    Returns:
        Configured Orchestrator instance.
    """
    orchestrator = Orchestrator(env_path=env_path)
    orchestrator.APPROVAL_THRESHOLD = approval_threshold
    orchestrator.MAX_ITERATIONS = max_iterations
    return orchestrator
