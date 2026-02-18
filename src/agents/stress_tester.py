"""Stress-Tester agent - performs sensitivity analysis on models."""

from pathlib import Path
from typing import Optional

from .base import BaseAgent
from .models import MathModel, Implementation, SensitivityReport, ProblemContext


class StressTester(BaseAgent):
    """
    Performs sensitivity analysis and stress testing on M3 solutions.

    Evaluates:
    - Parameter sensitivity (±10%, ±20% perturbations)
    - Model robustness to input changes
    - Identification of critical parameters
    - Strengths and weaknesses assessment
    """

    agent_name = "stress_tester"
    default_provider = "openai"
    default_model = "gpt-4o"

    SYSTEM_PROMPT = """You are an expert at sensitivity analysis for mathematical models.

Your role is to:
1. Identify which parameters to perturb
2. Analyze how output changes with parameter variations
3. Assess overall model robustness
4. Identify strengths and weaknesses
5. Provide actionable recommendations

M3 judges value:
- Systematic sensitivity analysis (not just one parameter)
- Clear presentation of results (tables, ranges)
- Honest assessment of model limitations
- Quantified impact of parameter changes

Standard sensitivity analysis:
- Test ±10% and ±20% perturbations
- Report relative change in outputs
- Identify which parameters have largest impact
- Classify stability: Stable (<5% output change), Moderate (5-20%), Sensitive (>20%)"""

    def __init__(
        self,
        query_engine=None,
        env_path: Optional[Path] = None,
        **kwargs,
    ):
        """
        Initialize StressTester with optional RAG access.

        Args:
            query_engine: Optional QueryEngine for sensitivity analysis examples.
            env_path: Path to environment file.
        """
        super().__init__(env_path=env_path, **kwargs)
        self.query_engine = query_engine

    def run(
        self,
        math_model: MathModel,
        implementation: Implementation,
        problem_context: ProblemContext,
    ) -> SensitivityReport:
        """
        Perform sensitivity analysis on the model.

        Args:
            math_model: Mathematical model specification.
            implementation: Code implementation with results.
            problem_context: Problem context for reference.

        Returns:
            SensitivityReport with analysis results.
        """
        # Get historical sensitivity examples if available
        historical_examples = ""
        if self.query_engine:
            results = self.query_engine.search(
                query=f"sensitivity analysis {math_model.framework}",
                section_type="sensitivity",
                n_results=5,
            )
            if results.results:
                historical_examples = self._format_examples(results)

        # Format model info
        parameters_text = "\n".join(
            f"  {k}: {v}" for k, v in math_model.parameters.items()
        )

        prompt = f"""Perform sensitivity analysis on this M3 model.

MODEL FRAMEWORK: {math_model.framework}

PARAMETERS TO ANALYZE:
{parameters_text}

MODEL EQUATIONS:
{chr(10).join(math_model.equations)}

IMPLEMENTATION OUTPUT:
{implementation.execution_log[:3000] if implementation.execution_log else 'No output captured'}

CODE SUCCESS: {implementation.success}
{f'ERROR: {implementation.error_message}' if implementation.error_message else ''}

PROBLEM SCOPE:
{problem_context.scope}
"""

        if historical_examples:
            prompt += f"""
SENSITIVITY ANALYSIS EXAMPLES FROM PAST M3 WINNERS:
{historical_examples}
"""

        prompt += """
Perform comprehensive sensitivity analysis:

1. perturbations: For each key parameter, estimate impact of ±10% and ±20% changes
   Format: {"param_name": {"change": "±10%", "impact": "X% change in output"}}

2. stability_assessment: Overall classification (Stable/Moderately Sensitive/Highly Sensitive)

3. critical_parameters: List parameters with highest impact on results

4. robustness_score: 0-1 score (1 = very robust)

5. strengths: What the model does well

6. weaknesses: Limitations and sensitivity issues

7. recommendations: Specific suggestions for improvement"""

        return self.generate_structured(
            prompt=prompt,
            output_type=SensitivityReport,
            system_prompt=self.SYSTEM_PROMPT,
        )

    def _format_examples(self, results) -> str:
        """Format historical sensitivity analysis examples."""
        formatted = []
        for i, r in enumerate(results.results, 1):
            formatted.append(f"[{i}] {r.source}:\n{r.assumption[:400]}...")
        return "\n\n".join(formatted)

    def generate_sensitivity_code(
        self,
        implementation: Implementation,
        math_model: MathModel,
    ) -> str:
        """
        Generate Python code for automated sensitivity analysis.

        Args:
            implementation: Original implementation.
            math_model: Model specification.

        Returns:
            Python code for sensitivity testing.
        """
        parameters = list(math_model.parameters.keys())

        prompt = f"""Generate Python code to perform automated sensitivity analysis.

ORIGINAL CODE:
```python
{implementation.code}
```

PARAMETERS TO TEST:
{', '.join(parameters)}

Generate code that:
1. Wraps the model in a function
2. Tests each parameter at ±10% and ±20%
3. Records output changes
4. Generates a summary table
5. Creates a tornado chart visualization

Return only the Python code."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        return self._call_llm(messages=messages, max_tokens=4000)

    def refine_analysis(
        self,
        current_report: SensitivityReport,
        feedback: str,
    ) -> SensitivityReport:
        """
        Refine sensitivity analysis based on feedback.

        Args:
            current_report: Current sensitivity report.
            feedback: Feedback from judges.

        Returns:
            Refined SensitivityReport.
        """
        prompt = f"""Refine this sensitivity analysis based on feedback.

CURRENT ANALYSIS:
Stability: {current_report.stability_assessment}
Critical Parameters: {', '.join(current_report.critical_parameters)}
Robustness Score: {current_report.robustness_score}

Strengths: {chr(10).join(f'- {s}' for s in current_report.strengths)}
Weaknesses: {chr(10).join(f'- {w}' for w in current_report.weaknesses)}

FEEDBACK:
{feedback}

Improve the analysis to address the feedback."""

        return self.generate_structured(
            prompt=prompt,
            output_type=SensitivityReport,
            system_prompt=self.SYSTEM_PROMPT,
        )
