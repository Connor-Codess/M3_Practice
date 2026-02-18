"""Mathematician agent - develops mathematical models based on problem and assumptions."""

from pathlib import Path
from typing import Optional

from .base import BaseAgent
from .models import ProblemContext, AssumptionSet, MathModel


class Mathematician(BaseAgent):
    """
    Develops rigorous mathematical models for M3 problems.

    Uses o1 reasoning model for:
    - Framework selection (ODE, optimization, stochastic, etc.)
    - Equation formulation
    - Variable and parameter definition
    - Boundary condition specification
    """

    agent_name = "mathematician"
    default_provider = "openai"
    default_model = "o1"

    SYSTEM_PROMPT = """You are an expert mathematical modeler for the M3 Math Modeling Challenge.

Your role is to develop rigorous mathematical frameworks that:
1. Directly address the problem questions
2. Are grounded in the stated assumptions
3. Use appropriate mathematical techniques for the domain
4. Include clear variable and parameter definitions
5. Specify boundary/initial conditions

M3 judges value:
- Appropriate model selection with justification
- Clear mathematical notation (LaTeX)
- Explicit connection between model and assumptions
- Acknowledgment of model limitations

Common modeling approaches in M3:
- Differential equations (logistic growth, SIR models, etc.)
- Optimization (linear programming, nonlinear optimization)
- Statistical models (regression, time series)
- Stochastic models (Markov chains, Monte Carlo)
- Agent-based models (for complex systems)

Select the most appropriate approach and justify your choice."""

    def __init__(
        self,
        query_engine=None,
        env_path: Optional[Path] = None,
        **kwargs,
    ):
        """
        Initialize Mathematician with optional RAG access.

        Args:
            query_engine: Optional QueryEngine for model lookup.
            env_path: Path to environment file.
        """
        # o1 model has different constraints
        kwargs.setdefault("temperature", 1.0)  # o1 requires temperature=1
        super().__init__(env_path=env_path, **kwargs)
        self.query_engine = query_engine

    def _call_llm(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> str:
        """Override for o1 model which has different API requirements."""
        if self.model.startswith("o1"):
            # o1 model doesn't support system messages or temperature
            # Combine system message into user message
            combined_content = ""
            user_content = ""

            for msg in messages:
                if msg["role"] == "system":
                    combined_content = msg["content"] + "\n\n"
                elif msg["role"] == "user":
                    user_content = msg["content"]

            messages = [{"role": "user", "content": combined_content + user_content}]

            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content

        return super()._call_llm(messages, max_tokens, response_format)

    def run(
        self,
        problem_context: ProblemContext,
        assumptions: AssumptionSet,
    ) -> MathModel:
        """
        Develop a mathematical model for the problem.

        Args:
            problem_context: Context from Scout agent.
            assumptions: Assumptions from Historian agent.

        Returns:
            MathModel specification.
        """
        # Optionally search for similar models
        historical_models = ""
        if self.query_engine:
            model_results = self.query_engine.search_models(
                query=problem_context.scope,
                n_results=5,
            )
            if model_results.results:
                historical_models = self._format_model_results(model_results)

        # Format assumptions
        assumptions_text = "\n".join(
            f"- [{a.category}] {a.assumption}"
            for a in assumptions.assumptions
        )

        prompt = f"""Develop a mathematical model for this M3 Challenge problem.

PROBLEM SCOPE:
{problem_context.scope}

REAL PROBLEM:
{problem_context.real_problem}

KEY VARIABLES TO MODEL:
{', '.join(problem_context.key_variables)}

QUESTIONS TO ANSWER:
{chr(10).join(f'Q{i+1}: {q}' for i, q in enumerate(problem_context.questions))}

ASSUMPTIONS (your model must respect these):
{assumptions_text}

SUCCESS CRITERIA:
{', '.join(problem_context.success_criteria)}
"""

        if historical_models:
            prompt += f"""
SIMILAR MODELS FROM PAST M3 WINNERS:
{historical_models}
"""

        prompt += """
Provide:
1. framework: The mathematical framework (e.g., "Logistic Growth ODE", "Markov Chain", "Linear Programming")
2. equations: List of LaTeX equations defining the model
3. variables: Dictionary mapping variable symbols to descriptions
4. parameters: Dictionary mapping parameter symbols to descriptions
5. rationale: Why this framework is appropriate for this problem
6. boundary_conditions: Initial or boundary conditions
7. assumptions_used: Which assumptions from the list this model relies on"""

        return self.generate_structured(
            prompt=prompt,
            output_type=MathModel,
            system_prompt=self.SYSTEM_PROMPT,
        )

    def _format_model_results(self, results) -> str:
        """Format model search results for prompt."""
        formatted = []
        for i, r in enumerate(results.results, 1):
            formatted.append(f"[{i}] {r.source}: {r.assumption[:300]}...")
        return "\n".join(formatted)

    def refine_model(
        self,
        current_model: MathModel,
        feedback: str,
        problem_context: ProblemContext,
    ) -> MathModel:
        """
        Refine model based on feedback.

        Args:
            current_model: Current mathematical model.
            feedback: Feedback from judges or other agents.
            problem_context: Original problem context.

        Returns:
            Refined MathModel.
        """
        prompt = f"""Refine this mathematical model based on feedback.

CURRENT MODEL:
Framework: {current_model.framework}
Equations: {chr(10).join(current_model.equations)}
Rationale: {current_model.rationale}

FEEDBACK:
{feedback}

PROBLEM SCOPE:
{problem_context.scope}

Improve the model to address the feedback while maintaining mathematical rigor."""

        return self.generate_structured(
            prompt=prompt,
            output_type=MathModel,
            system_prompt=self.SYSTEM_PROMPT,
        )
