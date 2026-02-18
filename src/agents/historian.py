"""Historian agent - retrieves and formulates justified assumptions using RAG."""

from pathlib import Path
from typing import Optional

from .base import BaseAgent
from .models import ProblemContext, JustifiedAssumption, AssumptionSet


class Historian(BaseAgent):
    """
    Formulates assumptions grounded in historical M3 winning solutions.

    Uses the RAG system to:
    - Find similar past problems
    - Retrieve justified assumptions from winners
    - Ground new assumptions in proven approaches
    """

    agent_name = "historian"
    default_provider = "openai"
    default_model = "gpt-4o"

    SYSTEM_PROMPT = """You are an expert at formulating justified assumptions for M3 Math Modeling problems.

Your role is to:
1. Analyze the problem context to identify what assumptions are needed
2. Use historical examples from past M3 winners to ground your assumptions
3. Provide clear justifications for each assumption

M3 judges highly value:
- Assumptions that are explicitly stated and justified
- Grounding in real-world reasoning or precedent
- Categorization of assumptions (data, model, scope, simplification)
- Recognition of which assumptions are most critical

Every assumption must have a justification. Unjustified assumptions lose points."""

    def __init__(
        self,
        query_engine=None,
        chroma_path: Optional[Path] = None,
        env_path: Optional[Path] = None,
        **kwargs,
    ):
        """
        Initialize Historian with RAG query engine.

        Args:
            query_engine: Pre-configured QueryEngine instance.
            chroma_path: Path to ChromaDB (if query_engine not provided).
            env_path: Path to environment file.
        """
        super().__init__(env_path=env_path, **kwargs)

        if query_engine:
            self.query_engine = query_engine
        else:
            # Initialize query engine from path
            from ..retrieval import get_query_engine
            self.query_engine = get_query_engine(
                chroma_path=chroma_path,
                env_path=env_path,
            )

    def run(
        self,
        problem_context: ProblemContext,
        n_historical_examples: int = 10,
    ) -> AssumptionSet:
        """
        Generate justified assumptions for the problem.

        Args:
            problem_context: Context from Scout agent.
            n_historical_examples: Number of historical examples to retrieve.

        Returns:
            AssumptionSet with justified assumptions.
        """
        # Build search query from problem context
        search_query = self._build_search_query(problem_context)

        # Retrieve historical assumptions
        historical_results = self.query_engine.search_assumptions(
            query=search_query,
            n_results=n_historical_examples,
            with_justification=True,
        )

        # Format historical examples for LLM
        historical_context = self._format_historical_results(historical_results)

        # Build data context section if available
        data_context_section = ""
        if problem_context.data_summary:
            data_context_section = f"""

AVAILABLE DATA:
{problem_context.data_context_string}
"""

        # Generate assumptions
        prompt = f"""Generate a comprehensive set of justified assumptions for this M3 problem.

PROBLEM CONTEXT:
Scope: {problem_context.scope}
Real Problem: {problem_context.real_problem}
Key Variables: {', '.join(problem_context.key_variables)}
Success Criteria: {', '.join(problem_context.success_criteria)}
Constraints: {', '.join(problem_context.constraints)}
{data_context_section}
HISTORICAL EXAMPLES FROM PAST M3 WINNERS:
{historical_context}

Generate assumptions that:
1. Cover data assumptions (about the given data - consider column types, missing values, date ranges)
2. Cover model assumptions (about the mathematical approach)
3. Cover scope assumptions (about problem boundaries)
4. Cover simplification assumptions (justified simplifications)

Each assumption must have:
- A clear statement
- A justification (why it's reasonable)
- A category (data, model, scope, simplification)
- Reference to historical source if applicable"""

        return self.generate_structured(
            prompt=prompt,
            output_type=AssumptionSet,
            system_prompt=self.SYSTEM_PROMPT,
        )

    def _build_search_query(self, context: ProblemContext) -> str:
        """Build a search query from problem context, including data characteristics."""
        parts = [context.scope]

        if context.key_variables:
            parts.append(" ".join(context.key_variables[:5]))

        if context.real_problem:
            parts.append(context.real_problem[:200])

        # Add data-aware query terms if data summary is available
        if context.data_summary:
            data_terms = self._extract_data_query_terms(context.data_summary)
            if data_terms:
                parts.append(data_terms)

        return " ".join(parts)

    def _extract_data_query_terms(self, data_summary) -> str:
        """Extract query-relevant terms from data summary."""
        terms = []

        # Add date range info for time series problems
        if data_summary.date_range:
            terms.append("time series")
            terms.append("temporal data")

        # Add geographic info
        if data_summary.geographic_scope:
            terms.append("geographic")
            terms.append("regional")

        # Look for common data patterns in column types
        numeric_cols = [
            name for name, info in data_summary.columns.items()
            if info.dtype == "numeric"
        ]
        categorical_cols = [
            name for name, info in data_summary.columns.items()
            if info.dtype == "categorical"
        ]

        if len(numeric_cols) > 3:
            terms.append("multivariate")

        if categorical_cols:
            terms.append("categorical variables")

        # Check for specific column patterns that suggest problem types
        col_names_lower = " ".join(c.lower() for c in data_summary.columns.keys())

        if "price" in col_names_lower or "cost" in col_names_lower:
            terms.append("economic")
            terms.append("price prediction")

        if "population" in col_names_lower or "demographic" in col_names_lower:
            terms.append("demographic")
            terms.append("population")

        if "growth" in col_names_lower or "rate" in col_names_lower:
            terms.append("growth modeling")

        return " ".join(terms[:5])  # Limit to avoid query noise

    def _format_historical_results(self, results) -> str:
        """Format RAG results for the LLM prompt."""
        if not results.results:
            return "No relevant historical examples found."

        formatted = []
        for i, result in enumerate(results.results, 1):
            entry = f"""[Example {i}] {result.source} (relevance: {result.relevance_score:.2f})
Assumption: {result.assumption}"""

            if result.justification:
                entry += f"\nJustification: {result.justification}"

            formatted.append(entry)

        return "\n\n".join(formatted)

    def search_similar_problems(
        self,
        problem_description: str,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Search for similar past problems.

        Args:
            problem_description: Description of current problem.
            n_results: Number of results to return.

        Returns:
            List of similar problem references.
        """
        results = self.query_engine.get_similar_problems(
            problem_description=problem_description,
            n_results=n_results,
        )

        return [
            {
                "source": r.source,
                "content": r.assumption,
                "relevance": r.relevance_score,
            }
            for r in results.results
        ]

    def refine_assumptions(
        self,
        current_assumptions: AssumptionSet,
        feedback: str,
        problem_context: ProblemContext,
    ) -> AssumptionSet:
        """
        Refine assumptions based on feedback.

        Args:
            current_assumptions: Current assumption set.
            feedback: Feedback from judges or other agents.
            problem_context: Original problem context.

        Returns:
            Refined AssumptionSet.
        """
        # Format current assumptions
        current_formatted = "\n".join(
            f"- [{a.category}] {a.assumption}: {a.justification}"
            for a in current_assumptions.assumptions
        )

        prompt = f"""Refine these assumptions based on the feedback received.

CURRENT ASSUMPTIONS:
{current_formatted}

FEEDBACK:
{feedback}

PROBLEM CONTEXT:
{problem_context.scope}

Improve the assumptions to address the feedback while maintaining mathematical rigor."""

        return self.generate_structured(
            prompt=prompt,
            output_type=AssumptionSet,
            system_prompt=self.SYSTEM_PROMPT,
        )
