"""Scout agent - deep problem analysis and context building."""

from .base import BaseAgent
from .models import ProblemContext
from .input_handler import ParsedInput


class Scout(BaseAgent):
    """
    Analyzes the parsed problem to understand scope and real objectives.

    Goes beyond surface-level parsing to identify:
    - What we're really solving (vs what is stated)
    - Hidden constraints and assumptions
    - Success criteria
    - Key variables for modeling
    """

    agent_name = "scout"
    default_provider = "openai"
    default_model = "gpt-4o"

    SYSTEM_PROMPT = """You are an expert M3 Math Modeling Challenge problem analyst.

Your role is to deeply analyze competition problems to uncover:
1. The TRUE problem being asked (often different from the stated problem)
2. Hidden constraints that aren't explicitly mentioned
3. What success looks like for judges
4. Critical variables that must be modeled

M3 judges value:
- Clear problem restatement that shows deep understanding
- Identification of the "real" problem beneath the surface
- Recognition of scope boundaries
- Anticipation of what makes a winning solution

Be thorough but concise. Focus on insights that will guide the mathematical modeling."""

    def run(
        self,
        parsed_input: ParsedInput,
        raw_text: str,
    ) -> ProblemContext:
        """
        Analyze parsed input to build full problem context.

        Args:
            parsed_input: Output from InputHandler.
            raw_text: Original problem text for reference.

        Returns:
            ProblemContext with deep analysis.
        """
        prompt = f"""Analyze this M3 Challenge problem to build a complete problem context.

PARSED QUESTIONS:
{self._format_questions(parsed_input.questions)}

DATA REFERENCES:
{', '.join(parsed_input.data_references) if parsed_input.data_references else 'None specified'}

STATED CONSTRAINTS:
{', '.join(parsed_input.constraints) if parsed_input.constraints else 'None specified'}

KEY TERMS:
{', '.join(parsed_input.key_terms) if parsed_input.key_terms else 'None identified'}

ORIGINAL PROBLEM TEXT:
{raw_text[:3000]}...

Provide:
1. scope: What exactly we are solving for (be specific about deliverables)
2. real_problem: The underlying issue or challenge (often different from stated problem)
3. key_variables: Critical variables that must be included in any model
4. success_criteria: What judges will look for in a winning solution
5. constraints: Both explicit and implicit constraints (expand on the parsed ones)
6. data_provided: What data is available and how it might be used"""

        return self.generate_structured(
            prompt=prompt,
            output_type=ProblemContext,
            system_prompt=self.SYSTEM_PROMPT,
        )

    def _format_questions(self, questions: list[str]) -> str:
        """Format questions for prompt."""
        if not questions:
            return "No questions parsed"

        formatted = []
        for i, q in enumerate(questions, 1):
            formatted.append(f"Q{i}: {q}")

        return "\n".join(formatted)

    def refine_context(
        self,
        context: ProblemContext,
        feedback: str,
    ) -> ProblemContext:
        """
        Refine problem context based on feedback from later agents.

        Args:
            context: Current problem context.
            feedback: Feedback from judges or other agents.

        Returns:
            Refined ProblemContext.
        """
        prompt = f"""Refine this problem context based on feedback received.

CURRENT CONTEXT:
Scope: {context.scope}
Real Problem: {context.real_problem}
Key Variables: {', '.join(context.key_variables)}
Success Criteria: {', '.join(context.success_criteria)}

FEEDBACK:
{feedback}

Update the context to address the feedback while maintaining the core understanding of the problem."""

        return self.generate_structured(
            prompt=prompt,
            output_type=ProblemContext,
            system_prompt=self.SYSTEM_PROMPT,
        )
