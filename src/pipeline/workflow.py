"""LangGraph workflow definition for M3 Challenge pipeline."""

from pathlib import Path
from typing import Optional, Literal

from langgraph.graph import StateGraph, END

from ..agents import AgentState
from ..agents.input_handler import InputHandler
from ..agents.scout import Scout
from ..agents.historian import Historian
from ..agents.mathematician import Mathematician
from ..agents.coder import Coder
from ..agents.stress_tester import StressTester
from ..agents.judges import AccuracyJudge, ClarityJudge, CreativityJudge
from ..agents.orchestrator import Orchestrator


class M3Workflow:
    """
    LangGraph-based workflow for M3 Challenge solutions.

    Pipeline phases:
    1. Input Handling - Parse problem text
    2. Scouting - Deep problem analysis
    3. History - RAG-based assumption generation
    4. Mathematics - Model development
    5. Coding - Implementation
    6. Stress Testing - Sensitivity analysis
    7. Judging - Multi-agent evaluation
    8. Orchestration - Iteration or approval
    """

    def __init__(
        self,
        env_path: Optional[Path] = None,
        chroma_path: Optional[Path] = None,
    ):
        """
        Initialize the workflow with all agents.

        Args:
            env_path: Path to environment file with API keys.
            chroma_path: Path to ChromaDB for RAG.
        """
        self.env_path = env_path
        self.chroma_path = chroma_path

        # Initialize agents
        self.input_handler = InputHandler(env_path=env_path)
        self.scout = Scout(env_path=env_path)
        self.historian = Historian(
            chroma_path=chroma_path,
            env_path=env_path,
        )
        self.mathematician = Mathematician(env_path=env_path)
        self.coder = Coder(env_path=env_path)
        self.stress_tester = StressTester(env_path=env_path)

        # Initialize judges
        self.accuracy_judge = AccuracyJudge(env_path=env_path)
        self.clarity_judge = ClarityJudge(env_path=env_path)
        self.creativity_judge = CreativityJudge(env_path=env_path)

        # Initialize orchestrator
        self.orchestrator = Orchestrator(env_path=env_path)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Define the graph with AgentState
        graph = StateGraph(AgentState)

        # Add nodes for each agent
        graph.add_node("input_handler", self._run_input_handler)
        graph.add_node("scout", self._run_scout)
        graph.add_node("historian", self._run_historian)
        graph.add_node("mathematician", self._run_mathematician)
        graph.add_node("coder", self._run_coder)
        graph.add_node("stress_tester", self._run_stress_tester)
        graph.add_node("judges", self._run_judges)
        graph.add_node("orchestrator", self._run_orchestrator)

        # Define edges (linear flow with iteration loop)
        graph.set_entry_point("input_handler")

        graph.add_edge("input_handler", "scout")
        graph.add_edge("scout", "historian")
        graph.add_edge("historian", "mathematician")
        graph.add_edge("mathematician", "coder")
        graph.add_edge("coder", "stress_tester")
        graph.add_edge("stress_tester", "judges")
        graph.add_edge("judges", "orchestrator")

        # Conditional edge from orchestrator
        graph.add_conditional_edges(
            "orchestrator",
            self._should_iterate,
            {
                "historian": "historian",
                "mathematician": "mathematician",
                "coder": "coder",
                "stress_tester": "stress_tester",
                "end": END,
            },
        )

        return graph.compile()

    def _run_input_handler(self, state: AgentState) -> AgentState:
        """Run input handler node."""
        state.current_phase = "input_handling"

        # Extract raw text from state (set by runner)
        raw_text = state.problem_context.raw_text if state.problem_context else ""

        parsed = self.input_handler.run(problem_text=raw_text)

        # Update state with parsed questions
        if state.problem_context:
            state.problem_context.questions = parsed.questions
            state.problem_context.constraints = parsed.constraints
            state.problem_context.data_provided = parsed.data_references

        return state

    def _run_scout(self, state: AgentState) -> AgentState:
        """Run scout node."""
        state.current_phase = "scouting"

        # Create parsed input from current state
        from ..agents.input_handler import ParsedInput
        parsed = ParsedInput(
            questions=state.problem_context.questions,
            data_references=state.problem_context.data_provided,
            constraints=state.problem_context.constraints,
            key_terms=[],
        )

        # Run scout
        context = self.scout.run(
            parsed_input=parsed,
            raw_text=state.problem_context.raw_text,
        )

        # Preserve raw_text and merge
        context.raw_text = state.problem_context.raw_text
        state.problem_context = context

        return state

    def _run_historian(self, state: AgentState) -> AgentState:
        """Run historian node."""
        state.current_phase = "assumption_generation"

        # Check if this is a refinement iteration
        if state.assumptions and state.orchestrator_decision:
            feedback = state.orchestrator_decision.final_feedback
            assumptions = self.historian.refine_assumptions(
                current_assumptions=state.assumptions,
                feedback=feedback,
                problem_context=state.problem_context,
            )
        else:
            assumptions = self.historian.run(
                problem_context=state.problem_context,
            )

        state.assumptions = assumptions
        return state

    def _run_mathematician(self, state: AgentState) -> AgentState:
        """Run mathematician node."""
        state.current_phase = "model_development"

        # Check if this is a refinement iteration
        if state.math_model and state.orchestrator_decision:
            feedback = state.orchestrator_decision.final_feedback
            model = self.mathematician.refine_model(
                current_model=state.math_model,
                feedback=feedback,
                problem_context=state.problem_context,
            )
        else:
            model = self.mathematician.run(
                problem_context=state.problem_context,
                assumptions=state.assumptions,
            )

        state.math_model = model
        return state

    def _run_coder(self, state: AgentState) -> AgentState:
        """Run coder node."""
        state.current_phase = "implementation"

        # Check if this is a refinement iteration
        if state.implementation and state.orchestrator_decision:
            feedback = state.orchestrator_decision.final_feedback
            implementation = self.coder.refine_code(
                implementation=state.implementation,
                feedback=feedback,
            )
        else:
            implementation = self.coder.run(
                math_model=state.math_model,
                problem_context=state.problem_context,
            )

        # Handle execution errors
        if not implementation.success and implementation.error_message:
            implementation = self.coder.debug_code(
                implementation=implementation,
                error_context=implementation.error_message,
            )

        state.implementation = implementation
        return state

    def _run_stress_tester(self, state: AgentState) -> AgentState:
        """Run stress tester node."""
        state.current_phase = "sensitivity_analysis"

        # Check if this is a refinement iteration
        if state.sensitivity_report and state.orchestrator_decision:
            feedback = state.orchestrator_decision.final_feedback
            report = self.stress_tester.refine_analysis(
                current_report=state.sensitivity_report,
                feedback=feedback,
            )
        else:
            report = self.stress_tester.run(
                math_model=state.math_model,
                implementation=state.implementation,
                problem_context=state.problem_context,
            )

        state.sensitivity_report = report
        return state

    def _run_judges(self, state: AgentState) -> AgentState:
        """Run all judges in parallel."""
        state.current_phase = "judging"

        verdicts = []

        # Run each judge
        accuracy_verdict = self.accuracy_judge.evaluate_state(state)
        clarity_verdict = self.clarity_judge.evaluate_state(state)
        creativity_verdict = self.creativity_judge.evaluate_state(state)

        verdicts = [accuracy_verdict, clarity_verdict, creativity_verdict]
        state.judge_verdicts = verdicts

        return state

    def _run_orchestrator(self, state: AgentState) -> AgentState:
        """Run orchestrator node."""
        state.current_phase = "orchestration"

        decision = self.orchestrator.run(
            state=state,
            judge_verdicts=state.judge_verdicts,
        )

        # Log iteration
        if decision.should_iterate:
            state.iteration_history.append(
                f"Iteration {decision.iteration_count + 1}: Routing to {decision.iteration_target} - {decision.iteration_reason}"
            )

        state.orchestrator_decision = decision
        return state

    def _should_iterate(self, state: AgentState) -> Literal["historian", "mathematician", "coder", "stress_tester", "end"]:
        """Determine next node based on orchestrator decision."""
        decision = state.orchestrator_decision

        if not decision or not decision.should_iterate:
            return "end"

        target = decision.iteration_target
        if target in ("historian", "mathematician", "coder", "stress_tester"):
            return target

        return "end"

    def run(self, problem_text: str) -> AgentState:
        """
        Execute the full workflow.

        Args:
            problem_text: Raw problem text.

        Returns:
            Final AgentState with all results.
        """
        from ..agents.models import ProblemContext

        # Initialize state with problem text
        initial_state = AgentState(
            problem_context=ProblemContext(
                raw_text=problem_text,
                questions=[],
                scope="",
                real_problem="",
            ),
        )

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        return final_state

    def run_with_context(self, context) -> AgentState:
        """
        Execute workflow with pre-built context including data.

        Args:
            context: ProblemContext with data summary already populated.

        Returns:
            Final AgentState with all results.
        """
        # Initialize state with enriched context
        initial_state = AgentState(
            problem_context=context,
        )

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        return final_state


def create_workflow(
    env_path: Optional[Path] = None,
    chroma_path: Optional[Path] = None,
) -> M3Workflow:
    """
    Factory function to create an M3Workflow.

    Args:
        env_path: Path to environment file.
        chroma_path: Path to ChromaDB.

    Returns:
        Configured M3Workflow instance.
    """
    return M3Workflow(env_path=env_path, chroma_path=chroma_path)
