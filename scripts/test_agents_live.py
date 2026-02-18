#!/usr/bin/env python3
"""
Live API tests for all M3 agents.

This script tests each agent with real API calls to verify:
1. API connectivity works
2. Structured output parsing is correct
3. No rate limits or model issues

Estimated cost: ~$1-2 for all tests

Usage:
    python scripts/test_agents_live.py
    python scripts/test_agents_live.py --agent input_handler  # Test single agent
    python scripts/test_agents_live.py --skip-expensive       # Skip o1 tests
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv


SAMPLE_PROBLEM = """
M3 Challenge 2024 - Housing Affordability

Background:
Housing affordability has become a critical issue in many metropolitan areas.
Young professionals are increasingly unable to purchase homes in the cities
where they work. This problem has implications for urban development,
transportation, and social mobility.

Data Provided:
- Table 1: Median home prices by city (2010-2023)
- Table 2: Median household income by city (2010-2023)
- Table 3: Population growth rates by metropolitan area

Q1: Develop a model to predict housing affordability (defined as the ratio of
median home price to median household income) for the next 10 years in a
metropolitan area of your choice.

Q2: Analyze the sensitivity of your model to changes in key parameters such as
interest rates, population growth, and income growth. Identify which factors
have the greatest impact on affordability.

Q3: Based on your model, provide policy recommendations that could improve
housing affordability while maintaining economic growth. Consider trade-offs
and unintended consequences.

Constraints:
- 25 pages maximum
- All assumptions must be justified
- Include sensitivity analysis
- Provide visualizations of key results
"""


class TestRunner:
    """Runs and reports on agent tests."""

    def __init__(self, env_path: Path, chroma_path: Optional[Path] = None):
        self.env_path = env_path
        self.chroma_path = chroma_path
        self.results = []
        load_dotenv(env_path)

    def run_test(
        self,
        name: str,
        test_func: Callable,
        skip: bool = False,
        skip_reason: str = "",
    ) -> dict:
        """Run a single test and record results."""
        result = {
            "name": name,
            "status": "unknown",
            "duration": 0,
            "error": None,
            "output_preview": None,
        }

        if skip:
            result["status"] = "skipped"
            result["error"] = skip_reason
            self.results.append(result)
            return result

        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print("="*60)

        start = time.time()
        try:
            output = test_func()
            result["status"] = "passed"
            result["duration"] = time.time() - start

            # Store preview of output
            if hasattr(output, "model_dump"):
                output_dict = output.model_dump()
                result["output_preview"] = json.dumps(output_dict, indent=2)[:500]
            elif isinstance(output, str):
                result["output_preview"] = output[:500]
            else:
                result["output_preview"] = str(output)[:500]

            print(f"  Status: PASSED ({result['duration']:.1f}s)")
            print(f"  Output preview: {result['output_preview'][:200]}...")

        except Exception as e:
            result["status"] = "failed"
            result["duration"] = time.time() - start
            result["error"] = str(e)
            print(f"  Status: FAILED")
            print(f"  Error: {e}")

        self.results.append(result)
        return result

    def print_summary(self):
        """Print test results summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        passed = sum(1 for r in self.results if r["status"] == "passed")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        skipped = sum(1 for r in self.results if r["status"] == "skipped")
        total = len(self.results)

        for r in self.results:
            status_icon = {
                "passed": "[PASS]",
                "failed": "[FAIL]",
                "skipped": "[SKIP]",
            }.get(r["status"], "[????]")

            duration = f"({r['duration']:.1f}s)" if r["duration"] > 0 else ""
            error = f" - {r['error']}" if r["error"] and r["status"] == "failed" else ""

            print(f"  {status_icon} {r['name']} {duration}{error}")

        print()
        print(f"Total: {passed} passed, {failed} failed, {skipped} skipped ({total} total)")

        if failed > 0:
            print("\nFailed tests need attention before running the full pipeline.")
            return False
        return True


def test_input_handler(env_path: Path):
    """Test InputHandler with Gemini 2.0 Flash."""
    from src.agents.input_handler import InputHandler

    handler = InputHandler(env_path=env_path)
    result = handler.run(problem_text=SAMPLE_PROBLEM)

    # Validate output
    assert result.questions, "No questions parsed"
    assert len(result.questions) >= 3, f"Expected 3 questions, got {len(result.questions)}"

    print(f"  Parsed {len(result.questions)} questions")
    print(f"  Found {len(result.data_references)} data references")
    print(f"  Found {len(result.constraints)} constraints")

    return result


def test_scout(env_path: Path):
    """Test Scout with Claude Sonnet 4."""
    from src.agents.scout import Scout
    from src.agents.input_handler import ParsedInput

    scout = Scout(env_path=env_path)

    # Create parsed input
    parsed = ParsedInput(
        questions=[
            "Develop a model to predict housing affordability for the next 10 years",
            "Analyze sensitivity to interest rates, population growth, income growth",
            "Provide policy recommendations for housing affordability",
        ],
        data_references=["Table 1: Median home prices", "Table 2: Median income"],
        constraints=["25 pages maximum", "Include sensitivity analysis"],
        key_terms=["housing affordability", "metropolitan area", "policy"],
    )

    result = scout.run(parsed_input=parsed, raw_text=SAMPLE_PROBLEM)

    # Validate output
    assert result.scope, "No scope defined"
    assert result.real_problem, "No real problem identified"
    assert result.key_variables, "No key variables identified"

    print(f"  Scope: {result.scope[:100]}...")
    print(f"  Key variables: {result.key_variables[:5]}")

    return result


def test_historian(env_path: Path, chroma_path: Path):
    """Test Historian with GPT-4o and RAG."""
    from src.agents.historian import Historian
    from src.agents.models import ProblemContext

    historian = Historian(
        chroma_path=chroma_path,
        env_path=env_path,
    )

    context = ProblemContext(
        raw_text=SAMPLE_PROBLEM,
        questions=[
            "Predict housing affordability for next 10 years",
            "Analyze sensitivity of the model",
            "Provide policy recommendations",
        ],
        scope="Predict housing affordability trends in a metropolitan area",
        real_problem="Balancing housing costs with income growth while maintaining economic development",
        key_variables=["home prices", "household income", "interest rates", "population growth"],
        constraints=["25 pages maximum"],
        data_provided=["Median home prices 2010-2023", "Median income 2010-2023"],
        success_criteria=["Accurate predictions", "Realistic assumptions", "Actionable recommendations"],
    )

    result = historian.run(problem_context=context)

    # Validate output
    assert result.assumptions, "No assumptions generated"
    assert len(result.assumptions) >= 3, f"Expected at least 3 assumptions, got {len(result.assumptions)}"

    print(f"  Generated {len(result.assumptions)} assumptions")
    for a in result.assumptions[:2]:
        print(f"    - [{a.category}] {a.assumption[:60]}...")

    return result


def test_mathematician(env_path: Path):
    """Test Mathematician with o1 model."""
    from src.agents.mathematician import Mathematician
    from src.agents.models import ProblemContext, AssumptionSet, JustifiedAssumption

    mathematician = Mathematician(env_path=env_path)

    context = ProblemContext(
        raw_text=SAMPLE_PROBLEM,
        questions=[
            "Predict housing affordability for next 10 years",
            "Analyze sensitivity of the model",
            "Provide policy recommendations",
        ],
        scope="Predict housing affordability trends",
        real_problem="Balancing housing costs with income growth",
        key_variables=["home prices", "income", "interest rates"],
        constraints=["25 pages"],
        data_provided=["Price data 2010-2023"],
        success_criteria=["Accurate predictions"],
    )

    assumptions = AssumptionSet(
        assumptions=[
            JustifiedAssumption(
                assumption="Housing prices follow a modified logistic growth pattern",
                justification="Historical data shows S-curve behavior in mature markets",
                category="model",
            ),
            JustifiedAssumption(
                assumption="Income grows at a constant rate within the forecast period",
                justification="Median income trends are relatively stable year-over-year",
                category="data",
            ),
            JustifiedAssumption(
                assumption="Interest rates vary within historical bounds (3-8%)",
                justification="Federal Reserve maintains rates within historical norms",
                category="scope",
            ),
        ],
        problem_context_summary="Housing affordability prediction for metropolitan area",
    )

    result = mathematician.run(problem_context=context, assumptions=assumptions)

    # Validate output
    assert result.framework, "No framework specified"
    assert result.equations, "No equations generated"
    assert result.variables, "No variables defined"

    print(f"  Framework: {result.framework}")
    print(f"  Equations: {len(result.equations)}")
    print(f"  Variables: {list(result.variables.keys())}")

    return result


def test_coder(env_path: Path):
    """Test Coder with Claude Sonnet 4."""
    from src.agents.coder import Coder
    from src.agents.models import ProblemContext, MathModel

    coder = Coder(env_path=env_path)

    context = ProblemContext(
        raw_text=SAMPLE_PROBLEM,
        questions=["Predict housing affordability for next 10 years"],
        scope="Housing affordability prediction",
        real_problem="Price to income ratio trends",
        key_variables=["price", "income"],
        constraints=[],
        data_provided=[],
        success_criteria=[],
    )

    model = MathModel(
        framework="Exponential Growth Model",
        equations=[
            r"P(t) = P_0 \cdot e^{r_p \cdot t}",
            r"I(t) = I_0 \cdot e^{r_i \cdot t}",
            r"A(t) = \frac{P(t)}{I(t)}",
        ],
        variables={
            "P": "Median home price",
            "I": "Median household income",
            "A": "Affordability ratio",
            "t": "Time in years",
        },
        parameters={
            "P_0": "Initial home price (baseline year)",
            "I_0": "Initial household income",
            "r_p": "Annual price growth rate",
            "r_i": "Annual income growth rate",
        },
        rationale="Exponential growth captures compound effects of year-over-year changes",
        boundary_conditions=["P(0) = P_0", "I(0) = I_0"],
        assumptions_used=["Price growth is exponential", "Income growth is constant"],
    )

    # Test without execution to avoid environment issues
    result = coder.run(math_model=model, problem_context=context, execute=False)

    # Validate output
    assert result.code, "No code generated"
    assert "import" in result.code, "Code missing imports"
    assert len(result.code) > 100, "Code seems too short"

    print(f"  Generated {len(result.code.split(chr(10)))} lines of code")
    print(f"  Dependencies: {result.dependencies}")

    return result


def test_stress_tester(env_path: Path):
    """Test StressTester with GPT-4o."""
    from src.agents.stress_tester import StressTester
    from src.agents.models import ProblemContext, MathModel, Implementation

    tester = StressTester(env_path=env_path)

    context = ProblemContext(
        raw_text=SAMPLE_PROBLEM,
        questions=["Predict housing affordability"],
        scope="Housing affordability prediction",
        real_problem="Price to income trends",
        key_variables=["price", "income"],
        constraints=[],
        data_provided=[],
        success_criteria=[],
    )

    model = MathModel(
        framework="Exponential Growth",
        equations=["P(t) = P_0 * exp(r_p * t)"],
        variables={"P": "Price", "t": "Time"},
        parameters={"r_p": "Growth rate 3-5%", "P_0": "Initial price $300k"},
        rationale="Captures compound growth",
        boundary_conditions=[],
        assumptions_used=[],
    )

    impl = Implementation(
        code="# Sample implementation\nprint('Affordability: 5.2')",
        execution_log="Affordability: 5.2\nForecast complete.",
        success=True,
    )

    result = tester.run(
        math_model=model,
        implementation=impl,
        problem_context=context,
    )

    # Validate output
    assert result.stability_assessment, "No stability assessment"
    assert result.critical_parameters, "No critical parameters identified"

    print(f"  Stability: {result.stability_assessment}")
    print(f"  Robustness: {result.robustness_score:.2f}")
    print(f"  Critical params: {result.critical_parameters}")

    return result


def test_orchestrator(env_path: Path):
    """Test Orchestrator with Kimi 128k."""
    from src.agents.orchestrator import Orchestrator
    from src.agents.models import AgentState, JudgeVerdict, ProblemContext

    orchestrator = Orchestrator(env_path=env_path)

    # Create minimal state
    state = AgentState(
        problem_context=ProblemContext(
            raw_text=SAMPLE_PROBLEM,
            questions=["Q1", "Q2", "Q3"],
            scope="Housing affordability",
            real_problem="Price trends",
            key_variables=[],
            constraints=[],
            data_provided=[],
            success_criteria=[],
        ),
        iteration_history=[],
    )

    verdicts = [
        JudgeVerdict(
            judge_id="accuracy",
            score=7.5,
            feedback="Good mathematical foundation but some issues with convergence",
            strengths=["Clear equations", "Good parameter definitions"],
            weaknesses=["Missing boundary check", "Numerical instability"],
            suggestions=["Add convergence tests"],
        ),
        JudgeVerdict(
            judge_id="clarity",
            score=8.0,
            feedback="Well-organized with clear explanations",
            strengths=["Good structure", "Clear visualizations"],
            weaknesses=["Could use more comments"],
            suggestions=["Add summary section"],
        ),
        JudgeVerdict(
            judge_id="creativity",
            score=7.0,
            feedback="Standard approach with some novel elements",
            strengths=["Good data usage"],
            weaknesses=["Could be more innovative"],
            suggestions=["Consider alternative models"],
        ),
    ]

    result = orchestrator.run(state=state, judge_verdicts=verdicts)

    # Validate output
    assert result.consensus_score > 0, "No consensus score"
    assert result.final_feedback, "No final feedback"

    print(f"  Consensus score: {result.consensus_score:.1f}")
    print(f"  Should iterate: {result.should_iterate}")
    print(f"  Target: {result.iteration_target}")

    return result


def test_accuracy_judge(env_path: Path):
    """Test AccuracyJudge with GPT-4o."""
    from src.agents.judges import AccuracyJudge
    from src.agents.models import ProblemContext, MathModel, Implementation, SensitivityReport

    judge = AccuracyJudge(env_path=env_path)

    context = ProblemContext(
        raw_text=SAMPLE_PROBLEM,
        questions=["Predict housing affordability"],
        scope="Housing affordability prediction",
        real_problem="Price to income trends",
        key_variables=["price", "income"],
        constraints=[],
        data_provided=[],
        success_criteria=[],
    )

    model = MathModel(
        framework="Exponential Growth",
        equations=["P(t) = P_0 * exp(r * t)"],
        variables={"P": "Price", "t": "Time"},
        parameters={"r": "Growth rate"},
        rationale="Captures compound growth",
        boundary_conditions=[],
        assumptions_used=[],
    )

    impl = Implementation(
        code="# Sample",
        execution_log="Result: 5.2",
        success=True,
    )

    sensitivity = SensitivityReport(
        perturbations={"r": {"change": "10%", "impact": "15%"}},
        stability_assessment="Moderately Sensitive",
        critical_parameters=["r"],
        robustness_score=0.7,
        strengths=["Clear model"],
        weaknesses=["Parameter sensitive"],
        recommendations=["Add bounds"],
    )

    result = judge.run(
        problem_context=context,
        math_model=model,
        implementation=impl,
        sensitivity_report=sensitivity,
    )

    # Validate output
    assert 0 <= result.score <= 10, f"Score out of range: {result.score}"
    assert result.feedback, "No feedback"

    print(f"  Score: {result.score}")
    print(f"  Strengths: {result.strengths[:2]}")

    return result


def test_clarity_judge(env_path: Path):
    """Test ClarityJudge with Claude Sonnet 4."""
    from src.agents.judges import ClarityJudge
    from src.agents.models import (
        ProblemContext, AssumptionSet, JustifiedAssumption,
        MathModel, Implementation, SensitivityReport
    )

    judge = ClarityJudge(env_path=env_path)

    context = ProblemContext(
        raw_text=SAMPLE_PROBLEM,
        questions=["Predict housing affordability"],
        scope="Housing affordability prediction",
        real_problem="Price to income trends",
        key_variables=["price", "income"],
        constraints=[],
        data_provided=[],
        success_criteria=[],
    )

    assumptions = AssumptionSet(
        assumptions=[
            JustifiedAssumption(
                assumption="Growth is exponential",
                justification="Historical pattern",
                category="model",
            )
        ],
        problem_context_summary="Housing",
    )

    model = MathModel(
        framework="Exponential Growth",
        equations=["P(t) = P_0 * exp(r * t)"],
        variables={"P": "Price"},
        parameters={"r": "Rate"},
        rationale="Simple model",
        boundary_conditions=[],
        assumptions_used=[],
    )

    impl = Implementation(
        code="import numpy as np\n# Implementation",
        execution_log="Done",
        success=True,
    )

    sensitivity = SensitivityReport(
        perturbations={},
        stability_assessment="Stable",
        critical_parameters=[],
        robustness_score=0.8,
        strengths=["Clear"],
        weaknesses=[],
        recommendations=[],
    )

    result = judge.run(
        problem_context=context,
        assumptions=assumptions,
        math_model=model,
        implementation=impl,
        sensitivity_report=sensitivity,
    )

    assert 0 <= result.score <= 10
    print(f"  Score: {result.score}")

    return result


def test_creativity_judge(env_path: Path):
    """Test CreativityJudge with Gemini 2.0 Flash."""
    from src.agents.judges import CreativityJudge
    from src.agents.models import (
        ProblemContext, AssumptionSet, JustifiedAssumption,
        MathModel, SensitivityReport
    )

    judge = CreativityJudge(env_path=env_path)

    context = ProblemContext(
        raw_text=SAMPLE_PROBLEM,
        questions=["Predict housing affordability"],
        scope="Housing affordability prediction",
        real_problem="Price to income trends",
        key_variables=["price", "income"],
        constraints=[],
        data_provided=[],
        success_criteria=[],
    )

    assumptions = AssumptionSet(
        assumptions=[
            JustifiedAssumption(
                assumption="Novel approach using machine learning",
                justification="Captures non-linear patterns",
                category="model",
            )
        ],
        problem_context_summary="Housing",
    )

    model = MathModel(
        framework="Hybrid ML-ODE Model",
        equations=["dP/dt = f_nn(P, I, r)"],
        variables={"P": "Price"},
        parameters={"theta": "NN weights"},
        rationale="Combines interpretability with flexibility",
        boundary_conditions=[],
        assumptions_used=[],
    )

    sensitivity = SensitivityReport(
        perturbations={},
        stability_assessment="Stable",
        critical_parameters=[],
        robustness_score=0.8,
        strengths=["Innovative"],
        weaknesses=[],
        recommendations=["Try ensemble"],
    )

    result = judge.run(
        problem_context=context,
        assumptions=assumptions,
        math_model=model,
        sensitivity_report=sensitivity,
    )

    assert 0 <= result.score <= 10
    print(f"  Score: {result.score}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Live API tests for M3 agents")
    parser.add_argument(
        "--agent",
        type=str,
        help="Test only a specific agent",
        choices=[
            "input_handler", "scout", "historian", "mathematician",
            "coder", "stress_tester", "orchestrator",
            "accuracy_judge", "clarity_judge", "creativity_judge",
        ],
    )
    parser.add_argument(
        "--skip-expensive",
        action="store_true",
        help="Skip expensive tests (o1 model)",
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip tests that require RAG (historian)",
    )

    args = parser.parse_args()

    # Setup paths
    env_path = project_root / "keyholder.env"
    chroma_path = project_root / "chroma_db"

    if not env_path.exists():
        print(f"Error: Environment file not found: {env_path}")
        sys.exit(1)

    runner = TestRunner(env_path=env_path, chroma_path=chroma_path)

    print("="*60)
    print("M3 Agent Live API Tests")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Define all tests
    tests = [
        ("InputHandler (Gemini 2.0 Flash)", "input_handler",
         lambda: test_input_handler(env_path), False, ""),

        ("Scout (Claude Sonnet 4)", "scout",
         lambda: test_scout(env_path), False, ""),

        ("Historian (GPT-4o + RAG)", "historian",
         lambda: test_historian(env_path, chroma_path),
         args.skip_rag or not chroma_path.exists(),
         "RAG skipped or ChromaDB not found"),

        ("Mathematician (o1)", "mathematician",
         lambda: test_mathematician(env_path),
         args.skip_expensive, "Skipped expensive o1 test"),

        ("Coder (Claude Sonnet 4)", "coder",
         lambda: test_coder(env_path), False, ""),

        ("StressTester (GPT-4o)", "stress_tester",
         lambda: test_stress_tester(env_path), False, ""),

        ("Orchestrator (Kimi 128k)", "orchestrator",
         lambda: test_orchestrator(env_path), False, ""),

        ("AccuracyJudge (GPT-4o)", "accuracy_judge",
         lambda: test_accuracy_judge(env_path), False, ""),

        ("ClarityJudge (Claude Sonnet 4)", "clarity_judge",
         lambda: test_clarity_judge(env_path), False, ""),

        ("CreativityJudge (Gemini 2.0 Flash)", "creativity_judge",
         lambda: test_creativity_judge(env_path), False, ""),
    ]

    # Filter to single agent if specified
    if args.agent:
        tests = [(name, key, func, skip, reason)
                 for name, key, func, skip, reason in tests
                 if key == args.agent]

    # Run tests
    for name, key, test_func, skip, reason in tests:
        runner.run_test(name, test_func, skip=skip, skip_reason=reason)

    # Print summary
    success = runner.print_summary()

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
