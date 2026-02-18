#!/usr/bin/env python3
"""
Test agent system components.

Usage:
    python scripts/test_agents.py

This script tests individual agents and the full pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    from src.agents import (
        BaseAgent,
        ProblemContext,
        JustifiedAssumption,
        AssumptionSet,
        MathModel,
        Implementation,
        SensitivityReport,
        JudgeVerdict,
        OrchestratorDecision,
        AgentState,
        InputHandler,
        Scout,
        Historian,
        Mathematician,
        Coder,
        StressTester,
        Orchestrator,
        AccuracyJudge,
        ClarityJudge,
        CreativityJudge,
    )

    from src.pipeline import M3Workflow, M3Runner

    print("  All imports successful!")
    return True


def test_models():
    """Test Pydantic model creation."""
    print("Testing models...")

    from src.agents.models import (
        ProblemContext,
        JustifiedAssumption,
        AssumptionSet,
        MathModel,
        AgentState,
    )

    # Test ProblemContext
    context = ProblemContext(
        raw_text="Test problem text",
        questions=["Q1: What is X?", "Q2: How does Y work?"],
        scope="Determine X and Y",
        real_problem="Understanding the relationship between X and Y",
    )
    assert context.questions[0] == "Q1: What is X?"
    print("  ProblemContext: OK")

    # Test JustifiedAssumption
    assumption = JustifiedAssumption(
        assumption="X is constant",
        justification="Based on historical data",
        category="model",
    )
    assert assumption.category == "model"
    print("  JustifiedAssumption: OK")

    # Test AssumptionSet
    assumption_set = AssumptionSet(
        assumptions=[assumption],
        problem_context_summary="Test summary",
    )
    assert assumption_set.total_assumptions == 1
    print("  AssumptionSet: OK")

    # Test MathModel
    model = MathModel(
        framework="Logistic Growth",
        equations=["dP/dt = rP(1 - P/K)"],
        variables={"P": "Population", "t": "Time"},
        parameters={"r": "Growth rate", "K": "Carrying capacity"},
        rationale="Logistic growth is appropriate for bounded population growth",
    )
    assert model.framework == "Logistic Growth"
    print("  MathModel: OK")

    # Test AgentState
    state = AgentState(
        problem_context=context,
        current_phase="testing",
    )
    assert state.current_phase == "testing"
    print("  AgentState: OK")

    print("  All models valid!")
    return True


def test_agent_initialization():
    """Test that agents can be initialized."""
    print("Testing agent initialization...")

    env_path = project_root / "keyholder.env"
    load_dotenv(env_path)

    from src.agents import InputHandler, Scout

    # Test InputHandler (uses Gemini)
    try:
        handler = InputHandler(env_path=env_path)
        print(f"  InputHandler: OK ({handler.provider}/{handler.model})")
    except Exception as e:
        print(f"  InputHandler: FAILED - {e}")
        return False

    # Test Scout (uses Claude)
    try:
        scout = Scout(env_path=env_path)
        print(f"  Scout: OK ({scout.provider}/{scout.model})")
    except Exception as e:
        print(f"  Scout: FAILED - {e}")
        return False

    print("  Agent initialization successful!")
    return True


def test_rag_integration():
    """Test that Historian can connect to RAG."""
    print("Testing RAG integration...")

    env_path = project_root / "keyholder.env"
    chroma_path = project_root / "chroma_db"

    load_dotenv(env_path)

    if not chroma_path.exists():
        print("  SKIPPED: ChromaDB not found. Run ingest_data.py first.")
        return True

    from src.agents import Historian

    try:
        historian = Historian(
            chroma_path=chroma_path,
            env_path=env_path,
        )

        # Test RAG query
        results = historian.search_similar_problems(
            "housing growth prediction",
            n_results=3,
        )

        print(f"  Historian: OK (found {len(results)} similar problems)")
        return True

    except Exception as e:
        print(f"  Historian: FAILED - {e}")
        return False


def main():
    print("=" * 50)
    print("M3 Agent System Tests")
    print("=" * 50)
    print()

    tests = [
        ("Imports", test_imports),
        ("Models", test_models),
        ("Agent Initialization", test_agent_initialization),
        ("RAG Integration", test_rag_integration),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print()
    print(f"Total: {passed}/{total} passed")

    if passed == total:
        print("\nAll tests passed! The agent system is ready.")
    else:
        print("\nSome tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
