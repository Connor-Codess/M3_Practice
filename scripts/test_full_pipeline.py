#!/usr/bin/env python3
"""
End-to-end pipeline test for M3 Challenge system.

Runs the complete LangGraph workflow on a sample problem to verify:
1. All agents execute correctly
2. State passes properly between nodes
3. Iteration logic works
4. Output generation succeeds

Usage:
    python scripts/test_full_pipeline.py
    python scripts/test_full_pipeline.py --output outputs/test_run
    python scripts/test_full_pipeline.py --max-iterations 1  # Quick test
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv


SAMPLE_PROBLEM = """
M3 Challenge 2024 - Electric Vehicle Adoption

Background:
Electric vehicles (EVs) are becoming increasingly popular as concerns about
climate change grow and battery technology improves. However, adoption rates
vary significantly by region, income level, and infrastructure availability.

Data Provided:
- Table 1: EV sales by state (2015-2023)
- Table 2: Charging station density by state
- Table 3: Average household income by state
- Table 4: Electricity prices by state

Q1: Develop a mathematical model to predict EV adoption rates over the next
decade for a state of your choice. Your model should account for economic
factors, infrastructure, and technological improvements.

Q2: Analyze how changes in key parameters (such as battery cost, charging
infrastructure, and gas prices) affect your adoption predictions. Which
factors are most influential?

Q3: Recommend policies that a state government could implement to accelerate
EV adoption while considering budget constraints and equity concerns.

Constraints:
- Maximum 25 pages
- All assumptions must be clearly stated and justified
- Include sensitivity analysis
- Provide at least two visualizations
- Consider environmental and economic impacts
"""


def run_pipeline_test(
    env_path: Path,
    chroma_path: Path,
    output_dir: Optional[Path],
    max_iterations: int = 3,
    verbose: bool = True,
):
    """Run the full pipeline and return results."""
    from src.pipeline import M3Runner

    print("="*60)
    print("M3 Full Pipeline Test")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Create runner
    runner = M3Runner(
        env_path=env_path,
        chroma_path=chroma_path,
        verbose=verbose,
    )

    # Override max iterations for testing
    runner.workflow.orchestrator.MAX_ITERATIONS = max_iterations

    print(f"\nProblem length: {len(SAMPLE_PROBLEM)} characters")
    print(f"Max iterations: {max_iterations}")
    print()

    # Track timing
    start_time = time.time()
    phase_times = {}

    try:
        # Run the pipeline
        final_state = runner.run(
            problem_text=SAMPLE_PROBLEM,
            output_dir=output_dir,
        )

        total_time = time.time() - start_time

        # Print results
        print("\n" + "="*60)
        print("PIPELINE RESULTS")
        print("="*60)

        # Check each component
        results = validate_state(final_state)

        for component, (status, message) in results.items():
            icon = "[OK]" if status else "[FAIL]"
            print(f"  {icon} {component}: {message}")

        # Print scores
        if final_state.orchestrator_decision:
            decision = final_state.orchestrator_decision
            print(f"\n  Final Score: {decision.consensus_score:.1f}/10")
            print(f"  Iterations: {decision.iteration_count}")
            print(f"  Individual Scores:")
            for judge_id, score in decision.individual_scores.items():
                print(f"    - {judge_id}: {score:.1f}")

        # Print timing
        print(f"\n  Total Time: {total_time:.1f}s")

        # Save detailed results
        if output_dir:
            save_detailed_results(final_state, output_dir, total_time)

        # Check if all passed
        all_passed = all(status for status, _ in results.values())

        print("\n" + "="*60)
        if all_passed:
            print("Pipeline test PASSED")
        else:
            print("Pipeline test FAILED - see details above")
        print("="*60)

        return final_state, all_passed

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def validate_state(state) -> dict:
    """Validate the final state has all required components."""
    results = {}

    # Check problem context
    if state.problem_context:
        pc = state.problem_context
        if pc.questions and pc.scope and pc.real_problem:
            results["Problem Context"] = (True, f"{len(pc.questions)} questions parsed")
        else:
            results["Problem Context"] = (False, "Missing questions/scope/real_problem")
    else:
        results["Problem Context"] = (False, "Not set")

    # Check assumptions
    if state.assumptions:
        count = len(state.assumptions.assumptions)
        if count >= 3:
            results["Assumptions"] = (True, f"{count} assumptions generated")
        else:
            results["Assumptions"] = (False, f"Only {count} assumptions (expected 3+)")
    else:
        results["Assumptions"] = (False, "Not set")

    # Check math model
    if state.math_model:
        mm = state.math_model
        if mm.framework and mm.equations:
            results["Math Model"] = (True, f"{mm.framework} with {len(mm.equations)} equations")
        else:
            results["Math Model"] = (False, "Missing framework or equations")
    else:
        results["Math Model"] = (False, "Not set")

    # Check implementation
    if state.implementation:
        impl = state.implementation
        if impl.code:
            lines = len(impl.code.split("\n"))
            status = "executed" if impl.success else f"failed: {impl.error_message[:50]}"
            results["Implementation"] = (True, f"{lines} lines, {status}")
        else:
            results["Implementation"] = (False, "No code generated")
    else:
        results["Implementation"] = (False, "Not set")

    # Check sensitivity report
    if state.sensitivity_report:
        sr = state.sensitivity_report
        if sr.stability_assessment:
            results["Sensitivity"] = (True, f"{sr.stability_assessment}, robustness={sr.robustness_score:.2f}")
        else:
            results["Sensitivity"] = (False, "Missing stability assessment")
    else:
        results["Sensitivity"] = (False, "Not set")

    # Check judge verdicts
    if state.judge_verdicts:
        count = len(state.judge_verdicts)
        if count == 3:
            avg = sum(v.score for v in state.judge_verdicts) / count
            results["Judge Verdicts"] = (True, f"{count} judges, avg score={avg:.1f}")
        else:
            results["Judge Verdicts"] = (False, f"Only {count} verdicts (expected 3)")
    else:
        results["Judge Verdicts"] = (False, "Not set")

    # Check orchestrator decision
    if state.orchestrator_decision:
        od = state.orchestrator_decision
        results["Orchestrator"] = (True, f"score={od.consensus_score:.1f}, iterate={od.should_iterate}")
    else:
        results["Orchestrator"] = (False, "Not set")

    return results


def save_detailed_results(state, output_dir: Path, total_time: float):
    """Save detailed JSON results for analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary dict
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "final_score": state.orchestrator_decision.consensus_score if state.orchestrator_decision else None,
        "iterations": state.orchestrator_decision.iteration_count if state.orchestrator_decision else 0,
        "individual_scores": state.orchestrator_decision.individual_scores if state.orchestrator_decision else {},
        "problem_scope": state.problem_context.scope if state.problem_context else None,
        "model_framework": state.math_model.framework if state.math_model else None,
        "assumption_count": len(state.assumptions.assumptions) if state.assumptions else 0,
        "code_lines": len(state.implementation.code.split("\n")) if state.implementation else 0,
        "code_success": state.implementation.success if state.implementation else False,
        "stability": state.sensitivity_report.stability_assessment if state.sensitivity_report else None,
        "robustness": state.sensitivity_report.robustness_score if state.sensitivity_report else None,
    }

    # Save summary JSON
    summary_path = output_dir / "test_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Detailed results saved to: {output_dir}")


def run_quick_validation(env_path: Path, chroma_path: Path):
    """Quick validation that components can be imported and initialized."""
    print("Running quick validation...")

    try:
        from src.pipeline import M3Workflow

        workflow = M3Workflow(
            env_path=env_path,
            chroma_path=chroma_path,
        )

        print("  [OK] Workflow initialized")
        print(f"  [OK] Graph compiled with {len(workflow.graph.nodes)} nodes")

        # Check all agents
        agents = [
            ("InputHandler", workflow.input_handler),
            ("Scout", workflow.scout),
            ("Historian", workflow.historian),
            ("Mathematician", workflow.mathematician),
            ("Coder", workflow.coder),
            ("StressTester", workflow.stress_tester),
            ("AccuracyJudge", workflow.accuracy_judge),
            ("ClarityJudge", workflow.clarity_judge),
            ("CreativityJudge", workflow.creativity_judge),
            ("Orchestrator", workflow.orchestrator),
        ]

        for name, agent in agents:
            print(f"  [OK] {name}: {agent.provider}/{agent.model}")

        return True

    except Exception as e:
        print(f"  [FAIL] Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run M3 pipeline end-to-end test")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Maximum iterations (default: 1 for quick test)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate imports and initialization",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Setup paths
    env_path = project_root / "keyholder.env"
    chroma_path = project_root / "chroma_db"

    if not env_path.exists():
        print(f"Error: Environment file not found: {env_path}")
        sys.exit(1)

    load_dotenv(env_path)

    # Set default output directory
    if args.output is None:
        args.output = project_root / "outputs" / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Quick validation only
    if args.validate_only:
        success = run_quick_validation(env_path, chroma_path)
        sys.exit(0 if success else 1)

    # Full pipeline test
    state, success = run_pipeline_test(
        env_path=env_path,
        chroma_path=chroma_path,
        output_dir=args.output,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
