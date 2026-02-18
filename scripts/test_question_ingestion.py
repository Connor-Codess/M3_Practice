#!/usr/bin/env python3
"""
Test script for the question ingestion module.

Tests:
1. Data loading from CSV files
2. Question parsing from text/PDF
3. Context building with data integration
4. Question store functionality

Usage:
    python scripts/test_question_ingestion.py
    python scripts/test_question_ingestion.py --create-sample-data
"""

import sys
import tempfile
import argparse
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


SAMPLE_PROBLEM_TEXT = """
M3 Challenge 2024 - Renewable Energy Transition

Background:
As countries work to reduce carbon emissions, the transition to renewable
energy sources has become increasingly important. This transition involves
complex trade-offs between economic costs, reliability, and environmental
benefits.

The data provided includes historical energy production by source, costs
of different generation technologies, and carbon emissions data.

Q1: Develop a model to predict the optimal mix of energy sources for a
state over the next 20 years, considering cost, reliability, and emissions.

Q2: Analyze how sensitive your model is to changes in technology costs,
policy incentives, and energy demand growth.

Q3: Recommend a phased transition plan that balances economic and
environmental goals. Consider political and social feasibility.

Constraints:
- Maximum 25 pages
- Must include sensitivity analysis
- All assumptions must be justified
- Include at least two visualizations
"""


def create_sample_csv(output_dir: Path) -> list[Path]:
    """Create sample CSV files for testing."""
    import csv

    files = []

    # Energy production data
    energy_path = output_dir / "energy_production.csv"
    with open(energy_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "state", "source", "production_gwh", "cost_per_mwh"])
        for year in range(2015, 2024):
            for source in ["coal", "natural_gas", "solar", "wind", "nuclear"]:
                writer.writerow([
                    year,
                    "California",
                    source,
                    1000 + year * 10 + (50 if source == "solar" else 0),
                    45 + (year - 2015) * 2 if source in ["solar", "wind"] else 60,
                ])
    files.append(energy_path)

    # Emissions data
    emissions_path = output_dir / "emissions_data.csv"
    with open(emissions_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "source", "co2_tons_per_gwh", "other_emissions"])
        for year in range(2015, 2024):
            for source, co2 in [("coal", 900), ("natural_gas", 400), ("solar", 20), ("wind", 10), ("nuclear", 15)]:
                writer.writerow([year, source, co2, co2 * 0.1])
    files.append(emissions_path)

    return files


def test_data_loader():
    """Test the DataLoader class."""
    print("\n[Test: DataLoader]")

    from src.question_ingestion import DataLoader

    loader = DataLoader()

    with tempfile.TemporaryDirectory() as tmpdir:
        files = create_sample_csv(Path(tmpdir))

        # Test loading files
        summary = loader.generate_summary(files)

        print(f"  Files loaded: {len(summary.files)}")
        print(f"  Total rows: {summary.total_rows}")
        print(f"  Columns: {len(summary.columns)}")
        print(f"  Missing data: {summary.missing_data_pct:.1f}%")
        print(f"  Date range: {summary.date_range}")

        # Verify columns
        assert len(summary.columns) > 0, "No columns found"
        assert summary.total_rows > 0, "No rows loaded"

        # Check numeric stats
        for col_name, col_info in summary.columns.items():
            if col_info.dtype == "numeric" and col_info.stats:
                print(f"  Column '{col_name}': mean={col_info.stats['mean']:.2f}, std={col_info.stats['std']:.2f}")

        print("  [PASS] DataLoader")
        return True


def test_question_parser():
    """Test the QuestionParser class."""
    print("\n[Test: QuestionParser]")

    from src.question_ingestion import QuestionParser

    parser = QuestionParser()

    # Test text parsing
    parsed = parser.parse_text(SAMPLE_PROBLEM_TEXT)

    print(f"  Title: {parsed.title}")
    print(f"  Questions: {len(parsed.questions)}")
    print(f"  Data references: {len(parsed.data_references)}")
    print(f"  Constraints: {len(parsed.constraints)}")

    # Verify parsing
    assert parsed.questions, "No questions parsed"
    assert len(parsed.questions) >= 2, f"Expected at least 2 questions, got {len(parsed.questions)}"
    assert parsed.background, "No background extracted"

    for i, q in enumerate(parsed.questions[:3], 1):
        print(f"  Q{i}: {q[:60]}...")

    print("  [PASS] QuestionParser")
    return True


def test_context_builder():
    """Test the ContextBuilder class."""
    print("\n[Test: ContextBuilder]")

    from src.question_ingestion import ContextBuilder

    builder = ContextBuilder()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample data
        files = create_sample_csv(tmpdir)

        # Create sample problem file
        problem_path = tmpdir / "problem.txt"
        problem_path.write_text(SAMPLE_PROBLEM_TEXT)

        # Build context
        context = builder.build_from_files(
            problem_path=problem_path,
            data_paths=files,
        )

        print(f"  Questions: {len(context.questions)}")
        print(f"  Key variables: {len(context.key_variables)}")
        print(f"  Has data summary: {context.data_summary is not None}")

        if context.data_summary:
            print(f"  Data files: {context.data_summary.files}")
            print(f"  Total rows: {context.data_summary.total_rows}")

        # Verify context
        assert context.raw_text, "No raw text"
        assert context.data_summary, "No data summary"
        assert context.data_context_string, "No data context string"

        # Test context string
        ctx_str = context.data_context_string
        print(f"  Context string length: {len(ctx_str)} chars")
        print(f"  Context string preview: {ctx_str[:100]}...")

        print("  [PASS] ContextBuilder")
        return True


def test_question_store():
    """Test the QuestionStore class."""
    print("\n[Test: QuestionStore]")

    from src.question_ingestion import QuestionStore
    from src.agents.models import ProblemContext, DataSummary

    # Clear any existing state
    QuestionStore.clear()

    # Create test context
    context = ProblemContext(
        raw_text=SAMPLE_PROBLEM_TEXT,
        questions=["Q1: First question", "Q2: Second question"],
        scope="Test scope",
        real_problem="Test problem",
        key_variables=["var1", "var2"],
        constraints=["25 pages"],
        data_provided=["data.csv"],
        success_criteria=["accuracy"],
        data_summary=DataSummary(
            files=["test.csv"],
            total_rows=100,
            columns={},
            missing_data_pct=5.0,
        ),
        data_context_string="Test data context",
    )

    # Set and retrieve
    QuestionStore.set_context(context)

    assert QuestionStore.has_context(), "Context not stored"
    assert QuestionStore.has_data(), "Data not detected"
    assert QuestionStore.get_scope() == "Test scope", "Wrong scope"
    assert len(QuestionStore.get_questions()) == 2, "Wrong question count"
    assert QuestionStore.get_data_context() == "Test data context", "Wrong data context"

    print(f"  Has context: {QuestionStore.has_context()}")
    print(f"  Has data: {QuestionStore.has_data()}")
    print(f"  Scope: {QuestionStore.get_scope()}")

    # Clear and verify
    QuestionStore.clear()
    assert not QuestionStore.has_context(), "Context not cleared"

    print("  [PASS] QuestionStore")
    return True


def test_integration():
    """Test full integration with sample files."""
    print("\n[Test: Integration]")

    from src.question_ingestion import ContextBuilder, QuestionStore

    builder = ContextBuilder()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        files = create_sample_csv(tmpdir)

        # Build context from text
        context = builder.build_from_text(
            problem_text=SAMPLE_PROBLEM_TEXT,
            data_paths=files,
        )

        # Store it
        QuestionStore.set_context(context)
        QuestionStore.set_data_files(files)

        # Verify integration
        assert QuestionStore.has_context()
        assert QuestionStore.has_data()
        assert len(QuestionStore.get_data_files()) == len(files)

        # Get data summary
        summary = QuestionStore.get_data_summary()
        assert summary is not None
        assert summary.total_rows > 0

        # Clean up
        QuestionStore.clear()

    print("  [PASS] Integration")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test question ingestion module")
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample data files in outputs/sample_data/",
    )

    args = parser.parse_args()

    if args.create_sample_data:
        output_dir = project_root / "outputs" / "sample_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        files = create_sample_csv(output_dir)
        print(f"Created sample data files in: {output_dir}")
        for f in files:
            print(f"  - {f.name}")

        # Also save sample problem
        problem_path = output_dir / "sample_problem.txt"
        problem_path.write_text(SAMPLE_PROBLEM_TEXT)
        print(f"  - {problem_path.name}")
        return

    print("="*60)
    print("M3 Question Ingestion Tests")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    tests = [
        ("DataLoader", test_data_loader),
        ("QuestionParser", test_question_parser),
        ("ContextBuilder", test_context_builder),
        ("QuestionStore", test_question_store),
        ("Integration", test_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for _, s in results if s)
    total = len(results)

    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")

    print()
    print(f"Total: {passed}/{total} passed")

    if passed == total:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
