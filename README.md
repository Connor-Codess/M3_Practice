# M3 RAG Multi-Agent System

A multi-agent AI system for solving M3 Math Modeling Challenge problems, powered by RAG (Retrieval-Augmented Generation) from 14 years of winning solutions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Current Status](#current-status)
3. [Architecture](#architecture)
4. [What's Been Built](#whats-been-built)
5. [What's Left to Build](#whats-left-to-build)
6. [File Structure](#file-structure)
7. [Setup & Installation](#setup--installation)
8. [Usage](#usage)
9. [API Keys Required](#api-keys-required)
10. [Technical Details](#technical-details)
11. [Development Notes](#development-notes)

---

## Project Overview

The M3 (MathWorks Math Modeling) Challenge is a 14-hour high school math modeling competition. This system uses multiple specialized AI agents to:

1. **Parse** competition problems
2. **Research** historical winning solutions via RAG
3. **Generate** justified assumptions grounded in past winners
4. **Develop** mathematical models
5. **Implement** solutions in Python
6. **Evaluate** using a multi-LLM judge panel
7. **Iterate** until quality threshold is met

### Goal
Automate the creation of competition-quality M3 solutions by learning from 42 past winning papers (2011-2024).

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| RAG System | COMPLETE | 1,002 chunks indexed in ChromaDB |
| Data Processing | COMPLETE | Schema normalization, semantic chunking |
| Embeddings | COMPLETE | OpenAI text-embedding-3-large (3072 dims) |
| Agent Framework | COMPLETE | 10 agents + 3 judges implemented |
| LangGraph Pipeline | COMPLETE | State machine with iteration loop |
| Pipeline Runner | COMPLETE | CLI and programmatic interface |
| Question + Data Ingestion | COMPLETE | CSV/Excel loading, stats generation |
| Live Agent Testing | COMPLETE | Test scripts with real API calls |
| End-to-End Testing | READY | Test scripts created, needs execution |
| Report Generation | NOT STARTED | LaTeX/PDF output |

**Overall Progress: ~80% complete**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR (Kimi 128k)                        │
│                   Coordinates flow, handles iterations                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌─────────────┐           ┌─────────────────┐           ┌─────────────────┐
│INPUT HANDLER│           │     SCOUT       │           │   HISTORIAN     │
│ Parse Text  │──────────▶│ Problem Analysis│──────────▶│ Assumptions+RAG │
│Gemini Flash │           │ Claude Sonnet   │           │    GPT-4o       │
└─────────────┘           └─────────────────┘           └────────┬────────┘
                                                                 │
                                                        ┌────────▼────────┐
                                                        │  MATHEMATICIAN  │
                                                        │ Model Dev (o1)  │
                                                        └────────┬────────┘
                                                                 │
                                                        ┌────────▼────────┐
                                                        │     CODER       │
                                                        │ Implementation  │
                                                        │ Claude Sonnet   │
                                                        └────────┬────────┘
                                                                 │
                                                        ┌────────▼────────┐
                                                        │  STRESS-TESTER  │
                                                        │  Sensitivity    │
                                                        │    GPT-4o       │
                                                        └────────┬────────┘
                                                                 │
                          ┌──────────────────────────────────────┘
                          ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                           JUDGES (Parallel)                          │
    ├─────────────────┬─────────────────────┬─────────────────────────────┤
    │ ACCURACY (GPT-4o)│ CLARITY (Claude)   │ CREATIVITY (Gemini)         │
    │ Math correctness │ Presentation       │ Innovation                   │
    └─────────────────┴─────────────────────┴─────────────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │  ORCHESTRATOR   │
                          │ Score >= 9? ────┼──▶ DONE
                          │ Else iterate    │
                          └────────┬────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
               historian    mathematician      coder
              (if assumptions) (if math)    (if clarity)
```

### RAG Integration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              ChromaDB                                    │
│                          1,002 indexed chunks                            │
├─────────────────────────────────────────────────────────────────────────┤
│ Assumptions (651) │ Models (125) │ Sensitivity (76) │ Comments (90)     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ Semantic + Metadata Search
                                    │
                          ┌─────────┴─────────┐
                          │   Query Engine    │
                          │ Hybrid Search     │
                          └───────────────────┘
                                    ▲
                                    │
                    ┌───────────────┴───────────────┐
                    │           HISTORIAN            │
                    │  "Find assumptions for         │
                    │   housing growth problems"     │
                    └────────────────────────────────┘
```

---

## What's Been Built

### Phase 1: RAG System (COMPLETE)

#### Data Processing (`src/data_processing/`)

| File | Purpose | Status |
|------|---------|--------|
| `schema_normalizer.py` | Normalizes JSON schema variations across years (2011-2024 have different formats) | COMPLETE |
| `content_extractor.py` | Extracts Q1/Q2/Q3 content blocks from normalized documents | COMPLETE |
| `semantic_chunker.py` | Creates typed chunks (assumption, model, sensitivity, comment) | COMPLETE |
| `__init__.py` | Exports `extract_document`, `chunk_document` | COMPLETE |

**Key Features:**
- Handles 2024 format: `{Assumption: str, Justification: str}`
- Handles 2011 format: plain string arrays
- Fixes malformed JSON (escape sequences, missing braces)
- Auto-tags chunks with topics and model types

#### Embeddings (`src/embeddings/`)

| File | Purpose | Status |
|------|---------|--------|
| `embedding_client.py` | OpenAI embedding wrapper with batch processing | COMPLETE |
| `__init__.py` | Exports `EmbeddingClient`, `get_embedding_client` | COMPLETE |

**Configuration:**
- Model: `text-embedding-3-large`
- Dimensions: 3072
- Batch size: 100 (rate-limited)

#### Vector Store (`src/vector_store/`)

| File | Purpose | Status |
|------|---------|--------|
| `chroma_client.py` | ChromaDB connection and operations | COMPLETE |
| `ingestor.py` | Batch ingestion with deduplication | COMPLETE |
| `__init__.py` | Exports clients and ingestion functions | COMPLETE |

**Database Stats:**
- Collection: `m3_rag_historical`
- Documents: 1,002 chunks
- Location: `chroma_db/`

#### Retrieval (`src/retrieval/`)

| File | Purpose | Status |
|------|---------|--------|
| `query_engine.py` | Hybrid search (semantic + metadata filtering) | COMPLETE |
| `__init__.py` | Exports `QueryEngine`, `get_query_engine` | COMPLETE |

**Query Methods:**
```python
engine.search(query, section_type=None, year_min=None, year_max=None, placement=None)
engine.search_assumptions(query, with_justification=True)
engine.search_models(query, model_type=None)
engine.get_similar_problems(problem_description)
```

#### Models (`src/models/`)

| File | Purpose | Status |
|------|---------|--------|
| `document.py` | `SourceDocument`, `Question`, `Assumption`, `Model` dataclasses | COMPLETE |
| `chunk.py` | `Chunk`, `ChunkType` for vector storage | COMPLETE |
| `results.py` | `AssumptionResult`, `RAGResponse` for query results | COMPLETE |
| `__init__.py` | Exports all models | COMPLETE |

---

### Phase 2: Agent System (COMPLETE)

#### Base Agent (`src/agents/base.py`)

| Feature | Status |
|---------|--------|
| Multi-provider LLM client (OpenAI, Anthropic, Google, Moonshot) | COMPLETE |
| Structured output generation with Pydantic parsing | COMPLETE |
| JSON response handling (strips markdown, finds JSON in text) | COMPLETE |
| Temperature and max_tokens configuration | COMPLETE |

#### Agent Data Models (`src/agents/models.py`)

| Model | Purpose | Status |
|-------|---------|--------|
| `ProblemContext` | Parsed problem with scope, constraints, questions | COMPLETE |
| `JustifiedAssumption` | Single assumption with justification and source | COMPLETE |
| `AssumptionSet` | Collection of assumptions for a solution | COMPLETE |
| `MathModel` | Framework, equations, variables, parameters | COMPLETE |
| `Implementation` | Code, outputs, visualizations, execution status | COMPLETE |
| `SensitivityReport` | Perturbations, stability, strengths/weaknesses | COMPLETE |
| `JudgeVerdict` | Score (0-10), feedback, suggestions | COMPLETE |
| `OrchestratorDecision` | Consensus score, iteration routing | COMPLETE |
| `AgentState` | Full pipeline state passed between nodes | COMPLETE |

#### Individual Agents (`src/agents/`)

| Agent | File | LLM | Purpose | Status |
|-------|------|-----|---------|--------|
| Input Handler | `input_handler.py` | Gemini 2.0 Flash | Parse problem text, extract Q1/Q2/Q3 | COMPLETE |
| Scout | `scout.py` | Claude Sonnet 4 | Deep problem analysis, identify real problem | COMPLETE |
| Historian | `historian.py` | GPT-4o | Generate justified assumptions using RAG | COMPLETE |
| Mathematician | `mathematician.py` | o1 | Develop mathematical model | COMPLETE |
| Coder | `coder.py` | Claude Sonnet 4 | Implement model in Python | COMPLETE |
| Stress-Tester | `stress_tester.py` | GPT-4o | Sensitivity analysis | COMPLETE |
| Orchestrator | `orchestrator.py` | Kimi 128k | Coordinate pipeline, handle iterations | COMPLETE |

#### Judges (`src/agents/judges/`)

| Judge | File | LLM | Evaluates | Status |
|-------|------|-----|-----------|--------|
| Accuracy | `judge_accuracy.py` | GPT-4o | Mathematical correctness, rigor | COMPLETE |
| Clarity | `judge_clarity.py` | Claude Sonnet 4 | Presentation, documentation | COMPLETE |
| Creativity | `judge_creativity.py` | Gemini 2.0 Flash | Innovation, insights | COMPLETE |

---

### Phase 3: Pipeline Orchestration (COMPLETE)

#### Workflow (`src/pipeline/workflow.py`)

| Feature | Status |
|---------|--------|
| LangGraph StateGraph definition | COMPLETE |
| Node functions for each agent | COMPLETE |
| Linear flow with conditional iteration | COMPLETE |
| Refinement methods for iteration loops | COMPLETE |

**Pipeline Flow:**
```
input_handler → scout → historian → mathematician → coder → stress_tester → judges → orchestrator
                                                                                          │
                          ←──────────────── (iterate if score < 9) ──────────────────────┘
```

#### Runner (`src/pipeline/runner.py`)

| Feature | Status |
|---------|--------|
| `M3Runner` class with progress logging | COMPLETE |
| `run(problem_text)` method | COMPLETE |
| `run_from_pdf(pdf_path)` method | COMPLETE |
| `run_with_data(problem_path, data_files)` method | COMPLETE |
| Output saving (code, summary, assumptions, model, sensitivity) | COMPLETE |
| CLI interface with argparse | COMPLETE |

---

### Phase 4: Question + Data Ingestion (COMPLETE)

#### Data Loader (`src/question_ingestion/data_loader.py`)

| Feature | Status |
|---------|--------|
| CSV/Excel file loading | COMPLETE |
| Column type detection (numeric, categorical, datetime) | COMPLETE |
| Automatic statistics generation (mean, std, min, max) | COMPLETE |
| Date range detection for time series | COMPLETE |
| Geographic scope detection | COMPLETE |
| Data quality issue detection | COMPLETE |

#### Question Parser (`src/question_ingestion/question_parser.py`)

| Feature | Status |
|---------|--------|
| PDF text extraction | COMPLETE |
| Question extraction (Q1, Q2, Q3) | COMPLETE |
| Constraint identification | COMPLETE |
| Data reference extraction | COMPLETE |

#### Context Builder (`src/question_ingestion/context_builder.py`)

| Feature | Status |
|---------|--------|
| Combine parsed question + data summary | COMPLETE |
| Generate text context for agent prompts | COMPLETE |
| Extract key variables from column names | COMPLETE |

#### Question Store (`src/question_ingestion/question_store.py`)

| Feature | Status |
|---------|--------|
| Singleton store for current question | COMPLETE |
| Agent access to data context | COMPLETE |

---

### Scripts (`scripts/`)

| Script | Purpose | Status |
|--------|---------|--------|
| `ingest_data.py` | Ingest JSON files into ChromaDB | COMPLETE |
| `test_queries.py` | Test RAG query functionality | COMPLETE |
| `test_agents.py` | Test agent system components | COMPLETE |
| `test_agents_live.py` | Test agents with real API calls | COMPLETE |
| `test_full_pipeline.py` | End-to-end pipeline test | COMPLETE |
| `test_question_ingestion.py` | Test data loading and context building | COMPLETE |

---

## What's Left to Build

### Priority 1: Execute Live Tests

**Status: READY TO RUN**

Test scripts are created. Need to execute them to verify API connectivity.

```bash
# Test individual agents
python scripts/test_agents_live.py

# Test full pipeline (with single iteration for speed)
python scripts/test_full_pipeline.py --max-iterations 1

# Test question ingestion
python scripts/test_question_ingestion.py
```

**Expected Issues to Fix:**
- LangGraph state serialization with Pydantic models
- o1 model API differences (no system messages)
- Parallel judge execution (currently sequential)
- Error handling in code execution

### Priority 2: Report Generation

**Status: NOT STARTED**

Generate competition-ready PDF reports.

**Files to Create:**
```
src/report/
├── __init__.py
├── latex_generator.py      # Generate LaTeX from AgentState
├── pdf_compiler.py         # Compile LaTeX to PDF
├── templates/
│   ├── main.tex            # Main document template
│   ├── assumptions.tex     # Assumptions section template
│   ├── model.tex           # Model section template
│   └── results.tex         # Results section template
└── assets/
    └── style.sty           # Custom styling
```

**Implementation Plan:**
1. Create Jinja2 templates for each section
2. Convert MathModel equations (already LaTeX) to document
3. Embed generated visualizations
4. Compile with pdflatex or latexmk

### Priority 3: Parallel Judge Execution

**Status: NOT STARTED**

Currently judges run sequentially. Should run in parallel.

**Changes Needed:**
```python
# In workflow.py, modify _run_judges:
import asyncio

async def _run_judges_parallel(self, state: AgentState) -> AgentState:
    tasks = [
        asyncio.to_thread(self.accuracy_judge.evaluate_state, state),
        asyncio.to_thread(self.clarity_judge.evaluate_state, state),
        asyncio.to_thread(self.creativity_judge.evaluate_state, state),
    ]
    verdicts = await asyncio.gather(*tasks)
    state.judge_verdicts = list(verdicts)
    return state
```

### Priority 4: Improved Error Handling

**Status: NOT STARTED**

Add robust error handling throughout the pipeline.

**Changes Needed:**
1. Wrap each agent call in try/except
2. Add retry logic for API failures (rate limits, timeouts)
3. Add fallback models if primary fails
4. Log errors to file for debugging

---

## File Structure

```
M3_RAG/
├── README.md                    # This file (update after changes!)
├── CLAUDE.md                    # Claude Code instructions
├── requirements.txt             # Python dependencies
├── keyholder.env                # API keys (DO NOT COMMIT)
│
├── M3 Compilation 2026/         # Source data (42 JSON files)
│   ├── 2024_S1.json
│   ├── 2024_S2.json
│   ├── ...
│   └── 2011_S3.json
│
├── chroma_db/                   # Vector database (gitignore)
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/                  # Data models for RAG
│   │   ├── __init__.py
│   │   ├── document.py          # SourceDocument, Question, Assumption
│   │   ├── chunk.py             # Chunk, ChunkType
│   │   └── results.py           # AssumptionResult, RAGResponse
│   │
│   ├── data_processing/         # JSON parsing and chunking
│   │   ├── __init__.py
│   │   ├── schema_normalizer.py # Handle year-specific schemas
│   │   ├── content_extractor.py # Extract content blocks
│   │   └── semantic_chunker.py  # Create typed chunks
│   │
│   ├── embeddings/              # Embedding generation
│   │   ├── __init__.py
│   │   └── embedding_client.py  # OpenAI embedding wrapper
│   │
│   ├── vector_store/            # ChromaDB operations
│   │   ├── __init__.py
│   │   ├── chroma_client.py     # ChromaDB connection
│   │   └── ingestor.py          # Batch ingestion
│   │
│   ├── retrieval/               # Query interface
│   │   ├── __init__.py
│   │   └── query_engine.py      # Hybrid search
│   │
│   ├── agents/                  # Agent system
│   │   ├── __init__.py
│   │   ├── base.py              # BaseAgent class
│   │   ├── models.py            # Pydantic models for agents
│   │   ├── input_handler.py     # Parse problem text
│   │   ├── scout.py             # Problem analysis
│   │   ├── historian.py         # Assumptions + RAG
│   │   ├── mathematician.py     # Model development
│   │   ├── coder.py             # Code implementation
│   │   ├── stress_tester.py     # Sensitivity analysis
│   │   ├── orchestrator.py      # Pipeline coordination
│   │   └── judges/
│   │       ├── __init__.py
│   │       ├── judge_accuracy.py
│   │       ├── judge_clarity.py
│   │       └── judge_creativity.py
│   │
│   ├── pipeline/                # Orchestration
│   │   ├── __init__.py
│   │   ├── workflow.py          # LangGraph state machine
│   │   └── runner.py            # CLI and programmatic runner
│   │
│   └── question_ingestion/      # Question + data loading
│       ├── __init__.py
│       ├── data_loader.py       # CSV/Excel loading with stats
│       ├── question_parser.py   # PDF/text parsing
│       ├── context_builder.py   # Combine question + data
│       └── question_store.py    # Store current question context
│
├── scripts/
│   ├── ingest_data.py           # Run: python scripts/ingest_data.py
│   ├── test_queries.py          # Run: python scripts/test_queries.py
│   ├── test_agents.py           # Run: python scripts/test_agents.py
│   ├── test_agents_live.py      # Run: python scripts/test_agents_live.py
│   ├── test_full_pipeline.py    # Run: python scripts/test_full_pipeline.py
│   └── test_question_ingestion.py # Run: python scripts/test_question_ingestion.py
│
└── outputs/                     # Pipeline outputs (gitignore)
```

---

## Setup & Installation

### 1. Clone and Setup Environment

```bash
cd M3_RAG
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `keyholder.env` with:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIzaSy...
MOONSHOT_API_KEY=sk-...
```

### 3. Ingest Historical Data

```bash
python scripts/ingest_data.py
```

This processes 42 JSON files and creates 1,002 chunks in ChromaDB.

### 4. Verify Setup

```bash
python scripts/test_queries.py   # Test RAG
python scripts/test_agents.py    # Test agents
```

---

## Usage

### Programmatic Usage

```python
from pathlib import Path
from src.pipeline import M3Runner

# Initialize runner
runner = M3Runner(
    env_path=Path("keyholder.env"),
    chroma_path=Path("chroma_db"),
    verbose=True,
)

# Run pipeline
problem_text = """
The M3 Challenge Problem...
Q1: What is the optimal...
Q2: How should we model...
Q3: What recommendations...
"""

state = runner.run(problem_text, output_dir=Path("outputs/my_solution"))

# Get summary
print(runner.get_summary(state))

# Access individual components
print(f"Score: {state.orchestrator_decision.consensus_score}")
print(f"Model: {state.math_model.framework}")
print(f"Assumptions: {len(state.assumptions.assumptions)}")
```

### CLI Usage

```bash
# From text file
python -m src.pipeline.runner problem.txt -o outputs/

# From PDF
python -m src.pipeline.runner problem.pdf -o outputs/

# With data files (NEW)
python -m src.pipeline.runner problem.pdf \
    --data housing_data.csv population.xlsx \
    --output outputs/

# With custom paths
python -m src.pipeline.runner problem.txt \
    --env keyholder.env \
    --chroma chroma_db \
    --output outputs/
```

### Usage with Data Files (NEW)

```python
from pathlib import Path
from src.pipeline import M3Runner

runner = M3Runner()

# Run with problem + data files
state = runner.run_with_data(
    problem_path=Path("2024_problem.pdf"),
    data_files=[
        Path("housing_data.csv"),
        Path("population_stats.xlsx"),
    ],
    output_dir=Path("outputs/"),
)

# Data summary is automatically generated and included in agent context
# Agents can see column names, data types, statistics, date ranges, etc.
```

### RAG-Only Usage

```python
from src.retrieval import get_query_engine

engine = get_query_engine(chroma_path=Path("chroma_db"))

# Search assumptions
results = engine.search_assumptions(
    query="population growth prediction",
    n_results=10,
    with_justification=True,
)

for r in results.results:
    print(f"[{r.source}] {r.assumption}")
    print(f"  Justification: {r.justification}")
```

---

## API Keys Required

| Provider | Model(s) Used | Purpose |
|----------|---------------|---------|
| OpenAI | gpt-4o, o1, text-embedding-3-large | Historian, Mathematician, Accuracy Judge, Embeddings |
| Anthropic | claude-sonnet-4-20250514 | Scout, Coder, Clarity Judge |
| Google | gemini-2.0-flash | Input Handler, Creativity Judge |
| Moonshot | moonshot-v1-128k | Orchestrator |

**Estimated Cost Per Run:** ~$2-5 depending on iterations

---

## Technical Details

### RAG Chunk Types

| Type | Count | Description |
|------|-------|-------------|
| `assumption` | 651 | Assumptions with justifications |
| `model` | 125 | Model development descriptions |
| `sensitivity` | 76 | Sensitivity analysis content |
| `comment` | 90 | Judge feedback |
| `strengths_weaknesses` | 60 | Self-assessments |

### Metadata Schema

```python
{
    "id": "2024_S1_Q1_assumption_1",
    "year": 2024,
    "placement": 1,              # 1=Champion, 2=Runner-up, 3=Third
    "question": "Q1",
    "section_type": "assumption",
    "topic_tags": "housing,growth",
    "model_type": "logistic_growth",
    "has_justification": True,
    "source_file": "2024_S1.json"
}
```

### Pipeline State Flow

```
AgentState {
  problem_context: ProblemContext     # Set by InputHandler + Scout
  assumptions: AssumptionSet          # Set by Historian
  math_model: MathModel               # Set by Mathematician
  implementation: Implementation      # Set by Coder
  sensitivity_report: SensitivityReport # Set by StressTester
  judge_verdicts: list[JudgeVerdict]  # Set by Judges
  orchestrator_decision: OrchestratorDecision # Set by Orchestrator
  iteration_history: list[str]        # Appended each iteration
  current_phase: str                  # Updated by each node
}
```

### Iteration Logic

```python
if consensus_score >= 9.0:
    return END
elif iteration_count >= 3:
    return END  # Force finish
else:
    # Route to agent with lowest-scoring domain
    if accuracy_lowest:
        return "mathematician"
    elif clarity_lowest:
        return "coder"
    else:
        return "historian"
```

---

## Development Notes

### Known Issues

1. **Python Version Warning**: Google libraries warn about Python 3.9. Consider upgrading to 3.10+.

2. **google.generativeai Deprecation**: The `google.generativeai` package is deprecated. Should migrate to `google.genai`.

3. **o1 Model Differences**: The o1 model doesn't support system messages or temperature. The Mathematician agent handles this with a custom `_call_llm` override.

4. **LangGraph Pydantic Compatibility**: May need to use `model_dump()` for state serialization.

### Testing Commands

```bash
# Test RAG queries
python scripts/test_queries.py

# Test agent imports and initialization
python scripts/test_agents.py

# Run full pipeline (once implemented)
python scripts/run_full_pipeline.py
```

### Adding a New Agent

1. Create `src/agents/new_agent.py`:
```python
from .base import BaseAgent
from .models import SomeOutputModel

class NewAgent(BaseAgent):
    agent_name = "new_agent"
    default_provider = "openai"
    default_model = "gpt-4o"

    def run(self, input_data) -> SomeOutputModel:
        prompt = f"..."
        return self.generate_structured(prompt, SomeOutputModel)
```

2. Add to `src/agents/__init__.py`

3. Add node to `src/pipeline/workflow.py`

### Modifying the Pipeline

1. Edit `src/pipeline/workflow.py`
2. Add/modify node functions
3. Update edges in `_build_graph()`
4. Update `AgentState` in `src/agents/models.py` if new data needed

---

## Changelog

### 2026-01-31
- Initial RAG system complete (1,002 chunks)
- All 10 agents implemented
- 3 judges implemented
- LangGraph pipeline complete
- CLI runner complete
- Basic test scripts added

---

## Contact

Project maintained by Connor Brady.

For issues, check the plan file at: `~/.claude/plans/immutable-stirring-cray.md`
