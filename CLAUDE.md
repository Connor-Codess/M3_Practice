# CLAUDE.md

## Project Overview

M3_RAG is a Python-based project for testing and evaluating multiple LLM provider APIs. Currently in early/experimental stage, it validates API connectivity and response formats across providers.

## Structure

```
M3_RAG/
├── .cursorrules          # Cursor AI IDE config (no emojis, clean output)
├── api_pulls.ipynb       # Main notebook - API integration tests
└── CLAUDE.md             # This file
```

## Tech Stack

- **Language:** Python (Jupyter Notebook)
- **Dependencies:** `requests`, `json` (stdlib)
- **No dependency manager** - no requirements.txt or pyproject.toml exists

## APIs Tested

| Provider     | Model                      | Auth Method        |
|-------------|----------------------------|--------------------|
| Moonshot/Kimi| moonshot-v1-128k          | Bearer token       |
| Google Gemini| gemini-3-flash-preview    | API key param      |
| OpenAI       | gpt-4o                    | Bearer token       |
| Anthropic    | claude-sonnet-4-5-20250929| x-api-key header   |

## Running

Open and run cells in `api_pulls.ipynb` via Jupyter Notebook or JupyterLab.

## Known Issues

- API keys are hardcoded in the notebook rather than loaded from environment variables or `.env` files.
- No `requirements.txt` or formal dependency management.
- No modular code structure - all code lives in notebook cells.

## Style

- No emojis in code, comments, or output.
- Write human-like comments that explain the "why", not the "what".
- Keep comments concise and natural, like a developer would write.
- Write clean, professional, production-ready code.
- Avoid over-commenting obvious code.
- Use clear variable and function names that reduce the need for comments.
