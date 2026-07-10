# Prompt Chaining Workbench

Chain multiple LLM calls together, piping the output of one step as the
input to the next. Includes a Python library for programmatic use and a
visual HTML dashboard for interactive design and testing.

## Problem

Complex tasks often exceed a single LLM call. You might need to:

1. Summarise text, then translate the summary
2. Brainstorm ideas, critique them, then refine
3. Decompose a question, answer sub-parts, then synthesise

Without a framework, you manually copy-paste outputs between calls or
write fragile glue code. A chaining library makes the pipeline explicit,
reproducible, and testable.

## Solution

A `PromptChain` is a sequence of `ChainStep` objects. Each step is an
LLM call whose prompt can reference prior outputs via template variables
(`{{input}}`, `{{step_1}}`, `{{step_name}}`).

- **Programmatic API** — build and run chains in Python against any
  llama.cpp server
- **6 pre-built templates** — common patterns ready to use
- **Visual workbench** — `tools/chain-workbench.html` for drag-free
  chain design, live execution, and JSON import/export
- **Context passing** — outputs flow between steps automatically
- **Retry & error handling** — each step retries on transient errors;
  stop-on-error or continue modes

## Quick Start

```python
from prompt_chaining.chain import PromptChain, ChainStep
from prompt_chaining.runner import run_chain

# Define a 2-step chain
chain = PromptChain(
    name="summarize-and-translate",
    steps=[
        ChainStep(
            name="summarise",
            system_prompt="You are a precise summariser.",
            user_prompt="Summarise: {{input}}",
        ),
        ChainStep(
            name="translate",
            user_prompt="Translate to French: {{step_1}}",
        ),
    ],
)

# Run it (requires llama.cpp server on port 8080)
result = run_chain(
    chain,
    input_text="Long article text here...",
    base_url="http://127.0.0.1:8080",
)

for sr in result.steps:
    print(f"[{sr.step_name}] {sr.output[:100]}...")
```

## Pre-built Chain Templates

| Template | Steps | Description |
|----------|-------|-------------|
| `summarize_and_translate()` | 2 | Summarise → Translate to French |
| `brainstorm_critique_refine()` | 3 | Ideas → Critique → Refined version |
| `expand_and_polish()` | 2 | Expand outline → Polish prose |
| `question_decomposition()` | 4 | Decompose → Answer Q1 → Answer Q2 → Synthesise |
| `fact_check_pipeline()` | 3 | Generate claim → Evidence → Confidence rating |
| `chain_of_thought()` | 2 | Step-by-step reasoning → Final answer |

```python
from prompt_chaining.templates import brainstorm_critique_refine

chain = brainstorm_critique_refine()
result = run_chain(chain, input_text="How to reduce plastic waste?")
```

## Visual Workbench

The HTML workbench lets you design and run chains without writing code:

```bash
cd llama-cpp-projects
python3 -m http.server 8000
# Open http://localhost:8000/tools/chain-workbench.html
```

Features:
- Add / remove / reorder steps
- Load 6 built-in templates with one click
- Edit prompts, temperature, max_tokens per step
- Run against any llama.cpp server
- Per-step results with timing, tokens, and rendered prompt viewer
- Import / export chains as JSON
- Share chains via URL hash

## Template Variables

The following variables are available in step prompts:

| Variable | Resolves To |
|----------|-------------|
| `{{input}}` | Original input to the chain |
| `{{step_1}}`, `{{step_2}}`, … | Output of step N (1-indexed) |
| `{{step_name}}` | Output of the step with that name |

## API Reference

### ChainStep
- `name` — Step identifier (used in template resolution)
- `user_prompt` — User message template (required)
- `system_prompt` — System message template (optional)
- `temperature` — Per-step override (uses chain default if None)
- `max_tokens` — Per-step override

### PromptChain
- `name` — Chain identifier
- `steps` — List of `ChainStep`
- `temperature` / `max_tokens` — Defaults for all steps
- `to_dict()` / `from_dict()` — JSON serialisation round-trip

### run_chain()
- `chain` — `PromptChain` to execute
- `input_text` — Initial input (available as `{{input}}`)
- `base_url` — llama.cpp server URL (default `http://127.0.0.1:8080`)
- `max_retries` — Retries per step on 5xx/timeout (default 2)
- `stop_on_error` — Stop chain on first failure (default True)
- `progress_cb` — Optional `(idx, total, name, status)` callback
- Returns `ChainResult`

### ChainResult
- `steps` — List of `StepResult`
- `all_ok` — True if every step succeeded
- `last_output` — Output of the last successful step
- `ok_steps` / `failed_steps` — Filtered lists
- `context()` — Builds resolution dict from results

### StepResult
- `step_name`, `step_index`, `output`, `error`
- `prompt_tokens`, `completion_tokens`, `elapsed_s`
- `rendered_system`, `rendered_user` — Resolved prompts sent to the API
- `is_ok` — True if no error

## Project Structure

```
3.4-prompt-chaining-workbench/
├── prompt_chaining/
│   ├── __init__.py
│   ├── chain.py        # ChainStep, PromptChain, template resolution
│   ├── runner.py       # run_chain(), StepResult, ChainResult, retry
│   └── templates.py    # 6 pre-built chain templates
├── tests/
│   ├── test_chain.py       # 12 tests (resolution, serialisation)
│   ├── test_runner.py      # 15 tests (execution, retry, context)
│   └── test_templates.py   # 20 tests (all 6 templates, list/load)
├── pyproject.toml
├── README.md
└── todo-plan.md
```

Also:
- `tools/chain-workbench.html` — visual chain designer + runner

## Requirements

- llama.cpp server running at `http://127.0.0.1:8080` (configurable)
- Python 3.11+
- `httpx`, `pydantic` (installed automatically)

## Running Tests

```bash
pip install -e ".[test]"
pytest -v
```

All 47 tests passing.
