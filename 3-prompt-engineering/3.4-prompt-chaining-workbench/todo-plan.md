# 3.4 — Prompt Chaining Workbench: Todo & Plan

## Goal

Build a prompt-chaining library that sequences multiple LLM calls, piping
output from one step as input to the next, plus a visual HTML workbench
for manual chain design and testing.

## Module Architecture

```
3.4-prompt-chaining-workbench/
├── prompt_chaining/
│   ├── __init__.py
│   ├── chain.py        # ChainStep, PromptChain, step definition
│   ├── runner.py       # execute chain, context passing, retry, streaming
│   └── templates.py    # pre-built chain templates
├── tests/
│   ├── test_chain.py
│   ├── test_runner.py
│   └── test_templates.py
├── pyproject.toml
├── README.md
└── todo-plan.md
```

Also:
- `tools/chain-workbench.html` — visual chain designer + runner

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Scaffold project (pyproject.toml, dirs, __init__.py) | ✅ |
| 2 | `chain.py` — ChainStep, PromptChain, template variable resolution | |
| 3 | `runner.py` — execute chain, context dict, retry, step-level results | |
| 4 | `templates.py` — 6 pre-built chain templates | |
| 5 | Tests for all 3 modules | |
| 6 | `tools/chain-workbench.html` — visual chain designer + runner | |
| 7 | `pytest -v` — all tests passing | |
| 8 | README.md with quick-start, API ref, chain templates | |

## Key Design Decisions

- Steps reference prior outputs via Jinja2-style `{{step_N.output}}` or `{{step_N.system_output}}`
- `{{input}}` always refers to the original input to the chain
- Context is a flat dict: `{"input": ..., "step_1": ..., "step_2": ..., ...}`
- Each step can override temperature/max_tokens from the chain default
- Runner returns `ChainResult` with per-step `StepResult` (output, tokens, timing, error)
- Workbench is a standalone HTML page (no build step), served via `python3 -m http.server`
- Chain templates stored as JSON-serialisable dicts for easy import/export
