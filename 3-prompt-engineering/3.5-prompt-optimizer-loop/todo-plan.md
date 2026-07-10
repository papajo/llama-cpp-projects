# 3.5 — Prompt Optimizer Loop: Todo & Plan

## Goal

Build a self-improving prompt optimization loop: define a prompt
template with variables, supply test cases with expected outputs,
run the prompt against an LLM, score the results, and use an LLM
to iteratively improve the prompt until scores meet a threshold.

## Module Architecture

```
3.5-prompt-optimizer-loop/
├── prompt_optimizer/
│   ├── __init__.py
│   ├── prompt.py        # PromptTemplate, TestCase
│   ├── evaluator.py     # scoring functions (exact, contains, rubric)
│   ├── optimizer.py     # optimization loop (run → eval → improve → repeat)
│   └── reporter.py      # iteration history tracking, markdown reports
├── tests/
│   ├── test_prompt.py
│   ├── test_evaluator.py
│   ├── test_optimizer.py
│   └── test_reporter.py
├── pyproject.toml
├── README.md
└── todo-plan.md
```

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Scaffold project | ✅ |
| 2 | `prompt.py` — PromptTemplate with variable substitution, TestCase | |
| 3 | `evaluator.py` — Scorecard, exact/contains/rubric evaluators (LLM-as-judge) | |
| 4 | `optimizer.py` — optimize() loop, ImprovementProposal, meta-prompt for improvements | |
| 5 | `reporter.py` — iteration history, markdown reports with trajectories | |
| 6 | Tests for all 4 modules | |
| 7 | `pytest -v` — all tests passing | |
| 8 | README.md with quick-start, API ref, usage examples | |

## Key Design Decisions

- PromptTemplate uses `{{variable}}` syntax (compatible with Jinja2-style)
- Evaluators return scores 0.0–1.0 for composability
- LLM-as-judge rubric evaluator uses the same llama.cpp server
- The optimizer's improvement step also calls llama.cpp with a meta-prompt
- History tracked as list of `IterationSnapshot` (prompt, scores, outputs)
- Stop conditions: score >= threshold, max_iterations reached, score plateau
