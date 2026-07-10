# 3.1 — Reasoning Budget Sweep: Todo & Plan

## Goal

Build a tool that systematically sweeps inference-time budgets (max tokens,
temperature, top-p, min-p, repeat penalty) and measures how each affects
output length, quality, diversity, and token efficiency against llama.cpp's
server API.

## Module Architecture

```
3.1-reasoning-budget-sweep/
├── budget_sweep/
│   ├── __init__.py
│   ├── budget.py        # SweepConfig, parameter ranges
│   ├── metrics.py       # Length, diversity, efficiency measurements
│   ├── sweep.py         # Orchestrator — run configs, collect results
│   └── reporter.py      # Text/markdown report generation
├── tests/
│   ├── test_budget.py
│   ├── test_metrics.py
│   ├── test_sweep.py
│   └── test_reporter.py
├── pyproject.toml
├── README.md
└── todo-plan.md
```

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Create project scaffold (pyproject.toml, dirs, __init__.py) | ✅ |
| 2 | `budget.py` — SweepConfig, parameter definitions, presets | ✅ |
| 3 | `metrics.py` — output length, repetition, diversity, token efficiency | ✅ |
| 4 | `sweep.py` — orchestrator, llama.cpp client, result collection | ✅ |
| 5 | `reporter.py` — text report generation with tables | ✅ |
| 6 | Tests for all 4 modules (47 tests) | ✅ |
| 7 | README.md documenting the project | ✅ |
| 8 | `pytest -v` — all 47 passing | ✅ |

## Key Design Decisions

- httpx `raise_for_status()` patched: httpx 0.28.1 removed `_request` attrs
  from `HTTPStatusError`. All error handling checks `resp.status_code >= 400`.
- SweepConfig uses a `param_grid` dict so arbitrary combinations can be swept
- Metrics are computed server-side where possible (`usage.prompt_tokens`,
  `usage.completion_tokens`)
- Reporter outputs markdown tables sorted by token efficiency
- `n_runs` repeats per config enable pairwise diversity measurement
