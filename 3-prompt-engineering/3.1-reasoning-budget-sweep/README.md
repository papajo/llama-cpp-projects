# Reasoning Budget Sweep

Systematically probe how inference-time budgets affect LLM output —
measure token efficiency, diversity, and speed as you sweep parameters
against llama.cpp's server API.

## Problem

LLM inference has several "budget knobs": `max_tokens`, `temperature`,
`top_p`, `min_p`, `repeat_penalty`, etc. The best settings depend on
your task, model, and latency requirements. Without systematic
measurement, it's easy to waste tokens on verbose outputs, pick a
temperature that reduces diversity, or choose a token budget far larger
than needed.

## Solution

Define a parameter grid and let the sweep engine run all combinations
against llama.cpp's `/v1/chat/completions`. It collects:

- **Token efficiency** — words per completion token (higher = more
  information per token)
- **Diversity** — pairwise Jaccard distance between runs at the same
  config (measures how much the output varies)
- **Repetition rate** — fraction of repeated bigrams (quality signal)
- **Speed** — tokens per second (latency impact)

Results are aggregated into markdown reports with leaderboards sorted
by efficiency, diversity, and speed.

## Quick Start

```python
from budget_sweep.budget import SweepConfig
from budget_sweep.sweep import run_sweep
from budget_sweep.metrics import compute_metrics
from budget_sweep.reporter import format_report

# Define the sweep
config = SweepConfig(
    base_url="http://127.0.0.1:8080",
    param_grid={
        "max_tokens": [64, 128, 256, 512],
        "temperature": [0.0, 0.7],
    },
    n_runs=3,                          # 3 repeats per config for diversity
)

# Run it (requires a running llama.cpp server)
results = run_sweep(config)

# Analyse and report
summaries = compute_metrics(results)
report = format_report(summaries, title="My First Sweep")
print(report)
```

## Presets

Common experiments are available as one-liners:

```python
from budget_sweep.budget import load_preset

# Sweep max_tokens 64→4096 at T=0 and T=0.7
cfg = load_preset("token-budget")

# Sweep temperature 0.0→2.0 at fixed max_tokens
cfg = load_preset("temperature-sweep")

# Sweep all major sampling params (3×3 grid)
cfg = load_preset("sampling-full")

# Test context window sizes 512→32768
cfg = load_preset("context-window")
```

## Project Structure

```
3.1-reasoning-budget-sweep/
├── budget_sweep/
│   ├── __init__.py
│   ├── budget.py       # SweepConfig, parameter definitions, presets
│   ├── metrics.py      # SweepResult, compute_metrics, PerPromptMetrics
│   ├── sweep.py        # Orchestrator — runs configs against llama.cpp
│   └── reporter.py     # Markdown report generation
├── tests/
│   ├── test_budget.py
│   ├── test_metrics.py
│   ├── test_sweep.py
│   └── test_reporter.py
├── pyproject.toml
├── README.md
└── todo-plan.md
```

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
