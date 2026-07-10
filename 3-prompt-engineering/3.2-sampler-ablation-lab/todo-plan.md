# 3.2 — Sampler Ablation Lab: Todo & Plan

## Goal

Build a tool that runs the **same prompt** through different sampler
configurations and produces a side-by-side comparison showing how each
sampler affects output — length, vocabulary, repetition, and tone.

## Module Architecture

```
3.2-sampler-ablation-lab/
├── sampler_ablation/
│   ├── __init__.py
│   ├── config.py       # SamplerConfig, presets, ablation generators
│   ├── runner.py       # Run prompts through muliple configs
│   └── comparator.py   # Side-by-side diff, per-config metrics
├── tests/
│   ├── test_config.py
│   ├── test_runner.py
│   └── test_comparator.py
├── pyproject.toml
├── README.md
└── todo-plan.md
```

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Create project scaffold (pyproject.toml, dirs, __init__.py) | |
| 2 | `config.py` — SamplerConfig, 6+ presets, ablation generator | |
| 3 | `runner.py` — run configs, collect outputs, retry logic | |
| 4 | `comparator.py` — side-by-side table, per-config metrics, word diff | |
| 5 | Tests for all 3 modules | |
| 6 | README.md documenting the project | |
| 7 | `pytest -v` all passing | |

## Key Design Decisions

- Ablation mode: generate N variants by toggling one param at a time from
  a reference config — shows the delta each sampler introduces.
- Comparison output: markdown table with metrics + full output sections.
- httpx error handling: check `resp.status_code >= 400` (httpx 0.28.1 compat).
- SamplerConfig stores ALL supported sampler params, with built-in presets
  for greedy, creative, balanced, precise, mirostat-v1, mirostat-v2.
