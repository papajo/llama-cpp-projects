# Sampler Ablation Lab

Run, compare, and analyse LLM sampler configurations side-by-side.
Toggle one sampling parameter at a time to see exactly what each knob
does to the output — temperature, top-k, top-p, min-p, repeat penalty,
mirostat, and more.

## Problem

llama.cpp exposes ~15 sampling parameters. Changing one can ripple
through the output in unexpected ways. "Ablation" — varying one knob
while holding everything else constant — reveals the causal effect of
each parameter, helping you choose the right settings for your task.

## Solution

Define any number of `SamplerConfig` objects, send a prompt through
each against the same llama.cpp server, and get a markdown report with:

- **Metrics table** — words, chars, tokens, speed, vocabulary size,
  repetition rate, and character entropy, per config
- **Full outputs** — verbatim responses for side-by-side reading
- **Word-level diff** — which words appear/disappear vs a reference
- **Presets** — 6 ready-to-use configurations (greedy, creative,
  balanced, precise, mirostat-v1, mirostat-v2)
- **Automatic ablation** — `ablate()` generates variants by changing
  one parameter at a time from a reference config

## Quick Start

```python
from sampler_ablation.config import preset_greedy, preset_creative
from sampler_ablation.runner import run_ablation
from sampler_ablation.comparator import format_comparison

# Compare two presets side-by-side
configs = [preset_greedy(), preset_creative()]
results = run_ablation(
    configs=configs,
    prompt="Write a haiku about Python.",
    base_url="http://127.0.0.1:8080",
)

# Generate a markdown report
report = format_comparison(results, title="Greedy vs Creative")
print(report)
```

## Presets

Six built-in presets covering the major sampling regimes:

| Preset | Temperature | Top-K | Top-P | Description |
|--------|-------------|-------|-------|-------------|
| `preset_greedy()` | 0.0 | 1 | 1.0 | Deterministic — always picks the most likely token |
| `preset_creative()` | 1.2 | 100 | 0.95 | Diverse and unpredictable |
| `preset_balanced()` | 0.7 | 40 | 0.9 | Default sensible trade-off |
| `preset_precise()` | 0.2 | 20 | 0.8 | Focused, factual output |
| `preset_mirostat_v1()` | 0.8 | 40 | 0.9 | Adaptive entropy (v1) |
| `preset_mirostat_v2()` | 0.8 | 40 | 0.9 | Improved adaptive entropy (v2) |

## Automatic Ablation

`ablate()` generates a family of configs that each change exactly one
parameter from a reference:

```python
from sampler_ablation.config import ablate, preset_balanced
from sampler_ablation.runner import run_ablation
from sampler_ablation.comparator import format_comparison

# Create variants: one knob changed per run
configs = ablate(reference=preset_balanced())
# → [balanced, balanced (temperature=0.0), balanced (temperature=1.5),
#    balanced (top_k=1), balanced (top_k=100), ...]

results = run_ablation(configs=configs, prompt="Explain quantum computing simply.")
report = format_comparison(results, title="Ablation from Balanced")
```

The default ablation grid covers 6 parameters with 2 values each
(temperature, top_k, top_p, min_p, repeat_penalty, mirostat), yielding
~13 configs including the reference.

## Custom Ablation Grids

Pass your own parameter pairs to ablate specific knobs:

```python
from sampler_ablation.config import ablate, preset_precise

configs = ablate(
    reference=preset_precise(),
    params=[
        ("temperature", [0.0, 0.5, 1.0]),
        ("repeat_penalty", [1.0, 1.1, 1.2]),
    ],
)
# → [precise, precise (temperature=0.0), precise (temperature=0.5),
#    precise (temperature=1.0), precise (repeat_penalty=1.0),
#    precise (repeat_penalty=1.1), ...]
```

## Project Structure

```
3.2-sampler-ablation-lab/
├── sampler_ablation/
│   ├── __init__.py
│   ├── config.py       # SamplerConfig, 6 presets, ablate() generator
│   ├── runner.py       # run_ablation(), AblationResult, retries
│   └── comparator.py   # format_comparison(), metrics, word-level diff
├── tests/
│   ├── test_config.py      # 16 tests (defaults, presets, ablation)
│   ├── test_runner.py      # 9 tests (HTTP calls, retries, progress)
│   └── test_comparator.py  # 15 tests (format, metrics, diffs)
├── pyproject.toml
├── README.md
└── todo-plan.md
```

## API Reference

### SamplerConfig (config.py)

Frozen dataclass with all llama.cpp sampling parameters:

| Field | Default | Description |
|-------|---------|-------------|
| `temperature` | 0.7 | Sampling temperature |
| `top_k` | 40 | Top-K filtering |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `min_p` | 0.0 | Minimum probability (≥0.05 recommended) |
| `tfs_z` | 1.0 | Tail-free sampling Z value |
| `typical_p` | 1.0 | Typical sampling threshold |
| `repeat_penalty` | 1.0 | Repetition penalty |
| `frequency_penalty` | 0.0 | Frequency penalty |
| `presence_penalty` | 0.0 | Presence penalty |
| `mirostat` | 0 | Mirostat mode (0=off, 1=v1, 2=v2) |
| `mirostat_tau` | 5.0 | Mirostat target entropy |
| `mirostat_eta` | 0.1 | Mirostat learning rate |
| `seed` | None | Fixed random seed (for reproducibility) |
| `max_tokens` | 512 | Max tokens to generate |
| `n_keep` | 0 | Tokens to keep from prompt |

Key methods:
- `to_request_body()` — returns `dict` of non-default params for the API
- `param_summary()` — multi-line string of what's non-default

### AblationResult (runner.py)

| Property | Description |
|----------|-------------|
| `label` | Config label (delegated from `config.label`) |
| `output` | Generated text from the model |
| `prompt_tokens` / `completion_tokens` | Token counts from usage |
| `elapsed_s` | Wall-clock time |
| `is_ok` | `True` if no error |
| `error` | Error string or `None` |

### format_comparison() (comparator.py)

Generates a markdown report with:
- Overview metrics table (vocabulary, repetition, entropy, speed)
- Per-config sampler details
- Full output sections
- Word-level diffs against a reference config
- Error listing and notes section

## Requirements

- llama.cpp server running at `http://127.0.0.1:8080` (configurable)
- Python 3.11+
- `httpx`, `pydantic` (installed automatically)

## Running Tests

```bash
pip install -e ".[test]"
pytest -v
```

All 40 tests passing.
