# Speculative Decoding Strategy Benchmark Suite

**Quantify the speedup of all 8 speculative decoding strategies in llama.cpp.**

Speculative decoding is the highest-leverage local-inference speedup technique,
but strategy choice is workload-dependent. This benchmark suite systematically
tests all 8 strategies across code, prose, and repetitive-log workloads to
answer the question: *which strategy should I use for my use case?*

---

## The 8 Strategies

### Draft-Model-Based (require a separate draft GGUF)

| # | Strategy | Flag | How it works |
|---|----------|------|-------------|
| 1 | **Draft-Simple** | `--spec-type draft-simple` | Small LM generates candidates; target verifies in one forward pass |
| 2 | **Draft-EAGLE3** | `--spec-type draft-eagle3` | Draft conditions on target's hidden states for higher acceptance |
| 3 | **Draft-MTP** | `--spec-type draft-mtp` | Multi-Token Prediction head, trained to predict multiple future tokens |

### N-Gram-Based (no separate model needed, use prompt context)

| # | Strategy | Flag | How it works |
|---|----------|------|-------------|
| 4 | **Ngram-Simple** | `--spec-type ngram-simple` | Standard n-gram lookup from prompt context |
| 5 | **Ngram-Map-K** | `--spec-type ngram-map-k` | n-gram with map of length K for faster lookups |
| 6 | **Ngram-Map-K4V** | `--spec-type ngram-map-k4v` | Compact 4-byte key-value n-gram map |
| 7 | **Ngram-Mod** | `--spec-type ngram-mod` | Modular arithmetic for n-gram indexing |
| 8 | **Ngram-Cache** | `--spec-type ngram-cache` | Persists n-grams across generations in the session |

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# List strategies with descriptions
python -m harness.cli list-strategies

# Run the full benchmark suite
python -m harness.cli run /path/to/model.gguf \
    --draft-model /path/to/draft.gguf \
    --gpu-layers -1 \
    --ctx-size 4096 \
    --n-predict 200 \
    --runs 3

# Generate report from saved results
python -m harness.cli report results/raw_results.json

# Run only the baseline (no speculation)
python -m harness.cli baseline /path/to/model.gguf
```

---

## Prompt Corpora

Three workload categories, each with 5 prompts:

| Category | Type | Characteristics | N-gram friendly? |
|----------|------|----------------|-----------------|
| **code/** | Bubble sort, merge sort, JSON parser, React component, FastAPI endpoints | Structured syntax, repetitive keywords | ✅ Yes |
| **prose/** | Explanations, creative stories, summaries, instructions, essays | Variable patterns, creative | ❌ Less |
| **logs/** | Web server logs, JSON log entries, CSV metrics, syslog, Prometheus metrics | Highly templated, repetitive | ✅✅ Very |

---

## Metrics

| Metric | Description |
|--------|-------------|
| **Acceptance Rate** | Fraction of draft tokens accepted by the target model |
| **Tokens/Second** | Generation throughput (wall time) |
| **Draft Efficiency** | Accepted tokens per draft model forward pass |
| **Verification Ratio** | Target forward passes per generated token (lower = better) |
| **p50/p95/p99 Latency** | Per-token generation latency percentiles |
| **Latency Jitter** | Coefficient of variation of inter-token latencies |

---

## Reports

The benchmark generates a complete report package:

```
results/
├── report.md          # Full Markdown report with tables
├── report.html        # Interactive Plotly dashboard with 6 charts
├── summary.csv        # Aggregated metrics table
├── summary.json       # Raw data for further analysis
└── raw_results.json   # Per-run metrics
```

### Dashboard Charts

1. **Throughput** — tok/s by strategy × prompt
2. **Acceptance Rate** — draft acceptance by strategy × prompt
3. **Latency Profile** — p50/p95/p99 latency comparison
4. **Strategy Radar** — normalised multi-metric comparison
5. **Draft Efficiency** — efficiency vs acceptance scatter
6. **Workload Heatmap** — speedup factor for every strategy × workload pair

---

## CLI Reference

```
Usage:
    spec-bench list-strategies              List all strategies with descriptions
    spec-bench run <model>                  Run full benchmark against all strategies
    spec-bench run-strategy <model> <strat> Run a single strategy
    spec-bench baseline <model>             Baseline only (no speculation)
    spec-bench report <results.json>        Generate report from saved results

Options:
    --server PATH         llama-server binary path (default: llama-server)
    --port PORT           Server port (default: 18080)
    --gpu-layers N        GPU layers (-1 = all)
    --ctx-size N          Context size (default: 4096)
    --n-predict N         Tokens per completion (default: 200)
    --temperature F       Sampling temperature (default: 0.7)
    --runs N              Runs per prompt for significance (default: 1)
    -o, --output DIR      Output directory (default: results)
    -v, --verbose         Show server output
```

---

## Docker

For reproducible runs with isolated GPU access:

```bash
# Build
docker-compose build

# Run full benchmark
docker-compose up benchmark

# Or just baseline
docker-compose up baseline

# Results land in ./results/
```

---

## Architecture

```
spec-decoding-benchmark/
├── harness/                    # Core benchmarking engine
│   ├── configs.py              # 8 strategy definitions with params
│   ├── runner.py               # Subprocess manager, server lifecycle, API client
│   ├── metrics.py              # Statistical metrics computation
│   └── cli.py                  # CLI interface
├── prompts/                    # Prompt corpora
│   ├── code.py                 # 5 code generation prompts
│   ├── prose.py                # 5 natural language prompts
│   └── logs.py                 # 5 repetitive/log prompts
├── report_generator/           # Analysis and visualisation
│   ├── analysis.py             # Aggregation, per-strategy stats, recommendations
│   ├── plots.py                # 6 Plotly chart types + dashboard
│   └── report.py              # Markdown/HTML/CSV report generation
├── results/                    # Output directory
├── docker-compose.yml          # Reproducible Docker setup
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Expected Findings

Based on the architecture of each strategy:

| Workload | Likely Best Strategy | Why |
|----------|---------------------|-----|
| **Code** | Draft-Simple or Draft-MTP | Structured syntax benefits from draft-model predictions; n-grams help with keywords but not logic |
| **Prose** | Draft-Simple or Draft-EAGLE3 | Variable patterns need a real draft model; EAGLE3's hidden-state conditioning helps with creative text |
| **Logs** | Ngram-Cache or Ngram-Map-K4V | Highly repetitive templated text is where n-gram strategies excel; no draft model needed |
| **Mixed** | Draft-Simple | Best all-rounder; reasonable on all workloads |
| **Repetitive** | Ngram-Simple | Zero overhead from a draft model, identical to target on repeated patterns |

---

## llama.cpp Flag Reference

| Flag | Effect |
|------|--------|
| `--spec-type TYPE` | Speculative decoding strategy |
| `--spec-draft-model PATH` | Draft model for draft-based strategies |
| `--spec-draft-n-max N` | Max draft tokens per iteration |
| `--spec-draft-n-min N` | Min draft tokens before verification |
| `--spec-draft-p-split F` | Probability split for draft vs verify |
| `--spec-draft-p-min F` | Minimum acceptance probability |
| `--spec-ngram-*-size-n N` | N-gram order N |
| `--spec-ngram-*-size-m M` | N-gram order M |
| `--spec-ngram-*-min-hits N` | Min hits before using n-gram |
| `--spec-draft-backend-sampling` | Use backend sampling for draft |

---

## Learning Outcomes

- How acceptance probability determines speculative decoding speedup
- Why EAGLE-3 and MTP differ architecturally from simple draft models
- When n-gram lookup decoding beats draft models (and vice versa)
- How `--spec-draft-p-split` and `--spec-draft-n-max` interact
- Which strategy to choose for code vs chat vs log generation
