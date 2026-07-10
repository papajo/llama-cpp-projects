# KV Cache Visualizer & Memory Calculator

**Understand why context length blows up VRAM — and what you can do about it.**

The KV (Key-Value) cache is the single biggest driver of serving cost for
local LLMs. This tool helps you reason about it quantitatively: given a
model's architecture and your hardware budget, compute exact KV cache
memory footprints, compare quantization strategies, predict OOM, and
visualise scaling curves.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# List known model architectures
python -m calculator.cli list

# Analyse Llama-3.1-8B at full 128K context
python -m calculator.cli show Llama-3.1-8B

# Compare with Mistral at reduced context
python -m calculator.cli --ctx 8192 show Mistral-7B-v0.3

# Predict max context for a 24 GB GPU
python -m calculator.cli oom Llama-3.1-8B --vram 24

# Run worked examples
python examples/example_usage.py

# Start the web dashboard
uvicorn web.backend.main:app --reload --port 8000
# Open http://localhost:8000/api/dashboard/Llama-3.1-8B
```

---

## The Formula

For each transformer layer, the KV cache stores the Key and Value tensors
for every token in the context window:

```
KV_cache_per_layer = 2 × n_kv_heads × head_dim × ctx_length × bytes_per_elem
                         ↑               ↑            ↑              ↑
                     (K + V)      per-head dim    # tokens    quant type size
```

With GQA (Grouped Query Attention), `n_kv_heads` is smaller than `n_heads`,
which is why models like Llama-3.1-8B (32 Q-heads, 8 KV-heads) have a
~4× smaller KV cache than a naive MHA model of the same size.

### KV Quantization Byte Sizes

| Type   | Bytes/elem | Savings vs f16 |
|--------|-----------|----------------|
| `f16`  | 2.0       | baseline       |
| `q8_0` | 1.0       | 50% smaller    |
| `q4_0` | 0.5       | 75% smaller    |

---

## CLI Reference

```
Usage:
    kv-cache list                       List known model architectures
    kv-cache show <model>               Detailed KV cache analysis
    kv-cache compare <m1> <m2> [...]    Side-by-side comparison
    kv-cache oom <model> --vram N       OOM prediction

Global flags:
    --ctx N         Context length to evaluate (default: model max)
    -q f16,q8_0,q4_0  KV quant types to include
    -s N            Number of server slots (default: 1)
    --allocation    per-slot or unified
    --json          Output raw JSON
```

### Examples

```bash
# Detailed analysis
python -m calculator.cli show Llama-3.1-8B

# JSON output for programmatic use
python -m calculator.cli --json show Mistral-7B-v0.3

# Compare models side by side
python -m calculator.cli compare Llama-3.1-8B Mistral-7B-v0.3 Qwen-2.5-7B

# OOM prediction with exact weight memory
python -m calculator.cli oom Llama-3.1-70B --vram 48 --weights 35
```

---

## Supported Models

18 pre-configured architectures including:

| Model | Layers | KV Heads | Max Ctx |
|-------|--------|----------|---------|
| Llama-3.1-8B | 32 | 8 | 131,072 |
| Llama-3.1-70B | 80 | 8 | 131,072 |
| Llama-3.1-405B | 126 | 16 | 131,072 |
| Mistral-7B-v0.3 | 32 | 8 | 32,768 |
| Mixtral-8x7B (MoE) | 32 | 8 | 32,768 |
| Qwen-2.5-7B | 28 | 4 | 32,768 |
| Qwen-2.5-72B | 80 | 8 | 32,768 |
| DeepSeek-V3 (MoE) | 61 | 128 | 65,536 |
| Gemma-2-9B | 42 | 16 | 8,192 |
| Gemma-2-27B | 46 | 16 | 8,192 |
| Phi-3-mini-4K | 32 | 32 | 4,096 |
| Phi-3-medium-128K | 40 | 10 | 131,072 |
| CodeLlama-34B | 48 | 8 | 16,384 |

Add new models by editing `calculator/presets.py`.

---

## Web API

Start the server:

```bash
uvicorn web.backend.main:app --reload --port 8000
```

| Endpoint | Description |
|----------|-------------|
| `GET /api/models` | List all known models |
| `GET /api/models/{name}?ctx=N` | KV cache analysis for a model |
| `POST /api/calculate` | Custom calculation (JSON body) |
| `POST /api/oom` | OOM prediction |
| `GET /api/dashboard/{name}` | Full HTML dashboard with Plotly charts |
| `GET /api/health` | Health check |

### Example: Custom calculation

```bash
curl -X POST http://localhost:8000/api/calculate \
  -H "Content-Type: application/json" \
  -d '{"model": "Llama-3.1-8B", "max_ctx": 65536, "quants": ["f16", "q4_0"]}'
```

### Example: OOM prediction

```bash
curl -X POST http://localhost:8000/api/oom \
  -H "Content-Type: application/json" \
  -d '{"model": "Llama-3.1-8B", "vram_gb": 24}'
```

---

## Visualizations

The dashboard (available at `/api/dashboard/{name}`) includes:

1. **Memory Scaling** — KV cache size vs context length (all quant types)
2. **Quant Comparison** — Per-layer / total / all-slots breakdown
3. **OOM Frontier** — Max context vs available VRAM
4. **Slot Allocation** — Per-slot vs unified, scaling with number of slots

Each chart is interactive (zoom, pan, hover data, export to PNG).

---

## Live Server Probing

Poll a running `llama-server` for real KV cache usage:

```python
from server_probe import ServerProbe

probe = ServerProbe("http://localhost:8080")
report = probe.report()

print(f"Active slots: {report['active_slots']}")
print(f"Total KV tokens cached: {report['total_kv_tokens_cached']}")
print(f"KV cache usage: {report['metrics']['kv_cache_usage_ratio']:.1%}")
```

---

## Architecture

```
kv-cache-visualizer/
├── calculator/           # Core math engine
│   ├── __init__.py
│   ├── models.py         # Data models (KVQuant, ModelConfig, etc.)
│   ├── calculator.py     # KVCacheCalculator — all formulas
│   ├── presets.py        # 18 pre-configured model architectures
│   ├── cli.py            # CLI interface
│   └── __main__.py       # python -m calculator entry point
├── viz/                  # Plotly chart generators
│   ├── __init__.py
│   └── plots.py          # Memory scaling, quant comparison, OOM frontier, etc.
├── web/
│   └── backend/
│       └── main.py       # FastAPI server
├── server-probe/         # Live llama-server polling
│   ├── __init__.py
│   └── probe.py          # ServerProbe class
├── examples/
│   ├── example_usage.py  # Comprehensive worked examples
│   ├── llama31-8b-analysis.json
│   └── qwen-72b-analysis.json
├── tests/                # (coming soon)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Learning Outcomes

After using this tool you'll understand:

- **Why 128K context on a 24 GB GPU is tight** — and exactly where the bytes go
- **How GQA saves KV cache memory** — Mistral's 4:1 ratio vs Gemma's full MHA
- **When to use `--cache-type-k q4_0`** — the quantified tradeoff between quality and memory
- **How `--kv-unified` changes the memory equation** — shared pool vs per-slot isolation
- **What context length your hardware can actually support** — before OOM

---

## llama.cpp Flag Reference

| Flag | Effect |
|------|--------|
| `-ctk TYPE` / `--cache-type-k TYPE` | Key cache quantization (f16, q8_0, q4_0) |
| `-ctv TYPE` / `--cache-type-v TYPE` | Value cache quantization |
| `-kvu` / `--kv-unified` | Unified KV cache across all slots |
| `-cram` / `--cache-ram` | RAM cache size for offloaded layers |
| `-ctxcp` / `--ctx-checkpoints` | Enable KV cache checkpoints |
| `--defrag-thold` | KV cache defragmentation threshold |
| `--context-shift` | Context shift mode for long sequences |
