# GGUF Quant Explorer

Understand quantisation tradeoffs for any GGUF model. Read raw GGUF metadata, compute "what-if" memory for every quant type, get hardware-aware recommendations, and generate visual HTML reports.

## What it does

- **GGUF metadata parser** — reads raw binary GGUF files and extracts architecture, dimensions, tensor info, and current quantisation
- **Memory estimator** — computes model weights, KV cache, and overhead for every quant type using formulas derived from llama.cpp's actual memory allocation
- **Hardware-aware recommendations** — specify your GPU VRAM and get ranked recommendations
- **OOM checker** — see at a glance which quant types fit your hardware
- **HTML dashboard** — interactive Plotly charts: memory waterfall, quant comparison, OOM frontier, context scaling
- **37 quant types** — full spectrum from F32 to TQ1_0, including K-quants, IQ quants, and experimental types

## Project structure

```
1.4-gguf-quant-explorer/
├── gguf_reader/
│   ├── parser.py       # Raw GGUF binary parser (struct-level)
│   └── models.py       # QuantType, GGUFModel, TensorInfo data models
├── quant_planner/
│   ├── estimator.py    # Memory formulas + HardwareProfile + recommendations
│   └── charts.py       # Plotly chart generators + HTML report builder
├── cli.py              # 7 CLI commands with rich tables
├── data/
│   └── benchmarks.json # Cached perplexity benchmarks for 8 model families
├── tests/
│   └── test_quant_explorer.py  # 14 tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt
```

### Commands

| Command | Description |
|---------|-------------|
| `inspect <file.gguf>` | Show model metadata (architecture, dimensions, file type) |
| `compare <file.gguf>` | Compare memory for all quant types |
| `recommend <file.gguf>` | Recommend best quants for your hardware |
| `oom <file.gguf>` | Show which quants fit in VRAM |
| `dashboard <file.gguf>` | Generate an interactive HTML report |
| `list-quants` | List all 37 known quant types |
| `hardware-list` | List built-in hardware profiles |

### Examples

**Inspect a model:**
```bash
python -m cli inspect path/to/model.gguf
```

**Compare memory across all quant types:**
```bash
python -m cli compare path/to/model.gguf --context 8192
```

**Get recommendations for an RTX 3090:**
```bash
python -m cli recommend path/to/model.gguf --hardware rtx3090-24gb
```

**Check what fits in 12 GB VRAM:**
```bash
python -m cli oom path/to/model.gguf --hardware 12
```

**Generate an HTML dashboard:**
```bash
python -m cli dashboard path/to/model.gguf --hardware rtx4090-24gb -o report.html
```

### Hardware profiles

| Name | VRAM | Type |
|------|------|------|
| `rtx3060-12gb` | 12 GB | Dedicated |
| `rtx3090-24gb` | 24 GB | Dedicated |
| `rtx4090-24gb` | 24 GB | Dedicated |
| `rtx5090-32gb` | 32 GB | Dedicated |
| `a100-80gb` | 80 GB | Dedicated |
| `m1-8gb` | 8 GB | Unified (Apple Silicon) |
| `m2-24gb` | 24 GB | Unified (Apple Silicon) |
| `m4max-128gb` | 128 GB | Unified (Apple Silicon) |

Or specify VRAM directly: `--hardware 16`

## Memory formulas

```
model_weights  = num_params * bits_per_weight / 8 * overhead_factor
kv_cache       = 2 * n_layers * n_kv_heads * head_dim * ctx_len * 2 (F16)
overhead       = 5-15% of weights (activations + scratch buffers)
total          = model_weights + kv_cache + overhead
```

- K-quants: ~2% overhead for block metadata
- IQ quants: ~1.5% overhead for importance maps
- KV cache is always stored in FP16 (2 bytes per element)

## Supported quant types

37 types across 4 families:

- **Float**: F32, F16, BF16, F64
- **Standard**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
- **K-quants** (block-aware): Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
- **Importance quants**: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS
- **Experimental**: TQ1_0, TQ2_0, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8, IQ4_NL_4_4, IQ4_NL_4_8, IQ4_NL_8_8
- **Integer**: I8, I16, I32, I64

## Tests

```bash
python tests/test_quant_explorer.py
```

14 tests covering: quant type metadata, bits-per-weight, quality scores, K-quant/IQ detection, memory estimation for multiple architectures (7B, 70B, MoE), OOM checking, hardware profiles, context scaling, and benchmark data validation.

## Benchmark data

Cached perplexity benchmarks for 8 model families across common quant types:
- Llama 3.1 (8B, 70B, 405B)
- Mistral 7B
- Qwen 2.5 (7B, 32B)
- Gemma 2 9B
- DeepSeek V2 Lite (16B MoE)

## Key learning outcomes

- How GGUF metadata encodes model architecture and quantisation info
- How quantization type affects memory across model dimensions
- The practical memory/quality tradeoff space for 37 quant types
- Why KV cache dominates memory at long contexts regardless of weight quant
- How K-quants and importance quants differ from standard block quants
