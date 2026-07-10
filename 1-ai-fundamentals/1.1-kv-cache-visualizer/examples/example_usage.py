#!/usr/bin/env python3
"""
Worked examples for the KV Cache Visualizer.

Shows common use cases:
  1. Analyse Llama-3.1-8B at full 128K context
  2. Compare Mistral-7B vs Qwen-2.5-7B at 32K
  3. Predict OOM for a 24 GB GPU running Llama-3.1-70B
  4. Visualise slot scaling
  5. Generate an HTML dashboard
"""

import sys
import os

# Add parent to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculator.calculator import KVCacheCalculator
from calculator.models import KVQuant
from calculator.presets import ModelPresets

print("=" * 70)
print("  KV Cache Visualizer — Worked Examples")
print("=" * 70)

# ── Example 1: Llama-3.1-8B at full context ──────────────────────
print("\n" + "─" * 70)
print("  1. Llama-3.1-8B — 128K context (f16 KV cache)")
print("─" * 70)

llama = ModelPresets.get("Llama-3.1-8B")
calc = KVCacheCalculator(llama)
result = calc.calculate()

for label, bd in result.breakdowns.items():
    print(f"    {label:8s} → {bd.gb_total:8.2f} GB total  |  "
          f"{bd.miB_per_layer:8.0f} MiB/layer  |  "
          f"{bd.gb_total_all_slots:8.2f} GB all slots")

# ── Example 2: Compare models ────────────────────────────────────
print("\n" + "─" * 70)
print("  2. Model comparison at 32K context (f16)")
print("─" * 70)

for name in ["Mistral-7B-v0.3", "Qwen-2.5-7B", "Gemma-2-9B"]:
    cfg = ModelPresets.get(name)
    c = KVCacheCalculator(cfg)
    r = c.calculate(max_ctx=32768, quants=[KVQuant.F16])
    bd = r.breakdowns["f16"]
    print(f"    {name:22s} → {bd.gb_total:6.2f} GB  "
          f"(layers={cfg.n_layers}, kv_heads={cfg.n_kv_heads}, "
          f"head_dim={cfg.head_dim})")

# ── Example 3: OOM prediction ────────────────────────────────────
print("\n" + "─" * 70)
print("  3. OOM prediction: Llama-3.1-70B on a 24 GB GPU")
print("─" * 70)

llama70 = ModelPresets.get("Llama-3.1-70B")
calc70 = KVCacheCalculator(llama70)
oom = calc70.predict_oom(vram_gb=24)
for r in oom:
    print(f"    {r['quant']:8s} → max ctx: {r['max_ctx_tokens']:>8,}  "
          f"({r['percentage_of_config']:5.1f}% of config)")

# ── Example 4: Slot scaling ──────────────────────────────────────
print("\n" + "─" * 70)
print("  4. Slot scaling: Mistral-7B at 32K, f16 KV cache")
print("─" * 70)

mistral = ModelPresets.get("Mistral-7B-v0.3")
for n_slots in [1, 2, 4, 8]:
    c = KVCacheCalculator(mistral, n_slots=n_slots)
    r = c.calculate(max_ctx=32768, quants=[KVQuant.F16])
    bd = r.breakdowns["f16"]
    print(f"    {n_slots} slot(s) → {bd.gb_total:6.2f} GB (per-slot)"
          f"  |  unified approx: {bd.gb_total_all_slots:6.2f} GB")

# ── Example 5: Savings from KV cache quantization ────────────────
print("\n" + "─" * 70)
print("  5. Savings from KV cache quant: CodeLlama-34B at 16K")
print("─" * 70)

codellama = ModelPresets.get("CodeLlama-34B")
calc_cl = KVCacheCalculator(codellama)
r_cl = calc_cl.calculate(max_ctx=16384)
f16_gb = r_cl.breakdowns["f16"].gb_total
q8_gb = r_cl.breakdowns["q8_0"].gb_total
q4_gb = r_cl.breakdowns["q4_0"].gb_total
print(f"    f16 → {f16_gb:.2f} GB")
print(f"    q8_0 → {q8_gb:.2f} GB  (saves {f16_gb - q8_gb:.2f} GB)")
print(f"    q4_0 → {q4_gb:.2f} GB  (saves {f16_gb - q4_gb:.2f} GB)")

print("\n" + "=" * 70)
print("  Done! Run `python -m calculator.cli show <model>` for details.")
print("=" * 70)
