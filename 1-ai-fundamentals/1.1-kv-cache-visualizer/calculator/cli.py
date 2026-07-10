"""
CLI interface for the KV Cache Visualizer calculator.

Usage:
    python -m calculator list                       # list known models
    python -m calculator show <model>                # detailed analysis
    python -m calculator compare <model1> <model2>   # side-by-side
    python -m calculator oom <model> --vram 12       # OOM prediction
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .calculator import KVCacheCalculator
from .models import KVQuant
from .presets import ModelPresets


# ── Formatting helpers ──────────────────────────────────────────────

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _cyan(s: str) -> str:
    return f"\033[96m{s}\033[0m"


def _gb(val: float) -> str:
    return f"{val:.2f} GB"


def _mib(val: float) -> str:
    return f"{val:,.0f} MiB"


def _header(text: str, width: int = 60) -> str:
    return f"\n{_bold('═' * width)}\n{_bold(f'  {text}')}\n{_bold('═' * width)}"


# ── Display functions ──────────────────────────────────────────────

def show_model(args: argparse.Namespace) -> None:
    try:
        cfg = ModelPresets.get(args.model)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    calc = KVCacheCalculator(cfg, n_slots=args.slots)
    result = calc.calculate(
        max_ctx=args.ctx or cfg.max_ctx,
        quants=[KVQuant.from_str(q) for q in args.quants.split(",")],
        allocation=args.allocation,
    )

    # ── Model info ──
    print(_header(f"📐  {cfg.name}  —  Architecture"))
    print(f"  {_bold('Layers')}:         {cfg.n_layers}")
    print(f"  {_bold('Q Heads')}:        {cfg.n_heads}")
    print(f"  {_bold('KV Heads')}:       {cfg.n_kv_heads}  (GQA ratio: {cfg.n_kv_groups}:1)")
    print(f"  {_bold('Head dim')}:       {cfg.head_dim}")
    print(f"  {_bold('Hidden dim')}:     {cfg.d_model}")
    print(f"  {_bold('Max ctx')}:        {cfg.max_ctx:,} tokens")
    print(f"  {_bold('MoE')}:            {'Yes — ' + str(cfg.n_experts) + ' experts, ' + str(cfg.n_active_experts) + ' active' if cfg.is_moe else 'No (dense)'}")
    print(f"  {_bold('Slots')}:          {result.n_slots}")
    print(f"  {_bold('Allocation')}:     {result.allocation}")

    # ── Breakdown table ──
    print(_header(f"💾  KV Cache at {result.max_ctx:,} tokens  ({result.allocation})"))
    header = f"  {'Quant':<8} {'Per-Layer':<14} {'Total':<14} {'All Slots':<14} {'Savings':<14}"
    sep = f"  {'-'*8} {'-'*14} {'-'*14} {'-'*14} {'-'*14}"
    print(header)
    print(sep)

    f16_gb = None
    for label, b in result.breakdowns.items():
        savings = ""
        if label == "f16":
            f16_gb = b.gb_total
            savings = "baseline"
        elif f16_gb is not None:
            saved = f16_gb - b.gb_total
            savings = f"save {saved:.2f} GB"
        print(
            f"  {label:<8} {_mib(b.miB_per_layer):<14} {_gb(b.gb_total):<14}"
            f" {_gb(b.gb_total_all_slots):<14} {savings:<14}"
        )

    # ── Recommendations ──
    if result.recommendations:
        print(_header("💡  Recommendations"))
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")

    # ── JSON output ──
    if args.json:
        print()
        print(json.dumps(result.as_dict(), indent=2))


def compare_models(args: argparse.Namespace) -> None:
    models = args.models
    cfgs = []
    for name in models:
        try:
            cfgs.append(ModelPresets.get(name))
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    compare_ctx = min(cfg.max_ctx for cfg in cfgs)
    print(_header(f"📊  KV Cache Comparison at {compare_ctx:,} tokens  (f16)"))
    # header
    print(f"  {'Model':<22} {'Layers':<8} {'KV Heads':<10} {'Ctx':<10} {'Total KV':<14}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*14}")

    for cfg in cfgs:
        calc = KVCacheCalculator(cfg, n_slots=1)
        actual_ctx = min(compare_ctx, cfg.max_ctx)
        b = calc.calculate(max_ctx=actual_ctx, quants=[KVQuant.F16], allocation="per-slot")
        bd = b.breakdowns["f16"]
        print(
            f"  {cfg.name:<22} {cfg.n_layers:<8} {cfg.n_kv_heads:<10}"
            f" {cfg.max_ctx:<10,} {_gb(bd.gb_total):<14}"
        )


def list_models(args: argparse.Namespace) -> None:
    print(_header("📋  Known Model Architectures"))
    print(f"  {'Name':<30} {'Layers':<8} {'KV Hd':<8} {'Ctx':<10} {'Hidden':<10}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for cfg in ModelPresets.all():
        print(
            f"  {cfg.name:<30} {cfg.n_layers:<8} {cfg.n_kv_heads:<8}"
            f" {cfg.max_ctx:<10,} {cfg.d_model:<10}"
        )


def predict_oom(args: argparse.Namespace) -> None:
    try:
        cfg = ModelPresets.get(args.model)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    calc = KVCacheCalculator(cfg, n_slots=args.slots)
    results = calc.predict_oom(
        vram_gb=args.vram,
        quants=[KVQuant.from_str(q) for q in args.quants.split(",")],
        model_weights_gb=args.weights,
    )

    print(_header(f"🔮  OOM Prediction — {cfg.name}  —  {args.vram} GB VRAM"))
    print(f"  {'Quant':<8} {'Weights':<12} {'KV Budget':<12} {'Max Ctx':<12} {'% of Config':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for r in results:
        pct = r["percentage_of_config"]
        pct_str = _green(f"{pct}%") if pct >= 100 else _yellow(f"{pct}%") if pct >= 50 else _red(f"{pct}%")
        print(
            f"  {r['quant']:<8} {_gb(r['model_weights_gb']):<12}"
            f" {_gb(r['kv_cache_budget_gb']):<12} {r['max_ctx_tokens']:<12,} {pct_str:<12}"
        )
    print(_yellow("\n  ⚠  Weight estimate is approximate.  Use --weights for exact value."))

    if args.json:
        print()
        print(json.dumps(results, indent=2))


# ── Main ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kv-cache-calculator",
        description="KV Cache Memory Calculator & Visualizer for llama.cpp",
        epilog="See https://github.com/your-org/kv-cache-visualizer for full docs.",
    )
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    p.add_argument("-s", "--slots", type=int, default=1, help="Number of server slots (default: 1)")
    p.add_argument("-q", "--quants", type=str, default="f16,q8_0,q4_0", help="Comma-separated KV quants")
    p.add_argument("--allocation", choices=["per-slot", "unified"], default="per-slot")
    p.add_argument("--ctx", type=int, default=None, help="Context length to evaluate (default: model max)")

    sub = p.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List known model architectures")

    # show
    sp = sub.add_parser("show", help="Show detailed KV cache analysis for a model")
    sp.add_argument("model", type=str, help="Model name (e.g. 'Llama-3.1-8B')")

    # compare
    cp = sub.add_parser("compare", help="Compare KV cache across models")
    cp.add_argument("models", type=str, nargs="+", help="Model names")

    # oom
    op = sub.add_parser("oom", help="Predict max context before OOM given VRAM budget")
    op.add_argument("model", type=str)
    op.add_argument("--vram", type=float, required=True, help="Available GPU VRAM in GB")
    op.add_argument("--weights", type=float, default=None, help="Exact model weights memory in GB")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        list_models(args)
    elif args.command == "show":
        show_model(args)
    elif args.command == "compare":
        compare_models(args)
    elif args.command == "oom":
        predict_oom(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
