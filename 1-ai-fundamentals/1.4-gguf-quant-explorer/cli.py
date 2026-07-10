#!/usr/bin/env python3
"""
GGUF Quant Explorer CLI.

Commands:
  inspect     <file.gguf>          — Show model metadata
  compare     <file.gguf>          — Compare all quant types
  recommend   <file.gguf>          — Recommend best quants for hardware
  oom         <file.gguf>          — Show which quants fit VRAM
  list-quants                     — List all known quant types
  dashboard   <file.gguf>          — Generate HTML report
  hardware-list                   — List available hardware profiles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from gguf_reader.parser import GGUFParser
from gguf_reader.models import GGUFModel, QuantType
from quant_planner.estimator import (
    HardwareProfile,
    MemoryEstimate,
    QuantEstimator,
)
from quant_planner.charts import generate_report_html

console = Console() if RICH_AVAILABLE else None


# ── Hardware profiles ─────────────────────────────────────────────

HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "rtx3060-12gb": HardwareProfile.rtx_3060_12gb(),
    "rtx3090-24gb": HardwareProfile.rtx_3090_24gb(),
    "rtx4090-24gb": HardwareProfile.rtx_4090_24gb(),
    "rtx5090-32gb": HardwareProfile.rtx_5090_32gb(),
    "a100-80gb": HardwareProfile.a100_80gb(),
    "m1-8gb": HardwareProfile.macbook_m1_8gb(),
    "m2-24gb": HardwareProfile.macbook_m2_24gb(),
    "m4max-128gb": HardwareProfile.macbook_m4_max_128gb(),
}


def _resolve_hardware(name: str) -> HardwareProfile:
    """Resolve hardware profile by name or custom VRAM value."""
    if name in HARDWARE_PROFILES:
        return HARDWARE_PROFILES[name]
    try:
        vram = float(name)
        return HardwareProfile(f"Custom {vram}GB", total_vram_gb=vram)
    except ValueError:
        console.print(f"[red]Unknown hardware profile: {name}[/red]")
        console.print(f"Available: {', '.join(HARDWARE_PROFILES.keys())}")
        sys.exit(1)


def _print_model_summary(model: GGUFModel):
    """Print a rich summary of the model."""
    if not RICH_AVAILABLE:
        print(f"Model: {model.summary()}")
        return

    console.print()
    console.print(Panel.fit(
        f"[bold]{model.model_name}[/bold]\n"
        f"[dim]{model.file_path}[/dim]",
        title="📋 GGUF Model",
    ))

    # Metadata table
    meta_table = Table(title="Metadata", box=box.SIMPLE, title_style="bold")
    meta_table.add_column("Property", style="cyan")
    meta_table.add_column("Value", style="white")
    meta_table.add_column("Notes", style="dim")

    meta_table.add_row("Architecture", model.architecture, "")
    meta_table.add_row("File size", f"{model.model_size_gb:.2f} GB", "")
    meta_table.add_row("File type", str(model.file_type), str(model.current_quant.name))
    meta_table.add_row("Parameters", f"{model.param_count_b:.2f}B", "")

    if model.is_moe:
        meta_table.add_row("MoE", f"{model.expert_count}E{model.expert_used_count}K", "")

    meta_table.add_row("Embedding dim", f"{model.embedding_dim:,}", "")
    meta_table.add_row("Layers", str(model.block_count), "")
    meta_table.add_row("Heads", f"{model.head_count}", f"KV: {model.head_count_kv}")
    meta_table.add_row("FFN dim", f"{model.feed_forward_dim:,}", "")
    meta_table.add_row("Vocab size", f"{model.vocab_size:,}", "")
    meta_table.add_row("Context length", f"{model.context_length:,}", "")
    meta_table.add_row("Tensors", str(model.tensor_count), "")
    meta_table.add_row("Metadata KVs", str(model.metadata_kv_count), "")

    console.print(meta_table)
    console.print()


def _print_quant_table(
    estimates: List[MemoryEstimate],
    title: str = "Quantisation Estimates",
):
    """Print a table of quantisation estimates."""
    if not RICH_AVAILABLE:
        print(f"\n{title}")
        print(f"{'Quant':<12} {'Weights':>10} {'KV':>10} {'Total':>10} {'Qual':>6} {'Fits':>6}")
        print("-" * 60)
        for e in sorted(estimates, key=lambda x: x.total_gb):
            fits = "✓" if e.fits_in_vram else "✗"
            print(f"{e.quant_type.name:<12} {e.model_weights_gb:>8.2f}GB "
                  f"{e.kv_cache_gb:>8.2f}GB {e.total_gb:>8.2f}GB "
                  f"{e.quality_score*100:>5.0f}% {fits:>6}")
        return

    table = Table(title=title, box=box.SIMPLE, title_style="bold")
    table.add_column("Quant", style="cyan", no_wrap=True)
    table.add_column("BPW", style="dim")
    table.add_column("Weights", justify="right")
    table.add_column("KV Cache", justify="right")
    table.add_column("Overhead", justify="right")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Quality", justify="right")
    table.add_column("Fits", justify="center")

    for e in sorted(estimates, key=lambda x: x.total_gb):
        qt = e.quant_type
        color = "green" if e.fits_in_vram else "red"
        fits_text = "✓" if e.fits_in_vram else "✗"
        table.add_row(
            qt.name,
            f"{qt.bits_per_weight:.2f}",
            f"{e.model_weights_gb:.2f} GB",
            f"{e.kv_cache_gb:.2f} GB",
            f"{e.overhead_gb:.2f} GB",
            f"[{color}]{e.total_gb:.2f} GB[/{color}]",
            f"{e.quality_score*100:.0f}%",
            f"[{color}]{fits_text}[/{color}]",
        )

    console.print(table)
    console.print()


# ── Commands ──────────────────────────────────────────────────────

def cmd_inspect(args: argparse.Namespace):
    """Inspect a GGUF file and show metadata."""
    model = GGUFParser.parse(args.file)
    _print_model_summary(model)

    # Show raw metadata in JSON mode
    if args.json:
        print(json.dumps(model.metadata, indent=2))


def cmd_compare(args: argparse.Namespace):
    """Compare memory across all quant types."""
    model = GGUFParser.parse(args.file)
    estimator = QuantEstimator(model)
    estimates = estimator.estimate_all(context_length=args.context)
    _print_model_summary(model)

    if args.hardware:
        hw = _resolve_hardware(args.hardware)
        for e in estimates:
            vram = hw.total_vram_gb if not hw.is_apple_silicon else hw.total_ram_gb
            e.fits_in_vram = e.total_gb <= vram * 0.9

    _print_quant_table(estimates, "Memory Estimates by Quant Type")


def cmd_recommend(args: argparse.Namespace):
    """Recommend best quant types for hardware."""
    model = GGUFParser.parse(args.file)
    hw = _resolve_hardware(args.hardware)
    estimator = QuantEstimator(model)
    estimates = estimator.recommend(hw, context_length=args.context, min_quality=args.min_quality / 100)

    _print_model_summary(model)
    console.print(f"[bold]Target:[/bold] {hw.name} (VRAM: {hw.total_vram_gb} GB)")
    console.print()

    if not estimates:
        console.print("[red]No quant types fit in the available VRAM.[/red]")
        console.print("Try a smaller model or reduce context length.")
        return

    _print_quant_table(estimates, f"Recommended Quant Types for {hw.name}")

    # Top pick
    best = estimates[0]
    console.print(Panel.fit(
        f"[bold green]Best choice:[/bold green] [cyan]{best.quant_type.name}[/cyan]\n"
        f"  Total: {best.total_gb:.2f} GB · "
        f"Quality: {best.quality_score*100:.0f}% · "
        f"Weights: {best.model_weights_gb:.2f} GB",
        title="🏆 Top Recommendation",
    ))


def cmd_oom(args: argparse.Namespace):
    """Check which quants fit in VRAM."""
    model = GGUFParser.parse(args.file)
    hw = _resolve_hardware(args.hardware)
    estimator = QuantEstimator(model)
    estimates = estimator.check_oom(hw, context_length=args.context)

    _print_model_summary(model)
    console.print(f"[bold]Target:[/bold] {hw.name} ({hw.total_vram_gb} GB)")
    console.print()

    fitting = [e for e in estimates if e.fits_in_vram]
    not_fitting = [e for e in estimates if not e.fits_in_vram]

    if fitting:
        _print_quant_table(fitting, "✅ Fits in VRAM")
    if not_fitting:
        _print_quant_table(not_fitting, "❌ Does NOT fit")

    console.print(f"\n[dim]{len(fitting)} quants fit, {len(not_fitting)} do not[/dim]")


def cmd_dashboard(args: argparse.Namespace):
    """Generate an HTML dashboard."""
    model = GGUFParser.parse(args.file)
    estimator = QuantEstimator(model)
    estimates = estimator.estimate_all(context_length=args.context)

    hw = None
    if args.hardware:
        hw = _resolve_hardware(args.hardware)
        for e in estimates:
            vram = hw.total_vram_gb if not hw.is_apple_silicon else hw.total_ram_gb
            e.fits_in_vram = e.total_gb <= vram * 0.9

    output = args.output or f"quant-report-{Path(args.file).stem}.html"
    generate_report_html(model, estimates, hw, output)
    console.print(f"[green]✓ Report generated:[/green] {output}")


def cmd_list_quants(args: argparse.Namespace):
    """List all known quant types."""
    if not RICH_AVAILABLE:
        print("Known quantisation types:")
        for qt in QuantType:
            print(f"  {qt.value:>3}  {qt.name:<12}  {qt.bits_per_weight:>5.2f} bpw  {qt.description}")
        return

    table = Table(title="Known Quantisation Types", box=box.SIMPLE, title_style="bold")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("BPW", justify="right")
    table.add_column("K-Quant", justify="center")
    table.add_column("IQ", justify="center")
    table.add_column("Quality", justify="right")
    table.add_column("Description")

    for qt in sorted(QuantType, key=lambda x: x.value):
        table.add_row(
            str(qt.value),
            qt.name,
            f"{qt.bits_per_weight:.2f}",
            "✓" if qt.is_k_quant else "",
            "✓" if qt.is_i_quant else "",
            f"{qt.relative_quality*100:.0f}%",
            qt.description,
        )

    console.print(table)


def cmd_hardware_list(args: argparse.Namespace):
    """List available hardware profiles."""
    if not RICH_AVAILABLE:
        print("Available hardware profiles:")
        for name, hw in HARDWARE_PROFILES.items():
            mem_type = "unified" if hw.is_apple_silicon else "dedicated VRAM"
            print(f"  {name:<18}  {hw.gpu_name:<20}  {hw.total_vram_gb:>5} GB {mem_type}")
        return

    table = Table(title="Hardware Profiles", box=box.SIMPLE, title_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("GPU", style="white")
    table.add_column("VRAM", justify="right")
    table.add_column("Type", style="dim")
    table.add_column("Notes")

    for name, hw in HARDWARE_PROFILES.items():
        mem_type = "Unified" if hw.is_apple_silicon else "Dedicated"
        vram = hw.total_vram_gb if not hw.is_apple_silicon else hw.total_ram_gb
        table.add_row(name, hw.gpu_name, f"{vram} GB", mem_type, "")

    console.print(table)
    console.print("\nYou can also specify VRAM directly: [cyan]--hardware 16[/cyan]")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GGUF Quant Explorer — understand quantisation tradeoffs",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON where applicable")
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p = sub.add_parser("inspect", help="Show model metadata")
    p.add_argument("file", help="Path to .gguf file")
    p.set_defaults(func=cmd_inspect)

    # compare
    p = sub.add_parser("compare", help="Compare memory across quant types")
    p.add_argument("file", help="Path to .gguf file")
    p.add_argument("--context", type=int, default=4096, help="Context length")
    p.add_argument("--hardware", help="Hardware profile or VRAM in GB")
    p.set_defaults(func=cmd_compare)

    # recommend
    p = sub.add_parser("recommend", help="Recommend best quants for hardware")
    p.add_argument("file", help="Path to .gguf file")
    p.add_argument("--hardware", default="rtx4090-24gb", help="Hardware profile")
    p.add_argument("--context", type=int, default=4096, help="Context length")
    p.add_argument("--min-quality", type=float, default=30,
                   help="Minimum quality threshold (0-100)")
    p.set_defaults(func=cmd_recommend)

    # oom
    p = sub.add_parser("oom", help="Check which quants fit in VRAM")
    p.add_argument("file", help="Path to .gguf file")
    p.add_argument("--hardware", default="rtx4090-24gb", help="Hardware profile")
    p.add_argument("--context", type=int, default=4096, help="Context length")
    p.set_defaults(func=cmd_oom)

    # dashboard
    p = sub.add_parser("dashboard", help="Generate HTML report")
    p.add_argument("file", help="Path to .gguf file")
    p.add_argument("--hardware", help="Hardware profile")
    p.add_argument("--context", type=int, default=4096, help="Context length")
    p.add_argument("-o", "--output", help="Output HTML path")
    p.set_defaults(func=cmd_dashboard)

    # list-quants
    p = sub.add_parser("list-quants", help="List all quant types")
    p.set_defaults(func=cmd_list_quants)

    # hardware-list
    p = sub.add_parser("hardware-list", help="List hardware profiles")
    p.set_defaults(func=cmd_hardware_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
