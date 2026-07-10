"""
CLI interface for the Speculative Decoding Benchmark Suite.

Usage:
    spec-bench list-strategies          # list all strategies with descriptions
    spec-bench run <model>              # run full benchmark
    spec-bench run-strategy <model> <strategy>  # test one strategy
    spec-bench baseline <model>         # baseline only
    spec-bench report <results.json>    # generate report from saved results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _bold(s: str) -> str: return f"\033[1m{s}\033[0m"
def _green(s: str) -> str: return f"\033[92m{s}\033[0m"
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m"
def _cyan(s: str) -> str: return f"\033[96m{s}\033[0m"


def list_strategies(args: argparse.Namespace) -> None:
    """Display all 8 strategies with descriptions and parameters."""
    from .configs import ALL_STRATEGIES, strategies_by_family

    families = strategies_by_family()
    for family_name, strats in families.items():
        print(f"\n{_bold(family_name.upper())} strategies:")
        print(f"  {'-'*60}")
        for s in strats:
            print(f"  {_green(s.label):<22} {s.flag_key}")
            print(f"  {'':22} {s.description}")
            print(f"  {'':22} Params: {s.param_summary()}")
            print()


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full benchmark suite."""
    from harness.configs import ALL_STRATEGIES
    from harness.runner import BenchmarkRunner
    from prompts import ALL_PROMPTS
    from report_generator import BenchmarkAnalysis, generate_report

    model_path = args.model
    draft_path = args.draft_model
    output_dir = args.output or "results"

    print(_bold(f"\n{'='*60}"))
    print(_bold(f"  Speculative Decoding Benchmark"))
    print(_bold(f"  Target model: {model_path}"))
    if draft_path:
        print(f"  Draft model:  {draft_path}")
    print(f"  Strategies:   {len(ALL_STRATEGIES)}")
    print(f"  Prompts:      {len(ALL_PROMPTS)}")
    print(f"  Runs/prompt:  {args.runs}")
    print(_bold(f"{'='*60}"))

    runner = BenchmarkRunner(
        model_path=model_path,
        draft_model_path=draft_path,
        server_binary=args.server,
        server_port=args.port,
        n_gpu_layers=args.gpu_layers,
        context_size=args.ctx_size,
        verbose=args.verbose,
    )

    try:
        all_results = runner.run_full_benchmark(
            prompts=ALL_PROMPTS,
            strategies=ALL_STRATEGIES if not args.strategy else [
                s for s in ALL_STRATEGIES
                if s.label.lower() == args.strategy.lower()
                or s.strategy.value.lower() == args.strategy.lower()
            ],
            n_predict=args.n_predict,
            temperature=args.temperature,
            n_runs=args.runs,
        )
    finally:
        runner.cleanup()

    # Save raw results
    results_path = Path(output_dir) / "raw_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_data = [
        {"strategy": r.strategy, "prompt_name": r.prompt_name,
         "metrics": r.metrics.as_dict()}
        for r in all_results
    ]
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"\n💾 Raw results saved to {results_path}")

    # Analyse and generate report
    analysis = BenchmarkAnalysis.from_results(all_results)
    generate_report(analysis, output_dir)

    # Print key findings
    print(f"\n{_bold('Key Findings:')}")
    for rec in analysis.recommendations():
        print(f"  • {rec}")


def run_baseline(args: argparse.Namespace) -> None:
    """Run only the baseline (no speculation)."""
    from harness.runner import BenchmarkRunner
    from prompts import ALL_PROMPTS
    from report_generator import BenchmarkAnalysis

    runner = BenchmarkRunner(
        model_path=args.model,
        server_binary=args.server,
        server_port=args.port,
        n_gpu_layers=args.gpu_layers,
        context_size=args.ctx_size,
        verbose=args.verbose,
    )

    try:
        results = runner.run_baseline(
            prompts=ALL_PROMPTS,
            n_predict=args.n_predict,
            temperature=args.temperature,
            n_runs=args.runs,
        )
    finally:
        runner.cleanup()

    print(f"\n{_bold('Baseline Results:')}")
    for r in results:
        m = r.metrics
        print(f"  {r.prompt_name:<25} {m.tokens_per_second:>8.1f} tok/s  "
              f"({m.total_tokens_generated} tokens in {m.wall_time_s:.1f}s)")


def generate_report_cmd(args: argparse.Namespace) -> None:
    """Generate a report from saved results."""
    from harness.metrics import BenchmarkMetrics
    from report_generator import BenchmarkAnalysis, generate_report

    # Load raw results
    path = Path(args.results)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text())

    # Reconstruct BenchmarkResult-like objects
    class FakeResult:
        def __init__(self, d):
            self.strategy = d["strategy"]
            self.prompt_name = d["prompt_name"]
            self.metrics = BenchmarkMetrics(**d["metrics"])

    results = [FakeResult(d) for d in data]
    analysis = BenchmarkAnalysis.from_results(results)
    generate_report(analysis, args.output or "results")
    print(f"\nReport generated from {path}")


def run_strategy(args: argparse.Namespace) -> None:
    """Run a single strategy (for focused testing)."""
    from harness.configs import get_strategy, ALL_STRATEGIES
    from harness.runner import BenchmarkRunner
    from prompts import ALL_PROMPTS

    strategy = get_strategy(args.strategy)
    if strategy is None:
        print(f"Error: Unknown strategy '{args.strategy}'", file=sys.stderr)
        print(f"  Available: {[s.label for s in ALL_STRATEGIES]}")
        sys.exit(1)

    runner = BenchmarkRunner(
        model_path=args.model,
        draft_model_path=args.draft_model,
        server_binary=args.server,
        server_port=args.port,
        n_gpu_layers=args.gpu_layers,
        context_size=args.ctx_size,
        verbose=args.verbose,
    )

    try:
        results = runner.run_strategy(
            strategy=strategy,
            prompts=ALL_PROMPTS,
            n_predict=args.n_predict,
            temperature=args.temperature,
            n_runs=args.runs,
        )
    finally:
        runner.cleanup()

    print(f"\n{_bold(f'Strategy Results: {strategy.label}')}")
    for r in results:
        m = r.metrics
        print(f"  {r.prompt_name:<25} {m.tokens_per_second:>8.1f} tok/s  "
              f"accept={m.acceptance_rate:.1%}  "
              f"({m.total_tokens_generated} tokens)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spec-bench",
        description="Speculative Decoding Benchmark Suite for llama.cpp",
    )

    # Global server options
    p.add_argument("--server", default="llama-server",
                   help="Path to llama-server binary")
    p.add_argument("--port", type=int, default=18080,
                   help="Server port (default: 18080)")
    p.add_argument("--gpu-layers", type=int, default=-1,
                   help="GPU layers (-1 = all)")
    p.add_argument("--ctx-size", type=int, default=4096,
                   help="Context size (default: 4096)")
    p.add_argument("--n-predict", type=int, default=200,
                   help="Tokens to generate per completion")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature")
    p.add_argument("--runs", type=int, default=1,
                   help="Runs per prompt (for statistical significance)")
    p.add_argument("-o", "--output", default="results",
                   help="Output directory")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show server output")

    sub = p.add_subparsers(dest="command", required=True)

    # list-strategies
    sub.add_parser("list-strategies", help="List all spec decoding strategies")

    # run (full benchmark)
    run_p = sub.add_parser("run", help="Run full benchmark against all strategies")
    run_p.add_argument("model", type=str, help="Path to target GGUF model")
    run_p.add_argument("--draft-model", type=str, default=None,
                       help="Path to draft GGUF model (required for draft strategies)")
    run_p.add_argument("--strategy", type=str, default=None,
                       help="Run only this strategy (label or value)")

    # run-strategy (single)
    rs_p = sub.add_parser("run-strategy", help="Run a single strategy")
    rs_p.add_argument("model", type=str)
    rs_p.add_argument("strategy", type=str, help="Strategy label (e.g. 'Draft-Simple')")
    rs_p.add_argument("--draft-model", type=str, default=None)

    # baseline
    bl_p = sub.add_parser("baseline", help="Run baseline (no speculation)")
    bl_p.add_argument("model", type=str)

    # report
    rep_p = sub.add_parser("report", help="Generate report from saved results")
    rep_p.add_argument("results", type=str, help="Path to raw_results.json")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-strategies":
        list_strategies(args)
    elif args.command == "run":
        run_benchmark(args)
    elif args.command == "run-strategy":
        run_strategy(args)
    elif args.command == "baseline":
        run_baseline(args)
    elif args.command == "report":
        generate_report_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
