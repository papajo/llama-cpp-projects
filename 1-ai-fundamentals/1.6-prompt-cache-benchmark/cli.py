#!/usr/bin/env python3
"""
Prompt Cache Benchmark CLI.

Commands:
  list-scenarios       — List available benchmark scenarios
  run <scenario>       — Run a specific scenario
  run-all              — Run all scenarios
  report               — Generate reports from results file
  health               — Check llama-server connectivity
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .harness.runner import CacheBenchmarkRunner, get_scenario_list
from .harness.scenarios import get_scenario
from .report.report import generate_markdown, generate_html, generate_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("cache-benchmark")


def cmd_list_scenarios(args: argparse.Namespace):
    """List available scenarios."""
    scenarios = get_scenario_list()
    if not scenarios:
        print("No scenarios available.")
        return

    print(f"\nAvailable scenarios ({len(scenarios)}):")
    print(f"{'Name':<35} {'Prompts':<8} {'Avg Len':<10} {'Type':<10} Description")
    print("-" * 100)
    for s in scenarios:
        print(
            f"{s['name']:<35} {s['num_prompts']:<8} "
            f"{s['avg_prompt_len']:<10} {s['cache_type']:<10} "
            f"{s['description']:<50}"
        )
    print()


def cmd_health(args: argparse.Namespace):
    """Check llama-server connection."""
    runner = CacheBenchmarkRunner(server_url=args.server)
    ok = runner.health_check()
    if ok:
        print(f"✅ llama-server at {args.server} is healthy")
    else:
        print(f"❌ Cannot connect to llama-server at {args.server}")
        print("   Start it with: llama-server -m <model> --host 0.0.0.0")
        sys.exit(1)


def cmd_run(args: argparse.Namespace):
    """Run a specific scenario."""
    runner = CacheBenchmarkRunner(
        server_url=args.server,
        n_predict=args.n_predict,
    )

    if not runner.health_check():
        print(f"❌ Cannot connect to llama-server at {args.server}")
        sys.exit(1)

    scenario = get_scenario(args.scenario)
    if scenario is None:
        print(f"Unknown scenario: {args.scenario}")
        print("Use 'list-scenarios' to see available scenarios.")
        sys.exit(1)

    print(f"Running scenario: {scenario.name}")
    print(f"  {scenario.description}")
    print(f"  {len(scenario.prompts)} prompts, up to {args.n_predict} tokens each")
    print()

    result = runner.run_scenario(scenario, warmup_runs=args.warmup)

    print(f"\nResults for '{scenario.name}':")
    print(f"  Uncached TTFT: {result.avg_ttft_uncached_ms:.1f} ms (avg)")
    print(f"  Cached TTFT:   {result.avg_ttft_cached_ms:.1f} ms (avg)")
    print(f"  Speedup:       {result.speedup_factor:.1f}x")
    print(f"  Reduction:     {result.speedup_pct:.1f}%")

    # Save results
    output = args.output or f"results-{scenario.name}.json"
    data = {scenario.name: result.to_dict()}
    Path(output).write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to: {output}")


def cmd_run_all(args: argparse.Namespace):
    """Run all scenarios."""
    runner = CacheBenchmarkRunner(
        server_url=args.server,
        n_predict=args.n_predict,
    )

    if not runner.health_check():
        print(f"❌ Cannot connect to llama-server at {args.server}")
        sys.exit(1)

    scenario_names = args.scenarios or [s["name"] for s in get_scenario_list()]
    print(f"Running {len(scenario_names)} scenarios...\n")

    all_results = runner.run_all_scenarios(scenario_names, warmup_runs=args.warmup)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<35} {'Uncached':<12} {'Cached':<12} {'Speedup':<10}")
    print("-" * 70)
    for name, r in all_results.items():
        print(
            f"{name:<35} {r.avg_ttft_uncached_ms:<8.1f}ms  "
            f"{r.avg_ttft_cached_ms:<8.1f}ms  "
            f"{r.speedup_factor:.1f}x"
        )

    # Generate reports
    stem = args.output_stem or "cache-benchmark"
    md_path = generate_markdown(all_results, f"{stem}.md")
    html_path = generate_html(all_results, f"{stem}.html")
    json_path = generate_json(all_results, f"{stem}.json")
    print(f"\nReports generated:")
    print(f"  📊 {html_path}")
    print(f"  📝 {md_path}")
    print(f"  📋 {json_path}")


def cmd_report(args: argparse.Namespace):
    """Generate reports from existing results JSON."""
    data = json.loads(Path(args.input).read_text())
    results = data.get("scenarios", data)

    # Reconstruct CacheBenchmarkResult objects
    from .harness.metrics import CacheBenchmarkResult, PromptResult
    parsed: Dict[str, CacheBenchmarkResult] = {}
    for name, rdata in results.items():
        cr = CacheBenchmarkResult(
            scenario_name=rdata.get("scenario", name),
            description=rdata.get("description", ""),
            runs_cached=[],
            runs_uncached=[],
            avg_ttft_cached_ms=rdata.get("avg_ttft_cached_ms", 0),
            avg_ttft_uncached_ms=rdata.get("avg_ttft_uncached_ms", 0),
            median_ttft_cached_ms=rdata.get("median_ttft_cached_ms", 0),
            median_ttft_uncached_ms=rdata.get("median_ttft_uncached_ms", 0),
            speedup_factor=rdata.get("speedup_factor", 0),
            speedup_pct=rdata.get("speedup_pct", 0),
            per_prompt=rdata.get("per_prompt", []),
        )
        parsed[name] = cr

    stem = args.output_stem or Path(args.input).stem
    md_path = generate_markdown(parsed, f"{stem}.md")
    html_path = generate_html(parsed, f"{stem}.html")
    print(f"Reports generated:")
    print(f"  📊 {html_path}")
    print(f"  📝 {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prompt Cache Benchmark — measure TTFT with and without caching",
    )
    parser.add_argument("--server", default=os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080"),
                        help="llama-server URL")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-scenarios", help="List scenarios")
    p.set_defaults(func=cmd_list_scenarios)

    p = sub.add_parser("health", help="Check server")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("run", help="Run a scenario")
    p.add_argument("scenario", help="Scenario name")
    p.add_argument("--n-predict", type=int, default=50)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("-o", "--output", help="Output JSON path")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("run-all", help="Run all scenarios")
    p.add_argument("--n-predict", type=int, default=50)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("scenarios", nargs="*", help="Specific scenarios (default: all)")
    p.add_argument("--output-stem", default="cache-benchmark", help="Output file stem")
    p.set_defaults(func=cmd_run_all)

    p = sub.add_parser("report", help="Generate reports from JSON")
    p.add_argument("input", help="Input JSON results file")
    p.add_argument("--output-stem", help="Output file stem")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
