"""Live demo for Category 7 — Bonus projects with real LLM inference."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.integrations import ChatAdapter, EmbedAdapter


def demo_cpu_affinity():
    print("\n  7.1 CPU Affinity Tuning [no LLM needed — system-level benchmarking]")


def demo_inference_profiling():
    adapter = ChatAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  7.2 Inference Profiling Dashboard [{tag}]")
    import time
    t0 = time.time()
    resp = adapter.call_llm("Say 'hello world'", max_tokens=10)
    elapsed = time.time() - t0
    print(f"     Inference time: {elapsed:.2f}s")
    print(f"     Response: {resp}")


def demo_numa_deployment():
    print("\n  7.3 NUMA Deployment Guide [no LLM needed — deployment topology]")


def demo_prompt_cache_hit():
    adapter = ChatAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  7.4 Prompt Cache Hit Rate [{tag}]")
    resp = adapter.call_llm("What is the meaning of life?", max_tokens=50)
    print(f"     Response: {resp}")


def demo_sleep_wake_profiler():
    print("\n  7.5 Sleep-Wake Cost Profiler [no LLM needed — power profiling]")


def demo():
    demo_cpu_affinity()
    demo_inference_profiling()
    demo_numa_deployment()
    demo_prompt_cache_hit()
    demo_sleep_wake_profiler()


if __name__ == "__main__":
    demo()
