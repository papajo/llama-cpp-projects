"""Live demo for Category 1 — AI Fundamentals with real LLM inference."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.integrations import ChatAdapter


def demo_kv_cache():
    adapter = ChatAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  1.1 KV Cache Visualizer [{tag}]")
    resp = adapter.call_llm(
        "Explain what the KV cache is in transformer models in 2-3 sentences.",
        max_tokens=100,
    )
    print(f"     {resp}")


def demo_spec_decoding():
    adapter = ChatAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  1.2 Speculative Decoding Benchmark [{tag}]")
    resp = adapter.call_llm(
        "What is speculative decoding and how does it speed up inference?",
        max_tokens=100,
    )
    print(f"     {resp}")


def demo_constrained_decoding():
    adapter = ChatAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  1.3 Constrained Decoding Playground [{tag}]")
    resp = adapter.call_llm(
        "Generate a JSON object with keys: name (string), age (integer), skills (array of strings). "
        "Return ONLY valid JSON.",
        temperature=0.1,
        max_tokens=150,
    )
    print(f"     JSON output: {resp}")


def demo_gguf_quant():
    print("\n  1.4 GGUF Quant Explorer [no LLM needed — reads GGUF file metadata]")


def demo_sampling_params():
    adapter = ChatAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  1.5 Sampling Param Explorer [{tag}]")
    resp = adapter.call_llm(
        "List the most important sampling parameters for LLM text generation and their effects.",
        max_tokens=150,
    )
    print(f"     {resp}")


def demo_prompt_cache():
    adapter = ChatAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  1.6 Prompt Cache Benchmark [{tag}]")
    resp = adapter.call_llm(
        "How does prompt caching improve LLM serving performance?",
        max_tokens=100,
    )
    print(f"     {resp}")


def demo():
    demo_kv_cache()
    demo_spec_decoding()
    demo_constrained_decoding()
    demo_gguf_quant()
    demo_sampling_params()
    demo_prompt_cache()


if __name__ == "__main__":
    demo()
