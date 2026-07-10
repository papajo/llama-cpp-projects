"""Live demo for Category 3 — Prompt Engineering projects with real LLM."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.integrations import PromptEngineeringAdapter


def demo_reasoning_budget():
    adapter = PromptEngineeringAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  3.1 Reasoning Budget Sweep [{tag}]")
    results = adapter.compare(
        "Explain quantum computing in one paragraph.",
        [{"max_tokens": 50}, {"max_tokens": 200}],
    )
    for r in results:
        print(f"     max_tokens={r['kwargs']['max_tokens']} ({r['time_s']}s): {r['text'][:80]}...")


def demo_sampler_ablation():
    adapter = PromptEngineeringAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  3.2 Sampler Ablation Lab [{tag}]")
    results = adapter.compare(
        "Write a tagline for a coffee shop.",
        [{"temperature": 0.1}, {"temperature": 0.9}],
    )
    for r in results:
        print(f"     temp={r['kwargs']['temperature']}: {r['text']}")


def demo_chat_template():
    adapter = PromptEngineeringAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  3.3 Chat Template Tester [{tag}]")
    resp = adapter.call_llm(
        "What is 2+2?",
        system="You are a math tutor. Answer briefly.",
        temperature=0.1,
    )
    print(f"     System: math tutor")
    print(f"     Response: {resp}")


def demo_prompt_chaining():
    adapter = PromptEngineeringAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  3.4 Prompt Chaining Workbench [{tag}]")
    step1 = adapter.call_llm("Generate a topic for a short story.", temperature=0.8)
    print(f"     Step 1 (topic): {step1}")
    step2 = adapter.call_llm(f"Write one paragraph for a story about: {step1}", temperature=0.7)
    print(f"     Step 2 (story): {step2[:150]}...")


def demo_prompt_optimizer():
    adapter = PromptEngineeringAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  3.5 Prompt Optimizer Loop [{tag}]")
    resp = adapter.call_llm(
        "Explain the water cycle briefly.",
        temperature=0.3,
    )
    print(f"     Response: {resp}")


def demo():
    demo_reasoning_budget()
    demo_sampler_ablation()
    demo_chat_template()
    demo_prompt_chaining()
    demo_prompt_optimizer()


if __name__ == "__main__":
    demo()
