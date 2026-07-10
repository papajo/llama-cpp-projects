"""Live demo for Category 2 — LangChain projects with real LLM inference."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.integrations import LangChainAdapter


def demo_router():
    """Demo 2.1: LangChain Router & Failover with real backends."""
    adapter = LangChainAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  2.1 LangChain Router & Failover [{tag}]")

    prompt = "What is the capital of Australia?"
    resp = adapter.call_llm(prompt, temperature=0.3)
    print(f"     Q: {prompt}")
    print(f"     A: {resp}")


def demo_grammar():
    """Demo 2.2: Grammar-Structured Output with JSON constraint."""
    adapter = LangChainAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  2.2 Grammar-Structured Output [{tag}]")

    prompt = (
        'Return a JSON object with keys "name", "age", "city" '
        "for a fictional person."
    )
    resp = adapter.call_llm(prompt, temperature=0.2)
    print(f"     Prompt: {prompt}")
    print(f"     Response: {resp}")


def demo_multimodal():
    """Demo 2.3: Multimodal Document Agent (text-only fallback)."""
    adapter = LangChainAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  2.3 Multimodal Document Agent [{tag}]")

    prompt = "Extract key information from this document: 'Invoice #1234, Date: 2024-01-15, Total: $450.00'"
    resp = adapter.call_llm(prompt, temperature=0.1)
    print(f"     Input: Invoice document text")
    print(f"     Extracted: {resp}")


def demo_lora():
    """Demo 2.4: LoRA Hotswap Personas with different system prompts."""
    adapter = LangChainAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  2.4 LoRA Hotswap Personas [{tag}]")

    personas = [
        ("pirate", "You are a salty pirate captain."),
        ("poet", "You are a romantic poet."),
    ]
    for name, system in personas:
        resp = adapter.call_llm(
            "Introduce yourself in one sentence.",
            system=system,
            temperature=0.8,
        )
        print(f"     [{name}] {resp}")


def demo():
    demo_router()
    demo_grammar()
    demo_multimodal()
    demo_lora()


if __name__ == "__main__":
    demo()
