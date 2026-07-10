"""Live demo for Category 5 — LangGraph projects with real LLM."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.integrations import LangGraphAdapter


def demo_draft_verify():
    adapter = LangGraphAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  5.1 Draft-Verify Graph [{tag}]")
    resp = adapter.call_node("draft", "Write a short tweet about AI.")
    print(f"     Draft output: {resp}")


def demo_checkpoint_rollback():
    adapter = LangGraphAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  5.2 Checkpoint & Rollback Agent [{tag}]")
    resp = adapter.call_node(
        "process_step",
        "Process this user request: 'Book a flight to London'",
        state={"step": 1, "completed": False},
    )
    print(f"     State: step=1")
    print(f"     Output: {resp}")


def demo_human_approval():
    adapter = LangGraphAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  5.3 Human Approval Loop [{tag}]")
    resp = adapter.call_llm(
        "Generate a summary of this action: 'Send email to customer about order delay'"
    )
    print(f"     Action summary: {resp}")


def demo_multi_model():
    adapter = LangGraphAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  5.4 Multi-Model Task Decomposition [{tag}]")
    resp = adapter.call_node(
        "decompose",
        "Break this down: 'Plan a birthday party'",
        state={"task": "birthday_party"},
    )
    print(f"     Decomposition: {resp}")


def demo_conditional_branching():
    adapter = LangGraphAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  5.5 Conditional Branching [{tag}]")
    resp = adapter.call_llm("Classify this query as 'technical' or 'general': 'How do I fix a memory leak?'")
    print(f"     Classification: {resp}")


def demo_parallel():
    adapter = LangGraphAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  5.6 Parallel Execution [{tag}]")
    resp = adapter.call_llm("List 3 pros and cons of remote work.")
    print(f"     Result: {resp}")


def demo_supervisor():
    adapter = LangGraphAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  5.7 Supervisor Agent [{tag}]")
    resp = adapter.call_llm(
        "As a supervisor, decide which specialist handles: 'Customer wants a refund'",
        system="You are a supervisor agent. Choose: billing, support, or escalation.",
    )
    print(f"     Decision: {resp}")


def demo_map_reduce():
    adapter = LangGraphAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  5.8 Map-Reduce [{tag}]")
    resp = adapter.call_node(
        "map",
        "Summarize: 'AI has many applications including healthcare, finance, and transportation.'",
        state={"mode": "map"},
    )
    print(f"     Map result: {resp}")


def demo():
    demo_draft_verify()
    demo_checkpoint_rollback()
    demo_human_approval()
    demo_multi_model()
    demo_conditional_branching()
    demo_parallel()
    demo_supervisor()
    demo_map_reduce()


if __name__ == "__main__":
    demo()
