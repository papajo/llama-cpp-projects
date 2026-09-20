"""Live demo for Category 6 — MCP projects with real LLM context."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.integrations import MCPAdapter, ChatAdapter


def demo_slots_metrics():
    adapter = MCPAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  6.1 Slots Metrics Server [{tag}]")

    # Real decode-slot state from llama-server's /slots endpoint. This used to
    # ask the model to role-play a tool response, which produced invented prose
    # under a heading that implied real server data.
    print(f"     Slots: {adapter.slots_summary()}")
    for slot in adapter.slots():
        state = "processing" if slot.get("is_processing") else "idle"
        line = (
            f"       slot {slot.get('id')}: {state:10s}"
            f"  n_ctx={slot.get('n_ctx')}"
            f"  prompt_tokens={slot.get('n_prompt_tokens', 0)}"
        )
        temp = (slot.get("params") or {}).get("temperature")
        if isinstance(temp, (int, float)):
            line += f"  temp={temp:.2f}"
        print(line)

    # The project's own MCP tool: system metrics, not model output.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "6.1-slots-metrics-server"))
        from metrics_server.collector import MetricsCollector, format_metrics

        print("     System metrics (the project's MCP tool):")
        for line in format_metrics(MetricsCollector().collect()).splitlines():
            print(f"       {line}")
    except Exception as exc:  # pragma: no cover - demo convenience
        print(f"     System metrics unavailable: {exc}")


def demo_fs_shell_bridge():
    print("\n  6.2 FS-Shell MCP Bridge [no LLM needed — file system operations]")


def demo_model_discovery():
    adapter = ChatAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  6.3 Model Discovery & Router [{tag}]")
    models = adapter.discover_models()
    print(f"     Available models: {', '.join(models) if models else 'none'}")
    if models:
        resp = adapter.call_llm(
            f"Recommend which of these models to use for general chat: {', '.join(models)}",
            temperature=0.3,
        )
        print(f"     Recommendation: {resp}")


def demo_enterprise_gateway():
    adapter = MCPAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  6.4 Enterprise MCP Gateway [{tag}]")
    resp = adapter.tool_response("route_request", {"source": "slack", "intent": "query_knowledge"})
    print(f"     Gateway response: {resp}")


def demo():
    demo_slots_metrics()
    demo_fs_shell_bridge()
    demo_model_discovery()
    demo_enterprise_gateway()


if __name__ == "__main__":
    demo()
