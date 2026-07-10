"""FastMCP server for model discovery and routing tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from model_router.registry import ModelRegistry, RoutingStrategy

mcp = FastMCP("model-router")
_registry = ModelRegistry()


@mcp.tool()
def list_models() -> str:
    """List all available models from the registry."""
    _registry.discover_if_empty()
    models = _registry.list_models()
    if not models:
        return "No models available."
    lines = []
    for m in models:
        mtype = _registry.infer_type(m.id)
        lines.append(f"- {m.id} (type: {mtype})")
    return "\n".join(lines)


@mcp.tool()
def get_model_info(model_id: str) -> str:
    """Get detailed information about a specific model."""
    _registry.discover_if_empty()
    info = _registry.get_model(model_id)
    if info is None:
        return f"Model '{model_id}' not found."
    return (
        f"ID: {info.id}\n"
        f"Object: {info.object}\n"
        f"Owned by: {info.owned_by}\n"
        f"Type: {_registry.infer_type(info.id)}\n"
    )


@mcp.tool()
def route_request(task_type: str = "chat", preferred_model: str = "") -> str:
    """Route a request to the best available model."""
    _registry.discover_if_empty()
    strategy = RoutingStrategy.FIRST_AVAILABLE
    pref = preferred_model if preferred_model else None
    model_id = _registry.route(strategy=strategy, preferred_model=pref)
    if model_id is None:
        return "No models available to route to."
    return f"Routed to '{model_id}' ({_registry.infer_type(model_id)})"
