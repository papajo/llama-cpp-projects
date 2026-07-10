# 6.3 Model Discovery & Router

A FastMCP server that discovers available LLM models via llama.cpp's `/v1/models` API, provides model info, and routes requests using configurable strategies.

## Tools

| Tool | Description |
|------|-------------|
| `list_models()` | List all available models with their inferred type |
| `get_model_info(model_id)` | Get details about a specific model |
| `route_request(task_type, preferred_model)` | Route a request to the best model |

## Routing Strategies

| Strategy | Behaviour |
|----------|-----------|
| `FIRST_AVAILABLE` | Return the first model (alphabetically) |
| `ROUND_ROBIN` | Cycle through all models evenly |
| `BY_NAME` | Return a specific named model |

## Architecture

```
  MCP Client ──▶ FastMCP Server ──▶ ModelRegistry
    (Claude)      (3 tools)            │
                                        ├── /v1/models (discovery)
                                        ├── type inference
                                        └── routing engine
```

## Files

| File | Purpose |
|------|---------|
| `model_router/registry.py` | `ModelInfo`, `ModelRegistry`, `RoutingStrategy`, error types |
| `model_router/server.py` | FastMCP server with 3 tools |
| `tests/` | 14 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
