# 5.1 Draft-Verify Graph

A reflexion loop that generates a draft, critiques it, and iteratively improves it — the core pattern behind self-correcting LLM agents.

## Architecture

```
┌──────────┐     ┌──────────┐     ┌───────────┐
│  Draft   │────▶│  Verify  │────▶│  Improve  │
│   Node   │     │   Node   │     │   Node    │
└──────────┘     └──────────┘     └───────────┘
                       │                │
                       ▼                │
                  pass? ──no────────────┘
                       │
                    yes ▼
                  ┌──────────┐
                  │  return  │
                  │  result  │
                  └──────────┘
```

1. **DraftNode** — generates an initial response using the LLM
2. **VerifyNode** — critiques the response (LLM-as-judge, returns `{score, issues, verdict}`)
3. **ImproveNode** — rewrites the response addressing the issues
4. **DraftVerifyGraph** — orchestrates the loop, stopping when `score >= 7` or `max_iterations` reached

## Usage

```python
from draft_verify_graph.llm import LlamaClient
from draft_verify_graph.nodes import DraftNode, VerifyNode, ImproveNode
from draft_verify_graph.graph import DraftVerifyGraph

client = LlamaClient("http://localhost:8080")

graph = DraftVerifyGraph(
    draft_node=DraftNode(client=client),
    verify_node=VerifyNode(client=client),
    improve_node=ImproveNode(client=client),
    max_iterations=5,
)

result = graph.run("Explain quantum computing in 3 sentences")
print(result.final_output)
print(f"Passed: {result.passed} after {result.num_iterations} iterations")
```

## Files

| File | Purpose |
|------|---------|
| `draft_verify_graph/llm.py` | `LlamaClient` — HTTP client for llama.cpp chat completions |
| `draft_verify_graph/nodes.py` | `DraftNode`, `VerifyNode`, `ImproveNode`, `VerificationResult` |
| `draft_verify_graph/graph.py` | `DraftVerifyGraph` — the loop orchestrator |
| `draft_verify_graph/reporting.py` | `result_to_markdown()` — format report |
| `tests/` | 18 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
