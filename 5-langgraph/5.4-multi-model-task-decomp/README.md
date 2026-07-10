# 5.4 Multi-Model Task Decomposition

Decompose complex tasks into subtasks, route each subtask to a specialised handler (analogous to different "models"), and aggregate results.

## Architecture

```
Complex Task
     │
     ▼
┌────────────────┐
│  TaskDecomposer │  splits task → DecompositionPlan
└────────┬───────┘
         │ subtasks: [summarize, codegen, extract]
         ▼
┌──────────────────────────────────────────┐
│  MultiModelOrchestrator                  │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │ summarize│─▶│  codegen │─▶│ extract ││
│  │ handler  │  │  handler │  │ handler ││
│  └────┬─────┘  └────┬─────┘  └────┬────┘│
│       │              │             │     │
│       ▼              ▼             ▼     │
│  shared_state ←── outputs merged ─────── │
└──────────────────────────────────────────┘
```

- **Subtask** — unit of work with name, handler key, and optional dependencies
- **HandlerRegistry** — maps handler keys to callables
- **TaskDecomposer** — creates a `DecompositionPlan` (override for custom strategies)
- **MultiModelOrchestrator** — executes in dependency order, accumulates shared state

## Usage

```python
from task_decomp.core import (
    DecompositionPlan, HandlerRegistry, MultiModelOrchestrator,
    Subtask, TaskDecomposer
)

def codegen_handler(subtask, state):
    return {"code": f"// Generated: {subtask.description}"}

def review_handler(subtask, state):
    code = state.get("code", "")
    return {"review": f"Reviewed {len(code)} chars"}

class MyDecomposer(TaskDecomposer):
    def decompose(self, task):
        return DecompositionPlan(subtasks=[
            Subtask(name="gen", description="parse function", handler_key="code"),
            Subtask(name="review", description="review code", handler_key="review",
                    dependencies=["gen"]),
        ])

reg = HandlerRegistry()
reg.register("code", codegen_handler)
reg.register("review", review_handler)

orch = MultiModelOrchestrator(decomposer=MyDecomposer(), registry=reg)
result = orch.execute("Write a JSON parser")

print(result.all_succeeded)           # True
print(result.get_output("gen", "code"))      # "// Generated: parse function"
print(result.get_output("review", "review")) # "Reviewed 24 chars"
```

## Files

| File | Purpose |
|------|---------|
| `task_decomp/core.py` | `Subtask`, `HandlerRegistry`, `DecompositionPlan`, `TaskDecomposer`, `MultiModelOrchestrator`, result types |
| `tests/` | 14 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
