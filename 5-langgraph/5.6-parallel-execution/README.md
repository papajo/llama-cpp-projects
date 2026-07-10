# 5.6 Parallel Execution Agent

Fan-out to concurrent subtasks, then fan-in with a merge function — enabling true parallel execution via `ThreadPoolExecutor`.

## Architecture

```
  pre_steps ──▶ Parallel Group ──▶ post_steps
                     │
            ┌───────┼───────┐
            ▼       ▼       ▼
        Task A   Task B   Task C
        (inc)   (double) (label)
            │       │       │
            └───────┼───────┘
                    ▼
              merge_fn
               (union)
                    │
                    ▼
            state updated
```

- **ParallelGroup** — set of `ParallelTask`s run concurrently, then merged
- **ParallelAgent** — serial pre/post steps + one or more parallel groups
- **Custom merge** — `merge_fn(outputs)` controls how task outputs combine

## Usage

```python
from parallel_execution.core import (
    ParallelAgent, ParallelGroup, ParallelTask, SerialStep
)

agent = ParallelAgent(
    pre_steps=[SerialStep("init", lambda s: {"items": [1, 2, 3]})],
    parallel_groups=[
        ParallelGroup("process", tasks=[
            ParallelTask("sum", lambda s: {"total": sum(s["items"])}),
            ParallelTask("count", lambda s: {"count": len(s["items"])}),
            ParallelTask("avg", lambda s: {"avg": sum(s["items"]) / len(s["items"])}),
        ]),
    ],
    post_steps=[SerialStep("report", lambda s: {"report": f"{s['count']} items"})],
)

result = agent.run()
print(result.final_state)
# {"items": [1,2,3], "total": 6, "count": 3, "avg": 2.0, "report": "3 items"}
```

## Files

| File | Purpose |
|------|---------|
| `parallel_execution/core.py` | `SerialStep`, `ParallelTask`, `ParallelGroup`, `ParallelAgent`, result types |
| `tests/` | 12 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
