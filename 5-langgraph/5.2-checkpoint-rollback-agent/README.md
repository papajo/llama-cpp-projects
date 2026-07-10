# 5.2 Checkpoint / Rollback Agent

A generic checkpoint system for multi-step agent workflows — saves immutable state snapshots at each step and supports rollback to any prior step.

## Architecture

```
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Step 0   │───▶│ Step 1   │───▶│ Step 2   │───▶ ...
  │ add_one  │    │ multiply │    │ add_one  │
  └────┬─────┘    └────┬─────┘    └────┬─────┘
       │               │               │
       ▼               ▼               ▼
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ CP #[0]  │    │ CP #[1]  │    │ CP #[2]  │
  │ {c: 1}   │    │ {c: 10}  │    │ {c: 11}  │
  └──────────┘    └──────────┘    └──────────┘
                       │
                       │ rollback_to(1)
                       ▼
                  {c: 1}  ← state restored
                  steps 2+ discarded
```

## Usage

```python
from checkpoint_rollback_agent.agent import AgentStep, CheckpointAgent

def add_one(state):
    return {"count": state.get("count", 0) + 1}

agent = CheckpointAgent([
    AgentStep(name="add_one", fn=add_one),
    AgentStep(name="double", fn=lambda s: {"count": s["count"] * 2}),
])

result = agent.run({"count": 0})
print(result.final_state)       # {"count": 2}
print(result.num_steps_executed)  # 2

# Rollback to step 0 (discard step 1)
state = agent.rollback_to(0)
print(state)                    # {"count": 1}
```

## Files

| File | Purpose |
|------|---------|
| `checkpoint_rollback_agent/store.py` | `Checkpoint` (immutable snapshot), `CheckpointStore` (append-only, rollback) |
| `checkpoint_rollback_agent/agent.py` | `AgentStep`, `StepResult`, `WorkflowResult`, `CheckpointAgent` |
| `tests/` | 17 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
