# 5.3 Human Approval Loop

An agent workflow with human-in-the-loop approval gates — the agent pauses at critical decision points, presents context to a human, and awaits their decision before proceeding.

## Architecture

```
  AutoStep ──▶ ApprovalGate ──▶ AutoStep
                  │
          ┌───────┼───────┐
          ▼       ▼       ▼
       APPROVE  REJECT   MODIFY
          │       │       │
          ▼       ▼       ▼
      Run fn   Skip    Apply alt
                    state
```

**Decisions:**
| Decision | Behaviour |
|----------|-----------|
| `APPROVE` | Run the gate's function and continue |
| `REJECT` | Skip the step; state unchanged |
| `MODIFY` | Apply the human's alternative state instead |

## Usage

```python
from human_approval_loop.gates import (
    ApprovalGate, ApprovalDecision, Approver,
    AutoStep, Decision, HumanApprovalAgent, WorkflowItem
)

def add_item(state):
    items = state.get("items", [])
    items.append(state.get("next_item", "?"))
    return {"items": items}

# Custom approver
class MyApprover(Approver):
    def review(self, request):
        if request.context.get("next_item") == "bad":
            return ApprovalDecision(Decision.REJECT, comment="Not allowed")
        return ApprovalDecision(Decision.APPROVE)

agent = HumanApprovalAgent([
    ("auto", AutoStep("prepare", lambda s: {"next_item": "widget"})),
    ("gate", ApprovalGate("review-add", add_item, "Add next_item to list")),
], approver=MyApprover())

result = agent.run({"items": []})
print(result.final_state)   # {"items": ["widget"], "next_item": "widget"}
print(result.num_gates)     # 1
print(result.num_approved)  # 1
```

## Files

| File | Purpose |
|------|---------|
| `human_approval_loop/gates.py` | `ApprovalRequest`, `ApprovalDecision`, `Approver`, `ApprovalGate`, `AutoStep`, `HumanApprovalAgent`, result types |
| `tests/` | 12 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
