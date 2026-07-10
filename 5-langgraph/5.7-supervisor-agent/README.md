# 5.7 Supervisor Agent

Hierarchical delegation — a supervisor routes tasks to specialised sub-agents based on keyword capability matching or custom router logic.

## Architecture

```
         Supervisor
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
 SubAgent  SubAgent  SubAgent
 (coder)  (review)  (deploy)
    │        │        │
    └────────┼────────┘
             ▼
       merged state
```

**Delegation strategies:**
| Strategy | Behaviour |
|----------|-----------|
| `FIRST_MATCH` | Route to the first sub-agent whose capability keywords match the task |
| `ALL_MATCH` | Route to **all** matching sub-agents; each builds on the previous output |
| `CUSTOM` | Use a user-provided `router(task, state, agents) → List[SubAgent]` |

## Usage

```python
from supervisor_agent.core import (
    DelegationStrategy, SubAgent, SupervisorAgent
)

def code_gen(state):
    return {"code": "# generated code"}

def code_review(state):
    return {"review": "Looks good"}

agent = SupervisorAgent(
    sub_agents=[
        SubAgent("coder", "code generation", code_gen),
        SubAgent("reviewer", "code review", code_review),
    ],
    strategy=DelegationStrategy.FIRST_MATCH,
)

result = agent.delegate("generate some code for parsing")
print(result.selected_agents)   # ["coder"]
print(result.final_state)       # {"code": "# generated code"}
print(result.all_succeeded)     # True
```

## Files

| File | Purpose |
|------|---------|
| `supervisor_agent/core.py` | `SubAgent`, `DelegationStrategy`, `SupervisorAgent`, `SupervisorResult` |
| `tests/` | 14 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
