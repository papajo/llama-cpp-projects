# 5.5 Conditional Branching Agent

Route execution along different paths based on state conditions — a decision-tree workflow with pre-steps, branch points, and post-steps.

## Architecture

```
  pre_steps ──▶ Branch Point ──▶ post_steps
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
    Condition 1  Condition 2  Default
    (is_high)    (is_low)     (fallback)
          │          │          │
          ▼          ▼          ▼
     Branch A    Branch B    Branch C
       steps       steps       steps
```

- **Condition** — named predicate (`Callable[[dict], bool]`)
- **Branch** — a condition + sequence of `ActionStep`s + optional default flag
- **ConditionalAgent** — runs pre-steps, evaluates conditions in order, takes first match (or default), runs post-steps

## Usage

```python
from conditional_branching.core import (
    ActionStep, Branch, Condition, ConditionalAgent
)

is_high = Condition("high", predicate=lambda s: s.get("count", 0) > 10)
is_low = Condition("low", predicate=lambda s: s.get("count", 0) <= 10)

agent = ConditionalAgent(
    pre_steps=[ActionStep("add_five", lambda s: {"count": s["count"] + 5})],
    branches=[
        Branch(name="high_branch", condition=is_high,
               steps=[ActionStep("double", lambda s: {"count": s["count"] * 2})]),
        Branch(name="low_branch", condition=is_low,
               steps=[ActionStep("triple", lambda s: {"count": s["count"] * 3})]),
    ],
    post_steps=[ActionStep("label", lambda s: {"label": f"v{s['count']}"})],
)

result = agent.run({"count": 2})
print(result.final_state)    # {"count": 21, "label": "v21"}  (2+5=7 → low: 7*3=21)
print(result.branch_taken)   # "low_branch"
```

## Files

| File | Purpose |
|------|---------|
| `conditional_branching/core.py` | `Condition`, `ActionStep`, `Branch`, `ConditionalAgent`, result types |
| `tests/` | 16 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
