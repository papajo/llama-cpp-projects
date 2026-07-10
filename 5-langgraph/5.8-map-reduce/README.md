# 5.8 Map-Reduce Agent

Split input into chunks, process each chunk independently (in parallel via `ThreadPoolExecutor`), then aggregate results — the classic map-reduce pattern.

## Architecture

```
  Input: "a b c\nd e f\ng h i"
            │
            ▼ split_lines
    ┌───────┼───────┐
    ▼       ▼       ▼
  "a b c" "d e f" "g h i"
    │       │       │
    ▼ MAP   ▼ MAP   ▼ MAP
  count=3  count=3  count=3
    │       │       │
    └───────┼───────┘
            ▼ reduce_sum
            9
```

**Split functions** (built-in): `split_fixed_size`, `split_lines`, `split_by_delimiter`  
**Reduce functions** (built-in): `reduce_concat`, `reduce_sum`, `reduce_list`

## Usage

```python
from map_reduce.core import MapReduceAgent, split_lines, reduce_concat

def shout(chunk: str, idx: int) -> str:
    return chunk.upper() + "!"

agent = MapReduceAgent(
    split_fn=split_lines,
    chunk_fn=shout,
    reduce_fn=reduce_concat,
    max_workers=4,
)

result = agent.run("hello\nworld")
print(result.num_chunks)          # 2
print(result.reduced_output)      # "HELLO!WORLD!"
print(result.all_succeeded)       # True
```

## Files

| File | Purpose |
|------|---------|
| `map_reduce/core.py` | `Chunk`, `MapReduceAgent`, split/reduce helpers, result types |
| `tests/` | 17 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
