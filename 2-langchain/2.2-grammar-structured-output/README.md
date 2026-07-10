# Structured-Output Extraction Chain via GBNF/JSON-Schema Backend

A LangChain output parser that converts a **Pydantic model** into a
`--json-schema` grammar constraint, sends it to llama.cpp's `/completion`
endpoint, and returns a validated Pydantic object — with **zero retries**
needed.

## Problem

LangChain's `with_structured_output` relies on tool-calling fine-tunes
and prompt-based retries. On small local models this is unreliable —
schema violations require multiple retry attempts, increasing latency
and cost.

## Solution

llama.cpp's `--json-schema` grammar constraint operates at the **sampler
level**. The model **cannot** produce tokens outside the grammar. Output
is guaranteed schema-conformant on the first try.

## Quick Start

```python
from pydantic import BaseModel, Field
from parser import GrammarOutputParser

class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(ge=0, le=150)
    email: str | None = None

parser = GrammarOutputParser(base_url="http://127.0.0.1:8080")

person = parser.invoke(
    prompt="Extract: John Smith is 30 years old",
    schema=Person,
)
# -> Person(name="John Smith", age=30, email=None)
# Guaranteed, no retries needed.
```

## Features

- **Zero-retry structured output** — grammar constraints at the sampler
  level guarantee JSON Schema conformance
- **Full Pydantic support** — nested models, `Enum`, `Literal`,
  `Optional`, `list[X]`, `dict[str, X]`, and field constraints
  (`ge`, `le`, `min_length`, `max_length`, `pattern`)
- **LangChain Runnable** — `GrammarStructuredOutput` wraps a prompt
  template + schema for drop-in use in chains
- **Streaming** — full SSE streaming support
- **Code fence cleanup** — strips ```json fences the model might add

## Schema Compiler

The `schema_compiler` module converts Pydantic models → JSON Schema:

```python
from schema_compiler import SchemaCompiler

compiler = SchemaCompiler()
schema = compiler.compile(Person)
# -> {"type": "object", "properties": {...}, "required": [...]}
```

## Compare with `with_structured_output`

Run the benchmark:

```bash
python -m benchmark_vs_native_langchain.benchmark
```

## Requirements

- llama.cpp server running at `http://127.0.0.1:8080` (or custom URL)
- `langchain-core>=0.3.0`
- `httpx>=0.27`
- `pydantic>=2.5`

> **Tip:** Use the `tools/llama-server-launcher.html` dashboard to quickly
> start a server with the correct flags. The **Basic Chat** mode works for
> structured output — grammar constraints run at the sampler level on any
> model.

## Project Structure

```
2.2-grammar-structured-output/
├── schema_compiler/
│   ├── __init__.py
│   └── compiler.py          # Pydantic → JSON Schema conversion
├── parser/
│   ├── __init__.py
│   ├── parser.py            # GrammarOutputParser (wraps /completion)
│   └── chain.py             # GrammarStructuredOutput (Runnable chain)
├── benchmark_vs_native_langchain/
│   └── benchmark.py         # Comparison against with_structured_output
├── tests/
│   ├── test_schema_compiler.py
│   └── test_parser.py
├── pyproject.toml
└── README.md
```
