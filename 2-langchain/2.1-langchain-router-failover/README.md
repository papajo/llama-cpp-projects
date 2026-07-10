# LangChain Local-Model Router with Automatic Failover

A custom LangChain `ChatModel` (`RunnableRouter`) that integrates with
llama.cpp's **router server** to intelligently route tasks across multiple
local models behind a single OpenAI-compatible endpoint — with graceful
cold-start failover.

## Problem

llama.cpp's router server (`--models-dir`, `--models-preset`) can host
several models behind one endpoint, but LangChain has no idiomatic
integration for dynamic multi-model routing.

## Solution

`RunnableRouter` extends `BaseChatModel` so it plugs into any LangChain
chain or agent.  You register models with **tags**, then route per-request
via a `routing_key` or explicit `model_alias`.  When a model is asleep
(`--sleep-idle-seconds`), the router detects the cold-start signal and
retries with exponential backoff.

## Quick Start

```python
from langchain_core.messages import HumanMessage
from langchain_llamacpp_router import RunnableRouter, RouterModel

router = RunnableRouter(base_url="http://127.0.0.1:8080")

router.register_model(RouterModel(
    alias="coder",
    tags=["code", "generation", "fast"],
    description="Qwen 2.5 7B — fast code generation",
))
router.register_model(RouterModel(
    alias="chat",
    tags=["chat", "creative", "slow"],
    description="Llama 3.1 70B — creative writing",
))

# Route by tag
response = router.invoke(
    [HumanMessage(content="Write a Python fib function")],
    config={"routing_key": "code"},
)
print(response.content)

# Route by explicit alias
response = router.invoke(
    [HumanMessage(content="Tell me a story")],
    config={"model_alias": "chat"},
)
print(response.content)
```

## Features

- **Tag-based routing** — assign tags to models, route by keyword
- **Explicit alias** — bypass tag lookup with a direct model name
- **Cold-start failover** — detects `503`/`504` from sleeping models and
  retries with exponential backoff
- **Fallback chain** — ordered list of backup models if the primary fails
- **Streaming** — full SSE streaming support
- **Async** — `ainvoke`, `astream`, `_agenerate` for async pipelines
- **INI preset loader** — load model definitions from `--models-preset`
  files directly

## Preset Files

Create an INI file compatible with llama.cpp's `--models-preset`:

```ini
[coder:Qwen2.5-7B-Q4_K_M]
model = /models/qwen2.5-7b-q4_k_m.gguf
tags = code, generation, fast
description = Lightweight coder — fast on CPU

[chat:Llama-3.1-70B-Q3_K_L]
model = /models/llama-3.1-70b-q3_k_l.gguf
tags = chat, creative, slow
description = Full 70B creative model — needs GPU
cold_start_seconds = 15
```

Load it with:

```python
from pathlib import Path
from langchain_llamacpp_router import load_presets_ini

presets = load_presets_ini(Path("presets/my-models.ini"))
router.register_presets(presets)
```

## Running the Router Server

Start llama.cpp's router server:

```bash
llama-server \
    --models-dir /path/to/ggufs \
    --models-preset presets/my-models.ini \
    --models-max 3 \
    --models-autoload \
    --sleep-idle-seconds 300 \
    --host 127.0.0.1 --port 8080
```

### Launcher Dashboard

The `tools/llama-server-launcher.html` dashboard has a **Router** mode
that generates the correct `--models-dir` / `--models-preset` command
for you. Open it in any browser:

```bash
open tools/llama-server-launcher.html
```

## Dependencies

- `langchain-core>=0.3.0`
- `httpx>=0.27`
- `pydantic>=2.5`

## Project Structure

```
2.1-langchain-router-failover/
├── langchain_llamacpp_router/
│   ├── __init__.py
│   ├── router.py          # RunnableRouter (BaseChatModel subclass)
│   ├── preset_parser.py   # --models-preset INI parser
│   └── exceptions.py      # Custom exceptions
├── examples/
│   └── demo_router.py     # End-to-end demo script
├── presets/
│   ├── coding-chat.ini    # Sample: coder + chat models
│   └── full-stack.ini     # Sample: coder + chat + embedder
├── tests/
│   ├── test_preset_parser.py
│   └── test_router.py
├── pyproject.toml
├── requirements.txt
└── README.md
```
