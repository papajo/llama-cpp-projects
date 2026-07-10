# LoRA-Adapter Hot-Swap Agent Personas

A LangChain agent that switches **LoRA adapters at runtime** to change
its "personality" mid-conversation — without reloading the base model.

## Problem

Serving multiple fine-tuned models typically means running separate
processes per model — expensive and slow. llama.cpp's `--lora-init-without-apply`
plus `POST /lora-adapters` lets you load many LoRA adapters at startup
and switch between them with a simple API call.

## Solution

`LoraAgent` extends LangChain's `BaseChatModel` and automatically selects
the right LoRA adapter based on the user's input keywords. When the
topic shifts (e.g. from "review this contract" to "write a haiku"), the
agent hot-swaps the adapter via API call — the base model stays loaded.

## Quick Start

```python
from adapters import load_adapter_registry
from persona_router import PersonaRouter
from agent import LoraAgent

# Load adapter definitions
registry = load_adapter_registry("adapters/registry.json")
router = PersonaRouter(registry)

# Create the agent
agent = LoraAgent(
    base_url="http://127.0.0.1:8080",
    router=router,
)

# The agent auto-selects the right LoRA adapter
response = agent.invoke(
    "Review this contract for liability issues"
)
# -> Applies "legal-tone" LoRA, generates response

response = agent.invoke(
    "Write a haiku about debugging"
)
# -> Switches to "creative-writer" LoRA mid-conversation
```

## Features

- **Auto-routing** — detects the topic from user input and applies the
  matching LoRA adapter
- **Keyword scoring** — adapters with more keyword matches win ties
- **Manual switch** — `agent.switch_persona("legal-tone")` for explicit control
- **5 built-in personas** — legal-tone, concise-summarizer, code-reviewer,
  creative-writer, teacher
- **Adapter scale control** — adjusts personality strength (`--lora-scaled`)
- **No reloads** — only the adapter weights change, base model stays in VRAM

## Personas

| Persona | Keywords | Scale |
|---------|----------|-------|
| `legal-tone` | contract, lawsuit, clause, legal, attorney | 1.0 |
| `concise-summarizer` | summarize, tl;dr, summary, brief | 0.8 |
| `code-reviewer` | code review, bug, refactor, vulnerability | 1.0 |
| `creative-writer` | story, poem, haiku, narrative, fiction | 1.2 |
| `teacher` | explain, teach, learn, tutorial, beginner | 0.9 |

## Starting the Server

### LoRA CLI flags

There are two ways to load LoRA adapters at server startup:

| Flag | Format | Example |
|------|--------|---------|
| `--lora` | `FNAME` (can repeat, or comma-separated) | `--lora /adapters/legal-tone.gguf --lora /adapters/chat.gguf` |
| `--lora-scaled` | `FNAME:SCALE,...` (colon format, comma-separated, or repeat the flag) | `--lora-scaled /adapters/legal-tone.gguf:1.0,...` |

Both flags can be repeated or accept comma-separated values. Using
`--lora-scaled` is preferred because it sets the scale factor per
adapter (controls personality strength).

### The `--lora-init-without-apply` flag

Normally, all loaded LoRA adapters are applied immediately at startup
(the last one wins). Adding `--lora-init-without-apply` sets every
adapter's initial scale to **0** — none is active. The agent then
activates one via `POST /lora-adapters`.

Without this flag, the agent would need to explicitly disable the
startup-default adapter on every request, which wastes tokens.

### Start command

```bash
llama-server \
    --model /models/base-model.gguf \
    --lora-scaled /adapters/legal-tone.gguf:1.0 \
    --lora-scaled /adapters/concise-summarizer.gguf:0.8 \
    --lora-scaled /adapters/code-reviewer.gguf:1.0 \
    --lora-scaled /adapters/creative-writer.gguf:1.2 \
    --lora-scaled /adapters/teacher.gguf:0.9 \
    --lora-init-without-apply \
    --host 127.0.0.1 --port 8080
```

### Launcher Dashboard

The `tools/llama-server-launcher.html` dashboard has a **LoRA Personas**
mode that generates the correct command with checkbox-toggled adapters.
Open it in any browser:

```bash
open tools/llama-server-launcher.html
```

This loads all 5 LoRA adapters into slots (ID 0–4) but activates none.
The agent calls `POST /lora-adapters` to switch between them at runtime.

You can also use the comma-separated form:

```bash
llama-server \
    --model /models/base-model.gguf \
    --lora-scaled /adapters/legal-tone.gguf:1.0,/adapters/concise-summarizer.gguf:0.8,/adapters/code-reviewer.gguf:1.0,/adapters/creative-writer.gguf:1.2,/adapters/teacher.gguf:0.9 \
    --lora-init-without-apply \
    --host 127.0.0.1 --port 8080
```

## API Reference

### `POST /lora-adapters`

Sets the **global** scale for one or more LoRA adapters. The body is
a JSON **array** (not a single object):

```json
[
  {"id": 0, "scale": 1.0},
  {"id": 1, "scale": 0.0}
]
```

Setting scale to `0.0` disables an adapter. This global default is
overridden for individual requests via the `lora` field in
`/v1/chat/completions`.

### `GET /lora-adapters`

Returns the currently loaded adapters and their scales:

```json
[
  {"id": 0, "path": "legal-tone.gguf", "scale": 1.0},
  {"id": 1, "path": "chat.gguf", "scale": 0.0}
]
```

### Per-request LoRA selection

The `/v1/chat/completions` endpoint accepts an optional `lora` field —
exactly the same JSON array format — to override the global scale for
that specific request. This is useful if you want different adapters
for different turns without a global switch.

## Project Structure

```
2.4-lora-hotswap-personas/
├── adapters/
│   ├── __init__.py
│   ├── registry.py          # AdapterDef, AdapterRegistry
│   └── registry.json        # 5 sample adapter definitions
├── persona_router/
│   ├── __init__.py
│   └── router.py            # Keyword/tag-based persona selection
├── agent/
│   ├── __init__.py
│   ├── lora_manager.py      # POST /lora-adapters (apply, set, disable)
│   └── lora_agent.py        # LoraAgent (BaseChatModel)
├── tests/
│   ├── test_adapter_registry.py
│   └── test_persona_router.py
├── pyproject.toml
└── README.md
```
