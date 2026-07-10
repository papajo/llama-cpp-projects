# llama.cpp Server Launcher Dashboard

A unified HTML dashboard for configuring and launching `llama-server` with
different model setups — chat, vision, router, and LoRA personas.

## Quick Start

Open the dashboard in any browser:

```bash
open tools/llama-server-launcher.html
```

Or use the convenience script:

```bash
bash tools/launch-server-ui.sh
```

## Four Quick Modes

| Mode | Use Case | Server Flags |
|------|----------|-------------|
| **Basic Chat** | Single text model | `--model` |
| **Vision** | Multimodal model (images, video, audio) | `--model` + `--mmproj` |
| **Router** | Multiple models behind one endpoint | `--models-dir` + `--models-preset` |
| **LoRA Personas** | Single model with swappable LoRA adapters | `--model` + `--lora-scaled` + `--lora-init-without-apply` |

Each mode dynamically shows/hides the relevant sections (multimodal model
cards, LoRA adapter manager, etc.).

## Key Features

### Auto-Download Models from Hugging Face

Use the `-hf repo:quant` syntax to auto-download without specifying a local path:

```
-hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
```

The server downloads to `~/.cache/huggingface/hub/` and reuses cached files
on subsequent runs — no re-download.

The dashboard highlights `-hf` paths in **purple** so you can tell at a glance
which models are remote vs. local.

### Multimodal Model Picker

9 pre-configured multimodal models with capability badges:

| Model | Image | Video | Audio |
|-------|-------|-------|-------|
| Gemma 3 4B / 12B | ✓ | | |
| Gemma 4 E2B / E4B | ✓ | | |
| Qwen2.5 VL 3B / 7B | ✓ | ✓ | |
| SmolVLM2 2.2B / 500M Video | ✓ | ✓ | |
| Pixtral 12B | ✓ | | |
| Qwen2.5 Omni 7B / 3B | ✓ | | ✓ |

The `--mmproj` file is auto-downloaded alongside the model when using `-hf`.

> **⚠️ Vision check:** To confirm a model supports vision, check the model info
> endpoint: `curl http://localhost:8080/v1/model/info`. If `"capabilities"`
> includes `"vision"` (or image/video/audio content in `"multimodal"`), it
> supports vision. Some models **claim** vision capabilities in their docs but
> don't actually support it (e.g. the Ornith model returns
> `"capabilities": ["completion"]` only).

### LoRA Adapter Manager

5 pre-configured personas from the 2.4 project:

| Persona | Scale | Description |
|---------|-------|-------------|
| legal-tone | 1.0 | Formal legal writing |
| concise-summarizer | 0.8 | Short summaries |
| code-reviewer | 1.0 | Code quality review |
| creative-writer | 1.2 | Stories, poems |
| teacher | 0.9 | Explanations |

Each can be enabled/disabled with checkbox toggles. The `--lora-scaled` flags
and `--lora-init-without-apply` are added automatically.

### Auto-Kill Toggle

When enabled, prepends `kill $(lsof -ti:<PORT>)` before the server start
command to free a stuck port. Essential when port 8080 (or another) is held
by a previous process.

## Important Server Notes

### GGUF Only

`llama-server` only loads **GGUF** format models. MLX safetensors directories
(commonly downloaded from Hugging Face) cannot be used directly — you must
convert them with `llama-quantize` or use a pre-quantized GGUF from Hugging Face.

### Port Conflicts

If the server fails to start with `bind: address already in use`, someone is
already listening on that port. The auto-kill toggle handles this, or run
manually:

```bash
kill $(lsof -ti:8080)
```

### Model Hallucination

Not all models that claim "multimodal" or "vision" support actually implement
it. Always verify against the server's actual capabilities endpoint:

```bash
curl -s http://localhost:8080/v1/model/info | python3 -m json.tool
```

Look for `"vision"` in the `"capabilities"` array, or image/video/audio
content types listed under `"multimodal"`.

## Accessing the Dashboard

- **File:** `tools/llama-server-launcher.html`
- **Script:** `tools/launch-server-ui.sh`
- **Open:** Just double-click the HTML file or open via the script above.
  No web server needed — it's fully self-contained.

---

## Prompt Chain Workbench

A visual workbench for designing, editing, and testing prompt chains —
sequences of LLM calls where each step's output feeds into the next.

- **File:** `tools/chain-workbench.html`
- **Open:** `open tools/chain-workbench.html`
- **Note:** Chain workbench makes API calls to llama.cpp, so it must be
  served via HTTP. Run from the project root:

  ```bash
  python3 -m http.server 8000
  # Then open http://localhost:8000/tools/chain-workbench.html
  ```

### ⚠️ CORS Requirement

The workbench runs in a browser at `http://localhost:8000` but makes API
calls to `http://localhost:8080` (the llama.cpp server). This is a
cross-origin request, so **llama-server must be started with `--cors *`**.

**Option 1 — Server Launcher:** Use `tools/llama-server-launcher.html`
and enable the **"Enable CORS"** toggle (on by default for new sessions).

**Option 2 — Manual:**
```bash
llama-server --model your-model.gguf --cors * --port 8080
```

Without `--cors *`, external browsers (Chrome, Safari, Firefox) will
block the API requests with a CORS error. The inline webview in some
terminals may not show this error.

### Features

- **Visual chain builder** — Add, remove, reorder steps with editable
  prompts, temperature, and max_tokens per step
- **6 built-in templates** — Summarize+Translate, Brainstorm→Critique→Refine,
  Expand+Polish, Question Decomposition, Fact Check, Chain of Thought
- **Template variables** — `{{input}}`, `{{step_1}}`, `{{step_name}}`
  reference prior outputs
- **Live execution** — Run chains against any llama.cpp server, see
  per-step results with timing and token counts
- **Import / Export** — Save chains as JSON, share via URL hash
- **Rendered prompt viewer** — Click any step header during/after
  execution to see the resolved prompts sent to the API
- **Auto-retry** — Each step retries up to 2 times on 5xx errors with
  exponential backoff
