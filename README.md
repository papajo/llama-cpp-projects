# 🦙 llama-cpp-projects

A collection of 29 hands-on projects exploring LLM inference, prompt engineering, RAG, agentic workflows, and MCP — all wired up to run with local AI models.

## Quick Start

```bash
# Start a local LLM server
ollama serve                    # Ollama (recommended)
# or: llama-server -m model.gguf   # llama.cpp
# or: omlx serve <model>           # MLX (Apple Silicon)

# Run everything with real LLM:
python3 demo_live_all.py
```

**Requirements:** Python 3.10+, a running local LLM server (Ollama, llama.cpp, or MLX).

## Project Categories

| # | Category | Projects | Live Demo |
|---|----------|----------|-----------|
| 1 | **AI Fundamentals** | KV Cache, Spec Decoding, Constrained Decoding, GGUF Quant, Sampling Params, Prompt Cache | `1-ai-fundamentals/demo_live.py` |
| 2 | **LangChain** | Router/Failover, Grammar Output, Multimodal Doc Agent, LoRA Personas | `2-langchain/demo_live.py` |
| 3 | **Prompt Engineering** | Reasoning Budget, Sampler Ablation, Chat Templates, Prompt Chaining, Optimizer | `3-prompt-engineering/demo_live.py` |
| 4 | **Vector DB & RAG** | Embedding Norms, Local Rerank, Cache Chunking, Multi-Tenant RAG | `4-vector-db-rag/demo_live.py` |
| 5 | **LangGraph** | Draft-Verify, Checkpoint/Rollback, Human Approval, Multi-Model, Branching, Parallel, Supervisor, Map-Reduce | `5-langgraph/demo_live.py` |
| 6 | **MCP** | Slots Metrics, FS-Shell Bridge, Model Discovery, Enterprise Gateway | `6-mcp/demo_live.py` |
| 7 | **Bonus** | CPU Affinity, Inference Profiling, NUMA Deployment, Cache Hit Rate, Sleep-Wake Profiler | `7-bonus/demo_live.py` |

## How It Works

Each category has a `demo_live.py` that auto-discovers your running LLM server and exercises every project with real inference. The shared integration layer (`_shared/`) handles:

| Module | Purpose |
|--------|---------|
| `llm_client.py` | Auto-detect Ollama / llama.cpp / MLX, manage connections |
| `integrations.py` | Adapter classes (Chat, Embed, LangChain, RAG, LangGraph, MCP, etc.) |
| `__init__.py` | Package exports |

All adapters gracefully fall back to mock responses when no server is running.

## Running a Single Category

```bash
# Just the AI Fundamentals demos:
python3 1-ai-fundamentals/demo_live.py

# Just the RAG demos:
python3 4-vector-db-rag/demo_live.py

# Just the LangGraph agent demos:
python3 5-langgraph/demo_live.py
```

## Supported Backends

| Backend | Status | Notes |
|---------|--------|-------|
| **Ollama** | ✅ Verified | Tested with llama3.2:latest, nomic-embed-text |
| **llama.cpp** | ✅ Supported | Auto-detects llama-server on port 8080 |
| **MLX** (Apple Silicon) | ✅ Supported | Auto-detects omlx server |
