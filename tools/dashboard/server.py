"""Interactive dashboard server — pick provider/model, click a project card, get a live LLM response."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _shared.integrations import (
    ChatAdapter,
    EmbedAdapter,
    LangChainAdapter,
    PromptEngineeringAdapter,
    RAGAdapter,
    LangGraphAdapter,
    MCPAdapter,
    get_adapter,
    live_projects_summary,
)
from _shared.llm_client import LLMClient, get_client, detect_server

logger = logging.getLogger("dash")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="llama-cpp Interactive Dashboard", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────

_current_provider: str = "ollama"
_active_model: str = "llama3.2:latest"
_client: Optional[LLMClient] = None

# ── Project definitions with domain-specific prompts ─────────────────────

PROJECTS: List[Dict[str, Any]] = [
    # Category 1 — AI Fundamentals
    {"id": "1.1", "cat": 1, "icon": "🧠", "name": "KV Cache Visualizer",
     "desc": "Visualise how the key-value cache works in transformer inference.",
     "prompt": "Explain what the KV cache is in transformer models in 2-3 sentences."},
    {"id": "1.2", "cat": 1, "icon": "⚡", "name": "Speculative Decoding",
     "desc": "Benchmark draft-model speculative decoding speedups.",
     "prompt": "What is speculative decoding and how does it speed up LLM inference?"},
    {"id": "1.3", "cat": 1, "icon": "🔒", "name": "Constrained Decoding",
     "desc": "Playground for grammar-guided JSON and structured output.",
     "prompt": "Generate a JSON object with keys: name (string), age (integer), skills (array of strings). Return ONLY valid JSON."},
    {"id": "1.4", "cat": 1, "icon": "📦", "name": "GGUF Quant Explorer",
     "desc": "Explore GGUF quantization levels and their trade-offs.",
     "prompt": "What are the different GGUF quantization levels and how do they affect model quality vs size?"},
    {"id": "1.5", "cat": 1, "icon": "🎛️", "name": "Sampling Param Explorer",
     "desc": "Interactively tweak temperature, top-k, top-p, repeat penalty.",
     "prompt": "List the most important sampling parameters for LLM text generation and describe how each affects output."},
    {"id": "1.6", "cat": 1, "icon": "💾", "name": "Prompt Cache Benchmark",
     "desc": "Measure prompt caching efficiency across repeated requests.",
     "prompt": "How does prompt caching improve LLM serving performance?"},

    # Category 2 — LangChain
    {"id": "2.1", "cat": 2, "icon": "🔀", "name": "LangChain Router & Failover",
     "desc": "Route prompts across models with automatic failover.",
     "prompt": "What is the capital of Australia?"},
    {"id": "2.2", "cat": 2, "icon": "📐", "name": "Grammar-Structured Output",
     "desc": "Generate constrained JSON using LangChain output parsers.",
     "prompt": "Return a JSON object with keys \"name\", \"age\", \"city\" for a fictional person."},
    {"id": "2.3", "cat": 2, "icon": "🖼️", "name": "Multimodal Doc Agent",
     "desc": "Extract structured data from invoices and documents.",
     "prompt": "Extract key information from this invoice: Invoice #1234, Date: 2024-01-15, Total: $450.00, Vendor: Acme Corp"},
    {"id": "2.4", "cat": 2, "icon": "🎭", "name": "LoRA Hotswap Personas",
     "desc": "Hot-swap LoRA adapters to change model persona mid-conversation.",
     "prompt": "Describe yourself in character as a helpful AI assistant with a friendly personality."},

    # Category 3 — Prompt Engineering
    {"id": "3.1", "cat": 3, "icon": "📏", "name": "Reasoning Budget Sweep",
     "desc": "Compare output quality at different max-token budgets.",
     "prompt": "Explain quantum computing in one paragraph."},
    {"id": "3.2", "cat": 3, "icon": "🌡️", "name": "Sampler Ablation Lab",
     "desc": "A/B test temperature, top-k, top-p, min-p side by side.",
     "prompt": "Write a tagline for a coffee shop."},
    {"id": "3.3", "cat": 3, "icon": "📋", "name": "Chat Template Tester",
     "desc": "Verify chat template rendering across model formats.",
     "prompt": "What is 2+2? (Answer briefly.)"},
    {"id": "3.4", "cat": 3, "icon": "⛓️", "name": "Prompt Chaining Workbench",
     "desc": "Design chains where each step's output feeds the next.",
     "prompt": "Generate a creative topic for a short story, then write a single sentence for it."},
    {"id": "3.5", "cat": 3, "icon": "🔁", "name": "Prompt Optimizer Loop",
     "desc": "Iteratively refine prompts using LLM-as-judge feedback.",
     "prompt": "Explain the water cycle briefly."},

    # Category 4 — Vector DB / RAG
    {"id": "4.1", "cat": 4, "icon": "📏", "name": "Embedding Norm Ablation",
     "desc": "Analyse how embedding normalisation affects similarity search.",
     "prompt": "What is the embedding dimension of this model and why does normalisation matter?"},
    {"id": "4.2", "cat": 4, "icon": "🎯", "name": "Local Rerank RAG",
     "desc": "Two-stage retrieval with a local cross-encoder reranker.",
     "prompt": "Who created the Python programming language?"},
    {"id": "4.3", "cat": 4, "icon": "🧩", "name": "Prompt Cache Chunking",
     "desc": "Chunk contexts to maximise prompt cache reuse.",
     "prompt": "What is deep learning and how does it differ from traditional machine learning?"},
    {"id": "4.4", "cat": 4, "icon": "🏢", "name": "Multi-Tenant RAG",
     "desc": "Isolate RAG data per tenant with shared embedding models.",
     "prompt": "What is the revenue trend for Q4 2024 based on the available data?"},

    # Category 5 — LangGraph
    {"id": "5.1", "cat": 5, "icon": "✏️", "name": "Draft-Verify Graph",
     "desc": "Two-stage graph: draft content then self-verify.",
     "prompt": "Write a short tweet about the future of AI."},
    {"id": "5.2", "cat": 5, "icon": "💾", "name": "Checkpoint & Rollback",
     "desc": "Graph agent with state checkpointing and rollback.",
     "prompt": "Process this request: 'Book a flight to London for next Monday.'"},
    {"id": "5.3", "cat": 5, "icon": "👤", "name": "Human Approval Loop",
     "desc": "Pause graph execution for human-in-the-loop approval.",
     "prompt": "Summarise this action for human approval: 'Send email to customer about order delay.'"},
    {"id": "5.4", "cat": 5, "icon": "🔀", "name": "Multi-Model Task Decomposition",
     "desc": "Decompose complex tasks across different models.",
     "prompt": "Break down the task: 'Plan a birthday party for a 10-year-old.'"},
    {"id": "5.5", "cat": 5, "icon": "🌿", "name": "Conditional Branching",
     "desc": "Branch execution based on LLM classifier output.",
     "prompt": "Classify this query as 'technical' or 'general': 'How do I fix a memory leak in Python?'"},
    {"id": "5.6", "cat": 5, "icon": "⚡", "name": "Parallel Execution",
     "desc": "Run multiple graph branches concurrently.",
     "prompt": "List 3 pros and 3 cons of remote work."},
    {"id": "5.7", "cat": 5, "icon": "👁️", "name": "Supervisor Agent",
     "desc": "Supervisor routes work to specialist sub-agents.",
     "prompt": "As a supervisor, decide which specialist handles this: 'Customer wants a refund for a damaged item.'"},
    {"id": "5.8", "cat": 5, "icon": "🗺️", "name": "Map-Reduce",
     "desc": "Map phase processes items in parallel; reduce phase aggregates.",
     "prompt": "Summarise this: 'AI has many applications including healthcare diagnostics, financial forecasting, autonomous vehicles, and creative content generation.'"},

    # Category 6 — MCP
    {"id": "6.1", "cat": 6, "icon": "📊", "name": "Slots Metrics Server",
     "desc": "MCP server exposing llama-server slot usage metrics.",
     "prompt": "Describe what information a slots metrics MCP server should expose for monitoring a model serving cluster."},
    {"id": "6.2", "cat": 6, "icon": "🔗", "name": "FS-Shell MCP Bridge",
     "desc": "Bridge file-system shell commands through MCP.",
     "prompt": "What are the security considerations when exposing file-system operations through an MCP server?"},
    {"id": "6.3", "cat": 6, "icon": "🔍", "name": "Model Discovery & Router",
     "desc": "Discover and route requests across multiple models.",
     "prompt": "What factors should a model router consider when choosing which LLM to use for a given request?"},
    {"id": "6.4", "cat": 6, "icon": "🏗️", "name": "Enterprise MCP Gateway",
     "desc": "Central gateway for tool discovery, auth, and rate limiting.",
     "prompt": "What features should an enterprise MCP gateway provide for security and governance?"},

    # Category 7 — Bonus
    {"id": "7.1", "cat": 7, "icon": "⚙️", "name": "CPU Affinity Tuning",
     "desc": "Pin threads to cores for latency predictability.",
     "prompt": "How does CPU thread affinity tuning improve LLM inference latency?"},
    {"id": "7.2", "cat": 7, "icon": "⏱️", "name": "Inference Profiling",
     "desc": "Profile prompt processing vs token generation latency.",
     "prompt": "Say 'Hello, world!' — timing this request."},
    {"id": "7.3", "cat": 7, "icon": "🏛️", "name": "NUMA Deployment",
     "desc": "NUMA-aware deployment topology for multi-socket servers.",
     "prompt": "Why does NUMA-aware memory placement matter for LLM serving on multi-socket servers?"},
    {"id": "7.4", "cat": 7, "icon": "💡", "name": "Prompt Cache Hit Rate",
     "desc": "Measure cache hit/miss ratios across request patterns.",
     "prompt": "What is the meaning of life?"},
    {"id": "7.5", "cat": 7, "icon": "🔋", "name": "Sleep-Wake Profiler",
     "desc": "Measure power-state transition costs in inference.",
     "prompt": "How do CPU sleep states (C-states) affect LLM inference latency and power consumption?"},
]

CATEGORIES: Dict[int, Dict[str, Any]] = {
    1: {"name": "AI Fundamentals", "icon": "🧠"},
    2: {"name": "LangChain", "icon": "🔗"},
    3: {"name": "Prompt Engineering", "icon": "💬"},
    4: {"name": "Vector DB / RAG", "icon": "📐"},
    5: {"name": "LangGraph", "icon": "🔀"},
    6: {"name": "MCP", "icon": "🔌"},
    7: {"name": "Bonus / Performance", "icon": "⚡"},
}


def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = get_client(prefer=_current_provider, auto_start=False)
    return _client


def _rebuild_client(provider: str):
    global _client, _current_provider
    _current_provider = provider
    _client = get_client(prefer=provider, auto_start=False, reset=True)


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/providers")
async def list_providers():
    """Return available providers and which one is active."""
    return {
        "providers": [
            {"id": "ollama", "name": "Ollama", "port": 11434},
            {"id": "llama.cpp", "name": "llama.cpp", "port": 8080},
            {"id": "mlx", "name": "MLX", "port": 9090},
        ],
        "active": _current_provider,
    }


@app.get("/api/models")
async def list_models(provider: Optional[str] = None):
    """Return models available on the given (or current) provider."""
    if provider and provider != _current_provider:
        _rebuild_client(provider)
    client = _get_client()
    if not client.connected:
        return {"models": [], "connected": False, "provider": _current_provider}
    return {
        "models": client.available_models,
        "connected": True,
        "provider": _current_provider,
        "active_model": _active_model,
        "embedding_models": [m for m in client.available_models if "embed" in m.lower() or "nomic" in m.lower()],
    }


@app.post("/api/switch-provider")
async def switch_provider(body: dict):
    """Switch the active provider."""
    provider = body.get("provider", "ollama")
    _rebuild_client(provider)
    client = _get_client()
    return {
        "provider": provider,
        "connected": client.connected,
        "error": None if client.connected else f"Could not connect to {provider}",
    }


@app.post("/api/switch-model")
async def switch_model(body: dict):
    """Switch the active model."""
    global _active_model
    model = body.get("model", "llama3.2:latest")
    _active_model = model
    return {"model": model}


@app.get("/api/projects")
async def list_projects():
    """Return all project definitions."""
    return {"categories": CATEGORIES, "projects": PROJECTS}


@app.post("/api/run")
async def run_project(body: dict):
    """Run a specific project with the chosen provider/model and return the LLM response."""
    project_id = body.get("project_id", "")
    model = body.get("model", _active_model)
    provider = body.get("provider", _current_provider)

    if provider != _current_provider:
        _rebuild_client(provider)

    project = next((p for p in PROJECTS if p["id"] == project_id), None)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    prompt = project["prompt"]
    cat = project["cat"]

    client = _get_client()
    if not client.connected:
        return {"response": "⚠️ No LLM server connected. Start Ollama with `ollama serve` and try again.", "timing_ms": 0}

    import time
    t0 = time.time()

    cat_name = CATEGORIES[cat]["name"]
    system_prompt = f"You are a helpful assistant demonstrating {cat_name}. Be concise and accurate."

    if cat == 4:
        # RAG — include embedding retrieval context
        embedder = EmbedAdapter(client)
        try:
            emb = embedder.embed_query(prompt)
            context_note = f"(embedding dim={len(emb) if emb else 'N/A'})"
        except Exception:
            context_note = ""
        messages = [
            {"role": "system", "content": f"{system_prompt} {context_note}"},
            {"role": "user", "content": prompt},
        ]
        text = client.chat(messages=messages, model=model, temperature=0.3, max_tokens=512)
    elif cat == 5:
        # LangGraph — simulate agentic response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = client.chat(messages=messages, model=model, temperature=0.5, max_tokens=512)
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = client.chat(messages=messages, model=model, temperature=0.5, max_tokens=512)

    elapsed = round((time.time() - t0) * 1000)

    return {
        "response": text,
        "timing_ms": elapsed,
        "model": model,
        "provider": _current_provider,
        "project_id": project_id,
    }


@app.post("/api/run-stream")
async def run_project_stream(body: dict):
    """Stream a response for a specific project."""
    project_id = body.get("project_id", "")
    model = body.get("model", _active_model)
    provider = body.get("provider", _current_provider)

    if provider != _current_provider:
        _rebuild_client(provider)

    project = next((p for p in PROJECTS if p["id"] == project_id), None)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    prompt = project["prompt"]
    cat = project["cat"]
    cat_name = CATEGORIES[cat]["name"]
    system_prompt = f"You are a helpful assistant demonstrating {cat_name}. Be concise and accurate."

    client = _get_client()
    if not client.connected:
        async def noop():
            yield json.dumps({"error": "No LLM server connected"}) + "\n"
        return StreamingResponse(noop(), media_type="application/x-ndjson")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    async def stream():
        import httpx
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.5, "num_predict": 512},
        }
        url = f"http://127.0.0.1:11434/api/chat"
        try:
            async with httpx.AsyncClient(timeout=300) as hc:
                async with hc.stream("POST", url, json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            yield line + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the interactive dashboard HTML."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found</h1>")


def main():
    port = int(os.getenv("DASH_PORT", "3000"))
    print(f"🦙 Interactive Dashboard: http://127.0.0.1:{port}")
    print(f"   Provider: {_current_provider}")
    print(f"   Model: {_active_model}")
    print(f"   Press Ctrl+C to stop")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
