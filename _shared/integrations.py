"""Integration adapters that wire projects to live LLM inference.

Each project gets an adapter that:
  1. Auto-discovers the running local LLM (ollama, llama.cpp, omlx)
  2. Provides the right API for that project's needs (chat, generate, embed, stream)
  3. Gracefully falls back to mock data when no server is available

Usage:
    from _shared.integrations import get_adapter

    adapter = get_adapter("langchain")         # Category 2: LangChain
    adapter = get_adapter("prompt-engineering") # Category 3
    adapter = get_adapter("rag")               # Category 4: Vector DB / RAG
    adapter = get_adapter("langgraph")          # Category 5: LangGraph
    adapter = get_adapter("mcp")               # Category 6: MCP
    adapter = get_adapter("chat")              # Generic chat
    adapter = get_adapter("embed")             # Embeddings only
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .llm_client import LLMClient, NoServerError, get_client, detect_server

logger = logging.getLogger(__name__)

# ── Module-level path setup ──────────────────────────────────────────────
# Ensure _shared is importable from any project directory
_SHARED_DIR = Path(__file__).parent.resolve()
_PROJECTS_ROOT = _SHARED_DIR.parent.resolve()
if str(_PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECTS_ROOT))


# ── Adapter Base ─────────────────────────────────────────────────────────

class AdapterBase:
    """Base class for project-specific LLM adapters.

    Each adapter provides a ``call_llm`` method and a ``summary``
    so the project can report connectivity to the user.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        default_model: Optional[str] = None,
    ):
        self._client = client or get_client(prefer="ollama", auto_start=False)
        self._default_model = default_model or "llama3.2:latest"

    @property
    def connected(self) -> bool:
        return self._client.connected

    @property
    def backend(self) -> Optional[str]:
        return self._client.backend if self._client.connected else None

    def discover_models(self) -> List[str]:
        """Return list of available models on the connected server."""
        return self._client.available_models if self._client.connected else []

    def summary(self) -> str:
        """Return a short connectivity string for CLI output."""
        if not self._client.connected:
            return "[offline — no LLM server detected]"
        return f"[live — {self._client.backend} @ {self._client.server.base_url}]"

    def vendor_summary(self) -> str:
        """Return the client's detailed summary box."""
        return self._client.summary()


# ── Concrete Adapters ────────────────────────────────────────────────────

class ChatAdapter(AdapterBase):
    """Generic chat adapter — used by most projects."""

    def call_llm(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Send a prompt to the LLM and return the response."""
        if not self._client.connected:
            return _mock_response(prompt, "chat")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self._client.chat(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class StreamingChatAdapter(ChatAdapter):
    """Adapter with streaming support for projects that show partial output."""

    def call_llm_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Stream a response from the LLM.

        If *on_chunk* is provided it is called with each token as it arrives.
        Returns the full accumulated text.
        """
        if not self._client.connected:
            text = _mock_response(prompt, "chat")
            if on_chunk:
                on_chunk(text)
            return text

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # The base client doesn't yet support streaming tokens,
        # so we fall back to non-streaming for now.
        return self.call_llm(prompt, system, model, temperature, max_tokens)


class EmbedAdapter(AdapterBase):
    """Embedding adapter — used by RAG / vector projects."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        if not self._client.connected:
            return _mock_embeddings(texts)
        return self._client.embed(texts, model="nomic-embed-text:latest")

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query string."""
        return self.embed([text])[0]


class LangChainAdapter(ChatAdapter):
    """Adapter for Category 2 (LangChain) projects.

    These projects often need to demonstrate routing / fallback / tool-use
    behaviour.  We provide a ``call_llm`` that accepts a list of messages
    (the LangChain ``BaseMessage`` format is converted internally).
    """

    def call_llm(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        return super().call_llm(prompt, system, model, temperature, max_tokens)

    def call_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Send a pre-formatted message list (role/content dicts)."""
        if not self._client.connected:
            return _mock_response(str(messages[-1].get("content", "")), "chat")
        return self._client.chat(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class PromptEngineeringAdapter(ChatAdapter):
    """Adapter for Category 3 (Prompt Engineering) projects.

    These projects run multiple trials with varying parameters and compare
    outputs.  We add a ``compare`` method for side-by-side evaluation.
    """

    def compare(
        self,
        prompt: str,
        variants: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run the same prompt with different parameters and return all outputs.

        Each *variant* is a dict of kwargs to pass to ``call_llm``
        (e.g. ``{"temperature": 0.1}``, ``{"temperature": 0.9}``).
        """
        results = []
        for i, kwargs in enumerate(variants):
            t0 = time.time()
            text = self.call_llm(prompt, model=model, **kwargs)
            elapsed = time.time() - t0
            results.append({"variant": i, "kwargs": kwargs, "text": text, "time_s": round(elapsed, 2)})
        return results


class RAGAdapter(AdapterBase):
    """Adapter for Category 4 (Vector DB / RAG) projects.

    Provides both embedding and generation so the project can demonstrate
    a complete retrieve-then-generate pipeline.
    """

    def __init__(self, client: Optional[LLMClient] = None):
        super().__init__(client)
        self._chat = ChatAdapter(client)
        self._embed = EmbedAdapter(client)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._embed.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed.embed_query(text)

    def generate(
        self,
        query: str,
        context: List[str],
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate an answer grounded in retrieved context chunks."""
        ctx_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(context))
        prompt = (
            f"Answer the question based **only** on the context below.\n\n"
            f"--- Context ---\n{ctx_block}\n\n"
            f"--- Question ---\n{query}\n\n"
            f"--- Answer ---"
        )
        return self._chat.call_llm(prompt, system=system, model=model)


class LangGraphAdapter(ChatAdapter):
    """Adapter for Category 5 (LangGraph) projects.

    Graph-based agents need multiple LLM calls per run.  We provide
    ``call_node`` to simulate a single graph-node invocation.
    """

    def call_node(
        self,
        node_name: str,
        input_text: str,
        state: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> str:
        """Call the LLM as a graph node, optionally passing state context."""
        context = ""
        if state:
            context = "Current state:\n" + json.dumps(state, indent=2) + "\n\n"
        prompt = f"{context}Node '{node_name}' received:\n{input_text}"
        return self.call_llm(prompt, model=model)


class MCPAdapter(ChatAdapter):
    """Adapter for Category 6 (MCP) projects.

    MCP servers often need to call the LLM to answer tool invocations
    or to provide model context to the host.
    """

    def tool_response(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        model: Optional[str] = None,
    ) -> str:
        """Generate a natural-language response for a tool invocation."""
        prompt = (
            f"You are an MCP server. The tool '{tool_name}' was called with:\n"
            f"{json.dumps(arguments, indent=2)}\n\n"
            f"Generate a helpful response."
        )
        return self.call_llm(prompt, model=model)


# ── Adapter Registry ─────────────────────────────────────────────────────

_ADAPTERS: Dict[str, type] = {
    "chat": ChatAdapter,
    "stream": StreamingChatAdapter,
    "embed": EmbedAdapter,
    "langchain": LangChainAdapter,
    "prompt-engineering": PromptEngineeringAdapter,
    "rag": RAGAdapter,
    "langgraph": LangGraphAdapter,
    "mcp": MCPAdapter,
}


def get_adapter(kind: str = "chat", **kwargs) -> AdapterBase:
    """Return an adapter instance for the given project category.

    Args:
        kind: One of ``chat``, ``stream``, ``embed``, ``langchain``,
              ``prompt-engineering``, ``rag``, ``langgraph``, ``mcp``.
        **kwargs: Forwarded to the adapter constructor (e.g. *default_model*).

    Returns:
        An adapter instance that will auto-detect the local LLM server.
    """
    cls = _ADAPTERS.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown adapter kind {kind!r}. "
            f"Available: {list(_ADAPTERS)}"
        )
    return cls(**kwargs)


def live_projects_summary() -> str:
    """Return a summary of all project categories and their LLM status."""
    client = get_client(prefer="ollama", auto_start=False)
    parts = [
        "╔══════════════════════════════════════════╗",
        "║  🦙  llama-cpp-projects  —  Live Status  ║",
        "╠══════════════════════════════════════════╣",
    ]
    if client.connected:
        parts.append(f"║  ✅  LLM Server: {client.backend:<26s}║")
        parts.append(f"║      {client.server.base_url:<34s}║")
        models = client.available_models[:3]
        if models:
            parts.append(f"║      Models: {', '.join(models):<29s}║")
    else:
        parts.append("║  ❌  No local LLM server detected      ║")
        parts.append("║  Start one with:                      ║")
        parts.append("║    ollama serve                       ║")
        parts.append("║    llama-server -m <model.gguf>       ║")
        parts.append("║    omlx serve <model>                 ║")

    parts.append("╠══════════════════════════════════════════╣")
    for kind, cls in sorted(_ADAPTERS.items()):
        name = f"{kind:<20s}"
        tag = "  ✅ live" if client.connected else "  ⏸️  offline"
        parts.append(f"║  📦 {name}{tag:<14s}║")
    parts.append("╚══════════════════════════════════════════╝")
    return "\n".join(parts)


# ── Mock Fallbacks ───────────────────────────────────────────────────────

_MOCK_RESPONSES: Dict[str, List[str]] = {
    "chat": [
        "This is a mock response. Start Ollama (ollama serve) for real LLM output.",
        "Mock: local inference not available. Run 'ollama serve' to enable.",
        "No LLM server detected. Install ollama and run 'ollama serve'.",
    ],
    "embed": [
        "[mock embedding — 768-dimensional zero vector]",
    ],
}


def _mock_response(prompt: str, kind: str = "chat") -> str:
    """Return a deterministic mock when no LLM server is available."""
    import hashlib
    idx = hash(hashlib.md5(prompt.encode()).hexdigest()) % len(_MOCK_RESPONSES.get(kind, _MOCK_RESPONSES["chat"]))
    return _MOCK_RESPONSES[kind][idx]


def _mock_embeddings(texts: List[str]) -> List[List[float]]:
    """Return mock zero-vector embeddings (768-d, same as nomic-embed-text)."""
    return [[0.0] * 768 for _ in texts]


__all__ = [
    "AdapterBase",
    "ChatAdapter",
    "StreamingChatAdapter",
    "EmbedAdapter",
    "LangChainAdapter",
    "PromptEngineeringAdapter",
    "RAGAdapter",
    "LangGraphAdapter",
    "MCPAdapter",
    "get_adapter",
    "live_projects_summary",
]
