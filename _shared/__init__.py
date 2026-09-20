"""Shared utilities for llama-cpp-projects.

Provides auto-discovering LLM client for local inference servers
(ollama, llama.cpp, omlx) with unified API and per-category integration adapters.

Importing this package applies the repo's server defaults for any LLM_* setting
the environment has not already set, so `python3 <category>/demo_live.py` works
without sourcing env.sh first. Anything already exported wins, so env.sh, a
shell export and CI configuration all still override these.
"""

import os as _os

# Keep in sync with env.sh. Without LLM_PREFER the auto-probe tries Ollama
# (11434) first and, if that answers, sends chat there -- which on this machine
# 404s with "model 'llama3.2:latest' not found". Without LLM_EMBED_BASE_URL,
# embeddings go to the chat server, which answers 501 because it was not
# started with --embeddings.
_DEFAULTS = {
    "LLM_PREFER": "llama.cpp",
    "LLAMACPP_HOST": "127.0.0.1",
    "LLAMACPP_PORT": "8090",
    "LLM_CHAT_BASE_URL": "http://127.0.0.1:8090",
    "LLM_EMBED_BASE_URL": "http://127.0.0.1:8081",
    "LLM_RERANK_BASE_URL": "http://127.0.0.1:8082",
    "LLM_CHAT_MODEL": "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0",
    "LLM_EMBED_MODEL": "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0",
}
for _k, _v in _DEFAULTS.items():
    _os.environ.setdefault(_k, _v)
del _k, _v

from .llm_client import LLMClient, ServerInfo, NoServerError, get_client, detect_server
from .integrations import (
    ChatAdapter,
    StreamingChatAdapter,
    EmbedAdapter,
    LangChainAdapter,
    PromptEngineeringAdapter,
    RAGAdapter,
    LangGraphAdapter,
    MCPAdapter,
    get_adapter,
    live_projects_summary,
)

__all__ = [
    "LLMClient", "ServerInfo", "NoServerError", "get_client", "detect_server",
    "ChatAdapter", "StreamingChatAdapter", "EmbedAdapter",
    "LangChainAdapter", "PromptEngineeringAdapter",
    "RAGAdapter", "LangGraphAdapter", "MCPAdapter",
    "get_adapter", "live_projects_summary",
]
