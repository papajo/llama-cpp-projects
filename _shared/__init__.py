"""Shared utilities for llama-cpp-projects.

Provides auto-discovering LLM client for local inference servers
(ollama, llama.cpp, omlx) with unified API and per-category integration adapters.
"""

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
