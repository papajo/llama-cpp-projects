"""
LangChain agent with LoRA-adapter hot-swap — switches personalities
mid-conversation without reloading the base model.

Usage::

    from adapters import load_adapter_registry
    from persona_router import PersonaRouter
    from agent import LoraAgent

    registry = load_adapter_registry("adapters/registry.json")
    router = PersonaRouter(registry)

    agent = LoraAgent(
        base_url="http://127.0.0.1:8080",
        router=router,
    )

    # The agent auto-selects the right adapter based on the input
    response = agent.chat("I need to review this contract for liability issues")
    # -> Applies "legal-tone" LoRA, then generates response

    response = agent.chat("Write a haiku about debugging")
    # -> Switches to "creative" LoRA mid-conversation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Type

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from adapters import AdapterDef, AdapterRegistry
from persona_router import PersonaRouter

from .lora_manager import LoraManager

logger = logging.getLogger(__name__)


class LoraAgent(BaseChatModel):
    """
    LangChain chat model that hot-swaps LoRA adapters mid-conversation.

    Before each generation, the agent classifies the input and applies
    the appropriate LoRA adapter via ``POST /lora-adapters``.  This
    changes the model's "personality" without reloading the base model.

    Requires llama.cpp server started with ``--lora-init-without-apply``
    and all LoRA files listed via ``--lora`` / ``--lora-scaled``.
    The ``POST /lora-adapters`` endpoint accepts a JSON **array** of
    ``{"id": …, "scale": …}`` objects (see server README for details).
    """

    base_url: str = Field(
        default="http://127.0.0.1:8080",
        description="llama.cpp server URL.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096)
    request_timeout: float = Field(default=120.0)

    # Non-pydantic fields
    _lora_manager: LoraManager = None
    _router: PersonaRouter = None
    _current_adapter: Optional[AdapterDef] = None
    _client: Optional[httpx.Client] = None

    def __init__(
        self,
        router: PersonaRouter,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._router = router
        self._lora_manager = LoraManager(
            base_url=self.base_url,
            request_timeout=self.request_timeout,
        )

    # ---- Public API ----

    @property
    def current_persona(self) -> Optional[str]:
        """Return the name of the currently active persona."""
        return self._current_adapter.name if self._current_adapter else None

    def switch_persona(self, persona_name: str) -> None:
        """
        Manually switch to a specific persona by name.

        Args:
            persona_name: Adapter name from the registry.

        Raises:
            KeyError: If the persona is not in the registry.
        """
        if persona_name not in self._router.registry:
            raise KeyError(
                f"Persona {persona_name!r} not found. "
                f"Available: {list(self._router.registry)}"
            )
        adapter = self._router.registry[persona_name]
        self._apply_adapter(adapter)

    def list_personas(self) -> List[Dict[str, str]]:
        """List all available personas."""
        return self._router.list_personas()

    # ---- Internal adapter management ----

    def _select_and_apply_adapter(self, text: str) -> None:
        """
        Classify the input and apply the matching adapter if it differs
        from the current one.
        """
        adapter = self._router.route_by_keywords(text)
        if self._current_adapter is None or adapter.name != self._current_adapter.name:
            logger.info(
                "Switching persona: %s -> %s",
                self._current_adapter.name if self._current_adapter else "(none)",
                adapter.name,
            )
            self._apply_adapter(adapter)

    def _apply_adapter(self, adapter: AdapterDef) -> None:
        """Apply a LoRA adapter via the API."""
        self._lora_manager.apply_adapter(
            lora_id=adapter.lora_id,
            scale=adapter.scale,
        )
        self._current_adapter = adapter

    # ---- HTTP client ----

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.request_timeout,
            )
        return self._client

    # ---- LangChain BaseChatModel ----

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response, first switching to the right adapter
        based on the user's last message.
        """
        # Extract the user's last message for routing
        last_text = self._extract_last_user_text(messages)
        if last_text:
            self._select_and_apply_adapter(last_text)

        # Standard non-vision chat completion
        lm_messages = self._convert_messages(messages)
        client = self._get_client()

        body: Dict[str, Any] = {
            "messages": lm_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        response = client.post("/v1/chat/completions", json=body)
        if response.status_code >= 400:
            try:
                req = response.request
            except RuntimeError:
                req = None
            raise httpx.HTTPStatusError(
                f"llama.cpp returned {response.status_code}",
                request=req,
                response=response,
            )

        data = response.json()
        return self._parse_chat_result(data)

    def _parse_chat_result(self, data: Dict[str, Any]) -> ChatResult:
        """Parse OpenAI-compatible response."""
        generations: List[ChatGeneration] = []
        usage: Optional[Dict[str, int]] = None

        if "usage" in data:
            usage = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }

        for choice in data.get("choices", []):
            message = choice.get("message", {})
            content = message.get("content", "")
            finish_reason = choice.get("finish_reason", "")

            ai_message = AIMessage(
                content=content,
                response_metadata={
                    "finish_reason": finish_reason,
                    "model": data.get("model", ""),
                    "persona": self.current_persona,
                    "usage": usage,
                },
            )
            generations.append(ChatGeneration(message=ai_message))

        return ChatResult(generations=generations, llm_output=usage)

    @staticmethod
    def _extract_last_user_text(messages: Sequence[BaseMessage]) -> Optional[str]:
        """Get the text content from the last user/human message."""
        for msg in reversed(messages):
            if msg.type == "human":
                content = msg.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [
                        c["text"] for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    ]
                    return texts[0] if texts else None
        return None

    @staticmethod
    def _convert_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to OpenAI format."""
        result: List[Dict[str, str]] = []
        for msg in messages:
            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            role = role_map.get(msg.type, "user")
            content = msg.content
            if isinstance(content, list):
                texts = [
                    c["text"] for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                content = texts[0] if texts else str(content)
            result.append({"role": role, "content": str(content)})
        return result

    @property
    def _llm_type(self) -> str:
        return "llamacpp-lora-agent"

    def close(self) -> None:
        if self._client:
            self._client.close()
        if self._lora_manager:
            self._lora_manager.close()

    def __del__(self) -> None:
        self.close()
