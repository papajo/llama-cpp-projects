"""
LangChain chat model for llama.cpp vision-enabled endpoints.

Wraps llama.cpp's ``/v1/chat/completions`` endpoint with support for
multimodal messages (text + images via base64 data URIs).

Usage::

    from extraction_chain import VisionChatModel

    model = VisionChatModel(base_url="http://127.0.0.1:8080")
    response = model.invoke_with_images(
        prompt="What's shown in this image?",
        images=["data:image/png;base64,iVBOR...", ...],
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field


class VisionChatModel(BaseChatModel):
    """
    LangChain chat model for llama.cpp's vision-enabled endpoint.

    Sends messages to ``/v1/chat/completions`` with multimodal content
    blocks (text + ``image_url``).  Requires a llama.cpp server started
    with ``--mmproj`` and a vision GGUF model.
    """

    base_url: str = Field(
        default="http://127.0.0.1:8080",
        description="llama.cpp server URL with vision support.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096)
    request_timeout: float = Field(default=120.0)

    _client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.request_timeout,
            )
        return self._client

    def invoke_with_images(
        self,
        prompt: str,
        images: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Send a prompt with images for vision-based inference.

        Args:
            prompt: Text prompt.
            images: List of base64 data URIs (``data:image/...;base64,...``).
            system_prompt: Optional system-level instruction.

        Returns:
            Generated text response.
        """
        messages = self._build_multimodal_messages(prompt, images, system_prompt)
        client = self._get_client()

        body: Dict[str, Any] = {
            "model": "vision",  # llama.cpp routes to the loaded vision model
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=body)
        if response.status_code >= 400:
            try:
                req = response.request
            except RuntimeError:
                req = None
            raise httpx.HTTPStatusError(
                f"llama.cpp vision endpoint returned {response.status_code}",
                request=req,
                response=response,
            )

        data = response.json()
        return self._extract_content(data)

    def _build_multimodal_messages(
        self,
        prompt: str,
        images: List[str],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build OpenAI-compatible multimodal messages.

        Returns a message list where the user message contains both
        text and image content parts.
        """
        messages: List[Dict[str, Any]] = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        # Build content parts: text + images
        content_parts: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt}
        ]
        for uri in images:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": uri},
            })

        messages.append({
            "role": "user",
            "content": content_parts,
        })

        return messages

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        """Extract text from an OpenAI-compatible response."""
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    # ---- BaseChatModel interface (non-vision) ----

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Standard text-only generate (falls back to non-vision)."""
        lm_messages = self._convert_standard_messages(messages)
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
        generations: List[ChatGeneration] = []
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            ai_msg = AIMessage(content=msg.get("content", ""))
            generations.append(ChatGeneration(message=ai_msg))

        return ChatResult(generations=generations)

    @staticmethod
    def _convert_standard_messages(
        messages: Sequence[BaseMessage],
    ) -> List[Dict[str, str]]:
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
        return "llamacpp-vision"

    def close(self) -> None:
        if self._client:
            self._client.close()

    def __del__(self) -> None:
        self.close()
