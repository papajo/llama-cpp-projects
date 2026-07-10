"""
LangChain ``RunnableRouter`` — multi-model router for llama.cpp router server.

Usage::

    router = RunnableRouter(base_url="http://127.0.0.1:8080")
    router.register_model("coder", tags=["code", "fast", "generation"])
    router.register_model("chat",  tags=["chat", "creative", "slow"])
    router.register_model("embed", tags=["embedding"])

    # Route by tag
    response = router.invoke(
        [HumanMessage(content="write a fib function")],
        config={"routing_key": "code"},   # matches "coder" via tags
    )

    # Route by explicit model alias
    response = router.invoke(
        [HumanMessage(content="tell me a story")],
        config={"model_alias": "chat"},
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
)

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from .exceptions import (
    ModelColdStartError,
    ModelNotAvailableError,
    NoSuitableModelError,
    RouterConnectionError,
)
from .preset_parser import PresetConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight model descriptor
# ---------------------------------------------------------------------------


@dataclass
class RouterModel:
    """A model available on the router server with routing metadata."""

    alias: str
    """Model alias as registered on the router (e.g. ``coder``)."""

    tags: List[str] = field(default_factory=list)
    """Tags used for routing lookups."""

    description: str = ""
    """Human-readable description."""

    cold_start_seconds: float = 0.0
    """Expected cold-start latency. 0 = unknown, probe at first use."""


# ---------------------------------------------------------------------------
# Routing configuration
# ---------------------------------------------------------------------------


class RoutingConfig(BaseModel):
    """Configuration for a single routing invocation."""

    routing_key: Optional[str] = Field(
        default=None,
        description=(
            "Tag or keyword used to select a model. "
            "The router picks the model whose tags contain this key."
        ),
    )
    model_alias: Optional[str] = Field(
        default=None,
        description="Explicit model alias — bypasses tag-based routing.",
    )
    fallback_aliases: List[str] = Field(
        default_factory=list,
        description="Ordered list of fallback aliases if the primary model is unavailable.",
    )
    max_cold_start_retries: int = Field(
        default=3,
        description="Number of cold-start retries before raising.",
        ge=1,
    )
    cold_start_retry_delay: float = Field(
        default=2.0,
        description="Base delay in seconds between cold-start retries (exponential backoff).",
        gt=0,
    )
    request_timeout: float = Field(
        default=120.0,
        description="HTTP request timeout in seconds (long enough for cold-starts).",
        gt=0,
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible request/response helpers
# ---------------------------------------------------------------------------


def _convert_messages(
    messages: Sequence[BaseMessage],
) -> List[Dict[str, str]]:
    """Convert LangChain messages to OpenAI-format dicts."""
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }
    result: List[Dict[str, str]] = []
    for msg in messages:
        role = role_map.get(msg.type, "user")
        content = msg.content
        if isinstance(content, list):
            # Handle multi-modal content — take first text block
            texts = [
                c["text"] for c in content if isinstance(c, dict) and c.get("type") == "text"
            ]
            content = texts[0] if texts else str(content)
        result.append({"role": role, "content": str(content)})
    return result


def _build_chat_request(
    model_alias: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """Build an OpenAI-compatible chat completion request body."""
    body: Dict[str, Any] = {
        "model": model_alias,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


# ---------------------------------------------------------------------------
# Router Chat Model
# ---------------------------------------------------------------------------


class RunnableRouter(BaseChatModel):
    """
    LangChain chat model that routes requests to models on a llama.cpp router server.

    The router server hosts multiple GGUF models behind one OpenAI-compatible
    endpoint.  This model selects the right backend per-request via **tags**
    or an explicit **alias**, and handles cold-start retries when a model
    is asleep (``--sleep-idle-seconds``).
    """

    # ---- Pydantic fields ----

    base_url: str = Field(
        default="http://127.0.0.1:8080",
        description="Base URL of the llama.cpp router server.",
    )
    default_routing_key: Optional[str] = Field(
        default=None,
        description="Default routing tag when none is provided in config.",
    )
    default_model_alias: Optional[str] = Field(
        default=None,
        description="Default model alias when no routing key or alias is provided.",
    )
    default_max_tokens: Optional[int] = Field(
        default=None,
        description="Default max_tokens for generation.",
    )
    default_temperature: float = Field(
        default=0.0,
        description="Default sampling temperature.",
        ge=0.0,
        le=2.0,
    )
    cold_start_retry_delay: float = Field(
        default=2.0,
        description="Base delay before retrying a cold-start model (exponential backoff).",
    )
    max_cold_start_retries: int = Field(
        default=3,
        description="Max cold-start retry attempts.",
    )
    request_timeout: float = Field(
        default=120.0,
        description="HTTP request timeout.",
    )
    verify_ssl: bool = Field(
        default=True,
        description="Verify SSL certificates when connecting to the router.",
    )

    # ---- Internal state (not serialized) ----

    _models: Dict[str, RouterModel] = {}
    _client: Optional[httpx.AsyncClient] = None
    _sync_client: Optional[httpx.Client] = None

    # ---- Pydantic init ----

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Use the model_post_init pattern for setup
        self._models = {}

    def _get_client(self) -> httpx.Client:
        """Lazy-init sync HTTP client."""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                timeout=self.request_timeout,
                verify=self.verify_ssl,
            )
        return self._sync_client

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Lazy-init async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.request_timeout,
                verify=self.verify_ssl,
            )
        return self._client

    # ---- Model registration ----

    def register_model(self, model: RouterModel) -> None:
        """Register a model for routing."""
        self._models[model.alias] = model

    def register_presets(self, presets: Dict[str, PresetConfig]) -> None:
        """Register models from a parsed preset INI."""
        for name, preset in presets.items():
            self.register_model(
                RouterModel(
                    alias=preset.name,
                    tags=preset.tags,
                    description=preset.description,
                    cold_start_seconds=preset.cold_start_seconds,
                )
            )

    def unregister_model(self, alias: str) -> None:
        """Remove a registered model."""
        self._models.pop(alias, None)

    @property
    def registered_models(self) -> Dict[str, RouterModel]:
        """Return the dict of registered models (alias → RouterModel)."""
        return dict(self._models)

    # ---- Routing logic ----

    def _select_model(self, config: RoutingConfig) -> str:
        """
        Resolve a model alias from the routing config.

        Priority:
        1. ``config.model_alias`` — explicit alias
        2. ``config.routing_key`` — tag-based lookup
        3. ``self.default_model_alias``
        4. ``self.default_routing_key``
        5. Fallback to the first registered model

        Raises:
            NoSuitableModelError: No model matched.
        """
        # 1. Explicit alias
        if config.model_alias:
            if config.model_alias not in self._models:
                raise ModelNotAvailableError(
                    f"Model alias {config.model_alias!r} is not registered. "
                    f"Available: {list(self._models)}"
                )
            return config.model_alias

        # 2. Tag-based routing
        key = config.routing_key or self.default_routing_key
        if key:
            matches = [
                name
                for name, m in self._models.items()
                if key.lower() in [t.lower() for t in m.tags]
            ]
            if not matches:
                raise NoSuitableModelError(
                    f"No registered model has tags matching {key!r}. "
                    f"Available models: {list(self._models)}"
                )
            # Prefer the first match (most specific/smallest)
            return matches[0]

        # 3. Default alias
        if self.default_model_alias:
            return self.default_model_alias

        # 4. Default routing key
        if self.default_routing_key:
            matches = [
                name
                for name, m in self._models.items()
                if self.default_routing_key.lower() in [t.lower() for t in m.tags]
            ]
            if matches:
                return matches[0]

        # 5. First registered
        if self._models:
            return next(iter(self._models))

        raise NoSuitableModelError(
            "No models registered and no default alias configured."
        )

    def _get_fallback_chain(self, config: RoutingConfig, primary: str) -> List[str]:
        """Build the ordered list of model aliases to try."""
        chain = [primary]
        for alias in config.fallback_aliases:
            if alias in self._models and alias not in chain:
                chain.append(alias)
        return chain

    # ---- Cold-start handling ----

    def _is_cold_start_error(self, response: httpx.Response) -> bool:
        """
        Detect a cold-start signal from the router.

        The router returns:
        - HTTP 503 (Service Unavailable) when a model is unloading/loading
        - HTTP 504 (Gateway Timeout) when a model takes too long to wake
        - A JSON body with ``"error"`` containing "cold" or "sleep"
        """
        if response.status_code in (503, 504):
            return True
        if response.status_code == 200:
            return False
        try:
            body = response.json()
            error_text = str(body.get("error", {})).lower()
            if "cold" in error_text or "sleep" in error_text or "unavailable" in error_text:
                return True
        except Exception:
            pass
        return False

    async def _a_is_cold_start_error(self, response: httpx.Response) -> bool:
        """Async version of _is_cold_start_error."""
        return self._is_cold_start_error(response)

    # ---- Core HTTP request (sync) ----

    def _chat_request(
        self,
        model_alias: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> httpx.Response:
        """
        Send a chat completion request to the router, with cold-start retries.

        Raises:
            ModelColdStartError: If all cold-start retries are exhausted.
            RouterConnectionError: If the server is unreachable.
        """
        client = self._get_client()
        body = _build_chat_request(
            model_alias=model_alias,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = client.post("/v1/chat/completions", json=body)
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                raise RouterConnectionError(
                    f"Cannot connect to router at {self.base_url}: {e}"
                ) from e

            if response.is_success:
                return response

            if self._is_cold_start_error(response):
                delay = retry_delay * (2 ** (attempt - 1))  # exponential backoff
                logger.warning(
                    "Cold-start for model %r (attempt %d/%d, retrying in %.1fs)",
                    model_alias,
                    attempt,
                    max_retries,
                    delay,
                )
                if attempt < max_retries:
                    time.sleep(delay)
                    last_error = ModelColdStartError(
                        f"Model {model_alias!r} is cold-starting "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    continue
                else:
                    raise ModelColdStartError(
                        f"Model {model_alias!r} failed to cold-start "
                        f"after {max_retries} retries"
                    ) from last_error

            # Non-cold-start error
            response.raise_for_status()

        # Should not reach here, but satisfy the type checker
        raise ModelColdStartError(
            f"Model {model_alias!r} cold-start failed after {max_retries} retries"
        )

    # ---- Async HTTP request ----

    async def _a_chat_request(
        self,
        model_alias: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> httpx.Response:
        """Async version of _chat_request."""
        client = await self._get_async_client()
        body = _build_chat_request(
            model_alias=model_alias,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post("/v1/chat/completions", json=body)
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                raise RouterConnectionError(
                    f"Cannot connect to router at {self.base_url}: {e}"
                ) from e

            if response.is_success:
                return response

            if await self._a_is_cold_start_error(response):
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Cold-start for model %r (attempt %d/%d, retrying in %.1fs)",
                    model_alias,
                    attempt,
                    max_retries,
                    delay,
                )
                if attempt < max_retries:
                    await asyncio.sleep(delay)
                    last_error = ModelColdStartError(
                        f"Model {model_alias!r} is cold-starting "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    continue
                else:
                    raise ModelColdStartError(
                        f"Model {model_alias!r} failed to cold-start "
                        f"after {max_retries} retries"
                    ) from last_error

            response.raise_for_status()

        raise ModelColdStartError(
            f"Model {model_alias!r} cold-start failed after {max_retries} retries"
        )

    # ---- LangChain BaseChatModel interface ----

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response by routing to the appropriate model."""
        config = self._extract_routing_config(kwargs)
        model_alias = self._select_model(config)

        lm_messages = _convert_messages(messages)
        max_tokens = kwargs.get("max_tokens") or self.default_max_tokens
        temperature = kwargs.get("temperature", self.default_temperature)
        max_retries = config.max_cold_start_retries
        retry_delay = config.cold_start_retry_delay

        chain = self._get_fallback_chain(config, model_alias)
        last_error: Optional[Exception] = None

        for alias in chain:
            try:
                response = self._chat_request(
                    model_alias=alias,
                    messages=lm_messages,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
            except (ModelColdStartError, ModelNotAvailableError) as e:
                logger.warning("Fallback from %r: %s", alias, e)
                last_error = e
                continue

            data = response.json()
            return self._parse_chat_result(data)

        raise NoSuitableModelError(
            f"No model could fulfill the request after trying: {chain}"
        ) from last_error

    async def _agenerate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate."""
        config = self._extract_routing_config(kwargs)
        model_alias = self._select_model(config)

        lm_messages = _convert_messages(messages)
        max_tokens = kwargs.get("max_tokens") or self.default_max_tokens
        temperature = kwargs.get("temperature", self.default_temperature)
        max_retries = config.max_cold_start_retries
        retry_delay = config.cold_start_retry_delay

        chain = self._get_fallback_chain(config, model_alias)
        last_error: Optional[Exception] = None

        for alias in chain:
            try:
                response = await self._a_chat_request(
                    model_alias=alias,
                    messages=lm_messages,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
            except (ModelColdStartError, ModelNotAvailableError) as e:
                logger.warning("Fallback from %r: %s", alias, e)
                last_error = e
                continue

            data = response.json()
            return self._parse_chat_result(data)

        raise NoSuitableModelError(
            f"No model could fulfill the request after trying: {chain}"
        ) from last_error

    def _stream(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream a response from the routed model."""
        config = self._extract_routing_config(kwargs)
        model_alias = self._select_model(config)

        lm_messages = _convert_messages(messages)
        max_tokens = kwargs.get("max_tokens") or self.default_max_tokens
        temperature = kwargs.get("temperature", self.default_temperature)

        client = self._get_client()
        body = _build_chat_request(
            model_alias=model_alias,
            messages=lm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        try:
            with client.stream("POST", "/v1/chat/completions", json=body) as response:
                if not response.is_success:
                    if self._is_cold_start_error(response):
                        raise ModelColdStartError(
                            f"Model {model_alias!r} cold-starting when streaming"
                        )
                    response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            break
                        import json

                        data = json.loads(payload)
                        chunk = self._parse_stream_chunk(data)
                        if chunk is not None:
                            if run_manager:
                                run_manager.on_llm_new_token(
                                    chunk.text, chunk=chunk
                                )
                            yield chunk
        except httpx.TimeoutException:
            raise RouterConnectionError(
                f"Stream timed out connecting to {self.base_url}"
            )

    async def _astream(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream."""
        config = self._extract_routing_config(kwargs)
        model_alias = self._select_model(config)

        lm_messages = _convert_messages(messages)
        max_tokens = kwargs.get("max_tokens") or self.default_max_tokens
        temperature = kwargs.get("temperature", self.default_temperature)

        client = await self._get_async_client()
        body = _build_chat_request(
            model_alias=model_alias,
            messages=lm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        try:
            async with client.stream(
                "POST", "/v1/chat/completions", json=body
            ) as response:
                if not response.is_success:
                    if await self._a_is_cold_start_error(response):
                        raise ModelColdStartError(
                            f"Model {model_alias!r} cold-starting when streaming"
                        )
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            break
                        import json

                        data = json.loads(payload)
                        chunk = self._parse_stream_chunk(data)
                        if chunk is not None:
                            if run_manager:
                                await run_manager.on_llm_new_token(
                                    chunk.text, chunk=chunk
                                )
                            yield chunk
        except httpx.TimeoutException:
            raise RouterConnectionError(
                f"Stream timed out connecting to {self.base_url}"
            )

    # ---- Response parsing ----

    def _parse_chat_result(self, data: Dict[str, Any]) -> ChatResult:
        """Parse an OpenAI-compatible response into a ChatResult."""
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
                    "usage": usage,
                },
            )
            gen = ChatGeneration(message=ai_message)
            generations.append(gen)

        return ChatResult(generations=generations, llm_output=usage)

    def _parse_stream_chunk(
        self, data: Dict[str, Any]
    ) -> Optional[ChatGenerationChunk]:
        """Parse a streaming SSE chunk into a ChatGenerationChunk."""
        choices = data.get("choices", [])
        if not choices:
            return None

        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        if not content:
            return None

        finish_reason = choices[0].get("finish_reason")

        chunk = AIMessageChunk(
            content=content,
            response_metadata={"finish_reason": finish_reason},
        )
        return ChatGenerationChunk(message=chunk)

    # ---- Config extraction ----

    def _extract_routing_config(self, kwargs: Dict[str, Any]) -> RoutingConfig:
        """Extract routing config from kwargs passed to invoke/generate."""
        config: Optional[RunnableConfig] = kwargs.get("config")
        if config and isinstance(config, dict):
            routing_key = config.get("routing_key") or config.get("tags")
            model_alias = config.get("model_alias")
            fallback_aliases: List[str] = config.get("fallback_aliases", [])
            max_retries = config.get(
                "max_cold_start_retries", self.max_cold_start_retries
            )
            retry_delay = config.get(
                "cold_start_retry_delay", self.cold_start_retry_delay
            )
            timeout = config.get("request_timeout", self.request_timeout)
        else:
            # Fall back to kwargs directly
            routing_key = kwargs.get("routing_key", self.default_routing_key)
            model_alias = kwargs.get("model_alias", self.default_model_alias)
            fallback_aliases = kwargs.get("fallback_aliases", [])
            max_retries = self.max_cold_start_retries
            retry_delay = self.cold_start_retry_delay
            timeout = self.request_timeout

        return RoutingConfig(
            routing_key=routing_key,
            model_alias=model_alias,
            fallback_aliases=fallback_aliases,
            max_cold_start_retries=max_retries,
            cold_start_retry_delay=retry_delay,
            request_timeout=timeout,
        )

    # ---- Required by BaseChatModel ----

    @property
    def _llm_type(self) -> str:
        return "llamacpp-router"

    # ---- Cleanup ----

    def close(self) -> None:
        """Close the sync HTTP client."""
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()

    async def aclose(self) -> None:
        """Close the async HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def __del__(self) -> None:
        self.close()
