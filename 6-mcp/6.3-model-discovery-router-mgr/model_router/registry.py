"""Model registry — discover, query, and manage LLM models."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class ModelInfo:
    """Information about an LLM model."""

    id: str
    object: str = "model"
    owned_by: str = "unknown"
    permission: List[Dict[str, Any]] = field(default_factory=list)
    context_length: int = 0
    type: str = "unknown"  # chat, completion, embedding

    @classmethod
    def from_openai_response(cls, data: Dict[str, Any]) -> ModelInfo:
        # llama.cpp reports the loaded context window as meta.n_ctx; plain
        # OpenAI payloads have no meta block, so fall back to 0.
        meta = data.get("meta") or {}
        return cls(
            id=data.get("id", "unknown"),
            object=data.get("object", "model"),
            owned_by=data.get("owned_by", "unknown"),
            permission=data.get("permission", []),
            context_length=int(meta.get("n_ctx", 0) or 0),
        )


class RoutingStrategy(Enum):
    FIRST_AVAILABLE = "first_available"
    ROUND_ROBIN = "round_robin"
    BY_NAME = "by_name"


@dataclass
class ModelRegistry:
    """Registry of available models, discovered via llama.cpp API."""

    server_url: str = "http://localhost:8080"
    models: Dict[str, ModelInfo] = field(default_factory=dict)
    _round_robin_index: int = 0

    def discover(self) -> List[ModelInfo]:
        """Fetch available models from llama.cpp ``/v1/models``."""
        try:
            req = urllib.request.Request(
                f"{self.server_url}/v1/models",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:
            raise ModelDiscoveryError(f"Failed to discover models: {exc}") from exc

        models: List[ModelInfo] = []
        for item in body.get("data", []):
            info = ModelInfo.from_openai_response(item)
            self.models[info.id] = info
            models.append(info)
        return models

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        return self.models.get(model_id)

    def list_models(self) -> List[ModelInfo]:
        return list(self.models.values())

    def route(
        self,
        strategy: RoutingStrategy = RoutingStrategy.FIRST_AVAILABLE,
        preferred_model: Optional[str] = None,
    ) -> Optional[str]:
        """Select a model ID based on the routing strategy."""
        if not self.models:
            return None

        if strategy == RoutingStrategy.BY_NAME and preferred_model:
            if preferred_model in self.models:
                return preferred_model
            return None

        model_ids = sorted(self.models.keys())

        if strategy == RoutingStrategy.ROUND_ROBIN:
            idx = self._round_robin_index % len(model_ids)
            self._round_robin_index = (self._round_robin_index + 1) % len(model_ids)
            return model_ids[idx]

        # FIRST_AVAILABLE
        return model_ids[0]

    def discover_if_empty(self) -> List[ModelInfo]:
        if not self.models:
            return self.discover()
        return self.list_models()

    @staticmethod
    def infer_type(model_id: str) -> str:
        mid = model_id.lower()
        if "embed" in mid:
            return "embedding"
        if "instruct" in mid or "chat" in mid:
            return "chat"
        return "completion"


class ModelDiscoveryError(Exception):
    pass
