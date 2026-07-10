"""Lightweight client for llama.cpp chat completions."""

from __future__ import annotations

import json
from typing import List, Optional


class LlamaClient:
    """Client for llama.cpp ``/v1/chat/completions``."""

    def __init__(self, server_url: str = "http://localhost:8080", timeout: int = 120):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        import urllib.error
        import urllib.request

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model is not None:
            payload["model"] = model
        data = json.dumps(payload).encode()
        try:
            req = urllib.request.Request(
                f"{self.server_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            raise LlamaError(str(exc)) from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaError(f"Unexpected response: {body}") from exc


class LlamaError(Exception):
    pass
