"""Auto-discovering LLM client for local inference servers.

Detects and connects to any running local server:
  1. Ollama      → http://localhost:11434  (v1/chat/completions, v1/embeddings)
  2. llama.cpp   → http://localhost:8080   (v1/chat/completions, v1/embeddings, /completion)
  3. omlx        → http://localhost:8081   (v1/chat/completions)

Usage:
    from llm_client import LLMClient

    llm = LLMClient()                     # auto-detect
    llm = LLMClient(prefer="ollama")       # force a backend
    llm.chat("What is the capital of France?")
    llm.embed(["text1", "text2"])
    llm.generate("Once upon a time")
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

def _env_host(backend: str, default: str = "127.0.0.1") -> str:
    """Host override, e.g. LLAMACPP_HOST / OLLAMA_HOST."""
    key = backend.upper().replace(".", "").replace("-", "") + "_HOST"
    return os.environ.get(key) or os.environ.get("LLM_HOST") or default


def _env_port(backend: str, default: int) -> int:
    """Port override, e.g. LLAMACPP_PORT=8082 / OLLAMA_PORT=11434."""
    key = backend.upper().replace(".", "").replace("-", "") + "_PORT"
    raw = os.environ.get(key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


SERVER_PROFILES: List[Dict[str, Any]] = [
    {
        "name": "ollama",
        "host": _env_host("ollama"),
        "port": _env_port("ollama", 11434),
        "chat_path": "/v1/chat/completions",
        "embed_path": "/v1/embeddings",
        "models_path": "/api/tags",
        "health_path": "/api/tags",
        "default_model": "llama3.2:latest",
    },
    {
        "name": "llama.cpp",
        "host": _env_host("llamacpp"),
        "port": _env_port("llamacpp", 8080),
        "chat_path": "/v1/chat/completions",
        "embed_path": "/v1/embeddings",
        "completion_path": "/completion",
        "models_path": "/v1/models",
        "health_path": "/health",
        "default_model": "default",
    },
    {
        "name": "omlx",
        "host": _env_host("omlx"),
        "port": _env_port("omlx", 8081),
        "chat_path": "/v1/chat/completions",
        "embed_path": "/v1/embeddings",
        "models_path": "/v1/models",
        "health_path": "/health",
        "default_model": "default",
    },
]


class NoServerError(RuntimeError):
    """Raised when no local inference server can be found."""


@dataclass
class ServerInfo:
    """Describes a detected local inference server."""

    name: str
    base_url: str
    chat_path: str
    embed_path: str
    completion_path: Optional[str] = None
    models_path: str = "/v1/models"
    default_model: str = "default"
    detected_at: float = 0.0
    models: List[str] = field(default_factory=list)

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get("LLM_API_KEY") or os.environ.get(
            f"{self.name.upper()}_API_KEY"
        )


# ── Server Discovery ───────────────────────────────────────────────────


def probe_server(profile: dict, timeout: float = 1.5) -> Optional[ServerInfo]:
    """Check if a server is running at the given profile's address."""
    base = f"http://{profile['host']}:{profile['port']}"
    health_path = profile.get("health_path", "/health")
    try:
        req = urllib.request.Request(f"{base}{health_path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
    except (urllib.error.URLError, OSError, TimeoutError):
        return None

    # Fetch model list
    models = []
    try:
        models_path = profile.get("models_path", "/v1/models")
        req = urllib.request.Request(f"{base}{models_path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            if profile["name"] == "ollama" and "models" in data:
                models = [m["name"] for m in data["models"]]
            elif "data" in data:
                models = [m["id"] for m in data["data"]]
    except Exception:
        pass

    return ServerInfo(
        name=profile["name"],
        base_url=base,
        chat_path=profile.get("chat_path", "/v1/chat/completions"),
        embed_path=profile.get("embed_path", "/v1/embeddings"),
        completion_path=profile.get("completion_path"),
        models_path=profile.get("models_path", "/v1/models"),
        default_model=profile.get("default_model", "default"),
        detected_at=time.time(),
        models=models,
    )


def detect_server(
    prefer: Optional[str] = None, timeout: float = 1.5
) -> Optional[ServerInfo]:
    """Detect a running local inference server.

    Args:
        prefer: If set, try this backend first ("ollama", "llama.cpp", "omlx").
        timeout: HTTP timeout per probe in seconds.

    Returns:
        ServerInfo if a server is found, None otherwise.
    """
    profiles = list(SERVER_PROFILES)

    # Environment default, e.g. LLM_PREFER=llama.cpp
    prefer = prefer or os.environ.get("LLM_PREFER") or None

    # Move preferred profile to front
    if prefer:
        idx = next(
            (i for i, p in enumerate(profiles) if p["name"] == prefer), None
        )
        if idx is not None:
            profiles.insert(0, profiles.pop(idx))

    for profile in profiles:
        info = probe_server(profile, timeout=timeout)
        if info is not None:
            logger.info(
                "Detected %s at %s (models: %s)",
                info.name,
                info.base_url,
                info.models or "unknown",
            )
            return info

    return None


def ensure_server(
    prefer: Optional[str] = None,
    auto_start: bool = False,
    model: Optional[str] = None,
    timeout: float = 2.0,
) -> ServerInfo:
    """Detect a server or optionally start one.

    Args:
        prefer: Preferred backend name.
        auto_start: If True and no server is running, attempt to start one.
        model: Model to use when starting a server.
        timeout: Probe / start timeout.

    Returns:
        ServerInfo.

    Raises:
        NoServerError: If no server is found and auto_start is False or fails.
    """
    info = detect_server(prefer=prefer, timeout=timeout)
    if info is not None:
        return info

    if not auto_start:
        raise NoServerError(
            "No local LLM server detected. Start one with:\n"
            "  ollama serve            # port 11434\n"
            "  llama-server -m <gguf>  # port 8080\n"
            "  omlx serve <model>      # port 8081\n"
            "Or set auto_start=True to attempt auto-start."
        )

    # Try to auto-start
    if not prefer:
        prefer = "ollama"
    started = _start_server(prefer, model=model, timeout=timeout)
    if started:
        info = detect_server(prefer=prefer, timeout=timeout)
        if info is not None:
            return info

    raise NoServerError(
        f"Could not detect or start any local LLM server. "
        f"Tried: {[p['name'] for p in SERVER_PROFILES]}"
    )


def _start_server(
    backend: str, model: Optional[str] = None, timeout: float = 10.0
) -> bool:
    """Try to start a local inference server (non-blocking)."""
    if backend == "ollama":
        if not _find_binary("ollama"):
            logger.warning("ollama binary not found")
            return False
        # ollama may already be running as a service; if not, start it
        if detect_server(prefer="ollama", timeout=1.0):
            return True
        logger.info("Starting ollama server...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for it to come up
            for _ in range(int(timeout * 2)):
                time.sleep(0.5)
                if detect_server(prefer="ollama", timeout=0.5):
                    return True
        except FileNotFoundError:
            return False

    elif backend == "llama.cpp":
        binary = _find_binary("llama-server")
        if not binary:
            logger.warning("llama-server binary not found")
            return False
        model_path = model or _find_gguf_model()
        if not model_path:
            logger.warning("No GGUF model found for llama-server")
            return False
        logger.info("Starting llama-server with %s...", model_path)
        try:
            subprocess.Popen(
                [str(binary), "-m", str(model_path), "--port", "8080"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(int(timeout * 2)):
                time.sleep(0.5)
                if detect_server(prefer="llama.cpp", timeout=0.5):
                    return True
        except FileNotFoundError:
            return False

    elif backend == "omlx":
        binary = _find_binary("omlx")
        if not binary:
            logger.warning("omlx binary not found")
            return False
        model_name = model or "mlx-community/Llama-3.2-3B-Instruct-4bit"
        logger.info("Starting omlx with %s...", model_name)
        try:
            subprocess.Popen(
                [str(binary), "serve", model_name, "--port", "8081"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(int(timeout * 2)):
                time.sleep(0.5)
                if detect_server(prefer="omlx", timeout=0.5):
                    return True
        except FileNotFoundError:
            return False

    return False


def _find_binary(name: str) -> Optional[Path]:
    """Find a binary in PATH or common Homebrew locations."""
    # Check PATH
    for p in os.environ.get("PATH", "").split(":"):
        candidate = Path(p) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    # Check Homebrew on macOS
    for prefix in ["/opt/homebrew/bin", "/usr/local/bin"]:
        candidate = Path(prefix) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _find_gguf_model() -> Optional[Path]:
    """Look for a GGUF model in common locations."""
    env_dir = os.environ.get("LLM_MODELS_DIR")
    search_dirs = [
        Path(env_dir) if env_dir else None,
        Path.home() / "Models",          # case-sensitive filesystems
        Path.home() / "models",
        Path.home() / ".cache" / "llama.cpp",
        Path.home() / ".cache" / "lm-studio" / "models",
        Path("/opt/homebrew/share/llama.cpp/models"),
    ]
    # Embedding models cannot serve chat; skip them when picking a default.
    skip = ("embed", "bge-", "minilm", "e5-")
    for d in search_dirs:
        if d is None or not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):        # recurse: models live in subdirs
            if f.suffix.lower() != ".gguf":
                continue
            if f.name.startswith("ggml-vocab-"):
                continue                       # llama.cpp tokenizer fixtures
            if any(k in f.name.lower() for k in skip):
                continue
            return f
    return None


# ── Main Client ────────────────────────────────────────────────────────


class LLMClient:
    """Auto-discovering client for local LLM inference servers.

    Probes common ports to find a running server, then provides a unified
    interface for chat, text generation, and embeddings.

    Args:
        prefer: Prefer a specific backend ("ollama", "llama.cpp", "omlx").
        auto_start: If True, attempt to start a server if none is detected.
        model: Default model name to use for generation.
        timeout: Probe timeout in seconds.
    """

    def __init__(
        self,
        prefer: Optional[str] = None,
        auto_start: bool = False,
        model: Optional[str] = None,
        timeout: float = 2.0,
    ):
        self._prefer = prefer
        self._auto_start = auto_start
        self._model = model
        self._timeout = timeout
        self._server: Optional[ServerInfo] = None
        self._detected = False

    @property
    def server(self) -> Optional[ServerInfo]:
        """Get the detected server info, probing if needed."""
        if not self._detected:
            try:
                self._server = ensure_server(
                    prefer=self._prefer,
                    auto_start=self._auto_start,
                    model=self._model,
                    timeout=self._timeout,
                )
            except NoServerError:
                self._server = None
            self._detected = True
        return self._server

    @property
    def connected(self) -> bool:
        """Whether a server has been detected."""
        return self.server is not None

    @property
    def backend(self) -> Optional[str]:
        """Name of the detected backend."""
        return self.server.name if self.server else None

    @property
    def available_models(self) -> List[str]:
        """List of models available on the detected server."""
        if not self.server:
            return []
        return self.server.models

    # ── Chat (OpenAI-compatible) ────────────────────────────────────

    def chat(
        self,
        messages: Any,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: Either a string prompt (converted to user message) or
                      a list of dicts with 'role' and 'content'.
            model: Model name (uses server default if not set).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stream: If True, return the response as it arrives (partial).
            **kwargs: Additional parameters passed to the API.

        Returns:
            Generated text content.

        Raises:
            NoServerError: If no server is available.
        """
        self._require_server()

        model = model or self._model or self.server.default_model

        # Normalize a string prompt to a chat message
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }

        data = self._request(
            method="POST",
            path=self.server.chat_path,
            payload=payload,
        )

        if stream:
            return data  # raw streaming response
        return data["choices"][0]["message"]["content"].strip()

    # ── Text Generation ─────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """Generate text from a prompt.

        Uses /v1/chat/completions by default (wraps prompt as user message).
        Falls back to /completion (llama.cpp legacy) if available and prefer.

        Args:
            prompt: Input text.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        # For llama.cpp, we could use the legacy /completion endpoint
        # for simpler text-in-text-out (non-chat) usage
        if self.server.completion_path and self._prefer == "llama.cpp":
            payload = {
                "prompt": prompt,
                "temperature": temperature,
                "n_predict": max_tokens,
                **kwargs,
            }
            data = self._request(
                method="POST",
                path=self.server.completion_path,
                payload=payload,
            )
            return data.get("content", "").strip()

        # Default: use chat API
        return self.chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens)

    # ── Embeddings ──────────────────────────────────────────────────

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            model: Model name (must support embeddings).

        Returns:
            List of embedding vectors (each is a list of floats).

        Raises:
            NoServerError: If no server is available.
        """
        self._require_server()

        if isinstance(texts, str):
            texts = [texts]

        # A plain llama-server serves ONE model and only offers /v1/embeddings
        # when started with --embeddings, so chat and embeddings usually live on
        # different processes. Prefer the dedicated embeddings endpoint when one
        # is configured; fall back to the detected server otherwise.
        embed_base = self._embed_base_url()
        model = (
            model
            or os.environ.get("LLM_EMBED_MODEL")
            or self._model
            or self.server.default_model
        )

        payload = {"model": model, "input": texts}
        data = self._request(
            method="POST",
            path=self.server.embed_path,
            payload=payload,
            base_url=embed_base,
        )

        # Handle both OpenAI format and ollama format
        if "data" in data:
            return [item["embedding"] for item in data["data"]]
        elif "embeddings" in data:
            return data["embeddings"]
        else:
            raise RuntimeError(f"Unexpected embeddings response: {list(data.keys())}")

    def _embed_base_url(self) -> Optional[str]:
        """Base URL for embeddings, when it differs from the chat server.

        Returns LLM_EMBED_BASE_URL when set, else None meaning "use the
        detected server". Importing the _shared package sets that variable to
        this repo's embeddings port, so it is normally populated; a caller that
        genuinely wants embeddings from the chat server can unset it.
        """
        explicit = os.environ.get("LLM_EMBED_BASE_URL")
        return explicit.rstrip("/") if explicit else None

    def embed_query(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get embedding for a single query string."""
        return self.embed([text], model=model)[0]

    # ── Tokenize ────────────────────────────────────────────────────

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate token count by sending a short completion."""
        # Simple approximation from response usage
        try:
            resp = self.chat(
                f"Count the tokens in: {text[:50]}",
                model=model,
                max_tokens=1,
            )
            # Approximate: 1 token ≈ 4 chars for English
            return max(1, len(text) // 4)
        except Exception:
            return len(text) // 4

    # ── System Info ──────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the connection status."""
        if not self.server:
            parts = [
                "╔══════════════════════════════════╗",
                "║  🔌 LLM Client: NOT CONNECTED    ║",
                "╠══════════════════════════════════╣",
            ]
            if self._detected:
                parts.append("║  No server found after probing.   ║")
            parts.append("║  Start one with:                 ║")
            parts.append("║    ollama serve                  ║")
            parts.append("║    llama-server -m <model.gguf> ║")
            parts.append("║    omlx serve <model>           ║")
            parts.append("╚══════════════════════════════════╝")
            return "\n".join(parts)

        s = self.server
        models_str = ", ".join(s.models[:5]) if s.models else "default"
        if len(s.models) > 5:
            models_str += f" … (+{len(s.models) - 5} more)"

        return (
            f"╔══════════════════════════════════╗\n"
            f"║  🔌 LLM Client: CONNECTED        ║\n"
            f"║  Backend : {s.name:30s}║\n"
            f"║  Server  : {s.base_url:30s}║\n"
            f"║  Models  : {models_str:30s}║\n"
            f"╚══════════════════════════════════╝"
        )

    def __repr__(self) -> str:
        if self.server:
            return f"LLMClient({self.server.name} @ {self.server.base_url})"
        return "LLMClient(disconnected)"

    # ── Internal ─────────────────────────────────────────────────────

    def _require_server(self) -> None:
        """Ensure a server is connected, raise otherwise."""
        if not self.server:
            if not self._detected:
                self.server  # triggers detection
            if not self.server:
                raise NoServerError(
                    "No local LLM server detected. Cannot make API calls."
                )

    def _request(
        self,
        method: str = "POST",
        path: str = "/v1/chat/completions",
        payload: Optional[dict] = None,
        base_url: Optional[str] = None,
    ) -> dict:
        """Make an HTTP request to the server.

        base_url overrides the detected server, for capabilities that live on
        a different process (embeddings, reranking).
        """
        url = f"{base_url or self.server.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
        }
        api_key = self.server.api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else str(e)
            raise RuntimeError(
                f"Server error {e.code} on {method} {path}: {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Connection failed to {url}: {e.reason}"
            ) from e


# ── Convenience Factory ────────────────────────────────────────────────

_default_client: Optional[LLMClient] = None


def get_client(
    prefer: Optional[str] = None,
    auto_start: bool = False,
    model: Optional[str] = None,
    reset: bool = False,
) -> LLMClient:
    """Get or create the shared default LLM client.

    Args:
        prefer: Preferred backend.
        auto_start: Whether to auto-start a server.
        model: Default model name.
        reset: Force re-detection of server.

    Returns:
        LLMClient instance (singleton unless reset=True).
    """
    global _default_client
    if _default_client is None or reset:
        _default_client = LLMClient(prefer=prefer, auto_start=auto_start, model=model)
    return _default_client


__all__ = [
    "LLMClient",
    "ServerInfo",
    "NoServerError",
    "detect_server",
    "ensure_server",
    "get_client",
]
