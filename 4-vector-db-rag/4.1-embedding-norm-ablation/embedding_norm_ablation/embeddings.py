"""Embedding generation via llama.cpp server with L2 normalization support."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

import numpy as np


class EmbeddingClient:
    """Client for generating embeddings from a llama.cpp server.

    Uses the OpenAI-compatible ``/v1/embeddings`` endpoint.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        timeout: int = 60,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def embed(self, text: str, model: Optional[str] = None) -> np.ndarray:
        """Generate an embedding vector for *text*.

        Returns a 1-D ``numpy`` array.
        Raises ``EmbeddingError`` on failure.
        """
        payload: dict = {"input": text}
        if model is not None:
            payload["model"] = model

        data = json.dumps(payload).encode()

        try:
            req = urllib.request.Request(
                f"{self.server_url}/v1/embeddings",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc

        try:
            vec = body["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError(
                f"Unexpected response format: {body}"
            ) from exc

        return np.array(vec, dtype=np.float32)

    def embed_many(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [self.embed(t, model=model) for t in texts]


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return the L2-normalised copy of *vector* (unit length).

    A zero vector is returned unchanged.
    """
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector.copy()
    return vector / norm


def batch_normalize(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """L2-normalise every vector in the list."""
    return [l2_normalize(v) for v in vectors]


def norm_stats(vectors: List[np.ndarray]) -> dict:
    """Compute L2-norm statistics across a collection of vectors.

    Returns ``{"mean": float, "std": float, "min": float, "max": float}``.
    For an empty list all values are ``float("nan")``.
    """
    if not vectors:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    norms = [float(np.linalg.norm(v)) for v in vectors]
    return {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
    }
