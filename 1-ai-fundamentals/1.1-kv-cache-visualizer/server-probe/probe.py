"""
ServerProbe — Polls a running llama-server for live KV cache metrics.

Endpoints polled:
  - GET /slots         — per-slot status (active, n_prompt_tokens, etc.)
  - GET /metrics       — Prometheus-format metrics (if --metrics enabled)
  - GET /props         — server properties (model info, llama.cpp version)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class SlotInfo:
    """Information about a single server slot."""
    id: int
    state: str  # idle / processing
    n_prompt_tokens: int = 0
    n_predicts: int = 0
    n_decoded: int = 0
    n_past: int = 0  # tokens in KV cache for this slot
    prompt: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "SlotInfo":
        # Current llama-server reports occupancy as the boolean
        # `is_processing` and nests decode progress under `next_token`.
        # Older builds exposed a flat `state` string plus `n_decoded` /
        # `n_past`, so fall back to those when present.
        if "state" in d:
            state = d["state"]
        elif "is_processing" in d:
            state = "processing" if d["is_processing"] else "idle"
        else:
            state = "unknown"

        next_token = d.get("next_token") or [{}]
        if isinstance(next_token, dict):
            next_token = [next_token]

        return cls(
            id=d.get("id", -1),
            state=state,
            n_prompt_tokens=d.get("n_prompt_tokens", 0),
            n_predicts=d.get("n_predicts", 0),
            n_decoded=d.get("n_decoded", next_token[0].get("n_decoded", 0)),
            n_past=d.get("n_past", d.get("n_prompt_tokens_cache", 0)),
            prompt=d.get("prompt", ""),
        )


@dataclass
class ServerMetrics:
    """Aggregated server metrics from /metrics (Prometheus)."""
    kv_cache_usage_ratio: float = 0.0   # 0.0 – 1.0
    tokens_per_second: float = 0.0
    active_slots: int = 0
    idle_slots: int = 0
    total_slots: int = 0
    raw: str = ""

    @classmethod
    def from_prometheus(cls, text: str) -> "ServerMetrics":
        m = cls(raw=text)

        # Parse key metrics from Prometheus text format
        patterns = {
            "kv_cache_usage_ratio": r"llamacpp:kv_cache_usage_ratio\s+([\d.]+)",
            "tokens_per_second": r"llamacpp:tokens_per_second\s+([\d.]+)",
            "active_slots": r"llamacpp:active_slots\s+(\d+)",
            "idle_slots": r"llamacpp:idle_slots\s+(\d+)",
            "total_slots": r"llamacpp:total_slots\s+(\d+)",
        }

        for attr, pat in patterns.items():
            match = re.search(pat, text, re.MULTILINE)
            if match:
                setattr(m, attr, float(match.group(1)))

        return m


class ServerProbe:
    """
    Probe a running llama-server for live status.

    Usage:
        probe = ServerProbe("http://localhost:8080")
        slots = probe.get_slots()
        metrics = probe.get_metrics()
        props = probe.get_props()
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ── Slot info ──────────────────────────────────────────────

    def get_slots(self) -> List[SlotInfo]:
        """Fetch /slots and return parsed SlotInfo list."""
        resp = self._client.get(f"{self.base_url}/slots")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            data = data.get("slots", [])
        return [SlotInfo.from_dict(s) for s in data]

    def get_active_slots(self) -> List[SlotInfo]:
        return [s for s in self.get_slots() if s.state == "processing"]

    def get_idle_slots(self) -> List[SlotInfo]:
        return [s for s in self.get_slots() if s.state == "idle"]

    # ── Metrics ────────────────────────────────────────────────

    def get_metrics(self) -> ServerMetrics:
        """Fetch /metrics and parse Prometheus metrics."""
        resp = self._client.get(f"{self.base_url}/metrics")
        resp.raise_for_status()
        return ServerMetrics.from_prometheus(resp.text)

    # ── Server properties ──────────────────────────────────────

    def get_props(self) -> Dict[str, Any]:
        """Fetch /props for server configuration metadata."""
        resp = self._client.get(f"{self.base_url}/props")
        resp.raise_for_status()
        return resp.json()

    # ── Aggregate report ───────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Collect all live data into a structured report."""
        slots = self.get_slots()
        try:
            metrics = self.get_metrics()
        except Exception:
            metrics = ServerMetrics()
        try:
            props = self.get_props()
        except Exception:
            props = {}

        total_kv = sum(s.n_past for s in slots)

        return {
            "server": self.base_url,
            "total_slots": len(slots),
            "active_slots": len([s for s in slots if s.state == "processing"]),
            "idle_slots": len([s for s in slots if s.state == "idle"]),
            "total_kv_tokens_cached": total_kv,
            "metrics": {
                "kv_cache_usage_ratio": metrics.kv_cache_usage_ratio,
                "tokens_per_second": metrics.tokens_per_second,
                "active_slots_prom": metrics.active_slots,
                "idle_slots_prom": metrics.idle_slots,
            },
            "slots": [
                {
                    "id": s.id,
                    "state": s.state,
                    "tokens_cached": s.n_past,
                    "prompt_preview": s.prompt[:80] if s.prompt else "",
                }
                for s in slots
            ],
            "props": props,
            "timestamp": time.time(),
        }

    def close(self):
        self._client.close()
