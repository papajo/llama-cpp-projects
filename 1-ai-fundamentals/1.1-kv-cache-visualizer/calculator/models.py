"""
Data models for KV cache memory calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class KVQuant(Enum):
    """KV cache quantization types supported by llama.cpp."""

    F16 = ("f16", 2.0)       # 16-bit float — 2 bytes per element
    Q8_0 = ("q8_0", 1.0)     # 8-bit quant — 1 byte per element
    Q4_0 = ("q4_0", 0.5)     # 4-bit quant — 0.5 bytes per element

    def __init__(self, label: str, bytes_per_elem: float):
        self.label = label
        self.bytes_per_elem = bytes_per_elem

    @classmethod
    def from_str(cls, s: str) -> "KVQuant":
        for q in cls:
            if q.label == s.lower():
                return q
        raise ValueError(f"Unknown KV quant type: {s!r}.  Options: {[q.label for q in cls]}")

    @classmethod
    def all(cls) -> list["KVQuant"]:
        return [cls.F16, cls.Q8_0, cls.Q4_0]

    def memory_savings_label(self) -> str:
        """Human-readable size relative to f16."""
        if self == KVQuant.F16:
            return "baseline"
        ratio = self.bytes_per_elem / KVQuant.F16.bytes_per_elem
        return f"{int((1 - ratio) * 100)}% smaller"


@dataclass
class ModelConfig:
    """Architecture dimensions for a transformer model.

    These are the numbers that determine KV cache size.
    All values are taken from model configs (config.json) or GGUF metadata.
    """
    name: str
    n_layers: int              # number of transformer layers
    n_heads: int               # number of query heads (Q)
    n_kv_heads: int            # number of key/value heads (K/V, for GQA/MQA)
    head_dim: int              # dimension per head
    max_ctx: int               # maximum context length (in tokens)
    n_experts: int = 0         # MoE: total experts (0 = dense)
    n_active_experts: int = 0  # MoE: active experts per token

    @property
    def n_kv_groups(self) -> int:
        """Number of query heads per KV head (GQA ratio)."""
        return self.n_heads // self.n_kv_heads

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    @property
    def d_model(self) -> int:
        """Approximate hidden dimension."""
        return self.n_heads * self.head_dim

    def kv_cache_bytes_per_token(self, quant: KVQuant) -> float:
        """Bytes of KV cache per layer, per token, for *one* sequence.

        Formula: 2 (K + V) × n_kv_heads × head_dim × bytes_per_elem
        """
        return 2.0 * self.n_kv_heads * self.head_dim * quant.bytes_per_elem

    def kv_cache_bytes_per_token_all_layers(self, quant: KVQuant) -> float:
        """Bytes for all layers, per token, one sequence."""
        return self.n_layers * self.kv_cache_bytes_per_token(quant, )

    def __post_init__(self):
        if self.n_kv_heads > self.n_heads:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) cannot exceed n_heads ({self.n_heads})"
            )


@dataclass
class MemoryBreakdown:
    """Detailed breakdown of KV cache memory at a specific context length."""
    ctx_tokens: int
    quant: KVQuant
    bytes_per_layer: float        # K+V for one layer at this ctx
    bytes_total: float             # all layers
    bytes_total_all_slots: float   # unified KV (router server) total
    miB_per_layer: float
    miB_total: float
    miB_total_all_slots: float
    gb_total: float
    gb_total_all_slots: float

    def as_dict(self) -> dict:
        return {
            "ctx_tokens": self.ctx_tokens,
            "quant": self.quant.label,
            "bytes_per_layer": self.bytes_per_layer,
            "bytes_total": self.bytes_total,
            "bytes_total_all_slots": self.bytes_total_all_slots,
            "MiB_per_layer": round(self.miB_per_layer, 2),
            "MiB_total": round(self.miB_total, 2),
            "MiB_total_all_slots": round(self.miB_total_all_slots, 2),
            "GB_total": round(self.gb_total, 3),
            "GB_total_all_slots": round(self.gb_total_all_slots, 3),
        }


@dataclass
class KVCacheResult:
    """Complete KV cache analysis for a model configuration."""
    model: ModelConfig
    max_ctx: int
    n_slots: int
    allocation: str  # "unified" or "per-slot"
    breakdowns: Dict[str, MemoryBreakdown]  # quant_label -> breakdown at max ctx
    scaling: List[dict]                      # token-by-token scaling data
    recommendations: List[str]               # memory-saving tips

    def as_dict(self) -> dict:
        return {
            "model": {
                "name": self.model.name,
                "n_layers": self.model.n_layers,
                "n_heads": self.model.n_heads,
                "n_kv_heads": self.model.n_kv_heads,
                "head_dim": self.model.head_dim,
                "max_ctx": self.model.max_ctx,
                "d_model": self.model.d_model,
                "is_moe": self.model.is_moe,
                "n_kv_groups": self.model.n_kv_groups,
            },
            "max_ctx": self.max_ctx,
            "n_slots": self.n_slots,
            "allocation": self.allocation,
            "breakdowns": {k: v.as_dict() for k, v in self.breakdowns.items()},
            "scaling": self.scaling,
            "recommendations": self.recommendations,
        }
