"""
Memory estimator for GGUF quantisation types.

Computes "what-if" memory usage for every quant type given a model's
dimensions, using formulas based on llama.cpp's actual memory allocation:

  model_weights = sum(tensor_size for each tensor at quant bits_per_weight)
  kv_cache      = 2 * n_layers * (n_heads_kv * d_head) * ctx_len * kv_size
  overhead      = activations + scratch buffers (heuristic)
  total         = model_weights + kv_cache + overhead
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from gguf_reader.models import GGUFModel, QuantType


# ── Hardware profiles ─────────────────────────────────────────────

@dataclass
class HardwareProfile:
    """Describes target hardware for recommendations."""
    name: str = "custom"
    total_vram_gb: float = 12.0       # Total GPU VRAM in GB
    total_ram_gb: float = 32.0        # Total system RAM in GB
    gpu_name: str = "GPU"
    is_apple_silicon: bool = False    # Unified memory (M-series)
    compute_capability: Optional[str] = None  # e.g., "8.9" (RTX 4090)

    @classmethod
    def rtx_3060_12gb(cls) -> "HardwareProfile":
        return cls("RTX 3060 12GB", total_vram_gb=12.0, gpu_name="RTX 3060")

    @classmethod
    def rtx_3090_24gb(cls) -> "HardwareProfile":
        return cls("RTX 3090 24GB", total_vram_gb=24.0, gpu_name="RTX 3090")

    @classmethod
    def rtx_4090_24gb(cls) -> "HardwareProfile":
        return cls("RTX 4090 24GB", total_vram_gb=24.0, gpu_name="RTX 4090")

    @classmethod
    def rtx_5090_32gb(cls) -> "HardwareProfile":
        return cls("RTX 5090 32GB", total_vram_gb=32.0, gpu_name="RTX 5090")

    @classmethod
    def a100_80gb(cls) -> "HardwareProfile":
        return cls("A100 80GB", total_vram_gb=80.0, gpu_name="A100")

    @classmethod
    def macbook_m1_8gb(cls) -> "HardwareProfile":
        return cls("MacBook M1 8GB", total_ram_gb=8.0, gpu_name="M1", is_apple_silicon=True)

    @classmethod
    def macbook_m2_24gb(cls) -> "HardwareProfile":
        return cls("MacBook M2 24GB", total_ram_gb=24.0, gpu_name="M2", is_apple_silicon=True)

    @classmethod
    def macbook_m4_max_128gb(cls) -> "HardwareProfile":
        return cls("MacBook M4 Max 128GB", total_ram_gb=128.0, gpu_name="M4 Max", is_apple_silicon=True)


# ── Memory estimate ───────────────────────────────────────────────

@dataclass
class MemoryEstimate:
    """Detailed memory breakdown for a single quantisation."""
    quant_type: QuantType
    model_weights_gb: float = 0.0     # Size of weights at this quant
    kv_cache_gb: float = 0.0          # KV cache at default context
    overhead_gb: float = 0.0          # Activations + scratch
    total_gb: float = 0.0             # Total = weights + kv + overhead
    context_length: int = 4096        # Context length used for estimate
    fits_in_vram: bool = False        # Whether it fits target hardware
    quality_score: float = 0.0        # 0.0–1.0 relative quality

    @property
    def weights_pct(self) -> float:
        return (self.model_weights_gb / self.total_gb * 100) if self.total_gb > 0 else 0

    @property
    def kv_cache_pct(self) -> float:
        return (self.kv_cache_gb / self.total_gb * 100) if self.total_gb > 0 else 0


# ── Estimator ─────────────────────────────────────────────────────

class QuantEstimator:
    """
    Memory estimator for quantisation types.

    Uses formulas derived from llama.cpp's memory allocation:
      - Weights: num_params * bits_per_weight / 8
      - KV cache: 2 * n_layers * (n_kv_heads * d / n_heads) * ctx_len * dtype_size
      - Overhead: heuristic 5-10% of weights
    """

    # GGML tensor overhead per quant type (bytes per block overhead)
    # Based on GGML block sizes
    _BLOCK_SIZES = {
        QuantType.F32: 1.0,        # 1 weight per byte * 4
        QuantType.F16: 1.0,
        QuantType.Q4_0: 4.5 / 32,  # 32 weights in 18 bytes
        QuantType.Q4_1: 5.0 / 32,
        QuantType.Q5_0: 5.5 / 32,
        QuantType.Q5_1: 6.0 / 32,
        QuantType.Q8_0: 8.5 / 32,
        QuantType.Q2_K: 2.5625 / 32,
        QuantType.Q3_K: 3.4375 / 32,
        QuantType.Q4_K: 4.5 / 32,
        QuantType.Q5_K: 5.5 / 32,
        QuantType.Q6_K: 6.5625 / 32,
        QuantType.Q8_K: 8.5 / 32,
    }

    def __init__(self, model: GGUFModel):
        self.model = model

    def estimate(
        self,
        quant_type: QuantType,
        context_length: int = 4096,
        batch_size: int = 512,
    ) -> MemoryEstimate:
        """Compute memory estimate for a given quant type."""
        d = self.model.embedding_dim
        n_layers = self.model.block_count
        n_heads = self.model.head_count
        n_kv_heads = self.model.head_count_kv
        vocab = self.model.vocab_size
        n_experts = self.model.expert_count

        # ── Model weights ──────────────────────────────────────────
        bpw = quant_type.bits_per_weight
        num_params = int(self.model.param_count_b * 1e9)

        if num_params == 0:
            num_params = self._estimate_num_params()

        # Account for quant overhead (block structure)
        overhead_factor = 1.0
        if quant_type.is_k_quant:
            overhead_factor = 1.02   # ~2% extra for K-quant metadata
        elif quant_type.is_i_quant:
            overhead_factor = 1.015  # ~1.5% extra for importance quants

        model_weights_bytes = num_params * bpw / 8 * overhead_factor
        model_weights_gb = model_weights_bytes / (1024 ** 3)

        # ── KV cache ───────────────────────────────────────────────
        # KV cache uses F16 (2 bytes) regardless of weight quant
        kv_size_bytes = 2  # F16

        if n_kv_heads > 0 and d > 0 and n_heads > 0:
            head_dim = d // n_heads
            kv_cache_bytes = (
                2                           # K + V
                * n_layers
                * n_kv_heads
                * head_dim
                * context_length
                * kv_size_bytes
            )
        else:
            kv_cache_bytes = 0

        kv_cache_gb = kv_cache_bytes / (1024 ** 3)

        # ── Overhead (activations + scratch) ────────────────────────
        # Heuristic: 5-15% of weights depending on batch size
        overhead_pct = 0.05 + (batch_size / 4096) * 0.10
        overhead_gb = model_weights_gb * overhead_pct

        total_gb = model_weights_gb + kv_cache_gb + overhead_gb

        return MemoryEstimate(
            quant_type=quant_type,
            model_weights_gb=round(model_weights_gb, 3),
            kv_cache_gb=round(kv_cache_gb, 3),
            overhead_gb=round(overhead_gb, 3),
            total_gb=round(total_gb, 3),
            context_length=context_length,
            quality_score=quant_type.relative_quality,
        )

    def estimate_all(
        self,
        context_length: int = 4096,
        quant_types: Optional[List[QuantType]] = None,
    ) -> List[MemoryEstimate]:
        """Compute estimates for all (or specified) quant types."""
        if quant_types is None:
            quant_types = [
                QuantType.F32, QuantType.F16,
                QuantType.Q8_0, QuantType.Q6_K,
                QuantType.Q5_K, QuantType.Q5_0,
                QuantType.Q4_K, QuantType.Q4_0,
                QuantType.Q3_K, QuantType.Q2_K,
                QuantType.IQ4_XS, QuantType.IQ3_S,
                QuantType.IQ2_S, QuantType.IQ1_S,
                QuantType.TQ2_0,
            ]

        return [
            self.estimate(qt, context_length)
            for qt in quant_types
        ]

    def check_oom(
        self,
        hardware: HardwareProfile,
        context_length: int = 4096,
    ) -> List[MemoryEstimate]:
        """Check which quant types fit in the given hardware."""
        estimates = self.estimate_all(context_length)
        vram = hardware.total_vram_gb if not hardware.is_apple_silicon else hardware.total_ram_gb
        for est in estimates:
            est.fits_in_vram = est.total_gb <= vram * 0.9  # 10% headroom
        return estimates

    def recommend(
        self,
        hardware: HardwareProfile,
        context_length: int = 4096,
        min_quality: float = 0.5,
    ) -> List[MemoryEstimate]:
        """
        Recommend the best quant types for given hardware.
        Returns estimates sorted by quality (best first) that fit in VRAM.
        """
        estimates = self.check_oom(hardware, context_length)
        fitting = [
            e for e in estimates
            if e.fits_in_vram and e.quality_score >= min_quality
        ]
        return sorted(fitting, key=lambda e: e.quality_score, reverse=True)

    def _estimate_num_params(self) -> int:
        """Fallback parameter count estimation."""
        model = self.model
        d = model.embedding_dim
        n_layers = model.block_count
        ffn = model.feed_forward_dim
        vocab = model.vocab_size

        if d == 0:
            return 0

        # Rough estimate: embeddings + (attention + FFN) * layers + output
        params = vocab * d  # embeddings
        params += n_layers * (d * d * 4 + d * ffn * 3)  # attn + FFN
        params += d  # final norm
        return int(params)
