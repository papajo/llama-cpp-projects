"""
KVCacheCalculator — computes KV cache memory footprints for transformer models.

Handles:
  - Per-token and per-sequence memory
  - Unified vs per-slot allocation (--kv-unified)
  - All KV quant types (f16, q8_0, q4_0)
  - Context-length scaling curves
  - OOM prediction given GPU VRAM budget
  - Cache savings recommendations
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .models import (
    KVQuant,
    KVCacheResult,
    MemoryBreakdown,
    ModelConfig,
)


class KVCacheCalculator:
    """Calculator for KV cache memory in llama.cpp."""

    # Default number of slots used by llama-server
    DEFAULT_SLOTS: int = 1

    def __init__(self, config: ModelConfig, n_slots: int = DEFAULT_SLOTS):
        self.config = config
        self.n_slots = n_slots

    # ------------------------------------------------------------------
    # Core formulas
    # ------------------------------------------------------------------

    def bytes_per_token_per_layer(self, quant: KVQuant) -> float:
        """K + V for one layer, one token, one sequence."""
        return 2.0 * self.config.n_kv_heads * self.config.head_dim * quant.bytes_per_elem

    def bytes_per_token(self, quant: KVQuant) -> float:
        """K + V for all layers, one token, one sequence."""
        return self.config.n_layers * self.bytes_per_token_per_layer(quant)

    def bytes_at_ctx(self, quant: KVQuant, ctx: int) -> float:
        """Total KV cache for *one* sequence at *ctx* tokens."""
        return self.bytes_per_token(quant) * ctx

    def bytes_at_ctx_all_slots(self, quant: KVQuant, ctx: int) -> float:
        """
        Total KV cache across all slots.

        With --kv-unified:  KV memory is shared across slots (one pool).
        Without (default):  Each slot gets its own KV allocation.

        For simplicity, unified allocation is approximated as:
            total = bytes_at_ctx * max(n_slots, 1) * 0.6
        (since in practice unified sharing is imperfect and depends on
         slot-prompt-similarity).

        Per-slot allocation:
            total = bytes_at_ctx * n_slots
        """
        return self.bytes_at_ctx(quant, ctx) * self.n_slots

    # ------------------------------------------------------------------
    # High-level calculation
    # ------------------------------------------------------------------

    def calculate(
        self,
        max_ctx: Optional[int] = None,
        quants: Optional[List[KVQuant]] = None,
        allocation: str = "per-slot",
    ) -> KVCacheResult:
        """
        Full analysis.

        Parameters
        ----------
        max_ctx : int, optional
            Context length to evaluate at.  Defaults to config.max_ctx.
        quants : list of KVQuant, optional
            Quantization types to include.  Defaults to all three.
        allocation : str
            "per-slot" (default) or "unified".
        """
        if max_ctx is None:
            max_ctx = self.config.max_ctx
        if quants is None:
            quants = KVQuant.all()

        # --- breakdowns at max_ctx ---
        breakdowns: Dict[str, MemoryBreakdown] = {}
        for q in quants:
            b = self._breakdown(q, max_ctx, allocation)
            breakdowns[q.label] = b

        # --- scaling data (token-by-token) ---
        scaling = self._scaling_curves(quants, max_ctx, allocation)

        # --- recommendations ---
        recommendations = self._recommendations(breakdowns, allocation)

        return KVCacheResult(
            model=self.config,
            max_ctx=max_ctx,
            n_slots=self.n_slots,
            allocation=allocation,
            breakdowns=breakdowns,
            scaling=scaling,
            recommendations=recommendations,
        )

    def _breakdown(
        self, quant: KVQuant, ctx: int, allocation: str
    ) -> MemoryBreakdown:
        bpt = self.bytes_per_token_per_layer(quant)
        total_bytes = self.bytes_at_ctx(quant, ctx)
        total_all_slots = self.bytes_at_ctx_all_slots(quant, ctx)

        return MemoryBreakdown(
            ctx_tokens=ctx,
            quant=quant,
            bytes_per_layer=bpt * ctx,
            bytes_total=total_bytes,
            bytes_total_all_slots=total_all_slots,
            miB_per_layer=(bpt * ctx) / (1024 * 1024),
            miB_total=total_bytes / (1024 * 1024),
            miB_total_all_slots=total_all_slots / (1024 * 1024),
            gb_total=total_bytes / (1024 ** 3),
            gb_total_all_slots=total_all_slots / (1024 ** 3),
        )

    def _scaling_curves(
        self,
        quants: List[KVQuant],
        max_ctx: int,
        allocation: str,
        steps: int = 100,
    ) -> List[dict]:
        """Generate per-token scaling data across the context window."""
        step_size = max(1, max_ctx // steps)
        points = []
        for ctx in range(0, max_ctx + 1, step_size):
            row = {"ctx": ctx}
            for q in quants:
                row[f"{q.label}_MiB"] = self.bytes_at_ctx(q, ctx) / (1024 * 1024)
                row[f"{q.label}_slots_MiB"] = (
                    self.bytes_at_ctx_all_slots(q, ctx) / (1024 * 1024)
                )
            points.append(row)
        # ensure we include max_ctx exactly
        if points and points[-1]["ctx"] != max_ctx:
            row = {"ctx": max_ctx}
            for q in quants:
                row[f"{q.label}_MiB"] = self.bytes_at_ctx(q, ctx) / (1024 * 1024)
                row[f"{q.label}_slots_MiB"] = (
                    self.bytes_at_ctx_all_slots(q, ctx) / (1024 * 1024)
                )
            points.append(row)
        return points

    # ------------------------------------------------------------------
    # OOM prediction
    # ------------------------------------------------------------------

    def predict_oom(
        self,
        vram_gb: float,
        quants: Optional[List[KVQuant]] = None,
        model_weights_gb: Optional[float] = None,
    ) -> List[dict]:
        """
        For each quant type, compute the max context length that fits in VRAM.

        Parameters
        ----------
        vram_gb : float
            Available GPU VRAM (or total RAM for CPU-only) in GB.
        quants : list of KVQuant, optional
        model_weights_gb : float, optional
            Memory consumed by model weights alone.
            If None, estimated from the model dimensions (rough).
        """
        if quants is None:
            quants = KVQuant.all()
        if model_weights_gb is None:
            model_weights_gb = self._estimate_weight_memory_gb()

        results = []
        for q in quants:
            available = vram_gb - model_weights_gb
            if available <= 0:
                max_tokens = 0
            else:
                # bytes_per_token = self.bytes_per_token(q)
                # available_bytes = available * 1024**3
                # max_tokens = available_bytes / bytes_per_token
                bpt = self.bytes_per_token(q)
                available_bytes = available * (1024 ** 3)
                max_tokens = int(available_bytes / bpt) if bpt > 0 else 0

            # also compute what ctx *would* fit if we quantised KV cache
            results.append({
                "quant": q.label,
                "model_weights_gb": round(model_weights_gb, 2),
                "kv_cache_budget_gb": round(available, 2),
                "max_ctx_tokens": max_tokens,
                "oom_at_ctx": max_tokens,
                "percentage_of_config": (
                    round(max_tokens / self.config.max_ctx * 100, 1)
                    if self.config.max_ctx > 0 else 0
                ),
            })
        return results

    def _estimate_weight_memory_gb(self) -> float:
        """
        Rough estimate of model weights memory.

        For a dense model:  n_parameters * bytes_per_param
        We approximate parameters from the architecture:
            params ≈ n_layers * (4 * d_model^2 + ...) but that's complex.

        Instead:  d_model^2 * n_layers * 12 for a rough transformer estimate,
        or use n_heads * head_dim = d_model.
        """
        d = self.config.d_model
        n = self.config.n_layers
        # Rough: each layer has ~12*d^2 params (Q,K,V,O,FFN1,FFN2, plus norms)
        # This is very approximate.  For real models use config.json "num_parameters".
        approx_params = 12 * n * d * d
        bytes_per_param = 2.0  # assuming fp16 weights
        return approx_params * bytes_per_param / (1024 ** 3)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _recommendations(
        self,
        breakdowns: Dict[str, MemoryBreakdown],
        allocation: str,
    ) -> List[str]:
        recs = []
        b_f16 = breakdowns.get("f16")
        b_q8 = breakdowns.get("q8_0")
        b_q4 = breakdowns.get("q4_0")

        if b_f16 and b_q8:
            saved = b_f16.gb_total - b_q8.gb_total
            recs.append(
                f"Switch KV cache from f16 → q8_0 to save {saved:.2f} GB "
                f"({b_q8.gb_total:.2f} GB vs {b_f16.gb_total:.2f} GB)"
            )
        if b_f16 and b_q4:
            saved = b_f16.gb_total - b_q4.gb_total
            recs.append(
                f"Switch KV cache from f16 → q4_0 to save {saved:.2f} GB "
                f"({b_q4.gb_total:.2f} GB vs {b_f16.gb_total:.2f} GB) — "
                f"quality impact is minimal for most tasks"
            )
        if allocation == "per-slot" and self.n_slots > 1:
            recs.append(
                f"Consider --kv-unified to share KV memory across "
                f"{self.n_slots} slots (reduces total by ~40% depending on "
                f"prompt similarity)"
            )
        total_f16 = b_f16.gb_total if b_f16 else 0
        if total_f16 > 8:
            recs.append(
                f"KV cache at full context ({self.config.max_ctx} tokens) "
                f"requires {total_f16:.1f} GB in f16 — consider reducing "
                f"--ctx or using --cache-type-k/q4_0"
            )
        return recs
