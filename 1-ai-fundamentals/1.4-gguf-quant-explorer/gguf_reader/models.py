"""
Data models for GGUF file structure and model metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


# ── GGML Quantisation Types ──────────────────────────────────────

class QuantType(IntEnum):
    """GGML quantisation type enum (subset of ggml_type)."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    Q4_0_4_4 = 31
    Q4_0_4_8 = 32
    Q4_0_8_8 = 33
    TQ1_0 = 34
    TQ2_0 = 35
    IQ4_NL_4_4 = 36
    IQ4_NL_4_8 = 37
    IQ4_NL_8_8 = 38

    @classmethod
    def from_file_type(cls, file_type: int) -> QuantType:
        """Convert a GGUF file_type to QuantType."""
        try:
            return cls(file_type)
        except ValueError:
            return cls.F32  # fallback

    @property
    def is_k_quant(self) -> bool:
        """K-quants are block-structure aware."""
        return self.name.startswith("Q") and "_K" in self.name

    @property
    def is_i_quant(self) -> bool:
        """Importance-aware quants (IQ)."""
        return self.name.startswith("IQ")

    @property
    def bits_per_weight(self) -> float:
        """Theoretical bits per weight for this quant type."""
        bits = {
            # Float
            "F32": 32.0, "F16": 16.0, "BF16": 16.0, "F64": 64.0,
            # Standard quants
            "Q4_0": 4.5, "Q4_1": 5.0,
            "Q5_0": 5.5, "Q5_1": 6.0,
            "Q8_0": 8.5, "Q8_1": 9.0,
            # K-quants (block-aware)
            "Q2_K": 2.5625, "Q3_K": 3.4375,
            "Q4_K": 4.5, "Q5_K": 5.5,
            "Q6_K": 6.5625, "Q8_K": 8.5,
            # Importance-aware quants
            "IQ1_S": 1.5625, "IQ1_M": 1.75,
            "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125,
            "IQ2_S": 2.5625,
            "IQ3_XXS": 3.0625, "IQ3_S": 3.4375,
            "IQ4_NL": 4.5, "IQ4_XS": 4.25,
            # Integer
            "I8": 8.0, "I16": 16.0, "I32": 32.0, "I64": 64.0,
            # 4-bit block
            "Q4_0_4_4": 4.5, "Q4_0_4_8": 4.5, "Q4_0_8_8": 4.5,
            # Ternary
            "TQ1_0": 1.0, "TQ2_0": 2.0,
            # IQ4 block variants
            "IQ4_NL_4_4": 4.5, "IQ4_NL_4_8": 4.5, "IQ4_NL_8_8": 4.5,
        }
        return bits.get(self.name, 16.0)

    @property
    def display_name(self) -> str:
        """Human-readable quant name (e.g. 'Q4_K_M' detection from file_type)."""
        name = self.name
        # K-quant variants are stored as file_type in GGUF, but the sub-variant
        # (M, S, L) is stored in the model filename, not the metadata.
        return name

    @property
    def kval(self) -> Optional[str]:
        """Return 'M', 'S', or 'L' if this is a K-quant with known sub-type."""
        # The GGUF file_type doesn't encode M/S/L — that's in the filename.
        return None

    @property
    def relative_quality(self) -> float:
        """Relative quality score 0.0–1.0 (higher = better)."""
        scores = {
            "F32": 1.0, "F16": 0.99, "BF16": 0.99,
            "Q8_0": 0.95, "Q8_K": 0.94,
            "Q6_K": 0.92,
            "Q5_1": 0.91, "Q5_0": 0.90, "Q5_K": 0.90,
            "Q4_1": 0.87, "Q4_0": 0.85, "Q4_K": 0.86,
            "Q3_K": 0.78,
            "Q2_K": 0.65,
            "IQ4_NL": 0.86, "IQ4_XS": 0.84,
            "IQ3_S": 0.78, "IQ3_XXS": 0.74,
            "IQ2_S": 0.65, "IQ2_XS": 0.60, "IQ2_XXS": 0.55,
            "IQ1_S": 0.40, "IQ1_M": 0.45,
            "TQ1_0": 0.30, "TQ2_0": 0.50,
        }
        return scores.get(self.name, 0.5)

    @property
    def description(self) -> str:
        """Short description of the quant type."""
        descs = {
            "F32": "32-bit float — no quantisation",
            "F16": "16-bit float — minimal compression",
            "BF16": "Brain float 16-bit",
            "Q4_0": "4-bit block quant (legacy, fast)",
            "Q4_1": "4-bit block quant with higher precision",
            "Q5_0": "5-bit block quant",
            "Q5_1": "5-bit block quant with higher precision",
            "Q8_0": "8-bit block quant — near lossless",
            "Q2_K": "2-bit K-quant — maximum compression",
            "Q3_K": "3-bit K-quant — good compression/quality balance",
            "Q4_K": "4-bit K-quant — most popular choice",
            "Q5_K": "5-bit K-quant — high quality",
            "Q6_K": "6-bit K-quant — near lossless K-quant",
            "Q8_K": "8-bit K-quant",
            "IQ1_S": "1.56-bit importance quant — extreme compression",
            "IQ1_M": "1.75-bit importance quant",
            "IQ2_XXS": "2.06-bit importance quant — very compact",
            "IQ2_XS": "2.31-bit importance quant",
            "IQ2_S": "2.56-bit importance quant",
            "IQ3_XXS": "3.06-bit importance quant",
            "IQ3_S": "3.44-bit importance quant",
            "IQ4_NL": "4.5-bit importance quant (no list)",
            "IQ4_XS": "4.25-bit importance quant (extra small)",
            "TQ1_0": "1-bit ternary quant — experimental",
            "TQ2_0": "2-bit ternary quant — experimental",
        }
        return descs.get(self.name, f"{self.name} quantisation")


# ── GGUF metadata value types ────────────────────────────────────

class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# ── GGUF model data ──────────────────────────────────────────────

@dataclass
class GGUFMetadata:
    """Key-value pair from the GGUF metadata header."""
    key: str
    value_type: GGUFValueType
    value: Any


@dataclass
class TensorInfo:
    """Information about a single tensor in the GGUF file."""
    name: str
    n_dims: int
    dimensions: List[int]
    ggml_type: QuantType
    offset: int
    size_bytes: int = 0

    @property
    def num_elements(self) -> int:
        import math
        return math.prod(self.dimensions) if self.dimensions else 0


@dataclass
class GGUFModel:
    """
    High-level model information extracted from a GGUF file.
    """
    file_path: str = ""
    file_size_bytes: int = 0

    # GGUF header
    gguf_version: int = 0
    tensor_count: int = 0
    metadata_kv_count: int = 0

    # Raw metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    all_kv_pairs: List[GGUFMetadata] = field(default_factory=list)

    # Tensor info
    tensors: List[TensorInfo] = field(default_factory=list)

    # Derived model architecture
    architecture: str = ""
    model_name: str = ""
    file_type: int = 0
    current_quant: Optional[QuantType] = None

    # Dimensions
    embedding_dim: int = 0       # hidden_size / d_model
    block_count: int = 0         # number of transformer layers
    head_count: int = 0          # attention heads
    head_count_kv: int = 0       # KV heads (for GQA/MQA)
    feed_forward_dim: int = 0    # intermediate_size / FFN hidden
    vocab_size: int = 0
    context_length: int = 0
    expert_count: int = 0        # MoE experts (0 = dense)
    expert_used_count: int = 0   # top-k experts

    # Computed properties
    param_count_b: float = 0.0   # total parameters in billions

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 0

    @property
    def is_gqa(self) -> bool:
        return self.head_count_kv > 0 and self.head_count_kv != self.head_count

    @property
    def kv_group_size(self) -> int:
        if self.head_count_kv == 0:
            return 1
        return self.head_count // self.head_count_kv

    @property
    def model_size_gb(self) -> float:
        """Raw file size in GB."""
        return self.file_size_bytes / (1024 ** 3)

    def summary(self) -> str:
        """One-line summary of the model."""
        parts = [
            f"{self.model_name or self.architecture or 'Unknown'}",
            f"{self.param_count_b:.1f}B params",
            f"{self.current_quant.name if self.current_quant else 'unknown'}",
        ]
        if self.is_moe:
            parts.append(f"MoE ({self.expert_count}E{self.expert_used_count}K)")
        if self.context_length:
            parts.append(f"ctx {self.context_length}")
        return " · ".join(parts)
