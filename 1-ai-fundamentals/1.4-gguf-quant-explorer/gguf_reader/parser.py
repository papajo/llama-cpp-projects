"""
GGUF binary parser — reads raw GGUF files and extracts metadata + tensor info.

GGUF format (v1–v3):
  [0x00] magic:      "GGUF" (4 bytes)
  [0x04] version:    uint32
  [0x08] tensor_count: uint64
  [0x10] metadata_kv_count: uint64
  [0x18] ... KV pairs ...
  ...    ... tensor infos ...
  ...    ... tensor data ...
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

from .models import (
    GGUFMetadata,
    GGUFModel,
    GGUFValueType,
    QuantType,
    TensorInfo,
)

# ── GGUF magic bytes ──────────────────────────────────────────────

GGUF_MAGIC = b"GGUF"

# Map metadata key suffixes to (numeric_type, default)
_KNOWN_SIZES = {
    "context_length": 4096,
    "embedding_length": 4096,
    "block_count": 32,
    "head_count": 32,
    "head_count_kv": 8,
    "feed_forward_length": 11008,
    "vocab_size": 32000,
    "expert_count": 0,
    "expert_used_count": 0,
}


# ── GGUF value reader ─────────────────────────────────────────────

def _read_gguf_value(f: BinaryIO, value_type: int, version: int) -> Any:
    """Read a single GGUF metadata value."""
    if value_type == GGUFValueType.UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif value_type == GGUFValueType.INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif value_type == GGUFValueType.UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif value_type == GGUFValueType.INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif value_type == GGUFValueType.UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif value_type == GGUFValueType.INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif value_type == GGUFValueType.UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif value_type == GGUFValueType.INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif value_type == GGUFValueType.FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif value_type == GGUFValueType.FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    elif value_type == GGUFValueType.BOOL:
        return bool(struct.unpack("<B", f.read(1))[0])
    elif value_type == GGUFValueType.STRING:
        # String: uint64 length + UTF-8 bytes
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")
    elif value_type == GGUFValueType.ARRAY:
        # Array: uint32 type + uint64 len + elements
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        return [_read_gguf_value(f, arr_type, version) for _ in range(arr_len)]
    else:
        raise ValueError(f"Unknown GGUF value type: {value_type}")


def _read_string(f: BinaryIO) -> str:
    """Read a GGUF string: uint64 length + UTF-8 bytes."""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _read_tensor_info(f: BinaryIO, version: int) -> TensorInfo:
    """Read a single tensor info block."""
    name = _read_string(f)
    n_dims = struct.unpack("<I", f.read(4))[0]
    dims = list(struct.unpack(f"<{'Q' * n_dims}", f.read(8 * n_dims)))
    ggml_type = QuantType(struct.unpack("<I", f.read(4))[0])
    offset = struct.unpack("<Q", f.read(8))[0]

    # Approximate size from dimensions and type
    num_elements = math.prod(dims) if dims else 0
    size_bytes = int(num_elements * ggml_type.bits_per_weight / 8)

    return TensorInfo(
        name=name,
        n_dims=n_dims,
        dimensions=dims,
        ggml_type=ggml_type,
        offset=offset,
        size_bytes=size_bytes,
    )


# ── Parser ────────────────────────────────────────────────────────

class GGUFParser:
    """Parse a raw GGUF file and extract model metadata."""

    @staticmethod
    def parse(file_path: str) -> GGUFModel:
        """Parse a GGUF file and return structured model info."""
        path = Path(file_path)
        model = GGUFModel(file_path=str(path.resolve()))
        model.file_size_bytes = path.stat().st_size

        with open(path, "rb") as f:
            _GGUFParser._parse_header(f, model)
            _GGUFParser._parse_metadata(f, model)
            _GGUFParser._parse_tensors(f, model)

        _GGUFParser._derive_model_info(model)
        return model


class _GGUFParser:
    """Internal parser implementation."""

    @staticmethod
    def _parse_header(f: BinaryIO, model: GGUFModel):
        """Parse the GGUF file header."""
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(
                f"Not a valid GGUF file (magic: {magic.hex()}, "
                f"expected 'GGUF')"
            )

        model.gguf_version = struct.unpack("<I", f.read(4))[0]
        model.tensor_count = struct.unpack("<Q", f.read(8))[0]
        model.metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

    @staticmethod
    def _parse_metadata(f: BinaryIO, model: GGUFModel):
        """Parse metadata KV pairs."""
        for _ in range(model.metadata_kv_count):
            key = _read_string(f)
            value_type = struct.unpack("<I", f.read(4))[0]
            value = _read_gguf_value(f, value_type, model.gguf_version)

            kv = GGUFMetadata(
                key=key,
                value_type=GGUFValueType(value_type),
                value=value,
            )
            model.all_kv_pairs.append(kv)
            model.metadata[key] = value

    @staticmethod
    def _parse_tensors(f: BinaryIO, model: GGUFModel):
        """Parse tensor info blocks."""
        for _ in range(model.tensor_count):
            tensor = _read_tensor_info(f, model.gguf_version)
            model.tensors.append(tensor)

    @staticmethod
    def _derive_model_info(model: GGUFModel):
        """Derive high-level model info from raw metadata."""
        meta = model.metadata

        # Architecture
        model.architecture = meta.get(
            "general.architecture",
            meta.get("general.name", "unknown"),
        ).lower()

        model.model_name = meta.get(
            "general.name",
            f"{model.architecture}-model",
        )

        # File type (quantisation)
        model.file_type = meta.get("general.file_type", 0)
        model.current_quant = QuantType.from_file_type(model.file_type)

        # Dimensions — try known key patterns
        arch = model.architecture
        prefixes = [f"{arch}.", "llama.", ""]  # fallback chain

        def _lookup(key: str):
            """First present value for `key` across the prefix chain."""
            for p in prefixes:
                val = meta.get(f"{p}{key}")
                if val is not None:
                    return val
            return None

        def _get(key: str) -> int:
            val = _lookup(key)
            if val is not None:
                return int(val)
            # Attention hyper-parameters are namespaced under
            # `<arch>.attention.*` in GGUF, not directly under `<arch>.`.
            val = _lookup(f"attention.{key}")
            if val is not None:
                return int(val)
            return _KNOWN_SIZES.get(key, 0)

        model.embedding_dim = _get("embedding_length")
        model.block_count = _get("block_count")
        model.head_count = _get("head_count")
        # A model with no explicit KV head count is plain MHA, so the KV
        # head count equals the head count. Only fall back to the generic
        # default when neither is known.
        kv = _lookup("head_count_kv") or _lookup("attention.head_count_kv")
        model.head_count_kv = int(kv) if kv is not None else model.head_count
        model.feed_forward_dim = _get("feed_forward_length")
        model.vocab_size = _get("vocab_size")
        model.context_length = _get("context_length")
        model.expert_count = _get("expert_count")
        model.expert_used_count = _get("expert_used_count")

        # GGUF rarely carries an explicit `vocab_size`; the authoritative
        # figure is the length of the tokenizer's token list.
        tokens = meta.get("tokenizer.ggml.tokens")
        if tokens:
            model.vocab_size = len(tokens)

        # Estimate parameter count from dimensions
        model.param_count_b = _GGUFParser._estimate_params(model)

    @staticmethod
    def _estimate_params(model: GGUFModel) -> float:
        """Rough parameter count estimate from model dimensions."""
        d = model.embedding_dim
        n_layers = model.block_count
        n_heads = model.head_count
        n_kv_heads = model.head_count_kv
        ffn = model.feed_forward_dim
        vocab = model.vocab_size
        n_experts = model.expert_count

        if d == 0 or n_layers == 0:
            return 0.0

        # Embedding
        params = vocab * d  # token embeddings

        # Per-layer: attention + FFN
        for _ in range(n_layers):
            # Self-attention: Q, K, V, O projections
            params += d * d * 3  # Q, K, V (simplified: d * d each)
            params += d * d      # O projection

            if n_kv_heads and n_kv_heads != n_heads:
                # GQA/MQA: adjust K,V sizes
                kv_dim = d * n_kv_heads // n_heads
                params -= d * d * 2  # remove standard K,V size
                params += d * kv_dim * 2  # add actual K,V size

            # FFN
            if n_experts > 0:
                # MoE: expert_count * (gate + up + down) / expert_used_count
                params += n_experts * (ffn * d * 2 + ffn * d) / max(model.expert_used_count, 1)
            else:
                params += d * ffn * 3  # gate, up, down projections

        # Final RMS norm
        params += d

        # LM head (tied embedding or separate)
        if not model.metadata.get(f"{model.architecture}.tied_embeddings", False):
            params += vocab * d

        return params / 1e9
