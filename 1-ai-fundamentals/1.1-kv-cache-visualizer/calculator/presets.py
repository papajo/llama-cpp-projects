"""
ModelPresets — known transformer architectures for quick KV cache analysis.

All values sourced from official model configs (config.json) or GGUF metadata.
"""

from __future__ import annotations

from typing import Dict, List

from .models import ModelConfig


class ModelPresets:
    """Registry of known model architectures."""

    _registry: Dict[str, ModelConfig] = {}

    @classmethod
    def register(cls, config: ModelConfig) -> None:
        cls._registry[config.name] = config

    @classmethod
    def get(cls, name: str) -> ModelConfig:
        if name in cls._registry:
            return cls._registry[name]
        # fuzzy match
        for key, cfg in cls._registry.items():
            if name.lower() in key.lower():
                return cfg
        raise KeyError(
            f"Unknown model: {name!r}.  Available: {', '.join(cls.list_names())}"
        )

    @classmethod
    def list_names(cls) -> List[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def all(cls) -> List[ModelConfig]:
        return list(cls._registry.values())


# ---------------------------------------------------------------------------
# Register presets
# All data from official config.json / GGUF metadata.
# n_kv_heads reflects GQA; for MQA all = 1.
# ---------------------------------------------------------------------------

ModelPresets.register(ModelConfig(
    name="Llama-3.1-8B",
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,     # GQA: 4 query heads per KV head
    head_dim=128,
    max_ctx=131072,   # 128K (with RoPE scaling)
))

ModelPresets.register(ModelConfig(
    name="Llama-3.2-1B",
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    max_ctx=131072,
))

ModelPresets.register(ModelConfig(
    name="Llama-3.2-3B",
    n_layers=28,
    n_heads=24,
    n_kv_heads=8,
    head_dim=128,
    max_ctx=131072,
))

ModelPresets.register(ModelConfig(
    name="Llama-3.1-70B",
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,     # GQA: 8:1
    head_dim=128,
    max_ctx=131072,
))

ModelPresets.register(ModelConfig(
    name="Llama-3.1-405B",
    n_layers=126,
    n_heads=128,
    n_kv_heads=16,    # GQA: 8:1
    head_dim=128,
    max_ctx=131072,
))

ModelPresets.register(ModelConfig(
    name="Mistral-7B-v0.3",
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,     # GQA
    head_dim=128,
    max_ctx=32768,
))

ModelPresets.register(ModelConfig(
    name="Mixtral-8x7B",
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    max_ctx=32768,
    n_experts=8,
    n_active_experts=2,
))

ModelPresets.register(ModelConfig(
    name="Qwen-2.5-7B",
    n_layers=28,
    n_heads=28,
    n_kv_heads=4,     # GQA: 7:1
    head_dim=128,
    max_ctx=32768,
))

ModelPresets.register(ModelConfig(
    name="Qwen-2.5-32B",
    n_layers=64,
    n_heads=40,
    n_kv_heads=8,     # GQA: 5:1
    head_dim=128,
    max_ctx=32768,
))

ModelPresets.register(ModelConfig(
    name="Qwen-2.5-72B",
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,     # GQA: 8:1
    head_dim=128,
    max_ctx=32768,
))

ModelPresets.register(ModelConfig(
    name="DeepSeek-V3",
    n_layers=61,
    n_heads=128,
    n_kv_heads=128,   # MHA (no GQA for DeepSeek-attention)
    head_dim=128,
    max_ctx=65536,
    n_experts=256,     # MoE: 256 routed experts
    n_active_experts=8,
))

ModelPresets.register(ModelConfig(
    name="DeepSeek-R1-Distill-Llama-8B",
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    max_ctx=131072,
))

ModelPresets.register(ModelConfig(
    name="Gemma-2-9B",
    n_layers=42,
    n_heads=16,
    n_kv_heads=16,    # MHA
    head_dim=256,
    max_ctx=8192,
))

ModelPresets.register(ModelConfig(
    name="Gemma-2-27B",
    n_layers=46,
    n_heads=32,
    n_kv_heads=16,    # GQA: 2:1
    head_dim=256,
    max_ctx=8192,
))

ModelPresets.register(ModelConfig(
    name="Phi-3-mini-4K",
    n_layers=32,
    n_heads=32,
    n_kv_heads=32,    # MHA
    head_dim=96,
    max_ctx=4096,
))

ModelPresets.register(ModelConfig(
    name="Phi-3-medium-128K",
    n_layers=40,
    n_heads=40,
    n_kv_heads=10,    # GQA: 4:1
    head_dim=128,
    max_ctx=131072,
))

ModelPresets.register(ModelConfig(
    name="Command-R-plus",
    n_layers=40,
    n_heads=64,
    n_kv_heads=8,     # GQA: 8:1
    head_dim=96,
    max_ctx=131072,
))

ModelPresets.register(ModelConfig(
    name="CodeLlama-34B",
    n_layers=48,
    n_heads=64,
    n_kv_heads=8,     # GQA: 8:1
    head_dim=128,
    max_ctx=16384,
))


ModelPresets.register(ModelConfig(
    name="Ornith-1.0-9B-MLX-4bit",
    n_layers=32,
    n_heads=16,
    n_kv_heads=4,
    head_dim=256,
    max_ctx=262144,
))
