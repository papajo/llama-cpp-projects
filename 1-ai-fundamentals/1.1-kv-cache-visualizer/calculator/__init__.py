"""
kv_cache_calculator — Core math for KV cache memory estimation in llama.cpp.

Exposes:
  - KVQuant: quantization types with byte sizes
  - ModelConfig: dataclass for transformer architecture dimensions
  - KVCacheCalculator: computes memory footprints
  - ModelPresets: known model architectures
"""
from .models import KVQuant, ModelConfig, MemoryBreakdown, KVCacheResult
from .calculator import KVCacheCalculator
from .presets import ModelPresets

__all__ = [
    "KVQuant",
    "ModelConfig",
    "MemoryBreakdown",
    "KVCacheResult",
    "KVCacheCalculator",
    "ModelPresets",
]
