"""
GGUF reader — parse and inspect GGUF model files.

Provides:
  - GGUFParser: raw binary → structured metadata
  - GGUFModel: high-level model info (architecture, dimensions, file_type)
  - QuantType: enum of all GGML quantization types
"""
