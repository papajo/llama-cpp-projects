"""
Constrained Decoding Playground backend.

Exposes:
  - GrammarConverter: JSON schema → GBNF grammar
  - GrammarDebugger: token-by-token grammar masking analysis
  - FastAPI server: proxy to llama-server with n_probs streaming
"""
