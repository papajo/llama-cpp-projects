# Real vs. mock drift — worker `rag` (4-vector-db-rag, 2-langchain)

Server build `b11046-60081bb2b`.
chat `http://127.0.0.1:8090` SmolLM2-360M-Instruct Q8_0, n_ctx 2048, n_embd 960.
embed `http://127.0.0.1:8081` nomic-embed-text-v1.5 Q8_0, n_embd 768, n_ctx 2048.

---

## 1. Embeddings come back already L2-normalised

- **Project / test:** `4.1-embedding-norm-ablation` — `tests/test_embeddings.py::TestEmbeddingClient`,
  `tests/test_ablation.py` (whole ablation premise).
- **What the mock asserted:** canned vectors are arbitrary and non-unit —
  `{"data": [{"embedding": [0.1, 0.2, 0.3]}]}` (norm ≈ 0.374), so
  `l2_normalize` visibly changes them and `norm-*` vs `unnorm-*` treatments are
  distinguishable.
- **What the real server returned:** every `/v1/embeddings` vector has
  L2 norm `1.0` (measured mean 0.99999998, std 2.9e-08 across the mini corpus).
  Consequence: `unnorm-cosine == norm-cosine` and `unnorm-dot == norm-dot`
  exactly, so the project's four-treatment ablation is **degenerate** against
  this server — it measures nothing. The project code is correct; the
  experiment simply has no signal to find here.
- **Cause:** llama-server normalises pooled embeddings by default,
  `--embd-normalize 2` (L2/Euclidean). Passing `--embd-normalize -1` disables
  it and would make the ablation meaningful again. The native `/embedding`
  endpoint is normalised too, so switching endpoints is not a workaround.
- **Covered live by:** `test_live_embeddings.py::test_server_returns_l2_normalised_vectors`
  and `::test_normalisation_treatments_coincide_on_real_vectors`.

## 2. Embedding response envelope is richer than the canned one

- **Project / test:** `4.1-embedding-norm-ablation` — `tests/test_embeddings.py::TestEmbeddingClient::test_embed_success`.
- **What the mock asserted:** `{"data": [{"embedding": [...]}]}` only.
- **What the real server returned:** per-item `index` (int) and
  `object: "embedding"` alongside `embedding`, plus top-level
  `model`, `object: "list"`, and `usage` with **only**
  `{"prompt_tokens", "total_tokens"}` — no `completion_tokens` on an embedding
  request. A list `input` returns one row per text with sequential `index`.
- **Cause:** the OpenAI-compat `/v1/embeddings` handler in llama-server always
  emits the full envelope; nothing turns it off. Mock updated to match.
- **Covered live by:** `test_live_embeddings.py::test_raw_response_envelope`,
  `::test_batch_input_returns_indexed_rows`.
