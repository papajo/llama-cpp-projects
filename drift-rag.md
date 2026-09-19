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

## 3. `_parse_score` crashes on a bare JSON number — REAL BUG (fixed)

- **Project / test:** `4.2-local-rerank-rag` — `local_rerank_rag/reranker.py::Reranker._parse_score`,
  exposed by `tests/test_live_rerank.py::test_score_returns_normalised_float`.
- **What the mock asserted:** every canned reply is either a JSON **object**
  (`'{"score": 8, "rationale": "..."}'`) or obviously non-JSON prose
  (`"The relevance is 7 out of 10."`). Both paths worked, so the offline suite
  was green.
- **What the real server returned:** SmolLM2-360M ignores the "output a JSON
  object" instruction and replies with the single character `0`. That is *valid
  JSON*, so `json.loads` succeeds and returns `int` 0 — then
  `parsed.get("score", 0)` raised `AttributeError: 'int' object has no
  attribute 'get'`. `AttributeError` is not a subclass of the caught
  `(json.JSONDecodeError, ValueError, TypeError)`, so it escaped `score()`
  entirely and crashed the whole rerank.
- **Cause:** not a server flag — a gap in the project's parser. No mock ever
  produced a bare JSON scalar, so only a real model surfaced it. Fixed by
  treating a bare JSON number as the score itself and routing other non-dict
  JSON (`null`, `true`, `"8"`, lists) to the existing regex fallback. All prior
  behaviour preserved; 3 regression tests added to `tests/test_reranker.py`.
- **Related, and NOT fixed:** the score SmolLM2 assigns is meaningless (it
  returns `0` for a document that plainly answers the query). That is
  *model too small* and the live tests assert only type and range.

## 4. Chat response envelope is far richer than the canned one

- **Project / test:** `4.2-local-rerank-rag` — `tests/test_reranker.py::_mock_llm_response`.
- **What the mock asserted:** `{"choices": [{"message": {"content": ...}}]}`.
- **What the real server returned:** additionally `id` (`chatcmpl-` + 32 chars),
  `created`, `model`, `object: "chat.completion"`, `system_fingerprint`
  (= build, `b11046-60081bb2b`), per-choice `index` and `finish_reason`, and
  `usage` with `prompt_tokens`/`completion_tokens`/`total_tokens` **plus
  `prompt_tokens_details.cached_tokens`**. There is also a top-level
  `timings` block (`cache_n`, `prompt_n`, `prompt_ms`, `predicted_n`,
  `predicted_ms`, `*_per_token_ms`, `*_per_second`) that OpenAI never sends.
- **Cause:** `timings` and `prompt_tokens_details.cached_tokens` are llama.cpp
  extensions, emitted unconditionally by the `/v1/chat/completions` handler
  (`timings_per_token` only adds *per-token* timings on top). The mock is left
  minimal on purpose — the client only reads `choices[0].message.content` — but
  the real shape is now pinned by a live test.
- **Covered live by:** `test_live_rerank.py::test_chat_response_has_llamacpp_only_keys`.

## 5. Native `/v1/rerank` is unavailable — unsupported on this build

- **Project / test:** `4.2-local-rerank-rag` — no offline test covers it;
  documented by `tests/test_live_rerank.py::test_native_rerank_endpoint_unsupported`.
- **What the mock asserted:** nothing. Worth stating explicitly because the
  project is named "local-rerank": it reranks via **LLM-as-judge over
  `/v1/chat/completions`** and never calls a native rerank endpoint, so this
  limitation does not block any of its tests.
- **What the real server returned:** `POST /v1/rerank` and `POST /rerank`
  return HTTP **501** on *both* servers:
  `{"error":{"code":501,"message":"This server does not support reranking. Start it with \`--reranking\`","type":"not_supported_error"}}`.
- **Cause:** llama-server's `--reranking` flag is not set (and it requires a
  cross-encoder/reranker GGUF; neither SmolLM2 nor nomic-embed is one).

## 6. `finish_reason` is not reproducible without `ignore_eos`

- **Project / test:** `4.2-local-rerank-rag` — `tests/test_live_rerank.py::test_finish_reason_length_when_truncated`.
- **What the mock asserted:** canned replies carry no `finish_reason` at all,
  so nothing offline depends on it.
- **What the real server returned:** with `max_tokens=8`, the *same* prompt
  yields `"stop"` on one run and `"length"` on the next — SmolLM2-360M often
  emits EOS within a few tokens even when asked to "count slowly from 1 to
  100". Adding llama.cpp's non-OpenAI `ignore_eos: true` makes it
  deterministically `"length"` with `completion_tokens == max_tokens`.
- **Cause:** model capacity plus the server's default sampling
  (`temperature 0.8`, `top_k 40`, `top_p 0.95` per `/props`). `ignore_eos`
  suppresses the EOS token so generation always runs to `max_tokens`.
