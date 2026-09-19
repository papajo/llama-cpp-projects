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

## 7. `cache_hit_ratio` was structurally always 0.0 — REAL BUG (fixed)

- **Project / test:** `4.3-prompt-cache-chunking` — `prompt_cache_chunking/cache_sim.py::simulate_cache`,
  vacuously "covered" by `tests/test_cache_sim.py`.
- **What the mock asserted:** nothing falsifiable. This project opens no
  sockets, so there is no canned HTTP response — but its assertions were
  tautological:
  `assert result.total_tokens > result.cached_tokens or True` can never fail,
  and `assert 0.0 <= result.cache_hit_ratio <= 1.0` / `cached_tokens >= 0` are
  both satisfied by a constant 0.
- **What the real behaviour was:** the chunk-level cache-hit branch was dead
  code. `chunk_prefix = f"{prefix}|{idx}"` was *tested* against
  `seen_prefixes` but only the bare `prefix` was ever *added* to it, so
  `chunk_prefix` could never match. Worse, `prefix` embeds the query, and all
  10 default-corpus queries are unique, so the system+query branch never fired
  either. Net effect: `cached_tokens == 0` and `cache_hit_ratio == 0.0` for
  every strategy, every run — the project's headline metric measured nothing,
  and the whole strategy comparison was vacuous.
- **What the real server returned (ground truth for the fix):** three prompts
  sharing a system prompt plus an identical chunk block, differing only in the
  trailing question, reported
  `usage.prompt_tokens_details.cached_tokens` = 9, then **135**, then **135**
  out of ~145 `prompt_tokens`. `timings.cache_n` carries the same number. So
  llama.cpp really does reuse a shared prefix across *different* queries —
  exactly what the docstring promised and the code failed to model.
- **Cause:** not a server flag — a missing `seen_prefixes.add(...)` plus a
  cache key that included the query. Fixed by keying the chunk region on the
  cumulative chunk sequence (what a KV cache actually prefix-matches on) and
  recording it. Hit ratios went from 0.0000 to ~0.19 across all four
  strategies; all 24 pre-existing tests still pass. The tautological assertion
  was made falsifiable and 3 regression tests added.
- **Covered live by:** `test_live_cache.py::test_shared_prefix_is_reused_across_different_queries`,
  `::test_more_shared_prefix_means_more_cached_tokens`,
  `::test_simulated_hit_ratio_is_non_degenerate`.

## 8. `chars // 4` over-estimates real token counts by ~20-26%

- **Project / test:** `4.3-prompt-cache-chunking` — `simulate_cache` counts
  every span as `max(1, len(text) // 4)`; no offline test compares this to a
  tokenizer.
- **What the mock asserted:** that the estimate *is* the token count — token
  totals are reported as `total_tokens` in the markdown/JSON reports with no
  caveat.
- **What the real server returned:** via `POST /tokenize` with the SmolLM2
  vocab — system prompt: 66 chars, estimated 16, **actual 13**; a chunk block:
  559 chars, estimated 139, **actual 110**. Consistently ~20-26% high.
- **Cause:** no flag; `chars/4` is an English-average heuristic and the real
  count depends entirely on the loaded model's vocab (SmolLM2 `n_vocab` 49152
  on the chat server vs 30522 on nomic-embed, so the two servers tokenize the
  same string differently). Left as-is deliberately — every strategy shares the
  estimator, so cross-strategy *ratios* remain valid — but now documented in
  the code and pinned by a live test.
- **Covered live by:** `test_live_cache.py::test_chars_over_four_overestimates_real_tokens`,
  `::test_estimator_is_monotonic_in_real_tokens`.

## 9. Canned vectors are orthogonal; real embeddings are crowded

- **Project / test:** `4.4-multi-tenant-rag` — `tests/test_tenant_store.py`,
  `tests/test_pipeline.py` (and the same pattern in 4.1/4.2).
- **What the mock asserted:** hand-written basis-like vectors
  (`[1, 0, 0]`, `[0, 1, 0]`, ...), so every non-matching similarity is exactly
  `0.0` and the "right" document wins by an unmissable margin.
- **What the real server returned:** 768-dim nomic embeddings sit in a narrow
  cone — two deliberately unrelated documents ("Acme Corp quarterly revenue
  report" vs "Healthplus patient intake policy") still score cosine **> 0.05**,
  and semantically close documents across *different tenants* compete directly
  for top-k slots. Measured on the real corpus: isolated retrieval leaked
  **0** documents (the guarantee holds — it is structural, not score-based),
  while the deliberately non-isolated baseline leaked **5** of 27 retrieved.
- **Cause:** no flag — a property of real sentence-embedding geometry
  (anisotropy) that clean synthetic vectors do not reproduce. Nothing to fix:
  4.4's isolation is enforced by dict partitioning and tenant-relative
  indices, so it is score-independent by construction. Recorded because a
  score-based or threshold-based isolation scheme would pass the offline suite
  and leak in production.
- **Covered live by:** `test_live_tenant_isolation.py::test_real_vectors_are_not_orthogonal`,
  `::test_search_only_returns_own_tenant_documents` (checks document identity,
  not just index), `::test_non_isolated_baseline_does_leak`.

**No code defects found in 4.4** — isolation held on real vectors in every path.

## 10. The `model` field is ignored; the response reports the loaded model

- **Project / test:** `2.1-langchain-router-failover` — `tests/test_router.py`
  (`make_fake_response`), whose whole premise is alias-based routing.
- **What the mock asserted:** the canned responses are built by the test, so an
  alias like `"coder"` round-trips unchallenged. Nothing checks what the server
  does with an unrecognised alias.
- **What the real server returned:** llama-server **silently ignores** the
  `model` field. Asking for `model: "coder"` returns
  `model: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0"` with HTTP 200 — no
  404, no error. So a mocked test asserting that an unknown model id produces
  an error would be asserting behaviour the real server does not have.
- **Cause:** a plain `llama-server` loads exactly one model (`-m`/`-hf`) and
  serves every request from it; the field only matters to a multi-model
  front-end (llama-swap, or router mode with several `--model-alias` entries).
  Nothing to fix — the router's own `ModelNotAvailableError` guard catches
  unregistered aliases *client-side*, before any request, which is the only
  thing standing between a typo and a silently wrong model.
- **Covered live by:** `test_live_router.py::test_server_ignores_the_model_field`,
  `::test_unregistered_alias_is_rejected_before_any_request`.

## 11. Real SSE opens with `content: null` and closes with an empty delta

- **Project / test:** `2.1-langchain-router-failover` — `tests/test_router.py::test_stream_basic`
  via `make_fake_stream_chunk`.
- **What the mock asserted:** every chunk is `{"delta": {"content": "<text>"}}`,
  with the terminal chunk modelled as `content: ""` plus a `finish_reason`.
- **What the real server returned:** three distinct chunk shapes —
  1. a role-priming first chunk: `{"delta": {"role": "assistant", "content": null}}`
     — `content` is JSON **null**, not `""`;
  2. text chunks: `{"delta": {"content": "..."}, "finish_reason": null}`;
  3. a terminal chunk with **no `content` key at all**:
     `{"delta": {}, "finish_reason": "length"}`, carrying a `timings` block;
  then a literal `data: [DONE]`. Every chunk also has `object:
  "chat.completion.chunk"`, `id`, `created`, `model` and `system_fingerprint`.
- **Cause:** llama-server's OpenAI-compatible streaming handler; the leading
  role chunk and trailing stats chunk are standard for it. `_parse_stream_chunk`
  survives all three only because `delta.get("content", "")` returns `None` for
  case 1 and `""` for case 3, and `if not content` is falsy for both — i.e. the
  `null` case works by luck rather than by design. Note the consequence:
  because the terminal chunk is dropped, **`finish_reason` is never surfaced
  when streaming**, only when using `invoke()`.
- **Also found (framework, not server):** LangChain's own
  `BaseChatModel.stream` appends a final empty `AIMessageChunk` with
  `chunk_position="last"`, so a live consumer sees one `content == ""` chunk
  that the router never produced.
- **Covered live by:** `test_live_router.py::test_stream_yields_text_chunks`.

## 12. Cold-start retry (503/504) is not reproducible on this build

- **Project / test:** `2.1-langchain-router-failover` — `tests/test_router.py::TestColdStart*`
  (six offline tests) and `test_cold_start_then_success` / `test_cold_start_exhausted`.
- **What the mock asserted:** the router retries with exponential backoff when
  the server answers 503/504 or an error body containing "cold"/"sleep"/
  "unavailable", then succeeds or raises `ModelColdStartError`.
- **What the real server returned:** always HTTP 200. A plain single-model
  llama-server never emits 503/504 for a sleeping model — the model is resident
  from startup, so there is nothing to wake.
- **Cause:** needs a front-end that unloads idle models — llama-swap, or router
  mode with `--sleep-idle-seconds`. Unsupported in this CPU-only single-model
  setup, so the path stays mock-only; the live test asserts only that the
  server is healthy and that a real 400 is *not* misread as a cold start
  (verified: `{"error":{"code":400,"type":"invalid_request_error"}}` contains
  none of the cold-start keywords, so it raises instead of burning 3 retries).
- **Covered live by:** `test_live_router.py::test_cold_start_path_is_not_reproducible_live`,
  `::test_real_400_is_raised_not_treated_as_cold_start`.

**Note (test-infra):** `pytest-asyncio` is not installed in the shared venv, so
2.1's live async tests drive `_agenerate`/`_astream` with `asyncio.run()` from
sync tests. The offline suite covers no async path at all, so these are the
only coverage those two methods have.

## 13. `json_schema` sent as a JSON string — REAL BUG (fixed), total feature failure

- **Project / test:** `2.2-grammar-structured-output` — `parser/parser.py::GrammarOutputParser.invoke`,
  exposed by `tests/test_live_grammar.py::test_returns_a_valid_model_instance`.
- **What the mock asserted:** nothing about the request at all. Every offline
  test in `test_parser.py` patches `_get_client` and asserts only that the
  parser parses the JSON the mock hands back. **No offline test inspected the
  request body**, so the field was never checked.
- **What the real server returned:** HTTP **400** for every single call —
  `{"error":{"code":400,"message":"Field 'json_schema': \"json_schema\": JSON
  schema conversion failed:\nJSON schema error at #: schema must be an
  object","type":"invalid_request_error"}}`. The parser did
  `schema_str = json.dumps(json_schema)` and sent the *string*; llama-server's
  `/completion` requires the schema **object**. So this project's one headline
  feature — guaranteed schema-conformant output — failed 100% of the time
  against a real server while its offline suite was fully green.
- **Cause:** not a server flag. llama-server parses `json_schema` with
  `json_value.is_object()` before converting it to GBNF and rejects any other
  JSON type. Fixed by passing the dict. Verified end to end: the real server
  now returns `Person(name='John Smith', age=30)`. Four regression tests added
  (`TestRequestBody`) that assert on the wire format — `json_schema` is a dict,
  the endpoint is `/completion`, and the native param is `n_predict`, not
  `max_tokens`.
- **Grammar itself works well:** with the fix, nested models, lists,
  `Literal` enums, bools and optionals all come back conformant from a 360M
  model — the sampler cannot emit an off-grammar token, so structure is right
  even when values are invented (`email: "john@example.com"` for a prompt that
  mentions no email). That is *model too small*, not a defect, and no live test
  asserts field values.

## 14. `/completion` returns nothing at all for an empty prompt

- **Project / test:** `2.2-grammar-structured-output` — `tests/test_live_grammar.py::test_empty_prompt_generates_nothing_and_raises_clearly`.
- **What the mock asserted:** that grammar-constrained output is always valid
  JSON, with the parser's `json.loads` failure branch commented "This should
  never happen with grammar constraints".
- **What the real server returned:** for `prompt: ""` — `content: ""`,
  `tokens_predicted: 0`, `stop_type: "none"`. It generates **zero tokens**, so
  the grammar has nothing to constrain and the "always valid JSON" guarantee is
  vacuous. Every non-empty prompt returned conformant JSON
  (`stop_type: "eos"`, 20-34 tokens).
- **Cause:** no flag — a grammar restricts *which* tokens may be sampled, it
  cannot compel sampling to happen. Left as-is deliberately: the parser's guard
  raises a clear `ValueError` naming the offending output, which is the right
  behaviour for a degenerate input. Now pinned by a live test so the guard is
  no longer dead code.

## 15. Native `/completion` response has keys the canned one omits

- **Project / test:** `2.2-grammar-structured-output` — `tests/test_parser.py::make_completion_response`.
- **What the mock asserted:** a 5-key body — `content`, `tokens_predicted`,
  `tokens_evaluated`, `truncated`, `model`.
- **What the real server returned:** all of those plus `stop`, `stop_type`
  (`"eos"` / `"none"` / `"limit"`), `timings`, `prompt`, `has_new_line`,
  `index`, `tokens_cached`, and a full `generation_settings` block that echoes
  the applied sampler config — including `generation_settings.grammar`, which
  is the compiled GBNF and therefore direct proof the constraint was applied.
- **Cause:** llama.cpp's native (non-OpenAI) `/completion` endpoint is verbose
  by design. The mock stays minimal since the parser reads only `content`, but
  the live test now pins the real shape and asserts the grammar was echoed back.
- **Covered live by:** `test_live_grammar.py::test_native_completion_response_shape`.

## 16. Vision is unsupported, and fails with 500 rather than 501

- **Project / test:** `2.3-multimodal-doc-agent` — `tests/test_extraction_chain.py`
  (all mocked vision paths), now documented by `tests/test_live_vision.py`.
- **What the mock asserted:** that `/v1/chat/completions` accepts a multimodal
  content-parts list (`{"type": "image_url", "image_url": {"url": "data:..."}}`)
  and returns a normal text completion describing the image.
- **What the real server returned:** HTTP **500** —
  `{"error":{"code":500,"message":"image input is not supported - hint: if this
  is unexpected, you may need to provide the mmproj","type":"server_error"}}`.
  `/props` confirms `modalities: {vision: false, video: false, audio: false}`.
  Note the **inconsistency**: llama-server reports missing *embeddings* and
  missing *reranking* as `501 / not_supported_error`, but missing *vision* as
  `500 / server_error`. Anything switching on the status code to distinguish
  "capability absent" from "server broke" will misclassify this one.
- **Cause:** no multimodal projector is loaded — needs `--mmproj` plus a vision
  GGUF. SmolLM2-360M-Instruct is a text-only model, so this is not fixable by a
  flag alone on this build. Unsupported on CPU; the two extraction-chain tests
  are marked `xfail(strict=True, raises=httpx.HTTPStatusError)` so they will
  start failing (and demand attention) the moment vision is enabled.
- **Verified the failure is loud, not silent:** `invoke_with_images` propagates
  `httpx.HTTPStatusError` rather than dropping the image and answering from the
  text alone — which would have produced plausible output with the image
  ignored, a much more dangerous outcome than an exception.
- **Non-vision paths do work live** and are covered: the text-only `_generate`
  path, and the multimodal envelope with an empty image list (which the server
  accepts, proving the request shape itself is valid and only the image block is
  refused). Real PNGs are loaded from disk by the project's own `ImageLoader`,
  including the resize path, and round-tripped back through base64.
