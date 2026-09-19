# Drift log — worker `graph` (5-langgraph, 6-mcp)

Real servers used:
- chat  `http://127.0.0.1:8090` — SmolLM2-360M-Instruct Q8_0, n_ctx 2048, build `b11046-60081bb2b`
- embed `http://127.0.0.1:8081` — nomic-embed-text-v1.5 Q8_0, n_embd 768

---

## 1. 5.1-draft-verify-graph — chat completion envelope is far richer than the mock

**Test:** `tests/test_llm.py::TestLlamaClient::test_complete_success`,
`tests/test_nodes.py::_mock_chat_response`

**What the mock asserted:** the response body is exactly
```json
{"choices": [{"message": {"content": "Hello!"}}]}
```

**What the real server returned:** every OpenAI envelope field plus two
llama.cpp-only blocks (`timings`, `usage.prompt_tokens_details`):
```json
{
  "choices": [{"finish_reason": "length", "index": 0,
               "message": {"role": "assistant", "content": "Hi, I'm here to help you"}}],
  "created": 1789829233,
  "model": "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0",
  "system_fingerprint": "b11046-60081bb2b",
  "object": "chat.completion",
  "usage": {"completion_tokens": 8, "prompt_tokens": 32, "total_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 31}},
  "id": "chatcmpl-aKdOb19SjynKzSDgbeclm4DEyuq5XX9z",
  "timings": {"cache_n": 31, "prompt_n": 1, "prompt_ms": 72.875, "predicted_n": 8,
              "predicted_ms": 240.826, "predicted_per_second": 29.07, ...}
}
```
Missing from the mock: `finish_reason`, `index`, `message.role`, `created`,
`model`, `object`, `id`, `system_fingerprint`, `usage.*`, `timings.*`.

**Cause:** `timings` is emitted by llama-server unless `--no-timings`;
`usage.prompt_tokens_details.cached_tokens` reflects llama.cpp's prompt-cache
reuse and has no OpenAI-cloud equivalent. `system_fingerprint` is the
llama.cpp build string, not a model hash.

**Impact:** `LlamaClient.complete()` only reads
`choices[0].message.content`, so the thin mock did not hide a bug — but it
would mislead any later change that reads `usage` (cost/token accounting) or
`finish_reason` (truncation detection). `finish_reason` is `"length"`, not
`"stop"`, on every short-`max_tokens` call, which is exactly the case these
tests use.

**Fixed:** mock envelopes in `test_llm.py` and `test_nodes.py` now carry the
real field set. Live coverage: `tests/test_live_graph.py`.

---

## 2. 5.1-draft-verify-graph — unknown `model` is silently served, never rejected

**Test:** no offline test asserted this; recorded because it is a trap for one.

**What a mock would assert:** an unknown model id produces an error response.

**What the real server returned:** HTTP 200 and a normal completion, with
`"model"` echoing the *loaded* model id
(`HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0`), not the requested one.

**Cause:** llama-server loads exactly one model per process (`-m`/`-hf`) and
ignores the request's `model` field entirely. There is no model-routing layer,
so there is nothing to 404 on.

**Live coverage:** `test_live_graph.py::test_complete_ignores_unknown_model`.

---

## 3. 5.1-draft-verify-graph — `VerifyNode` never gets JSON from this model

**Classification: model too small — NOT a code bug, nothing fixed.**

`VERIFY_SYSTEM` asks for `{"score", "issues", "verdict"}`. Real output for
task "Name one colour." / draft "Blue.":

> "The response evaluates the response against the task. The response does not
> mention any specific colour, suggesting that it does not recognize the answer
> to the question. However, it does mention the response, indicating that it is
> a response."

`_parse()` therefore always takes the `json.JSONDecodeError` fallback, finds no
digit, and yields `score=0, verdict="fail",
issues=["Could not parse structured feedback"]`. The mocked tests feed
hand-written clean JSON, so they only ever exercise the happy path.

The offline tests are not wrong to test the JSON path (a larger model would hit
it), so they are left alone. The live test asserts structure only.

---

## 4. 6.3-model-discovery-router-mgr — mocks invented a multi-model OpenAI catalogue

**Test:** `tests/test_registry.py` (`SAMPLE_MODELS`, `test_discover`,
`test_list_models`, `test_discover_if_empty`, `test_from_openai_response`),
`tests/test_server.py` (all four tool tests)

**What the mock asserted:** `/v1/models` returns a four-model catalogue from
cloud providers, and `discover()` yields 4 `ModelInfo`s:
```json
{"data": [
  {"id": "gpt-4",                  "object": "model", "owned_by": "openai"},
  {"id": "gpt-3.5-turbo",          "object": "model", "owned_by": "openai"},
  {"id": "text-embedding-ada-002", "object": "model", "owned_by": "openai"},
  {"id": "llama-3-70b-instruct",   "object": "model", "owned_by": "meta"}
]}
```

**What the real server returned:** one model, `owned_by: "llamacpp"`, a `meta`
block, and an Ollama-style `models` list beside the OpenAI `data` list:
```json
{
  "models": [{"name": "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0",
              "capabilities": ["completion"], "details": {"format": "gguf", ...}, ...}],
  "object": "list",
  "data": [{"id": "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0",
            "aliases": ["HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0"],
            "tags": [], "object": "model", "created": 1789829232,
            "owned_by": "llamacpp",
            "meta": {"vocab_type": true, "n_vocab": 49152, "n_ctx": 2048,
                     "n_ctx_train": 8192, "n_embd": 960, "n_params": 361821120,
                     "size": 384618240, "ftype": "Q8_0"}}]
}
```
The embed server on 8081 is a *separate process* reporting
`nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0` with `meta.n_embd: 768`.
No `permission` list is ever sent.

**Cause:** llama-server loads exactly one model per process (`-m` / `-hf`), so
`/v1/models` is a one-element list by construction — there is no multi-model
router to discover. The dual payload is deliberate: llama-server serves both
the OpenAI `/v1/models` contract and Ollama's `/api/tags`-style contract from
one handler, so `data` and `models` describe the same single model. `meta` is
llama.cpp-specific (no OpenAI equivalent).

**Fixed:** both test files now mock the real payload and assert one
`llamacpp`-owned model. The pure routing tests (`test_route_round_robin` etc.)
still build two-model registries directly — they never touch HTTP, and
round-robin needs >1 model to mean anything, so they were left alone.

---

## 5. 6.3-model-discovery-router-mgr — `context_length` was never populated

**Classification: real bug — project code fixed.**

**Test:** `tests/test_live_registry.py::test_discover_populates_context_length`
(failed before the fix: `assert 0 == 2048`)

**What the mock asserted:** nothing. No offline test touched
`ModelInfo.context_length`, and the mocked payloads had no `meta` block, so the
gap was invisible offline.

**What the real server returned:** `data[0].meta.n_ctx == 2048`.

**Cause:** `ModelInfo` declares `context_length: int = 0`, but
`from_openai_response()` only mapped `id`, `object`, `owned_by` and
`permission`. The field was dead — permanently 0 for every discovered model —
even though the server supplies the value. llama.cpp reports the loaded context
window as `meta.n_ctx` (2048 here, vs `n_ctx_train` 8192, because the server
was started with `--ctx-size 2048`); a plain OpenAI payload has no `meta`, which
is why the mapping was missed.

**Fixed:** `from_openai_response()` now reads `meta.n_ctx`, defaulting to 0 when
there is no `meta` block. Regression-covered offline by
`test_from_openai_response` and `test_from_openai_response_without_meta`.

---

## 6. 6.3-model-discovery-router-mgr — `server._registry` cache leaked across tests

**Classification: real bug (in the offline tests) — fixed.**

`model_router/server.py` holds a module-global `_registry` and every tool calls
`discover_if_empty()`. `tests/test_server.py` never reset it, so the first test
to run populated the cache and the remaining three were served from it,
ignoring their own `urlopen` mock entirely. The suite passed only because every
test mocked the same `gpt-4` body.

**Fixed:** an `autouse` `reset_registry` fixture gives each test a fresh
`ModelRegistry`. The live tests use the same pattern (`fresh_server_registry`),
which is required there — otherwise a cached offline registry would satisfy the
live assertions without a single real request.

---

## 7. 6.1-slots-metrics-server — `/metrics` is unavailable on this build

**Classification: unsupported (llama-server flag not enabled) — xfail, not faked.**

**Test:** `tests/test_live_metrics.py::test_llamacpp_metrics_endpoint` (xfail,
`strict=True`), with the gap asserted positively by
`test_llamacpp_metrics_really_is_501`

**What the mock asserted:** nothing — `metrics_server` never calls
llama-server. Worth recording anyway, because the project's name invites
someone to add a `/metrics` scrape and assume it works.

**What the real server returned:** HTTP 501
```json
{"error": {"code": 501,
           "message": "This server does not support metrics endpoint. Start it with `--metrics`",
           "type": "not_supported_error"}}
```
`/props` corroborates: `endpoint_metrics: false`.

**Cause:** the `--metrics` flag. llama-server only registers the Prometheus
`/metrics` handler when started with it; this instance was not. It is a startup
flag, not a CPU/GPU limitation — but it cannot be enabled without restarting the
server, so no code change can reach it.

The xfail is `strict=True` deliberately: if someone restarts llama-server with
`--metrics`, the test XPASSes, which pytest reports as a failure and forces a
real assertion to be written here.

---

## 8. 6.1-slots-metrics-server — `/slots` is a bare LIST, not a dict

**Test:** `tests/test_live_metrics.py::test_llamacpp_slots_endpoint_shape`

**What a mock would assert:** a wrapper object, e.g. `{"slots": [...]}` —
the natural guess, and the shape most other llama-server endpoints use
(`/v1/models` wraps in `data`, `/v1/embeddings` wraps in `data`).

**What the real server returned:** a top-level JSON **array**:
```json
[{"id": 0, "n_ctx": 2048, "speculative": false, "is_processing": false,
  "id_task": 1967, "n_prompt_tokens": 205, "n_prompt_tokens_processed": 0,
  "n_prompt_tokens_cache": 0,
  "params": {"seed": 4294967295, "temperature": 0.0, "top_k": 40,
             "top_p": 0.949999988079071, "min_p": 0.05000000074505806,
             "repeat_last_n": 64, "repeat_penalty": 1.0, ...}}, ...]
```
Three slots (`id` 0-2), matching `/props` `total_slots: 3`. Any code doing
`body["slots"]` gets a `TypeError`.

**Cause:** `/slots` is a llama.cpp-native endpoint, not an OpenAI-compatible
one, so it does not follow the `{"object": "list", "data": [...]}` convention.
It is gated on `--slots` (`/props` reports `endpoint_slots: true` here).
Slot count comes from `--parallel` / `-np` (3 on this instance). Note the
floats are float32-rounded (`0.949999988079071`, not `0.95`), so exact
equality against `0.95` would fail.

**Note on scope:** `metrics_server/collector.py` reports *host* CPU/memory/disk
via psutil and has no llama-server integration at all, despite the project
name. There is no project code path to fix here; these two entries document the
real endpoint contract so the merged REAL_VS_MOCK.md can carry it.

---

## 9. All 6-mcp projects — SmolLM2 cannot do tool calling

**Classification: unsupported (model chat template) — affects any MCP→tool round trip.**

`/props` → `chat_template_caps`:
```json
{"supports_tools": false, "supports_tool_calls": false,
 "supports_parallel_tool_calls": false, "supports_object_arguments": false,
 "supports_typed_content": false, "supports_reasoning_effort": false,
 "supports_string_content": true, "supports_system_role": true}
```

**Cause:** the model's bundled Jinja chat template has no tool/function-calling
section, so llama-server advertises no tool support. Passing `tools=[...]` to
this server cannot produce `tool_calls`. This is a property of the SmolLM2-360M
template, not of any project's code, so no live test drives an MCP tool through
the model. The MCP tools in 6.1/6.3 are therefore called directly (as the
offline tests do) rather than via model-chosen invocation.

---

## 10. 5.6 / 5.8 — real concurrency is capped at 3 by `--parallel`

**Classification: environment constraint — live tests sized to it, no code changed.**

**Test:** `5.6-parallel-execution/tests/test_live_core.py`,
`5.8-map-reduce/tests/test_live_core.py`

**What the mocks asserted:** offline fan-out is unbounded — synthetic lambdas
return instantly, so `ParallelAgent(max_workers=4)` and
`MapReduceAgent(max_workers=4)` appear to scale freely.

**What the real server does:** `/props` reports `total_slots: 3`, and `/slots`
lists exactly three slot objects (`id` 0-2), each with `n_ctx: 2048`. A fourth
concurrent request waits for a slot instead of running in parallel.

**Cause:** llama-server's `--parallel` / `-np` flag (3 on this instance) fixes
the number of decode slots. `max_workers` above that number buys nothing; the
extra threads block on the HTTP response.

**Effect on the tests:** live fan-out is capped at 3 so the wall-clock
assertion in
`5.6::test_parallel_is_faster_than_serial_would_be` stays meaningful — it
asserts `total < sum(per_task) * 0.9`, which genuinely fails if the executor
serialises but tolerates a loaded CPU. With a fan-out above the slot count that
assertion would flake, since queued requests serialise for reasons that have
nothing to do with the code under test. 5.8 uses the embed server for its wider
fan-outs because embeddings are far cheaper on CPU than generation.
