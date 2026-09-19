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
