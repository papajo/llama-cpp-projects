# Real servers vs. the mock — drift report

Produced by a three-worker parallel run (`core`, `rag`, `graph`) against two real
`llama-server` instances, merged from `work-core`, `work-rag` and `work-graph`.

| | |
|---|---|
| Build | `b11046-60081bb2b` |
| Chat | `http://127.0.0.1:8090` — SmolLM2-360M-Instruct Q8_0, `n_ctx` 2048 (`n_ctx_train` 8192), `n_embd` 960, `n_vocab` 49152, 3 slots |
| Embeddings | `http://127.0.0.1:8081` — nomic-embed-text-v1.5 Q8_0, `n_embd` 768, `n_ctx` 2048, `n_vocab` 30522 |
| Reranker | `http://127.0.0.1:8082` — bge-reranker-v2-m3 Q8_0, cross-encoder, `--reranking --pooling rank` |
| Result | 36 projects · **808 offline tests** · **290 live tests** · 3 xfailed · 0 failing |

Full per-entry detail lives in [`drift-core.md`](drift-core.md),
[`drift-rag.md`](drift-rag.md) and [`drift-graph.md`](drift-graph.md).
This file is the merged index: **51 drift entries**, grouped by what caused them.

---

## 0. The headline finding: there was never a mock server

The task began from the premise that a mock `llama-server` was running on port
8080 and that tests should be repointed at the real servers. Neither half held.

**There is no mock server anywhere in this repo.** What the suite calls "the
mock" is *in-process HTTP interception*: 17 test files patch
`urllib.request.urlopen` through `unittest.mock`, others use `pytest-httpx`'s
`httpx_mock` fixture, and the rest touch no HTTP at all. No socket is ever
opened. The `http://127.0.0.1:8080` and `http://mock:8080` strings inside those
tests (145 occurrences of `8080` repo-wide) are **inert constructor arguments** —
`http://mock:8080` is not even a resolvable host. Changing an env var, a
conftest, or a client default could not have altered their behaviour by one byte.

**Port 8080 is Open WebUI**, a third-party web app, not this repo's mock and not
llama.cpp. Its `/v1/models` serves a SvelteKit application shell and `ss -ltnp`
shows no `llama-server` bound to it. The cheapest discriminator is the health
endpoint:

| Service | `GET /health` |
|---|---|
| real `llama-server` | `{"status":"ok"}` |
| Open WebUI (8080) | `{"status":true}` |

Because a wrong target here would silently produce plausible-looking results,
`_shared/live_fixtures.py` **hard-fails** on any base URL containing `8080` *and*
on any health response that is not `{"status":"ok"}`. No test in the real-server
run can reach Open WebUI, by construction.

So the work was inverted: rather than repointing existing tests, a **live layer**
was added alongside them — `test_live_*.py`, marked `@pytest.mark.live`, gated
behind `LLM_LIVE=1`, driven by shared fixtures — leaving the fast offline suite
intact. Drift is the *difference* between the two layers, which is what this
document records.

---

## 1. Drift caused by a missing llama-server startup flag

These are deployment choices, not defects. Each is marked `xfail`/`skip` with the
flag named, never faked into a pass.

| # | Drift | Endpoint | Missing flag |
|---|---|---|---|
| core-2, graph-7 | `/metrics` returns **501** `not_supported_error`; `/props` reports `endpoint_metrics: false` | `GET /metrics` | `--metrics` |
| rag-5 | Native reranking returns **501** on the chat and embeddings servers | `POST /rerank`, `/v1/rerank` | `--reranking` **+ a cross-encoder GGUF**. Now served on a third server (see §1a); the two original servers must stay 501 |
| — | Embeddings return **501** on the chat server | `POST /v1/embeddings` @8090 | `--embeddings` (only 8081 has it) |
| core-5 | No speculative-decoding telemetry; draft sampler never engages | `/completion` timings | `--model-draft` |
| rag-17, rag-19 | No LoRA adapter slots | `GET/POST /lora-adapters` | `--lora` / `--lora-scaled` |
| rag-16 | Vision unsupported; `/props` reports `modalities.vision: false` | `/v1/chat/completions` with image parts | `--mmproj` + a vision GGUF (SmolLM2 is text-only, so the flag alone is not enough) |
| rag-12 | Cold-start 503/504 retry path unreachable | — | needs llama-swap or router `--sleep-idle-seconds`; a single-model server never unloads |
| graph-10 | Real concurrency caps at **3** regardless of `max_workers` | `/v1/chat/completions` | `--parallel` / `-np` (3 here) fixes the decode-slot count |
| rag-1 | Embeddings arrive **already L2-normalised** (norm 1.0, σ 2.9e-08), so the project's norm-vs-unnorm ablation is *degenerate* — it measures nothing | `POST /v1/embeddings` | `--embd-normalize 2` is the **default**; `--embd-normalize -1` would restore signal |
| core-8, graph-8 | `/slots` is served at all only because slots are enabled | `GET /slots` | `--slots` (`endpoint_slots: true`) |

**Two of these are silent, which makes them the dangerous ones.** `POST
/lora-adapters` returns `{"success": true}` for *every* body — id 99, id -5, a
string id, a missing id — while `GET` keeps returning `[]` (rag-17). A live test
asserting merely "apply_adapter didn't raise" would pass and look like proof that
hot-swapping works. And vision fails with **500 `server_error`**, not the
**501 `not_supported_error`** that missing embeddings and reranking give
(rag-16) — so switching on status code misclassifies it.

### 1a. `--reranking` cannot share a server with embeddings

Enabling `--reranking` forces the pooling type from `mean` to `rank`. Verified on
a scratch port: `/v1/rerank` then works, but `/v1/embeddings` on the same
process returns **garbage** — denormals and values like `151722240.0`,
`3.3e-41`. The two capabilities are mutually exclusive in one `llama-server`.

So the cross-encoder runs as a **third server**:

| | |
|---|---|
| Port | `127.0.0.1:8082` (`LLM_RERANK_BASE_URL`) |
| Model | `gpustack/bge-reranker-v2-m3-GGUF:Q8_0` |
| Flags | `--reranking --pooling rank` |

**`relevance_score` is a raw logit, not 0-1.** Hosted rerank APIs (Cohere,
Jina) normalise to `[0,1]`; llama.cpp does not. Measured: a matching document
scores `+7.86`, unrelated ones `-10.91` and `-11.04`. Code assuming a 0-1 range
silently misreads these. `usage` carries only `prompt_tokens`/`total_tokens` —
no `completion_tokens`.

Project 4.2 still reranks via chat completions; the native endpoint is now
covered separately by five live tests that skip when 8082 is not running.

---

## 2. Drift caused by endpoint shape: llama.cpp supersets OpenAI

The OpenAI-compatible layer emits strictly more than the OpenAI schema, and the
native endpoints follow no OpenAI convention at all. Every canned body in the
suite was written to the *narrow* OpenAI shape.

| # | Drift | Endpoint |
|---|---|---|
| core-20, rag-4, graph-1 | Real responses carry llama.cpp-only `timings` and `usage.prompt_tokens_details.cached_tokens`; `system_fingerprint` is the build id `b11046-60081bb2b`. Mocks had a 3-key stub | `/v1/chat/completions` |
| core-12 | Errors are a **nested object** `{"error":{"code":<int>,"message":str,"type":str}}`, not a flat string. Types seen: `invalid_request_error`, `not_supported_error`, `exceed_context_size_error` | all |
| rag-2 | Embedding envelope adds per-item `index` + `object`, top-level `model`/`object`/`usage` — and `usage` has **no** `completion_tokens` | `/v1/embeddings` |
| rag-11 | Real SSE **opens** with `content: null` (role chunk) and **closes** with an empty delta (stats chunk) | streaming `/v1/chat/completions` |
| core-3, core-4 | Streaming stop chunk is a **summary frame, not a text frame**: empty `content`, and naive parsers emit a phantom token from it | streaming `/completion` |
| core-6 | Logprob entries key the token as `token`, not `text` | `/completion` with `n_probs>0` |
| core-1 | `/slots` replaced flat `state` with boolean `is_processing`; decode progress moved into `next_token` | `GET /slots` |
| graph-8 | `/slots` is a **bare JSON array**, not `{"object":"list","data":[…]}` — it is native, not OpenAI-shaped. Its floats are float32-rounded (`0.949999988079071`), so exact `== 0.95` comparisons fail | `GET /slots` |
| rag-15 | Native `/completion` is verbose by design: `generation_settings`, `stop_type`, `tokens_cached`, `tokens_evaluated`, `truncated`, … | `POST /completion` |

---

## 3. Drift caused by "one process, one model"

A plain `llama-server` loads exactly one model via `-m`/`-hf` and serves every
request from it. Several mocks had invented a multi-model world.

| # | Drift |
|---|---|
| rag-10, graph-2 | **An unknown `model` value is silently accepted and served by the loaded model.** No 404, no 400. Any mocked test asserting an error for a bad model id was asserting fiction. |
| graph-4 | Mocks invented a multi-model OpenAI catalogue; `/v1/models` is a one-element list by construction. It also returns **both** an OpenAI `data` list **and** an Ollama-style `models` list in one payload. |
| core-16 | Unknown *sampler* keys are likewise accepted without error — the server does not validate request bodies strictly. |

---

## 4. Drift caused by the model, not the code

`SmolLM2-360M-Instruct` is a 360M model on CPU. Weak output is expected and was
never treated as a code defect; no project code was changed to chase output
quality, and live tests assert **structure only**, never semantics.

| # | Finding |
|---|---|
| core-19, graph-9 | **No tool calling.** `/props` reports `supports_tools: false` and `supports_tool_calls: false` — the model's bundled Jinja template has no tool section, so `tools=[…]` can never yield `tool_calls`. A model property, not a CPU or flag limit. |
| core-21, graph-3 | LLM-as-judge and `VerifyNode` never get usable JSON; parsers always take their regex fallback. The server *does* support `response_format`/GBNF — the model does not follow the instruction. |
| core-10 | Prompt-cache **speedup** is not observable at this size, though reuse *is* confirmed via `timings.cache_n > 0`. The test asserts reuse, not speed. |
| core-17 | SmolLM2's chat template **injects its own default system turn** when none is supplied, so a client-side ChatML renderer cannot match the server byte-for-byte without knowing it. |
| rag-6 | `finish_reason` is **not reproducible** without `ignore_eos` — default sampling (`temp 0.8`, `top_k 40`, `top_p 0.95`) sometimes emits EOS early. This bit us for real: see §6. |
| rag-9 | Real embeddings are anisotropic and crowded; canned vectors were conveniently orthogonal. |
| core-13/14/15, core-18, rag-8 | `tfs_z` was **removed from llama.cpp upstream** yet is still silently accepted; `mirostat` is parsed but absent from the active sampler chain; `typ_p` (chain) vs `typical_p` (request field) is a naming quirk, not a defect; `chars//4` over-estimates real token counts by 20-26%. |

Active sampler chain per `/props`: `penalties, dry, top_n_sigma, top_k, typ_p,
top_p, min_p, xtc, temperature`.

---

## 5. Real bugs found and fixed

Twelve genuine defects, every one invisible to the offline suite. These are the
return on running against a real server.

| Project | Bug | Why the mock never caught it |
|---|---|---|
| **2.2** grammar-structured-output | **`json_schema` was sent as a JSON *string***; the server rejects it with `400 "schema must be an object"`. The project's core feature was **totally broken** against any real server. | No mocked test ever inspected the *request body* — only responses were asserted. |
| **6.4** enterprise-mcp-gateway | **Auth bypass.** `if self.config.require_auth and token:` meant `require_auth=True` with the default empty token skipped authentication entirely — an *unauthenticated* caller passed while a *bad-token* caller was rejected. | Every offline test set `require_auth=False`, except one passing an invalid token. Pure policy logic; no live call would surface it either. |
| **4.3** prompt-cache-chunking | `cache_hit_ratio` was **structurally always 0.0** — a missing `seen_prefixes.add(...)` plus a cache key that wrongly included the query. | Canned data never exercised prefix reuse. |
| **1.3** constrained-decoding | `SchemaConverter` emitted GBNF llama.cpp **cannot parse** (4 distinct defects, e.g. rule names must match `[a-zA-Z0-9-]`), and `GBNFParser.validate` accepted grammars the server rejects — it only counted `(` vs `)`. | Nothing ever fed the grammar to a real parser. |
| **1.4** gguf-quant-explorer | Attention/vocab keys **silently fell back to llama-2 defaults** (32/8/32000) for *every* model. | Assertions matched the same wrong defaults. |
| **1.2** spec-decoding-benchmark | Streaming `/completion` parsing broke on the real stop frame. | The canned stream had no summary frame. |
| **1.1/1.2** | `/slots` parsing used the removed `state` field; `1.2` also shipped an **invalid `pyproject.toml`** so its tests never ran at all. | Tests were never collected. |
| **1.5** sampling-param-explorer | `softmax` returned **NaN** when every logit was masked — its documented fallback was unreachable because `NaN <= 0` is `False`. Pre-existing ~1-in-3 flake. | Intermittent; masked by luck. |
| **4.2** local-rerank-rag | `_parse_score` **crashed on a bare JSON number**. | No mock ever produced a bare JSON scalar; only a real model did. |
| **6.3** model-discovery-router | `context_length` was **never populated** — `from_openai_response()` didn't map it, so it was permanently 0. Also, `server._registry` **leaked across tests**. | Mocks asserted the same empty value. |
| **2.4** lora-hotswap | `list_adapters` had the wrong return shape (`GET /lora-adapters` returns an **array**, not an object). | Mock returned the invented object shape. |

---

## 6. One bug this process introduced, and caught

`5.5-conditional-branching`'s live test assumed `max_tokens=8` always yields
`finish_reason == "length"`. SmolLM2 sometimes emits EOS inside 8 tokens and
returns `"stop"`, so the test passed in the worker's own run and **failed on
merged `main`**. Fixed by passing `ignore_eos: true`, making the token cap the
only stop condition — deterministic by construction rather than by luck.
Verified 6/6 consecutive runs.

Notably the `rag` worker had already recorded this exact non-reproducibility
(rag-6) while `graph` hit it independently. Cross-reading the drift logs would
have caught it before the merge.

---

## 7. Environment notes

- **`7-bonus/inference-profiling-dashboard` defaults `DASHBOARD_PORT` to 8090**,
  colliding with the real chat server. Set `DASHBOARD_PORT` before running it.
- **Project defaults still point at 8080**, which is Open WebUI here. Left
  unchanged deliberately — upstream llama.cpp really does default to 8080, and
  `env.sh` plus the fixture guard already cover tests and demos. Anyone running a
  project *without* sourcing `env.sh` will hit Open WebUI.
- **`mcp` is pinned `<2`**: the 6-mcp projects import `mcp.server.fastmcp`, which
  mcp 2.x renamed to `MCPServer`.
- **`langgraph` is never imported** despite the directory name; those projects
  hand-roll their graphs.
- Each project ships its own `pyproject.toml`, so **a repo-root `conftest.py` is
  cut off** by pytest's rootdir detection. The shared fixtures load via
  `PYTEST_PLUGINS=_shared.live_fixtures` instead.
- `env.sh` derives its root from `BASH_SOURCE`, so a git worktree imports its own
  `_shared/` rather than the main checkout's.

## 8. Behaviour changes to be aware of

The 6.4 auth fix is a **real behaviour change**: the default `GatewayConfig`
pairs `require_auth=True` with `AllowAllAuthProvider`, so a tokenless call
against the bare default is now **denied**. That is the correct posture —
`AllowAllAuthProvider` governs *which* token is acceptable, not *whether* one is
required — but any caller relying on `EnterpriseGateway()` with no config will
see previously-passing requests rejected.

## 9. Reproducing

```bash
source env.sh                                   # 8090 chat, 8081 embed
python scripts/run_all_tests.py --mode offline  # 808 tests, no server needed
python scripts/run_all_tests.py --mode live     # 290 tests against real servers
python scripts/run_all_tests.py --mode both
```
