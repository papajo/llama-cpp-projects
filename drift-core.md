# Drift log — worker `core` (1-ai-fundamentals, 3-prompt-engineering)

Real servers used:
- chat  `http://127.0.0.1:8090` — SmolLM2-360M-Instruct Q8_0, n_ctx 2048, 3 slots
- embed `http://127.0.0.1:8081` — nomic-embed-text-v1.5 Q8_0, n_embd 768
- llama.cpp build `b11046-60081bb2b`

---

## 1. `/slots` reports `is_processing`, not `state`

- **Project / test**: `1-ai-fundamentals/1.1-kv-cache-visualizer` —
  `tests/test_live_probe.py::test_get_slots_parses_real_payload`
- **What the mock asserted**: nothing — `server-probe/probe.py` had *no* test
  coverage at all, so `SlotInfo.from_dict` was never run against a payload.
  It read `d["state"]` (a string, `"idle"`/`"processing"`), `d["n_decoded"]`,
  `d["n_past"]` and `d["prompt"]`.
- **What the real server returned**: slot objects with keys
  `id, id_task, is_processing, n_ctx, n_prompt_tokens, n_prompt_tokens_cache,
  n_prompt_tokens_processed, next_token, speculative`.
  There is no `state`, no `n_past`, no `n_predicts` and no `prompt`.
  `n_decoded` is nested at `next_token[0].n_decoded`.
- **Cause**: llama-server's `/slots` schema changed — flat `state` was replaced
  by the boolean `is_processing`, and per-token decode progress moved into the
  `next_token` array. Endpoint is only served at all because the server runs
  without `--no-slots`.
- **Consequence**: every slot parsed as `state="unknown"`, so
  `get_active_slots()` and `get_idle_slots()` both returned `[]` unconditionally,
  and `report()["active_slots"]`/`["idle_slots"]` were always `0`.
- **Fixed**: `SlotInfo.from_dict` now derives `state` from `is_processing`,
  reads `n_decoded` out of `next_token[0]`, and falls back to
  `n_prompt_tokens_cache` for `n_past`. Legacy flat keys still win if present.

## 2. `/metrics` is not compiled into this deployment

- **Project / test**: `1-ai-fundamentals/1.1-kv-cache-visualizer` —
  `tests/test_live_probe.py::test_metrics_endpoint_is_not_compiled_in`
- **What the mock asserted**: nothing; `ServerMetrics.from_prometheus` was
  untested.
- **What the real server returned**: `HTTP 501` with
  `{"error":{"code":501,"message":"This server does not support metrics
  endpoint. Start it with `--metrics`","type":"not_supported_error"}}`
- **Cause**: llama-server flag `--metrics` was not passed at startup.
- **Classification**: unsupported in this deployment. The Prometheus parser is
  left untested rather than faked; the live test asserts the honest 501 and
  that `report()` degrades to zeroed metrics.

## 3. Streaming `/completion`: stop chunk has empty `content`

- **Project / test**: `1-ai-fundamentals/1.2-spec-decoding-benchmark` —
  `tests/test_live_spec_decoding.py::test_full_text_is_the_accumulated_stream`
- **What the mock asserted**: nothing — 1.2 shipped no tests at all.
  `BenchmarkRunner.send_completion` set
  `result["full_text"] = data.get("content", "")` on the chunk where
  `stop == True`.
- **What the real server returned**: the terminating SSE chunk carries the
  stop metadata (`stop, stop_type, timings, tokens_cached, truncated,
  generation_settings, …`) with `content` set to the empty string. The
  generated text only ever exists as the concatenation of the preceding
  chunks' `content` fields.
- **Cause**: llama-server's streaming protocol on `/completion` — the final
  frame is a summary frame, not a text frame.
- **Consequence**: `full_text` was `""` for every run.
- **Fixed**: `full_text` is now `"".join(result["tokens"])`.

## 4. Streaming `/completion`: phantom token from the stop chunk

- **Project / test**: `1-ai-fundamentals/1.2-spec-decoding-benchmark` —
  `tests/test_live_spec_decoding.py::test_token_count_matches_server_accounting`
- **What the mock asserted**: nothing (no tests shipped).
  The token loop tested `if "content" in data:` — key presence.
- **What the real server returned**: the stop chunk *has* a `content` key
  whose value is `""`. Measured: 11 chunks carrying a `content` key, 10 with
  non-empty content, and the server's own `timings.predicted_n == 10`.
- **Cause**: same summary-frame protocol as above.
- **Consequence**: `total_tokens` was inflated by exactly one on every
  request, and one bogus entry was appended to `tokens`/`timestamps`. This
  skews every tokens-per-second figure the benchmark reports.
- **Fixed**: the loop now tests `if data.get("content"):` (truthiness).

## 5. No speculative-decoding telemetry on this server

- **Project / test**: `1-ai-fundamentals/1.2-spec-decoding-benchmark` —
  `test_draft_acceptance_telemetry_absent_without_draft_model`,
  `test_server_reports_no_speculative_slot`, and a skipped
  `test_speculative_strategy_end_to_end`.
- **What the mock asserted**: nothing (no tests shipped). `send_completion`
  looks for `draft_accepted` / `draft_total` keys in the stream.
- **What the real server returned**: neither key ever appears. `/slots`
  reports `"speculative": false` for all 3 slots and `/props` reports
  `"speculative.types": "none"`.
- **Cause**: llama-server was started without `--model-draft`, so no draft
  model is loaded and the speculative sampler is never engaged.
- **Classification**: unsupported in this deployment. The end-to-end
  speculative strategy test is skipped with that reason rather than faked;
  the counters are asserted to be zero, which is the honest result.

## 6. `completion_probabilities` entries key the token as `token`, not `text`

- **Project / test**: `1-ai-fundamentals/1.3-constrained-decoding-playground` —
  `tests/test_grammar.py::test_trace_analysis` (mock) and
  `tests/test_live_grammar.py::test_analyse_real_constrained_completion` (live)
- **What the mock asserted**: the canned `completion_probabilities` list used
  `{"text": '{"', "id": 1, "logprob": -0.1, "top_logprobs": [{"id":…,
  "logprob":…}]}` — a `text` key for the chosen token, and candidate entries
  carrying only `id` and `logprob`. The test asserted only `total_tokens` and
  `full_text`, so it never noticed which token string came back.
- **What the real server returned** (`POST /completion` with `n_probs`):
  `{"id": 32, "token": "0", "bytes": [48], "logprob": -4.70,
  "top_logprobs": [{"id": 216, "token": " ", "bytes": [32], "logprob": -0.13}, …]}`
  — the field is `token`, there is no `text`, and every entry (chosen and
  candidate) additionally carries `bytes`.
- **Cause**: `/completion` with `n_probs > 0`; llama-server aligned these
  entries with the OpenAI logprobs schema, which names the field `token`.
- **Consequence**: `GrammarDebugger.analyse_completion` read
  `prob_entry.get("text", "")`, so `GrammarMaskStep.token_text` was `""` for
  every token of every real trace. The masking heuristic (which matches on
  `id`) still worked, but the rendered trace was blank.
- **Fixed**: read `token` first, falling back to `text` for older payloads.
  Mock data corrected to the real shape and the offline test now asserts
  `token_text`.

## 7. SchemaConverter emitted GBNF that llama.cpp cannot parse

- **Project / test**: `1-ai-fundamentals/1.3-constrained-decoding-playground` —
  `tests/test_live_grammar.py::test_json_schema_grammar_round_trip`
- **What the mock asserted**: `tests/test_grammar.py` only checked that the
  generated text *contained* substrings — `assert "root ::=" in gbnf`,
  `assert "location" in gbnf or "prop_location" in gbnf`. It never fed the
  grammar to a parser, so four independent syntax defects went unnoticed.
- **What the real server returned**: `HTTP 400`
  `{"error":{"code":400,"message":"Failed to initialize samplers: failed to
  parse grammar","type":"invalid_request_error"}}` for every object schema.
- **Cause**: llama.cpp's GBNF parser (`/completion` `grammar` field). Four
  separate defects, each confirmed in isolation against the server:
  1. **Rule names may only contain `[a-zA-Z0-9-]`.** The converter generated
     `prop_n`, `<rule>_item`, `<rule>_opt0` and `<base>_<counter>`.
     Verified: `prop-n ::= "1"` parses, `prop_n ::= "1"` is a 400.
  2. **`" "}"` is not a literal.** Object/array rules ended with
     `' " "}"'`, which lexes as the literal `" "` then a bare `}` and a
     dangling quote. Needs `" " "}"` (likewise `" " "]"`).
  3. **The default string rule was escaped one level too deep** —
     `"\\"" ( [^"\\\\] | "\\\\" . )* "\\""` instead of
     `"\"" ( [^"\\] | "\\" . )* "\""`.
  4. **Enum alternatives lost their closing quote** — produced
     `( "\"r\" | "\"g\" )` instead of `( "\"r\"" | "\"g\"" )`.
  Also fixed alongside: the `email` pattern emitted `"+"` as a *literal plus*
  (`[a-zA-Z0-9._%+-] "+" "@" ...`) rather than the repetition operator.
- **Consequence**: JSON-Schema-constrained decoding — the project's headline
  feature — was non-functional against any real llama-server.
- **Fixed**: names sanitised to the legal subset at the single choke point
  (`_fresh_name`), literals separated, string/enum/email rules corrected.
  The live test now round-trips five schema shapes (boolean, enum, string
  with an underscored key, array, two-property) and parses the output as JSON.
- **Note (model too small, not a bug)**: an *unbounded* integer or the email
  char-class lets SmolLM2-360M emit digits until `n_predict` runs out. The
  grammar is accepted and correct; the model is simply too weak to stop. The
  live tests therefore use bounded schemas only.

## 8. `GBNFParser.validate` accepted grammars the server rejects

- **Project / test**: `1-ai-fundamentals/1.3-constrained-decoding-playground` —
  `tests/test_live_grammar.py::test_invalid_grammar_is_rejected_by_server`
- **What the mock asserted**: the offline tests only exercised balanced
  parentheses.
- **What the real server returned**: `HTTP 400 ... failed to parse grammar`
  for `root ::= [0-9`, while `validate()` returned `[]` ("valid").
- **Cause**: `validate()` compared only `count("(")` vs `count(")")`. It
  ignored character-class brackets and string literals entirely — so it both
  missed unterminated `[`/`"` *and* would have mis-flagged a grammar whose
  literals contain a delimiter, e.g. `root ::= "(" "[" "]" ")"`.
- **Fixed**: the checker now walks the definition through three lexical
  contexts (bare / string literal / character class) honouring backslash
  escapes, and reports unterminated literals, unterminated classes and stray
  closers. `root ::= "(" "[" "]" ")"` still validates clean.

## 9. GGUF attention + vocab keys silently fell back to llama-2 defaults

- **Project / test**: `1-ai-fundamentals/1.4-gguf-quant-explorer` —
  `tests/test_live_gguf.py::test_derived_dims_match_live_server` and
  `::test_attention_dims_are_read_not_defaulted`
- **What the mock asserted**: the offline suite parses synthetic GGUF byte
  streams it builds itself, using flat keys like `llama.head_count`. Those
  streams are internally consistent, so the fallback table was never hit and
  the tests passed.
- **What the real files / server returned**: in a real GGUF the attention
  hyper-parameters are namespaced under `<arch>.attention.*`, and there is no
  `vocab_size` key at all — the authoritative vocab is
  `len(tokenizer.ggml.tokens)`. Measured:

  | field | parser said | real file | live server |
  |---|---|---|---|
  | `head_count` (nomic-bert) | 32 | 12 (`nomic-bert.attention.head_count`) | — |
  | `head_count_kv` (nomic-bert) | 8 | absent → MHA, = 12 | — |
  | `vocab_size` (nomic) | 32000 | 30522 (`len(tokenizer.ggml.tokens)`) | `n_vocab` = 30522 |
  | `head_count` / `_kv` (qwen2) | 32 / 8 | 12 / 2 | — |
  | `vocab_size` (qwen2) | 32000 | 151936 | — |

- **Cause**: not a llama-server flag — this is the GGUF metadata layout
  itself. `_derive_model_info` searched the prefix chain
  `["<arch>.", "llama.", ""]` for a bare `head_count` / `vocab_size`, missed,
  and silently returned `_KNOWN_SIZES` (the llama-2 defaults: 32 heads,
  8 KV heads, 32000 vocab). Every value looked plausible, which is why it
  went unnoticed.
- **Consequence**: `head_count`, `head_count_kv` and `vocab_size` were the
  same constants for *every* model. `is_gqa` was therefore always `True`
  (nomic-bert is actually MHA), and the parameter-count and KV-cache
  estimates — the whole point of the tool — were computed from wrong dims.
- **Fixed**: the lookup now also tries `attention.<key>`; a model with no
  explicit KV head count is treated as MHA (`head_count_kv = head_count`)
  instead of taking the generic default; and `vocab_size` is taken from the
  tokenizer token list when present. Verified against the live server:
  nomic `n_embd` 768 / `n_vocab` 30522 now match exactly.

## 10. Cache speedup is not observable at this model size (not a defect)

- **Project / test**: `1-ai-fundamentals/1.6-prompt-cache-benchmark` —
  `tests/test_live_cache.py::test_speedup_is_reported_but_not_asserted_to_be_favourable`
- **What the mock asserted**: `test_cache_result_compute` feeds hand-picked
  TTFTs (cached 50/60/70 ms vs uncached 200/220/240 ms) and asserts
  `speedup_factor ≈ 3.67`. That is arithmetic on invented numbers, not a
  measurement.
- **What the real server returned**: prompt caching demonstrably *works* —
  `timings.cache_n > 0` and `tokens_cached > 0` on a repeated prefix — but
  with SmolLM2-360M and a ~40-token prefix the prompt-eval phase is already
  sub-millisecond, so the cached arm is not reliably faster end to end.
  `speedup_factor` fluctuates around 1.0 in either direction.
- **Cause**: model size and prefix length, not a flag. The KV-reuse
  machinery is confirmed active via `timings.cache_n`.
- **Classification**: model too small. The live test asserts cache *reuse*
  (`cache_n > 0`, the thing the code actually controls) and asserts only
  that `speedup_factor` is finite and positive. No project code was changed
  to chase a favourable ratio.

## 11. `softmax` returned NaN when every logit was masked (pre-existing flake)

- **Project / test**: `1-ai-fundamentals/1.5-sampling-param-explorer` —
  `tests/test_sampler.py::test_full_pipeline` (offline, was intermittently
  failing) and `tests/test_live_sampler.py::test_aggressive_truncation_never_yields_nan`
- **What the mock asserted**: `assert result["summary"]["final_entropy"] >= 0`
  against the randomly generated "flat" distribution preset. It failed roughly
  one run in three with `assert nan >= 0`.
- **What was actually happening**: not server drift — a latent numerical bug,
  surfaced here because the preset's generator is random. `apply_typical`
  could zero every candidate; the pipeline then rebuilt logits as all `-inf`;
  `softmax` computed `max_l = -inf` and `l - max_l` = `inf - inf` = `NaN`.
  Its own documented fallback (`# All logits are -inf: return uniform`) was
  unreachable, because the guard is `if total <= 0` and `NaN <= 0` is `False`.
- **Cause**: no llama-server flag involved; pure arithmetic.
- **Fixed**: `softmax` now checks `math.isfinite(max_l)` *before* subtracting
  and returns the uniform distribution, which is what the existing comment
  always intended. Separately, `apply_typical` now keeps the most likely
  token when its threshold would empty the candidate set — real samplers
  never return an empty set. Offline test now passes 8/8 consecutive runs.

## 12. Error bodies are a nested object, not a flat string

- **Project / test**: `3-prompt-engineering/3.1-reasoning-budget-sweep` —
  `tests/test_sweep.py::test_http_error_captured` (mock) and
  `tests/test_live_sweep.py::test_real_400_is_captured_as_a_result_not_an_exception`
- **What the mock asserted**: the canned 400 body was
  `{"error": "bad request"}` — `error` mapped to a plain string — and the
  test only asserted `"HTTP 400" in result.error`.
- **What the real server returned**: a nested object, on every 4xx:
  `{"error":{"code":400,"message":"Field 'temperature': [json.exception.type_error.302] type must be number, but is string","type":"invalid_request_error"}}`
  Context overflow uses the same envelope with extra fields and a distinct
  type: `{"error":{"code":400,"message":"request (6031 tokens) exceeds the
  available context size (2048 tokens), try increasing it",
  "type":"exceed_context_size_error","n_prompt_tokens":6031,"n_ctx":2048}}`
- **Cause**: llama-server's standard OpenAI-compatible error envelope on
  `/v1/chat/completions`. The `exceed_context_size_error` variant is
  produced because the server was started with `n_ctx` 2048
  (`n_ctx_train` is 8192, so this is a launch flag, not a model limit).
- **Consequence**: no project bug — `run_sweep` only interpolates
  `response.text` into `SweepResult.error`, so both shapes format fine. But
  the mock could not express an assertion on the error *type*, so nothing
  offline pinned the contract.
- **Fixed**: canned body corrected to the real nested shape and the offline
  test now asserts `"invalid_request_error"` appears. The live tests assert
  both the plain 400 and the `exceed_context_size_error` path.

## 13. `tfs_z` no longer exists in llama.cpp, but is accepted silently

- **Project / test**: `3-prompt-engineering/3.2-sampler-ablation-lab` —
  `tests/test_live_ablation.py::test_tfs_z_is_not_in_this_builds_sampler_chain`
- **What the mock asserted**: `test_config.py` only checks that
  `SamplerConfig.to_request_body()` *emits* `tfs_z`, and `test_runner.py`
  never sends it. Nothing asserted the server does anything with it.
- **What the real server returned**: `HTTP 200`, normal output. But
  `/props` shows the active sampler chain is
  `["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p",
  "xtc", "temperature"]` — no `tfs_z` — and `tfs_z` is not even present
  among `default_generation_settings.params`.
- **Cause**: tail-free sampling was removed from llama.cpp upstream. The
  server ignores unrecognised body keys rather than rejecting them.
- **Classification**: unsupported in this build. No code changed — the
  request is well-formed and the project is right to be able to emit it.
  The live test pins the fact that ablating `tfs_z` measures nothing here,
  so a future "tfs_z had no effect" result is not read as a finding.

## 14. `mirostat` is a recognised param but not in the active sampler chain

- **Project / test**: `3-prompt-engineering/3.2-sampler-ablation-lab` —
  `tests/test_live_ablation.py::test_mirostat_is_accepted_but_absent_from_the_sampler_chain`
- **What the mock asserted**: `test_config.py` asserts `cfg.mirostat == 1`
  / `== 2` for the two presets — a dataclass field check with no server
  involvement.
- **What the real server returned**: `HTTP 200` with normal output for both
  `preset_mirostat_v1()` and `preset_mirostat_v2()`. `/props` still lists
  `mirostat: 0`, `mirostat_tau: 5.0`, `mirostat_eta: 0.1` among the default
  params (so the field is recognised), but `"mirostat"` does **not** appear
  in the `samplers` chain.
- **Cause**: llama.cpp keeps the mirostat request fields but the chain
  reported by `/props` is the standard truncation chain. Nothing in this
  deployment demonstrates mirostat engaging.
- **Classification**: unsupported in this build. Not faked as a pass — the
  live test asserts only that the presets are accepted and produce output,
  and records that the mirostat arm of an ablation is not measurable here.

## 15. `typ_p` (chain name) vs `typical_p` (request field) — not a defect

- **Project / test**: `3-prompt-engineering/3.2-sampler-ablation-lab` —
  `tests/test_live_ablation.py::test_config_uses_typical_p_the_name_the_server_accepts`
- **What the real server returned**: `/props` names the sampler `typ_p` in
  the chain, while the request-side parameter and the default-params block
  both use `typical_p`.
- **Cause**: llama.cpp uses the short name internally for the chain listing
  and the long name on the API surface.
- **Classification**: no defect. The project emits `typical_p`, which is the
  correct request field. Recorded only so the two names are not mistaken
  for drift during the merge.

## 16. Unknown sampler keys are accepted without error

- **Project / test**: `3-prompt-engineering/3.2-sampler-ablation-lab` —
  `tests/test_live_ablation.py::test_unknown_sampler_params_are_silently_ignored`
- **What the real server returned**: `HTTP 200` for a body containing
  `"definitely_not_a_sampler": 1.23`.
- **Cause**: llama-server ignores unrecognised keys on
  `/v1/chat/completions` rather than validating the body strictly. (Matches
  the conductor's finding that an unknown *model* id is also silently
  accepted and served by the loaded model.)
- **Classification**: server behaviour, no code change. Pinned because it is
  the failure mode behind entries 13 and 14: a misspelled or removed
  sampler name in an ablation grid yields a clean run that measured nothing.

## 17. SmolLM2's chat template injects its own default system turn

- **Project / test**: `3-prompt-engineering/3.3-chat-template-tester` —
  `tests/test_live_templates.py::test_server_injects_a_default_system_turn_when_none_is_given`
- **What the mock asserted**: nothing about the server — 3.3 is pure
  client-side string formatting and never opens a socket, so the offline
  suite (68 tests) has no server contract at all.
- **What the real server returned**: `POST /apply-template` with a single
  user turn renders
  `<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n…`
  — a system block the caller never supplied. Supplying *any* system
  message suppresses it.
- **Cause**: the model's own Jinja `chat_template` (visible in `/props`)
  begins
  `{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\n…' }}{% endif %}`.
  Not a llama-server flag — it ships inside the GGUF.
- **Classification**: no defect. A client-side ChatML renderer cannot know
  a given model's default system string. The project's `chatml` render is
  byte-identical to `/apply-template` whenever a system message is
  supplied; the live tests pin both that equality and this one exception.

## 18. `estimate_tokens()` is a heuristic, not a tokenizer

- **Project / test**: `3-prompt-engineering/3.3-chat-template-tester` —
  `tests/test_live_templates.py::test_token_estimate_is_only_an_estimate`
- **What the mock asserted**: the offline tests treat `estimated_tokens`
  as a plain integer property.
- **What the real server returned**: `POST /tokenize` gives the true count
  from the model's vocabulary; `estimate_tokens()` is `len(text) // 4` and
  does not equal it.
- **Classification**: no defect — it is documented as an estimate. Pinned
  with an order-of-magnitude bound (within 4x either way) so the figure is
  never mistaken for a measurement, and so a future tokenizer change that
  blows past that band is caught.

## 19. Tool calling unsupported by this model's chat template

- **Project / test**: `3-prompt-engineering/3.3-chat-template-tester` —
  `tests/test_live_templates.py::test_tool_calling_is_unsupported_by_this_chat_template`
- **What the real server returned**: `/props` `chat_template_caps` reports
  `supports_tools: false` and `supports_tool_calls: false` (also
  `supports_parallel_tool_calls: false`, `supports_typed_content: false`).
  `supports_system_role: true`, which the byte-equality tests depend on.
- **Cause**: SmolLM2's chat template has no tool/function-calling section.
  Not a CPU or build limitation — a property of the model's template.
- **Classification**: unsupported by this model. No tool-calling assertions
  were written; the capability flags are asserted directly so the gap is
  explicit rather than silently skipped.

## 20. OpenAI-shaped mocks omit most of the real response envelope

- **Project / test**: all four HTTP projects in `3-prompt-engineering`
  (3.1, 3.2, 3.4, 3.5). Pinned by
  `3.4-prompt-chaining-workbench/tests/test_live_chain.py::test_real_usage_block_is_richer_than_the_canned_one`
- **What the mocks assert**: every canned success body across these
  projects is the same two-key minimum —
  `{"choices":[{"message":{"content": "..."}}],
    "usage":{"prompt_tokens":N,"completion_tokens":M}}`
- **What the real server returns**: that plus
  `id`, `model`, `created`, `object: "chat.completion"`,
  `system_fingerprint` (`"b11046-60081bb2b"`), and a llama.cpp-only
  `timings` block (`prompt_n`, `prompt_ms`, `predicted_n`,
  `predicted_per_second`, `cache_n`, …). Inside `choices[0]` there is also
  `finish_reason` (`"stop"` or `"length"`), `index`, and `message.role`.
  `usage` additionally carries `total_tokens` and
  `prompt_tokens_details.cached_tokens`.
- **Cause**: llama-server's OpenAI-compatible layer supersets the OpenAI
  schema. `timings` is llama.cpp-specific; `cached_tokens` reflects the
  prompt-cache reuse also seen in entry 10.
- **Consequence**: no project bug — 3.1/3.2/3.4/3.5 all read only
  `choices[0].message.content` and the two `usage` counts, which the thin
  mocks do supply. But nothing offline pinned the real contract, so a
  project that later started reading `finish_reason` (to distinguish a
  truncated completion from a finished one) would pass its mocked tests
  while reading `None` in production.
- **Classification**: mock drift, recorded rather than "fixed" — widening
  every canned body would add noise without changing behaviour. The live
  test asserts the full envelope once, in 3.4, on behalf of all four.

## 21. LLM-as-judge: SmolLM2 cannot produce the rubric's JSON

- **Project / test**: `3-prompt-engineering/3.5-prompt-optimizer-loop` —
  `tests/test_live_optimizer.py::test_rubric_parse_fallback_is_exercised_by_this_model`
- **What the mock asserted**: the offline tests patch `urlopen` to return a
  judge reply of exactly `{"score": 8, "rationale": "..."}`, so
  `RubricEvaluator` always takes its strict `json.loads` path and yields a
  clean 0.8. Every offline rubric assertion is really an assertion about
  that canned string.
- **What the real server returned**: asked to rate `"The capital of France
  is Paris."` against expected `"Paris"` and to reply with ONLY a JSON
  object, SmolLM2-360M replied with the bare word `Paris`. The strict parse
  fails, the `"score"\s*:\s*(\d+)` regex fails, and the neutral fallback
  fires — `(5, "Could not parse judge output")`, i.e. a flat 0.5.
- **Cause**: model capability. Not a flag, not an endpoint, not a build
  option — a 360M instruct model does not reliably follow a
  "respond with only JSON" instruction. (Note: the server *does* support
  `response_format: json_schema` and GBNF `grammar`, either of which would
  force valid JSON — but using them would be a redesign of the evaluator,
  not a test fix, so nothing was changed.)
- **Classification**: model too small. No project code was changed to chase
  a better score. The live tests assert only the contract that holds
  regardless of which parse tier fires: the result is a `Score` with
  `scorer == "rubric"` and a normalised value in [0,1]. Judge *accuracy*
  is never asserted, and neither is score improvement across iterations.
- **Knock-on**: `OptimizationLoop` is therefore tested for termination and
  history well-formedness only. `test_loop_stops_at_max_iterations_without_improving`
  deliberately encodes the realistic path — the meta-prompt rewrite does
  not help, the target is never reached, and the loop exits on
  `max_iterations` rather than on success.
