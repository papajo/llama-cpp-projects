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
