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
