"""Live integration tests for the sampler ablation lab.

The offline suite drives run_ablation through pytest-httpx with canned
bodies. These run the same code path against the real chat server, and
additionally check which sampler knobs this llama.cpp build actually
honours - the lab's whole premise is that toggling a knob changes
something.

max_tokens is forced small on every config: the presets default to 512,
which is far too slow on a CPU-only 360M model.

Assertions are structural. SmolLM2-360M output quality is never asserted.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live
"""

from __future__ import annotations

import dataclasses
import json
import urllib.request

import pytest

from sampler_ablation.comparator import format_comparison
from sampler_ablation.config import (
    SamplerConfig,
    ablate,
    preset_balanced,
    preset_creative,
    preset_greedy,
    preset_mirostat_v1,
    preset_mirostat_v2,
    preset_precise,
)
from sampler_ablation.runner import run_ablation

PROMPT = "Name one colour."


def _small(cfg: SamplerConfig, max_tokens: int = 16) -> SamplerConfig:
    """Same config, but with a CPU-friendly token budget."""
    return dataclasses.replace(cfg, max_tokens=max_tokens)


@pytest.fixture(scope="module")
def server_samplers(chat_base_url):
    """The sampler chain this build actually runs, from /props."""
    with urllib.request.urlopen(f"{chat_base_url}/props", timeout=10) as r:
        props = json.loads(r.read().decode())
    return props["default_generation_settings"]["params"]["samplers"]


@pytest.mark.live
def test_single_config_against_real_server(chat_base_url):
    results = run_ablation(
        configs=[_small(preset_greedy())],
        prompt=PROMPT,
        base_url=chat_base_url,
        max_retries=0,
    )
    assert len(results) == 1
    r = results[0]
    assert r.is_ok, r.error
    assert r.output
    assert r.label == "greedy"
    assert r.prompt_tokens > 0
    assert r.completion_tokens > 0
    assert r.elapsed_s > 0


@pytest.mark.live
def test_multiple_configs_keep_order_and_labels(chat_base_url):
    configs = [_small(preset_greedy()), _small(preset_precise())]
    results = run_ablation(
        configs=configs, prompt=PROMPT, base_url=chat_base_url, max_retries=0
    )
    assert [r.label for r in results] == ["greedy", "precise"]
    assert all(r.is_ok for r in results), [r.error for r in results]


@pytest.mark.live
def test_greedy_is_deterministic_on_the_real_model(chat_base_url):
    """temperature=0 + top_k=1 + fixed seed must reproduce exactly.

    This is a structural property of the sampler, not of output quality,
    so it is safe to assert even on a 360M model.
    """
    cfg = _small(preset_greedy())
    first = run_ablation([cfg], PROMPT, base_url=chat_base_url, max_retries=0)[0]
    second = run_ablation([cfg], PROMPT, base_url=chat_base_url, max_retries=0)[0]
    assert first.is_ok and second.is_ok
    assert first.output == second.output


@pytest.mark.live
def test_max_tokens_binds_per_config(chat_base_url):
    """Each config's own max_tokens caps its own completion."""
    results = run_ablation(
        configs=[_small(preset_balanced(), 8), _small(preset_balanced(), 24)],
        prompt="Write a long essay about the sea.",
        base_url=chat_base_url,
        max_retries=0,
    )
    assert all(r.is_ok for r in results), [r.error for r in results]
    assert results[0].completion_tokens <= 8
    assert results[1].completion_tokens <= 24


@pytest.mark.live
def test_sampler_params_are_accepted_by_the_real_server(chat_base_url):
    """Every preset's request body is accepted without a 4xx."""
    configs = [
        _small(p())
        for p in (
            preset_greedy,
            preset_creative,
            preset_balanced,
            preset_precise,
        )
    ]
    results = run_ablation(
        configs=configs, prompt=PROMPT, base_url=chat_base_url, max_retries=0
    )
    for r in results:
        assert r.is_ok, f"{r.label} rejected: {r.error}"


@pytest.mark.live
def test_ablation_sweep_runs_end_to_end(chat_base_url):
    """ablate() variants all execute; the report renders over real output."""
    configs = [_small(c) for c in ablate(preset_balanced())][:4]
    results = run_ablation(
        configs=configs, prompt=PROMPT, base_url=chat_base_url, max_retries=0
    )
    assert len(results) == len(configs)
    assert all(r.is_ok for r in results), [r.error for r in results]

    report = format_comparison(results, reference_label=results[0].label)
    assert report.startswith("# ")
    assert PROMPT in report
    for r in results:
        assert r.label in report


# ── Unsupported samplers ────────────────────────────────────────────


@pytest.mark.live
def test_tfs_z_is_not_in_this_builds_sampler_chain(server_samplers, chat_base_url):
    """tfs_z (tail-free sampling) no longer exists in llama.cpp.

    It is not in the sampler chain and not even among the server's default
    generation params, yet a request carrying it is accepted silently
    rather than rejected. Ablating it therefore cannot change anything.
    """
    assert "tfs_z" not in server_samplers
    with urllib.request.urlopen(f"{chat_base_url}/props", timeout=10) as r:
        params = json.loads(r.read().decode())["default_generation_settings"]["params"]
    assert "tfs_z" not in params

    cfg = _small(SamplerConfig(tfs_z=0.5, label="tfs"))
    assert cfg.to_request_body()["tfs_z"] == 0.5
    result = run_ablation([cfg], PROMPT, base_url=chat_base_url, max_retries=0)[0]
    # Accepted, not rejected - which is precisely the trap.
    assert result.is_ok, result.error


@pytest.mark.live
def test_mirostat_is_accepted_but_absent_from_the_sampler_chain(
    server_samplers, chat_base_url
):
    """mirostat is a recognised param but is not in the active chain.

    The server still reports mirostat/mirostat_tau/mirostat_eta among its
    defaults, so the request is accepted, but 'mirostat' does not appear
    in the sampler chain this build runs. The ablation lab's mirostat
    presets cannot be shown to do anything here.
    """
    assert "mirostat" not in server_samplers

    for preset in (preset_mirostat_v1, preset_mirostat_v2):
        cfg = _small(preset())
        assert "mirostat" in cfg.to_request_body()
        result = run_ablation([cfg], PROMPT, base_url=chat_base_url, max_retries=0)[0]
        assert result.is_ok, f"{cfg.label} rejected: {result.error}"
        assert result.output


@pytest.mark.live
def test_config_uses_typical_p_the_name_the_server_accepts(server_samplers):
    """The chain calls it typ_p; the request field is typical_p.

    Both refer to the same sampler. The project emits `typical_p`, which
    matches the server's request-side parameter name, so this is a naming
    difference in /props only - not a defect.
    """
    assert "typ_p" in server_samplers
    body = SamplerConfig(typical_p=0.5).to_request_body()
    assert "typical_p" in body
    assert "typ_p" not in body


@pytest.mark.live
def test_unknown_sampler_params_are_silently_ignored(chat_base_url):
    """llama-server accepts unknown body keys without complaint.

    Worth pinning: it means a typo'd sampler name in an ablation grid
    produces a clean-looking run that measured nothing.
    """
    body = {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 8,
        "definitely_not_a_sampler": 1.23,
    }
    req = urllib.request.Request(
        f"{chat_base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    assert data["choices"][0]["message"]["content"] is not None
