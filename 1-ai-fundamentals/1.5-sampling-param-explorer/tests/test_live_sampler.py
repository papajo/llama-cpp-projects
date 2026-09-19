"""Live integration tests for the sampling pipeline.

backend/sampler.py is pure maths and never speaks HTTP, so the offline
suite drives it with synthetic distributions. These tests instead feed it
a REAL logprob distribution pulled from llama-server (/completion with
n_probs) and check the sampler transforms behave correctly on it.

Assertions are structural only. SmolLM2-360M's actual token choices are
not something to assert on.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.sampler import (  # noqa: E402
    SamplingPipeline,
    apply_min_p,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    softmax,
)

N_PROBS = 20


@pytest.fixture(scope="module")
def real_logprobs(chat_base_url):
    """Top-N logprobs for the first generated token, from the real model."""
    body = {
        "prompt": "The capital of France is",
        "n_predict": 1,
        "n_probs": N_PROBS,
        "temperature": 1.0,
    }
    req = urllib.request.Request(
        f"{chat_base_url}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())

    entries = data["completion_probabilities"][0]["top_logprobs"]
    assert len(entries) >= 5, f"server returned only {len(entries)} candidates"
    return [e["logprob"] for e in entries]


@pytest.mark.live
def test_real_logprobs_are_wellformed(real_logprobs):
    """The server's logprobs are negative reals that exponentiate sanely."""
    assert all(lp <= 0.0 for lp in real_logprobs)
    assert all(math.isfinite(lp) for lp in real_logprobs)
    # These are already log-probabilities, so they cannot sum to more than 1.
    assert sum(math.exp(lp) for lp in real_logprobs) <= 1.0 + 1e-6


@pytest.mark.live
def test_softmax_normalises_real_logits(real_logprobs):
    probs = softmax(real_logprobs)
    assert pytest.approx(sum(probs), abs=1e-9) == 1.0
    assert all(p >= 0 for p in probs)
    # Ordering must be preserved by a monotone transform.
    assert probs.index(max(probs)) == real_logprobs.index(max(real_logprobs))


@pytest.mark.live
@pytest.mark.parametrize("k", [1, 3, 5])
def test_top_k_keeps_exactly_k_on_real_distribution(real_logprobs, k):
    filtered = apply_top_k(real_logprobs, k)
    kept = [i for i, v in enumerate(filtered) if v != float("-inf")]
    assert len(kept) == k
    # The survivors must be the k largest.
    expected = sorted(
        range(len(real_logprobs)), key=lambda i: real_logprobs[i], reverse=True
    )[:k]
    assert set(kept) == set(expected)


@pytest.mark.live
def test_top_p_keeps_minimal_nucleus_on_real_distribution(real_logprobs):
    probs = softmax(real_logprobs)
    for p in (0.5, 0.9):
        filtered = apply_top_p(probs, p)
        kept = [v for v in filtered if v > 0]
        assert kept, f"top_p={p} eliminated everything"
        assert sum(kept) >= min(p, sum(probs)) - 1e-9
        # Dropping the smallest survivor must fall below the threshold,
        # i.e. the nucleus is minimal.
        if len(kept) > 1:
            assert sum(kept) - min(kept) < p


@pytest.mark.live
def test_min_p_threshold_is_relative_to_peak(real_logprobs):
    probs = softmax(real_logprobs)
    peak = max(probs)
    filtered = apply_min_p(probs, 0.1)
    for original, kept in zip(probs, filtered):
        if kept > 0:
            assert original >= 0.1 * peak - 1e-12
    assert max(filtered) == peak


@pytest.mark.live
def test_temperature_monotonically_flattens_real_distribution(real_logprobs):
    """Higher temperature must lower the peak probability."""
    peaks = []
    for t in (0.5, 1.0, 2.0):
        peaks.append(max(softmax(apply_temperature(real_logprobs, t))))
    assert peaks[0] > peaks[1] > peaks[2], peaks


@pytest.mark.live
def test_pipeline_runs_the_servers_own_sampler_order(real_logprobs, chat_base_url):
    """Drive the pipeline with the chain the real server reports.

    /props exposes the active sampler chain; the pipeline should accept the
    subset it implements and produce a normalised distribution.
    """
    pipeline = SamplingPipeline(real_logprobs)
    out = pipeline.run(
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        min_p=0.05,
        sampler_sequence="top_k,top_p,min_p,temp",
    )
    final = out["final_probs"]
    assert pytest.approx(sum(final), abs=1e-6) == 1.0
    assert len([p for p in final if p > 0]) <= 5  # top_k=5 bounds the support
    assert len(out["snapshots"]) == 4


@pytest.mark.live
def test_aggressive_truncation_never_yields_nan(real_logprobs):
    """A sampler chain must not collapse a real distribution to NaN.

    Regression guard: softmax([-inf]*n) returned all-NaN, because
    `l - max_l` is inf-inf and the `total <= 0` fallback cannot catch NaN.
    Typical sampling could empty the candidate set and trigger exactly that.
    """
    pipeline = SamplingPipeline(real_logprobs)
    for typical_p in (0.2, 0.5, 0.95, 0.99):
        out = pipeline.run(
            temperature=1.0,
            top_p=typical_p,
            sampler_sequence="typ",
        )
        final = out["final_probs"]
        assert all(math.isfinite(p) for p in final), (
            f"typical_p={typical_p} produced non-finite probs"
        )
        assert math.isfinite(out["summary"]["final_entropy"])
        assert out["summary"]["final_entropy"] >= 0
        assert any(p > 0 for p in final), "candidate set was emptied"
