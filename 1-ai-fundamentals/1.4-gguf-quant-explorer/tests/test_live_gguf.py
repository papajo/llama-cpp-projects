"""Live integration tests for the GGUF parser and quant estimator.

The offline suite parses synthetic GGUF byte streams. These tests parse the
real .gguf files on disk and cross-check the derived architecture against
what the live llama-server reports for the same model family via
/v1/models, which is the only independent source of truth available.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gguf_reader.parser import GGUFParser  # noqa: E402
from gguf_reader.models import QuantType  # noqa: E402
from quant_planner.estimator import QuantEstimator  # noqa: E402

MODELS_DIR = Path(os.environ.get("LLM_MODELS_DIR", "/home/pa-joshi/Models"))
NOMIC = MODELS_DIR / "nomic-embed-text-v1.5-GGUF" / "nomic-embed-text-v1.5.f16.gguf"
QWEN = MODELS_DIR / "Qwen2.5-1.5B-Instruct-GGUF" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"


@pytest.fixture(scope="module")
def nomic():
    if not NOMIC.exists():
        pytest.skip(f"GGUF not present: {NOMIC}")
    return GGUFParser.parse(str(NOMIC))


@pytest.fixture(scope="module")
def qwen():
    if not QWEN.exists():
        pytest.skip(f"GGUF not present: {QWEN}")
    return GGUFParser.parse(str(QWEN))


@pytest.mark.live
def test_parses_real_gguf_header(nomic):
    """A real file parses, with a sane header and tensor table."""
    assert nomic.gguf_version == 3
    assert nomic.architecture == "nomic-bert"
    assert nomic.tensor_count == len(nomic.tensors) > 0
    assert nomic.file_size_bytes == NOMIC.stat().st_size


@pytest.mark.live
def test_derived_dims_match_live_server(nomic, live_server_meta):
    """Fields derived from the GGUF agree with the running server.

    Regression guard: embedding_length and the tokenizer vocab used to fall
    back to hardcoded llama-2 defaults (4096 / 32000) because the real keys
    sit elsewhere in the metadata.
    """
    meta = live_server_meta["embed"]["data"][0]["meta"]
    assert nomic.embedding_dim == meta["n_embd"] == 768
    assert nomic.vocab_size == meta["n_vocab"] == 30522
    assert nomic.context_length == meta["n_ctx_train"]


@pytest.mark.live
def test_attention_dims_are_read_not_defaulted(nomic, qwen):
    """Head counts come from <arch>.attention.*, not the default table."""
    # nomic-bert is plain MHA: 12 heads, no separate KV head count.
    assert nomic.head_count == 12
    assert nomic.head_count_kv == 12
    assert nomic.is_gqa is False

    # qwen2 genuinely is GQA.
    assert qwen.head_count == 12
    assert qwen.head_count_kv == 2
    assert qwen.is_gqa is True


@pytest.mark.live
def test_quant_type_reflects_real_file(nomic, qwen):
    """file_type maps to the quantisation the filename advertises."""
    assert "f16" in str(nomic.current_quant).lower() or nomic.file_type == 1
    # q4_k_m
    assert qwen.file_type == 15


@pytest.mark.live
def test_param_estimate_is_in_the_right_ballpark(nomic, live_server_meta):
    """The parameter estimate tracks the server's exact count.

    This is a dimensional estimate, not a tensor-by-tensor sum, so it is
    only asserted to within a wide band.
    """
    exact = live_server_meta["embed"]["data"][0]["meta"]["n_params"]
    estimated = nomic.param_count_b * 1e9
    assert 0.6 * exact < estimated < 1.6 * exact, (
        f"estimated {estimated:.0f} vs exact {exact}"
    )


@pytest.mark.live
def test_estimator_runs_on_real_model(nomic):
    """QuantEstimator produces a coherent estimate from a real model."""
    est = QuantEstimator(nomic)
    result = est.estimate(QuantType.Q4_K, context_length=2048)
    assert result.model_weights_gb > 0
    assert result.kv_cache_gb > 0
    assert result.total_gb >= result.model_weights_gb + result.kv_cache_gb
    assert 0.0 < result.weights_pct <= 100.0
    assert result.context_length == 2048

    # A coarser quant must not need more space for the same weights.
    bigger = est.estimate(QuantType.Q8_0, context_length=2048)
    assert bigger.model_weights_gb > result.model_weights_gb
    # KV cache depends on context, not on weight quant.
    assert bigger.kv_cache_gb == pytest.approx(result.kv_cache_gb)
