"""
Tests for the GGUF Quant Explorer.

Tests the quant models, estimator, and CLI without needing
a real GGUF file (uses synthetic model data).
"""

import sys
import json
import math
from pathlib import Path

# Add project root to path
PROJ_ROOT = str(Path(__file__).parent.parent)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from gguf_reader.models import QuantType, GGUFModel, TensorInfo, GGUFMetadata, GGUFValueType
from quant_planner.estimator import QuantEstimator, HardwareProfile, MemoryEstimate


def _make_test_model(
    name: str = "test-model",
    architecture: str = "llama",
    embedding_dim: int = 4096,
    block_count: int = 32,
    head_count: int = 32,
    head_count_kv: int = 8,
    feed_forward_dim: int = 11008,
    vocab_size: int = 32000,
    context_length: int = 8192,
    param_count_b: float = 7.0,
    file_type: int = 12,  # Q4_K
) -> GGUFModel:
    """Create a synthetic GGUFModel for testing."""
    return GGUFModel(
        file_path="/fake/model.gguf",
        file_size_bytes=int(7 * 1e9),
        gguf_version=3,
        tensor_count=300,
        metadata_kv_count=20,
        metadata={
            "general.architecture": architecture,
            "general.name": name,
            "general.file_type": file_type,
            f"{architecture}.embedding_length": embedding_dim,
            f"{architecture}.block_count": block_count,
            f"{architecture}.head_count": head_count,
            f"{architecture}.head_count_kv": head_count_kv,
            f"{architecture}.feed_forward_length": feed_forward_dim,
            f"{architecture}.vocab_size": vocab_size,
            f"{architecture}.context_length": context_length,
        },
        architecture=architecture,
        model_name=name,
        file_type=file_type,
        current_quant=QuantType.from_file_type(file_type),
        embedding_dim=embedding_dim,
        block_count=block_count,
        head_count=head_count,
        head_count_kv=head_count_kv,
        feed_forward_dim=feed_forward_dim,
        vocab_size=vocab_size,
        context_length=context_length,
        param_count_b=param_count_b,
    )


# ── Tests ─────────────────────────────────────────────────────────

def test_quant_type_bits():
    """Verify bits_per_weight for common quant types."""
    assert QuantType.F32.bits_per_weight == 32.0
    assert QuantType.F16.bits_per_weight == 16.0
    assert QuantType.Q4_0.bits_per_weight == 4.5
    assert QuantType.Q4_K.bits_per_weight == 4.5
    assert QuantType.Q6_K.bits_per_weight == 6.5625
    assert QuantType.Q8_0.bits_per_weight == 8.5
    assert QuantType.IQ2_XXS.bits_per_weight == 2.0625
    assert QuantType.IQ1_S.bits_per_weight == 1.5625
    print("  ✓ QuantType bits_per_weight correct")


def test_quant_type_from_file_type():
    """Test file_type to QuantType mapping."""
    assert QuantType.from_file_type(0) == QuantType.F32
    assert QuantType.from_file_type(8) == QuantType.Q8_0
    assert QuantType.from_file_type(12) == QuantType.Q4_K
    assert QuantType.from_file_type(14) == QuantType.Q6_K
    print("  ✓ QuantType file_type mapping correct")


def test_quant_quality_scores():
    """Quality scores should decrease with more aggressive quant."""
    assert QuantType.F32.relative_quality > QuantType.F16.relative_quality
    assert QuantType.F16.relative_quality > QuantType.Q8_0.relative_quality
    assert QuantType.Q8_0.relative_quality > QuantType.Q4_K.relative_quality
    assert QuantType.Q4_K.relative_quality > QuantType.Q2_K.relative_quality
    assert QuantType.Q2_K.relative_quality > QuantType.IQ1_S.relative_quality
    print("  ✓ Quality scores ordered correctly")


def test_k_quant_detection():
    """K-quants should be identified correctly."""
    assert QuantType.Q4_K.is_k_quant
    assert QuantType.Q6_K.is_k_quant
    assert not QuantType.F32.is_k_quant
    assert not QuantType.Q8_0.is_k_quant
    print("  ✓ K-quant detection correct")


def test_i_quant_detection():
    """Importance quants should be identified correctly."""
    assert QuantType.IQ1_S.is_i_quant
    assert QuantType.IQ2_XXS.is_i_quant
    assert QuantType.IQ4_XS.is_i_quant
    assert not QuantType.F32.is_i_quant
    assert not QuantType.Q4_K.is_i_quant
    print("  ✓ IQ detection correct")


def test_estimate_llama_8b():
    """Estimate memory for a Llama-3.1-8B class model."""
    model = _make_test_model(
        name="test-llama-8b",
        embedding_dim=4096,
        block_count=32,
        head_count=32,
        head_count_kv=8,
        feed_forward_dim=14336,
        vocab_size=128256,
        context_length=8192,
        param_count_b=8.03,
    )
    estimator = QuantEstimator(model)

    # F32 estimate
    f32 = estimator.estimate(QuantType.F32, context_length=4096)
    assert f32.model_weights_gb > 20  # 8B params * 32bps should be ~30GB
    assert f32.kv_cache_gb > 0
    assert f32.total_gb > f32.model_weights_gb
    print(f"  ✓ Llama-8B F32: {f32.total_gb:.1f} GB (weights: {f32.model_weights_gb:.1f} GB, "
          f"kv: {f32.kv_cache_gb:.1f} GB)")

    # Q4_K estimate should be much smaller
    q4k = estimator.estimate(QuantType.Q4_K, context_length=4096)
    assert q4k.total_gb < f32.total_gb
    assert q4k.total_gb > 0
    print(f"  ✓ Llama-8B Q4_K: {q4k.total_gb:.1f} GB")

    # Q2_K should be smaller than Q4_K
    q2k = estimator.estimate(QuantType.Q2_K, context_length=4096)
    assert q2k.total_gb < q4k.total_gb
    print(f"  ✓ Llama-8B Q2_K: {q2k.total_gb:.1f} GB")


def test_estimate_all():
    """Estimate_all should return estimates for multiple quants."""
    model = _make_test_model()
    estimator = QuantEstimator(model)
    estimates = estimator.estimate_all()

    assert len(estimates) > 5
    # Estimates should be sorted by total_gb after sorting
    totals = [e.total_gb for e in estimates]
    assert all(t > 0 for t in totals)
    print(f"  ✓ {len(estimates)} quant estimates generated")


def test_oom_check():
    """OOM check should correctly identify fitting/non-fitting quants."""
    model = _make_test_model(param_count_b=70.0)  # 70B model
    estimator = QuantEstimator(model)

    # 24GB VRAM — Q4_K should not fit for 70B
    hw = HardwareProfile.rtx_3090_24gb()
    estimates = estimator.check_oom(hw)
    fitting = [e for e in estimates if e.fits_in_vram]
    not_fitting = [e for e in estimates if not e.fits_in_vram]

    # At least some quants should fit (small ones like IQ1_S, Q2_K)
    # and some shouldn't (F32, F16)
    assert len(fitting) > 0, "At least some quants should fit in 24GB"
    assert len(not_fitting) > 0, "At least some quants should not fit in 24GB"

    # F32 should never fit for 70B in 24GB
    f32_est = next(e for e in estimates if e.quant_type == QuantType.F32)
    assert not f32_est.fits_in_vram

    print(f"  ✓ OOM check: {len(fitting)} fit, {len(not_fitting)} don't fit in 24GB")


def test_recommend():
    """Recommendation should return best fitting quants sorted by quality."""
    model = _make_test_model(param_count_b=7.0)
    estimator = QuantEstimator(model)
    hw = HardwareProfile.rtx_4090_24gb()

    recs = estimator.recommend(hw, min_quality=0.5)
    assert len(recs) > 0
    # Should be sorted by quality descending
    for i in range(len(recs) - 1):
        assert recs[i].quality_score >= recs[i + 1].quality_score
    # Top pick should have high quality
    assert recs[0].quality_score >= 0.8

    print(f"  ✓ Top recommendation for 7B on 24GB: {recs[0].quant_type.name} "
          f"({recs[0].total_gb:.1f} GB, {recs[0].quality_score*100:.0f}%)")


def test_hardware_profiles():
    """Hardware profiles have reasonable values."""
    assert HardwareProfile.rtx_4090_24gb().total_vram_gb == 24.0
    assert HardwareProfile.a100_80gb().total_vram_gb == 80.0
    assert HardwareProfile.macbook_m2_24gb().is_apple_silicon
    assert not HardwareProfile.rtx_3060_12gb().is_apple_silicon
    print("  ✓ Hardware profiles correct")


def test_estimate_moe():
    """MoE models should have reasonable estimates."""
    model = _make_test_model(name="test-moe", param_count_b=16.0)
    model.expert_count = 8
    model.expert_used_count = 2
    estimator = QuantEstimator(model)
    est = estimator.estimate(QuantType.Q4_K)
    assert est.total_gb > 0
    print(f"  ✓ MoE 16B Q4_K: {est.total_gb:.1f} GB")


def test_context_scaling():
    """Larger context should increase KV cache."""
    model = _make_test_model()
    estimator = QuantEstimator(model)

    est_4k = estimator.estimate(QuantType.Q4_K, context_length=4096)
    est_8k = estimator.estimate(QuantType.Q4_K, context_length=8192)
    est_32k = estimator.estimate(QuantType.Q4_K, context_length=32768)

    assert est_8k.kv_cache_gb > est_4k.kv_cache_gb
    assert est_32k.kv_cache_gb > est_8k.kv_cache_gb
    print(f"  ✓ Context scaling: 4K={est_4k.kv_cache_gb:.2f}GB, "
          f"8K={est_8k.kv_cache_gb:.2f}GB, "
          f"32K={est_32k.kv_cache_gb:.2f}GB")


def test_quality_scores_exist():
    """All quant types have a quality score."""
    for qt in QuantType:
        score = qt.relative_quality
        assert 0.0 <= score <= 1.0, f"{qt.name} quality score {score} out of range"
    print(f"  ✓ All {len(list(QuantType))} quant types have quality scores")


def test_benchmark_data():
    """Benchmark data file is valid JSON with expected structure."""
    path = Path(__file__).parent.parent / "data" / "benchmarks.json"
    data = json.loads(path.read_text())
    assert "models" in data
    assert "meta" in data
    assert len(data["models"]) > 0

    for name, model_data in data["models"].items():
        assert "param_count" in model_data
        assert "f16_ppl" in model_data
        assert "quants" in model_data
        # F16 ppl should be the lowest
        for qname, qppl in model_data["quants"].items():
            assert qppl >= model_data["f16_ppl"], (
                f"{name}/{qname} ppl ({qppl}) < f16 ppl ({model_data['f16_ppl']})"
            )

    print(f"  ✓ Benchmark data valid ({len(data['models'])} models)")


if __name__ == "__main__":
    print("GGUF Quant Explorer Tests")
    print("=" * 50)
    passed = 0
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                passed += 1
            except Exception as e:
                print(f"  ✗ FAILED: {name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
    print("=" * 50)
    print(f"Result: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
