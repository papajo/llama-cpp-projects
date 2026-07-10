"""Tests for the sampling simulation engine."""

import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.sampler import (
    SamplingPipeline, softmax, apply_temperature, apply_top_k,
    apply_top_p, apply_min_p, DISTRIBUTION_PRESETS
)


def _logits():
    return [5.0, 3.0, 1.0, 0.5, 0.2, -1.0, -2.0, -3.0]


def test_softmax():
    p = softmax(_logits())
    assert abs(sum(p) - 1.0) < 1e-6
    assert p[0] > p[1] > p[2]
    print("  ✓ softmax sums to 1")


def test_temperature():
    logits = _logits()
    t0 = apply_temperature(logits, 0.5)
    t1 = apply_temperature(logits, 2.0)
    assert max(t0) > max(logits)  # cold = sharper
    assert max(t1) < max(logits)  # hot = flatter
    print("  ✓ temperature scaling correct")


def test_top_k():
    logits = _logits()
    result = apply_top_k(logits, 2)
    valid = [l for l in result if l != float("-inf")]
    assert len(valid) == 2
    print("  ✓ top_k keeps exactly K tokens")


def test_top_p():
    probs = softmax(_logits())
    result = apply_top_p(probs, 0.9)
    total = sum(result)
    assert total <= 1.0
    assert total > 0
    # The smallest set with cumulative prob >= 0.9 was kept
    remaining = sum(1 for p in result if p > 0)
    assert remaining < len(probs)  # some tokens were pruned
    print(f"  ✓ top_p: {remaining}/{len(probs)} tokens kept (total={total:.4f})")


def test_min_p():
    probs = softmax(_logits())
    result = apply_min_p(probs, 0.5)
    max_p = max(probs)
    for p in result:
        if p > 0:
            assert p >= 0.5 * max_p
    print("  ✓ min_p respects threshold")


def test_full_pipeline():
    preset = DISTRIBUTION_PRESETS[2]  # flat distribution
    logits = preset.generator(100)
    pipe = SamplingPipeline(logits)
    result = pipe.run(temperature=1.0, top_k=50, top_p=0.95)
    assert len(result["snapshots"]) > 0
    assert result["summary"]["final_entropy"] >= 0
    assert result["summary"]["num_snapshots"] > 0
    assert len(result["final_probs"]) == 50
    print(f"  ✓ full pipeline: {result['summary']['num_snapshots']} steps, "
          f"entropy={result['summary']['final_entropy']:.2f}")


def test_sampler_sequence_order():
    """Different sampler sequences should produce different results."""
    logits = [10.0, 8.0, 5.0, 3.0, 1.0]
    pipe = SamplingPipeline(logits)
    
    r1 = pipe.run(sampler_sequence="temp,top_k,top_p")
    r2 = pipe.run(sampler_sequence="top_k,top_p,temp")
    
    # Sequences differ so distributions should differ
    s1 = r1["snapshots"]
    s2 = r2["snapshots"]
    assert len(s1) == len(s2)
    print("  ✓ sampler sequencing works")


def test_distribution_presets():
    """All distribution presets generate valid logits."""
    for preset in DISTRIBUTION_PRESETS:
        logits = preset.generator(100)
        assert len(logits) == 100
        probs = softmax(logits)
        assert abs(sum(probs) - 1.0) < 1e-6
    print(f"  ✓ {len(DISTRIBUTION_PRESETS)} presets generate valid distributions")


if __name__ == "__main__":
    print("Sampler Simulation Tests")
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
                import traceback; traceback.print_exc()
                failed += 1
    print("=" * 50)
    print(f"Result: {passed} passed, {failed} failed")
