"""Tests for the prompt cache benchmark."""

import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.scenarios import (
    list_scenarios, get_scenario,
    multi_turn_conversation, repeated_prefix, long_prefix_short_query,
    code_edit_loop, batch_processing,
)
from harness.metrics import CacheBenchmarkResult, PromptResult, estimate_tokens
from harness.runner import get_scenario_list


def test_scenario_list():
    """list_scenarios returns metadata for all scenarios."""
    scenarios = list_scenarios()
    assert len(scenarios) >= 10
    for s in scenarios:
        assert "name" in s
        assert "description" in s
        assert "num_prompts" in s
        assert s["num_prompts"] > 0
    print(f"  ✓ {len(scenarios)} scenarios available")


def test_get_scenario():
    """get_scenario returns the right scenario by name."""
    s = get_scenario("multi-turn-5turns")
    assert s is not None
    assert s.name == "multi-turn-5turns"
    assert len(s.prompts) == 5
    print(f"  ✓ Got scenario '{s.name}' with {len(s.prompts)} prompts")


def test_shared_prefix():
    """Scenarios should have a shared prefix."""
    s = repeated_prefix(3)
    assert len(s.shared_prefix) > 0
    for p in s.prompts:
        assert p.startswith(s.shared_prefix)
    print(f"  ✓ Shared prefix present in all {len(s.prompts)} prompts")


def test_multi_turn_prompts_grow():
    """Multi-turn prompts should grow with each turn."""
    s = multi_turn_conversation(5)
    lengths = [len(p) for p in s.prompts]
    assert lengths == sorted(lengths), "Prompts should grow monotonically"
    assert lengths[-1] > lengths[0]
    print(f"  ✓ {len(s.prompts)} prompts grow from {lengths[0]} to {lengths[-1]} chars")


def test_estimate_tokens():
    """Token estimation should be reasonable."""
    assert estimate_tokens("hello world") == 2  # 11 // 4 = 2
    assert estimate_tokens("a") == 1
    assert estimate_tokens("") == 1
    print("  ✓ Token estimation correct")


def test_cache_result_compute():
    """CacheBenchmarkResult.compute() computes correct metrics."""
    result = CacheBenchmarkResult(scenario_name="test")

    # Add 3 cached and 3 uncached runs
    for i in range(3):
        result.runs_cached.append(PromptResult(
            prompt_index=i, prompt_length_chars=100,
            ttft_ms=50.0 + i * 10,
            cached=True,
        ))
        result.runs_uncached.append(PromptResult(
            prompt_index=i, prompt_length_chars=100,
            ttft_ms=200.0 + i * 20,
            cached=False,
        ))

    result.compute()

    assert abs(result.avg_ttft_cached_ms - 60.0) < 0.1
    assert abs(result.avg_ttft_uncached_ms - 220.0) < 0.1
    assert abs(result.speedup_factor - 3.67) < 0.1
    assert len(result.per_prompt) == 3
    print(f"  ✓ Results computed: cached={result.avg_ttft_cached_ms:.1f}ms, "
          f"uncached={result.avg_ttft_uncached_ms:.1f}ms, "
          f"speedup={result.speedup_factor:.1f}x")


def test_cache_result_empty():
    """Empty results should not crash."""
    result = CacheBenchmarkResult(scenario_name="empty")
    result.compute()  # Should not raise
    assert result.speedup_factor == 0
    print("  ✓ Empty result handles gracefully")


def test_long_prefix_scenario():
    """Long prefix scenario should have long shared prefix."""
    s = long_prefix_short_query(3)
    assert len(s.shared_prefix) > 200  # Long document prefix
    print(f"  ✓ Long prefix: {len(s.shared_prefix)} chars")


def test_code_edit_scenario():
    """Code edit scenario should include file context."""
    s = code_edit_loop(3)
    for p in s.prompts:
        assert "calculator" in p or "def " in p
    print(f"  ✓ Code edit context present in {len(s.prompts)} prompts")


def test_batch_scenario():
    """Batch scenario should have many prompts."""
    s = batch_processing(2, 2)
    expected = 2 * 3 * 2  # 2 batches × 3 templates × 2 texts
    assert len(s.prompts) == expected
    print(f"  ✓ Batch scenario: {len(s.prompts)} prompts")


def test_scenario_names_unique():
    """All scenario names should be unique."""
    seen = set()
    for s in list_scenarios():
        assert s["name"] not in seen, f"Duplicate: {s['name']}"
        seen.add(s["name"])
    print(f"  ✓ All {len(seen)} scenario names unique")


if __name__ == "__main__":
    print("Cache Benchmark Tests")
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
