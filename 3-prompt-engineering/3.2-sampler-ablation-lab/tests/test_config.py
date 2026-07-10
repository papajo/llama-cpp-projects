"""Tests for config.py — SamplerConfig, presets, ablate."""

from sampler_ablation.config import (
    SamplerConfig,
    preset_greedy,
    preset_creative,
    preset_balanced,
    preset_precise,
    preset_mirostat_v1,
    preset_mirostat_v2,
    ablate,
)


class TestSamplerConfig:
    def test_default_values(self):
        cfg = SamplerConfig()
        assert cfg.temperature == 0.7
        assert cfg.top_k == 40
        assert cfg.top_p == 0.9
        assert cfg.mirostat == 0

    def test_auto_label(self):
        cfg = SamplerConfig(temperature=0.0)
        assert "greedy" in cfg.label

        cfg = SamplerConfig(temperature=0.7)
        assert "T=0.7" in cfg.label

    def test_explicit_label(self):
        cfg = SamplerConfig(temperature=0.5, label="my-test")
        assert cfg.label == "my-test"

    def test_frozen(self):
        cfg = SamplerConfig()
        import pytest
        with pytest.raises(Exception):
            cfg.temperature = 0.0  # frozen dataclass

    def test_to_request_body_empty(self):
        """Config with all defaults produces empty body."""
        cfg = SamplerConfig(max_tokens=512)  # all defaults
        body = cfg.to_request_body()
        assert body == {}

    def test_to_request_body_non_default(self):
        cfg = SamplerConfig(temperature=0.0, max_tokens=256)
        body = cfg.to_request_body()
        assert body["temperature"] == 0.0
        assert body["max_tokens"] == 256
        # Default params should not be in body
        assert "top_k" not in body

    def test_to_request_body_seed(self):
        cfg = SamplerConfig(seed=42)
        body = cfg.to_request_body()
        assert body["seed"] == 42

    def test_param_summary(self):
        cfg = SamplerConfig(temperature=0.2, max_tokens=128)
        summary = cfg.param_summary()
        assert "temperature" in summary
        assert "max_tokens" in summary
        assert "0.2" in summary

    def test_param_summary_all_defaults(self):
        cfg = SamplerConfig()
        summary = cfg.param_summary()
        assert "all defaults" in summary


class TestPresets:
    def test_greedy(self):
        cfg = preset_greedy()
        assert cfg.temperature == 0.0
        assert cfg.top_k == 1
        assert cfg.label == "greedy"

    def test_creative(self):
        cfg = preset_creative()
        assert cfg.temperature == 1.2
        assert cfg.top_k == 100
        assert cfg.repeat_penalty == 1.1

    def test_balanced(self):
        cfg = preset_balanced()
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9

    def test_precise(self):
        cfg = preset_precise()
        assert cfg.temperature == 0.2
        assert cfg.top_k == 20

    def test_mirostat_v1(self):
        cfg = preset_mirostat_v1()
        assert cfg.mirostat == 1

    def test_mirostat_v2(self):
        cfg = preset_mirostat_v2()
        assert cfg.mirostat == 2

    def test_all_presets_have_descriptions(self):
        for factory in [
            preset_greedy, preset_creative, preset_balanced,
            preset_precise, preset_mirostat_v1, preset_mirostat_v2,
        ]:
            cfg = factory()
            assert cfg.description, f"{cfg.label} missing description"


class TestAblate:
    def test_ablate_default_reference(self):
        configs = ablate()
        assert len(configs) >= 2
        assert configs[0].label == "balanced"

    def test_ablate_custom_reference(self):
        configs = ablate(reference=preset_greedy())
        assert configs[0].label == "greedy"
        # Should have variants for each ablated param
        assert len(configs) > 1

    def test_ablate_no_duplicate_labels(self):
        configs = ablate()
        labels = [c.label for c in configs]
        assert len(labels) == len(set(labels))

    def test_ablate_all_variants_differ_from_reference(self):
        configs = ablate(reference=preset_balanced())
        ref = configs[0]
        for variant in configs[1:]:
            # At least one param should differ
            ref_body = ref.to_request_body()
            var_body = variant.to_request_body()
            assert ref_body != var_body or variant.label != ref.label

    def test_ablate_with_custom_params(self):
        configs = ablate(
            reference=preset_balanced(),
            params=[("temperature", [0.0, 1.5])],
        )
        # Reference + 2 variants (0.0 and 1.5). But 0.7 (the ref val) is skipped.
        assert len(configs) == 3
        # The variants should have temperature 0.0 and 1.5
        temps = {c.temperature for c in configs[1:]}
        assert 0.0 in temps
        assert 1.5 in temps
