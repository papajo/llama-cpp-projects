"""Tests for budget.py — SweepConfig and presets."""

from budget_sweep.budget import (
    SweepConfig,
    load_preset,
    preset_token_budget,
    preset_temperature_sweep,
    preset_sampling_full,
)


class TestSweepConfig:
    def test_default_param_grid(self):
        """Defaults to a simple max_tokens sweep."""
        cfg = SweepConfig()
        assert cfg.param_grid is not None
        assert "max_tokens" in cfg.param_grid
        assert len(cfg.param_grid["max_tokens"]) >= 3

    def test_n_combinations(self):
        cfg = SweepConfig(
            param_grid={"max_tokens": [64, 128], "temperature": [0.0, 0.7]},
        )
        assert cfg.n_combinations == 4  # 2 × 2

    def test_n_combinations_single(self):
        cfg = SweepConfig(
            param_grid={"max_tokens": [256]},
        )
        assert cfg.n_combinations == 1

    def test_n_total_runs(self):
        cfg = SweepConfig(
            param_grid={"max_tokens": [64, 128]},
            prompts=["p1", "p2"],
            n_runs=3,
        )
        # 2 × 2 × 3 = 12
        assert cfg.n_total_runs == 12

    def test_iter_configs(self):
        cfg = SweepConfig(
            param_grid={"temperature": [0.0, 1.0], "max_tokens": [128]},
        )
        configs = cfg.iter_configs()
        assert len(configs) == 2  # 2 × 1
        assert configs[0]["temperature"] == 0.0
        assert configs[0]["max_tokens"] == 128
        assert configs[1]["temperature"] == 1.0
        assert configs[1]["max_tokens"] == 128

    def test_iter_configs_empty_grid(self):
        cfg = SweepConfig(param_grid={})
        configs = cfg.iter_configs()
        # An empty grid produces a single empty config
        assert len(configs) == 1
        assert configs[0] == {}

    def test_annotate(self):
        cfg = SweepConfig()
        assert cfg.annotate("max_tokens", 512) == "tok=512"
        assert cfg.annotate("temperature", 0.7) == "T=0.7"
        assert cfg.annotate("top_p", 0.9) == "p=0.9"
        assert cfg.annotate("top_k", 40) == "k=40"
        assert cfg.annotate("min_p", 0.05) == "minp=0.05"
        assert cfg.annotate("foo", "bar") == "foo=bar"

    def test_default_prompts(self):
        cfg1 = SweepConfig()
        cfg2 = SweepConfig()
        # Should be independent copies
        assert cfg1.prompts == cfg2.prompts
        assert cfg1.prompts is not cfg2.prompts


class TestPresets:
    def test_preset_token_budget(self):
        cfg = preset_token_budget()
        assert "max_tokens" in cfg.param_grid
        assert 64 in cfg.param_grid["max_tokens"]
        assert 4096 in cfg.param_grid["max_tokens"]
        assert cfg.n_runs == 3
        assert "token-budget" in cfg.tags

    def test_preset_temperature_sweep(self):
        cfg = preset_temperature_sweep()
        assert "temperature" in cfg.param_grid
        assert 2.0 in cfg.param_grid["temperature"]
        assert cfg.n_runs == 5

    def test_preset_sampling_full(self):
        cfg = preset_sampling_full()
        assert "temperature" in cfg.param_grid
        assert "top_p" in cfg.param_grid
        assert "max_tokens" in cfg.param_grid

    def test_load_preset_by_name(self):
        cfg = load_preset("token-budget", base_url="http://localhost:9999")
        assert cfg.base_url == "http://localhost:9999"
        assert "token-budget" in cfg.tags

    def test_load_preset_invalid(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent")
