"""
Parameter budget definitions for reasoning budget sweeps.

Provides ``SweepConfig`` — a collection of parameter ranges to sweep
over, along with built-in presets for common experiments.

Usage::

    from budget_sweep.budget import SweepConfig, preset_token_budget

    config = SweepConfig(
        base_url="http://127.0.0.1:8080",
        prompts=[
            "Write a haiku about Python.",
            "Explain the meaning of life in one paragraph.",
        ],
        param_grid={
            "max_tokens": [64, 128, 256, 512, 1024],
            "temperature": [0.0, 0.7, 1.5],
        },
        n_runs=3,
    )
    # Produces 5 × 3 × 3 × 2 = 90 total runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_PROMPTS: List[str] = [
    "Write a concise summary of how neural networks work.",
    "Explain the difference between supervised and unsupervised learning in one paragraph.",
    "List three benefits of using Python for data science.",
    "Write a short poem about artificial intelligence.",
]


@dataclass
class SweepConfig:
    """
    Configuration for a reasoning budget sweep experiment.

    The ``param_grid`` defines all parameter combinations to test.
    Each key maps to a list of values.  The Cartesian product of all
    lists is used.

    Args:
        base_url: llama.cpp server URL.
        prompts: List of prompt strings to run for each config.
        param_grid: Dict of parameter name → list of values.
            Supported keys match llama.cpp ``/v1/chat/completions`` body:
            ``max_tokens``, ``temperature``, ``top_p``, ``top_k``,
            ``min_p``, ``repeat_penalty``, ``frequency_penalty``,
            ``presence_penalty``, ``seed``, ``n_keep``.
        n_runs: Number of repeat runs per config (for diversity measurement).
        system_prompt: Optional system prompt prepended to all requests.
        tags: Optional metadata tags for this sweep.
    """

    base_url: str = "http://127.0.0.1:8080"
    prompts: List[str] = field(default_factory=lambda: DEFAULT_PROMPTS.copy())
    param_grid: Optional[Dict[str, List[Any]]] = None
    n_runs: int = 1

    # Optional cross-cutting settings
    system_prompt: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.param_grid is None:
            self.param_grid = {"max_tokens": [64, 128, 256, 512]}

    @property
    def n_combinations(self) -> int:
        """Number of parameter combinations in the grid."""
        total = 1
        for values in self.param_grid.values():
            total *= len(values)
        return total

    @property
    def n_total_runs(self) -> int:
        """Total API calls: combinations × prompts × runs."""
        return self.n_combinations * len(self.prompts) * self.n_runs

    def iter_configs(self) -> List[Dict[str, Any]]:
        """Yield a dict of resolved parameters for each combination."""
        import itertools

        keys = list(self.param_grid.keys())
        value_lists = [self.param_grid[k] for k in keys]

        configs: List[Dict[str, Any]] = []
        for combo in itertools.product(*value_lists):
            configs.append(dict(zip(keys, combo)))
        return configs

    def annotate(self, param: str, value: Any) -> str:
        """Short label for a (param, value) pair (used in reports)."""
        if param == "max_tokens":
            return f"tok={value}"
        if param == "temperature":
            return f"T={value}"
        if param == "top_p":
            return f"p={value}"
        if param == "top_k":
            return f"k={value}"
        if param == "min_p":
            return f"minp={value}"
        return f"{param}={value}"


# ── Presets ──────────────────────────────────────────────────────


def preset_token_budget(
    base_url: str = "http://127.0.0.1:8080",
    temperatures: Optional[List[float]] = None,
) -> SweepConfig:
    """Sweep ``max_tokens`` from 64→4096 at one or more temperatures."""
    return SweepConfig(
        base_url=base_url,
        param_grid={
            "max_tokens": [64, 128, 256, 512, 1024, 2048, 4096],
            "temperature": temperatures or [0.0, 0.7],
            "top_p": [0.9],
        },
        n_runs=3,
        tags=["token-budget"],
    )


def preset_temperature_sweep(
    base_url: str = "http://127.0.0.1:8080",
) -> SweepConfig:
    """Sweep temperature from 0.0→2.0 with fixed max_tokens."""
    return SweepConfig(
        base_url=base_url,
        param_grid={
            "temperature": [0.0, 0.3, 0.7, 1.0, 1.5, 2.0],
            "max_tokens": [512],
            "top_p": [0.9],
        },
        n_runs=5,
        tags=["temperature-sweep"],
    )


def preset_sampling_full(
    base_url: str = "http://127.0.0.1:8080",
) -> SweepConfig:
    """
    Sweep all major sampling parameters at a moderate budget.

    Useful for finding which knob has the biggest impact on output
    diversity vs. coherence.
    """
    return SweepConfig(
        base_url=base_url,
        param_grid={
            "temperature": [0.0, 0.7, 1.5],
            "top_p": [0.5, 0.9, 1.0],
            "max_tokens": [256],
        },
        n_runs=3,
        tags=["sampling-full"],
    )


def preset_context_window(
    base_url: str = "http://127.0.0.1:8080",
) -> SweepConfig:
    """
    Test how context length affects output coherence.

    Note: ``n_ctx`` is a server-level param, not per-request.  The
    sweep will set it in the request body (llama.cpp ignores if the
    server was started with a smaller value).  Best run with dedicated
    server instances per context length.
    """
    return SweepConfig(
        base_url=base_url,
        param_grid={
            "n_ctx": [512, 2048, 8192, 32768],
            "max_tokens": [256],
            "temperature": [0.3],
        },
        n_runs=2,
        tags=["context-window"],
    )


PRESET_MAP = {
    "token-budget": preset_token_budget,
    "temperature-sweep": preset_temperature_sweep,
    "sampling-full": preset_sampling_full,
    "context-window": preset_context_window,
}


def load_preset(name: str, **kwargs: Any) -> SweepConfig:
    """
    Load a preset by name, forwarding extra kwargs.

    Raises:
        ValueError: If the preset name is unknown.
    """
    factory = PRESET_MAP.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown preset {name!r}.  Available: {sorted(PRESET_MAP)}"
        )
    return factory(**kwargs)
