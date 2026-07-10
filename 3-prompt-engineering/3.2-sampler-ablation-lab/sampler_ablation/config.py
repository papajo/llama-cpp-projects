"""
Sampler configuration definitions for the ablation lab.

Provides ``SamplerConfig`` — a frozen dataclass holding every sampling
parameter that llama.cpp accepts, along with built-in presets and an
ablation generator that creates variants by toggling one parameter at
a time.

Usage::

    from sampler_ablation.config import (
        SamplerConfig, preset_greedy, preset_creative, ablate,
    )

    # Compare two named presets
    configs = [preset_greedy(), preset_creative()]
    for cfg in configs:
        body = cfg.to_request_body()
        print(cfg.label, body)

    # Or ablate from a reference: toggle each param individually
    ablation_configs = ablate(preset_balanced())
    # Returns [reference, variant-1, variant-2, ...]
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Sampler parameter defaults ───────────────────────────────────
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_K: int = 40
DEFAULT_TOP_P: float = 0.9
DEFAULT_MIN_P: float = 0.0
DEFAULT_TFS_Z: float = 1.0
DEFAULT_TYPICAL_P: float = 1.0
DEFAULT_REPEAT_PENALTY: float = 1.0
DEFAULT_FREQUENCY_PENALTY: float = 0.0
DEFAULT_PRESENCE_PENALTY: float = 0.0
DEFAULT_MIROSTAT: int = 0        # 0=off, 1=v1, 2=v2
DEFAULT_MIROSTAT_TAU: float = 5.0
DEFAULT_MIROSTAT_ETA: float = 0.1
DEFAULT_SEED: Optional[int] = None
DEFAULT_MAX_TOKENS: int = 512
DEFAULT_N_KEEP: int = 0


@dataclass(frozen=True)
class SamplerConfig:
    """
    Complete sampling parameter configuration for llama.cpp.

    Each field maps directly to a key in the ``/v1/chat/completions``
    request body.  Setting ``mirostat=0`` disables mirostat.
    """

    # Core sampling
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    min_p: float = DEFAULT_MIN_P

    # Advanced samplers
    tfs_z: float = DEFAULT_TFS_Z
    typical_p: float = DEFAULT_TYPICAL_P

    # Penalties
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY

    # Mirostat (0=off, 1=v1, 2=v2)
    mirostat: int = DEFAULT_MIROSTAT
    mirostat_tau: float = DEFAULT_MIROSTAT_TAU
    mirostat_eta: float = DEFAULT_MIROSTAT_ETA

    # Determinism / budget
    seed: Optional[int] = DEFAULT_SEED
    max_tokens: int = DEFAULT_MAX_TOKENS
    n_keep: int = DEFAULT_N_KEEP

    # Display
    label: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", self._auto_label())

    def _auto_label(self) -> str:
        """Generate a short label from the salient params."""
        parts: list[str] = []
        if self.mirostat > 0:
            parts.append(f"mirostat-v{self.mirostat}")
        elif self.temperature == 0.0:
            parts.append("greedy")
        else:
            parts.append(f"T={self.temperature}")
        if self.top_p < 1.0:
            parts.append(f"p={self.top_p}")
        if self.min_p > 0:
            parts.append(f"minp={self.min_p}")
        if self.repeat_penalty != 1.0:
            parts.append(f"rp={self.repeat_penalty}")
        return ", ".join(parts)

    def to_request_body(self) -> Dict[str, Any]:
        """
        Convert this config to a llama.cpp API request body fragment.

        Only includes params that differ from default (or are explicitly
        set) to keep the request clean.
        """
        body: Dict[str, Any] = {}

        if self.temperature != DEFAULT_TEMPERATURE:
            body["temperature"] = self.temperature
        if self.top_k != DEFAULT_TOP_K:
            body["top_k"] = self.top_k
        if self.top_p != DEFAULT_TOP_P:
            body["top_p"] = self.top_p
        if self.min_p != DEFAULT_MIN_P:
            body["min_p"] = self.min_p
        if self.tfs_z != DEFAULT_TFS_Z:
            body["tfs_z"] = self.tfs_z
        if self.typical_p != DEFAULT_TYPICAL_P:
            body["typical_p"] = self.typical_p
        if self.repeat_penalty != DEFAULT_REPEAT_PENALTY:
            body["repeat_penalty"] = self.repeat_penalty
        if self.frequency_penalty != DEFAULT_FREQUENCY_PENALTY:
            body["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != DEFAULT_PRESENCE_PENALTY:
            body["presence_penalty"] = self.presence_penalty
        if self.mirostat != DEFAULT_MIROSTAT:
            body["mirostat"] = self.mirostat
        if self.mirostat_tau != DEFAULT_MIROSTAT_TAU:
            body["mirostat_tau"] = self.mirostat_tau
        if self.mirostat_eta != DEFAULT_MIROSTAT_ETA:
            body["mirostat_eta"] = self.mirostat_eta
        if self.seed is not None:
            body["seed"] = self.seed
        if self.max_tokens != DEFAULT_MAX_TOKENS:
            body["max_tokens"] = self.max_tokens

        return body

    def param_summary(self) -> str:
        """Multi-line description of non-default parameters."""
        body = self.to_request_body()
        if not body:
            return "  (all defaults)"
        return "\n".join(
            f"  {k}: {v}" for k, v in body.items()
        )


# ── Presets ──────────────────────────────────────────────────────


def preset_greedy() -> SamplerConfig:
    """Greedy decoding: temperature=0, no randomness."""
    return SamplerConfig(
        temperature=0.0,
        top_k=1,           # only the single most likely token
        top_p=1.0,
        repeat_penalty=1.0,
        seed=42,
        label="greedy",
        description="Temperature=0, top-k=1 — deterministic, always picks the most likely token.",
    )


def preset_creative() -> SamplerConfig:
    """Creative: high temperature, broad sampling."""
    return SamplerConfig(
        temperature=1.2,
        top_k=100,
        top_p=0.95,
        repeat_penalty=1.1,
        label="creative",
        description="T=1.2, top-k=100, top-p=0.95 — diverse and unpredictable.",
    )


def preset_balanced() -> SamplerConfig:
    """Balanced: moderate temperature, nucleus sampling."""
    return SamplerConfig(
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.0,
        label="balanced",
        description="T=0.7, top-k=40, top-p=0.9 — default sensible trade-off.",
    )


def preset_precise() -> SamplerConfig:
    """Precise: low temperature, narrow sampling."""
    return SamplerConfig(
        temperature=0.2,
        top_k=20,
        top_p=0.8,
        repeat_penalty=1.0,
        label="precise",
        description="T=0.2, top-k=20, top-p=0.8 — focused, factual output.",
    )


def preset_mirostat_v1() -> SamplerConfig:
    """Mirostat v1: adaptive entropy sampling."""
    return SamplerConfig(
        temperature=0.8,
        mirostat=1,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        label="mirostat-v1",
        description="Mirostat v1 — adaptively maintains target perplexity.",
    )


def preset_mirostat_v2() -> SamplerConfig:
    """Mirostat v2: improved adaptive entropy sampling."""
    return SamplerConfig(
        temperature=0.8,
        mirostat=2,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        label="mirostat-v2",
        description="Mirostat v2 — improved entropy control.",
    )


# ── Ablation ─────────────────────────────────────────────────────

ABLATION_PARAMS = [
    ("temperature", [0.0, 1.5]),
    ("top_k", [1, 100]),
    ("top_p", [0.5, 1.0]),
    ("min_p", [0.1, 0.0]),
    ("repeat_penalty", [1.0, 1.2]),
    ("mirostat", [1, 2]),
]


def ablate(
    reference: Optional[SamplerConfig] = None,
    params: Optional[List[tuple]] = None,
) -> List[SamplerConfig]:
    """
    Generate ablation variants by toggling one parameter at a time.

    Each variant keeps everything from ``reference`` except changes
    one parameter to a contrasting value.  This makes it easy to see
    the delta each knob introduces.

    Args:
        reference: Base config.  Defaults to ``preset_balanced()``.
        params: List of ``(param_name, [value_a, value_b])`` tuples.
            Defaults to a curated set of the most impactful samplers.

    Returns:
        ``[reference, variant_1, variant_2, ...]`` — the first element
        is always the reference config.
    """
    if reference is None:
        reference = preset_balanced()
    if params is None:
        params = ABLATION_PARAMS

    results: List[SamplerConfig] = [reference]
    seen_labels: set = set()

    for param_name, values in params:
        ref_val = getattr(reference, param_name)

        for alt_val in values:
            if alt_val == ref_val:
                continue

            kwargs = {
                f.name: getattr(reference, f.name)
                for f in dataclass_fields(SamplerConfig)
            }
            kwargs[param_name] = alt_val
            # Build a label indicating what changed
            label = f"{reference.label} ({param_name}={alt_val})"
            if label in seen_labels:
                continue
            seen_labels.add(label)
            kwargs["label"] = label
            kwargs["description"] = (
                f"Like {reference.label} but {param_name}={alt_val} "
                f"(was {ref_val})"
            )

            results.append(SamplerConfig(**kwargs))

    return results


def dataclass_fields(cls):
    """Return a list of fields for a dataclass (compat helper)."""
    import dataclasses
    return dataclasses.fields(cls)
