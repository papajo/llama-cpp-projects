"""
Speculative decoding strategy configurations.

llama.cpp exposes 8 distinct speculative decoding strategies
across two families:

  Draft-model-based:
    1. draft-simple     — standard draft model (e.g. a small LM)
    2. draft-eagle3     — EAGLE-3 style (draft model conditions on
                           hidden states from the target)
    3. draft-mtp        — Multi-Token Prediction (MTP) draft head

  N-gram-based (no separate model needed):
    4. ngram-simple     — standard n-gram lookup from the prompt
    5. ngram-map-k      — n-gram with a map of length K
    6. ngram-map-k4v    — n-gram with K4-Variant map
    7. ngram-mod        — modular n-gram
    8. ngram-cache      — n-gram cache from prior generations

Each strategy has tunable hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SpecStrategy(Enum):
    """The eight speculative decoding strategies in llama.cpp."""

    DRAFT_SIMPLE = "draft-simple"
    DRAFT_EAGLE3 = "draft-eagle3"
    DRAFT_MTP = "draft-mtp"
    NGRAM_SIMPLE = "ngram-simple"
    NGRAM_MAP_K = "ngram-map-k"
    NGRAM_MAP_K4V = "ngram-map-k4v"
    NGRAM_MOD = "ngram-mod"
    NGRAM_CACHE = "ngram-cache"

    @classmethod
    def draft_based(cls) -> list[SpecStrategy]:
        return [cls.DRAFT_SIMPLE, cls.DRAFT_EAGLE3, cls.DRAFT_MTP]

    @classmethod
    def ngram_based(cls) -> list[SpecStrategy]:
        return [cls.NGRAM_SIMPLE, cls.NGRAM_MAP_K, cls.NGRAM_MAP_K4V,
                cls.NGRAM_MOD, cls.NGRAM_CACHE]

    @classmethod
    def all(cls) -> list[SpecStrategy]:
        return cls.draft_based() + cls.ngram_based()

    def family(self) -> str:
        return "draft-model" if self in self.draft_based() else "ngram"

    def requires_draft_model(self) -> bool:
        return self.family() == "draft-model"


@dataclass
class StrategyConfig:
    """
    Full configuration for one speculative decoding strategy run.

    Combines the strategy type, llama-server CLI flags it maps to,
    and the hyperparameter sweep space.
    """
    strategy: SpecStrategy
    label: str                          # short human label for plots
    description: str                    # what this strategy does

    # llama-server CLI flags this config maps to
    cli_flags: Dict[str, str] = field(default_factory=dict)

    # Default hyperparameters
    # --spec-draft-n-max: max draft tokens per iteration
    draft_n_max: int = 5
    # --spec-draft-n-min: min draft tokens before verification
    draft_n_min: int = 2
    # --spec-draft-p-split: probability split for draft vs verify
    draft_p_split: float = 0.5
    # --spec-draft-p-min: min acceptance probability
    draft_p_min: float = 0.05
    # --spec-ngram-*-size-n: n-gram order N
    ngram_n: int = 4
    # --spec-ngram-*-size-m: n-gram order M
    ngram_m: int = 8
    # --spec-ngram-*-min-hits: min hits before using n-gram
    ngram_min_hits: int = 1
    # --spec-draft-backend-sampling: use backend sampling for draft
    draft_backend_sampling: bool = False

    @property
    def flag_key(self) -> str:
        """Return the --spec-type value for llama-server."""
        return self.strategy.value

    def to_llama_server_flags(self) -> List[str]:
        """Convert this config to a list of llama-server CLI flags."""
        flags = [f"--spec-type", self.strategy.value]

        for flag, value in self.cli_flags.items():
            flags.extend([flag, value])

        # Draft-model parameters
        if self.strategy.requires_draft_model():
            flags.extend([
                "--spec-draft-n-max", str(self.draft_n_max),
                "--spec-draft-n-min", str(self.draft_n_min),
                "--spec-draft-p-split", str(self.draft_p_split),
                "--spec-draft-p-min", str(self.draft_p_min),
            ])
            if self.draft_backend_sampling:
                flags.append("--spec-draft-backend-sampling")

        # N-gram parameters
        else:
            flags.extend([
                f"--spec-ngram-{self._ngram_flag()}-size-n", str(self.ngram_n),
                f"--spec-ngram-{self._ngram_flag()}-size-m", str(self.ngram_m),
                f"--spec-ngram-{self._ngram_flag()}-min-hits", str(self.ngram_min_hits),
            ])

        return flags

    def _ngram_flag(self) -> str:
        """Return the ngram variant suffix for CLI flags."""
        mapping = {
            SpecStrategy.NGRAM_SIMPLE: "simple",
            SpecStrategy.NGRAM_MAP_K: "map-k",
            SpecStrategy.NGRAM_MAP_K4V: "map-k4v",
            SpecStrategy.NGRAM_MOD: "mod",
            SpecStrategy.NGRAM_CACHE: "cache",
        }
        return mapping.get(self.strategy, "simple")

    def param_summary(self) -> str:
        """Human-readable summary of the active parameters."""
        parts = [f"n_max={self.draft_n_max}"]
        if self.strategy.requires_draft_model():
            parts.append(f"p_split={self.draft_p_split}")
            parts.append(f"p_min={self.draft_p_min}")
        else:
            parts.append(f"n={self.ngram_n}")
            parts.append(f"m={self.ngram_m}")
            parts.append(f"min_hits={self.ngram_min_hits}")
        return ", ".join(parts)


# ── Pre-built strategy configs ────────────────────────────────────

def _draft_config(strategy: SpecStrategy, label: str, desc: str,
                  **overrides) -> StrategyConfig:
    """Helper to create a draft-model strategy config with sensible defaults."""
    base = StrategyConfig(
        strategy=strategy,
        label=label,
        description=desc,
        draft_n_max=5,
        draft_n_min=2,
        draft_p_split=0.5,
        draft_p_min=0.05,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _ngram_config(strategy: SpecStrategy, label: str, desc: str,
                  **overrides) -> StrategyConfig:
    """Helper to create an n-gram strategy config with sensible defaults."""
    base = StrategyConfig(
        strategy=strategy,
        label=label,
        description=desc,
        ngram_n=4,
        ngram_m=8,
        ngram_min_hits=1,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


ALL_STRATEGIES: List[StrategyConfig] = [
    # ── Draft-model strategies ──
    _draft_config(
        SpecStrategy.DRAFT_SIMPLE,
        "Draft-Simple",
        "Standard draft model. A smaller LM generates candidate tokens; "
        "the target model verifies them in a single forward pass.",
    ),
    _draft_config(
        SpecStrategy.DRAFT_EAGLE3,
        "Draft-EAGLE3",
        "EAGLE-3 style draft. The draft model conditions on the target "
        "model's hidden states for higher acceptance rates. Requires "
        "a compatible EAGLE checkpoint.",
        draft_n_max=8,      # EAGLE benefits from longer drafts
    ),
    _draft_config(
        SpecStrategy.DRAFT_MTP,
        "Draft-MTP",
        "Multi-Token Prediction draft. Uses an MTP head trained to "
        "predict multiple future tokens simultaneously.",
        draft_n_max=6,
    ),

    # ── N-gram strategies ──
    _ngram_config(
        SpecStrategy.NGRAM_SIMPLE,
        "Ngram-Simple",
        "Standard n-gram lookup from the prompt context. Fast and "
        "effective for repetitive/textual patterns.",
        ngram_n=4,
        ngram_m=10,
    ),
    _ngram_config(
        SpecStrategy.NGRAM_MAP_K,
        "Ngram-Map-K",
        "N-gram with map of length K. Maintains a map of recently seen "
        "n-grams for faster lookups on repeated patterns.",
        ngram_n=4,
        ngram_m=8,
    ),
    _ngram_config(
        SpecStrategy.NGRAM_MAP_K4V,
        "Ngram-Map-K4V",
        "N-gram with K4-Variant map. Uses a 4-byte key-value map for "
        "compact representation of the n-gram table.",
        ngram_n=4,
        ngram_m=8,
    ),
    _ngram_config(
        SpecStrategy.NGRAM_MOD,
        "Ngram-Mod",
        "Modular n-gram. Uses modular arithmetic for n-gram indexing, "
        "offering a different tradeoff in lookup speed vs hit rate.",
        ngram_n=4,
        ngram_m=8,
    ),
    _ngram_config(
        SpecStrategy.NGRAM_CACHE,
        "Ngram-Cache",
        "N-gram cache from prior generations. Persists n-grams across "
        "generations in the same session, improving with repeated calls.",
        ngram_n=4,
        ngram_m=8,
        ngram_min_hits=2,   # require 2 hits before using cached ngram
    ),
]


def get_strategy(label: str) -> Optional[StrategyConfig]:
    """Look up a strategy config by its label or strategy value."""
    label_lower = label.lower()
    for s in ALL_STRATEGIES:
        if s.label.lower() == label_lower or s.strategy.value == label_lower:
            return s
    return None


def strategies_by_family() -> Dict[str, List[StrategyConfig]]:
    """Group strategies by family."""
    families: Dict[str, List[StrategyConfig]] = {}
    for s in ALL_STRATEGIES:
        f = s.strategy.family()
        families.setdefault(f, []).append(s)
    return families
