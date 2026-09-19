"""
Sampling simulation engine.

Models how each llama.cpp sampling parameter transforms logits:

  Temperature  →  softmax( logits / temp )
  Top-K        →  zero out all but top K logits
  Top-P        →  keep smallest set with cumulative prob >= p
  Min-P        →  keep tokens with prob >= min_p * max_prob
  Repeat penalty →  scale seen tokens' logits by penalty
  Frequency penalty →  subtract count * freq from logits
  Presence penalty  →  subtract presence * pres from logits
  Mirostat     →  entropy-based adaptive truncation
  XTC          →  exponential temperature control
  Dynatemp     →  temperature varies in [range_low, range_high]

Supports logit previews from:
  - Synthetic distributions (unimodal, bimodal, flat, sharp, long-tail)
  - Real pre-cached logit data from sample-logs/
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


# ── Data models ───────────────────────────────────────────────────

@dataclass
class SamplerState:
    """The current state of the sampling pipeline."""
    logits: List[float]
    probabilities: List[float]
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    mirostat: int = 0       # 0=off, 1=v1, 2=v2
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    xtc_threshold: float = 0.1
    xtc_probability: float = 0.0
    dynatemp_range: float = 0.0
    dynatemp_exponent: float = 1.0
    sampler_sequence: str = "mirostat,top_k,top_p,min_p,temp,typ"

    # Penalty state
    seen_tokens: Dict[int, int] = field(default_factory=dict)

    # Step-by-step trace
    steps: List[SamplerStep] = field(default_factory=list)


@dataclass
class SamplerStep:
    """Record of what happened at one step of the pipeline."""
    sampler_name: str
    parameters: Dict[str, float]
    logits_before: List[float]
    logits_after: List[float]
    num_pruned: int = 0
    description: str = ""


@dataclass
class DistributionPreset:
    """A named synthetic distribution type."""
    name: str
    description: str
    generator: Callable[[int], List[float]]


# ── Synthetic distribution generators ─────────────────────────────

def _gen_unimodal(vocab_size: int = 100) -> List[float]:
    """Single peak — most common real-world distribution."""
    peak = random.randint(10, vocab_size - 10)
    return [10 * math.exp(-((i - peak) ** 2) / 200) +
            random.gauss(0, 1) for i in range(vocab_size)]


def _gen_bimodal(vocab_size: int = 100) -> List[float]:
    """Two peaks — model is torn between two plausible tokens."""
    peak1 = random.randint(5, vocab_size // 2 - 5)
    peak2 = random.randint(vocab_size // 2 + 5, vocab_size - 5)
    return [8 * math.exp(-((i - peak1) ** 2) / 100) +
            7 * math.exp(-((i - peak2) ** 2) / 120) +
            random.gauss(0, 0.8) for i in range(vocab_size)]


def _gen_flat(vocab_size: int = 100) -> List[float]:
    """Uniform — model is uncertain (all tokens equally likely)."""
    return [random.gauss(0, 2) for _ in range(vocab_size)]


def _gen_sharp(vocab_size: int = 100) -> List[float]:
    """Very sharp peak — model is highly confident."""
    peak = random.randint(5, vocab_size - 5)
    return [30 * math.exp(-((i - peak) ** 4) / 50000) +
            random.gauss(0, 0.3) for i in range(vocab_size)]


def _gen_long_tail(vocab_size: int = 100) -> List[float]:
    """One strong candidate + many weaker ones."""
    peak = random.randint(5, vocab_size - 5)
    logits = [random.gauss(0, 0.5) for _ in range(vocab_size)]
    logits[peak] = 25
    # Add a gentle slope
    for i in range(vocab_size):
        logits[i] += 5 * math.exp(-abs(i - peak) / 15)
    return logits


DISTRIBUTION_PRESETS: List[DistributionPreset] = [
    DistributionPreset(
        "unimodal", "Single peak — most common real distribution", _gen_unimodal,
    ),
    DistributionPreset(
        "bimodal", "Two peaks — model is torn between two tokens", _gen_bimodal,
    ),
    DistributionPreset(
        "flat", "Uniform — model is uncertain, all tokens equally likely", _gen_flat,
    ),
    DistributionPreset(
        "sharp", "Very sharp peak — model is highly confident", _gen_sharp,
    ),
    DistributionPreset(
        "long-tail", "One strong candidate + many weaker ones", _gen_long_tail,
    ),
]


# ── Core math ─────────────────────────────────────────────────────

def softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax."""
    n = len(logits)
    if n == 0:
        return []
    max_l = max(logits)
    if not math.isfinite(max_l):
        # Every logit is -inf (or the input is degenerate). `l - max_l`
        # would be inf-inf = nan, and the `total <= 0` guard below cannot
        # catch nan, so handle it up front.
        return [1.0 / n] * n
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    if total <= 0:
        return [1.0 / n] * n
    return [e / total for e in exps]


def apply_temperature(logits: List[float], temp: float) -> List[float]:
    """Temperature scaling: divide logits by temp."""
    if temp <= 0 or temp > 10:
        return logits
    return [l / temp for l in logits]


def apply_top_k(logits: List[float], k: int) -> List[float]:
    """Zero out all but top-K logits."""
    if k <= 0 or k >= len(logits):
        return logits
    threshold = sorted(logits, reverse=True)[k - 1]
    return [l if l >= threshold else float("-inf") for l in logits]


def apply_top_p(probs: List[float], p: float) -> List[float]:
    """Nucleus sampling: keep smallest set with cumulative prob >= p."""
    if p >= 1.0 or p <= 0:
        return probs
    sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    cumsum = 0.0
    mask = [False] * len(probs)
    for i in sorted_idx:
        mask[i] = True
        cumsum += probs[i]
        if cumsum >= p:
            break
    return [probs[i] if mask[i] else 0.0 for i in range(len(probs))]


def apply_min_p(probs: List[float], min_p: float) -> List[float]:
    """Keep tokens with prob >= min_p * max_prob."""
    if min_p <= 0:
        return probs
    max_prob = max(probs)
    threshold = min_p * max_prob
    return [p if p >= threshold else 0.0 for p in probs]


def apply_repeat_penalty(
    logits: List[float],
    seen_tokens: Dict[int, int],
    penalty: float,
) -> List[float]:
    """Scale logits of previously seen tokens."""
    if penalty == 1.0 or not seen_tokens:
        return logits
    result = list(logits)
    for token_id, _ in seen_tokens.items():
        if 0 <= token_id < len(result):
            if result[token_id] < 0:
                result[token_id] *= penalty
            else:
                result[token_id] /= penalty
    return result


def apply_frequency_penalty(
    logits: List[float],
    seen_tokens: Dict[int, int],
    penalty: float,
) -> List[float]:
    """Subtract count * penalty from logits."""
    if penalty == 0.0 or not seen_tokens:
        return logits
    result = list(logits)
    for token_id, count in seen_tokens.items():
        if 0 <= token_id < len(result):
            result[token_id] -= count * penalty
    return result


def apply_presence_penalty(
    logits: List[float],
    seen_tokens: Dict[int, int],
    penalty: float,
) -> List[float]:
    """Subtract penalty from logits if token has been seen."""
    if penalty == 0.0 or not seen_tokens:
        return logits
    result = list(logits)
    for token_id in seen_tokens:
        if 0 <= token_id < len(result):
            result[token_id] -= penalty
    return result


def apply_typical(probs: List[float], p: float) -> List[float]:
    """Typical sampling: keep tokens with entropy close to expected."""
    if p >= 1.0 or p <= 0:
        return probs
    # Compute entropy of each token
    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
    neg_log_probs = [-math.log(max(p, 1e-10)) for p in probs]
    # Keep tokens whose neg_log_prob is within p of entropy
    result = []
    for i, prob in enumerate(probs):
        if abs(neg_log_probs[i] - entropy) < entropy * (1 - p):
            result.append(prob)
        else:
            result.append(0.0)
    if not any(result):
        # A truncation sampler must never empty the candidate set; real
        # samplers always leave at least the most likely token standing.
        keep = max(range(len(probs)), key=lambda i: probs[i])
        result[keep] = probs[keep]
    return result


# ── Distribution tracking ─────────────────────────────────────────

@dataclass
class DistributionSnapshot:
    """A snapshot of the distribution at a point in the pipeline."""
    step_label: str
    logits: List[float]
    probs: List[float]
    top_tokens: List[Tuple[int, float, float]]  # (id, logit, prob)
    num_zero: int
    entropy: float
    peak_prob: float
    description: str = ""


# ── Pipeline ──────────────────────────────────────────────────────

class SamplingPipeline:
    """
    Runs a sequence of sampling operations on logits and tracks
    how the distribution changes at each step.
    """

    def __init__(self, logits: List[float]):
        self.original_logits = list(logits)
        self.vocab_size = len(logits)

    def run(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        xtc_threshold: float = 0.1,
        xtc_probability: float = 0.0,
        dynatemp_range: float = 0.0,
        dynatemp_exponent: float = 1.0,
        sampler_sequence: str = "top_k,top_p,min_p,temp,typ",
        seen_tokens: Optional[Dict[int, int]] = None,
    ) -> Dict:
        """
        Run the full sampling pipeline and return snapshots at each step.
        
        Returns a dict with:
          - snapshots: list of DistributionSnapshot
          - final_probs: final probability distribution
          - final_logits: final logits
          - summary: dict with final metrics
        """
        logits = list(self.original_logits)
        snapshots: List[DistributionSnapshot] = []
        steps = [s.strip() for s in sampler_sequence.split(",")]

        if seen_tokens is None:
            seen_tokens = {}

        for step_name in steps:
            before = list(logits)
            description = ""

            if step_name == "penalty" or step_name == "repeat":
                logits = apply_repeat_penalty(logits, seen_tokens, repeat_penalty)
                description = f"repeat_penalty={repeat_penalty}"
            elif step_name == "freq" or step_name == "frequency":
                logits = apply_frequency_penalty(logits, seen_tokens, frequency_penalty)
                description = f"frequency_penalty={frequency_penalty}"
            elif step_name == "pres" or step_name == "presence":
                logits = apply_presence_penalty(logits, seen_tokens, presence_penalty)
                description = f"presence_penalty={presence_penalty}"
            elif step_name == "temp" or step_name == "temperature":
                effective_temp = temperature
                if dynatemp_range > 0:
                    effective_temp = random.uniform(
                        max(0.1, temperature - dynatemp_range / 2),
                        temperature + dynatemp_range / 2,
                    )
                    description = (f"temperature={effective_temp:.3f} "
                                   f"(dynatemp range={dynatemp_range})")
                else:
                    effective_temp = temperature
                    description = f"temperature={effective_temp}"
                logits = apply_temperature(logits, effective_temp)
            elif step_name == "top_k":
                logits = apply_top_k(logits, top_k)
                description = f"top_k={top_k}"
            elif step_name == "top_p" or step_name == "nucleus":
                probs = softmax(logits)
                probs = apply_top_p(probs, top_p)
                # Convert back to logits (approximate)
                logits = [math.log(max(p, 1e-10)) if p > 0 else float("-inf")
                         for p in probs]
                description = f"top_p={top_p}"
            elif step_name == "min_p":
                probs = softmax(logits)
                probs = apply_min_p(probs, min_p)
                logits = [math.log(max(p, 1e-10)) if p > 0 else float("-inf")
                         for p in probs]
                description = f"min_p={min_p}"
            elif step_name == "typ" or step_name == "typical":
                probs = softmax(logits)
                probs = apply_typical(probs, top_p)
                logits = [math.log(max(p, 1e-10)) if p > 0 else float("-inf")
                         for p in probs]
                description = f"typical_p={top_p}"

            # Record snapshot
            probs = softmax(logits)
            num_zero = sum(1 for p in probs if p == 0.0)
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            top_indices = sorted(
                range(len(probs)), key=lambda i: probs[i], reverse=True
            )[:5]
            top_tokens = [
                (i, logits[i], probs[i]) for i in top_indices
            ]
            peak_prob = max(probs)

            snapshots.append(DistributionSnapshot(
                step_label=step_name,
                logits=list(logits),
                probs=list(probs),
                top_tokens=top_tokens,
                num_zero=num_zero,
                entropy=entropy,
                peak_prob=peak_prob,
                description=description,
            ))

        # Final distribution
        final_probs = softmax(logits)

        return {
            "snapshots": [
                {
                    "step": s.step_label,
                    "description": s.description,
                    "logits_sample": s.logits[:20],  # first 20 for display
                    "probs_sample": s.probs[:20],
                    "top_tokens": [
                        {"id": t[0], "logit": round(t[1], 4), "prob": round(t[2], 6)}
                        for t in s.top_tokens
                    ],
                    "num_zero": s.num_zero,
                    "entropy": round(s.entropy, 4),
                    "peak_prob": round(s.peak_prob, 6),
                }
                for s in snapshots
            ],
            "final_probs": final_probs[:50],  # first 50 for display
            "final_logits_sample": logits[:20],
            "summary": {
                "vocab_size": self.vocab_size,
                "final_entropy": round(-sum(p * math.log(p + 1e-10)
                                          for p in final_probs), 4),
                "final_peak_prob": round(max(final_probs), 6),
                "final_zero_count": sum(1 for p in final_probs if p == 0.0),
                "num_snapshots": len(snapshots),
            },
        }
