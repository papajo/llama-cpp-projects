"""
Pre-cached sample logit data from real distributions.

These represent typical token probability distributions seen in
real LLM inference across different contexts. Useful for the
interactive explorer when a live model is not available.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List


# ── Seed for reproducibility ──────────────────────────────────────

_random = random.Random(42)


# ── Sample logit generators ───────────────────────────────────────

def _gaussian_peak(
    center: float,
    width: float,
    height: float,
    vocab_size: int = 100,
    noise: float = 0.5,
) -> List[float]:
    """Generate logits with a gaussian peak."""
    return [
        height * math.exp(-((i - center) ** 2) / (2 * width ** 2)) +
        _random.gauss(0, noise)
        for i in range(vocab_size)
    ]


def generate_sample_logits() -> Dict[str, Dict]:
    """Generate a set of sample logit distributions."""
    samples = {}

    # 1. Next-token prediction (low entropy — model is confident)
    samples["next_token_confident"] = {
        "name": "Confident next-token prediction",
        "description": "Model is very sure about the next token (e.g. 'The capital of France is ___')",
        "logits": _gaussian_peak(42, 2, 25, noise=0.3),
    }

    # 2. Mid-generation (moderate entropy)
    samples["mid_generation"] = {
        "name": "Mid-generation uncertainty",
        "description": "Several plausible continuations (e.g. story generation)",
        "logits": _gaussian_peak(30, 8, 15, noise=1.0),
    }

    # 3. Highly uncertain (high entropy)
    samples["highly_uncertain"] = {
        "name": "Highly uncertain / creative mode",
        "description": "Many equally plausible tokens (e.g. brainstorming)",
        "logits": _gaussian_peak(50, 25, 5, noise=2.0),
    }

    # 4. Bimodal — two competing choices
    peak1 = 25
    peak2 = 65
    samples["bimodal"] = {
        "name": "Bimodal — two competing options",
        "description": "Model is split between two plausible tokens (e.g. 'He ___ to the store' → went/runs)",
        "logits": [
            12 * math.exp(-((i - peak1) ** 2) / 80) +
            10 * math.exp(-((i - peak2) ** 2) / 100) +
            _random.gauss(0, 0.8)
            for i in range(100)
        ],
    }

    # 5. Flat / uniform
    samples["flat"] = {
        "name": "Flat / uniform distribution",
        "description": "All tokens roughly equally likely (model has no preference)",
        "logits": [_random.gauss(0, 1.5) for _ in range(100)],
    }

    # 6. Long-tail
    samples["long_tail"] = {
        "name": "Long-tail distribution",
        "description": "One strong candidate with many weaker ones falling off slowly",
        "logits": [
            20 * math.exp(-abs(i - 15) / 30) + _random.gauss(0, 0.5)
            for i in range(100)
        ],
    }

    # 7. Code generation (very sharp peaks)
    samples["code_generation"] = {
        "name": "Code generation",
        "description": "Very sharp peaks — model is highly confident about syntax tokens",
        "logits": _gaussian_peak(10, 1.5, 35, noise=0.2),
    }

    # 8. Repetition (model is stuck in a loop)
    samples["repetition"] = {
        "name": "Repetition pattern",
        "description": "Model repeating the same token pattern (high peak at recently seen tokens)",
        "logits": _gaussian_peak(8, 1, 20, noise=0.5),
    }

    return samples


# ── Pre-cached data ───────────────────────────────────────────────

def get_samples() -> Dict:
    """Return the pre-cached sample logits."""
    return generate_sample_logits()


# ── Export to JSON ────────────────────────────────────────────────

def export_samples(output_path: str = "sample-logs/samples.json"):
    """Export sample logits to a JSON file."""
    samples = get_samples()
    # Serialize logits as lists
    data = {}
    for key, sample in samples.items():
        data[key] = {
            "name": sample["name"],
            "description": sample["description"],
            "logits": [round(l, 6) for l in sample["logits"]],
        }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return str(path)
