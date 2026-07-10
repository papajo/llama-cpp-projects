"""
FastAPI server for the Sampling Parameter Explorer.

Endpoints:
  GET  /api/presets          — List distribution presets
  GET  /api/samples          — List pre-cached sample logits
  GET  /api/samples/{name}   — Get pre-cached sample logits
  POST /api/simulate         — Run sampling pipeline with given params
  POST /api/compare          — Compare two parameter sets side-by-side
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .sampler import (
    DISTRIBUTION_PRESETS,
    SamplingPipeline,
)
from .sample_logits import get_samples

# ── FastAPI app ────────────────────────────────────────────────────

app = FastAPI(
    title="Sampling Parameter Explorer",
    description="Interactively explore how sampling parameters shape token distributions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-cached samples
_SAMPLES = get_samples()

# ── Pydantic models ────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    """Request to run the sampling pipeline."""
    logits: List[float] = Field(default=[], description="Raw logits")
    distribution: str = Field(default="unimodal", description="Synthetic distribution preset")
    temperature: float = Field(default=1.0, ge=0.01, le=10.0)
    top_k: int = Field(default=0, ge=0, le=100)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repeat_penalty: float = Field(default=1.0, ge=0.0, le=10.0)
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=10.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=10.0)
    mirostat: int = Field(default=0, ge=0, le=2)
    mirostat_tau: float = Field(default=5.0, ge=0.1, le=10.0)
    mirostat_eta: float = Field(default=0.1, ge=0.01, le=1.0)
    dynatemp_range: float = Field(default=0.0, ge=0.0, le=5.0)
    dynatemp_exponent: float = Field(default=1.0, ge=0.1, le=5.0)
    sampler_sequence: str = Field(default="top_k,top_p,min_p,temp,typ")
    seed: int = Field(default=42)

class CompareRequest(BaseModel):
    """Compare two parameter sets."""
    logits: List[float] = Field(default=[])
    distribution: str = Field(default="unimodal")
    params_a: Dict[str, Any]
    params_b: Dict[str, Any]

class SimulateResponse(BaseModel):
    snapshots: List[Dict]
    final_probs: List[float]
    final_logits_sample: List[float]
    summary: Dict
    distribution_name: str = ""
    distribution_description: str = ""

# ── Helper ─────────────────────────────────────────────────────────

def _get_logits(distribution: str, logits_override: List[float]) -> List[float]:
    """Resolve logits from distribution preset, sample, or override."""
    if logits_override:
        return logits_override

    # Check sample logs
    if distribution in _SAMPLES:
        return _SAMPLES[distribution]["logits"]

    # Check synthetic presets
    for preset in DISTRIBUTION_PRESETS:
        if preset.name == distribution:
            import random as rnd
            rnd.seed(42)
            return preset.generator(100)

    raise HTTPException(status_code=400, detail=f"Unknown distribution: {distribution}")

# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/api/presets")
async def list_presets():
    """List available synthetic distribution presets."""
    return {
        "presets": [
            {
                "name": p.name,
                "description": p.description,
            }
            for p in DISTRIBUTION_PRESETS
        ]
    }


@app.get("/api/samples")
async def list_samples():
    """List pre-cached sample logit distributions."""
    return {
        "samples": [
            {
                "name": key,
                "title": s["name"],
                "description": s["description"],
            }
            for key, s in _SAMPLES.items()
        ]
    }


@app.get("/api/samples/{name}")
async def get_sample(name: str):
    """Get a specific pre-cached sample."""
    if name not in _SAMPLES:
        raise HTTPException(status_code=404, detail=f"Sample '{name}' not found")
    return _SAMPLES[name]


@app.post("/api/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest):
    """Run the sampling pipeline and return per-step snapshots."""
    logits = _get_logits(req.distribution, req.logits)

    pipeline = SamplingPipeline(logits)
    result = pipeline.run(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        min_p=req.min_p,
        repeat_penalty=req.repeat_penalty,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        mirostat=req.mirostat,
        mirostat_tau=req.mirostat_tau,
        mirostat_eta=req.mirostat_eta,
        dynatemp_range=req.dynatemp_range,
        dynatemp_exponent=req.dynatemp_exponent,
        sampler_sequence=req.sampler_sequence,
    )

    dist_name = req.distribution
    dist_desc = ""
    for p in DISTRIBUTION_PRESETS:
        if p.name == req.distribution:
            dist_desc = p.description
            break
    if req.distribution in _SAMPLES:
        dist_desc = _SAMPLES[req.distribution]["description"]

    return SimulateResponse(
        snapshots=result["snapshots"],
        final_probs=result["final_probs"],
        final_logits_sample=result["final_logits_sample"],
        summary=result["summary"],
        distribution_name=dist_name,
        distribution_description=dist_desc,
    )


@app.post("/api/compare")
async def compare(req: CompareRequest):
    """Compare two parameter sets side-by-side."""
    logits = _get_logits(req.distribution, req.logits)

    pipeline = SamplingPipeline(logits)

    result_a = pipeline.run(**req.params_a)
    result_b = pipeline.run(**req.params_b)

    return {
        "a": {
            "snapshots": result_a["snapshots"],
            "final_probs": result_a["final_probs"],
            "summary": result_a["summary"],
        },
        "b": {
            "snapshots": result_b["snapshots"],
            "final_probs": result_b["final_probs"],
            "summary": result_b["summary"],
        },
    }


@app.get("/api/presets/parameters")
async def get_parameter_presets():
    """Get common parameter presets (deterministic, creative, balanced, etc.)."""
    return {
        "presets": [
            {
                "name": "deterministic",
                "label": "Deterministic (low temp)",
                "description": "Low temperature, high top-k — for factual tasks",
                "params": {
                    "temperature": 0.1,
                    "top_k": 10,
                    "top_p": 0.9,
                    "min_p": 0.0,
                    "repeat_penalty": 1.1,
                },
            },
            {
                "name": "balanced",
                "label": "Balanced (default)",
                "description": "llama.cpp defaults — good for general use",
                "params": {
                    "temperature": 0.8,
                    "top_k": 40,
                    "top_p": 0.95,
                    "min_p": 0.05,
                    "repeat_penalty": 1.0,
                },
            },
            {
                "name": "creative",
                "label": "Creative (high temp)",
                "description": "Higher temperature for creative writing",
                "params": {
                    "temperature": 1.2,
                    "top_k": 0,
                    "top_p": 0.9,
                    "min_p": 0.1,
                    "repeat_penalty": 1.1,
                },
            },
            {
                "name": "precise",
                "label": "Precise (min-p focused)",
                "description": "Min-P sampling for precise control",
                "params": {
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                    "min_p": 0.2,
                    "repeat_penalty": 1.0,
                },
            },
            {
                "name": "code",
                "label": "Code generation",
                "description": "Low temperature, high repeat penalty for code",
                "params": {
                    "temperature": 0.2,
                    "top_k": 20,
                    "top_p": 0.95,
                    "min_p": 0.0,
                    "repeat_penalty": 1.2,
                },
            },
            {
                "name": "experimental_mirostat",
                "label": "Mirostat v2",
                "description": "Entropy-based sampling for perplexity control",
                "params": {
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                    "min_p": 0.0,
                    "mirostat": 2,
                    "mirostat_tau": 5.0,
                    "mirostat_eta": 0.1,
                },
            },
        ]
    }


# ── Main ───────────────────────────────────────────────────────────

def main():
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8200"))
    print(f"🔧 Sampling Parameter Explorer")
    print(f"   Listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
