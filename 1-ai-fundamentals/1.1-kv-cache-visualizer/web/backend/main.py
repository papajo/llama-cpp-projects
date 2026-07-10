"""
FastAPI web backend for the KV Cache Visualizer.

Provides:
  - GET /api/models          — list known model architectures
  - GET /api/models/{name}   — detailed analysis for a model
  - POST /api/calculate      — custom calculation
  - POST /api/oom            — OOM prediction
  - GET /api/dashboard/{name}— full HTML dashboard
  - GET /                    — serves the React frontend (or fallback)
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from calculator.calculator import KVCacheCalculator
from calculator.models import KVCacheResult, KVQuant, ModelConfig
from calculator.presets import ModelPresets
from viz.plots import build_dashboard


# ── Pydantic request/response models ──────────────────────────────

class CalculateRequest(BaseModel):
    model: str
    max_ctx: Optional[int] = None
    quants: List[str] = Field(default_factory=lambda: ["f16", "q8_0", "q4_0"])
    n_slots: int = 1
    allocation: str = "per-slot"


class OOMRequest(BaseModel):
    model: str
    vram_gb: float
    quants: List[str] = Field(default_factory=lambda: ["f16", "q8_0", "q4_0"])
    n_slots: int = 1
    model_weights_gb: Optional[float] = None


# ── App setup ─────────────────────────────────────────────────────

app = FastAPI(
    title="KV Cache Visualizer",
    description="Interactive KV Cache Memory Calculator for llama.cpp",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────────

def _get_calculator(model_name: str, n_slots: int = 1) -> KVCacheCalculator:
    try:
        cfg = ModelPresets.get(model_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return KVCacheCalculator(cfg, n_slots=n_slots)


def _parse_quants(quant_strs: List[str]) -> List[KVQuant]:
    quants = []
    for s in quant_strs:
        try:
            quants.append(KVQuant.from_str(s))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return quants


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/api/models")
def list_models() -> JSONResponse:
    """List all known model architectures."""
    models = []
    for cfg in ModelPresets.all():
        models.append({
            "name": cfg.name,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "head_dim": cfg.head_dim,
            "max_ctx": cfg.max_ctx,
            "d_model": cfg.d_model,
            "is_moe": cfg.is_moe,
        })
    return JSONResponse({"models": models})


@app.get("/api/models/{name}")
def get_model(name: str, ctx: Optional[int] = Query(None)) -> JSONResponse:
    """Get detailed KV cache analysis for a specific model."""
    calc = _get_calculator(name)
    quants = KVQuant.all()
    max_ctx = ctx or calc.config.max_ctx
    result = calc.calculate(max_ctx=max_ctx, quants=quants)
    return JSONResponse(result.as_dict())


@app.post("/api/calculate")
def calculate(req: CalculateRequest) -> JSONResponse:
    """Custom calculation with specified parameters."""
    calc = _get_calculator(req.model, req.n_slots)
    quants = _parse_quants(req.quants)
    result = calc.calculate(
        max_ctx=req.max_ctx or calc.config.max_ctx,
        quants=quants,
        allocation=req.allocation,
    )
    return JSONResponse(result.as_dict())


@app.post("/api/oom")
def predict_oom(req: OOMRequest) -> JSONResponse:
    """OOM prediction: max context length given VRAM budget."""
    calc = _get_calculator(req.model, req.n_slots)
    quants = _parse_quants(req.quants)
    results = calc.predict_oom(
        vram_gb=req.vram_gb,
        quants=quants,
        model_weights_gb=req.model_weights_gb,
    )
    return JSONResponse({"results": results})


@app.get("/api/dashboard/{name}", response_class=HTMLResponse)
def get_dashboard(name: str) -> str:
    """Full HTML dashboard for a model."""
    calc = _get_calculator(name)
    html = build_dashboard(calc)
    return html


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ── Serve SPA frontend (if built) ─────────────────────────────────

from pathlib import Path
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "build"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    print(f"  → Serving frontend from {FRONTEND_DIR}")
else:
    print("  → Frontend build not found. API-only mode.")
    print(f"    Build the React app at: {FRONTEND_DIR.parent}")
