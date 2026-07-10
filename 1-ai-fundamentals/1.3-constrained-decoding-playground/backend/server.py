"""
FastAPI server — proxy to llama-server with token-level streaming.

Endpoints:
  POST /api/completion         — Stream a constrained generation
  POST /api/convert-schema     — Convert JSON Schema → GBNF grammar
  POST /api/validate-grammar   — Validate GBNF grammar text
  POST /api/grammar-info       — Parse and explain a GBNF grammar
  GET  /api/sample-schemas     — List available sample schemas
  GET  /api/health             — Health check + llama-server status
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .grammar import GBNFParser, GrammarDebugger, GrammarMaskStep, TokenTraceResult
from .schema_converter import SchemaConverter

# ── Config ─────────────────────────────────────────────────────────

LLAMA_SERVER_URL = os.getenv(
    "LLAMA_SERVER_URL", "http://127.0.0.1:8080"
)
LLAMA_SERVER_BINARY = os.getenv(
    "LLAMA_SERVER_BINARY", "llama-server"
)
LLAMA_MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH", ""
)
SAMPLE_SCHEMAS_DIR = Path(__file__).parent.parent / "sample-schemas"

logger = logging.getLogger("constrained-decoding")
logging.basicConfig(level=logging.INFO)

# ── FastAPI app ────────────────────────────────────────────────────

app = FastAPI(
    title="Constrained Decoding Playground",
    description="Visualise how grammar constraints affect token generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    """Request to proxied /completion endpoint."""
    prompt: str = Field(default="", description="Input prompt")
    grammar: str = Field(default="", description="GBNF grammar text")
    json_schema: str = Field(default="", description="JSON Schema (alternative to grammar)")
    n_predict: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    n_probs: int = Field(default=0, description="Top-K probabilities per token (0=disable)")
    stream: bool = Field(default=False, description="Stream tokens via SSE")
    seed: int = Field(default=42)
    cache_prompt: bool = Field(default=True)
    slot_id: int = Field(default=-1)

class CompletionResponse(BaseModel):
    """Response from /completion."""
    content: str = ""
    timings: Dict[str, float] = {}
    tokens_generated: int = 0
    grammar_used: str = ""
    json_schema_used: str = ""

class SchemaConversionRequest(BaseModel):
    schema_text: str = Field(..., description="JSON Schema text")

class SchemaConversionResponse(BaseModel):
    grammar: str = ""
    errors: List[str] = []

class GrammarValidationRequest(BaseModel):
    grammar_text: str = Field(..., description="GBNF grammar to validate")

class GrammarValidationResponse(BaseModel):
    valid: bool = False
    errors: List[str] = []
    rules: List[Dict[str, Any]] = []

class GrammarInfoRequest(BaseModel):
    grammar_text: str = Field(..., description="GBNF grammar to analyse")

class GrammarInfoResponse(BaseModel):
    rules: List[Dict[str, Any]] = []
    num_rules: int = 0
    has_root: bool = False

class TraceRequest(BaseModel):
    """Analyse a completed response for grammar masking."""
    completion_data: Dict[str, Any]
    grammar_text: str = ""
    json_schema: str = ""

class TraceResponse(BaseModel):
    trace: Dict[str, Any] = {}
    total_tokens: int = 0
    tokens_masked: int = 0
    mask_rate: float = 0.0

class HealthResponse(BaseModel):
    status: str = "ok"
    llama_server: str = "unknown"
    llama_server_connected: bool = False

# ── llama-server proxy ─────────────────────────────────────────────

async def _call_llama_completion(
    payload: dict,
) -> dict:
    """Call llama-server /completion endpoint."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{LLAMA_SERVER_URL}/completion",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to llama-server at {LLAMA_SERVER_URL}. "
                       "Start it with: llama-server -m <model> --host 0.0.0.0",
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail="llama-server request timed out",
            )


async def _stream_llama_completion(
    payload: dict,
):
    """Stream tokens from llama-server /completion endpoint."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            async with client.stream(
                "POST",
                f"{LLAMA_SERVER_URL}/completion",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield f"data: {line}\n\n"
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot connect to llama-server'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def _llama_health() -> bool:
    """Check if llama-server is reachable."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{LLAMA_SERVER_URL}/health")
            return response.status_code == 200
        except Exception:
            return False


def _resolve_grammar(request: CompletionRequest) -> str:
    """Determine effective grammar to use."""
    if request.json_schema:
        try:
            converter = SchemaConverter()
            return converter.convert_from_json(request.json_schema)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON Schema: {e}",
            )
    return request.grammar


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    connected = await _llama_health()
    return HealthResponse(
        status="ok",
        llama_server=LLAMA_SERVER_URL,
        llama_server_connected=connected,
    )


@app.post("/api/completion")
async def completion(request: CompletionRequest):
    """
    Proxy a completion request to llama-server with optional grammar.
    
    If stream=True, returns SSE stream of tokens.
    If n_probs > 0 and stream=False, includes per-token probability data.
    """
    grammar = _resolve_grammar(request)

    payload = {
        "prompt": request.prompt,
        "n_predict": request.n_predict,
        "temperature": request.temperature,
        "top_k": request.top_k,
        "top_p": request.top_p,
        "seed": request.seed,
        "cache_prompt": request.cache_prompt,
        "slot_id": request.slot_id,
    }

    if grammar:
        payload["grammar"] = grammar

    if request.n_probs > 0 and not request.stream:
        payload["n_probs"] = request.n_probs

    if request.stream:
        return StreamingResponse(
            _stream_llama_completion(payload),
            media_type="text/event-stream",
        )

    result = await _call_llama_completion(payload)
    tokens = result.get("tokens_generated", 0)
    probs = result.get("completion_probabilities", [])

    return {
        "content": result.get("content", ""),
        "tokens_generated": tokens,
        "timings": result.get("timings", {}),
        "grammar_used": grammar,
        "json_schema_used": request.json_schema,
        "completion_probabilities": probs if probs else None,
    }


@app.post("/api/convert-schema",
          response_model=SchemaConversionResponse)
async def convert_schema(req: SchemaConversionRequest):
    """Convert JSON Schema to GBNF grammar."""
    try:
        converter = SchemaConverter()
        grammar = converter.convert_from_json(req.schema_text)
        return SchemaConversionResponse(grammar=grammar, errors=[])
    except json.JSONDecodeError as e:
        return SchemaConversionResponse(grammar="", errors=[str(e)])
    except Exception as e:
        return SchemaConversionResponse(grammar="", errors=[str(e)])


@app.post("/api/validate-grammar",
          response_model=GrammarValidationResponse)
async def validate_grammar(req: GrammarValidationRequest):
    """Validate GBNF grammar syntax."""
    errors = GBNFParser.validate(req.grammar_text)
    rules = GBNFParser.parse(req.grammar_text)
    return GrammarValidationResponse(
        valid=len(errors) == 0,
        errors=errors,
        rules=[
            {"name": r.name, "summary": r.summary(), "is_root": r.is_root}
            for r in rules
        ],
    )


@app.post("/api/grammar-info", response_model=GrammarInfoResponse)
async def grammar_info(req: GrammarInfoRequest):
    """Parse and display grammar rules."""
    rules = GBNFParser.parse(req.grammar_text)
    return GrammarInfoResponse(
        rules=[
            {"name": r.name, "definition": r.definition, "is_root": r.is_root}
            for r in rules
        ],
        num_rules=len(rules),
        has_root=any(r.is_root for r in rules),
    )


@app.post("/api/trace", response_model=TraceResponse)
async def trace_completion(req: TraceRequest):
    """Analyse a completion to show grammar masking effects."""
    result = GrammarDebugger.analyse_completion(
        req.completion_data,
        grammar_text=req.grammar_text,
        json_schema=req.json_schema,
    )
    return TraceResponse(
        trace=result.as_dict(),
        total_tokens=result.total_tokens,
        tokens_masked=result.tokens_masked,
        mask_rate=result.mask_rate,
    )


@app.get("/api/sample-schemas")
async def list_sample_schemas():
    """List available sample schema files."""
    schemas = []
    if SAMPLE_SCHEMAS_DIR.exists():
        for f in sorted(SAMPLE_SCHEMAS_DIR.iterdir()):
            if f.suffix in (".json", ".gbnf"):
                schemas.append({
                    "name": f.stem,
                    "type": f.suffix.lstrip("."),
                    "path": f.name,
                })
    return {"schemas": schemas}


@app.get("/api/sample-schema/{name}")
async def get_sample_schema(name: str):
    """Get a sample schema file content."""
    for ext in (".json", ".gbnf"):
        path = SAMPLE_SCHEMAS_DIR / f"{name}{ext}"
        if path.exists():
            return {
                "name": name,
                "type": ext.lstrip("."),
                "content": path.read_text(),
            }
    raise HTTPException(status_code=404, detail=f"Schema '{name}' not found")


# ── Main ───────────────────────────────────────────────────────────

def main():
    """Run the server with uvicorn."""
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8100"))
    print(f"🔧 Constrained Decoding Playground backend")
    print(f"   Listening on http://{host}:{port}")
    print(f"   llama-server: {LLAMA_SERVER_URL}")
    print(f"   Sample schemas: {SAMPLE_SCHEMAS_DIR}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
