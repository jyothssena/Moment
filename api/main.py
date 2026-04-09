"""
api/main.py
===========
FastAPI wrapper for the Moment compatibility pipeline.

MODEL_MODE env var controls which backend is used:
  MODEL_MODE=stub  (default) — uses model_interface_stub.py
                               no Gemini key required, all endpoints work
  MODEL_MODE=real            — uses model_interface.py (real Gemini agents)
                               requires GEMINI_API_KEY_MOMENT in environment

Endpoints:
  GET  /health                     — liveness + readiness probe
  GET  /info                       — version, mode, config
  POST /decompose                  — decompose one reader moment into sub-claims
  POST /compatibility              — score one user pair on one passage
  POST /batch                      — score anchor user vs all users on a passage
  POST /validate                   — run validate_model.py on a results list
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import time
from fastapi.responses import Response

try:
    from api.metrics import (
        setup_metrics, track_request, track_model_call,
        track_model_error, track_confidence, track_verdict,
        get_metrics_response, ACTIVE_REQUESTS,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("moment.api")

# ── sys.path — resolve Moment root regardless of how uvicorn was launched ─────
# Works for: `uvicorn api.main:app`, `python api/main.py`, Docker, Kubernetes
_HERE = os.path.abspath(__file__)       # .../Moment/api/main.py
_API  = os.path.dirname(_HERE)          # .../Moment/api/
_ROOT = os.path.dirname(_API)           # .../Moment/  ← validate_model.py lives here
for _p in (_ROOT, _API, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── validate_model — import at startup so path errors surface immediately ─────
try:
    from validate_model import run_validation_gate
    _VALIDATE_AVAILABLE = True
except ImportError as _e:
    logger.warning(f"validate_model not importable: {_e} — /validate will return 503")
    _VALIDATE_AVAILABLE = False

# ── Model backend selection ───────────────────────────────────────────────────

MODEL_MODE = os.environ.get("MODEL_MODE", "stub").lower()

if MODEL_MODE == "real":
    try:
        from model_interface import (
            decompose_moment,
            run_compatibility_pipeline,
            run_batch_compatibility,
            health_check as _model_health_check,
        )
        logger.info("Model mode: REAL (using model_interface.py + Gemini agents)")
    except Exception as exc:
        logger.error(f"Failed to import real model_interface: {exc}")
        logger.warning("Falling back to STUB mode")
        MODEL_MODE = "stub"

if MODEL_MODE == "stub":
    from api.model_interface_stub import (
        decompose_moment,
        run_compatibility_pipeline,
        run_batch_compatibility,
        health_check as _model_health_check,
    )
    logger.info("Model mode: STUB (model_interface_stub.py — no Gemini key required)")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Moment Compatibility API",
    description=(
        "Compatibility scoring API for the Moment reading platform. "
        "Matches readers based on how they interpret literary passages."
    ),
    version="1.0.0",
)
if _METRICS_AVAILABLE:
    setup_metrics(
        model_mode=MODEL_MODE,
        git_sha=os.environ.get("GIT_SHA", "local"),
    )
@app.middleware("http")
async def metrics_middleware(request, call_next):
    if not _METRICS_AVAILABLE:
        return await call_next(request)
    ACTIVE_REQUESTS.inc()
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    ACTIVE_REQUESTS.dec()
    track_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=duration,
    )
    return response

_start_time = datetime.now(timezone.utc)

# ── Request / Response models ─────────────────────────────────────────────────

class DecomposeRequest(BaseModel):
    user_id:      str
    passage_id:   str
    book_id:      str = ""
    moment_text:  str
    word_count:   int = Field(default=0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "user_id":     "user_emma_chen_a1b2c3d4",
                "passage_id":  "gutenberg_84_passage_1",
                "book_id":     "gutenberg_84",
                "moment_text": "Victor's immediate reaction of horror reveals his inability to take responsibility for what he created. The yellow eye feels like a symbol of everything he refused to confront.",
                "word_count":  30,
            }
        }


class CompatibilityRequest(BaseModel):
    user_a:     str
    user_b:     str
    book:       str
    passage_id: str
    moment_a:   dict
    moment_b:   dict

    class Config:
        json_schema_extra = {
            "example": {
                "user_a":     "user_emma_chen_a1b2c3d4",
                "user_b":     "user_marcus_williams_b2c3d4e5",
                "book":       "Frankenstein",
                "passage_id": "gutenberg_84_passage_1",
                "moment_a":   {"interpretation": "Victor refuses to see the creature as his responsibility.", "passage_id": "gutenberg_84_passage_1"},
                "moment_b":   {"interpretation": "The creature's yellow eye is the moment innocence turns to horror.", "passage_id": "gutenberg_84_passage_1"},
            }
        }


class BatchRequest(BaseModel):
    user_a_id:   str
    book_id:     str
    passage_id:  str
    moments_map: dict  # {user_id: moment_dict}

    class Config:
        json_schema_extra = {
            "example": {
                "user_a_id":   "user_emma_chen_a1b2c3d4",
                "book_id":     "gutenberg_84",
                "passage_id":  "gutenberg_84_passage_1",
                "moments_map": {
                    "user_emma_chen_a1b2c3d4":    {"interpretation": "Victor refuses responsibility.", "passage_id": "gutenberg_84_passage_1"},
                    "user_marcus_williams_b2c3":  {"interpretation": "The yellow eye is the turning point.", "passage_id": "gutenberg_84_passage_1"},
                },
            }
        }


class ValidateRequest(BaseModel):
    results: list  # list of compatibility result dicts

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    """
    Liveness + readiness probe.
    Returns 200 if the API is up and the model backend is responsive.
    Kubernetes uses this for both liveness and readiness probes.
    """
    model_status = _model_health_check()
    uptime_seconds = (datetime.now(timezone.utc) - _start_time).total_seconds()

    return {
        "status":        "ok",
        "mode":          MODEL_MODE,
        "uptime_seconds": round(uptime_seconds, 1),
        "model":         model_status,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


@app.get("/info", tags=["ops"])
def info():
    """Version, configuration, and environment info."""
    return {
        "service":   "moment-compatibility-api",
        "version":   "1.0.0",
        "mode":      MODEL_MODE,
        "project":   os.environ.get("GCP_PROJECT_ID", "moment-486719"),
        "region":    os.environ.get("GCP_REGION", "us-central1"),
        "started_at": _start_time.isoformat(),
    }


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics():
    if not _METRICS_AVAILABLE:
        return {"error": "prometheus_client not installed"}
    body, content_type = get_metrics_response()
    return Response(content=body, media_type=content_type)


@app.post("/decompose", tags=["model"])
def decompose(req: DecomposeRequest):
    """
    Decompose a single reader moment into weighted sub-claims with emotional modes.
    Called when a user writes or updates a moment in the app.
    """
    logger.info(f"[decompose] user={req.user_id} passage={req.passage_id}")
    try:
        result = decompose_moment(
            passage_id=req.passage_id,
            user_id=req.user_id,
            moment_text=req.moment_text,
            word_count=req.word_count,
            book_id=req.book_id,
        )
    except Exception as exc:
        if _METRICS_AVAILABLE:
            track_model_error("decompose", type(exc).__name__)
        logger.error(f"[decompose] failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    if _METRICS_AVAILABLE:
        track_model_call("decompose")

    return result


@app.post("/compatibility", tags=["model"])
def compatibility(req: CompatibilityRequest):
    """
    Score compatibility between two readers on one passage.
    Returns R/C/D percentages for both Think and Feel dimensions, plus a confidence score.
    """
    logger.info(f"[compatibility] {req.user_a} × {req.user_b} | {req.book} / {req.passage_id}")
    try:
        result = run_compatibility_pipeline(
            user_a=req.user_a,
            user_b=req.user_b,
            book=req.book,
            passage_id=req.passage_id,
            moment_a=req.moment_a,
            moment_b=req.moment_b,
        )
    except Exception as exc:
        logger.error(f"[compatibility] failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    if _METRICS_AVAILABLE:
        track_model_call("compatibility")
        if "confidence" in result:
            track_confidence(result["confidence"])
        if "dominant_think" in result:
            track_verdict(result["dominant_think"])

    return result


@app.post("/batch", tags=["model"])
@app.post("/batch", tags=["model"])
def batch(req: BatchRequest):
    """
    Score an anchor user against all other users who interpreted the same passage.
    Returns a list sorted by confidence descending.
    """
    logger.info(
        f"[batch] anchor={req.user_a_id} book={req.book_id} passage={req.passage_id} "
        f"users={len(req.moments_map)}"
    )
    try:
        results = run_batch_compatibility(
            user_a_id=req.user_a_id,
            book_id=req.book_id,
            passage_id=req.passage_id,
            moments_map=req.moments_map,
        )
    except Exception as exc:
        logger.error(f"[batch] failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return {"results": results, "count": len(results)}


@app.post("/validate", tags=["ops"])
def validate(req: ValidateRequest):
    """
    Run validate_model.py thresholds against a list of compatibility results.
    Used by the CI pipeline to gate deployment — also callable directly for testing.
    """
    if not _VALIDATE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="validate_model not importable — check server sys.path",
        )
    try:
        gate = run_validation_gate(req.results)
    except Exception as exc:
        logger.error(f"[validate] failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    status_code = 200 if gate.get("passed") else 422
    return JSONResponse(content=gate, status_code=status_code)


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


# ── Local dev entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
