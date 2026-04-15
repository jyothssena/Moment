"""
main.py — FastAPI application for the Momento matching pipeline.

Pipeline flow for POST /match:
  1. Check BQ for existing decomposition  →  run decomposer if missing
  2. Check BQ for existing scoring        →  run scorer if missing
  3. aggregate()                          →  log compat run to BQ
  4. Return compatibility result

Ranking flow (background):
  POST /feedback or POST /rankings/{user_id}/refit
  →  load compat runs + comparisons + conversations from BQ
  →  fit Bradley-Terry per user (mirrors run_rankings.py logic)
  →  write ranked results to BQ rankings table

Endpoints:
  POST /match                      — run full pipeline for a user pair
  POST /decompose/{user_id}        — force re-decomposition for a user + passage
  POST /feedback                   — log engagement comparison (feeds BT model)
  GET  /rankings/{user_id}         — retrieve stored ranked results
  POST /rankings/{user_id}/refit   — manually trigger BT rerank

Auth:
  Uses ADC (Application Default Credentials).
  Set GCP_PROJECT and BQ_DATASET env vars before starting.
"""

import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from tools import (
    insert_comparison,
    get_rankings,
    get_moments_for_passage,
)
from compatibility_agent import run_compatibility_agent
from run_rankings import refit_user

app = FastAPI(
    title="Momento Matching API",
    description="Reader compatibility pipeline: decompose → score → aggregate → rank",
    version="1.0.0",
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    user_id:           str
    winner_run_id:     str
    loser_run_id:      str
    session_id:        str
    winner_confidence: Optional[float] = None
    winner_verdict:    Optional[str]   = None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

class NewMomentRequest(BaseModel):
    user_id:        str
    book_id:        str
    passage_id:     str
    interpretation: str

@app.post("/match/batch")
def match_batch(req: NewMomentRequest, background_tasks: BackgroundTasks):
    """
    Called when a user submits a new moment.
    Finds all other users with moments on the same passage and runs
    compatibility against each one in the background.
    Returns immediately with a count of comparisons queued.
    """
    # add preprocessing here 
    background_tasks.add_task(
        _run_batch_compatibility,
        req.user_id, req.book_id, req.passage_id, req.interpretation
    )
    return {
        "status":     "queued",
        "user_id":    req.user_id,
        "passage_id": req.passage_id,
        "book_id":    req.book_id,
    }





@app.post("/feedback")
def feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    """Log an implicit pairwise comparison. Triggers BT rerank in background."""
    record = {
        "comparison_id":     f"cmp_{abs(hash((req.user_id, req.winner_run_id, req.loser_run_id, req.session_id))):08x}",
        "user_id":           req.user_id,
        "session_id":        req.session_id,
        "winner_run_id":     req.winner_run_id,
        "loser_run_id":      req.loser_run_id,
        "winner_confidence": req.winner_confidence,
        "winner_verdict":    req.winner_verdict,
        "timestamp":         datetime.utcnow().isoformat(),
    }
    insert_comparison(record)
    background_tasks.add_task(_refit_and_save_rankings, req.user_id)
    return {"status": "recorded", "comparison_id": record["comparison_id"]}


@app.get("/rankings/{user_id}")
def rankings(
    user_id:    str,
    book_id:    Optional[str] = None,
    passage_id: Optional[str] = None,
    k:          int = 5,
):
    """
    Return top-k ranked matches for a user.
    Always refits BT model first so the UI always gets fresh rankings.
    Refit runs synchronously so results are ready before returning.
    """
    _refit_and_save_rankings(user_id, book_id=book_id, passage_id=passage_id, k=k)

    rows = get_rankings(user_id, book_id=book_id, passage_id=passage_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No rankings found for {user_id}")
    return {"user_id": user_id, "rankings": rows}





# ── Batch compatibility ──────────────────────────────────────────────────────

def _run_batch_compatibility(
    user_id: str,
    book_id: str,
    passage_id: str,
    interpretation: str,
) -> None:
    """
    Run compatibility between user_id and every other user who has a moment
    on the same passage. Delegates entirely to run_compatibility_agent()
    which handles caching, decomposition, scoring, aggregation and BQ logging.
    """
    print(f"[Batch] finding matches for {user_id} on {book_id}/{passage_id}")

    other_moments = get_moments_for_passage(book_id, passage_id, exclude_user_id=user_id)
    if not other_moments:
        print(f"[Batch] no other users found for {book_id}/{passage_id}")
        return

    print(f"[Batch] running {len(other_moments)} comparisons")
    results = []
    for other in other_moments:
        other_user_id = str(other["user_id"])
        other_text    = other.get("cleaned_interpretation", "")
        if not other_text:
            continue

        try:
            result = run_compatibility_agent(
                user_a_id=user_id,
                user_b_id=other_user_id,
                book_id=book_id,
                moment_a={"passage_id": passage_id, "interpretation": interpretation},
                moment_b={"passage_id": passage_id, "cleaned_interpretation": other_text},
            )
            if "error" not in result:
                results.append(result)
                print(f"[Batch] {user_id} × {other_user_id}: {result.get('dominant_think')} ({result.get('confidence')})")
            else:
                print(f"[Batch] error for {other_user_id}: {result['error']}")
        except Exception as e:
            print(f"[Batch] exception for {other_user_id}: {e}")
            continue

    print(f"[Batch] complete — {len(results)} new compat runs logged")


# ── Background refit ──────────────────────────────────────────────────────────

def _refit_and_save_rankings(
    user_id: str,
    book_id: Optional[str]    = None,
    passage_id: Optional[str] = None,
    k: int = 5,
) -> None:
    """Thin wrapper — delegates all BT logic to run_rankings.refit_user()."""
    refit_user(user_id, book_id=book_id, passage_id=passage_id, k=k)