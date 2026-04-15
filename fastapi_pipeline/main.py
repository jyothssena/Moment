"""
main.py — MOMENT FastAPI Data Pipeline
=======================================
Daily pipeline — processes only today's moments:
  POST /pipeline/run
    1. Cloud SQL → fetch moments where DATE(created_at) = CURRENT_DATE
    2. Preprocess → write to BQ (moments, passages, books, users)  [synchronous]
    3. For each valid moment → _run_batch_compatibility() in background
       (matches against existing BQ users only, not the incoming batch)
    4. Rankings → refit BT model per processed user → write to BQ (background)

Other endpoints:
  POST /feedback                   — log pairwise comparison (feeds BT model)
  GET  /rankings/{user_id}         — retrieve stored ranked results
  GET  /health
  GET  /pipeline/status
"""

import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from compatibility_agent import run_compatibility_agent
from cloudsql_loader import CloudSQLLoader
from preprocessor_fastapi import preprocess_all
from bq_writer import write_to_bq
from tools import get_moments_for_passage, insert_comparison, get_rankings
from run_rankings import refit_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MOMENT Data Pipeline",
    description="Daily: Cloud SQL (today's moments) → Preprocess → BQ → Compatibility (background) → Rankings (background)",
    version="6.0.0",
)

_last_run: dict = {}


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


@app.post("/pipeline/run")
def run_pipeline(background_tasks: BackgroundTasks):
    """
    Daily pipeline — triggered manually or via Cloud Scheduler.
    Steps 1–3 (load, preprocess, BQ write) run synchronously.
    Compatibility and rankings run in the background per valid moment.
    """
    global _last_run
    start = datetime.utcnow()
    logger.info("=" * 60)
    logger.info(f"Pipeline run started at {start.isoformat()}")

    # ── Step 1: Load from Cloud SQL ───────────────────────────────
    logger.info("Step 1/3: Loading today's moments from Cloud SQL...")
    try:
        loader = CloudSQLLoader()
        loader.run()
        dfs = loader.get_dataframes()
    except Exception as e:
        logger.error(f"Cloud SQL load failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cloud SQL load failed: {str(e)}")

    new_count = len(dfs["moments_raw"])
    if new_count == 0:
        logger.info("No moments created today — pipeline skipped")
        return {
            "status":       "skipped",
            "reason":       "no moments created today",
            "duration_sec": round((datetime.utcnow() - start).total_seconds(), 2),
        }

    logger.info(f"  {new_count} moments to process today")

    # ── Step 2: Preprocess ────────────────────────────────────────
    logger.info("Step 2/3: Preprocessing...")
    try:
        moments, passages, books, users = preprocess_all(
            moments_df=dfs["moments_raw"],
            books_df=dfs["books_raw"],
            users_df=dfs["users_raw"],
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    valid_moments = [m for m in moments if m.get("is_valid", False)]
    logger.info(f"  {len(moments)} processed ({len(valid_moments)} valid)")

    # ── Step 3: Write to BQ ───────────────────────────────────────
    logger.info("Step 3/3: Writing to BQ...")
    try:
        bq_tables = write_to_bq(moments, passages, books, users)
    except Exception as e:
        logger.error(f"BQ write failed: {e}")
        raise HTTPException(status_code=500, detail=f"BQ write failed: {str(e)}")

    logger.info("Synchronous steps complete. Queuing compatibility + rankings in background.")

    # ── Steps 4+5: Compatibility + Rankings (background) ─────────
    # Each valid moment is matched independently against existing BQ users.
    # Moments in today's batch do NOT match each other (only match against
    # users already in BQ before this run).
    for m in valid_moments:
        user_id    = str(m.get("user_id", ""))
        passage_id = str(m.get("passage_id", ""))
        book_id    = str(m.get("book_id", ""))
        interp     = m.get("cleaned_interpretation", "")

        if not user_id or not passage_id or not interp:
            continue

        background_tasks.add_task(
            _run_batch_compatibility,
            user_id, book_id, passage_id, interp,
        )

    duration = (datetime.utcnow() - start).total_seconds()
    logger.info(f"Pipeline synchronous phase complete in {duration:.2f}s")
    logger.info("=" * 60)

    result = {
        "status":           "success",
        "timestamp":        start.isoformat(),
        "moments_count":    len(moments),
        "passages_count":   len(passages),
        "books_count":      len(books),
        "users_count":      len(users),
        "valid_moments":    len(valid_moments),
        "bq_tables":        bq_tables,
        "compat_queued":    len(valid_moments),
        "duration_sec":     round(duration, 2),
    }
    _last_run = result
    return result


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
    Refits BT model synchronously so results are always fresh.
    """
    _refit_and_save_rankings(user_id, book_id=book_id, passage_id=passage_id, k=k)
    rows = get_rankings(user_id, book_id=book_id, passage_id=passage_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No rankings found for {user_id}")
    return {"user_id": user_id, "rankings": rows}


@app.get("/pipeline/status")
def pipeline_status():
    if not _last_run:
        return {"status": "no_runs_yet"}
    return _last_run


# ── Background: batch compatibility ──────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed

# Max parallel Gemini calls per user batch.
# Each worker fires up to 2 Gemini requests (decompose + score), so
# MAX_COMPAT_WORKERS=8 means up to 16 simultaneous API calls.
# Tune down if you see HTTP 429s from Gemini in the logs.
MAX_COMPAT_WORKERS = 8
def _run_batch_compatibility(
    user_id:        str,
    book_id:        str,
    passage_id:     str,
    interpretation: str,
) -> None:
    """
    Run compatibility between one processed moment and every existing BQ user
    on the same passage — in parallel using ThreadPoolExecutor.

    Work is I/O-bound (Gemini API + BigQuery), so threads give real speedup:
    N matches that took N×~4s sequentially now take ~4s regardless of N
    (up to MAX_COMPAT_WORKERS concurrent pairs).

    Only matches against users already in BQ — moments ingested in the same
    pipeline run are excluded (get_moments_for_passage uses exclude_user_id,
    and BQ is written before this task starts so same-batch users will appear,
    but passage-level deduplication in the compat agent handles exact pairs).

    BQ streaming insert (log_compat_run) is thread-safe — each call is an
    independent HTTP request, no shared state.
    """
    logger.info(f"[Compat] starting batch for {user_id} on {book_id}/{passage_id}")

    try:
        other_moments = get_moments_for_passage(book_id, passage_id, exclude_user_id=user_id)
    except Exception as e:
        logger.warning(f"[Compat] could not fetch passage moments for {passage_id}: {e}")
        return

    if not other_moments:
        logger.info(f"[Compat] no other users on passage {passage_id} — skipping")
        return

    logger.info(
        f"[Compat] running {len(other_moments)} comparisons for {user_id} "
        f"(parallel, max_workers={MAX_COMPAT_WORKERS})"
    )
    compat_runs   = 0
    compat_errors = 0

    def _run_one(other: dict) -> tuple:
        """
        Run a single compat pair.
        Always returns (other_user_id, result_or_None) — never raises.

        Returning a tuple (instead of just result) means the caller always
        knows which user the result belongs to without reaching into
        futures[future], which was the source of the
        'error for <uid>: None' log noise when result was None.
        """
        other_user_id = str(other["user_id"])
        other_text    = other.get("cleaned_interpretation", "")
        if not other_text:
            return other_user_id, None   # skip silently — empty interpretation
        result = run_compatibility_agent(
            user_a_id=user_id,
            user_b_id=other_user_id,
            book_id=book_id,
            moment_a={"passage_id": passage_id, "cleaned_interpretation": interpretation},
            moment_b={"passage_id": passage_id, "cleaned_interpretation": other_text},
        )
        return other_user_id, result

    with ThreadPoolExecutor(max_workers=MAX_COMPAT_WORKERS) as executor:
        futures = {executor.submit(_run_one, other): other for other in other_moments}

        for future in as_completed(futures):
            try:
                other_user_id, result = future.result()  # always (str, dict|None)
                if result is None:
                    pass  # empty interpretation — skipped silently inside _run_one
                elif "error" in result:
                    logger.warning(f"[Compat] error {user_id}×{other_user_id}: {result['error']}")
                    compat_errors += 1
                else:
                    compat_runs += 1
                    logger.info(
                        f"[Compat] ✓ {user_id}×{other_user_id}: "
                        f"{result.get('dominant_think')} ({result.get('confidence')})"
                    )
            except Exception as e:
                logger.error(f"[Compat] exception in future: {e}")
                compat_errors += 1

    logger.info(f"[Compat] complete for {user_id} — {compat_runs} runs, {compat_errors} errors")

    # Refit rankings for this user now that new compat runs are logged
    try:
        refit_user(user_id)
        logger.info(f"[Rankings] ✓ updated for {user_id}")
    except Exception as e:
        logger.error(f"[Rankings] error for {user_id}: {e}")


# ── Background: rankings refit ────────────────────────────────────────────────

def _refit_and_save_rankings(
    user_id:    str,
    book_id:    Optional[str] = None,
    passage_id: Optional[str] = None,
    k:          int = 5,
) -> None:
    """Thin wrapper — delegates all BT logic to run_rankings.refit_user()."""
    refit_user(user_id, book_id=book_id, passage_id=passage_id, k=k)# test
# timeout fix
# test1