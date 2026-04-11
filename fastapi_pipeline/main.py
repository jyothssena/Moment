"""
main.py — MOMENT FastAPI Data Pipeline
=======================================
Endpoints:
  POST /pipeline/run   — pull from Cloud SQL → preprocess → write to BQ
  GET  /pipeline/status — last run result
  GET  /health          — sanity check
"""

import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from cloudsql_loader import CloudSQLLoader
from preprocessor_fastapi import preprocess_all
from bq_writer import write_to_bq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MOMENT Data Pipeline",
    description="Cloud SQL → Preprocess → BigQuery",
    version="1.0.0",
)

# ── In-memory status store ────────────────────────────────────────────────────
_last_run: dict = {}


# ── Pydantic models ───────────────────────────────────────────────────────────

class RunResponse(BaseModel):
    status:          str
    timestamp:       str
    moments_count:   int
    books_count:     int
    users_count:     int
    valid_moments:   int
    bq_tables:       list
    duration_sec:    float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/pipeline/run", response_model=RunResponse)
def run_pipeline():
    """Full pipeline: Cloud SQL → preprocess → BigQuery."""
    global _last_run
    start = datetime.utcnow()
    logger.info("=" * 60)
    logger.info(f"Pipeline run started at {start.isoformat()}")

    # ── Step 1: Load from Cloud SQL ──
    logger.info("Step 1/3: Loading data from Cloud SQL...")
    try:
        loader = CloudSQLLoader()
        loader.run()
        dfs = loader.get_dataframes()
    except Exception as e:
        logger.error(f"Cloud SQL load failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cloud SQL load failed: {str(e)}")

    logger.info(f"  interpretations: {len(dfs['interpretations_train'])} rows")
    logger.info(f"  passages:        {len(dfs['passage_details_new'])} rows")
    logger.info(f"  users:           {len(dfs['user_details_new'])} rows")

    # ── Step 2: Preprocess ──
    logger.info("Step 2/3: Preprocessing...")
    try:
        moments, books, users = preprocess_all(
            interpretations_df = dfs["interpretations_train"],
            passages_df        = dfs["passage_details_new"],
            users_df           = dfs["user_details_new"],
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    valid_count = sum(1 for m in moments if m.get("is_valid", False))
    logger.info(f"  moments: {len(moments)} ({valid_count} valid)")
    logger.info(f"  books:   {len(books)}")
    logger.info(f"  users:   {len(users)}")

    # ── Step 3: Write to BQ ──
    logger.info("Step 3/3: Writing to BigQuery...")
    try:
        bq_tables = write_to_bq(moments, books, users)
    except Exception as e:
        logger.error(f"BQ write failed: {e}")
        raise HTTPException(status_code=500, detail=f"BQ write failed: {str(e)}")

    duration = (datetime.utcnow() - start).total_seconds()
    logger.info(f"Pipeline complete in {duration:.2f}s")
    logger.info("=" * 60)

    result = {
        "status":        "success",
        "timestamp":     start.isoformat(),
        "moments_count": len(moments),
        "books_count":   len(books),
        "users_count":   len(users),
        "valid_moments": valid_count,
        "bq_tables":     bq_tables,
        "duration_sec":  round(duration, 2),
    }
    _last_run = result
    return RunResponse(**result)


@app.get("/pipeline/status")
def pipeline_status():
    """Return the result of the last pipeline run."""
    if not _last_run:
        return {"status": "no_runs_yet"}
    return _last_run