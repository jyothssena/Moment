"""
bq_loader.py
============
MOMENT Data Pipeline — BigQuery Integration

Handles ALL BigQuery operations for the pipeline:

  1. load_raw_from_bq()     — replaces GCS JSON reads in data_acquisition.py
                              Pulls raw interpretations, passages, and user data
                              from BQ staging tables directly into memory as dicts.

  2. write_processed_to_bq() — replaces write_outputs() local JSON writes in preprocessor.py
                               Upserts processed moments, books, and users into
                               their respective BQ tables after pipeline processing.

  3. upload_reports_to_bq()  — stores TFDV schema stats + validation + bias reports
                               in a BQ reports table so every run is queryable.

  4. get_bq_client()         — shared authenticated BigQuery client (lazy singleton).

Dataset layout in BigQuery (project: moment-486719):
  moment-486719.moment_raw.interpretations      ← raw user interpretations
  moment-486719.moment_raw.passages             ← raw passage details
  moment-486719.moment_raw.user_data            ← raw character/user data
  moment-486719.moment_processed.moments        ← processed moments (main ML table)
  moment-486719.moment_processed.books          ← processed book/passage records
  moment-486719.moment_processed.users          ← processed user profiles
  moment-486719.moment_reports.pipeline_runs    ← per-run pipeline stats & reports

All writes use WRITE_TRUNCATE by default (full refresh each pipeline run),
which is safe for this batch pipeline. Pass write_disposition='WRITE_APPEND'
to append incrementally instead.

Error handling:
  - All BQ operations are wrapped in try/except.
  - On failure: logs error, returns empty list / False.
  - Pipeline continues using local JSON fallback if BQ is unreachable.
  - Schema mismatches are auto-healed via autodetect=True on initial load.

Author: MOMENT Group 23 | IE7374 MLOps | Northeastern University
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Project / Dataset constants ────────────────────────────────────────────

PROJECT_ID   = os.environ.get("GOOGLE_CLOUD_PROJECT", "moment-486719")

# Raw staging tables (written by acquisition, read by preprocessing)
RAW_DATASET  = "moment_raw"
RAW_INTERP_TABLE   = "interpretations"
RAW_PASSAGES_TABLE = "passages"
RAW_USERS_TABLE    = "user_data"

# Processed tables (written by preprocessing / post-pipeline)
PROC_DATASET        = "moment_processed"
PROC_MOMENTS_TABLE  = "moments"
PROC_BOOKS_TABLE    = "books"
PROC_USERS_TABLE    = "users"

# Reports table (written by schema_stats + validation + notify)
REPORTS_DATASET      = "moment_reports"
REPORTS_RUNS_TABLE   = "pipeline_runs"

# ─── Lazy BQ client singleton ────────────────────────────────────────────────

_bq_client = None


def get_bq_client():
    """
    Return a cached BigQuery client.

    Uses Application Default Credentials (ADC) — works inside GCP (Vertex AI,
    Cloud Composer / Airflow) and locally when `gcloud auth application-default
    login` has been run.

    Returns:
        google.cloud.bigquery.Client or None if unavailable.
    """
    global _bq_client
    if _bq_client is None:
        try:
            from google.cloud import bigquery  # type: ignore
            _bq_client = bigquery.Client(project=PROJECT_ID)
            logger.info(f"[bq_loader] BigQuery client initialized for project={PROJECT_ID}")
        except Exception as exc:
            logger.error(f"[bq_loader] Failed to initialize BigQuery client: {exc}")
            return None
    return _bq_client


# ─── Helper: ensure dataset exists ──────────────────────────────────────────

def _ensure_dataset(client, dataset_id: str, location: str = "US") -> bool:
    """Create BQ dataset if it does not already exist."""
    from google.cloud import bigquery  # type: ignore
    dataset_ref = f"{PROJECT_ID}.{dataset_id}"
    try:
        client.get_dataset(dataset_ref)
        return True
    except Exception:
        try:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            client.create_dataset(dataset, exists_ok=True)
            logger.info(f"[bq_loader] Created dataset {dataset_ref}")
            return True
        except Exception as exc:
            logger.error(f"[bq_loader] Could not create dataset {dataset_ref}: {exc}")
            return False


# ─── Helper: flatten nested dicts/lists for BQ ──────────────────────────────

def _flatten_record(record: dict) -> dict:
    """
    Flatten a single record so it is BQ-compatible:
      - nested dicts  → JSON string
      - lists         → comma-joined string (for simple lists) or JSON string
      - None          → kept as None (BQ handles nullable columns)
    """
    flat = {}
    for k, v in record.items():
        if isinstance(v, dict):
            flat[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, list):
            if all(isinstance(i, str) for i in v):
                flat[k] = ", ".join(v)
            else:
                flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v
    return flat


def _records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert a list of pipeline records to a BQ-ready DataFrame."""
    if not records:
        return pd.DataFrame()
    flat = [_flatten_record(r) for r in records]
    df = pd.DataFrame(flat)
    # BQ does not support column names with dots — replace with underscores
    df.columns = [c.replace(".", "_") for c in df.columns]
    return df


# ─── 1. Load raw data from BigQuery ─────────────────────────────────────────

def load_raw_from_bq() -> dict[str, list[dict]]:
    """
    Pull raw input tables from BigQuery into memory as lists of dicts.

    Replaces the GCS → local file download step in data_acquisition.py.
    The pipeline reads directly from BQ staging tables, which are pre-populated
    from GCS via the existing upload workflow (or manually loaded once).

    Returns:
        {
          "interpretations": [...],   # list of raw interpretation dicts
          "passages":        [...],   # list of passage dicts
          "user_data":       [...],   # list of user/character dicts
        }
        Any missing table returns an empty list; pipeline falls back to local files.
    """
    client = get_bq_client()
    if client is None:
        logger.warning("[bq_loader] BQ client unavailable — returning empty raw data")
        return {"interpretations": [], "passages": [], "user_data": []}

    results: dict[str, list[dict]] = {}

    tables = {
        "interpretations": f"`{PROJECT_ID}.{RAW_DATASET}.{RAW_INTERP_TABLE}`",
        "passages":        f"`{PROJECT_ID}.{RAW_DATASET}.{RAW_PASSAGES_TABLE}`",
        "user_data":       f"`{PROJECT_ID}.{RAW_DATASET}.{RAW_USERS_TABLE}`",
    }

    for key, full_table in tables.items():
        try:
            query = f"SELECT * FROM {full_table}"
            logger.info(f"[bq_loader] Querying {full_table}...")
            df = client.query(query).to_dataframe()
            results[key] = df.to_dict(orient="records")
            logger.info(f"[bq_loader] Loaded {len(results[key])} rows from {full_table}")
        except Exception as exc:
            logger.warning(f"[bq_loader] Could not read {full_table}: {exc}")
            results[key] = []

    return results


# ─── 2. Load raw GCS files INTO BigQuery (acquisition → BQ staging) ─────────

def upload_raw_to_bq(dataframes: dict[str, pd.DataFrame]) -> bool:
    """
    Write the DataFrames acquired from GCS into BigQuery raw staging tables.

    Called at the end of task_acquire_data() so the raw data is immediately
    available in BQ for ad-hoc querying and auditing.

    Args:
        dataframes: {filename: DataFrame} dict from DataAcquisition.run()
                    Keys should contain one of: interpretations/user_interpretations,
                    passages/passage_details, user_data/characters.

    Returns:
        True if at least one table was written successfully.
    """
    client = get_bq_client()
    if client is None:
        logger.warning("[bq_loader] BQ client unavailable — skipping raw upload")
        return False

    _ensure_dataset(client, RAW_DATASET)

    # Map filename fragments to canonical table names
    table_map = {
        "interpretation": RAW_INTERP_TABLE,
        "passage":        RAW_PASSAGES_TABLE,
        "user":           RAW_USERS_TABLE,
        "character":      RAW_USERS_TABLE,   # characters.csv → user_data table
    }

    success_count = 0
    for filename, df in dataframes.items():
        fname_lower = filename.lower()
        target_table = None
        for fragment, table_name in table_map.items():
            if fragment in fname_lower:
                target_table = table_name
                break

        if target_table is None:
            logger.warning(f"[bq_loader] Could not map '{filename}' to a BQ table — skipping")
            continue

        full_table = f"{PROJECT_ID}.{RAW_DATASET}.{target_table}"
        try:
            from google.cloud.bigquery import LoadJobConfig, WriteDisposition  # type: ignore
            job_config = LoadJobConfig(
                write_disposition=WriteDisposition.WRITE_TRUNCATE,
                autodetect=True,
            )
            job = client.load_table_from_dataframe(df, full_table, job_config=job_config)
            job.result()  # wait for completion
            logger.info(f"[bq_loader] Uploaded {len(df)} rows → {full_table}")
            success_count += 1
        except Exception as exc:
            logger.error(f"[bq_loader] Failed to upload '{filename}' → {full_table}: {exc}")

    logger.info(f"[bq_loader] Raw upload complete: {success_count}/{len(dataframes)} tables written")
    return success_count > 0


# ─── 3. Write processed data to BigQuery ────────────────────────────────────

def write_processed_to_bq(
    moments: list[dict],
    books:   list[dict],
    users:   list[dict],
    write_disposition: str = "WRITE_TRUNCATE",
) -> bool:
    """
    Write the three processed datasets (moments, books, users) to BigQuery.

    Called after preprocessor.write_outputs() so both local JSON files AND BQ
    are always in sync. If BQ write fails, local JSON is still intact.

    Args:
        moments:           list of processed moment records
        books:             list of processed book/passage records
        users:             list of processed user profile records
        write_disposition: "WRITE_TRUNCATE" (default, full refresh per run)
                           or "WRITE_APPEND" (incremental)

    Returns:
        True if all three tables were written successfully.
    """
    client = get_bq_client()
    if client is None:
        logger.warning("[bq_loader] BQ client unavailable — skipping processed write")
        return False

    _ensure_dataset(client, PROC_DATASET)

    datasets = [
        (PROC_MOMENTS_TABLE, moments, "moments"),
        (PROC_BOOKS_TABLE,   books,   "books"),
        (PROC_USERS_TABLE,   users,   "users"),
    ]

    all_success = True
    for table_name, records, label in datasets:
        full_table = f"{PROJECT_ID}.{PROC_DATASET}.{table_name}"
        try:
            if not records:
                logger.warning(f"[bq_loader] No {label} records to write — skipping {full_table}")
                continue

            df = _records_to_dataframe(records)

            from google.cloud.bigquery import LoadJobConfig, WriteDisposition  # type: ignore

            wd = (WriteDisposition.WRITE_TRUNCATE
                  if write_disposition == "WRITE_TRUNCATE"
                  else WriteDisposition.WRITE_APPEND)

            job_config = LoadJobConfig(
                write_disposition=wd,
                autodetect=True,
            )
            job = client.load_table_from_dataframe(df, full_table, job_config=job_config)
            job.result()  # block until done
            logger.info(f"[bq_loader] Wrote {len(df)} {label} rows → {full_table}")

        except Exception as exc:
            logger.error(f"[bq_loader] Failed to write {label} → {full_table}: {exc}")
            all_success = False

    return all_success


# ─── 4. Upload pipeline reports / stats to BigQuery ─────────────────────────

def upload_reports_to_bq(
    run_id:            str,
    schema_stats:      dict | None = None,
    validation_report: dict | None = None,
    bias_results:      dict | None = None,
    anomaly_summary:   dict | None = None,
) -> bool:
    """
    Insert one row per pipeline run into moment_reports.pipeline_runs.

    This makes every run's quality metrics queryable in BigQuery — useful for
    trend analysis, dashboards, and automated alerting on metric degradation.

    Args:
        run_id:            unique identifier for this pipeline run (e.g. ISO timestamp)
        schema_stats:      output of generate_schema_stats.run_schema_stats()
        validation_report: output of task_validation()
        bias_results:      output of task_bias_detection()
        anomaly_summary:   dict with anomaly counts from anomalies.detect_anomalies()

    Returns:
        True on success.
    """
    client = get_bq_client()
    if client is None:
        logger.warning("[bq_loader] BQ client unavailable — skipping report upload")
        return False

    _ensure_dataset(client, REPORTS_DATASET)

    full_table = f"{PROJECT_ID}.{REPORTS_DATASET}.{REPORTS_RUNS_TABLE}"

    # Build a single flat row for this run
    row = {
        "run_id":            run_id,
        "run_timestamp":     datetime.utcnow().isoformat(),

        # Validation gate results
        "validation_status":       (validation_report or {}).get("status", "UNKNOWN"),
        "validation_moments_rows": _nested_get(validation_report, "validations", "moments_processed.json", "row_count"),
        "validation_valid_rate":   _nested_get(validation_report, "validations", "moments_processed.json", "valid_rate"),

        # Bias detection results
        "bias_age_max_dev":       _nested_get(bias_results, "age", "max_dev"),
        "bias_gender_max_dev":    _nested_get(bias_results, "gender", "max_dev"),
        "bias_book_assessment":   _nested_get(bias_results, "book", "assessment"),
        "bias_char_assessment":   _nested_get(bias_results, "character", "assessment"),

        # Anomaly counts
        "anomaly_wc_outliers":    (anomaly_summary or {}).get("word_count_outliers", None),
        "anomaly_read_outliers":  (anomaly_summary or {}).get("readability_outliers", None),
        "anomaly_duplicates":     (anomaly_summary or {}).get("duplicates", None),
        "anomaly_style_mismatch": (anomaly_summary or {}).get("style_mismatches", None),

        # Schema stats — serialized as JSON string for flexibility
        "schema_stats_json":      json.dumps(schema_stats, default=str) if schema_stats else None,

        # Raw reports stored as JSON strings for full auditability
        "validation_report_json": json.dumps(validation_report, default=str) if validation_report else None,
        "bias_report_json":       json.dumps(bias_results, default=str) if bias_results else None,
    }

    try:
        df = pd.DataFrame([row])
        from google.cloud.bigquery import LoadJobConfig, WriteDisposition  # type: ignore
        job_config = LoadJobConfig(
            write_disposition=WriteDisposition.WRITE_APPEND,
            autodetect=True,
        )
        job = client.load_table_from_dataframe(df, full_table, job_config=job_config)
        job.result()
        logger.info(f"[bq_loader] Pipeline run report written → {full_table} (run_id={run_id})")
        return True
    except Exception as exc:
        logger.error(f"[bq_loader] Failed to write report to {full_table}: {exc}")
        return False


# ─── 5. Query helpers (used by model pipeline / downstream consumers) ────────

def query_processed_moments(
    book_id:    str | None = None,
    passage_id: str | None = None,
    valid_only: bool = True,
    limit:      int | None = None,
) -> list[dict]:
    """
    Query processed moments from BigQuery with optional filters.

    Used by the model pipeline (interpretation_ingestion.py) instead of
    reading local JSON files.

    Args:
        book_id:    filter by book_id (e.g. 'gutenberg_84')
        passage_id: filter by passage_id
        valid_only: if True, only return records where is_valid = TRUE
        limit:      max rows to return (None = all)

    Returns:
        list of moment dicts, or [] on error.
    """
    client = get_bq_client()
    if client is None:
        return []

    conditions = []
    if valid_only:
        conditions.append("is_valid = TRUE")
    if book_id:
        conditions.append(f"book_id = '{book_id}'")
    if passage_id:
        conditions.append(f"passage_id = '{passage_id}'")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = (
        f"SELECT * FROM `{PROJECT_ID}.{PROC_DATASET}.{PROC_MOMENTS_TABLE}` "
        f"{where_clause} {limit_clause}"
    )

    try:
        df = client.query(query).to_dataframe()
        records = df.to_dict(orient="records")
        logger.info(f"[bq_loader] query_processed_moments → {len(records)} rows")
        return records
    except Exception as exc:
        logger.error(f"[bq_loader] query_processed_moments failed: {exc}")
        return []


def query_processed_users() -> list[dict]:
    """
    Return all processed user profiles from BigQuery.
    Used by the model pipeline for reader profile lookups.
    """
    client = get_bq_client()
    if client is None:
        return []
    try:
        query = f"SELECT * FROM `{PROJECT_ID}.{PROC_DATASET}.{PROC_USERS_TABLE}`"
        df = client.query(query).to_dataframe()
        records = df.to_dict(orient="records")
        logger.info(f"[bq_loader] query_processed_users → {len(records)} rows")
        return records
    except Exception as exc:
        logger.error(f"[bq_loader] query_processed_users failed: {exc}")
        return []


def query_pipeline_runs(last_n: int = 10) -> list[dict]:
    """
    Return the last N pipeline run reports from moment_reports.pipeline_runs.
    Useful for dashboards and trend monitoring.
    """
    client = get_bq_client()
    if client is None:
        return []
    try:
        query = (
            f"SELECT * FROM `{PROJECT_ID}.{REPORTS_DATASET}.{REPORTS_RUNS_TABLE}` "
            f"ORDER BY run_timestamp DESC LIMIT {last_n}"
        )
        df = client.query(query).to_dataframe()
        return df.to_dict(orient="records")
    except Exception as exc:
        logger.error(f"[bq_loader] query_pipeline_runs failed: {exc}")
        return []


# ─── Private helpers ─────────────────────────────────────────────────────────

def _nested_get(d: dict | None, *keys: str) -> Any:
    """Safely traverse nested dict. Returns None if any key is missing."""
    if d is None:
        return None
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


# ─── Standalone smoke-test ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = get_bq_client()
    if client:
        print(f"[bq_loader] Connected to BigQuery project: {PROJECT_ID}")
        print(f"[bq_loader] Raw dataset:       {RAW_DATASET}")
        print(f"[bq_loader] Processed dataset: {PROC_DATASET}")
        print(f"[bq_loader] Reports dataset:   {REPORTS_DATASET}")
    else:
        print("[bq_loader] Could not connect to BigQuery — check credentials")
