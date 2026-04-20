"""
bq_client.py — BigQuery data layer + shared utilities for Momento.

Fully replaces tools.py. Import everything from here instead.

Replaces all _read_json_file / _write_json_file calls in tools.py,
decomposing_agent.py, agent.py (compat), and aggregator.py.

Actual BQ tables in dataset `new_moments_process`:

  compatibility_results  (existing):
    run_id INTEGER, user_a INTEGER, user_b INTEGER, book_id STRING,
    passage_id STRING, think RECORD, feel RECORD, dominant_think STRING,
    dominant_feel STRING, verdict STRING, confidence FLOAT,
    think_rationale STRING, feel_rationale STRING, timestamp TIMESTAMP

  moments_processed  (existing, read-only):
    interpretation_id STRING, user_id STRING, book_id STRING,
    passage_id STRING, book_title STRING, passage_number INTEGER,
    character_id INTEGER, character_name STRING,
    cleaned_interpretation STRING, original_word_count INTEGER,
    is_valid BOOLEAN, quality_score FLOAT, quality_issues INTEGER,
    detected_issues RECORD, anomalies RECORD, metrics RECORD, timestamp STRING

  decompositions, scoring_runs, comparisons, conversations, rankings:
    tables we own — schema defined in bq_schema.py

CRITICAL type notes:
  - user_a / user_b in compatibility_results are INTEGER (= character_id)
  - think / feel in compatibility_results are RECORD (BQ nested struct)
    → read back as dicts natively; write as dicts via streaming insert
  - moments are identified by user_id (STRING) in our pipeline,
    but compatibility_results uses character_id (INTEGER)
  - get_character_id() bridges the two

Auth: Application Default Credentials (ADC).
  Local:    gcloud auth application-default login
  Cloud Run: attach service account with BigQuery Data Editor + Job User roles.

Env vars:
  GCP_PROJECT  — GCP project ID
  BQ_DATASET   — dataset name (default: new_moments_process)
"""

import os
import json
from datetime import datetime
from typing import Optional

from google.cloud import bigquery

# ── Config ────────────────────────────────────────────────────────────────────

_PROJECT = os.environ.get("MOMENT_GCP_PROJECTID", "your-gcp-project")
_DATASET = os.environ.get("BQ_DATASET",  "new_moments_processed")

_TABLES = {
    "decompositions":     "decompositions",
    "scoring_runs":       "scoring_runs",
    "compatibility_runs": "compatibility_results",  # actual BQ name
    "comparisons":        "comparisons",
    "conversations":      "conversations",
    "rankings":           "rankings",
    "moments":            "moments_processed",       # actual BQ name
    "users":              "users_processed",
    "books":              "books_processed",
}

_client: Optional[bigquery.Client] = None


def get_client() -> bigquery.Client:
    global _client
    if _client is None:
        _client = bigquery.Client(project=_PROJECT)
    return _client


def _table(name: str) -> str:
    bq_name = _TABLES.get(name, name)
    return f"`{_PROJECT}.{_DATASET}.{bq_name}`"


# ── Generic helpers ───────────────────────────────────────────────────────────

def _run_query(sql: str, params: list = None) -> list[dict]:
    client = get_client()
    cfg    = bigquery.QueryJobConfig(query_parameters=params or [])
    rows   = client.query(sql, job_config=cfg).result()
    return [dict(row) for row in rows]


def _insert_rows(table_name: str, rows: list[dict]) -> None:
    client = get_client()
    # Use the raw table ref without backticks for the Python client
    bq_name = _TABLES.get(table_name, table_name)
    tbl_ref = f"{_PROJECT}.{_DATASET}.{bq_name}"
    errors  = client.insert_rows_json(tbl_ref, rows)
    if errors:
        raise RuntimeError(f"BQ insert error into {table_name}: {errors}")


def _insert_row(table_name: str, row: dict) -> None:
    _insert_rows(table_name, [row])


def _upsert_row(table_name: str, row: dict, key_cols: list[str]) -> None:
    """
    MERGE upsert. Scalars become query parameters.
    Dicts/lists (RECORD columns) are kept as-is — BQ streaming handles nesting.
    For MERGE we serialise them as JSON strings; simple INSERT uses _insert_rows.
    """
    client = get_client()
    tbl    = _table(table_name)

    all_cols   = list(row.keys())
    value_cols = [c for c in all_cols if c not in key_cols]

    placeholders: list[str] = []
    params: list             = []

    for col, val in row.items():
        if isinstance(val, bool):
            placeholders.append(f"@{col} AS {col}")
            params.append(bigquery.ScalarQueryParameter(col, "BOOL", val))
        elif isinstance(val, int):
            placeholders.append(f"@{col} AS {col}")
            params.append(bigquery.ScalarQueryParameter(col, "INT64", val))
        elif isinstance(val, float):
            placeholders.append(f"@{col} AS {col}")
            params.append(bigquery.ScalarQueryParameter(col, "FLOAT64", val))
        elif isinstance(val, str):
            placeholders.append(f"@{col} AS {col}")
            params.append(bigquery.ScalarQueryParameter(col, "STRING", val))
        else:
            # Nested dict/list → JSON string for the MERGE source
            placeholders.append(f"@{col} AS {col}")
            params.append(bigquery.ScalarQueryParameter(col, "STRING", json.dumps(val)))

    source_select = ", ".join(placeholders)
    key_cond      = " AND ".join(f"CAST(T.{c} AS STRING) = CAST(S.{c} AS STRING)" for c in key_cols)
    update_set    = ", ".join(f"T.{c} = S.{c}" for c in value_cols)
    insert_cols   = ", ".join(all_cols)
    insert_vals   = ", ".join(f"S.{c}" for c in all_cols)

    sql = f"""
    MERGE {tbl} T
    USING (SELECT {source_select}) S
    ON {key_cond}
    WHEN MATCHED THEN
      UPDATE SET {update_set}
    WHEN NOT MATCHED THEN
      INSERT ({insert_cols}) VALUES ({insert_vals})
    """
    client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()


# ── Moments (read-only) ───────────────────────────────────────────────────────

def get_moments(user_id: str, book_id: Optional[str] = None) -> list[dict]:
    """
    Fetch valid reader moments from moments_processed.
    Returns cleaned_interpretation as the moment text.
    """
    conditions = ["user_id = @uid", "is_valid = TRUE"]
    params     = [bigquery.ScalarQueryParameter("uid", "STRING", user_id)]
    if book_id:
        conditions.append("book_id = @bid")
        params.append(bigquery.ScalarQueryParameter("bid", "STRING", book_id))
    where = " AND ".join(conditions)
    return _run_query(
        f"""
        SELECT user_id, character_id, character_name,
               book_id, passage_id, cleaned_interpretation,
               original_word_count, quality_score
        FROM {_table('moments')}
        WHERE {where}
        """,
        params,
    )


def get_moments_for_passage(book_id: str, passage_id: str, exclude_user_id: str) -> list[dict]:
    """
    Return all valid moments for a passage, excluding the anchor user.
    Used by batch compatibility to find all users to compare against.
    """
    return _run_query(
        f"""
        SELECT user_id, cleaned_interpretation
        FROM {_table('moments')}
        WHERE CAST(book_id AS STRING)    = @bid
          AND CAST(passage_id AS STRING) = @pid
          AND CAST(user_id AS STRING)   != @uid
          AND is_valid = TRUE
        """,
        [
            bigquery.ScalarQueryParameter("bid", "STRING", str(book_id)),
            bigquery.ScalarQueryParameter("pid", "STRING", str(passage_id)),
            bigquery.ScalarQueryParameter("uid", "STRING", str(exclude_user_id)),
        ],
    )


def get_moment_text(user_id: str, passage_id: str, book_id: str) -> Optional[str]:
    """Return the cleaned_interpretation string for one user+passage+book."""
    rows = _run_query(
        f"""
        SELECT cleaned_interpretation
        FROM {_table('moments')}
        WHERE user_id    = @uid
          AND passage_id = @pid
          AND book_id    = @bid
          AND is_valid   = TRUE
        LIMIT 1
        """,
        [
            bigquery.ScalarQueryParameter("uid", "STRING", user_id),
            bigquery.ScalarQueryParameter("pid", "STRING", passage_id),
            bigquery.ScalarQueryParameter("bid", "STRING", book_id),
        ],
    )
    return rows[0]["cleaned_interpretation"] if rows else None




# ── Decompositions ────────────────────────────────────────────────────────────
# Schema (our table):
#   user_id STRING, passage_id STRING, book_id STRING,
#   subclaims STRING (JSON array), computed_at TIMESTAMP

def get_decomposition(user_id: str, passage_id: str, book_id: str) -> Optional[dict]:
    rows = _run_query(
        f"""
        SELECT * FROM {_table('decompositions')}
        WHERE CAST(user_id AS STRING)    = @uid
          AND CAST(passage_id AS STRING) = @pid
          AND CAST(book_id AS STRING)    = @bid
        LIMIT 1
        """,
        [
            bigquery.ScalarQueryParameter("uid", "STRING", str(user_id)),
            bigquery.ScalarQueryParameter("pid", "STRING", str(passage_id)),
            bigquery.ScalarQueryParameter("bid", "STRING", str(book_id)),
        ],
    )
    if not rows:
        return None
    row = rows[0]
    if isinstance(row.get("subclaims"), str):
        row["subclaims"] = json.loads(row["subclaims"])
    return row


def save_decomposition(result: dict) -> None:
    """
    Insert a new decomposition into BQ.
    Only called when get_decomposition() returned nothing, so no deletion needed.
    subclaims is ARRAY<STRUCT> in BQ — pass as list of dicts directly.
    """
    client = get_client()
    bq_name = _TABLES.get("decompositions", "decompositions")
    row = {
        "user_id":    str(result["user_id"]),
        "passage_id": str(result["passage_id"]),
        "book_id":    str(result.get("book_id", "")),
        "subclaims":  result.get("subclaims", []),
    }
    errors = client.insert_rows_json(f"{_PROJECT}.{_DATASET}.{bq_name}", [row])
    if errors:
        raise RuntimeError(f"BQ insert error into decompositions: {errors}")
    print(f"[BQ] decomposition saved: {result['user_id']} / {result['passage_id']}")


# ── Scoring runs ──────────────────────────────────────────────────────────────
# Schema (our table):
#   user_a_id STRING, user_b_id STRING, passage_id STRING, book_id STRING,
#   scoring STRING (JSON), timestamp TIMESTAMP

def get_scoring(user_a_id: str, user_b_id: str, passage_id: str, book_id: str) -> Optional[dict]:
    rows = _run_query(
        f"""
        SELECT scoring FROM {_table('scoring_runs')}
        WHERE CAST(passage_id AS STRING) = @pid
          AND CAST(book_id AS STRING)    = @bid
          AND (
            (CAST(user_a_id AS STRING) = @ua AND CAST(user_b_id AS STRING) = @ub)
            OR
            (CAST(user_a_id AS STRING) = @ub AND CAST(user_b_id AS STRING) = @ua)
          )
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        [
            bigquery.ScalarQueryParameter("pid", "STRING", passage_id),
            bigquery.ScalarQueryParameter("bid", "STRING", book_id),
            bigquery.ScalarQueryParameter("ua",  "STRING", user_a_id),
            bigquery.ScalarQueryParameter("ub",  "STRING", user_b_id),
        ],
    )
    if not rows:
        return None
    # scoring is a STRUCT in BQ — comes back as dict, no JSON parsing needed
    return rows[0]["scoring"]


def save_scoring(user_a_id: str, user_b_id: str, passage_id: str, book_id: str, scoring: dict) -> None:
    """
    Insert scoring result. scoring is a STRUCT in BQ so pass as dict directly.
    Only called when no cached scoring exists so no upsert needed.
    """
    client = get_client()
    bq_name = _TABLES.get("scoring_runs", "scoring_runs")
    row = {
        "user_a_id":  str(user_a_id),
        "user_b_id":  str(user_b_id),
        "passage_id": str(passage_id),
        "book_id":    str(book_id),
        "scoring":    scoring,
        "timestamp":  datetime.utcnow().isoformat(),
    }
    errors = client.insert_rows_json(f"{_PROJECT}.{_DATASET}.{bq_name}", [row])
    if errors:
        raise RuntimeError(f"BQ insert error into scoring_runs: {errors}")
    print(f"[BQ] scoring saved: {user_a_id} × {user_b_id} / {passage_id}")


# ── Compatibility results ─────────────────────────────────────────────────────
# compatibility_results: user_a / user_b are STRING (user_id).
# think / feel are flat columns: think_R, think_C, think_D, feel_R, feel_C, feel_D.

def get_compat_run(user_a: int, user_b: int, book_id: str, passage_id: str) -> Optional[dict]:
    """
    Most recent compat result for a pair. Pair order is symmetric.
    user_a / user_b are integer character_ids.
    think / feel return as dicts already (BQ RECORD → Python dict).
    """
    rows = _run_query(
        f"""
        SELECT * FROM {_table('compatibility_runs')}
        WHERE CAST(book_id AS STRING)    = @bid
          AND CAST(passage_id AS STRING) = @pid
          AND (
            (CAST(user_a AS STRING) = @ua AND CAST(user_b AS STRING) = @ub)
            OR
            (CAST(user_a AS STRING) = @ub AND CAST(user_b AS STRING) = @ua)
          )
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        [
            bigquery.ScalarQueryParameter("bid", "STRING", book_id),
            bigquery.ScalarQueryParameter("pid", "STRING", passage_id),
            bigquery.ScalarQueryParameter("ua",  "STRING", str(user_a)),
            bigquery.ScalarQueryParameter("ub",  "STRING", str(user_b)),
        ],
    )
    if not rows:
        return None
    row = rows[0]
    # Reconstruct think/feel dicts from flat columns for pipeline compatibility
    row["think"] = {"R": row.get("think_R", 0), "C": row.get("think_C", 0), "D": row.get("think_D", 0)}
    row["feel"]  = {"R": row.get("feel_R",  0), "C": row.get("feel_C",  0), "D": row.get("feel_D",  0)}
    return row


def log_compat_run(record: dict) -> None:
    """
    Stream-insert a new compatibility result.
    think / feel are now flat columns (think_R, think_C, think_D etc.)
    after the CREATE OR REPLACE that flattened the RECORD columns.
    user_a / user_b are STRING user_ids.
    """
    think = record.get("think", {})
    feel  = record.get("feel",  {})
    row = {
        "run_id":          abs(hash((
            record.get("user_a"), record.get("user_b"),
            record.get("passage_id"), record.get("timestamp", "")
        ))) % (2 ** 31),
        "user_a":          str(record.get("user_a", "")),
        "user_b":          str(record.get("user_b", "")),
        "book_id":         record.get("book_id", ""),
        "passage_id":      record.get("passage_id", ""),
        "think_R":         int(think.get("R", 0)),
        "think_C":         int(think.get("C", 0)),
        "think_D":         int(think.get("D", 0)),
        "feel_R":          int(feel.get("R",  0)),
        "feel_C":          int(feel.get("C",  0)),
        "feel_D":          int(feel.get("D",  0)),
        "dominant_think":  record.get("dominant_think", ""),
        "dominant_feel":   record.get("dominant_feel", ""),
        "verdict":         record.get("verdict", ""),
        "confidence":      float(record.get("confidence", 0.0)),
        "think_rationale": record.get("think_rationale", ""),
        "feel_rationale":  record.get("feel_rationale", ""),
        "timestamp":       record.get("timestamp", datetime.utcnow().isoformat()),
    }
    _insert_row("compatibility_runs", row)
    print(f"[BQ] compat run logged: user_a={row['user_a']} × user_b={row['user_b']}")


def get_compat_runs_for_user(user_id: str, min_confidence: float = 0.0) -> list[dict]:
    """All compat results for a user. user_a/user_b are STRING in compatibility_results."""
    rows = _run_query(
        f"""
        SELECT * FROM {_table('compatibility_runs')}
        WHERE (CAST(user_a AS STRING) = @uid OR CAST(user_b AS STRING) = @uid)
          AND confidence >= @min_conf
        ORDER BY timestamp DESC
        """,
        [
            bigquery.ScalarQueryParameter("uid",      "STRING",  str(user_id)),
            bigquery.ScalarQueryParameter("min_conf", "FLOAT64", min_confidence),
        ],
    )
    for row in rows:
        row["think"] = {"R": row.get("think_R", 0), "C": row.get("think_C", 0), "D": row.get("think_D", 0)}
        row["feel"]  = {"R": row.get("feel_R",  0), "C": row.get("feel_C",  0), "D": row.get("feel_D",  0)}
    return rows


# ── Comparisons ───────────────────────────────────────────────────────────────

def insert_comparison(record: dict) -> None:
    _insert_row("comparisons", record)
    print(f"[BQ] comparison recorded: user={record['user_id']}")


def get_comparisons(user_id: Optional[str] = None) -> list[dict]:
    if user_id:
        return _run_query(
            f"SELECT * FROM {_table('comparisons')} WHERE user_id = @uid",
            [bigquery.ScalarQueryParameter("uid", "STRING", user_id)],
        )
    return _run_query(f"SELECT * FROM {_table('comparisons')}")


# ── Conversations ─────────────────────────────────────────────────────────────

def get_conversation_weights() -> dict[str, dict[str, float]]:
    """{ user_id: { run_id: engagement_score } }"""
    rows    = _run_query(
        f"SELECT user_id, match_run_id, engagement_score FROM {_table('conversations')}"
    )
    weights: dict[str, dict[str, float]] = {}
    for row in rows:
        weights.setdefault(row["user_id"], {})[row["match_run_id"]] = float(row["engagement_score"])
    return weights


# ── Rankings ──────────────────────────────────────────────────────────────────

def save_rankings(user_id: str, book_id: str, passage_id: str, ranked: list[dict]) -> None:
    client = get_client()
    bq_name = _TABLES.get("rankings", "rankings")
    # Delete stale rows
    client.query(
        f"""
        DELETE FROM `{_PROJECT}.{_DATASET}.{bq_name}`
        WHERE CAST(user_id AS STRING) = @uid AND book_id = @bid AND passage_id = @pid
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("uid", "STRING", str(user_id)),
            bigquery.ScalarQueryParameter("bid", "STRING", book_id),
            bigquery.ScalarQueryParameter("pid", "STRING", passage_id),
        ]),
    ).result()

    rows = []
    for pos, r in enumerate(ranked, 1):
        wu = r.get("weights_used", {})
        match_user = r.get("user_b") if str(r.get("user_a")) == str(user_id) else r.get("user_a")
        rows.append({
            "user_id":       str(user_id),   # pass as string to avoid float precision loss
            "book_id":       book_id,
            "passage_id":    passage_id,
            "rank_position": pos,
            "run_id":        str(r.get("run_id", 0)),
            "match_user":    str(match_user) if match_user else "0",
            "verdict":       r.get("verdict", ""),
            "confidence":    float(r.get("confidence", 0.0)),
            "bt_score":      float(r.get("bt_score", 0.0)),
            "blend_score":   float(r.get("blend_score", 0.0)),
            "conf_weight":   float(wu.get("conf", 0.4)),
            "bt_weight":     float(wu.get("bt", 0.6)),
            "n_comparisons": int(wu.get("n_comparisons", 0)),
            "generated_at":  datetime.utcnow().isoformat(),
        })
    if rows:
        _insert_rows("rankings", rows)
    print(f"[BQ] {len(rows)} rankings saved: {user_id} / {book_id} / {passage_id}")


def get_rankings(user_id: str, book_id: Optional[str] = None, passage_id: Optional[str] = None) -> list[dict]:
    conditions = ["CAST(user_id AS STRING) = @uid"]
    params     = [bigquery.ScalarQueryParameter("uid", "STRING", str(user_id))]
    if book_id:
        conditions.append("book_id = @bid")
        params.append(bigquery.ScalarQueryParameter("bid", "STRING", book_id))
    if passage_id:
        conditions.append("passage_id = @pid")
        params.append(bigquery.ScalarQueryParameter("pid", "STRING", passage_id))
    where = " AND ".join(conditions)
    # Deduplicate using ROW_NUMBER — keeps only the most recent insert per rank slot
    # This avoids DELETE on streaming buffer which BQ does not allow
    return _run_query(
        f"""
        SELECT * EXCEPT(rn) FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(user_id AS STRING), book_id, passage_id, rank_position
                    ORDER BY generated_at DESC
                ) AS rn
            FROM {_table('rankings')}
            WHERE {where}
        ) WHERE rn = 1
        ORDER BY book_id, passage_id, rank_position
        """,
        params,
    )


# ── Utilities (replaces tools.py entirely) ────────────────────────────────────

import re
import functools

def extract_json(text: str) -> dict:
    """Robustly extract a JSON object from LLM output."""
    if not text:
        return {"error": "empty response", "raw": text}
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "could not extract JSON", "raw": text}


def count_new_moments(user_id: str, since_iso: str) -> int:
    """Count moments saved after a given ISO timestamp. Used by profile agent."""
    rows = _run_query(
        f"""
        SELECT COUNT(*) AS n FROM {_table('moments')}
        WHERE user_id   = @uid
          AND timestamp > @since
          AND is_valid  = TRUE
        """,
        [
            bigquery.ScalarQueryParameter("uid",   "STRING", user_id),
            bigquery.ScalarQueryParameter("since", "STRING", since_iso),
        ],
    )
    return int(rows[0]["n"]) if rows else 0


# COMPAT_LOG_FILE kept as empty string so any legacy imports don't break
COMPAT_LOG_FILE = ""