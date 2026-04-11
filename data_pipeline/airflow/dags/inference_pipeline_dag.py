"""
MOMENT Inference Pipeline DAG
==============================
Runs the full inference pipeline on test data.

Flow:
    load_test_data
        → compatibility   (decompose → score → aggregate per pair)
        → validate_compat_output   (gate: row count, required cols, confidence range)
        → bradley_terry_ranking    (BT rerank per user × book × passage)
        → upload_to_bq             (staging → new_moments_processed)
        → notify                   (ALL_DONE)

BQ sources (read-only):
    moment-486719.new_moments_processed.interpretations_test
    moment-486719.new_moments_processed.books_processed
    moment-486719.new_moments_processed.users_processed
    moment-486719.new_moments_processed.comparisons
    moment-486719.new_moments_processed.conversations

BQ final destination (written incrementally per pair):
    moment-486719.new_moments_processed.compat_results  (upsert by run_id)
    moment-486719.new_moments_processed.decompositions         (upsert)
    moment-486719.new_moments_processed.rankings               (overwrite)

XCom strategy: tasks push/pull only BQ table IDs (strings), never raw records.

run_id per pair: stable hash of sorted(user_a, user_b) + book_id + passage_id
  → same pair always gets same run_id regardless of DAG run
  → allows safe upsert / deduplication
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
from functools import wraps
import sys, os, json, logging

# ════════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════════

logger = logging.getLogger('airflow.task')

LOGS_DIR = os.environ.get('LOGS_DIR', '/opt/airflow/logs')
os.makedirs(LOGS_DIR, exist_ok=True)

def _make_handler(path, level, filter_fn=None):
    h = logging.FileHandler(path)
    h.setLevel(level)
    if filter_fn:
        h.addFilter(filter_fn)
    h.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    return h

logger.addHandler(_make_handler(os.path.join(LOGS_DIR, 'inference_INFO.log'),    logging.INFO,    lambda r: r.levelno == logging.INFO))
logger.addHandler(_make_handler(os.path.join(LOGS_DIR, 'inference_WARNING.log'), logging.WARNING, lambda r: r.levelno == logging.WARNING))
logger.addHandler(_make_handler(os.path.join(LOGS_DIR, 'inference_ERROR.log'),   logging.ERROR))
logger.addHandler(_make_handler(os.path.join(LOGS_DIR, 'inference_ALL.log'),     logging.DEBUG))


# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

SCRIPTS_DIR = os.environ.get(
    'SCRIPTS_DIR',
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'scripts'
    )
)

REPO_ROOT = os.environ.get('REPO_ROOT', '/opt/airflow')

for _p in [SCRIPTS_DIR, REPO_ROOT, os.path.join(REPO_ROOT, 'models')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

PIPELINE_CONFIG = os.environ.get(
    'PIPELINE_CONFIG',
    os.path.join(os.path.dirname(SCRIPTS_DIR), 'config', 'config.yaml')
)

BQ_PROJECT        = os.environ.get('GOOGLE_CLOUD_PROJECT', 'moment-486719')
BQ_RAW_DATASET    = 'new_moments_processed'
BQ_FINAL_DATASET  = 'new_moments_processed'

COMPAT_REQUIRED_COLS  = {'confidence', 'dominant_think', 'dominant_feel',
                          'think', 'feel', 'user_a', 'user_b', 'passage_id', 'book_id'}
CONFIDENCE_MIN        = 0.20
CONFIDENCE_MAX        = 0.95
MAX_ERROR_RATE        = 0.20

TEAM_EMAILS = [
    'chandrasekar.s@northeastern.edu',
    'vasisht.h@northeastern.edu',
    'shurpali.t@northeastern.edu',
    'sreenivaasan.j@northeastern.edu',
    'patel.heetp@northeastern.edu',
    'wang.gre@northeastern.edu',
]


# ════════════════════════════════════════════════════════════════
#  BIGQUERY HELPERS
# ════════════════════════════════════════════════════════════════

def _bq_client():
    from google.cloud import bigquery
    return bigquery.Client(project=BQ_PROJECT)

def staging_dataset(run_id: str) -> str:
    safe = (run_id
            .replace('-', '_').replace(':', '_')
            .replace('+', '_').replace('.', '_').replace('T', '_'))
    return f"moments_staging_{safe}"[:1024]

def bq_table_id(run_id: str, table: str) -> str:
    return f"{BQ_PROJECT}.{staging_dataset(run_id)}.{table}"

def ensure_staging_dataset(run_id: str):
    from google.cloud import bigquery
    client  = _bq_client()
    ds_id   = f"{BQ_PROJECT}.{staging_dataset(run_id)}"
    ds      = bigquery.Dataset(ds_id)
    ds.location = "US"
    client.create_dataset(ds, exists_ok=True)
    logger.debug(f"Staging dataset ready: {ds_id}")

def bq_write(df, table_id: str, append: bool = False):
    from google.cloud import bigquery
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v
            )
    disposition = (bigquery.WriteDisposition.WRITE_APPEND
                   if append else bigquery.WriteDisposition.WRITE_TRUNCATE)
    job = _bq_client().load_table_from_dataframe(
        df, table_id,
        job_config=bigquery.LoadJobConfig(
            write_disposition=disposition,
            autodetect=True,
        )
    )
    job.result()
    logger.debug(f"Wrote {len(df)} rows → {table_id}")

def bq_read(table_id: str):
    import pandas as pd
    df = _bq_client().query(f"SELECT * FROM `{table_id}`").to_dataframe()
    logger.debug(f"Read {len(df)} rows ← {table_id}")
    return df

def bq_copy_table(src: str, dst: str):
    from google.cloud import bigquery
    job = _bq_client().copy_table(
        src, dst,
        job_config=bigquery.CopyJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
    )
    job.result()
    logger.debug(f"Copied {src} → {dst}")


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def make_pair_run_id(user_a: str, user_b: str, book_id: str, passage_id: str) -> str:
    """
    Stable deterministic run_id for a compatibility pair.
    Users sorted so (Alice, Bob) == (Bob, Alice).
    Used as the key in bradley_terry runs dict.
    """
    import hashlib
    user_lo, user_hi = sorted([user_a, user_b])
    raw = f"{user_lo}|{user_hi}|{book_id}|{passage_id}"
    return "run_" + hashlib.md5(raw.encode()).hexdigest()[:16]


def upsert_compat_result(result: dict, final_tid: str, tmp_run_id: str):
    """
    Upsert a single compat result into the final BQ table by run_id.
    Creates the table on first insert, then uses MERGE for subsequent runs.
    """
    import pandas as pd
    tmp_tid = f"{BQ_PROJECT}.{staging_dataset(tmp_run_id)}.compat_tmp"
    bq_write(pd.DataFrame([result]), tmp_tid)

    # Check if final table exists
    try:
        _bq_client().get_table(final_tid)
        table_exists = True
    except Exception:
        table_exists = False

    if not table_exists:
        # First run — just copy tmp to final directly
        bq_copy_table(tmp_tid, final_tid)
        logger.info(f"   Created new table: {final_tid}")
        return

    merge_sql = f"""
        MERGE `{final_tid}` T
        USING `{tmp_tid}` S
        ON T.run_id = S.run_id
        WHEN MATCHED THEN
            UPDATE SET
                user_a         = S.user_a,
                user_b         = S.user_b,
                book_id        = S.book_id,
                passage_id     = S.passage_id,
                confidence     = S.confidence,
                dominant_think = S.dominant_think,
                dominant_feel  = S.dominant_feel,
                think_R        = S.think_R,
                think_C        = S.think_C,
                think_D        = S.think_D,
                feel_R         = S.feel_R,
                feel_C         = S.feel_C,
                feel_D         = S.feel_D,
                verdict        = S.verdict,
                timestamp      = S.timestamp,
                run_id_dag     = S.run_id_dag
        WHEN NOT MATCHED THEN
            INSERT ROW
    """
    _bq_client().query(merge_sql).result()


# ════════════════════════════════════════════════════════════════
#  DECORATOR
# ════════════════════════════════════════════════════════════════

def log_task_execution(task_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()
            logger.info("=" * 70)
            logger.info(f"🚀 Starting task: {task_name}  [{start:%Y-%m-%d %H:%M:%S}]")
            try:
                result = func(*args, **kwargs)
                dur = (datetime.now() - start).total_seconds()
                if dur > 300:
                    logger.warning(f"⚠️  {task_name} took {dur:.1f}s (>5 min)")
                logger.info(f"✅ Completed: {task_name}  [{dur:.2f}s]")
                logger.info("=" * 70)
                return result
            except Exception as e:
                dur = (datetime.now() - start).total_seconds()
                logger.error(f"❌ Failed: {task_name} — {type(e).__name__}: {e}  [{dur:.2f}s]")
                logger.exception("Full traceback:")
                raise
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════
#  TASK 1 — LOAD TEST DATA
# ════════════════════════════════════════════════════════════════

@log_task_execution("Load Test Data")
def task_load_test_data(**context):
    import pandas as pd
    run_id = context['run_id']
    ti     = context['task_instance']

    logger.info(f"📥 Loading test data from BQ  [run={run_id}]")

    interp_tid  = f"{BQ_PROJECT}.{BQ_RAW_DATASET}.interpretations_test"
    passage_tid = f"{BQ_PROJECT}.{BQ_RAW_DATASET}.books_processed"
    user_tid    = f"{BQ_PROJECT}.{BQ_RAW_DATASET}.users_processed"

    interp_df  = bq_read(interp_tid)
    passage_df = bq_read(passage_tid)
    user_df    = bq_read(user_tid)

    logger.info(f"   interpretations_test : {len(interp_df)} rows")
    logger.info(f"   books_processed      : {len(passage_df)} rows")
    logger.info(f"   users_processed      : {len(user_df)} rows")

    if len(interp_df) == 0:
        raise ValueError("interpretations_test is empty — nothing to run")

    pairs_count = 0
    passage_ids = interp_df['passage_id'].unique()
    for pid in passage_ids:
        users_on_passage = interp_df[interp_df['passage_id'] == pid]['character_name'].unique()
        n = len(users_on_passage)
        pairs_count += n * (n - 1) // 2

    logger.info(f"   Passages: {len(passage_ids)}  |  Estimated pairs: {pairs_count}")

    if pairs_count == 0:
        raise ValueError("No user pairs found in interpretations_test")

    ensure_staging_dataset(run_id)
    raw_table_ids = {
        'interpretations_test': interp_tid,
        'books_processed':      passage_tid,
        'users_processed':      user_tid,
    }

    logger.info(f"✅ Test data ready — {pairs_count} pairs to process")
    ti.xcom_push(key='raw_table_ids', value=raw_table_ids)
    return {'pair_count': pairs_count, 'passage_count': int(len(passage_ids))}


# ════════════════════════════════════════════════════════════════
#  TASK 2 — COMPATIBILITY
#
#  Key behaviours:
#  1. user_a always lexicographically smaller (user_lo)
#  2. Stable pair run_id via make_pair_run_id()
#  3. Each successful pair upserted immediately to
#     new_moments_processed.compat_results via MERGE
#  4. Resume: reads already-done pairs from final table at startup
# ════════════════════════════════════════════════════════════════

@log_task_execution("Compatibility")
def task_compatibility(**context):
    import pandas as pd
    import time
    ti      = context['task_instance']
    run_id  = context['run_id']
    raw_ids = ti.xcom_pull(task_ids='load_test_data', key='raw_table_ids')

    logger.info(f"🤝 Running compatibility  [run={run_id}]")

    # ── Path injection (survives Airflow fork) ──────────────
    import sys, os
    _repo_root = os.environ.get('REPO_ROOT', '/opt/airflow')
    for _p in [_repo_root, os.path.join(_repo_root, 'models')]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from models.decomposing_agent import run_decomposer           # type: ignore
    from models.compatibility_agent import _call_scorer, _build_scorer_prompt  # type: ignore
    from models.aggregator import aggregate                        # type: ignore

    # ── Load data ───────────────────────────────────────────
    interp_df  = bq_read(raw_ids['interpretations_test'])
    passage_df = bq_read(raw_ids['books_processed'])

    moment_lookup = {}
    for m in interp_df.to_dict('records'):
        key = (m['character_name'], m['passage_id'])
        if key not in moment_lookup:
            moment_lookup[key] = m

    # ── Load user_id lookup: character_name → user_id ──
    user_id_lookup = {}
    try:
        users_df = bq_read(raw_ids['users_processed'])
        for _, row in users_df.iterrows():
            user_id_lookup[row['character_name']] = row['user_id']
        logger.info(f"   User ID lookup: {len(user_id_lookup)} users")
    except Exception as e:
        logger.warning(f"⚠️  Could not load user IDs: {e}")

    # ── Final table — all pairs land here ──────────────────
    final_compat_tid = f"{BQ_PROJECT}.{BQ_FINAL_DATASET}.compat_results"

    # ── Resume: load already-processed pairs ───────────────
    processed_pairs = set()
    try:
        existing_df = _bq_client().query(f"""
            SELECT user_a, user_b, passage_id
            FROM `{final_compat_tid}`
        """).to_dataframe()
        for _, row in existing_df.iterrows():
            processed_pairs.add((row['user_a'], row['user_b'], row['passage_id']))
        logger.info(f"   Resuming — {len(processed_pairs)} pairs already in final table")
    except Exception:
        logger.info("   No previous results found — starting fresh")

    # ── Main loop ───────────────────────────────────────────
    compat_results = []
    decomp_results = []
    error_count    = 0

    passage_ids = list(interp_df['passage_id'].unique())
    logger.info(f"   Processing {len(passage_ids)} passages")

    for passage_id in passage_ids:
        # Get book_id directly from interpretations_test — no join needed
        # books_processed uses different passage_id format (gutenberg_84_passage_1 vs passage_1)
        passage_interps = interp_df[interp_df['passage_id'] == passage_id]
        book_id = passage_interps['book'].iloc[0] if len(passage_interps) > 0 else 'unknown'

        users_on_passage = interp_df[
            interp_df['passage_id'] == passage_id
        ]['character_name'].unique().tolist()

        checked = set()
        for i, user_a in enumerate(users_on_passage):
            for user_b in users_on_passage[i+1:]:
                pair_key = (user_a, user_b, passage_id)
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                # Always normalise order
                user_lo, user_hi = sorted([user_a, user_b])

                # ── Skip if already done ──────────────────
                if (user_lo, user_hi, passage_id) in processed_pairs:
                    logger.info(f"   ⏭️  Already done: {user_lo} × {user_hi} / {passage_id}")
                    continue

                moment_lo_txt = moment_lookup.get((user_lo, passage_id), {}).get('interpretation', '')
                moment_hi_txt = moment_lookup.get((user_hi, passage_id), {}).get('interpretation', '')

                if not moment_lo_txt or not moment_hi_txt:
                    logger.warning(f"⚠️  Missing moment for {user_lo} × {user_hi} / {passage_id}")
                    error_count += 1
                    continue

                # ── Decompose A (user_lo) ─────────────────
                try:
                    decomp_a = None
                    for _attempt in range(5):
                        try:
                            decomp_a = run_decomposer(user_lo, passage_id, book_id, moment_lo_txt)
                            break
                        except Exception as _e:
                            if '503' in str(_e) and _attempt < 4:
                                wait = 30 * (_attempt + 1)
                                logger.warning(f"⚠️  Gemini 503 attempt {_attempt+1}/5, retrying in {wait}s...")
                                time.sleep(wait)
                            else:
                                raise
                    if decomp_a and 'error' not in decomp_a:
                        decomp_results.append(decomp_a)
                except Exception as e:
                    logger.error(f"❌ Decompose A failed {user_lo}/{passage_id}: {e}")
                    error_count += 1
                    continue

                # ── Decompose B (user_hi) ─────────────────
                try:
                    decomp_b = None
                    for _attempt in range(5):
                        try:
                            decomp_b = run_decomposer(user_hi, passage_id, book_id, moment_hi_txt)
                            break
                        except Exception as _e:
                            if '503' in str(_e) and _attempt < 4:
                                wait = 30 * (_attempt + 1)
                                logger.warning(f"⚠️  Gemini 503 attempt {_attempt+1}/5, retrying in {wait}s...")
                                time.sleep(wait)
                            else:
                                raise
                    if decomp_b and 'error' not in decomp_b:
                        decomp_results.append(decomp_b)
                except Exception as e:
                    logger.error(f"❌ Decompose B failed {user_hi}/{passage_id}: {e}")
                    error_count += 1
                    continue

                # ── Score ─────────────────────────────────
                try:
                    scoring = None
                    prompt  = _build_scorer_prompt(decomp_a, decomp_b)
                    for _attempt in range(5):
                        try:
                            scoring = _call_scorer(prompt)
                            break
                        except Exception as _e:
                            if '503' in str(_e) and _attempt < 4:
                                wait = 30 * (_attempt + 1)
                                logger.warning(f"⚠️  Gemini 503 attempt {_attempt+1}/5, retrying in {wait}s...")
                                time.sleep(wait)
                            else:
                                raise
                except Exception as e:
                    logger.error(f"❌ Scorer failed {user_lo} × {user_hi} / {passage_id}: {e}")
                    error_count += 1
                    continue

                # ── Aggregate ─────────────────────────────
                try:
                    if not scoring or 'error' in scoring or 'matched_pairs' not in scoring:
                        logger.warning(f"⚠️  Bad scoring for {user_lo} × {user_hi}: {scoring}")
                        error_count += 1
                        continue

                    scoring_normalized = dict(scoring)
                    scoring_normalized['unmatched_a'] = [
                        u['id'] if isinstance(u, dict) else u
                        for u in scoring.get('unmatched_a', [])
                    ]
                    scoring_normalized['unmatched_b'] = [
                        u['id'] if isinstance(u, dict) else u
                        for u in scoring.get('unmatched_b', [])
                    ]
                    # Convert think_q/feel_q boolean lists → think/feel dicts
                    # that aggregate() expects per matched pair
                    # Q mapping (per compatibility_agent.py scorer prompt):
                    # Think: Q1=R, Q2=YES→R/NO→C, Q3=YES→R/NO→C,
                    #        Q4=YES→C/NO→R, Q5=YES→R/NO→C
                    # Feel:  Q1=R, Q2=YES→R/NO→C, Q3=YES→R/NO→C,
                    #        Q4=YES→R/NO→C, Q5=YES→R/NO→C
                    def bools_to_rcd(q, is_think=True):
                        if not q or len(q) < 5:
                            return {"R": 0.0, "C": 0.0, "D": 0.0}
                        if is_think:
                            r = sum([q[0], q[1], q[2], not q[3], q[4]])
                            c = sum([not q[1], not q[2], q[3], not q[4]])
                        else:
                            r = sum([q[0], q[1], q[2], q[3], q[4]])
                            c = sum([not q[1], not q[2], not q[3], not q[4]])
                        total = 5.0
                        return {"R": r/total, "C": c/total, "D": 0.0}

                    converted_pairs = []
                    for p in scoring_normalized.get('matched_pairs', []):
                        cp = dict(p)
                        tq = p.get('think_q', [False]*5)
                        fq = p.get('feel_q',  [False]*5)
                        cp['think'] = bools_to_rcd(tq, is_think=True)
                        cp['feel']  = bools_to_rcd(fq, is_think=False)
                        converted_pairs.append(cp)

                    scoring_normalized['matched_pairs'] = converted_pairs

                    combined_decomp = {
                        "reader_a": decomp_a,
                        "reader_b": decomp_b,
                    }
                    result = aggregate(combined_decomp, scoring_normalized)
                    
                    
                except Exception as e:
                    logger.error(f"❌ Aggregate failed {user_lo} × {user_hi}: {e}")
                    error_count += 1
                    continue

                # ── Attach metadata ───────────────────────
                pair_run_id = make_pair_run_id(user_lo, user_hi, book_id, passage_id)

                # Flatten think/feel dicts into separate columns
                think = result.pop('think', {}) or {}
                feel  = result.pop('feel',  {}) or {}

                result['think_R']      = think.get('R', 0)
                result['think_C']      = think.get('C', 0)
                result['think_D']      = think.get('D', 0)
                result['feel_R']       = feel.get('R', 0)
                result['feel_C']       = feel.get('C', 0)
                result['feel_D']       = feel.get('D', 0)
                result['verdict']      = result.get('dominant_think')
                result['user_a']       = user_id_lookup.get(user_lo, user_lo)
                result['user_b']       = user_id_lookup.get(user_hi, user_hi)
                result['book_id']      = book_id
                result['passage_id']   = passage_id
                result['timestamp']    = datetime.utcnow().isoformat()
                result['run_id']       = pair_run_id
                result['run_id_dag']   = run_id
                # Remove nested fields not needed in BQ
                result.pop('character_a', None)
                result.pop('character_b', None)

                compat_results.append(result)

                # ── Upsert directly to final BQ table ─────
                try:
                    upsert_compat_result(result, final_compat_tid, run_id)
                    logger.info(f"   ✓ Upserted: {user_lo} × {user_hi} / {passage_id}  [{pair_run_id}]")
                    processed_pairs.add((user_lo, user_hi, passage_id))
                except Exception as e:
                    logger.warning(f"⚠️  Could not upsert pair to BQ: {e}")

    total_pairs = len(compat_results) + error_count
    logger.info(f"   Pairs processed: {total_pairs}  |  Success: {len(compat_results)}  |  Errors: {error_count}")

    if len(compat_results) == 0:
        raise ValueError(f"Compatibility produced 0 results — {error_count} errors. Check Gemini API key.")

    # ── Write decompositions to staging ────────────────────
    decomp_tid = None
    if decomp_results:
        decomp_tid = bq_table_id(run_id, 'decompositions')
        bq_write(pd.DataFrame(decomp_results), decomp_tid)
        logger.info(f"   ✓ decompositions ({len(decomp_results)} rows) → {decomp_tid}")

    error_rate = error_count / max(total_pairs, 1)
    if error_rate > MAX_ERROR_RATE:
        logger.warning(f"⚠️  High error rate: {error_rate:.1%} ({error_count}/{total_pairs} pairs failed)")

    # XCom: point downstream at final table
    ti.xcom_push(key='compat_table_id', value=final_compat_tid)
    ti.xcom_push(key='decomp_table_id', value=decomp_tid)
    return {
        'total_pairs':   total_pairs,
        'success_count': len(compat_results),
        'error_count':   error_count,
        'error_rate':    round(error_rate, 3),
    }


# ════════════════════════════════════════════════════════════════
#  TASK 3 — VALIDATE COMPAT OUTPUT
# ════════════════════════════════════════════════════════════════

@log_task_execution("Validate Compatibility Output")
def task_validate_compat_output(**context):
    import pandas as pd
    import numpy as np
    ti       = context['task_instance']
    run_id   = context['run_id']

    compat_tid   = ti.xcom_pull(task_ids='compatibility', key='compat_table_id')
    compat_stats = ti.xcom_pull(task_ids='compatibility') or {}

    logger.info(f"🔍 Validating compatibility output  [run={run_id}]")

    df     = bq_read(compat_tid)
    issues = []
    status = 'PASSED'

    # HARD: row count
    if len(df) == 0:
        raise ValueError("❌ HARD FAIL — compat_results table is empty")
    logger.info(f"   Row count: {len(df)} ✓")

    # HARD: required columns
    missing_cols = COMPAT_REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"❌ HARD FAIL — missing required columns: {missing_cols}")
    logger.info(f"   Required columns: all present ✓")

    # SOFT: confidence range
    try:
        conf_vals    = pd.to_numeric(df['confidence'], errors='coerce').dropna()
        out_of_range = ((conf_vals < CONFIDENCE_MIN) | (conf_vals > CONFIDENCE_MAX)).sum()
        if out_of_range > 0:
            pct = out_of_range / len(conf_vals)
            msg = f"confidence out of [{CONFIDENCE_MIN}, {CONFIDENCE_MAX}]: {out_of_range} rows ({pct:.1%})"
            issues.append(msg)
            logger.warning(f"⚠️  {msg}")
            status = 'WARNING'
        else:
            logger.info(f"   Confidence range: all in range ✓")
    except Exception as e:
        issues.append(f"confidence check skipped: {e}")
        status = 'WARNING'

    # SOFT: error rate
    error_rate = compat_stats.get('error_rate', 0.0)
    if error_rate > MAX_ERROR_RATE:
        msg = f"error rate {error_rate:.1%} exceeds {MAX_ERROR_RATE:.0%}"
        issues.append(msg)
        logger.warning(f"⚠️  {msg}")
        status = 'WARNING'
    else:
        logger.info(f"   Error rate: {error_rate:.1%} ✓")

    # SOFT: dominant_think values
    valid_verdicts = {'resonate', 'contradict', 'diverge'}
    if 'dominant_think' in df.columns:
        unexpected = set(df['dominant_think'].dropna().unique()) - valid_verdicts
        if unexpected:
            msg = f"unexpected dominant_think values: {unexpected}"
            issues.append(msg)
            logger.warning(f"⚠️  {msg}")
            status = 'WARNING'
        else:
            logger.info(f"   dominant_think values: all valid ✓")

    report_tid = bq_table_id(run_id, 'compat_validation_report')
    bq_write(
        pd.DataFrame([{
            'run_id':      run_id,
            'status':      status,
            'valid_rows':  len(df),
            'error_rate':  error_rate,
            'issues':      json.dumps(issues),
            'timestamp':   datetime.utcnow().isoformat(),
        }]),
        report_tid
    )
    logger.info(f"   Validation report → {report_tid}")

    if status == 'PASSED':
        logger.info("✅ Compatibility validation PASSED")
    else:
        logger.warning(f"⚠️  Compatibility validation {status} — {len(issues)} issue(s)")

    return {
        'status':     status,
        'valid_rows': len(df),
        'error_rate': error_rate,
        'issues':     issues,
    }


# ════════════════════════════════════════════════════════════════
#  TASK 4 — BRADLEY-TERRY RANKING
# ════════════════════════════════════════════════════════════════

@log_task_execution("Bradley-Terry Ranking")
def task_bradley_terry_ranking(**context):
    import pandas as pd
    ti     = context['task_instance']
    run_id = context['run_id']

    compat_tid = ti.xcom_pull(task_ids='compatibility', key='compat_table_id')

    logger.info(f"🏆 Bradley-Terry ranking  [run={run_id}]")

    import sys, os
    _repo_root = os.environ.get('REPO_ROOT', '/opt/airflow')
    for _p in [_repo_root, os.path.join(_repo_root, 'models')]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from models.bradley_terry import (   # type: ignore
        fit_global_bt,
        fit_user_bt,
        rerank_topk,
    )

    compat_df = bq_read(compat_tid)
    logger.info(f"   Loaded {len(compat_df)} compat results")

    # Build runs dict using stable pair run_id
    runs = {}
    for _, row in compat_df.iterrows():
        row_dict = row.to_dict()
        rid = row_dict.get('run_id') or make_pair_run_id(
            row_dict.get('user_a', ''),
            row_dict.get('user_b', ''),
            row_dict.get('book_id', ''),
            row_dict.get('passage_id', ''),
        )
        row_dict['run_id'] = rid
        runs[rid] = row_dict

    logger.info(f"   Runs index: {len(runs)} entries")

    comparisons_tid = f"{BQ_PROJECT}.{BQ_FINAL_DATASET}.comparisons"
    try:
        comp_df     = bq_read(comparisons_tid)
        comparisons = comp_df.to_dict('records')
        logger.info(f"   Comparisons loaded: {len(comparisons)}")
    except Exception as e:
        logger.warning(f"⚠️  Could not load comparisons ({e}) — using pure confidence")
        comparisons = []

    conversations_tid = f"{BQ_PROJECT}.{BQ_FINAL_DATASET}.conversations"
    try:
        conv_df = bq_read(conversations_tid)
        from collections import defaultdict
        conv_weights: dict = defaultdict(dict)
        for _, row in conv_df.iterrows():
            conv_weights[row['user_id']][row['match_run_id']] = row['engagement_score']
        logger.info(f"   Conversations loaded: {len(conv_df)} rows, {len(conv_weights)} users")
    except Exception as e:
        logger.warning(f"⚠️  Could not load conversations ({e}) — using uniform weights")
        conv_weights = {}

    if comparisons:
        global_bt = fit_global_bt(comparisons, conv_weights)
        logger.info(f"   Global BT model: {len(global_bt)} runs covered")
    else:
        global_bt = {}
        logger.info("   No comparisons — using confidence only")

    all_users = set()
    passages  = set()
    for r in runs.values():
        all_users.add(r.get('user_a'))
        all_users.add(r.get('user_b'))
        b = r.get('book_id')
        p = r.get('passage_id')
        if b and p:
            passages.add((b, p))

    all_users.discard(None)
    logger.info(f"   Users: {len(all_users)}  |  Passages: {len(passages)}")

    all_results = {}
    for user in sorted(all_users):
        candidate_ids = [
            rid for rid, r in runs.items()
            if r.get('user_a') == user or r.get('user_b') == user
        ]
        all_results[user] = {}
        books = sorted({b for b, _ in passages})

        for book_id in books:
            book_passages = sorted({p for b, p in passages if b == book_id})
            all_results[user][book_id] = {}

            for passage_id in book_passages:
                ranked = rerank_topk(
                    candidate_run_ids=candidate_ids,
                    user_id=user,
                    runs=runs,
                    comparisons=comparisons,
                    conv_weights=conv_weights,
                    global_bt=global_bt if global_bt else None,
                    k=5,
                    book_id=book_id,
                    passage_id=passage_id,
                )
                if ranked:
                    all_results[user][book_id][passage_id] = ranked

    ranking_rows = []
    for user, books in all_results.items():
        for book_id, book_passages in books.items():
            for passage_id, ranked_list in book_passages.items():
                for rank_pos, entry in enumerate(ranked_list, start=1):
                    ranking_rows.append({
                        'anchor_user':    user,
                        'rank':           rank_pos,
                        'run_id':         entry.get('run_id'),
                        'matched_user':   entry.get('user_b') if entry.get('user_a') == user else entry.get('user_a'),
                        'book_id':        book_id,
                        'passage_id':     passage_id,
                        'verdict':        entry.get('verdict'),
                        'confidence':     entry.get('confidence'),
                        'bt_score':       entry.get('bt_score'),
                        'bt_score_norm':  entry.get('bt_score_norm'),
                        'blend_score':    entry.get('blend_score'),
                        'weights_used':   json.dumps(entry.get('weights_used', {})),
                        'generated_at':   datetime.utcnow().isoformat(),
                        'run_id_dag':     run_id,
                    })

    if not ranking_rows:
        logger.warning("⚠️  No rankings produced")

    rankings_tid = bq_table_id(run_id, 'rankings')
    if ranking_rows:
        bq_write(pd.DataFrame(ranking_rows), rankings_tid)
        logger.info(f"   ✓ rankings ({len(ranking_rows)} rows) → {rankings_tid}")
    else:
        bq_write(
            pd.DataFrame([{'anchor_user': None, 'rank': None, 'generated_at': datetime.utcnow().isoformat()}]),
            rankings_tid
        )

    users_ranked     = len(all_results)
    passages_covered = len(passages)
    logger.info(f"✅ Ranking complete — {users_ranked} users, {passages_covered} passages")

    ti.xcom_push(key='rankings_table_id', value=rankings_tid)
    return {
        'users_ranked':      users_ranked,
        'passages_covered':  passages_covered,
        'ranking_rows':      len(ranking_rows),
    }


# ════════════════════════════════════════════════════════════════
#  TASK 5 — UPLOAD TO FINAL BQ DATASET
#  compat_results already in final table (incremental upsert).
#  This task copies decompositions + rankings only.
# ════════════════════════════════════════════════════════════════

@log_task_execution("Upload to Final BigQuery Dataset")
def task_upload_to_bq(**context):
    ti     = context['task_instance']
    run_id = context['run_id']

    decomp_tid   = ti.xcom_pull(task_ids='compatibility', key='decomp_table_id')
    rankings_tid = ti.xcom_pull(task_ids='bradley_terry_ranking', key='rankings_table_id')

    logger.info(f"📤 Copying staging → {BQ_FINAL_DATASET}  [run={run_id}]")

    copy_map = {
        rankings_tid: f"{BQ_PROJECT}.{BQ_FINAL_DATASET}.rankings",
    }
    if decomp_tid:
        copy_map[decomp_tid] = f"{BQ_PROJECT}.{BQ_FINAL_DATASET}.decompositions"

    results = {'copied': ['compat_results (incremental upsert)'], 'errors': []}
    for src, dst in copy_map.items():
        if not src:
            continue
        try:
            bq_copy_table(src, dst)
            logger.info(f"   ✓ {src}  →  {dst}")
            results['copied'].append(dst.split('.')[-1])
        except Exception as e:
            logger.error(f"❌ Copy failed: {src} → {dst}  [{type(e).__name__}: {e}]")
            results['errors'].append(dst.split('.')[-1])

    logger.info(f"✅ Upload complete — {results['copied']}")
    if results['errors']:
        logger.warning(f"⚠️  Copy errors: {results['errors']}")

    return results


# ════════════════════════════════════════════════════════════════
#  TASK 6 — NOTIFY
# ════════════════════════════════════════════════════════════════

@log_task_execution("Inference Pipeline Notification")
def task_notify(**context):
    import pandas as pd
    ti     = context['task_instance']
    run_id = context['run_id']

    compat_stats  = ti.xcom_pull(task_ids='compatibility')              or {}
    val_results   = ti.xcom_pull(task_ids='validate_compat_output')     or {}
    bt_stats      = ti.xcom_pull(task_ids='bradley_terry_ranking')      or {}
    upload_stats  = ti.xcom_pull(task_ids='upload_to_bq')               or {}

    dag_run      = ti.dag_run
    failed_tasks = [t.task_id for t in dag_run.get_task_instances() if t.state == 'failed'] if dag_run else []
    status_emoji = "✅" if not failed_tasks else "⚠️"

    body = f"""
{status_emoji} MOMENT Inference Pipeline — Complete
{'='*50}
Run ID:       {run_id}
Time:         {datetime.now():%Y-%m-%d %H:%M:%S}

── Compatibility ──────────────────────────────
  Pairs processed : {compat_stats.get('total_pairs', 'N/A')}
  Successful      : {compat_stats.get('success_count', 'N/A')}
  Errors          : {compat_stats.get('error_count', 'N/A')}
  Error rate      : {compat_stats.get('error_rate', 'N/A')}

── Validation ─────────────────────────────────
  Status          : {val_results.get('status', 'N/A')}
  Valid rows      : {val_results.get('valid_rows', 'N/A')}
  Issues          : {val_results.get('issues', [])}

── Bradley-Terry Ranking ──────────────────────
  Users ranked    : {bt_stats.get('users_ranked', 'N/A')}
  Passages        : {bt_stats.get('passages_covered', 'N/A')}
  Ranking rows    : {bt_stats.get('ranking_rows', 'N/A')}

── Upload ─────────────────────────────────────
  Tables copied   : {upload_stats.get('copied', [])}
  Errors          : {upload_stats.get('errors', [])}

Failed tasks:  {failed_tasks or 'None'}
Final data:    {BQ_PROJECT}.{BQ_FINAL_DATASET}.*

Pipeline: load_test_data → compatibility → validate_compat → bradley_terry → upload → notify
Team: MOMENT Group 23 | DADS7305 MLOps
"""
    logger.info(body)

    bq_write(
        pd.DataFrame([{
            'run_id':       run_id,
            'body':         body,
            'failed_tasks': json.dumps(failed_tasks),
            'timestamp':    datetime.utcnow().isoformat(),
        }]),
        bq_table_id(run_id, 'inference_notification')
    )

    try:
        from utils import send_email_alert  # type: ignore
        send_email_alert("MOMENT Inference Pipeline Complete", body, TEAM_EMAILS)
        logger.info("✅ Email sent")
    except Exception as e:
        logger.debug(f"Email not sent (expected in dev): {type(e).__name__}")

    return {'status': 'sent', 'failed_tasks': failed_tasks}


# ════════════════════════════════════════════════════════════════
#  DAG DEFINITION
# ════════════════════════════════════════════════════════════════

default_args = {
    'owner':             'moment-group23',
    'depends_on_past':   False,
    'start_date':        datetime(2025, 2, 10),
    'email':             TEAM_EMAILS,
    'email_on_failure':  True,
    'email_on_retry':    False,
    'retries':           1,
    'retry_delay':       timedelta(minutes=3),
    'execution_timeout': timedelta(minutes=120),
}

dag = DAG(
    'moment_inference_pipeline',
    default_args=default_args,
    description='Inference pipeline: compatibility → BT ranking on interpretations_test',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=['moment', 'mlops', 'inference', 'compatibility', 'group23'],
)

load_data  = PythonOperator(task_id='load_test_data',          python_callable=task_load_test_data,          provide_context=True, dag=dag)
compat     = PythonOperator(task_id='compatibility',           python_callable=task_compatibility,           provide_context=True, dag=dag)
val_compat = PythonOperator(task_id='validate_compat_output',  python_callable=task_validate_compat_output,  provide_context=True, dag=dag)
bt_rank    = PythonOperator(task_id='bradley_terry_ranking',   python_callable=task_bradley_terry_ranking,   provide_context=True, dag=dag)
upload     = PythonOperator(task_id='upload_to_bq',            python_callable=task_upload_to_bq,            provide_context=True, dag=dag)
notify     = PythonOperator(task_id='notify',                  python_callable=task_notify,                  provide_context=True, trigger_rule=TriggerRule.ALL_DONE, dag=dag)

load_data >> compat >> val_compat >> bt_rank >> upload >> notify