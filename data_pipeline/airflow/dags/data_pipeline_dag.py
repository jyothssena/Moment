"""
MOMENT Data Pipeline DAG - BIGQUERY INTERMEDIATE STORE VERSION
==============================================================
Data flows entirely through BigQuery staging tables — no local disk,
no GCS intermediate files.

BQ layout:
    moment-486719.moments_staging_{run_id}.interpretations_raw
    moment-486719.moments_staging_{run_id}.passage_details_new
    moment-486719.moments_staging_{run_id}.user_details_new
    moment-486719.moments_staging_{run_id}.moments_processed
    moment-486719.moments_staging_{run_id}.books_processed
    moment-486719.moments_staging_{run_id}.users_processed

Final destination (unchanged):
    moment-486719.new_moments_processed.*

XCom usage: tasks push/pull only BQ table IDs (tiny strings), never raw records.

Flow:
    acquire → bias_detection → preprocessing → [schema_stats, validation] → upload → notify
                                               (parallel)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import sys, os, json, logging
from functools import wraps
import pandas as pd 

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

logger.addHandler(_make_handler(os.path.join(LOGS_DIR, 'pipeline_INFO.log'),    logging.INFO,    lambda r: r.levelno == logging.INFO))
logger.addHandler(_make_handler(os.path.join(LOGS_DIR, 'pipeline_WARNING.log'), logging.WARNING, lambda r: r.levelno == logging.WARNING))
logger.addHandler(_make_handler(os.path.join(LOGS_DIR, 'pipeline_ERROR.log'),   logging.ERROR))
logger.addHandler(_make_handler(os.path.join(LOGS_DIR, 'pipeline_ALL.log'),     logging.DEBUG))


# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

SCRIPTS_DIR = os.environ.get(
    'SCRIPTS_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'scripts')
)
sys.path.insert(0, SCRIPTS_DIR)

PIPELINE_CONFIG = os.environ.get(
    'PIPELINE_CONFIG',
    os.path.join(os.path.dirname(SCRIPTS_DIR), 'config', 'config.yaml')
)
REPO_ROOT = os.environ.get(
    'REPO_ROOT',
    os.path.dirname(os.path.dirname(SCRIPTS_DIR))
)

BQ_PROJECT       = os.environ.get('GOOGLE_CLOUD_PROJECT', '')
BQ_FINAL_DATASET = 'new_moments_processed'   # unchanged final destination

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
    """
    Sanitize run_id into a valid BQ dataset name.
    Airflow run_ids look like 'scheduled__2026-03-30T00:00:00+00:00' —
    BQ dataset names allow only letters, digits, underscores.
    """
    safe = run_id.replace('-', '_').replace(':', '_').replace('+', '_').replace('.', '_').replace('T', '_')
    return f"moments_staging_{safe}"[:1024]  # BQ dataset name max 1024 chars

def bq_table_id(run_id: str, table: str) -> str:
    return f"{BQ_PROJECT}.{staging_dataset(run_id)}.{table}"

def ensure_staging_dataset(run_id: str):
    """Create the per-run staging dataset if it doesn't exist."""
    from google.cloud import bigquery
    client = _bq_client()
    ds_id  = f"{BQ_PROJECT}.{staging_dataset(run_id)}"
    ds     = bigquery.Dataset(ds_id)
    ds.location = "US"
    client.create_dataset(ds, exists_ok=True)
    logger.debug(f"Staging dataset ready: {ds_id}")

def bq_write(df, table_id: str):
    """Write a DataFrame to a BQ table (overwrite). No local file."""
    from google.cloud import bigquery
    job = _bq_client().load_table_from_dataframe(
        df, table_id,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=True,
        )
    )
    job.result()
    logger.debug(f"Wrote {len(df)} rows → {table_id}")

def bq_read(table_id: str):
    """Read a BQ table into a DataFrame. No local file."""
    import pandas as pd
    df = _bq_client().query(f"SELECT * FROM `{table_id}`").to_dataframe()
    logger.debug(f"Read {len(df)} rows ← {table_id}")
    return df

def bq_copy_table(src_table_id: str, dst_table_id: str):
    """Server-side BQ table copy — no data leaves BigQuery."""
    from google.cloud import bigquery
    client = _bq_client()
    job = client.copy_table(
        src_table_id, dst_table_id,
        job_config=bigquery.CopyJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
    )
    job.result()
    logger.debug(f"Copied {src_table_id} → {dst_table_id}")


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
                if isinstance(e, (MemoryError, SystemError)):
                    logger.critical(f"🚨 CRITICAL system error in {task_name}")
                logger.exception("Full traceback:")
                raise
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════
#  TASK 1 — ACQUIRE
#  BQLoader already reads from BQ into DataFrames.
#  We write those DataFrames into a run-scoped staging dataset
#  so downstream tasks can query them independently.
#  XCom out: dict of { table_name: full BQ table ID }
# ════════════════════════════════════════════════════════════════

@log_task_execution("Data Acquisition")
def task_acquire_data(**context):
    run_id = context['run_id']
    logger.info(f"📥 Acquiring data from BigQuery  [run={run_id}]")

    from bq_loader import BQLoader  # type: ignore

    # Load source tables
    acq = BQLoader(config_path=PIPELINE_CONFIG)
    metadata = acq.run()
    dfs = acq.get_dataframes()  # dict[str, pd.DataFrame]

    total_rows = metadata.get('total_rows', 0)
    if total_rows < 100:
        logger.warning(f"⚠️  Low row count from BigQuery: {total_rows}")

    # Write each DataFrame into the run-scoped staging dataset
    ensure_staging_dataset(run_id)
    raw_table_ids = {}
    for table_name, df in dfs.items():
        tid = bq_table_id(run_id, table_name)
        bq_write(df, tid)
        raw_table_ids[table_name] = tid
        logger.info(f"   ✓ {table_name}: {len(df)} rows → {tid}")

    logger.info(f"✅ Acquisition complete — {len(raw_table_ids)} tables staged in BQ")
    context['task_instance'].xcom_push(key='raw_table_ids', value=raw_table_ids)
    return metadata


# ════════════════════════════════════════════════════════════════
#  TASK 2 — BIAS DETECTION
#  Reads raw staging tables from BQ, runs analysis in-memory.
#  XCom out: bias results dict (small — safe for XCom default)
# ════════════════════════════════════════════════════════════════

@log_task_execution("Bias Detection")
def task_bias_detection(**context):
    ti     = context['task_instance']
    run_id = context['run_id']
    raw_ids = ti.xcom_pull(task_ids='acquire_data', key='raw_table_ids')
    logger.info(f"raw_ids received: {raw_ids}")
    logger.info(f"⚖️  Bias analysis  [run={run_id}]")

    df = bq_read(raw_ids['interpretations_train'])
    logger.info(f"   Loaded {len(df)} records from {raw_ids['interpretations_train']}")
    char_df = bq_read(raw_ids['user_details_new'])
    df = df.merge(char_df, left_on='character_name', right_on='Name', how='left')
    logger.info(f"[OK] Merged: {len(df)} records")
    df = df.rename(columns={'book': 'book_title'})
    age_groups = []
    for age_val in df['Age']:
        if pd.isna(age_val):
            age_groups.append("Unknown")
        else:
            try:
                a = int(age_val)
                if a < 25: age_groups.append("18-24 (Gen Z)")
                elif a < 35: age_groups.append("25-34 (Millennial)")
                elif a < 45: age_groups.append("35-44 (Gen X/Mill)")
                else: age_groups.append("45+ (Gen X/Boom)")
            except:
                age_groups.append("Unknown")
        
    df['age_group'] = age_groups
    original_dir = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        from bias_detection import run_analysis  # type: ignore
        results = run_analysis(df)
    finally:
        os.chdir(original_dir)

    age_dev    = results['age']['max_dev']
    gender_dev = results['gender']['max_dev']
    logger.info(f"✅ Bias detection complete — age_dev={age_dev:.1f}%  gender_dev={gender_dev:.1f}%")
    if gender_dev > 20:
        logger.warning(f"⚠️  High gender bias: {gender_dev:.1f}%")
    if age_dev > 30:
        logger.warning(f"⚠️  Significant age bias: {age_dev:.1f}%")

    return results  # small dict, fine for XCom


# ════════════════════════════════════════════════════════════════
#  TASK 3 — PREPROCESSING
#  Reads raw staging tables, processes in-memory,
#  writes processed DataFrames back into staging dataset.
#  XCom out: dict of { "moments": BQ table ID, "books": ..., "users": ... }
# ════════════════════════════════════════════════════════════════

@log_task_execution("Preprocessing Pipeline")
def task_preprocessing(**context):
    import yaml  # type: ignore
    ti      = context['task_instance']
    run_id  = context['run_id']
    raw_ids = ti.xcom_pull(task_ids='acquire_data', key='raw_table_ids')

    logger.info(f"🔧 Preprocessing  [run={run_id}]")

    with open(PIPELINE_CONFIG, 'r') as f:
        cfg = yaml.safe_load(f)

    from preprocessor import lookup_books, process_books, process_users, process_moments_pass1  # type: ignore
    from anomalies import detect_anomalies  # type: ignore
    import pandas as pd

    # ── Read raw records from BQ staging ──
    logger.info("Step 1/5: Reading raw staging tables from BQ")
    interp_df  = bq_read(raw_ids['interpretations_train'])
    passage_df = bq_read(raw_ids['passage_details_new'])
    char_df    = bq_read(raw_ids['user_details_new'])

    # Convert to the list-of-dicts format your preprocessor expects
    interpretations = interp_df.to_dict('records')
    passages        = passage_df.to_dict('records')
    characters      = char_df.to_dict('records')

    logger.info(f"   interpretations={len(interpretations)}, passages={len(passages)}, characters={len(characters)}")

    if not interpretations:
        raise ValueError("Empty interpretations dataset")
    if len(interpretations) < 100:
        logger.warning(f"⚠️  Low interpretation count: {len(interpretations)}")

    # ── Process in-memory (unchanged logic) ──
    logger.info("Step 2/5: Book metadata lookup")
    book_meta = lookup_books(cfg)

    logger.info("Step 3/5: Processing passages → books")
    books = process_books(passages, book_meta, cfg)

    logger.info("Step 4/5: Processing users")
    users = process_users(characters, interpretations, cfg)
    if len(users) < 5:
        logger.warning(f"⚠️  Very few users: {len(users)}")

    logger.info("Step 5/5: Moments + anomaly detection")
    moments = process_moments_pass1(interpretations, book_meta, cfg)
    moments = detect_anomalies(moments, characters, cfg)
    valid   = sum(1 for m in moments if m.get('is_valid', False))
    invalid = len(moments) - valid
    if invalid / max(len(moments), 1) > 0.20:
        logger.error(f"❌ High invalid rate: {invalid/len(moments):.1%}")
    elif invalid:
        logger.warning(f"⚠️  {invalid} invalid moments ({invalid/len(moments):.1%})")

    # ── Write processed records into BQ staging ──
    processed_ids = {
        'moments': bq_table_id(run_id, 'moments_processed'),
        'books':   bq_table_id(run_id, 'books_processed'),
        'users':   bq_table_id(run_id, 'users_processed'),
    }
    bq_write(pd.DataFrame(moments), processed_ids['moments'])
    bq_write(pd.DataFrame(books),   processed_ids['books'])
    bq_write(pd.DataFrame(users),   processed_ids['users'])

    for name, tid in processed_ids.items():
        logger.info(f"   ✓ {name} → {tid}")

    logger.info(f"✅ Preprocessing complete — moments={len(moments)}, books={len(books)}, users={len(users)}, valid={valid}")
    ti.xcom_push(key='processed_table_ids', value=processed_ids)
    return {'moments': len(moments), 'books': len(books), 'users': len(users), 'valid': valid}


# ════════════════════════════════════════════════════════════════
#  TASK 4a — SCHEMA & STATS (TFDV)
#  Reads processed staging tables, runs TFDV in-memory.
#  Writes stats/schema back to BQ as JSON records (no local files).
# ════════════════════════════════════════════════════════════════

@log_task_execution("Schema & Statistics (TFDV)")
def task_schema_stats(**context):
    import pandas as pd
    ti   = context['task_instance']
    run_id = context['run_id']
    proc_ids = ti.xcom_pull(task_ids='preprocessing', key='processed_table_ids')

    logger.info(f"📊 TFDV schema + stats  [run={run_id}]")

    import tensorflow_data_validation as tfdv  # type: ignore

    results = {'datasets': {}}
    schema_rows = []  # we'll write these back to BQ

    for name, tid in proc_ids.items():
        try:
            df     = bq_read(tid)
            stats  = tfdv.generate_statistics_from_dataframe(df)
            schema = tfdv.infer_schema(stats)
            anomalies = tfdv.validate_statistics(stats, schema)

            anom_dict = {f.name: f.description for f in anomalies.anomaly_info.values()}
            summary   = {
                'total_records': len(df),
                'total_fields':  len(df.columns),
                'anomalies':     anom_dict,
            }
            results['datasets'][name] = summary

            # Store serialised schema as a BQ row for lineage
            schema_rows.append({
                'dataset':   name,
                'run_id':    run_id,
                'schema_pb': schema.SerializeToString().hex(),  # hex-encode bytes for BQ STRING
                'anomalies': json.dumps(anom_dict),
                'recorded_at': datetime.now().isoformat(),
            })

            if anom_dict:
                logger.warning(f"⚠️  {name}: {len(anom_dict)} TFDV anomalies")
                for feat, desc in anom_dict.items():
                    logger.warning(f"     {feat}: {desc}")
            else:
                logger.info(f"   {name}: {len(df)} records, {len(df.columns)} fields — no anomalies")

        except Exception as e:
            logger.error(f"❌ {name} TFDV failed: {e}")
            results['datasets'][name] = {'error': str(e)}

    # Persist schemas to BQ staging for lineage (no local write)
    if schema_rows:
        import pandas as pd
        schema_tid = bq_table_id(run_id, 'tfdv_schemas')
        bq_write(pd.DataFrame(schema_rows), schema_tid)
        logger.info(f"   Schemas → {schema_tid}")

    logger.info("✅ TFDV complete")
    return results


# ════════════════════════════════════════════════════════════════
#  TASK 4b — VALIDATION
#  Reads processed staging tables, validates in-memory,
#  writes validation report back to BQ.
# ════════════════════════════════════════════════════════════════

@log_task_execution("Data Validation")
def task_validation(**context):
    import pandas as pd
    ti       = context['task_instance']
    run_id   = context['run_id']
    proc_ids = ti.xcom_pull(task_ids='preprocessing', key='processed_table_ids')

    logger.info(f"✓ Data validation  [run={run_id}]")

    results = {'timestamp': datetime.now().isoformat(), 'validations': {}, 'status': 'PASSED'}

    NAME_MAP = {
        'moments_processed.json': 'moments',
        'books_processed.json':   'books',
        'users_processed.json':   'users',
    }

    for fname, key in NAME_MAP.items():
        if key not in proc_ids:
            logger.error(f"❌ BQ table ID missing for {key}")
            results['validations'][fname] = {'status': 'MISSING'}
            results['status'] = 'FAILED'
            continue

        df = bq_read(proc_ids[key])
        checks = {
            'row_count':  len(df),
            'null_count': int(df.isnull().sum().sum()),
            'has_data':   len(df) > 0,
        }

        if key == 'moments':
            checks['has_interpretation_id'] = 'interpretation_id' in df.columns
            checks['has_is_valid']           = 'is_valid' in df.columns
            if 'is_valid' in df.columns:
                checks['valid_count'] = int(df['is_valid'].sum())
                checks['valid_rate']  = round(float(df['is_valid'].mean()), 3)
                if checks['valid_rate'] < 0.8:
                    logger.warning(f"⚠️  Low valid rate: {checks['valid_rate']:.1%}")

        if not checks['has_data']:
            checks['status'] = 'FAILED'
            logger.error(f"❌ {fname} has no data")
        elif checks['null_count'] > 0:
            checks['status'] = 'WARNING'
            logger.warning(f"⚠️  {fname}: {checks['null_count']} null values")
        else:
            checks['status'] = 'PASSED'
            logger.info(f"   {fname}: PASSED ({checks['row_count']} rows, 0 nulls)")

        results['validations'][fname] = checks

    statuses = [v['status'] for v in results['validations'].values()]
    if 'FAILED' in statuses:
        results['status'] = 'FAILED'
        logger.error("❌ Validation FAILED")
    elif 'WARNING' in statuses:
        results['status'] = 'WARNING'
        logger.warning("⚠️  Validation passed with warnings")
    else:
        logger.info("✅ All validation checks PASSED")

    # Write report to BQ staging (no local file)
    report_tid = bq_table_id(run_id, 'validation_report')
    bq_write(
        pd.DataFrame([{
            'run_id':     run_id,
            'status':     results['status'],
            'timestamp':  results['timestamp'],
            'detail':     json.dumps(results['validations'], default=str),
        }]),
        report_tid
    )
    logger.info(f"   Report → {report_tid}")

    return results


# ════════════════════════════════════════════════════════════════
#  TASK 5 — UPLOAD TO FINAL BIGQUERY DATASET
#  Server-side BQ table copy: staging → new_moments_processed.
#  Nothing leaves BigQuery.
# ════════════════════════════════════════════════════════════════

@log_task_execution("Upload to Final BigQuery Dataset")
def task_upload_to_bq(**context):
    ti       = context['task_instance']
    run_id   = context['run_id']
    proc_ids = ti.xcom_pull(task_ids='preprocessing', key='processed_table_ids')

    logger.info(f"📤 Copying staging → {BQ_FINAL_DATASET}  [run={run_id}]")

    FINAL_TABLE_MAP = {
        'moments': 'moments_processed',
        'books':   'books_processed',
        'users':   'users_processed',
    }

    results = {'copied': [], 'errors': []}

    for key, final_table in FINAL_TABLE_MAP.items():
        if key not in proc_ids:
            logger.warning(f"⚠️  Skipping {key} — table ID not found in XCom")
            continue

        src = proc_ids[key]
        dst = f"{BQ_PROJECT}.{BQ_FINAL_DATASET}.{final_table}"

        try:
            bq_copy_table(src, dst)
            logger.info(f"   ✓ {src}  →  {dst}")
            results['copied'].append(final_table)
        except Exception as e:
            logger.error(f"❌ Copy failed for {key}: {type(e).__name__}: {e}")
            results['errors'].append(final_table)

    logger.info(f"✅ Upload complete — {len(results['copied'])} tables copied to {BQ_FINAL_DATASET}")
    if results['errors']:
        logger.warning(f"⚠️  Errors: {results['errors']}")

    return results


# ════════════════════════════════════════════════════════════════
#  TASK 6 — NOTIFY
# ════════════════════════════════════════════════════════════════

@log_task_execution("Pipeline Notification")
def task_notify(**context):
    ti         = context['task_instance']
    run_id     = context['run_id']
    preprocess = ti.xcom_pull(task_ids='preprocessing')  or {}
    validation = ti.xcom_pull(task_ids='validation')     or {}
    upload     = ti.xcom_pull(task_ids='upload_to_bq')   or {}

    dag_run      = ti.dag_run
    failed_tasks = [t.task_id for t in dag_run.get_task_instances() if t.state == 'failed'] if dag_run else []
    if failed_tasks:
        logger.warning(f"⚠️  {len(failed_tasks)} tasks failed: {failed_tasks}")

    status_emoji = "✅" if not failed_tasks else "⚠️"
    staging_ds   = f"{BQ_PROJECT}.{staging_dataset(run_id)}"

    body = f"""
{status_emoji} MOMENT Data Pipeline - Complete
{'='*50}
Run ID:     {run_id}
Time:       {datetime.now():%Y-%m-%d %H:%M:%S}

Preprocessing:  {preprocess}
Validation:     {validation.get('status', 'N/A')}
BQ copied:      {len(upload.get('copied', []))} tables → {BQ_FINAL_DATASET}
Failed tasks:   {failed_tasks or 'None'}

Staging data:   {staging_ds}.*
Final data:     {BQ_PROJECT}.{BQ_FINAL_DATASET}.*

Pipeline: acquire → bias → preprocess → [schema + validate] → upload → notify
Team: MOMENT Group 23 | DADS7305 MLOps
"""
    logger.info(body)

    # Write notification to BQ staging
    import pandas as pd
    bq_write(
        pd.DataFrame([{'run_id': run_id, 'body': body, 'failed_tasks': json.dumps(failed_tasks)}]),
        bq_table_id(run_id, 'notification')
    )

    try:
        from utils import send_email_alert  # type: ignore
        send_email_alert("MOMENT Pipeline Complete", body, TEAM_EMAILS)
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
    'retries':           2,
    'retry_delay':       timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=30),
}

dag = DAG(
    'moment_team_pipeline_BQ',
    default_args=default_args,
    description='BigQuery intermediate store — no local writes, no GCS intermediates',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
    tags=['moment', 'mlops', 'data-pipeline', 'group23', 'BQ'],
)

acquire  = PythonOperator(task_id='acquire_data',   python_callable=task_acquire_data,   provide_context=True, dag=dag)
bias     = PythonOperator(task_id='bias_detection', python_callable=task_bias_detection, provide_context=True, dag=dag)
preproc  = PythonOperator(task_id='preprocessing',  python_callable=task_preprocessing,  provide_context=True, dag=dag)
schema   = PythonOperator(task_id='schema_stats',   python_callable=task_schema_stats,   provide_context=True, dag=dag)
validate = PythonOperator(task_id='validation',     python_callable=task_validation,     provide_context=True, dag=dag)
upload   = PythonOperator(task_id='upload_to_bq',   python_callable=task_upload_to_bq,   provide_context=True, dag=dag)
notify   = PythonOperator(task_id='notify',         python_callable=task_notify,         provide_context=True, trigger_rule=TriggerRule.ALL_DONE, dag=dag)

acquire >> bias >> preproc >> [schema, validate] >> upload >> notify