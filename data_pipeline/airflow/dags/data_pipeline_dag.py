"""
MOMENT Data Pipeline DAG - MULTI-LEVEL LOGGING VERSION
=======================================================
Enhanced with INFO, WARNING, ERROR, DEBUG logging levels.

Flow:
    acquire → bias_detection → preprocessing → [schema_stats, validation] → upload → notify
                                              (parallel)

Scripts (by team member):
    data_acquisition.py  – Jyothssena (GCS data loading)
    bias_detection.py    – Santhosh   (demographic bias analysis)
    preprocessor.py      – Tanmayi    (text cleaning, validation, metrics, anomalies)
    anomalies.py         – Tanmayi    (batch anomaly detection)
    validation.py        – Jyothssena (schema validation)
    generate_schema_stats.py – Team   (TFDV schema & statistics)
    utils.py             – Team       (email/slack alerts)

DAG Author: Heet Patel | Enhanced Logging: Greta | Group 23 | DADS7305 MLOps
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import sys
import os
import json
import logging
from functools import wraps

# ════════════════════════════════════════════════════════════════
#  MULTI-LEVEL LOGGING CONFIGURATION
# ════════════════════════════════════════════════════════════════

logger = logging.getLogger('airflow.task')

# Configure logging directory
LOGS_DIR = os.environ.get('LOGS_DIR', '/opt/airflow/logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Create separate handlers for each log level
info_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'team_pipeline_INFO.log'))
info_handler.setLevel(logging.INFO)
info_handler.addFilter(lambda record: record.levelno == logging.INFO)

warning_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'team_pipeline_WARNING.log'))
warning_handler.setLevel(logging.WARNING)
warning_handler.addFilter(lambda record: record.levelno == logging.WARNING)

error_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'team_pipeline_ERROR.log'))
error_handler.setLevel(logging.ERROR)
error_handler.addFilter(lambda record: record.levelno >= logging.ERROR)

all_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'team_pipeline_ALL.log'))
all_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
for handler in [info_handler, warning_handler, error_handler, all_handler]:
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("=" * 70)
logger.info("🔧 MOMENT Team Pipeline - Multi-Level Logging Initialized")
logger.info("   Log Files: team_pipeline_INFO.log, WARNING.log, ERROR.log, ALL.log")
logger.info("=" * 70)


# ════════════════════════════════════════════════════════════════
#  MULTI-LEVEL LOGGING DECORATOR
# ════════════════════════════════════════════════════════════════

def log_task_execution(task_name):
    """Enhanced decorator with INFO, WARNING, ERROR, DEBUG logging."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            logger.debug(f"🔍 DEBUG: Entering {task_name}")
            logger.info("=" * 70)
            logger.info(f"🚀 INFO: Starting task: {task_name}")
            logger.info(f"   Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # WARNING: Check for slow execution
                if duration > 300:
                    logger.warning(f"⚠️  WARNING: {task_name} took {duration:.1f}s (>5min threshold)")
                    logger.warning(f"   Consider optimizing this task")
                
                logger.info("=" * 70)
                logger.info(f"✅ INFO: Task completed: {task_name}")
                logger.info(f"   Duration: {duration:.2f}s ({duration/60:.2f}min)")
                if result and isinstance(result, dict):
                    logger.info(f"   Result summary: {list(result.keys())}")
                logger.info("=" * 70)
                logger.debug(f"🔍 DEBUG: Exiting {task_name} successfully")
                
                return result
                
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                logger.error("=" * 70)
                logger.error(f"❌ ERROR: Task failed: {task_name}")
                logger.error(f"   Error: {type(e).__name__}: {str(e)}")
                logger.error(f"   Duration before failure: {duration:.2f}s")
                logger.error("=" * 70)
                
                if isinstance(e, (MemoryError, SystemError)):
                    logger.critical(f"🚨 CRITICAL: System error in {task_name}!")
                
                logger.exception("Full traceback:")
                raise
                
        return wrapper
    return decorator


# ─── Path setup for imports ───
SCRIPTS_DIR = os.environ.get(
    'SCRIPTS_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'scripts')
)
sys.path.insert(0, SCRIPTS_DIR)

PIPELINE_CONFIG = os.environ.get(
    'PIPELINE_CONFIG',
    os.path.join(os.path.dirname(SCRIPTS_DIR), 'config', 'config.yaml')
)
PREPROCESSING_CONFIG = os.environ.get(
    'PREPROCESSING_CONFIG',
    os.path.join(os.path.dirname(SCRIPTS_DIR), 'config', 'preprocessing_config.yaml')
)

REPO_ROOT = os.environ.get(
    'REPO_ROOT',
    os.path.dirname(os.path.dirname(SCRIPTS_DIR))
)

TEAM_EMAILS = [
    'chandrasekar.s@northeastern.edu',
    'vasisht.h@northeastern.edu',
    'shurpali.t@northeastern.edu',
    'sreenivaasan.j@northeastern.edu',
    'patel.heetp@northeastern.edu',
    'wang.gre@northeastern.edu',
]

logger.debug(f"🔍 DEBUG: Configuration loaded")
logger.debug(f"   SCRIPTS_DIR: {SCRIPTS_DIR}")
logger.debug(f"   REPO_ROOT: {REPO_ROOT}")


# ════════════════════════════════════════════════════════════════
#  TASK FUNCTIONS WITH MULTI-LEVEL LOGGING
# ════════════════════════════════════════════════════════════════

@log_task_execution("Data Acquisition")
def task_acquire_data(**context):
    """
    Task 1: Acquire raw data.

    Priority order:
      1. BigQuery staging tables (moment_raw.*) — fastest, no file I/O
      2. GCS download → save to data/raw/ + upload to BQ staging
      3. Local data/raw/ fallback (already present from previous run)

    The BQ path is used when staging tables are populated (normal operation).
    The GCS path runs on first-time setup or when BQ tables are empty/missing.
    """
    logger.info("📥 INFO: Attempting to acquire data")

    raw_dir = os.path.join(REPO_ROOT, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    # ── Path 1: Try BigQuery staging tables ──────────────────────────────
    logger.info("   Checking BigQuery staging tables (moment_raw)...")
    try:
        from bq_loader import load_raw_from_bq
        bq_data = load_raw_from_bq()

        # BQ data is valid if all three tables have rows
        if (bq_data.get('interpretations') and
                bq_data.get('passages') and
                bq_data.get('user_data')):

            import json as _json
            # Write BQ data to local raw/ so preprocessor can read as normal
            with open(os.path.join(raw_dir, 'user_interpretations.json'), 'w', encoding='utf-8') as f:
                _json.dump(bq_data['interpretations'], f, indent=2, ensure_ascii=False)
            import pandas as _pd
            _pd.DataFrame(bq_data['passages']).to_csv(
                os.path.join(raw_dir, 'passage_details.csv'), index=False)
            _pd.DataFrame(bq_data['user_data']).to_csv(
                os.path.join(raw_dir, 'user_data.csv'), index=False)

            total_rows = (len(bq_data['interpretations']) +
                          len(bq_data['passages']) +
                          len(bq_data['user_data']))
            logger.info(f"✅ INFO: Data loaded from BigQuery staging tables")
            logger.info(f"   interpretations={len(bq_data['interpretations'])}, "
                        f"passages={len(bq_data['passages'])}, "
                        f"user_data={len(bq_data['user_data'])}")

            if len(bq_data['interpretations']) < 100:
                logger.warning(f"⚠️  WARNING: Low interpretation count from BQ: "
                                f"{len(bq_data['interpretations'])}")

            return {
                'source':     'bigquery',
                'num_files':  3,
                'total_rows': total_rows,
                'files':      ['user_interpretations.json', 'passage_details.csv', 'user_data.csv'],
                'raw_dir':    raw_dir,
            }
        else:
            missing = [k for k, v in bq_data.items() if not v]
            logger.warning(f"⚠️  WARNING: BQ staging tables empty or missing: {missing}")
            logger.info("   Falling through to GCS download...")

    except Exception as e:
        logger.warning(f"⚠️  WARNING: BigQuery read failed ({type(e).__name__}) — trying GCS...")

    # ── Path 2: GCS download (original behaviour) ─────────────────────────
    try:
        from data_acquisition import DataAcquisition

        logger.debug(f"🔍 DEBUG: Loading DataAcquisition with config: {PIPELINE_CONFIG}")
        acq = DataAcquisition(config_path=PIPELINE_CONFIG)
        metadata = acq.run()

        # save_to_local also calls upload_raw_to_bq() inside bq_loader
        acq.save_to_local(output_dir=raw_dir)

        num_files  = metadata.get('num_files', 0)
        total_rows = metadata.get('total_rows', 0)

        logger.info(f"✅ INFO: GCS → data/raw/ + BigQuery staging successful")
        logger.info(f"   Files: {num_files}, Rows: {total_rows}")

        if total_rows < 100:
            logger.warning(f"⚠️  WARNING: Low row count from GCS: {total_rows}")

        return {**metadata, 'source': 'gcs'}

    except Exception as e:
        logger.warning(f"⚠️  WARNING: GCS unavailable - {type(e).__name__}")
        logger.info("   Falling back to local file verification...")

    # ── Path 3: Local fallback ────────────────────────────────────────────
    required = ['user_interpretations.json', 'user_data.csv', 'passage_details.csv']
    missing  = [f for f in required if not os.path.exists(os.path.join(raw_dir, f))]

    if missing:
        logger.error(f"❌ ERROR: Missing local files in {raw_dir}: {missing}")
        logger.error(f"   Fix GCS credentials or place files in data/raw/")
        raise FileNotFoundError(f"Missing: {missing}")

    logger.info(f"✅ INFO: Local data/raw/ verified — {len(required)} files present")
    return {'source': 'local', 'raw_dir': raw_dir}


@log_task_execution("Bias Detection")
def task_bias_detection(**context):
    """Task 2: Bias detection on raw data (Santhosh's code)."""
    logger.info("⚖️ INFO: Analyzing demographic bias in dataset")
    
    original_dir = os.getcwd()
    logger.debug(f"🔍 DEBUG: Changing to repo root: {REPO_ROOT}")
    os.chdir(REPO_ROOT)

    try:
        from bias_detection import load_data, run_analysis

        logger.debug("🔍 DEBUG: Loading data for bias analysis")
        df = load_data()
        
        if df is None:
            logger.error("❌ ERROR: Failed to load data for bias detection")
            raise RuntimeError("Could not load data for bias detection")
        
        logger.info(f"   Loaded {len(df)} records for analysis")
        
        results = run_analysis(df)
        
        # Check bias levels
        age_dev = results['age']['max_dev']
        gender_dev = results['gender']['max_dev']
        
        logger.info(f"✅ INFO: Bias detection complete")
        logger.info(f"   Age deviation: {age_dev:.1f}% (intentional)")
        logger.info(f"   Gender deviation: {gender_dev:.1f}%")
        logger.info(f"   Book distribution: {results['book']['assessment']}")
        
        # WARNING: High bias detected
        if gender_dev > 20:
            logger.warning(f"⚠️  WARNING: High gender bias: {gender_dev:.1f}%")
            logger.warning(f"   Consider data balancing strategies")
        
        if age_dev > 30:
            logger.warning(f"⚠️  WARNING: Significant age bias: {age_dev:.1f}%")
        
        logger.debug(f"🔍 DEBUG: Full bias results: {json.dumps(results, indent=2, default=str)}")
        
        return results

    finally:
        os.chdir(original_dir)
        logger.debug(f"🔍 DEBUG: Restored working directory")


@log_task_execution("Preprocessing Pipeline")
def task_preprocessing(**context):
    """Task 3: Preprocessing (data/raw/ → data/processed/)"""
    logger.info("INFO: Running full preprocessing pipeline")
    logger.info("   Flow: read → lookup → process → anomalies → write")
    
    import yaml
    from preprocessor import (
        read_raw_data, lookup_books, process_books,
        process_users, process_moments_pass1, write_outputs
    )
    from anomalies import detect_anomalies

    # Load config
    logger.debug(f"🔍 DEBUG: Loading config from {PIPELINE_CONFIG}")
    with open(PIPELINE_CONFIG, 'r') as f:
        cfg = yaml.safe_load(f)

    # Adjust paths
    for key in cfg['paths']['raw']:
        if not os.path.isabs(cfg['paths']['raw'][key]):
            cfg['paths']['raw'][key] = os.path.join(REPO_ROOT, cfg['paths']['raw'][key])
    for key in cfg['paths']['processed']:
        if not os.path.isabs(cfg['paths']['processed'][key]):
            cfg['paths']['processed'][key] = os.path.join(REPO_ROOT, cfg['paths']['processed'][key])

    os.makedirs(os.path.join(REPO_ROOT, 'data', 'processed'), exist_ok=True)

    # Step 1: Read raw data
    logger.info("Step 1/6: Reading from data/raw/")
    interpretations, passages, characters = read_raw_data(cfg)
    logger.info(f"   Loaded: {len(interpretations)} interpretations, {len(passages)} passages, {len(characters)} characters")
    
    if len(interpretations) == 0:
        logger.error("❌ ERROR: No interpretations loaded")
        raise ValueError("Empty interpretations dataset")
    
    if len(interpretations) < 100:
        logger.warning(f"⚠️  WARNING: Low interpretation count: {len(interpretations)}")

    # Step 2: Lookup books
    logger.info("Step 2/6: Looking up book metadata")
    book_meta = lookup_books(cfg)
    logger.info(f"   Found {len(book_meta)} books")
    logger.debug(f"🔍 DEBUG: Book metadata keys: {list(book_meta.keys()) if book_meta else 'None'}")

    # Step 3: Process passages
    logger.info("Step 3/6: Processing passages into books")
    books = process_books(passages, book_meta, cfg)
    logger.info(f"   Processed {len(books)} book records")

    # Step 4: Process users
    logger.info("Step 4/6: Processing user profiles")
    users = process_users(characters, interpretations, cfg)
    logger.info(f"   Processed {len(users)} user profiles")
    
    if len(users) < 5:
        logger.warning(f"⚠️  WARNING: Very few users: {len(users)}")
        logger.warning(f"   Limited diversity for analysis")

    # Step 5: Process moments + anomalies
    logger.info("Step 5/6: Processing moments and detecting anomalies")
    moments = process_moments_pass1(interpretations, book_meta, cfg)
    logger.info(f"   Initial moments: {len(moments)}")
    
    moments = detect_anomalies(moments, characters, cfg)
    valid = sum(1 for m in moments if m.get('is_valid', False))
    invalid = len(moments) - valid
    
    logger.info(f"   After anomaly detection: {valid} valid, {invalid} invalid")
    
    if invalid > 0:
        invalid_pct = (invalid / len(moments)) * 100
        logger.warning(f"⚠️  WARNING: {invalid} invalid moments ({invalid_pct:.1f}%)")
        
        if invalid_pct > 20:
            logger.error(f"❌ ERROR: High invalid rate: {invalid_pct:.1f}%")
            logger.error(f"   Data quality issues detected")

    # Step 6: Write outputs
    logger.info("Step 6/6: Writing to data/processed/")
    logger.debug(f"🔍 DEBUG: Output paths configured in config")
    write_outputs(moments, books, users, cfg)
    
    logger.info(f"✅ INFO: Preprocessing complete")
    logger.info(f"   Moments: {len(moments)}, Books: {len(books)}, Users: {len(users)}")
    logger.info(f"   Valid rate: {valid}/{len(moments)} ({(valid/len(moments)*100):.1f}%)")

    return {'moments': len(moments), 'books': len(books), 'users': len(users), 'valid': valid}


@log_task_execution("Schema & Statistics (TFDV)")
def task_schema_stats(**context):
    """Task 4a: Generate schema, statistics, and anomaly detection using TFDV."""
    logger.info("📊 INFO: Generating schema and statistics with TFDV")

    from generate_schema_stats import run_schema_stats

    processed_dir = os.path.join(REPO_ROOT, 'data', 'processed')
    reports_dir = os.path.join(REPO_ROOT, 'data', 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    results = run_schema_stats(processed_dir, reports_dir)

    for name, summary in results.get('datasets', {}).items():
        if 'error' in summary:
            logger.error(f"❌ ERROR: {name} failed: {summary['error']}")
        else:
            anomalies = summary.get('anomalies', {})
            logger.info(f"   {name}: {summary['total_records']} records, {summary['total_fields']} fields")
            if anomalies:
                logger.warning(f"⚠️  WARNING: {name} has {len(anomalies)} anomalies")
                for feat, desc in anomalies.items():
                    logger.warning(f"     - {feat}: {desc}")

    logger.info(f"✅ INFO: TFDV schema stats complete")
    logger.info(f"   Analyzed {len(results.get('datasets', {}))} datasets")

    return results


@log_task_execution("Data Validation")
def task_validation(**context):
    """Task 4b: Validate processed data quality."""
    logger.info("✓ INFO: Starting data validation checks")
    
    import pandas as pd
    processed_dir = os.path.join(REPO_ROOT, "data", "processed")
    reports_dir = os.path.join(REPO_ROOT, 'data', 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    results = {'timestamp': datetime.now().isoformat(), 'validations': {}, 'status': 'PASSED'}

    for fname in ['moments_processed.json', 'books_processed.json', 'users_processed.json']:
        fpath = os.path.join(processed_dir, fname)
        
        if not os.path.exists(fpath):
            logger.error(f"❌ ERROR: Missing file for validation: {fname}")
            results['validations'][fname] = {'status': 'MISSING'}
            results['status'] = 'FAILED'
            continue

        logger.debug(f"🔍 DEBUG: Validating {fname}")
        df = pd.read_json(fpath)
        
        checks = {
            'row_count': len(df),
            'null_count': int(df.isnull().sum().sum()),
            'has_data': len(df) > 0,
        }

        if fname == 'moments_processed.json':
            checks['has_interpretation_id'] = 'interpretation_id' in df.columns
            checks['has_is_valid'] = 'is_valid' in df.columns
            
            if 'is_valid' in df.columns:
                checks['valid_count'] = int(df['is_valid'].sum())
                checks['valid_rate'] = round(float(df['is_valid'].mean()), 3)
                
                # WARNING: Low validation rate
                if checks['valid_rate'] < 0.8:
                    logger.warning(f"⚠️  WARNING: Low validation rate: {checks['valid_rate']:.1%}")
                    logger.warning(f"   Only {checks['valid_count']}/{checks['row_count']} moments are valid")

        # Determine status
        if not checks['has_data']:
            checks['status'] = 'FAILED'
            logger.error(f"❌ ERROR: {fname} has no data")
        elif checks['null_count'] > 0:
            checks['status'] = 'WARNING'
            logger.warning(f"⚠️  WARNING: {fname} has {checks['null_count']} null values")
        else:
            checks['status'] = 'PASSED'
            logger.info(f"   {fname}: PASSED ({checks['row_count']} rows, 0 nulls)")
        
        results['validations'][fname] = checks

    # Overall status
    if any(v['status'] == 'FAILED' for v in results['validations'].values()):
        results['status'] = 'FAILED'
        logger.error("❌ ERROR: Validation FAILED - critical issues found")
    elif any(v['status'] == 'WARNING' for v in results['validations'].values()):
        results['status'] = 'WARNING'
        logger.warning("⚠️  WARNING: Validation passed with warnings")
    else:
        logger.info("✅ INFO: All validation checks PASSED")

    validation_path = os.path.join(reports_dir, 'validation_report.json')
    with open(validation_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"   Report saved: {validation_path}")
    return results


@log_task_execution("Upload to GCS + BigQuery")
def task_upload_to_gcs(**context):
    """
    Task 5: Upload processed data to GCS (JSON backup) AND to BigQuery.

    Two parallel destinations:
      A) GCS  — gs://moment_data/preprocessed/  (JSON files, existing behaviour)
      B) BQ   — moment-486719.moment_processed.*  (queryable tables, new)

    Both succeed/fail independently — one failure does not block the other.
    The BigQuery write via bq_loader.write_processed_to_bq() is the primary
    destination; GCS acts as a durable JSON backup.
    """
    logger.info("✓ INFO: Uploading processed data to GCS + BigQuery")

    processed_dir = os.path.join(REPO_ROOT, 'data', 'processed')
    bucket_name   = 'moment_data'
    gcs_prefix    = 'preprocessed/'

    logger.info(f"   Source: {processed_dir}")
    logger.info(f"   GCS destination:       gs://{bucket_name}/{gcs_prefix}")
    logger.info(f"   BigQuery destination:  moment-486719.moment_processed.*")

    result: dict = {'gcs': {}, 'bigquery': {}}

    # ── A) GCS upload (existing behaviour, unchanged) ────────────────────
    try:
        from google.cloud import storage as gcs_storage
        project = os.environ.get('GOOGLE_CLOUD_PROJECT', 'moment-486719')
        client  = gcs_storage.Client(project=project)
        bucket  = client.bucket(bucket_name)

        uploaded = []
        for fname in os.listdir(processed_dir):
            fpath = os.path.join(processed_dir, fname)
            if os.path.isfile(fpath):
                file_size_mb = os.path.getsize(fpath) / (1024 * 1024)
                if file_size_mb > 50:
                    logger.warning(f"⚠️  WARNING: Large file: {fname} ({file_size_mb:.2f} MB)")
                blob = bucket.blob(f"{gcs_prefix}{fname}")
                blob.upload_from_filename(fpath)
                logger.info(f"   GCS ✓  {fname} → gs://{bucket_name}/{gcs_prefix}{fname}")
                uploaded.append(fname)

        logger.info(f"✅ INFO: GCS upload complete — {len(uploaded)} files")
        result['gcs'] = {'uploaded': uploaded, 'destination': f'gs://{bucket_name}/{gcs_prefix}'}

    except Exception as e:
        logger.error(f"❌ ERROR: GCS upload failed: {type(e).__name__}: {str(e)}")
        logger.warning("   Processed data remains in data/processed/ — pipeline continues")
        result['gcs'] = {'uploaded': [], 'error': str(e)}

    # ── B) BigQuery upload (new) ─────────────────────────────────────────
    # Read the processed JSON files (just written by preprocessing) and
    # load them into BigQuery processed tables. bq_loader handles flattening
    # and schema autodetect.
    try:
        import json as _json
        from bq_loader import write_processed_to_bq

        moments_path = os.path.join(processed_dir, 'moments_processed.json')
        books_path   = os.path.join(processed_dir, 'books_processed.json')
        users_path   = os.path.join(processed_dir, 'users_processed.json')

        moments, books, users = [], [], []

        if os.path.exists(moments_path):
            with open(moments_path, encoding='utf-8') as f:
                moments = _json.load(f)
        if os.path.exists(books_path):
            with open(books_path, encoding='utf-8') as f:
                books = _json.load(f)
        if os.path.exists(users_path):
            with open(users_path, encoding='utf-8') as f:
                users = _json.load(f)

        logger.info(f"   BQ writing: {len(moments)} moments, {len(books)} books, {len(users)} users")
        bq_success = write_processed_to_bq(moments, books, users)

        if bq_success:
            logger.info(
                f"✅ INFO: BigQuery upload complete — "
                f"moment_processed.moments ({len(moments)} rows), "
                f"moment_processed.books ({len(books)} rows), "
                f"moment_processed.users ({len(users)} rows)"
            )
            result['bigquery'] = {
                'status':  'success',
                'moments': len(moments),
                'books':   len(books),
                'users':   len(users),
                'dataset': 'moment-486719.moment_processed',
            }
        else:
            logger.warning("⚠️  WARNING: BigQuery upload failed — GCS backup intact")
            result['bigquery'] = {'status': 'failed'}

    except Exception as e:
        logger.error(f"❌ ERROR: BigQuery upload task failed: {type(e).__name__}: {str(e)}")
        result['bigquery'] = {'status': 'error', 'error': str(e)}

    logger.info(f"   Upload task complete — GCS: {result['gcs'].get('uploaded', [])}, "
                f"BQ: {result['bigquery'].get('status', 'skipped')}")
    return result


@log_task_execution("Pipeline Notification")
def task_notify(**context):
    """Task 6: Pipeline notification. Runs on ALL_DONE."""
    logger.info("INFO: Preparing pipeline completion notification")
    
    ti = context.get('task_instance')
    preprocess = ti.xcom_pull(task_ids='preprocessing') if ti else {}
    validation = ti.xcom_pull(task_ids='validation') if ti else {}
    upload = ti.xcom_pull(task_ids='upload_to_gcs') if ti else {}
    
    logger.debug(f"🔍 DEBUG: Preprocessing result: {preprocess}")
    logger.debug(f"🔍 DEBUG: Validation result: {validation}")
    logger.debug(f"🔍 DEBUG: Upload result: {upload}")

    # Check for failed tasks
    dag_run = ti.dag_run if ti else None
    failed_tasks = []
    if dag_run:
        failed_tasks = [t.task_id for t in dag_run.get_task_instances() if t.state == 'failed']
        if failed_tasks:
            logger.warning(f"⚠️  WARNING: {len(failed_tasks)} tasks failed")
            for task_id in failed_tasks:
                logger.warning(f"   - Failed: {task_id}")

    reports_dir = os.path.join(REPO_ROOT, 'data', 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    # Build notification
    status_emoji = "✅" if not failed_tasks else "⚠️"
    validation_status = validation.get('status', 'N/A') if isinstance(validation, dict) else 'N/A'
    upload_count = len(upload.get('uploaded', [])) if isinstance(upload, dict) else 0
    
    body = f"""
{status_emoji} MOMENT Data Pipeline – Complete
{'='*50}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Preprocessing: {preprocess}
Validation: {validation_status}
GCS Upload: {upload_count} files uploaded
Failed Tasks: {failed_tasks if failed_tasks else 'None'}

Pipeline: acquire → bias → preprocess → [schema + validate] → upload → notify
Team: MOMENT Group 23 | DADS7305 MLOps
"""
    logger.info(body)

    notification_path = os.path.join(reports_dir, 'notification.txt')
    with open(notification_path, 'w', encoding='utf-8') as f:
        f.write(body)
    
    logger.info(f"   Notification saved: {notification_path}")
    logger.info(f"   Recipients: {len(TEAM_EMAILS)} team members")

    # Try email
    try:
        from utils import send_email_alert
        send_email_alert("MOMENT Pipeline Complete", body, TEAM_EMAILS)
        logger.info("✅ INFO: Email notification sent")
    except Exception as e:
        logger.debug(f"🔍 DEBUG: Email not sent (expected in dev): {type(e).__name__}")

    # ── Write pipeline run report to BigQuery ─────────────────────────────
    # Stores per-run metrics in moment_reports.pipeline_runs for trend analysis.
    try:
        import json as _json
        from bq_loader import upload_reports_to_bq

        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Load reports generated earlier in this run
        reports_dir = os.path.join(REPO_ROOT, 'data', 'reports')
        schema_stats, validation_report = None, None

        schema_path = os.path.join(reports_dir, 'schema_stats.json')
        if os.path.exists(schema_path):
            with open(schema_path, encoding='utf-8') as f:
                schema_stats = _json.load(f)

        val_path = os.path.join(reports_dir, 'validation_report.json')
        if os.path.exists(val_path):
            with open(val_path, encoding='utf-8') as f:
                validation_report = _json.load(f)

        # Rebuild anomaly summary from preprocessing XCom result
        anomaly_summary = None
        if isinstance(preprocess, dict):
            anomaly_summary = {
                'word_count_outliers':  preprocess.get('anomalies_wc', None),
                'readability_outliers': preprocess.get('anomalies_read', None),
                'duplicates':           preprocess.get('anomalies_dup', None),
                'style_mismatches':     preprocess.get('anomalies_style', None),
            }

        bq_report_ok = upload_reports_to_bq(
            run_id=run_id,
            schema_stats=schema_stats,
            validation_report=validation_report,
            bias_results=isinstance(bias, dict) and bias or None,
            anomaly_summary=anomaly_summary,
        )
        if bq_report_ok:
            logger.info(f"✅ INFO: Pipeline run report written to BigQuery (run_id={run_id})")
        else:
            logger.warning("⚠️  WARNING: BigQuery report write failed — notification.txt intact")
    except Exception as e:
        logger.debug(f"🔍 DEBUG: BigQuery report write skipped: {type(e).__name__}: {e}")

    return {'status': 'sent', 'failed_tasks': failed_tasks}


# ════════════════════════════════════════════════════════════════
#  DAG DEFINITION
# ════════════════════════════════════════════════════════════════

default_args = {
    'owner': 'moment-group23',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 10),
    'email': TEAM_EMAILS,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=30),
}

dag = DAG(
    'moment_team_pipeline_MULTILEVEL',
    default_args=default_args,
    description='⭐ MULTI-LEVEL LOGGING - MOMENT: acquire → bias → preprocess → [schema + validate] → upload → notify',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
    tags=['moment', 'mlops', 'data-pipeline', 'group23', 'MULTILEVEL', 'TEAM'],
)

logger.info("INFO: Initializing MOMENT Team Pipeline DAG (MULTILEVEL)")
logger.info(f"   Schedule: Daily")
logger.debug(f"🔍 DEBUG: Repo root: {REPO_ROOT}")

# ─── Task Definitions ───
acquire  = PythonOperator(task_id='acquire_data',     python_callable=task_acquire_data,   provide_context=True, dag=dag)
bias     = PythonOperator(task_id='bias_detection',   python_callable=task_bias_detection, provide_context=True, dag=dag)
preproc  = PythonOperator(task_id='preprocessing',    python_callable=task_preprocessing,  provide_context=True, dag=dag)
schema   = PythonOperator(task_id='schema_stats',     python_callable=task_schema_stats,   provide_context=True, dag=dag)
validate = PythonOperator(task_id='validation',       python_callable=task_validation,     provide_context=True, dag=dag)
upload   = PythonOperator(task_id='upload_to_gcs',    python_callable=task_upload_to_gcs,  provide_context=True, dag=dag)
notify   = PythonOperator(task_id='notify',           python_callable=task_notify,         provide_context=True, trigger_rule=TriggerRule.ALL_DONE, dag=dag)

# ─── Task Dependencies ───
acquire >> bias >> preproc >> [schema, validate] >> upload >> notify

logger.info("✅ INFO: DAG configured with 7 tasks in pipeline flow")

if __name__ == "__main__":
    print("✅ DAG valid - MOMENT Team Pipeline (MULTILEVEL)")
    for t in dag.tasks:
        print(f"  {t.task_id} → {[d.task_id for d in t.downstream_list]}")