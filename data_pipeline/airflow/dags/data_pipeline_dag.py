"""
MOMENT Data Pipeline DAG
========================
Connects all team code into a single Airflow pipeline.

Flow:
    acquire → bias_detection → preprocessing → [schema_stats, validation] → notify
                                                  (parallel)

Scripts (by team member):
    data_acquisition.py  — Jyothssena (GCS data loading)
    bias_detection.py    — Santhosh   (demographic bias analysis)
    preprocessor.py      — Tanmayi    (text cleaning, validation, metrics, anomalies)
    anomalies.py         — Tanmayi    (batch anomaly detection)
    validation.py        — Jyothssena (schema validation)
    generate_schema_stats.py — Team   (TFDV schema & statistics)
    utils.py             — Team       (email/slack alerts)

DAG Author: Heet Patel | Group 23 | DADS7305 MLOps
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import sys
import os
import json
import logging

logger = logging.getLogger(__name__)

# ─── Path setup for imports ───
# Docker mounts: scripts/ → /opt/airflow/scripts/
SCRIPTS_DIR = os.environ.get(
    'SCRIPTS_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'scripts')
)
sys.path.insert(0, SCRIPTS_DIR)

# Config paths
PIPELINE_CONFIG = os.environ.get(
    'PIPELINE_CONFIG',
    os.path.join(os.path.dirname(SCRIPTS_DIR), 'config', 'config.yaml')
)
PREPROCESSING_CONFIG = os.environ.get(
    'PREPROCESSING_CONFIG',
    os.path.join(os.path.dirname(SCRIPTS_DIR), 'config', 'preprocessing_config.yaml')
)

# Repo root (for data paths that bias_detection.py uses)
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


# ════════════════════════════════════════════════════════════
#  TASK FUNCTIONS
# ════════════════════════════════════════════════════════════

def task_acquire_data(**context):
    """
    Task 1: Download data from GCS → save to data/raw/
    This is the FIRST step. All other tasks read from data/raw/.
    Falls back to checking local files if GCS unavailable.
    """
    logger.info("=" * 55)
    logger.info("  TASK 1: Data Acquisition (GCS → data/raw/)")
    logger.info("=" * 55)

    raw_dir = os.path.join(REPO_ROOT, 'data', 'raw')

    try:
        from data_acquisition import DataAcquisition
        acq = DataAcquisition(config_path=PIPELINE_CONFIG)

        # Step 1: Download from GCS to memory
        metadata = acq.run()

        # Step 2: Save to data/raw/ so other tasks can read files
        acq.save_to_local(output_dir=raw_dir)

        logger.info(f"✅ GCS → data/raw/: {metadata['num_files']} files, {metadata['total_rows']} rows")
        return metadata

    except Exception as e:
        logger.warning(f"GCS unavailable ({e}), checking if data/raw/ already has files...")

        # Fallback: check if files already exist in data/raw/
        required = [
            'user_interpretations.json',
            'user_data.csv',
            'passage_details.csv',
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(raw_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"GCS failed and local files missing from {raw_dir}: {missing}\n"
                f"Either fix GCS credentials or place files in data/raw/"
            )

        logger.info(f"✅ Local data/raw/ verified — {len(required)} files present")
        return {'source': 'local', 'raw_dir': raw_dir}


def task_bias_detection(**context):
    """
    Task 2: Bias detection on raw data (Santhosh's code).
    Analyzes: age, gender, reader type, personality, book distribution.
    """
    logger.info("=" * 55)
    logger.info("  TASK 2: Bias Detection")
    logger.info("=" * 55)

    # Change to repo root so bias_detection.py finds its hardcoded paths
    original_dir = os.getcwd()
    os.chdir(REPO_ROOT)

    try:
        from bias_detection import load_data, run_analysis

        df = load_data()
        if df is None:
            raise RuntimeError("Could not load data for bias detection")

        results = run_analysis(df)

        logger.info(f"✅ Bias detection complete")
        logger.info(f"   Age: {results['age']['max_dev']:.1f}% (intentional)")
        logger.info(f"   Gender: {results['gender']['max_dev']:.1f}%")
        logger.info(f"   Books: {results['book']['assessment']}")
        return results

    finally:
        os.chdir(original_dir)


def task_preprocessing(**context):
    """
    Task 3: Preprocessing pipeline (Tanmayi's code).
    READS FROM: data/raw/ (written by task_acquire_data)
    WRITES TO:  data/processed/ (read by schema_stats and validation)
    """
    logger.info("=" * 55)
    logger.info("  TASK 3: Preprocessing (data/raw/ → data/processed/)")
    logger.info("=" * 55)

    import yaml
    from preprocessor import (
        read_raw_data, lookup_books, process_books,
        process_users, process_moments_pass1, write_outputs
    )
    from anomalies import detect_anomalies

    # Load the unified config
    with open(PIPELINE_CONFIG, 'r') as f:
        cfg = yaml.safe_load(f)

    # Paths are relative — make them absolute from repo root
    for key in cfg['paths']['raw']:
        if not os.path.isabs(cfg['paths']['raw'][key]):
            cfg['paths']['raw'][key] = os.path.join(REPO_ROOT, cfg['paths']['raw'][key])
    for key in cfg['paths']['processed']:
        if not os.path.isabs(cfg['paths']['processed'][key]):
            cfg['paths']['processed'][key] = os.path.join(REPO_ROOT, cfg['paths']['processed'][key])

    # Ensure output directory exists
    os.makedirs(os.path.join(REPO_ROOT, 'data', 'processed'), exist_ok=True)

    # Run pipeline — each step passes data to the next IN MEMORY
    # Only disk I/O happens at read_raw_data (input) and write_outputs (output)
    logger.info("Step 1/6: Reading from data/raw/...")
    interpretations, passages, characters = read_raw_data(cfg)

    logger.info("Step 2/6: Looking up book metadata...")
    book_meta = lookup_books(cfg)

    logger.info("Step 3/6: Processing passages...")
    books = process_books(passages, book_meta, cfg)

    logger.info("Step 4/6: Processing users...")
    users = process_users(characters, interpretations, cfg)

    logger.info("Step 5/6: Processing moments + anomaly detection...")
    moments = process_moments_pass1(interpretations, book_meta, cfg)
    moments = detect_anomalies(moments, characters, cfg)

    logger.info("Step 6/6: Writing to data/processed/...")
    write_outputs(moments, books, users, cfg)

    valid = sum(1 for m in moments if m.get('is_valid', False))
    logger.info(f"✅ data/raw/ → data/processed/: {len(moments)} moments, {len(books)} books, {len(users)} users (valid: {valid})")

    return {'moments': len(moments), 'books': len(books), 'users': len(users), 'valid': valid}


def task_schema_stats(**context):
    """
    Task 4a: Generate schema and statistics.
    READS FROM: data/processed/ (written by preprocessing task)
    WRITES TO:  data/reports/
    """
    logger.info("=" * 55)
    logger.info("  TASK 4a: Schema & Stats (data/processed/ → data/reports/)")
    logger.info("=" * 55)

    import pandas as pd
    processed_dir = os.path.join(REPO_ROOT, 'data', 'processed')
    reports_dir = os.path.join(REPO_ROOT, 'data', 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    stats = {'generated_at': datetime.now().isoformat(), 'datasets': {}}

    for fname in ['moments_processed.json', 'books_processed.json', 'users_processed.json']:
        fpath = os.path.join(processed_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_json(fpath)
            ds = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'null_counts': df.isnull().sum().to_dict(),
            }
            # Numeric stats
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                ds['numeric_stats'] = df[numeric_cols].describe().to_dict()
            stats['datasets'][fname] = ds
            logger.info(f"  {fname}: {len(df)} rows, {len(df.columns)} cols")

    stats_path = os.path.join(reports_dir, 'schema_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info(f"✅ Schema stats saved to {stats_path}")
    return stats


def task_validation(**context):
    """
    Task 4b: Validate processed data quality.
    Runs in parallel with schema_stats.
    """
    logger.info("=" * 55)
    logger.info("  TASK 4b: Data Validation")
    logger.info("=" * 55)

    import pandas as pd
    processed_dir = os.path.join(REPO_ROOT, "data", "processed")
    reports_dir = os.path.join(REPO_ROOT, 'data', 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    results = {'timestamp': datetime.now().isoformat(), 'validations': {}, 'status': 'PASSED'}

    for fname in ['moments_processed.json', 'books_processed.json', 'users_processed.json']:
        fpath = os.path.join(processed_dir, fname)
        if not os.path.exists(fpath):
            results['validations'][fname] = {'status': 'MISSING'}
            results['status'] = 'FAILED'
            continue

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

        checks['status'] = 'PASSED' if checks['has_data'] and checks['null_count'] == 0 else 'WARNING'
        results['validations'][fname] = checks
        logger.info(f"  {fname}: {checks['status']} ({checks['row_count']} rows, {checks['null_count']} nulls)")

    validation_path = os.path.join(reports_dir, 'validation_report.json')
    with open(validation_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"✅ Validation: {results['status']}")
    return results


def task_upload_to_gcs(**context):
    """
    Task 5: Upload processed data back to GCS.
    Uploads data/processed/ → gs://moment_data/preprocessed/
    """
    logger.info("=" * 55)
    logger.info("  TASK 5: Upload to GCS (data/processed/ → GCS)")
    logger.info("=" * 55)

    from google.cloud import storage as gcs_storage

    processed_dir = os.path.join(REPO_ROOT, 'data', 'processed')
    bucket_name = 'moment_data'
    gcs_prefix = 'preprocessed/'

    project = os.environ.get('GOOGLE_CLOUD_PROJECT', 'moment-486719')
    client = gcs_storage.Client(project=project)
    bucket = client.bucket(bucket_name)

    uploaded = []
    for fname in os.listdir(processed_dir):
        fpath = os.path.join(processed_dir, fname)
        if os.path.isfile(fpath):
            blob = bucket.blob(f"{gcs_prefix}{fname}")
            blob.upload_from_filename(fpath)
            logger.info(f"  Uploaded {fname} → gs://{bucket_name}/{gcs_prefix}{fname}")
            uploaded.append(fname)

    logger.info(f"✅ Uploaded {len(uploaded)} files to gs://{bucket_name}/{gcs_prefix}")
    return {'uploaded': uploaded, 'destination': f'gs://{bucket_name}/{gcs_prefix}'}


def task_notify(**context):
    """
    Task 5: Pipeline notification. Runs on ALL_DONE.
    """
    logger.info("=" * 55)
    logger.info("  TASK 5: Notification")
    logger.info("=" * 55)

    ti = context.get('task_instance')
    preprocess = ti.xcom_pull(task_ids='preprocessing') if ti else {}
    validation = ti.xcom_pull(task_ids='validation') if ti else {}

    reports_dir = os.path.join(REPO_ROOT, 'data', 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    body = f"""
MOMENT Data Pipeline — Complete
{'='*50}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Preprocessing: {preprocess}
Validation: {validation.get('status', 'N/A') if isinstance(validation, dict) else 'N/A'}

Pipeline: acquire → bias → preprocess → [schema + validate] → notify
Team: MOMENT Group 23 | DADS7305 MLOps
"""
    logger.info(body)

    with open(os.path.join(reports_dir, 'notification.txt'), 'w', encoding='utf-8') as f:
        f.write(body)

    # Try email (if configured)
    try:
        from utils import send_email_alert
        send_email_alert("MOMENT Pipeline Complete", body, TEAM_EMAILS)
    except Exception as e:
        logger.info(f"Email not sent (expected in local dev): {e}")

    logger.info("✅ Notification saved")


# ════════════════════════════════════════════════════════════
#  DAG DEFINITION
# ════════════════════════════════════════════════════════════

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
    'moment_data_pipeline',
    default_args=default_args,
    description='MOMENT: acquire → bias → preprocess → [schema + validate] (parallel) → upload → notify',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
    tags=['moment', 'mlops', 'data-pipeline', 'group23'],
)

# ── Tasks ──
acquire  = PythonOperator(task_id='acquire_data',     python_callable=task_acquire_data,   provide_context=True, dag=dag)
bias     = PythonOperator(task_id='bias_detection',    python_callable=task_bias_detection,  provide_context=True, dag=dag)
preproc  = PythonOperator(task_id='preprocessing',     python_callable=task_preprocessing,   provide_context=True, dag=dag)
schema   = PythonOperator(task_id='schema_stats',      python_callable=task_schema_stats,    provide_context=True, dag=dag)
validate = PythonOperator(task_id='validation',        python_callable=task_validation,      provide_context=True, dag=dag)
upload   = PythonOperator(task_id='upload_to_gcs',     python_callable=task_upload_to_gcs,   provide_context=True, dag=dag)
notify   = PythonOperator(task_id='notify',            python_callable=task_notify,          provide_context=True, trigger_rule=TriggerRule.ALL_DONE, dag=dag)

# ── Flow ──
# acquire → bias → preprocess → [schema + validate] (parallel) → upload → notify
acquire >> bias >> preproc >> [schema, validate] >> upload >> notify

if __name__ == "__main__":
    print("✅ DAG valid")
    for t in dag.tasks:
        print(f"  {t.task_id} → {[d.task_id for d in t.downstream_list]}")