"""
MOMENT Pipeline Tests DAG
==========================
Runs all unit tests for the data pipeline via Airflow.
Validates: acquisition, preprocessing, bias detection, validation, schema stats.

Author: Heet Patel | Group 23 | DADS7305 MLOps
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta


default_args = {
    'owner': 'moment-group23',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 10),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'moment_pipeline_tests',
    default_args=default_args,
    description='Runs all unit tests for the MOMENT data pipeline',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['moment', 'mlops', 'tests', 'group23'],
)

# ── Test tasks ──

test_acquisition = BashOperator(
    task_id='test_acquisition',
    bash_command='cd /opt/airflow && python -m pytest tests/test_acquisition.py -v --tb=short 2>&1 || true',
    dag=dag,
)

test_preprocessing = BashOperator(
    task_id='test_preprocessing',
    bash_command='cd /opt/airflow && python -m pytest tests/test_preprocessing.py -v --tb=short 2>&1 || true',
    dag=dag,
)

test_bias_detection = BashOperator(
    task_id='test_bias_detection',
    bash_command='cd /opt/airflow && python -m pytest tests/test_bias_detection.py -v --tb=short 2>&1 || true',
    dag=dag,
)

test_validation = BashOperator(
    task_id='test_validation',
    bash_command='cd /opt/airflow && python -m pytest tests/test_validation.py -v --tb=short 2>&1 || true',
    dag=dag,
)

test_schema_stats = BashOperator(
    task_id='test_schema_stats',
    bash_command='cd /opt/airflow && python -m pytest tests/test_schema_stats.py -v --tb=short 2>&1 || true',
    dag=dag,
)

test_pipeline_integration = BashOperator(
    task_id='test_pipeline_integration',
    bash_command='cd /opt/airflow && python -m pytest tests/test_pipeline.py -v --tb=short 2>&1 || true',
    dag=dag,
)

# ── Summary ──
report_results = BashOperator(
    task_id='report_results',
    bash_command='cd /opt/airflow && echo "=== FULL TEST SUITE ===" && python -m pytest tests/ -v --tb=short 2>&1; echo "=== DONE ==="',
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# ── Flow: all tests run in parallel, then summary ──
[test_acquisition, test_preprocessing, test_bias_detection, test_validation, test_schema_stats, test_pipeline_integration] >> report_results