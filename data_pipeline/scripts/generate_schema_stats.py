"""
TFDV Schema & Statistics Generation
Generates schema, statistics, and anomaly reports for processed data.
Callable from DAG via run_schema_stats() or standalone via __main__.

Path resolution: Uses REPO_ROOT env var (set by docker-compose) or defaults.
"""

import tensorflow_data_validation as tfdv
import pandas as pd
import json
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


def flatten_dataframe(df, dataset_name):
    """Aggressively flatten nested structures in dataframe."""
    logger.info(f"Flattening {dataset_name}...")
    df_flat = df.copy()


    for col in df_flat.columns:
        if df_flat[col].dtype == 'object':
            first_val = df_flat[col].dropna().iloc[0] if len(df_flat[col].dropna()) > 0 else None

            if isinstance(first_val, dict):
                nested_df = pd.json_normalize(df_flat[col])
                nested_df.columns = [f'{col}_{subcol}' for subcol in nested_df.columns]
                df_flat = pd.concat([df_flat.drop(col, axis=1), nested_df], axis=1)
            elif isinstance(first_val, list):
                df_flat[col] = df_flat[col].apply(
                    lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x)
                )

    for col in df_flat.columns:
        if df_flat[col].dtype == 'object':
            first_val = df_flat[col].dropna().iloc[0] if len(df_flat[col].dropna()) > 0 else None
            if isinstance(first_val, (dict, list)):
                df_flat[col] = df_flat[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x)
                )

    logger.info(f"  Final column count: {len(df_flat.columns)}")
    return df_flat


def process_dataset(name, fpath, schemas_dir, reports_dir, output_dir):
    """Run TFDV on a single dataset: flatten, generate stats, infer schema, detect anomalies."""
    logger.info(f"Processing {name}...")

    with open(fpath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    logger.info(f"  Loaded {len(df)} records")

    df_flat = flatten_dataframe(df, name)

    # Generate statistics & infer schema
    stats = tfdv.generate_statistics_from_dataframe(df_flat)
    schema = tfdv.infer_schema(statistics=stats)

    schema_path = os.path.join(schemas_dir, f'{name}_schema.pbtxt')
    tfdv.write_schema_text(schema, schema_path)
    logger.info(f"  Schema saved: {schema_path}")

    # Validate for anomalies
    anomalies = tfdv.validate_statistics(statistics=stats, schema=schema)
    anomaly_list = {}
    if anomalies.anomaly_info:
        for feature_name, anomaly_info in anomalies.anomaly_info.items():
            anomaly_list[feature_name] = anomaly_info.description
            logger.warning(f"  Anomaly in {feature_name}: {anomaly_info.description}")
        anomalies_path = os.path.join(reports_dir, f'{name}_anomalies.txt')
        with open(anomalies_path, 'w') as f:
            f.write(str(anomalies))
    else:
        logger.info(f"  No anomalies detected")

    # Build summary dict
    numeric_cols = df_flat.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_flat.select_dtypes(include=['object', 'bool']).columns

    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': name,
        'total_records': len(df_flat),
        'total_fields': len(df_flat.columns),
        'field_names': list(df_flat.columns),
        'schema_location': schema_path,
        'anomalies': anomaly_list,
    }

    if len(numeric_cols) > 0:
        summary['numeric_fields'] = list(numeric_cols)
        summary['numeric_statistics'] = df_flat[numeric_cols].describe().to_dict()

    if len(categorical_cols) > 0:
        summary['categorical_fields'] = list(categorical_cols)
        summary['categorical_summary'] = {}
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': int(df_flat[col].nunique()),
                'most_common': str(df_flat[col].mode()[0]) if len(df_flat[col].mode()) > 0 else None
            }

    summary_path = os.path.join(output_dir, f'{name}_statistics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Statistics saved: {summary_path}")

    return summary


def run_schema_stats(processed_dir, reports_dir):
    """
    Run TFDV schema & statistics on all processed datasets.
    Called by the DAG's schema_stats task.

    Args:
        processed_dir: path to data/processed/
        reports_dir: path to data/reports/
    Returns:
        dict with results for each dataset
    """
    schemas_dir = os.path.join(os.path.dirname(reports_dir), 'schemas')

    os.makedirs(schemas_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    datasets = {
        'moments': 'moments_processed.json',
        'books': 'books_processed.json',
        'users': 'users_processed.json',
    }

    results = {'generated_at': datetime.now().isoformat(), 'datasets': {}}

    for name, fname in datasets.items():
        fpath = os.path.join(processed_dir, fname)
        if not os.path.exists(fpath):
            logger.warning(f"File not found: {fpath}, skipping {name}")
            continue
        try:
            summary = process_dataset(name, fpath, schemas_dir, reports_dir, reports_dir)
            results['datasets'][name] = summary
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")
            results['datasets'][name] = {'error': str(e)}

    # Save combined report
    combined_path = os.path.join(reports_dir, 'schema_stats.json')
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Combined schema stats saved: {combined_path}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Path resolution — works in Docker and locally
    base_dir = os.environ.get('REPO_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    reports_dir = os.path.join(base_dir, 'data', 'reports')

    print(f"Input:  {processed_dir}")
    print(f"Output: {reports_dir}")

    results = run_schema_stats(processed_dir, reports_dir)
    print(f"\nDone: {len(results['datasets'])} datasets processed")