import tensorflow_data_validation as tfdv
import pandas as pd
import json
from datetime import datetime
import os

# Set up paths
INPUT_DIR = 'data/input'
OUTPUT_DIR = 'data/output'
SCHEMA_DIR = 'schemas'
REPORTS_DIR = 'reports'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCHEMA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def flatten_dataframe(df, dataset_name):
    """Aggressively flatten nested structures in dataframe"""
    print(f"\n[Flattening] Processing {dataset_name}...")
    df_flat = df.copy()
    
    for col in df_flat.columns:
        if df_flat[col].dtype == 'object':
            first_val = df_flat[col].dropna().iloc[0] if len(df_flat[col].dropna()) > 0 else None
            
            # If it's a dict, flatten it
            if isinstance(first_val, dict):
                print(f"  ✓ Flattening dict column: {col}")
                nested_df = pd.json_normalize(df_flat[col])
                nested_df.columns = [f'{col}_{subcol}' for subcol in nested_df.columns]
                df_flat = pd.concat([df_flat.drop(col, axis=1), nested_df], axis=1)
            # If it's a list, convert to string
            elif isinstance(first_val, list):
                print(f"  ✓ Converting list to string: {col}")
                df_flat[col] = df_flat[col].apply(
                    lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x)
                )
    
    # Check for any remaining complex types and convert to JSON strings
    for col in df_flat.columns:
        if df_flat[col].dtype == 'object':
            first_val = df_flat[col].dropna().iloc[0] if len(df_flat[col].dropna()) > 0 else None
            if isinstance(first_val, (dict, list)):
                print(f"  ✓ Converting remaining complex type to JSON string: {col}")
                df_flat[col] = df_flat[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x)
                )
    
    print(f"  ✓ Final column count: {len(df_flat.columns)}")
    return df_flat

print("=" * 80)
print("STEP 10: Data Schema & Statistics Generation with TFDV")
print("=" * 80)

# ============================================================================
# PART 1: PROCESS MOMENTS DATA
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: Processing moments_processed.json")
print("=" * 80)

print("\n[1/6] Loading moments_processed.json...")
with open(f'{INPUT_DIR}/moments_processed.json', 'r') as f:
    moments_data = json.load(f)

moments_df = pd.DataFrame(moments_data)
print(f"   ✓ Loaded {len(moments_df)} moment interpretations")

print("\n[2/6] Flattening nested data structures...")
moments_flat = flatten_dataframe(moments_df, "moments")

print("\n[3/6] Generating statistics with TFDV...")
moments_stats = tfdv.generate_statistics_from_dataframe(moments_flat)
print("   ✓ Statistics generated successfully")

print("\n[4/6] Inferring schema from data...")
moments_schema = tfdv.infer_schema(statistics=moments_stats)
print("   ✓ Schema inferred successfully")

print("\n[5/6] Displaying schema...")
tfdv.display_schema(moments_schema)

print("\n[6/6] Saving outputs...")
moments_schema_path = f'{SCHEMA_DIR}/moments_schema.pbtxt'
tfdv.write_schema_text(moments_schema, moments_schema_path)
print(f"   ✓ Schema saved to: {moments_schema_path}")

# Validate data
moments_anomalies = tfdv.validate_statistics(statistics=moments_stats, schema=moments_schema)
if moments_anomalies.anomaly_info:
    print("   ⚠ Anomalies detected:")
    for feature_name, anomaly_info in moments_anomalies.anomaly_info.items():
        print(f"      - {feature_name}: {anomaly_info.description}")
    moments_anomalies_path = f'{REPORTS_DIR}/moments_anomalies.txt'
    with open(moments_anomalies_path, 'w') as f:
        f.write(str(moments_anomalies))
    print(f"   ✓ Anomalies report saved to: {moments_anomalies_path}")
else:
    print("   ✓ No anomalies detected - data is valid!")

# Generate statistics summary
print("\n" + "-" * 80)
print("Moments Statistics Summary")
print("-" * 80)

numeric_cols = moments_flat.select_dtypes(include=['int64', 'float64']).columns
moments_stats_dict = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'moments',
    'total_records': len(moments_flat),
    'total_fields': len(moments_flat.columns),
    'field_names': list(moments_flat.columns)
}

if len(numeric_cols) > 0:
    print("\nNumeric Field Statistics:")
    summary_stats = moments_flat[numeric_cols].describe()
    print(summary_stats)
    moments_stats_dict['numeric_fields'] = list(numeric_cols)
    moments_stats_dict['numeric_statistics'] = summary_stats.to_dict()

categorical_cols = moments_flat.select_dtypes(include=['object', 'bool']).columns
if len(categorical_cols) > 0:
    moments_stats_dict['categorical_fields'] = list(categorical_cols)
    moments_stats_dict['categorical_summary'] = {}
    print("\nCategorical Field Summary (first 5):")
    for col in list(categorical_cols)[:5]:
        unique_count = moments_flat[col].nunique()
        moments_stats_dict['categorical_summary'][col] = {
            'unique_values': int(unique_count),
            'most_common': str(moments_flat[col].mode()[0]) if len(moments_flat[col].mode()) > 0 else None
        }
        print(f"\n  {col}: {unique_count} unique values")

moments_stats_dict['schema_location'] = moments_schema_path

moments_stats_json_path = f'{OUTPUT_DIR}/moments_statistics_summary.json'
with open(moments_stats_json_path, 'w') as f:
    json.dump(moments_stats_dict, f, indent=2)
print(f"\n✓ Moments statistics JSON saved to: {moments_stats_json_path}")

# ============================================================================
# PART 2: PROCESS USERS DATA
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: Processing users_processed.json")
print("=" * 80)

print("\n[1/6] Loading users_processed.json...")
with open(f'{INPUT_DIR}/users_processed.json', 'r') as f:
    users_data = json.load(f)

users_df = pd.DataFrame(users_data)
print(f"   ✓ Loaded {len(users_df)} user records")

print("\n[2/6] Flattening nested data structures...")
users_flat = flatten_dataframe(users_df, "users")

print("\n[3/6] Generating statistics with TFDV...")
users_stats = tfdv.generate_statistics_from_dataframe(users_flat)
print("   ✓ Statistics generated successfully")

print("\n[4/6] Inferring schema from data...")
users_schema = tfdv.infer_schema(statistics=users_stats)
print("   ✓ Schema inferred successfully")

print("\n[5/6] Displaying schema...")
tfdv.display_schema(users_schema)

print("\n[6/6] Saving outputs...")
users_schema_path = f'{SCHEMA_DIR}/users_schema.pbtxt'
tfdv.write_schema_text(users_schema, users_schema_path)
print(f"   ✓ Schema saved to: {users_schema_path}")

users_anomalies = tfdv.validate_statistics(statistics=users_stats, schema=users_schema)
if users_anomalies.anomaly_info:
    print("   ⚠ Anomalies detected:")
    for feature_name, anomaly_info in users_anomalies.anomaly_info.items():
        print(f"      - {feature_name}: {anomaly_info.description}")
    users_anomalies_path = f'{REPORTS_DIR}/users_anomalies.txt'
    with open(users_anomalies_path, 'w') as f:
        f.write(str(users_anomalies))
    print(f"   ✓ Anomalies report saved to: {users_anomalies_path}")
else:
    print("   ✓ No anomalies detected - data is valid!")

print("\n" + "-" * 80)
print("Users Statistics Summary")
print("-" * 80)

numeric_user_cols = users_flat.select_dtypes(include=['int64', 'float64']).columns
users_stats_dict = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'users',
    'total_records': len(users_flat),
    'total_fields': len(users_flat.columns),
    'field_names': list(users_flat.columns)
}

if len(numeric_user_cols) > 0:
    print("\nNumeric Field Statistics:")
    user_summary_stats = users_flat[numeric_user_cols].describe()
    print(user_summary_stats)
    users_stats_dict['numeric_fields'] = list(numeric_user_cols)
    users_stats_dict['numeric_statistics'] = user_summary_stats.to_dict()

categorical_user_cols = users_flat.select_dtypes(include=['object', 'bool']).columns
if len(categorical_user_cols) > 0:
    users_stats_dict['categorical_fields'] = list(categorical_user_cols)
    users_stats_dict['categorical_summary'] = {}
    print("\nCategorical Field Summary (first 5):")
    for col in list(categorical_user_cols)[:5]:
        unique_count = users_flat[col].nunique()
        users_stats_dict['categorical_summary'][col] = {
            'unique_values': int(unique_count),
            'most_common': str(users_flat[col].mode()[0]) if len(users_flat[col].mode()) > 0 else None
        }
        print(f"\n  {col}: {unique_count} unique values")

users_stats_dict['schema_location'] = users_schema_path

users_stats_json_path = f'{OUTPUT_DIR}/users_statistics_summary.json'
with open(users_stats_json_path, 'w') as f:
    json.dump(users_stats_dict, f, indent=2)
print(f"\n✓ Users statistics JSON saved to: {users_stats_json_path}")

# ============================================================================
# PART 3: PROCESS BOOKS DATA
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: Processing books_processed.json")
print("=" * 80)

print("\n[1/6] Loading books_processed.json...")
with open(f'{INPUT_DIR}/books_processed.json', 'r') as f:
    books_data = json.load(f)

books_df = pd.DataFrame(books_data)
print(f"   ✓ Loaded {len(books_df)} book passages")

print("\n[2/6] Flattening nested data structures...")
books_flat = flatten_dataframe(books_df, "books")

print("\n[3/6] Generating statistics with TFDV...")
books_stats = tfdv.generate_statistics_from_dataframe(books_flat)
print("   ✓ Statistics generated successfully")

print("\n[4/6] Inferring schema from data...")
books_schema = tfdv.infer_schema(statistics=books_stats)
print("   ✓ Schema inferred successfully")

print("\n[5/6] Displaying schema...")
tfdv.display_schema(books_schema)

print("\n[6/6] Saving outputs...")
books_schema_path = f'{SCHEMA_DIR}/books_schema.pbtxt'
tfdv.write_schema_text(books_schema, books_schema_path)
print(f"   ✓ Schema saved to: {books_schema_path}")

books_anomalies = tfdv.validate_statistics(statistics=books_stats, schema=books_schema)
if books_anomalies.anomaly_info:
    print("   ⚠ Anomalies detected:")
    for feature_name, anomaly_info in books_anomalies.anomaly_info.items():
        print(f"      - {feature_name}: {anomaly_info.description}")
    books_anomalies_path = f'{REPORTS_DIR}/books_anomalies.txt'
    with open(books_anomalies_path, 'w') as f:
        f.write(str(books_anomalies))
    print(f"   ✓ Anomalies report saved to: {books_anomalies_path}")
else:
    print("   ✓ No anomalies detected - data is valid!")

print("\n" + "-" * 80)
print("Books Statistics Summary")
print("-" * 80)

numeric_books_cols = books_flat.select_dtypes(include=['int64', 'float64']).columns
books_stats_dict = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'books',
    'total_records': len(books_flat),
    'total_fields': len(books_flat.columns),
    'field_names': list(books_flat.columns)
}

if len(numeric_books_cols) > 0:
    print("\nNumeric Field Statistics:")
    books_summary_stats = books_flat[numeric_books_cols].describe()
    print(books_summary_stats)
    books_stats_dict['numeric_fields'] = list(numeric_books_cols)
    books_stats_dict['numeric_statistics'] = books_summary_stats.to_dict()

categorical_books_cols = books_flat.select_dtypes(include=['object', 'bool']).columns
if len(categorical_books_cols) > 0:
    books_stats_dict['categorical_fields'] = list(categorical_books_cols)
    books_stats_dict['categorical_summary'] = {}
    print("\nCategorical Field Summary:")
    for col in list(categorical_books_cols)[:10]:
        unique_count = books_flat[col].nunique()
        books_stats_dict['categorical_summary'][col] = {
            'unique_values': int(unique_count),
            'most_common': str(books_flat[col].mode()[0]) if len(books_flat[col].mode()) > 0 else None
        }
        print(f"\n  {col}: {unique_count} unique values")

books_stats_dict['schema_location'] = books_schema_path

books_stats_json_path = f'{OUTPUT_DIR}/books_statistics_summary.json'
with open(books_stats_json_path, 'w') as f:
    json.dump(books_stats_dict, f, indent=2)
print(f"\n✓ Books statistics JSON saved to: {books_stats_json_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ STEP 10 COMPLETE!")
print("=" * 80)

print(f"\n📊 MOMENTS DATA:")
print(f"  • Schema: {moments_schema_path}")
print(f"  • Statistics: {moments_stats_json_path}")
if moments_anomalies.anomaly_info:
    print(f"  • Anomalies: {moments_anomalies_path}")

print(f"\n👥 USERS DATA:")
print(f"  • Schema: {users_schema_path}")
print(f"  • Statistics: {users_stats_json_path}")
if users_anomalies.anomaly_info:
    print(f"  • Anomalies: {users_anomalies_path}")

print(f"\n📚 BOOKS DATA:")
print(f"  • Schema: {books_schema_path}")
print(f"  • Statistics: {books_stats_json_path}")
if books_anomalies.anomaly_info:
    print(f"  • Anomalies: {books_anomalies_path}")

print("\n" + "=" * 80)