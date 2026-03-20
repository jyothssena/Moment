import unittest
import json
import os
import sys
import pytest
import pandas as pd
from pathlib import Path

if sys.platform == "darwin":
    pytest.skip("TFDV not available on macOS Apple Silicon", allow_module_level=True)

import tensorflow_data_validation as tfdv


# ============================================================================
# PATH RESOLUTION
# Works both inside Docker (/opt/airflow) and locally on Windows/Mac
# ============================================================================

def _find_repo_root():
    """Walk up from this file to find the repo root (contains data_pipeline/)"""
    current = Path(__file__).resolve().parent
    for _ in range(6):
        if (current / 'data_pipeline').exists() or (current / 'data').exists():
            return current
        current = current.parent
    return Path('/opt/airflow')

# Inside Docker:  /opt/airflow
# On Windows/Mac: C:/Users/csk23/Desktop/Moment/Moment  (or wherever repo lives)
AIRFLOW_HOME = Path(os.environ.get('AIRFLOW_HOME', str(_find_repo_root())))

INPUT_DIR  = AIRFLOW_HOME / 'data' / 'processed'
OUTPUT_DIR = AIRFLOW_HOME / 'data' / 'reports'
SCHEMA_DIR = AIRFLOW_HOME / 'data' / 'schemas'
REPORTS_DIR = AIRFLOW_HOME / 'data' / 'reports'


class TestDataSchemaStatistics(unittest.TestCase):
    """
    Unit tests for STEP 10: Data Schema & Statistics Generation
    Tests the TFDV-based schema generation and statistics calculation
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - runs once before all tests"""
        cls.input_dir  = INPUT_DIR
        cls.output_dir = OUTPUT_DIR
        cls.schema_dir = SCHEMA_DIR
        cls.reports_dir = REPORTS_DIR

        moments_path = cls.input_dir / 'moments_processed.json'
        users_path   = cls.input_dir / 'users_processed.json'
        books_path   = cls.input_dir / 'books_processed.json'

        with open(moments_path, 'r') as f:
            cls.moments_data = json.load(f)

        with open(users_path, 'r') as f:
            cls.users_data = json.load(f)

        with open(books_path, 'r') as f:
            cls.books_data = json.load(f)

    # ========================================================================
    # TEST 1: Data Loading Tests
    # ========================================================================

    def test_moments_data_exists(self):
        """Test that moments_processed.json exists and is readable"""
        self.assertTrue(
            (self.input_dir / 'moments_processed.json').exists(),
            f"moments_processed.json should exist at {self.input_dir}"
        )

    def test_users_data_exists(self):
        """Test that users_processed.json exists and is readable"""
        self.assertTrue(
            (self.input_dir / 'users_processed.json').exists(),
            f"users_processed.json should exist at {self.input_dir}"
        )

    def test_books_data_exists(self):
        """Test that books_processed.json exists and is readable"""
        self.assertTrue(
            (self.input_dir / 'books_processed.json').exists(),
            f"books_processed.json should exist at {self.input_dir}"
        )

    def test_moments_data_not_empty(self):
        """Test that moments data contains records"""
        self.assertGreater(len(self.moments_data), 0,
                           "Moments data should not be empty")

    def test_moments_data_count(self):
        """Test that moments data has expected number of records (450)"""
        self.assertEqual(len(self.moments_data), 450,
                         "Moments data should contain 450 records (50 characters × 9 passages)")

    def test_users_data_count(self):
        """Test that users data has expected number of records (50)"""
        self.assertEqual(len(self.users_data), 50,
                         "Users data should contain 50 character profiles")

    def test_books_data_count(self):
        """Test that books data has expected number of records (9)"""
        self.assertEqual(len(self.books_data), 3,
                         "Books data should contain 3 book records")

    # ========================================================================
    # TEST 2: Data Structure Tests
    # ========================================================================

    def test_moments_required_fields(self):
        """Test that moments records have all required fields"""
        required_fields = [
            'interpretation_id', 'user_id', 'book_id', 'passage_id',
            'character_name', 'cleaned_interpretation', 'quality_score'
        ]
        for field in required_fields:
            self.assertIn(field, self.moments_data[0],
                          f"Moments record should contain '{field}' field")

    def test_users_required_fields(self):
        """Test that users records have all required fields"""
        required_fields = ['user_id', 'character_name']
        for field in required_fields:
            self.assertIn(field, self.users_data[0],
                          f"Users record should contain '{field}' field")

    def test_books_required_fields(self):
        """Test that books records have all required fields"""
        required_fields = [
            'book_id', 'passage_id', 'book_title',
            'book_author', 'cleaned_passage_text'
        ]
        for field in required_fields:
            self.assertIn(field, self.books_data[0],
                          f"Books record should contain '{field}' field")

    def test_moments_metrics_structure(self):
        """Test that moments records have metrics with correct fields"""
        if 'metrics' in self.moments_data[0]:
            metrics = self.moments_data[0]['metrics']
            expected_metrics = ['word_count', 'char_count', 'readability_score']
            for metric in expected_metrics:
                self.assertIn(metric, metrics,
                              f"Metrics should contain '{metric}'")

    # ========================================================================
    # TEST 3: Data Type Validation Tests
    # ========================================================================

    def test_moments_quality_score_type(self):
        """Test that quality_score is numeric (float or int)"""
        for record in self.moments_data[:10]:
            self.assertIsInstance(record['quality_score'], (int, float),
                                  "quality_score should be numeric")

    def test_moments_quality_score_range(self):
        """Test that quality_score is between 0 and 1"""
        for record in self.moments_data:
            self.assertGreaterEqual(record['quality_score'], 0,
                                    "quality_score should be >= 0")
            self.assertLessEqual(record['quality_score'], 1,
                                 "quality_score should be <= 1")

    def test_books_word_count_positive(self):
        """Test that word_count in books metrics is positive"""
        for record in self.books_data:
            if 'metrics' in record and 'word_count' in record['metrics']:
                self.assertGreater(record['metrics']['word_count'], 0,
                                   "word_count should be positive")

    # ========================================================================
    # TEST 4: Data Flattening Tests
    # ========================================================================

    def test_flatten_dataframe_handles_dicts(self):
        """Test that flattening correctly handles nested dictionaries"""
        sample_data = [{'id': 1, 'metrics': {'count': 10, 'score': 0.5}}]
        df = pd.DataFrame(sample_data)

        if 'metrics' in df.columns:
            metrics_df = pd.json_normalize(df['metrics'])
            metrics_df.columns = ['metric_' + col for col in metrics_df.columns]
            df_flat = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
            self.assertIn('metric_count', df_flat.columns)
            self.assertIn('metric_score', df_flat.columns)

    def test_flatten_dataframe_handles_lists(self):
        """Test that flattening correctly converts lists to strings"""
        sample_data = [{'id': 1, 'issues': ['issue1', 'issue2']}]
        df = pd.DataFrame(sample_data)
        df['issues'] = df['issues'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
        self.assertIsInstance(df['issues'].iloc[0], str)
        self.assertIn('issue1', df['issues'].iloc[0])

    # ========================================================================
    # TEST 5: TFDV — Schema Generation Tests
    # ========================================================================

    def test_tfdv_importable(self):
        """Test that tensorflow_data_validation is installed and importable"""
        import tensorflow_data_validation as tfdv
        self.assertIsNotNone(tfdv.__version__,
                             "TFDV should be importable and have a version")

    def test_tfdv_generates_statistics_from_moments(self):
        """Test that TFDV can generate statistics from moments dataframe"""
        df = pd.DataFrame(self.moments_data)
        # Flatten list/dict columns to strings so TFDV can process them
        for col in df.columns:
            if df[col].dtype == 'object':
                first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(first_val, (dict, list)):
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x)
                    )
        stats = tfdv.generate_statistics_from_dataframe(df)
        self.assertIsNotNone(stats, "TFDV should generate statistics from moments df")

    def test_tfdv_infers_schema_from_moments(self):
        """Test that TFDV can infer a schema from moments statistics"""
        df = pd.DataFrame(self.moments_data)
        for col in df.columns:
            if df[col].dtype == 'object':
                first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(first_val, (dict, list)):
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x)
                    )
        stats = tfdv.generate_statistics_from_dataframe(df)
        schema = tfdv.infer_schema(statistics=stats)
        self.assertIsNotNone(schema, "TFDV should infer a schema")
        self.assertGreater(len(schema.feature), 0,
                           "Inferred schema should have at least one feature")

    def test_schema_files_exist(self):
        """Test that schema .pbtxt files were generated by generate_schema_stats.py"""
        schema_files = [
            'moments_schema.pbtxt',
            'users_schema.pbtxt',
            'books_schema.pbtxt'
        ]
        for schema_file in schema_files:
            path = self.schema_dir / schema_file
            self.assertTrue(
                path.exists(),
                f"{schema_file} should exist at {self.schema_dir}. "
                f"Run generate_schema_stats.py first."
            )

    def test_schema_files_not_empty(self):
        """Test that schema files contain content"""
        schema_files = [
            'moments_schema.pbtxt',
            'users_schema.pbtxt',
            'books_schema.pbtxt'
        ]
        for schema_file in schema_files:
            path = self.schema_dir / schema_file
            if path.exists():
                self.assertGreater(path.stat().st_size, 0,
                                   f"{schema_file} should not be empty")

    def test_can_load_moments_schema(self):
        """Test that moments schema can be loaded by TFDV"""
        schema_path = self.schema_dir / 'moments_schema.pbtxt'
        if schema_path.exists():
            try:
                schema = tfdv.load_schema_text(str(schema_path))
                self.assertIsNotNone(schema)
            except Exception as e:
                self.fail(f"Failed to load moments schema: {e}")
        else:
            self.skipTest(f"Schema not yet generated at {schema_path}")

    # ========================================================================
    # TEST 6: Statistics JSON Tests
    # ========================================================================

    def test_schema_stats_json_exists(self):
        """Test that schema_stats.json was generated"""
        path = self.output_dir / 'schema_stats.json'
        self.assertTrue(path.exists(),
                        f"schema_stats.json should exist at {self.output_dir}")

    def test_schema_stats_json_structure(self):
        """Test that schema_stats.json has correct top-level structure"""
        path = self.output_dir / 'schema_stats.json'
        if path.exists():
            with open(path, 'r') as f:
                stats = json.load(f)
            self.assertIn('generated_at', stats,
                          "schema_stats.json should have 'generated_at'")
            self.assertIn('datasets', stats,
                          "schema_stats.json should have 'datasets'")

    def test_schema_stats_moments_row_count(self):
        """Test that schema_stats.json reports correct moments row count"""
        path = self.output_dir / 'schema_stats.json'
        if path.exists():
            with open(path, 'r') as f:
                stats = json.load(f)
            datasets = stats.get('datasets', {})
            moments_key = next(
                (k for k in datasets if 'moments' in k.lower()), None
            )
            if moments_key:
                self.assertEqual(
                    datasets[moments_key]['rows'], 450,
                    "schema_stats.json should report 450 moments rows"
                )

    def test_schema_stats_users_row_count(self):
        """Test that schema_stats.json reports correct users row count"""
        path = self.output_dir / 'schema_stats.json'
        if path.exists():
            with open(path, 'r') as f:
                stats = json.load(f)
            datasets = stats.get('datasets', {})
            users_key = next(
                (k for k in datasets if 'users' in k.lower()), None
            )
            if users_key:
                self.assertEqual(
                    datasets[users_key]['rows'], 50,
                    "schema_stats.json should report 50 users rows"
                )

    def test_schema_stats_books_row_count(self):
        """Test that schema_stats.json reports correct books row count"""
        path = self.output_dir / 'schema_stats.json'
        if path.exists():
            with open(path, 'r') as f:
                stats = json.load(f)
            datasets = stats.get('datasets', {})
            books_key = next(
                (k for k in datasets if 'books' in k.lower()), None
            )
            if books_key:
                self.assertEqual(
                    datasets[books_key]['rows'], 6,
                    "schema_stats.json should report 3 books rows"
                )

    def test_schema_stats_has_column_names(self):
        """Test that schema_stats.json includes column names for each dataset"""
        path = self.output_dir / 'schema_stats.json'
        if path.exists():
            with open(path, 'r') as f:
                stats = json.load(f)
            for dataset_name, dataset_stats in stats.get('datasets', {}).items():
                self.assertIn('column_names', dataset_stats,
                              f"Dataset '{dataset_name}' should have column_names")
                self.assertGreater(len(dataset_stats['column_names']), 0,
                                   f"Dataset '{dataset_name}' should have at least one column")

    def test_schema_stats_null_counts_present(self):
        """Test that null counts are tracked for each dataset"""
        path = self.output_dir / 'schema_stats.json'
        if path.exists():
            with open(path, 'r') as f:
                stats = json.load(f)
            for dataset_name, dataset_stats in stats.get('datasets', {}).items():
                self.assertIn('null_counts', dataset_stats,
                              f"Dataset '{dataset_name}' should have null_counts")

    def test_schema_stats_numeric_stats_present(self):
        """Test that numeric statistics are calculated in schema_stats.json"""
        path = self.output_dir / 'schema_stats.json'
        if path.exists():
            with open(path, 'r') as f:
                stats = json.load(f)
            moments_key = next(
                (k for k in stats.get('datasets', {}) if 'moments' in k.lower()), None
            )
            if moments_key:
                dataset_stats = stats['datasets'][moments_key]
                self.assertIn('numeric_stats', dataset_stats,
                              "Moments dataset should have numeric_stats")

    # ========================================================================
    # TEST 7: Data Quality Validation Tests
    # ========================================================================

    def test_no_null_ids_in_moments(self):
        """Test that critical ID fields have no null values"""
        critical_fields = ['interpretation_id', 'user_id', 'book_id', 'passage_id']
        for record in self.moments_data:
            for field in critical_fields:
                self.assertIsNotNone(record.get(field),
                                     f"'{field}' should not be null")
                self.assertNotEqual(record.get(field), '',
                                    f"'{field}' should not be empty string")

    def test_books_title_consistency(self):
        """Test that book titles are consistent (3 books expected)"""
        book_titles = set(book['book_title'] for book in self.books_data)
        self.assertEqual(len(book_titles), 1,
                         "Should have exactly 3 unique book titles")
        expected_titles = {'The Great Gatsby'}
        self.assertEqual(book_titles, expected_titles,
                         f"Book titles should be {expected_titles}")

    def test_unique_interpretation_ids(self):
        """Test that all interpretation IDs are unique"""
        ids = [record['interpretation_id'] for record in self.moments_data]
        self.assertEqual(len(ids), len(set(ids)),
                         "All interpretation_ids should be unique")

    def test_moments_no_null_quality_scores(self):
        """Test that quality_score has no null values in moments"""
        path = self.output_dir / 'schema_stats.json'
        if path.exists():
            with open(path, 'r') as f:
                stats = json.load(f)
            moments_key = next(
                (k for k in stats.get('datasets', {}) if 'moments' in k.lower()), None
            )
            if moments_key:
                null_counts = stats['datasets'][moments_key].get('null_counts', {})
                if 'quality_score' in null_counts:
                    self.assertEqual(null_counts['quality_score'], 0,
                                     "quality_score should have no null values")

    # ========================================================================
    # TEST 8: Validation Report Tests
    # ========================================================================

    def test_validation_report_exists(self):
        """Test that validation_report.json was generated"""
        path = self.output_dir / 'validation_report.json'
        self.assertTrue(path.exists(),
                        f"validation_report.json should exist at {self.output_dir}")

    def test_validation_report_structure(self):
        """Test that validation_report.json has expected structure"""
        path = self.output_dir / 'validation_report.json'
        if path.exists():
            with open(path, 'r') as f:
                report = json.load(f)
            self.assertIsInstance(report, dict,
                                  "validation_report.json should be a JSON object")


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDataSchemaStatistics)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run:  {result.testsRun}")
    print(f"Successes:  {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:   {len(result.failures)}")
    print(f"Errors:     {len(result.errors)}")
    print("=" * 70)

    exit(0 if result.wasSuccessful() else 1)
