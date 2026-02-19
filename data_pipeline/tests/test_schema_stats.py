import unittest
import json
import os
import pandas as pd
import tensorflow_data_validation as tfdv
from pathlib import Path

class TestDataSchemaStatistics(unittest.TestCase):
    """
    Unit tests for STEP 10: Data Schema & Statistics Generation
    Tests the TFDV-based schema generation and statistics calculation
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - runs once before all tests"""
        cls.input_dir = 'data/input'
        cls.output_dir = 'data/output'
        cls.schema_dir = 'schemas'
        
        # Load test data
        with open(f'{cls.input_dir}/moments_processed.json', 'r') as f:
            cls.moments_data = json.load(f)
        
        with open(f'{cls.input_dir}/users_processed.json', 'r') as f:
            cls.users_data = json.load(f)
        
        with open(f'{cls.input_dir}/books_processed.json', 'r') as f:
            cls.books_data = json.load(f)
    
    # ========================================================================
    # TEST 1: Data Loading Tests
    # ========================================================================
    
    def test_moments_data_exists(self):
        """Test that moments_processed.json exists and is readable"""
        self.assertTrue(
            os.path.exists(f'{self.input_dir}/moments_processed.json'),
            "moments_processed.json file should exist"
        )
    
    def test_users_data_exists(self):
        """Test that users_processed.json exists and is readable"""
        self.assertTrue(
            os.path.exists(f'{self.input_dir}/users_processed.json'),
            "users_processed.json file should exist"
        )
    
    def test_books_data_exists(self):
        """Test that books_processed.json exists and is readable"""
        self.assertTrue(
            os.path.exists(f'{self.input_dir}/books_processed.json'),
            "books_processed.json file should exist"
        )
    
    def test_moments_data_not_empty(self):
        """Test that moments data contains records"""
        self.assertGreater(
            len(self.moments_data), 0,
            "Moments data should not be empty"
        )
    
    def test_moments_data_count(self):
        """Test that moments data has expected number of records (450)"""
        self.assertEqual(
            len(self.moments_data), 450,
            "Moments data should contain 450 records (50 characters × 9 passages)"
        )
    
    def test_users_data_count(self):
        """Test that users data has expected number of records (50)"""
        self.assertEqual(
            len(self.users_data), 50,
            "Users data should contain 50 character profiles"
        )
    
    def test_books_data_count(self):
        """Test that books data has expected number of records (9)"""
        self.assertEqual(
            len(self.books_data), 9,
            "Books data should contain 9 passages (3 per book)"
        )
    
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
            self.assertIn(
                field, self.moments_data[0],
                f"Moments record should contain '{field}' field"
            )
    
    def test_users_required_fields(self):
        """Test that users records have all required fields"""
        required_fields = ['user_id', 'character_name']
        
        for field in required_fields:
            self.assertIn(
                field, self.users_data[0],
                f"Users record should contain '{field}' field"
            )
    
    def test_books_required_fields(self):
        """Test that books records have all required fields"""
        required_fields = [
            'book_id', 'passage_id', 'book_title', 
            'book_author', 'cleaned_passage_text'
        ]
        
        for field in required_fields:
            self.assertIn(
                field, self.books_data[0],
                f"Books record should contain '{field}' field"
            )
    
    def test_moments_metrics_structure(self):
        """Test that moments records have metrics with correct fields"""
        if 'metrics' in self.moments_data[0]:
            metrics = self.moments_data[0]['metrics']
            expected_metrics = [
                'word_count', 'char_count', 'readability_score'
            ]
            
            for metric in expected_metrics:
                self.assertIn(
                    metric, metrics,
                    f"Metrics should contain '{metric}'"
                )
    
    # ========================================================================
    # TEST 3: Data Type Validation Tests
    # ========================================================================
    
    def test_moments_quality_score_type(self):
        """Test that quality_score is numeric (float or int)"""
        for record in self.moments_data[:10]:  # Sample test
            self.assertIsInstance(
                record['quality_score'], (int, float),
                "quality_score should be numeric"
            )
    
    def test_moments_quality_score_range(self):
        """Test that quality_score is between 0 and 1"""
        for record in self.moments_data:
            self.assertGreaterEqual(
                record['quality_score'], 0,
                "quality_score should be >= 0"
            )
            self.assertLessEqual(
                record['quality_score'], 1,
                "quality_score should be <= 1"
            )
    
    def test_books_word_count_positive(self):
        """Test that word_count in books metrics is positive"""
        for record in self.books_data:
            if 'metrics' in record and 'word_count' in record['metrics']:
                self.assertGreater(
                    record['metrics']['word_count'], 0,
                    "word_count should be positive"
                )
    
    # ========================================================================
    # TEST 4: Data Flattening Tests
    # ========================================================================
    
    def test_flatten_dataframe_handles_dicts(self):
        """Test that flattening correctly handles nested dictionaries"""
        # Create sample data with nested dict
        sample_data = [
            {
                'id': 1,
                'metrics': {'count': 10, 'score': 0.5}
            }
        ]
        df = pd.DataFrame(sample_data)
        
        # Flatten
        if 'metrics' in df.columns:
            metrics_df = pd.json_normalize(df['metrics'])
            metrics_df.columns = ['metric_' + col for col in metrics_df.columns]
            df_flat = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
            
            self.assertIn('metric_count', df_flat.columns)
            self.assertIn('metric_score', df_flat.columns)
    
    def test_flatten_dataframe_handles_lists(self):
        """Test that flattening correctly converts lists to strings"""
        # Create sample data with list
        sample_data = [
            {
                'id': 1,
                'issues': ['issue1', 'issue2']
            }
        ]
        df = pd.DataFrame(sample_data)
        
        # Convert lists to strings
        df['issues'] = df['issues'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
        
        self.assertIsInstance(df['issues'].iloc[0], str)
        self.assertIn('issue1', df['issues'].iloc[0])
    
    # ========================================================================
    # TEST 5: Schema Generation Tests
    # ========================================================================
    
    def test_schema_files_exist(self):
        """Test that schema files were generated"""
        schema_files = [
            'moments_schema.pbtxt',
            'users_schema.pbtxt',
            'books_schema.pbtxt'
        ]
        
        for schema_file in schema_files:
            self.assertTrue(
                os.path.exists(f'{self.schema_dir}/{schema_file}'),
                f"{schema_file} should exist in schemas directory"
            )
    
    def test_schema_files_not_empty(self):
        """Test that schema files contain content"""
        schema_files = [
            'moments_schema.pbtxt',
            'users_schema.pbtxt',
            'books_schema.pbtxt'
        ]
        
        for schema_file in schema_files:
            file_path = f'{self.schema_dir}/{schema_file}'
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                self.assertGreater(
                    file_size, 0,
                    f"{schema_file} should not be empty"
                )
    
    def test_can_load_moments_schema(self):
        """Test that moments schema can be loaded by TFDV"""
        schema_path = f'{self.schema_dir}/moments_schema.pbtxt'
        if os.path.exists(schema_path):
            try:
                schema = tfdv.load_schema_text(schema_path)
                self.assertIsNotNone(schema)
            except Exception as e:
                self.fail(f"Failed to load moments schema: {e}")
    
    # ========================================================================
    # TEST 6: Statistics Generation Tests
    # ========================================================================
    
    def test_statistics_json_files_exist(self):
        """Test that statistics JSON files were generated"""
        stats_files = [
            'moments_statistics_summary.json',
            'users_statistics_summary.json',
            'books_statistics_summary.json'
        ]
        
        for stats_file in stats_files:
            self.assertTrue(
                os.path.exists(f'{self.output_dir}/{stats_file}'),
                f"{stats_file} should exist in output directory"
            )
    
    def test_moments_statistics_structure(self):
        """Test that moments statistics JSON has correct structure"""
        stats_path = f'{self.output_dir}/moments_statistics_summary.json'
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            required_keys = [
                'timestamp', 'dataset', 'total_records', 
                'total_fields', 'field_names'
            ]
            
            for key in required_keys:
                self.assertIn(
                    key, stats,
                    f"Statistics should contain '{key}'"
                )
    
    def test_moments_statistics_record_count(self):
        """Test that statistics record count matches actual data"""
        stats_path = f'{self.output_dir}/moments_statistics_summary.json'
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            self.assertEqual(
                stats['total_records'], len(self.moments_data),
                "Statistics record count should match actual data"
            )
    
    def test_users_statistics_record_count(self):
        """Test that users statistics record count matches actual data"""
        stats_path = f'{self.output_dir}/users_statistics_summary.json'
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            self.assertEqual(
                stats['total_records'], len(self.users_data),
                "Users statistics record count should match actual data"
            )
    
    def test_books_statistics_record_count(self):
        """Test that books statistics record count matches actual data"""
        stats_path = f'{self.output_dir}/books_statistics_summary.json'
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            self.assertEqual(
                stats['total_records'], len(self.books_data),
                "Books statistics record count should match actual data"
            )
    
    def test_numeric_statistics_present(self):
        """Test that numeric statistics are calculated"""
        stats_path = f'{self.output_dir}/moments_statistics_summary.json'
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            if 'numeric_statistics' in stats and stats['numeric_statistics']:
                # Check that statistics include standard measures
                first_field = list(stats['numeric_statistics'].keys())[0]
                field_stats = stats['numeric_statistics'][first_field]
                
                expected_stats = ['mean', 'std', 'min', 'max', '50%']
                for stat in expected_stats:
                    self.assertIn(
                        stat, field_stats,
                        f"Numeric statistics should include '{stat}'"
                    )
    
    # ========================================================================
    # TEST 7: Data Quality Validation Tests
    # ========================================================================
    
    def test_no_null_ids_in_moments(self):
        """Test that critical ID fields have no null values"""
        critical_fields = ['interpretation_id', 'user_id', 'book_id', 'passage_id']
        
        for record in self.moments_data:
            for field in critical_fields:
                self.assertIsNotNone(
                    record.get(field),
                    f"'{field}' should not be null"
                )
                self.assertNotEqual(
                    record.get(field), '',
                    f"'{field}' should not be empty string"
                )
    
    def test_books_title_consistency(self):
        """Test that book titles are consistent (3 books expected)"""
        book_titles = set([book['book_title'] for book in self.books_data])
        self.assertEqual(
            len(book_titles), 3,
            "Should have exactly 3 unique book titles"
        )
        
        expected_titles = {'Frankenstein', 'Pride and Prejudice', 'The Great Gatsby'}
        self.assertEqual(
            book_titles, expected_titles,
            f"Book titles should be {expected_titles}"
        )
    
    def test_unique_interpretation_ids(self):
        """Test that all interpretation IDs are unique"""
        ids = [record['interpretation_id'] for record in self.moments_data]
        unique_ids = set(ids)
        
        self.assertEqual(
            len(ids), len(unique_ids),
            "All interpretation_ids should be unique"
        )
    
    # ========================================================================
    # TEST 8: HTML Report Tests
    # ========================================================================
    
    def test_html_report_exists(self):
        """Test that HTML report was generated"""
        self.assertTrue(
            os.path.exists('reports/complete_statistics_report.html'),
            "HTML report should exist"
        )
    
    def test_html_report_not_empty(self):
        """Test that HTML report has content"""
        report_path = 'reports/complete_statistics_report.html'
        if os.path.exists(report_path):
            file_size = os.path.getsize(report_path)
            self.assertGreater(
                file_size, 10000,  # At least 10KB
                "HTML report should have substantial content"
            )
    
    def test_html_report_contains_data(self):
        """Test that HTML report contains expected data sections"""
        report_path = 'reports/complete_statistics_report.html'
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertIn('Moments Data', content)
            self.assertIn('Users Data', content)
            self.assertIn('Books Data', content)
            self.assertIn('450', content)  # Moments count
            self.assertIn('50', content)   # Users count
            self.assertIn('9', content)    # Books count

# ========================================================================
# TEST RUNNER
# ========================================================================

if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDataSchemaStatistics)
    
    # Run the tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)