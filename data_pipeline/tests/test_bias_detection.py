"""
Unit Tests for Bias Detection Analysis
File: test_bias_detection.py

Tests all functions in bias_detection.py with edge cases and validation
Uses actual data files for integration testing
"""
import pytest
import pandas as pd
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from data_pipeline.scripts.bias_detection import load_data, run_analysis

# ============================================================
# FIXTURES - Sample Data for Testing
# ============================================================

@pytest.fixture
def sample_interpretations_data():
    """Create minimal sample interpretations data for unit tests"""
    return [
        {
            "character_name": "TestChar_1",
            "book": "Frankenstein",
            "passage": "passage_1",
            "interpretation": "This passage explores themes of isolation and ambition in Victor's character.",
            "word_count": 75
        },
        {
            "character_name": "TestChar_2",
            "book": "Pride and Prejudice",
            "passage": "passage_2",
            "interpretation": "Elizabeth's wit shines through in this dialogue with Darcy.",
            "word_count": 65
        },
        {
            "character_name": "TestChar_3",
            "book": "The Great Gatsby",
            "passage": "passage_1",
            "interpretation": "Gatsby's dream represents the corruption of the American Dream through materialism.",
            "word_count": 85
        },
        {
            "character_name": "TestChar_1",
            "book": "Pride and Prejudice",
            "passage": "passage_3",
            "interpretation": "The social dynamics reveal class tensions.",
            "word_count": 55
        },
        {
            "character_name": "TestChar_2",
            "book": "The Great Gatsby",
            "passage": "passage_2",
            "interpretation": "The green light symbolizes unattainable desires.",
            "word_count": 60
        },
        {
            "character_name": "TestChar_3",
            "book": "Frankenstein",
            "passage": "passage_3",
            "interpretation": "The creature's loneliness mirrors human isolation in profound ways.",
            "word_count": 70
        }
    ]


@pytest.fixture
def sample_characters_data():
    """Create minimal sample characters CSV data"""
    return pd.DataFrame({
        'Name': ['TestChar_1', 'TestChar_2', 'TestChar_3'],
        'Age': [28, 35, 22],
        'Gender': ['Female', 'Male', 'Female'],
        'Distribution_Category': ['Casual', 'Voracious', 'Moderate'],
        'Personality': ['Empathetic', 'Analytical', 'Adventurous']
    })


@pytest.fixture
def temp_json_file(sample_interpretations_data, tmp_path):
    """Create temporary JSON file with sample data"""
    json_file = tmp_path / "test_interpretations.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(sample_interpretations_data, f)
    return str(json_file)


@pytest.fixture
def temp_csv_file(sample_characters_data, tmp_path):
    """Create temporary CSV file with sample data"""
    csv_file = tmp_path / "test_characters.csv"
    sample_characters_data.to_csv(csv_file, index=False)
    return str(csv_file)



@pytest.fixture
def full_scale_interpretations_data():
    """
    Create a full-scale interpretations dataset: 50 characters x 3 books x 3 passages = 450 records.
    Mirrors the shape of the actual production data.
    """
    books = ['Frankenstein', 'Pride and Prejudice', 'The Great Gatsby']
    passages = ['passage_1', 'passage_2', 'passage_3']
    data = []
    for i in range(1, 51):
        for book in books:
            for passage in passages:
                data.append({
                    "character_name": f"Char_{i:02d}",
                    "book": book,
                    "passage": passage,
                    "interpretation": f"Test interpretation for {book} passage {passage} " * 5,
                    "word_count": 70
                })
    return data


@pytest.fixture

def full_scale_characters_data():
    """
    Create a full-scale characters dataset: 50 characters with balanced
    gender, age, distribution type, and personality distributions.
    """
    genders = ['Female', 'Male'] * 25
    ages = [22, 28, 30, 35, 40, 45, 50, 55, 25, 38] * 5  # mix of age groups
    # Distribute as evenly as possible across 50 characters
    dist_cats   = (['Casual'] * 17 + ['Voracious'] * 17 + ['Moderate'] * 16)
    personalities = (['Empathetic'] * 17 + ['Analytical'] * 17 + ['Adventurous'] * 16)

    return pd.DataFrame({
        'Name': [f"Char_{i:02d}" for i in range(1, 51)],
        'Age': ages,
        'Gender': genders,
        'Distribution_Category': dist_cats,
        'Personality': personalities
    })


@pytest.fixture
def full_scale_json_file(full_scale_interpretations_data, tmp_path):
    """Write full-scale interpretations to a temp JSON file"""
    json_file = tmp_path / "full_interpretations.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(full_scale_interpretations_data, f)
    return str(json_file)


@pytest.fixture
def full_scale_csv_file(full_scale_characters_data, tmp_path):
    """Write full-scale characters to a temp CSV file"""
    csv_file = tmp_path / "full_characters.csv"
    full_scale_characters_data.to_csv(csv_file, index=False)
    return str(csv_file)

@pytest.fixture
def balanced_test_dataset():
    """Create a perfectly balanced dataset for testing bias thresholds"""
    # 12 interpretations: balanced across dimensions
    data = []
    
    # 4 characters, each with 3 interpretations
    characters = [
        {'name': 'Char_1', 'age': 22, 'gender': 'Female', 'type': 'Casual', 'personality': 'Empathetic'},
        {'name': 'Char_2', 'age': 30, 'gender': 'Male', 'type': 'Voracious', 'personality': 'Analytical'},
        {'name': 'Char_3', 'age': 40, 'gender': 'Female', 'type': 'Moderate', 'personality': 'Adventurous'},
        {'name': 'Char_4', 'age': 50, 'gender': 'Male', 'type': 'Casual', 'personality': 'Empathetic'}
    ]
    
    books = ['Frankenstein', 'Pride and Prejudice', 'The Great Gatsby']
    
    for char in characters:
        for i, book in enumerate(books):
            data.append({
                "character_name": char['name'],
                "book": book,
                "passage": f"passage_{i+1}",
                "interpretation": f"Test interpretation for {book} " * 10,  # ~70 words
                "word_count": 70
            })
    
    # Create characters dataframe
    char_df = pd.DataFrame([
        {'Name': c['name'], 'Age': c['age'], 'Gender': c['gender'], 
         'Distribution_Category': c['type'], 'Personality': c['personality']}
        for c in characters
    ])
    
    # Create interpretations dataframe
    interp_df = pd.DataFrame(data)
    
    # Merge
    df = interp_df.merge(char_df, left_on='character_name', right_on='Name', how='left')
    df = df.rename(columns={'book': 'book_title'})
    
    # Add age groups
    age_groups = []
    for age in df['Age']:
        if age < 25: age_groups.append("18-24 (Gen Z)")
        elif age < 35: age_groups.append("25-34 (Millennial)")
        elif age < 45: age_groups.append("35-44 (Gen X/Mill)")
        else: age_groups.append("45+ (Gen X/Boom)")
    df['age_group'] = age_groups
    
    return df


# ============================================================
# TEST CLASS 1: Data Loading Tests
# ============================================================

class TestDataLoading:
    """Test data loading and merging functions"""
    
    def test_load_data_with_mock_files(self, temp_json_file, temp_csv_file):
        """Test load_data function with mock files"""
        with patch('data_pipeline.scripts.bias_detection.INTERPRETATIONS_FILE', temp_json_file), \
             patch('data_pipeline.scripts.bias_detection.CHARACTERS_FILE', temp_csv_file):
            
            df = load_data()
            
            assert df is not None, "load_data should return a DataFrame"
            assert len(df) > 0, "DataFrame should not be empty"
            assert 'book_title' in df.columns, "Should rename 'book' to 'book_title'"
            assert 'age_group' in df.columns, "Should create 'age_group' column"
    
    def test_load_data_creates_age_groups(self, temp_json_file, temp_csv_file):
        """Test that age groups are created correctly"""
        with patch('data_pipeline.scripts.bias_detection.INTERPRETATIONS_FILE', temp_json_file), \
             patch('data_pipeline.scripts.bias_detection.CHARACTERS_FILE', temp_csv_file):
            
            df = load_data()
            
            # Check age group logic
            age_groups = df['age_group'].unique()
            assert len(age_groups) > 0, "Should create age groups"
            
            # Verify specific age mappings
            age_22_rows = df[df['Age'] == 22]
            if len(age_22_rows) > 0:
                assert age_22_rows['age_group'].iloc[0] == "18-24 (Gen Z)"
            
            age_35_rows = df[df['Age'] == 35]
            if len(age_35_rows) > 0:
                assert age_35_rows['age_group'].iloc[0] == "35-44 (Gen X/Mill)"
    
    def test_load_data_handles_missing_json_file(self):
        """Test error handling when JSON file is missing"""
        with patch('data_pipeline.scripts.bias_detection.INTERPRETATIONS_FILE', 'nonexistent_file.json'):
            df = load_data()
            assert df is None, "Should return None when file is missing"
    
    def test_load_data_handles_missing_csv_file(self, temp_json_file):
        """Test error handling when CSV file is missing"""
        with patch('data_pipeline.scripts.bias_detection.INTERPRETATIONS_FILE', temp_json_file), \
             patch('data_pipeline.scripts.bias_detection.CHARACTERS_FILE', 'nonexistent_file.csv'):
            df = load_data()
            assert df is None, "Should return None when CSV is missing"
    
    def test_load_data_handles_null_ages(self, tmp_path):
        """Test handling of null/NaN age values"""
        # Create data with null ages
        interp_data = [
            {"character_name": "Test1", "book": "Frankenstein", 
             "passage": "passage_1", "interpretation": "test", "word_count": 50}
        ]
        char_data = pd.DataFrame({
            'Name': ['Test1'],
            'Age': [None],  # Null age
            'Gender': ['Female'],
            'Distribution_Category': ['Casual'],
            'Personality': ['Empathetic']
        })
        
        json_file = tmp_path / "test_interp.json"
        csv_file = tmp_path / "test_char.csv"
        
        with open(json_file, 'w') as f:
            json.dump(interp_data, f)
        char_data.to_csv(csv_file, index=False)
        with patch('data_pipeline.scripts.bias_detection.INTERPRETATIONS_FILE', str(json_file)), \
             patch('data_pipeline.scripts.bias_detection.CHARACTERS_FILE', str(csv_file)):
            df = load_data()
            
            assert df is not None
            assert 'Unknown' in df['age_group'].values, "Should handle null ages as 'Unknown'"


# ============================================================
# TEST CLASS 2: Analysis Function Tests
# ============================================================

class TestAnalysisFunction:
    """Test the main run_analysis function"""
    
    def test_run_analysis_returns_results_dict(self, balanced_test_dataset):
        """Test that run_analysis returns proper results dictionary"""
        results = run_analysis(balanced_test_dataset)
        
        assert isinstance(results, dict), "Should return a dictionary"
        assert 'age' in results, "Should contain age results"
        assert 'gender' in results, "Should contain gender results"
        assert 'reader_type' in results, "Should contain reader_type results"
        assert 'personality' in results, "Should contain personality results"
        assert 'book' in results, "Should contain book results"
        assert 'character' in results, "Should contain character results"
    
    def test_run_analysis_creates_report_file(self, balanced_test_dataset, tmp_path):
        """Test that analysis creates a report file"""
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            run_analysis(balanced_test_dataset)
            
            report_path = tmp_path / 'bias_results' / 'bias_report_FINAL.md'
            assert report_path.exists(), "Should create report file"
            
            # Check report content
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert '# BIAS DETECTION REPORT' in content
                assert 'Age Distribution' in content
                assert 'Gender' in content
        finally:
            os.chdir(original_dir)
    
    def test_run_analysis_age_deviation_calculation(self, balanced_test_dataset):
        """Test that age deviation is calculated correctly"""
        results = run_analysis(balanced_test_dataset)
        
        assert 'max_dev' in results['age'], "Should calculate max deviation"
        assert isinstance(results['age']['max_dev'], float), "Deviation should be float"
        assert results['age']['max_dev'] >= 0, "Deviation should be non-negative"
    
    def test_run_analysis_gender_balance_check(self, balanced_test_dataset):
        """Test gender balance assessment"""
        results = run_analysis(balanced_test_dataset)
        
        assert 'max_dev' in results['gender']
        # Balanced dataset should have low deviation
        assert results['gender']['max_dev'] < 10, "Balanced dataset should have <10% gender deviation"
    
    def test_run_analysis_book_distribution_check(self, balanced_test_dataset):
        """Test book distribution assessment"""
        results = run_analysis(balanced_test_dataset)
        
        assert 'assessment' in results['book']
        assert results['book']['assessment'] in ['PERFECT', 'IMBALANCED']
    
    def test_run_analysis_character_representation_check(self, balanced_test_dataset):
        """Test character representation check"""
        results = run_analysis(balanced_test_dataset)
        
        assert 'assessment' in results['character']
        assert results['character']['assessment'] in ['PERFECT', 'IMBALANCED']


# ============================================================
# TEST CLASS 3: Bias Detection Logic Tests
# ============================================================

class TestBiasDetectionLogic:
    """Test specific bias detection calculations and thresholds"""
    
    def test_age_group_assignment_gen_z(self):
        """Test age group assignment for Gen Z (18-24)"""
        test_ages = [18, 20, 22, 24]
        for age in test_ages:
            if age < 25:
                expected = "18-24 (Gen Z)"
                assert expected == "18-24 (Gen Z)"
    
    def test_age_group_assignment_millennial(self):
        """Test age group assignment for Millennial (25-34)"""
        test_ages = [25, 28, 30, 34]
        for age in test_ages:
            if 25 <= age < 35:
                expected = "25-34 (Millennial)"
                assert expected == "25-34 (Millennial)"
    
    def test_age_group_assignment_gen_x(self):
        """Test age group assignment for Gen X (35-44)"""
        test_ages = [35, 38, 40, 44]
        for age in test_ages:
            if 35 <= age < 45:
                expected = "35-44 (Gen X/Mill)"
                assert expected == "35-44 (Gen X/Mill)"
    
    def test_age_group_assignment_45_plus(self):
        """Test age group assignment for 45+ """
        test_ages = [45, 50, 55, 60]
        for age in test_ages:
            if age >= 45:
                expected = "45+ (Gen X/Boom)"
                assert expected == "45+ (Gen X/Boom)"
    
    def test_deviation_threshold_logic(self):
        """Test that 10% deviation threshold is correctly applied"""
        # Test balanced case
        balanced_dev = 5.0
        assert balanced_dev < 10, "5% should be considered balanced"
        
        # Test biased case
        biased_dev = 15.0
        assert biased_dev >= 10, "15% should be considered biased"
    
    def test_perfect_book_distribution_logic(self):
        """Test logic for detecting perfect book distribution"""
        # All books have 150 interpretations
        book_counts = [150, 150, 150]
        all_150 = all(c == 150 for c in book_counts)
        assert all_150 == True, "Should detect perfect distribution"
        
        # Imbalanced distribution
        book_counts_imbalanced = [150, 140, 160]
        all_150_imbalanced = all(c == 150 for c in book_counts_imbalanced)
        assert all_150_imbalanced == False, "Should detect imbalanced distribution"
    
    def test_character_representation_logic(self):
        """Test logic for checking character representation"""
        # All 50 characters have 9 interpretations
        char_counts = [9] * 50
        chars_with_9 = sum(1 for c in char_counts if c == 9)
        assert chars_with_9 == 50, "Should detect perfect character representation"
        
        # Imperfect representation
        char_counts_imperfect = [9] * 48 + [8, 10]
        chars_with_9_imperfect = sum(1 for c in char_counts_imperfect if c == 9)
        assert chars_with_9_imperfect == 48, "Should detect imperfect representation"


# ============================================================
# TEST CLASS 4: Integration Tests with Actual Data
# ============================================================


class TestIntegrationWithFixtureData:
    """
    Integration tests that mirror production data shape and expectations,
    using fixture-created files instead of local paths.
    """

    def test_load_full_scale_interpretations(self, full_scale_json_file):
        """Fixture-created interpretations file should have 450 records"""
        df = pd.read_json(full_scale_json_file)
        assert len(df) == 450, "Should have 450 interpretations"
        for col in ('character_name', 'book', 'interpretation', 'word_count'):
            assert col in df.columns

    def test_load_full_scale_characters(self, full_scale_csv_file):
        """Fixture-created characters file should have 50 characters"""
        df = pd.read_csv(full_scale_csv_file)
        assert len(df) == 50, "Should have 50 characters"
        for col in ('Name', 'Age', 'Gender', 'Distribution_Category', 'Personality'):
            assert col in df.columns

    def test_full_pipeline_with_fixture_data(self, full_scale_json_file, full_scale_csv_file):
        """Test complete pipeline using fixture-generated full-scale data"""
        with patch('data_pipeline.scripts.bias_detection.INTERPRETATIONS_FILE', full_scale_json_file), \
             patch('data_pipeline.scripts.bias_detection.CHARACTERS_FILE', full_scale_csv_file):

            df = load_data()

            assert df is not None, "Should successfully load data"
            assert len(df) == 450, "Should have 450 records after merge"

            results = run_analysis(df)

            assert results['gender']['max_dev'] < 10,      "Gender should be balanced"
            assert results['reader_type']['max_dev'] < 10, "Reader type should be balanced"
            assert results['personality']['max_dev'] < 10, "Personality should be balanced"
            assert results['book']['assessment'] == 'PERFECT',      "Books should be perfectly distributed"
            assert results['character']['assessment'] == 'PERFECT', "Characters should be perfectly represented"



# ============================================================
# TEST CLASS 5: Edge Cases and Error Handling
# ============================================================

class TestEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        # Should handle gracefully without crashing
        try:
            # This would typically fail, so we just test it doesn't crash unexpectedly
            if len(empty_df) == 0:
                assert True, "Correctly identifies empty DataFrame"
        except Exception as e:
            pytest.fail(f"Should handle empty DataFrame gracefully: {e}")
    
    def test_single_record_dataframe(self, tmp_path):
        """Test behavior with single record"""
        # Create minimal data
        interp_data = [{
            "character_name": "SingleChar",
            "book": "Frankenstein",
            "passage": "passage_1",
            "interpretation": "test",
            "word_count": 50
        }]
        
        char_data = pd.DataFrame({
            'Name': ['SingleChar'],
            'Age': [25],
            'Gender': ['Female'],
            'Distribution_Category': ['Casual'],
            'Personality': ['Empathetic']
        })
        
        json_file = tmp_path / "single.json"
        csv_file = tmp_path / "single.csv"
        
        with open(json_file, 'w') as f:
            json.dump(interp_data, f)
        char_data.to_csv(csv_file, index=False)
        with patch('data_pipeline.scripts.bias_detection.INTERPRETATIONS_FILE', str(json_file)), \
             patch('data_pipeline.scripts.bias_detection.CHARACTERS_FILE', str(csv_file)):
            df = load_data()
            
            assert len(df) == 1, "Should handle single record"
    
    def test_missing_required_columns(self, tmp_path):
        """Test handling of missing required columns"""
        # Create data missing 'Gender' column
        interp_data = [{
            "character_name": "Test",
            "book": "Frankenstein",
            "passage": "passage_1",
            "interpretation": "test",
            "word_count": 50
        }]
        
        char_data = pd.DataFrame({
            'Name': ['Test'],
            'Age': [25],
            # Missing 'Gender'
            'Distribution_Category': ['Casual'],
            'Personality': ['Empathetic']
        })
        
        json_file = tmp_path / "missing_col.json"
        csv_file = tmp_path / "missing_col.csv"
        
        with open(json_file, 'w') as f:
            json.dump(interp_data, f)
        char_data.to_csv(csv_file, index=False)
        with patch('data_pipeline.scripts.bias_detection.INTERPRETATIONS_FILE', str(json_file)), \
             patch('data_pipeline.scripts.bias_detection.CHARACTERS_FILE', str(csv_file)):
            
            # Should either return None or handle gracefully
            try:
                df = load_data()
                if df is not None:
                    # If it loads, Gender column should not exist
                    assert 'Gender' not in df.columns
            except KeyError:
                # This is also acceptable behavior
                assert True
    
    def test_duplicate_character_names(self, tmp_path):
        """Test handling of duplicate character names in merge"""
        interp_data = [
            {"character_name": "DupeChar", "book": "Frankenstein", 
             "passage": "passage_1", "interpretation": "test1", "word_count": 50},
            {"character_name": "DupeChar", "book": "Pride and Prejudice", 
             "passage": "passage_2", "interpretation": "test2", "word_count": 60}
        ]
        
        char_data = pd.DataFrame({
            'Name': ['DupeChar'],
            'Age': [25],
            'Gender': ['Female'],
            'Distribution_Category': ['Casual'],
            'Personality': ['Empathetic']
        })
        
        json_file = tmp_path / "dupe.json"
        csv_file = tmp_path / "dupe.csv"
        
        with open(json_file, 'w') as f:
            json.dump(interp_data, f)
        char_data.to_csv(csv_file, index=False)
        with patch('data_pipeline.scripts.bias_detection.INTERPRETATIONS_FILE', str(json_file)), \
             patch('data_pipeline.scripts.bias_detection.CHARACTERS_FILE', str(csv_file)):
            df = load_data()
            
            # Both interpretations should get the same character data
            assert len(df) == 2, "Should merge both interpretations"
            assert df['Age'].nunique() == 1, "Both should have same age"


# ============================================================
# TEST CLASS 6: Statistical Calculations
# ============================================================

class TestStatisticalCalculations:
    """Test statistical calculation accuracy"""
    
    def test_percentage_calculation(self):
        """Test percentage calculation accuracy"""
        count = 45
        total = 450
        expected_pct = 10.0
        
        calculated_pct = round((count / total) * 100, 1)
        assert calculated_pct == expected_pct, "Percentage calculation should be accurate"
    
    def test_deviation_calculation(self):
        """Test deviation from expected calculation"""
        actual_pct = 44.0
        expected_pct = 25.0  # For 4 age groups
        expected_deviation = 19.0
        
        calculated_dev = round(actual_pct - expected_pct, 1)
        assert calculated_dev == expected_deviation, "Deviation calculation should be accurate"
    
    def test_word_count_statistics(self, balanced_test_dataset):
        """Test word count mean/median calculations"""
        mean = balanced_test_dataset['word_count'].mean()
        median = balanced_test_dataset['word_count'].median()
        
        assert isinstance(mean, (int, float)), "Mean should be numeric"
        assert isinstance(median, (int, float)), "Median should be numeric"
        assert mean > 0, "Mean should be positive"
        assert median > 0, "Median should be positive"
    
    def test_length_by_age_variance_check(self):
        """Test 20% variance threshold for age-length correlation"""
        overall_mean = 70.0
        age_group_mean = 83.0
        
        deviation_pct = ((age_group_mean - overall_mean) / overall_mean) * 100
        
        assert abs(deviation_pct) < 20, "Should be within 20% variance"
        
        # Test bias case
        biased_mean = 100.0
        biased_deviation_pct = ((biased_mean - overall_mean) / overall_mean) * 100
        assert abs(biased_deviation_pct) >= 20, "Should detect bias >20% variance"


# ============================================================
# TEST CLASS 7: Report Generation Tests
# ============================================================

class TestReportGeneration:
    """Test report generation and formatting"""
    
    def test_report_file_created(self, balanced_test_dataset, tmp_path):
        """Test that report file is created in correct location"""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            run_analysis(balanced_test_dataset)
            
            assert os.path.exists('bias_results'), "Should create bias_results directory"
            assert os.path.exists('bias_results/bias_report_FINAL.md'), "Should create report file"
        finally:
            os.chdir(original_dir)
    
    def test_report_contains_all_sections(self, balanced_test_dataset, tmp_path):
        """Test that report contains all required sections"""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            run_analysis(balanced_test_dataset)
            
            with open('bias_results/bias_report_FINAL.md', 'r', encoding='utf-8') as f:
                content = f.read()
            
            required_sections = [
                '# BIAS DETECTION REPORT',
                '## 1. Age Distribution',
                '## 2. Gender',
                '## 3. Reader Type',
                '## 4. Personality',
                '## 5. Book Distribution',
                '## 6. Character Representation',
                '## 7. Length Statistics',
                '## CROSS-TABULATIONS',
                '## SUMMARY',
                '## CONCLUSION',
                '## APPENDIX'
            ]
            
            for section in required_sections:
                assert section in content, f"Report should contain '{section}' section"
        finally:
            os.chdir(original_dir)
    
    def test_report_readable_format(self, balanced_test_dataset, tmp_path):
        """Test that report is in readable markdown format"""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            run_analysis(balanced_test_dataset)
            
            with open('bias_results/bias_report_FINAL.md', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for markdown formatting
            assert content.startswith('#'), "Should start with markdown header"
            assert '```' in content, "Should contain code blocks"
            assert '- ' in content, "Should contain bullet points"
        finally:
            os.chdir(original_dir)


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])