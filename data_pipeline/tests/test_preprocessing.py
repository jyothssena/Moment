"""
Unit tests for preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance"""
        return DataPreprocessor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'num1': [1, 2, np.nan, 4, 5],
            'num2': [10, 20, 30, 40, 50],
            'cat1': ['A', 'B', 'A', np.nan, 'C'],
            'target': [0, 1, 0, 1, 1]
        })
    
    def test_initialization(self, preprocessor):
        """Test that DataPreprocessor initializes correctly"""
        assert preprocessor is not None
        assert preprocessor.config is not None
    
    def test_handle_missing_values_drop(self, preprocessor, sample_data):
        """Test missing value handling with drop method"""
        result = preprocessor.handle_missing_values(sample_data)
        assert result.isnull().sum().sum() == 0
    
    def test_remove_duplicates(self, preprocessor):
        """Test duplicate removal"""
        data_with_dupes = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        
        result = preprocessor.remove_duplicates(data_with_dupes)
        assert len(result) == 3  # One duplicate removed
    
    def test_handle_outliers(self, preprocessor):
        """Test outlier handling"""
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            'target': [0, 0, 0, 1, 1, 1]
        })
        
        result = preprocessor.handle_outliers(data, columns=['value'])
        assert len(result) < len(data)  # Outlier removed
    
    def test_encode_categorical(self, preprocessor, sample_data):
        """Test categorical encoding"""
        result = preprocessor.encode_categorical(sample_data.dropna(), columns=['cat1'])
        assert result['cat1'].dtype in [np.int64, np.int32, np.float64]
    
    def test_preprocess_pipeline(self, preprocessor, sample_data):
        """Test full preprocessing pipeline"""
        result = preprocessor.preprocess(sample_data)
        
        # Should have no missing values
        assert result.isnull().sum().sum() == 0
        
        # Should have same or fewer rows (due to cleaning)
        assert len(result) <= len(sample_data)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()
    
    def test_empty_dataframe(self, preprocessor):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        # Should not raise an error
        try:
            result = preprocessor.handle_missing_values(empty_df)
            assert len(result) == 0
        except Exception as e:
            pytest.fail(f"Should handle empty dataframe: {e}")
    
    def test_all_missing_values(self, preprocessor):
        """Test handling of column with all missing values"""
        data = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [1, 2, 3]
        })
        
        result = preprocessor.handle_missing_values(data)
        # Should handle gracefully
        assert result is not None
