"""
Unit tests for validation module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from validation import DataValidator


class TestDataValidator:
    """Test suite for DataValidator class"""
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance"""
        return DataValidator()
    
    @pytest.fixture
    def valid_data(self):
        """Create valid sample data"""
        return pd.DataFrame({
            'feature_1': [10, 20, 30, 40, 50],
            'feature_2': ['A', 'B', 'C', 'A', 'B'],
            'feature_3': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [0, 1, 0, 1, 1]
        })
    
    def test_initialization(self, validator):
        """Test that DataValidator initializes correctly"""
        assert validator is not None
        assert validator.config is not None
    
    def test_detect_missing_values(self, validator):
        """Test missing value detection"""
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [np.nan, np.nan, 3, 4, 5]
        })
        
        alerts = validator.detect_missing_values(data)
        # col2 has 40% missing, should trigger alert
        assert len(alerts) > 0
    
    def test_detect_outliers(self, validator):
        """Test outlier detection"""
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        
        validator.detect_outliers(data)
        assert 'outliers' in validator.validation_results
    
    def test_generate_statistics(self, validator, valid_data):
        """Test statistics generation"""
        stats = validator.generate_statistics(valid_data)
        
        assert 'total_rows' in stats
        assert 'total_columns' in stats
        assert stats['total_rows'] == 5
        assert stats['total_columns'] == 4
    
    def test_validate_pipeline(self, validator, valid_data):
        """Test full validation pipeline"""
        results = validator.validate(valid_data)
        
        assert 'overall_valid' in results
        assert 'statistics' in results
        assert 'anomalies' in results
