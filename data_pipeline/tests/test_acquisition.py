"""
Unit tests for data acquisition module
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add scripts to path
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from data_acquisition import DataAcquisition


class TestDataAcquisition:
    """Test suite for DataAcquisition class"""
    
    @pytest.fixture
    def acquisition(self):
        """Create DataAcquisition instance"""
        return DataAcquisition()
    
    def test_initialization(self, acquisition):
        """Test that DataAcquisition initializes correctly"""
        assert acquisition is not None
        assert acquisition.config is not None
        assert acquisition.raw_data_path.exists()
    
    def test_fetch_data_returns_dataframe(self, acquisition):
        """Test that fetch_data returns a pandas DataFrame"""
        data = acquisition.fetch_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
    
    def test_fetch_data_has_required_columns(self, acquisition):
        """Test that fetched data has required columns"""
        data = acquisition.fetch_data()
        # TODO: Add your required columns
        # required_columns = ['feature_1', 'feature_2', 'target']
        # assert all(col in data.columns for col in required_columns)
        assert len(data.columns) > 0
    
    def test_save_data(self, acquisition, tmp_path):
        """Test that data is saved correctly"""
        # Create test data
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Save data
        output_path = acquisition.save_data(test_data)
        
        # Verify file exists
        assert output_path.exists()
        
        # Verify data can be loaded
        loaded_data = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(test_data, loaded_data)
    
    def test_fetch_data_no_duplicates(self, acquisition):
        """Test that fetched data has no duplicate rows"""
        data = acquisition.fetch_data()
        assert len(data) == len(data.drop_duplicates())
