"""
Pytest configuration and shared fixtures
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data"""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.randint(0, 100, 100),
        'feature_2': np.random.choice(['A', 'B', 'C'], 100),
        'feature_3': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })
