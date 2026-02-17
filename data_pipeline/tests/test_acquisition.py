import sys
sys.path.append("/Users/jyothssena/Moment/data_pipeline/scripts/")
import pytest
import pandas as pd
import yaml
import io
from unittest.mock import Mock, patch, MagicMock, mock_open
from google.cloud import storage
from data_acquisition import DataAcquisition 

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Sample configuration for testing."""
    return {
        'acquisition': {
            'source_bucket': 'test-bucket',
            'prefix': 'test/',
            'file_format': 'csv'
        }
    }


@pytest.fixture
def sample_csv_data():
    """Sample CSV data as bytes."""
    csv_content = "id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300"
    return csv_content.encode('utf-8')


@pytest.fixture
def sample_json_data():
    """Sample JSON data as bytes."""
    json_content = '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'
    return json_content.encode('utf-8')


@pytest.fixture
def sample_json_lines_data():
    """Sample JSON lines data as bytes."""
    json_content = '{"id": 1, "name": "Alice"}\n{"id": 2, "name": "Bob"}'
    return json_content.encode('utf-8')


@pytest.fixture
def sample_parquet_data():
    """Sample Parquet data as bytes."""
    df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def mock_storage_client():
    """Mock Google Cloud Storage client."""
    with patch('data_acquisition.storage.Client') as mock_client:
        yield mock_client


# ============================================================================
# Test DataAcquisition Initialization
# ============================================================================

class TestDataAcquisitionInit:
    """Test initialization of DataAcquisition class."""
    
    def test_init_with_valid_config(self, mock_storage_client, tmp_path):
        """Test initialization with valid config file."""
        # Create config file
        config_file = tmp_path / "config.yaml"
        config_content = {
            'acquisition': {
                'source_bucket': 'test-bucket',
                'prefix': 'test/',
                'file_format': 'csv'
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        acq = DataAcquisition(config_path=str(config_file))
        
        assert acq.config is not None
        assert acq.config['acquisition']['source_bucket'] == 'test-bucket'
        # Check for dataframes attribute (should exist after run())
        assert acq.timestamp is not None
    
    def test_init_with_missing_config(self, mock_storage_client):
        """Test initialization with missing config file."""
        with pytest.raises(Exception):
            DataAcquisition(config_path="nonexistent_config.yaml")
    
    def test_timestamp_format(self, mock_storage_client, tmp_path):
        """Test timestamp is in correct format."""
        # Create config file
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        acq = DataAcquisition(config_path=str(config_file))
        
        # Check format: YYYYMMDD_HHMMSS
        assert len(acq.timestamp) == 15
        assert acq.timestamp[8] == '_'


# ============================================================================
# Test Config Loading
# ============================================================================

class TestConfigLoading:
    """Test configuration loading functionality."""
    
    def test_load_valid_config(self, mock_storage_client, tmp_path):
        """Test loading valid YAML config."""
        config_file = tmp_path / "config.yaml"
        config_content = {
            'acquisition': {
                'source_bucket': 'test-bucket',
                'prefix': 'test/',
                'file_format': 'csv'
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        acq = DataAcquisition(config_path=str(config_file))
        
        assert 'acquisition' in acq.config
        assert acq.config['acquisition']['source_bucket'] == 'test-bucket'
    
    def test_load_config_with_invalid_yaml(self, mock_storage_client, tmp_path):
        """Test loading invalid YAML raises error."""
        bad_config = tmp_path / "bad_config.yaml"
        with open(bad_config, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(Exception):
            DataAcquisition(config_path=str(bad_config))


# ============================================================================
# Test Blob Listing
# ============================================================================

class TestListBlobs:
    """Test blob listing functionality."""
    
    def test_list_blobs_success(self, mock_storage_client, tmp_path):
        """Test successful blob listing."""
        # Create config
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        # Setup mock
        mock_blob1 = Mock()
        mock_blob1.name = 'test/file1.csv'
        mock_blob2 = Mock()
        mock_blob2.name = 'test/file2.csv'
        mock_blob3 = Mock()
        mock_blob3.name = 'test/folder/'  # Should be excluded
        
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        # Test
        acq = DataAcquisition(config_path=str(config_file))
        blob_names = acq.list_blobs('test-bucket', 'test/')
        
        assert len(blob_names) == 2
        assert 'test/file1.csv' in blob_names
        assert 'test/file2.csv' in blob_names
        assert 'test/folder/' not in blob_names
    
    def test_list_blobs_empty(self, mock_storage_client, tmp_path):
        """Test listing blobs when bucket is empty."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = []
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        acq = DataAcquisition(config_path=str(config_file))
        blob_names = acq.list_blobs('test-bucket', 'test/')
        
        assert len(blob_names) == 0


# ============================================================================
# Test Reading Single Blob
# ============================================================================

class TestReadSingleBlob:
    """Test reading individual blob files."""
    
    def test_read_csv_blob(self, mock_storage_client, tmp_path, sample_csv_data):
        """Test reading CSV blob."""
        # Create config
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        # Setup mock
        mock_blob = Mock()
        mock_blob.download_as_bytes.return_value = sample_csv_data
        
        mock_bucket = Mock()
        mock_bucket.blob.return_value = mock_blob
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        # Test
        acq = DataAcquisition(config_path=str(config_file))
        df = acq.read_single_blob('test-bucket', 'test/file.csv', 'csv')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name', 'value']
        assert df.iloc[0]['name'] == 'Alice'
    
    def test_read_json_blob_array(self, mock_storage_client, tmp_path, sample_json_data):
        """Test reading JSON blob (array format)."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        mock_blob = Mock()
        mock_blob.download_as_bytes.return_value = sample_json_data
        
        mock_bucket = Mock()
        mock_bucket.blob.return_value = mock_blob
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        acq = DataAcquisition(config_path=str(config_file))
        df = acq.read_single_blob('test-bucket', 'test/file.json', 'json')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'id' in df.columns
        assert 'name' in df.columns
    
    def test_read_json_blob_lines(self, mock_storage_client, tmp_path, sample_json_lines_data):
        """Test reading JSON lines blob."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        mock_blob = Mock()
        mock_blob.download_as_bytes.return_value = sample_json_lines_data
        
        mock_bucket = Mock()
        mock_bucket.blob.return_value = mock_blob
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        acq = DataAcquisition(config_path=str(config_file))
        df = acq.read_single_blob('test-bucket', 'test/file.json', 'json')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_read_parquet_blob(self, mock_storage_client, tmp_path, sample_parquet_data):
        """Test reading Parquet blob."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        mock_blob = Mock()
        mock_blob.download_as_bytes.return_value = sample_parquet_data
        
        mock_bucket = Mock()
        mock_bucket.blob.return_value = mock_blob
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        acq = DataAcquisition(config_path=str(config_file))
        df = acq.read_single_blob('test-bucket', 'test/file.parquet', 'parquet')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_read_unsupported_format(self, mock_storage_client, tmp_path):
        """Test reading unsupported file format raises error."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        mock_blob = Mock()
        mock_blob.download_as_bytes.return_value = b'some data'
        
        mock_bucket = Mock()
        mock_bucket.blob.return_value = mock_blob
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        acq = DataAcquisition(config_path=str(config_file))
        
        with pytest.raises(ValueError, match="Unsupported format"):
            acq.read_single_blob('test-bucket', 'test/file.txt', 'txt')


# ============================================================================
# Test Reading All Blobs
# ============================================================================

class TestReadAllBlobs:
    """Test reading multiple blobs."""
    
    def test_read_all_blobs_success(self, mock_storage_client, tmp_path, sample_csv_data):
        """Test reading all blobs successfully."""
        # Create config
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        # Setup mocks
        mock_blob1 = Mock()
        mock_blob1.name = 'test/file1.csv'
        mock_blob2 = Mock()
        mock_blob2.name = 'test/file2.csv'
        
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        
        # Mock blob downloads
        mock_blob_obj = Mock()
        mock_blob_obj.download_as_bytes.return_value = sample_csv_data
        mock_bucket.blob.return_value = mock_blob_obj
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        # Test
        acq = DataAcquisition(config_path=str(config_file))
        result = acq.read_all_blobs('test-bucket', 'test/', 'csv')
        
        # Check if result is a dict (individual DataFrames) or DataFrame (combined)
        if isinstance(result, dict):
            # Individual DataFrames
            assert len(result) == 2
            assert 'file1.csv' in result
            assert 'file2.csv' in result
            assert isinstance(result['file1.csv'], pd.DataFrame)
        else:
            # Combined DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
    
    def test_read_all_blobs_with_auto_detect(self, mock_storage_client, tmp_path, 
                                              sample_csv_data, sample_json_data):
        """Test reading blobs with auto format detection."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        # Setup mocks
        mock_blob1 = Mock()
        mock_blob1.name = 'test/file1.csv'
        mock_blob2 = Mock()
        mock_blob2.name = 'test/file2.json'
        
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        
        # Mock different content for different files
        mock_blob_csv = Mock()
        mock_blob_csv.download_as_bytes.return_value = sample_csv_data
        
        mock_blob_json = Mock()
        mock_blob_json.download_as_bytes.return_value = sample_json_data
        
        def blob_side_effect(path):
            if path.endswith('.csv'):
                return mock_blob_csv
            else:
                return mock_blob_json
        
        mock_bucket.blob.side_effect = blob_side_effect
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        # Test
        acq = DataAcquisition(config_path=str(config_file))
        result = acq.read_all_blobs('test-bucket', 'test/', 'auto')
        
        # Check if result is dict or DataFrame
        if isinstance(result, dict):
            assert len(result) == 2
            assert 'file1.csv' in result or 'file2.json' in result
        else:
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
    
    def test_read_all_blobs_no_files(self, mock_storage_client, tmp_path):
        """Test reading when no files are found."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = []
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        acq = DataAcquisition(config_path=str(config_file))
        
        with pytest.raises(ValueError, match="No files found"):
            acq.read_all_blobs('test-bucket', 'test/', 'csv')
    
    def test_read_all_blobs_all_fail(self, mock_storage_client, tmp_path):
        """Test when all blob reads fail."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'acquisition': {}}, f)
        
        mock_blob = Mock()
        mock_blob.name = 'test/file1.csv'
        
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = [mock_blob]
        
        # Make download fail
        mock_blob_obj = Mock()
        mock_blob_obj.download_as_bytes.side_effect = Exception("Download failed")
        mock_bucket.blob.return_value = mock_blob_obj
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        acq = DataAcquisition(config_path=str(config_file))
        
        with pytest.raises(ValueError, match="No data could be loaded"):
            acq.read_all_blobs('test-bucket', 'test/', 'csv')


# ============================================================================
# Test Run Method
# ============================================================================

class TestRun:
    """Test the main run method."""
    
    def test_run_success(self, mock_storage_client, tmp_path, sample_csv_data):
        """Test successful run."""
        # Create config
        config_file = tmp_path / "config.yaml"
        config_content = {
            'acquisition': {
                'source_bucket': 'test-bucket',
                'prefix': 'test/',
                'file_format': 'csv'
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        # Setup mocks
        mock_blob = Mock()
        mock_blob.name = 'test/file1.csv'
        
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = [mock_blob]
        
        mock_blob_obj = Mock()
        mock_blob_obj.download_as_bytes.return_value = sample_csv_data
        mock_bucket.blob.return_value = mock_blob_obj
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        # Test
        acq = DataAcquisition(config_path=str(config_file))
        metadata = acq.run()
        
        # Verify metadata based on your actual run() return
        assert 'timestamp' in metadata
        assert 'num_files' in metadata
        assert 'total_rows' in metadata
        assert 'files' in metadata
        assert metadata['source_bucket'] == 'test-bucket'
        assert metadata['num_files'] == 1
        assert 'file1.csv' in metadata['files']
    
    def test_run_stores_dataframe(self, mock_storage_client, tmp_path, sample_csv_data):
        """Test run stores dataframes in instance."""
        config_file = tmp_path / "config.yaml"
        config_content = {
            'acquisition': {
                'source_bucket': 'test-bucket',
                'prefix': 'test/',
                'file_format': 'csv'
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        mock_blob = Mock()
        mock_blob.name = 'test/file1.csv'
        
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = [mock_blob]
        
        mock_blob_obj = Mock()
        mock_blob_obj.download_as_bytes.return_value = sample_csv_data
        mock_bucket.blob.return_value = mock_blob_obj
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        acq = DataAcquisition(config_path=str(config_file))
        acq.run()
        
        # Verify dataframes dict is stored (your code uses self.dataframes)
        assert hasattr(acq, 'dataframes'), "DataAcquisition should have 'dataframes' attribute"
        assert acq.dataframes is not None
        assert isinstance(acq.dataframes, dict)
        assert len(acq.dataframes) > 0
        
        # Check first DataFrame in dict
        first_df = list(acq.dataframes.values())[0]
        assert isinstance(first_df, pd.DataFrame)


# ============================================================================
# Test Get DataFrames Methods - REMOVED (no longer exists in code)
# ============================================================================

# These tests have been removed since get_dataframes() and get_dataframe() 
# methods were removed from the DataAcquisition class


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_full_workflow_csv(self, mock_storage_client, tmp_path, sample_csv_data):
        """Test complete workflow with CSV files."""
        # Create config
        config_file = tmp_path / "config.yaml"
        config_content = {
            'acquisition': {
                'source_bucket': 'test-bucket',
                'prefix': 'test/',
                'file_format': 'csv'
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        # Setup
        mock_blob1 = Mock()
        mock_blob1.name = 'test/data1.csv'
        mock_blob2 = Mock()
        mock_blob2.name = 'test/data2.csv'
        
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        
        mock_blob_obj = Mock()
        mock_blob_obj.download_as_bytes.return_value = sample_csv_data
        mock_bucket.blob.return_value = mock_blob_obj
        
        mock_client = mock_storage_client.return_value
        mock_client.bucket.return_value = mock_bucket
        
        # Execute full workflow
        acq = DataAcquisition(config_path=str(config_file))
        metadata = acq.run()
        
        # Verify metadata
        assert 'timestamp' in metadata
        assert 'num_files' in metadata
        assert metadata['num_files'] == 2
        
        # Verify dataframes are stored
        assert hasattr(acq, 'dataframes')
        assert acq.dataframes is not None
        assert isinstance(acq.dataframes, dict)
        assert len(acq.dataframes) == 2
        
        # Verify DataFrames
        assert 'data1.csv' in acq.dataframes
        assert 'data2.csv' in acq.dataframes
        assert isinstance(acq.dataframes['data1.csv'], pd.DataFrame)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=data_acquisition", "--cov-report=html"])