import logging
import pandas as pd
from datetime import datetime
import yaml
import io
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAcquisition:
    """Simple data acquisition from GCP - reads YAML config and loads data to memory."""
    
    def __init__(self, config_path="data_pipeline/config/config.yaml"):
        """Initialize with config file path."""
        
        try:
            with open(config_path, 'r') as f:
                self.config=  yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.storage_client = storage.Client()
        self.dataframe = None
        
    def list_blobs(self, bucket_name, prefix=None):
        """List all blobs in a bucket with optional prefix."""
        logger.info(f"Listing blobs in bucket '{bucket_name}' with prefix '{prefix}'")
        
        bucket = self.storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        # Get blob names, exclude folders (ending with '/')
        blob_names = [blob.name for blob in blobs if not blob.name.endswith('/')]
        
        logger.info(f"Found {len(blob_names)} files")
        return blob_names
    
    def read_single_blob(self, bucket_name, blob_path, file_format):
        """Read a single blob from GCS into DataFrame."""
        logger.info(f"Reading: {blob_path}")
        
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob_content = blob.download_as_bytes()
        
        if file_format == 'auto':
            if blob_path.endswith('.csv'):         file_format = 'csv'
            elif blob_path.endswith('.parquet'):   file_format = 'parquet'
            elif blob_path.endswith('.json'):      file_format = 'json'
            else:
                raise ValueError(f"Cannot auto-detect format for: {blob_path}")

        # Load into DataFrame based on format
        if file_format == 'csv':
            df = pd.read_csv(io.BytesIO(blob_content))
        elif file_format == 'parquet':
            df = pd.read_parquet(io.BytesIO(blob_content))
        elif file_format == 'json':
            try:
                df = pd.read_json(io.BytesIO(blob_content))
            except ValueError:
                df = pd.read_json(io.BytesIO(blob_content), lines=True)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
                
        logger.info(f"Loaded {len(df)} rows from {blob_path}")
        return df
    def read_all_blobs(self, bucket_name, prefix, file_format):
        """Read all blobs and combine into single DataFrame."""
        logger.info(f"Reading all blobs from bucket '{bucket_name}'")
        
        # Get all blob paths
        blob_paths = self.list_blobs(bucket_name, prefix)
        
        if not blob_paths:
            raise ValueError(f"No files found in bucket '{bucket_name}' with prefix '{prefix}'")
        
        # Read each blob and collect DataFrames
        dataframes_dict = {}
        for blob_path in blob_paths:
            try:
                df = self.read_single_blob(bucket_name, blob_path, file_format)
                # Use filename as key (without path)
                filename = blob_path.split('/')[-1]
                dataframes_dict[filename] = df
                logger.info(f"Stored DataFrame for '{filename}': {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to read {blob_path}: {e}")
                continue
        
        if not dataframes_dict:
            raise ValueError("No data could be loaded from any file")
        
        total_rows = sum(len(df) for df in dataframes_dict.values())
        logger.info(f"Total: {total_rows} rows from {len(dataframes_dict)} files")
        
        return dataframes_dict

    def run(self):
        """Load all data from GCS bucket into memory."""
        logger.info("Starting data acquisition")
        
        # Get config values
        bucket_name = self.config['acquisition']['source_bucket']
        prefix = self.config['acquisition'].get('prefix', '')
        file_format = self.config['acquisition']['file_format']
        
        # Read all blobs from bucket
        self.dataframes = self.read_all_blobs(bucket_name, prefix, file_format)
        
        total_rows = sum(len(df) for df in self.dataframes.values())
        
        logger.info(f"Acquisition complete: {len(self.dataframes)} files, {total_rows} total rows")
        
        return {
            'timestamp': self.timestamp,
            'num_files': len(self.dataframes),
            'total_rows': total_rows,
            'files': list(self.dataframes.keys()),
            'source_bucket': bucket_name,
            'prefix': prefix
        }
        
    def get_dataframes(self):
        """Return the loaded DataFrame."""
        if self.dataframes is None:
            raise ValueError("No data loaded. Run acquisition first.")
        return self.dataframes
'''
def read_raw_files_from_gcs(bucket_name):
    """Reads the contents of all blobs from a GCS bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="raw/")
    blob_names = [blob.name for blob in blobs if not blob.name.endswith('/')]
    for blob_name in blob_names:
        blob = bucket.blob(blob_name)
        blob_content = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(blob_content))
        print(f"Contents of {blob_name}")
        print(df)
    return


read_raw_files_from_gcs("moment_data")
'''
