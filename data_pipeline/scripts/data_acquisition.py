"""
Data Acquisition Module
Fetches data from source and saves to raw data directory
"""

import os
import logging
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataAcquisition:
    """Handle data acquisition from various sources"""
    
    def __init__(self, config_path="Data-Pipeline/config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = Path(self.config['data']['raw_data_path'])
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
    
    def fetch_data(self):
        """
        Fetch data from source
        
        Returns:
            pd.DataFrame: Raw data
        """
        logger.info("Starting data acquisition...")
        
        source_type = self.config['data']['source']['type']
        
        try:
            if source_type == 'api':
                data = self._fetch_from_api()
            elif source_type == 'database':
                data = self._fetch_from_database()
            elif source_type == 'file':
                data = self._fetch_from_file()
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            logger.info(f"Successfully fetched {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def _fetch_from_api(self):
        """Fetch data from API"""
        url = self.config['data']['source']['url']
        logger.info(f"Fetching data from API: {url}")
        
        # TODO: Implement your API fetching logic
        # Example:
        # response = requests.get(url, headers={'Authorization': f'Bearer {api_key}'})
        # data = pd.DataFrame(response.json())
        
        # Placeholder - replace with actual implementation
        data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': ['A', 'B', 'A', 'C', 'B'],
            'feature_3': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 1]
        })
        
        return data
    
    def _fetch_from_database(self):
        """Fetch data from database"""
        logger.info("Fetching data from database")
        
        # TODO: Implement database fetching logic
        # Example with SQLAlchemy:
        # from sqlalchemy import create_engine
        # engine = create_engine(connection_string)
        # data = pd.read_sql_query("SELECT * FROM table", engine)
        
        raise NotImplementedError("Database fetching not implemented")
    
    def _fetch_from_file(self):
        """Fetch data from file"""
        file_path = self.config['data']['source'].get('path')
        logger.info(f"Loading data from file: {file_path}")
        
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def save_data(self, data):
        """
        Save raw data
        
        Args:
            data (pd.DataFrame): Data to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.raw_data_path / f"dataset_{timestamp}.csv"
        
        data.to_csv(output_path, index=False)
        logger.info(f"Data saved to: {output_path}")
        
        # Also save as latest
        latest_path = self.raw_data_path / "dataset.csv"
        data.to_csv(latest_path, index=False)
        logger.info(f"Latest data saved to: {latest_path}")
        
        return output_path
    
    def run(self):
        """Execute data acquisition pipeline"""
        try:
            data = self.fetch_data()
            output_path = self.save_data(data)
            logger.info("Data acquisition completed successfully")
            return output_path
        except Exception as e:
            logger.error(f"Data acquisition failed: {str(e)}")
            raise


if __name__ == "__main__":
    acquisition = DataAcquisition()
    acquisition.run()
