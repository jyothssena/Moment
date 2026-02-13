"""
Feature Engineering Module
Creates new features from preprocessed data
"""

import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handle feature engineering tasks"""
    
    def __init__(self, config_path="Data-Pipeline/config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.features_path = Path(self.config['data']['features_path'])
        self.features_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, file_path):
        """Load preprocessed data"""
        logger.info(f"Loading data from: {file_path}")
        return pd.read_csv(file_path)
    
    def create_interaction_features(self, df):
        """Create interaction features between numerical columns"""
        logger.info("Creating interaction features...")
        
        # TODO: Implement your interaction features
        # Example:
        # df['feature_1_x_feature_3'] = df['feature_1'] * df['feature_3']
        
        return df
    
    def create_polynomial_features(self, df, columns=None, degree=2):
        """Create polynomial features"""
        logger.info(f"Creating polynomial features (degree={degree})...")
        
        # TODO: Implement polynomial features
        # from sklearn.preprocessing import PolynomialFeatures
        
        return df
    
    def create_statistical_features(self, df):
        """Create statistical aggregation features"""
        logger.info("Creating statistical features...")
        
        # TODO: Implement statistical features
        # Example:
        # df['feature_sum'] = df[['feature_1', 'feature_3']].sum(axis=1)
        # df['feature_mean'] = df[['feature_1', 'feature_3']].mean(axis=1)
        
        return df
    
    def engineer_features(self, df):
        """
        Run feature engineering pipeline
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Add your feature engineering steps here
        df = self.create_interaction_features(df)
        df = self.create_statistical_features(df)
        
        logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        
        return df
    
    def save_features(self, df):
        """Save engineered features"""
        output_path = self.features_path / "features.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to: {output_path}")
        return output_path
    
    def run(self, input_path="Data-Pipeline/data/processed/processed_data.csv"):
        """Execute feature engineering pipeline"""
        try:
            df = self.load_data(input_path)
            df_features = self.engineer_features(df)
            output_path = self.save_features(df_features)
            logger.info("Feature engineering completed successfully")
            return output_path
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise


if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.run()
