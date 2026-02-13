"""
Data Preprocessing Module
Cleans, transforms, and prepares data for feature engineering
"""

import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data preprocessing tasks"""
    
    def __init__(self, config_path="Data-Pipeline/config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_path = Path(self.config['data']['processed_data_path'])
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_data(self, file_path):
        """Load raw data"""
        logger.info(f"Loading data from: {file_path}")
        return pd.read_csv(file_path)
    
    def handle_missing_values(self, df):
        """
        Handle missing values
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        logger.info("Handling missing values...")
        
        method = self.config['preprocessing'].get('handle_missing', 'drop')
        
        missing_before = df.isnull().sum().sum()
        logger.info(f"Missing values before: {missing_before}")
        
        if method == 'drop':
            df = df.dropna()
        elif method == 'impute':
            # Impute numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
            
            # Impute categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values after: {missing_after}")
        
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        logger.info("Removing duplicates...")
        
        duplicates_before = df.duplicated().sum()
        logger.info(f"Duplicates before: {duplicates_before}")
        
        df = df.drop_duplicates()
        
        duplicates_after = df.duplicated().sum()
        logger.info(f"Duplicates after: {duplicates_after}")
        
        return df
    
    def handle_outliers(self, df, columns=None, method='iqr', threshold=3.0):
        """
        Handle outliers in numerical columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to check for outliers
            method (str): 'iqr' or 'zscore'
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Dataframe with outliers handled
        """
        logger.info(f"Handling outliers using {method} method...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        rows_before = len(df)
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        rows_after = len(df)
        logger.info(f"Removed {rows_before - rows_after} outliers")
        
        return df
    
    def encode_categorical(self, df, columns=None):
        """
        Encode categorical variables
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to encode
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        logger.info("Encoding categorical variables...")
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns
        
        encoding_method = self.config['preprocessing'].get('encoding', 'label')
        
        if encoding_method == 'label':
            for col in columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
        elif encoding_method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
        
        logger.info(f"Encoded {len(columns)} categorical columns")
        
        return df
    
    def normalize_features(self, df, exclude_columns=None):
        """
        Normalize numerical features
        
        Args:
            df (pd.DataFrame): Input dataframe
            exclude_columns (list): Columns to exclude from normalization
            
        Returns:
            pd.DataFrame: Dataframe with normalized features
        """
        if not self.config['preprocessing'].get('normalization', False):
            return df
        
        logger.info("Normalizing numerical features...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if exclude_columns:
            numerical_cols = [col for col in numerical_cols if col not in exclude_columns]
        
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        logger.info(f"Normalized {len(numerical_cols)} numerical columns")
        
        return df
    
    def preprocess(self, df):
        """
        Run full preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        logger.info("Starting preprocessing pipeline...")
        
        # 1. Handle missing values
        df = self.handle_missing_values(df)
        
        # 2. Remove duplicates
        df = self.remove_duplicates(df)
        
        # 3. Handle outliers
        df = self.handle_outliers(df)
        
        # 4. Encode categorical variables
        df = self.encode_categorical(df)
        
        # 5. Normalize features (exclude target column)
        df = self.normalize_features(df, exclude_columns=['target'])
        
        logger.info("Preprocessing completed successfully")
        
        return df
    
    def save_processed_data(self, df):
        """Save preprocessed data"""
        output_path = self.processed_path / "processed_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to: {output_path}")
        return output_path
    
    def run(self, input_path="Data-Pipeline/data/raw/dataset.csv"):
        """Execute preprocessing pipeline"""
        try:
            df = self.load_data(input_path)
            df_processed = self.preprocess(df)
            output_path = self.save_processed_data(df_processed)
            logger.info("Preprocessing completed successfully")
            return output_path
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run()
