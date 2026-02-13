"""
Data Validation Module
Validates data quality, schema, and detects anomalies
"""

import logging
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Handle data validation and quality checks"""
    
    def __init__(self, config_path="Data-Pipeline/config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.schema_path = Path(self.config['validation']['schema_path'])
        self.reports_path = Path("Data-Pipeline/reports/statistics")
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        self.validation_results = {}
        self.anomalies = []
    
    def load_data(self, file_path):
        """Load feature data"""
        logger.info(f"Loading data from: {file_path}")
        return pd.read_csv(file_path)
    
    def load_schema(self):
        """Load data schema"""
        if self.schema_path.exists():
            with open(self.schema_path, 'r') as f:
                return yaml.safe_load(f)
        return None
    
    def validate_schema(self, df):
        """
        Validate data against schema
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            bool: True if valid, False otherwise
        """
        logger.info("Validating data schema...")
        
        schema = self.load_schema()
        if not schema:
            logger.warning("No schema found, skipping schema validation")
            return True
        
        is_valid = True
        issues = []
        
        # Check required features
        required_features = [f['name'] for f in schema.get('features', []) if f.get('required', False)]
        for feature in required_features:
            if feature not in df.columns:
                is_valid = False
                issue = f"Missing required feature: {feature}"
                issues.append(issue)
                logger.error(issue)
        
        # Check data types and constraints
        for feature_spec in schema.get('features', []):
            feature_name = feature_spec['name']
            if feature_name not in df.columns:
                continue
            
            feature_type = feature_spec['type']
            
            # Check numerical constraints
            if feature_type == 'numerical':
                if 'min' in feature_spec:
                    violations = (df[feature_name] < feature_spec['min']).sum()
                    if violations > 0:
                        is_valid = False
                        issue = f"{feature_name}: {violations} values below minimum {feature_spec['min']}"
                        issues.append(issue)
                        logger.error(issue)
                
                if 'max' in feature_spec:
                    violations = (df[feature_name] > feature_spec['max']).sum()
                    if violations > 0:
                        is_valid = False
                        issue = f"{feature_name}: {violations} values above maximum {feature_spec['max']}"
                        issues.append(issue)
                        logger.error(issue)
            
            # Check categorical constraints
            elif feature_type == 'categorical':
                if 'allowed_values' in feature_spec:
                    invalid = ~df[feature_name].isin(feature_spec['allowed_values'])
                    violations = invalid.sum()
                    if violations > 0:
                        is_valid = False
                        issue = f"{feature_name}: {violations} invalid categorical values"
                        issues.append(issue)
                        logger.error(issue)
        
        self.validation_results['schema_validation'] = {
            'valid': is_valid,
            'issues': issues
        }
        
        return is_valid
    
    def detect_missing_values(self, df):
        """Detect missing values"""
        logger.info("Detecting missing values...")
        
        threshold = self.config['validation']['anomaly_thresholds']['missing_values']
        
        missing_stats = {}
        alerts = []
        
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            missing_stats[col] = {
                'count': int(df[col].isnull().sum()),
                'percentage': float(missing_pct)
            }
            
            if missing_pct > threshold:
                alert = f"{col}: {missing_pct:.2%} missing values (threshold: {threshold:.2%})"
                alerts.append(alert)
                logger.warning(alert)
                self.anomalies.append({
                    'type': 'missing_values',
                    'column': col,
                    'severity': 'high' if missing_pct > 2 * threshold else 'medium',
                    'message': alert
                })
        
        self.validation_results['missing_values'] = missing_stats
        
        return alerts
    
    def detect_outliers(self, df):
        """Detect outliers in numerical columns"""
        logger.info("Detecting outliers...")
        
        threshold = self.config['validation']['anomaly_thresholds']['outlier_std']
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_stats = {}
        
        for col in numerical_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > threshold).sum()
            outlier_pct = outliers / len(df)
            
            outlier_stats[col] = {
                'count': int(outliers),
                'percentage': float(outlier_pct)
            }
            
            if outlier_pct > 0.05:  # 5% threshold
                alert = f"{col}: {outlier_pct:.2%} outliers detected"
                logger.warning(alert)
                self.anomalies.append({
                    'type': 'outliers',
                    'column': col,
                    'severity': 'medium',
                    'message': alert
                })
        
        self.validation_results['outliers'] = outlier_stats
    
    def generate_statistics(self, df):
        """Generate comprehensive data statistics"""
        logger.info("Generating data statistics...")
        
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'timestamp': datetime.now().isoformat()
        }
        
        # Numerical statistics
        numerical_stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            numerical_stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
        
        stats['numerical_statistics'] = numerical_stats
        
        # Categorical statistics
        categorical_stats = {}
        for col in df.select_dtypes(include=['object']).columns:
            value_counts = df[col].value_counts().to_dict()
            categorical_stats[col] = {
                'unique_values': int(df[col].nunique()),
                'top_values': {str(k): int(v) for k, v in list(value_counts.items())[:5]}
            }
        
        stats['categorical_statistics'] = categorical_stats
        
        self.validation_results['statistics'] = stats
        
        return stats
    
    def validate(self, df):
        """
        Run complete validation pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Validation results
        """
        logger.info("Starting data validation...")
        
        # Validate schema
        schema_valid = self.validate_schema(df)
        
        # Detect missing values
        self.detect_missing_values(df)
        
        # Detect outliers
        self.detect_outliers(df)
        
        # Generate statistics
        self.generate_statistics(df)
        
        # Overall validation status
        self.validation_results['overall_valid'] = schema_valid and len(self.anomalies) == 0
        self.validation_results['anomalies'] = self.anomalies
        
        logger.info(f"Validation completed. Valid: {self.validation_results['overall_valid']}")
        
        return self.validation_results
    
    def save_validation_report(self):
        """Save validation results to JSON"""
        output_path = self.reports_path / "validation_metrics.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info(f"Validation report saved to: {output_path}")
        
        return output_path
    
    def trigger_alerts(self):
        """Trigger alerts for anomalies"""
        if not self.anomalies:
            logger.info("No anomalies detected, no alerts triggered")
            return
        
        logger.warning(f"Found {len(self.anomalies)} anomalies!")
        
        # TODO: Implement actual alerting (email, Slack, etc.)
        for anomaly in self.anomalies:
            logger.warning(f"ALERT - {anomaly['type']}: {anomaly['message']}")
        
        # Placeholder for email/Slack alerts
        # self._send_email_alert()
        # self._send_slack_alert()
    
    def run(self, input_path="Data-Pipeline/data/features/features.csv"):
        """Execute validation pipeline"""
        try:
            df = self.load_data(input_path)
            self.validate(df)
            self.save_validation_report()
            self.trigger_alerts()
            logger.info("Validation completed successfully")
            return self.validation_results
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise


if __name__ == "__main__":
    validator = DataValidator()
    validator.run()
