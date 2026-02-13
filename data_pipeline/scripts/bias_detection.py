"""
Bias Detection Module
Detects bias in data using slicing techniques
"""

import logging
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from collections import defaultdict


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BiasDetector:
    """Detect bias in data using slicing techniques"""
    
    def __init__(self, config_path="Data-Pipeline/config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.bias_enabled = self.config['bias_detection']['enabled']
        self.slice_features = self.config['bias_detection']['slice_features']
        self.fairness_threshold = self.config['bias_detection']['fairness_threshold']
        
        self.reports_path = Path("Data-Pipeline/reports/bias_detection")
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        self.bias_results = {}
    
    def load_data(self, file_path):
        """Load feature data"""
        logger.info(f"Loading data from: {file_path}")
        return pd.read_csv(file_path)
    
    def create_slices(self, df):
        """
        Create data slices based on configured features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Dictionary of slices
        """
        logger.info("Creating data slices...")
        
        slices = {}
        
        for feature in self.slice_features:
            if feature not in df.columns:
                logger.warning(f"Slice feature '{feature}' not found in data")
                continue
            
            unique_values = df[feature].unique()
            for value in unique_values:
                slice_name = f"{feature}_{value}"
                slices[slice_name] = df[df[feature] == value]
                logger.info(f"Created slice '{slice_name}' with {len(slices[slice_name])} rows")
        
        return slices
    
    def analyze_slice_statistics(self, slices):
        """
        Analyze statistics for each slice
        
        Args:
            slices (dict): Dictionary of data slices
            
        Returns:
            dict: Statistics for each slice
        """
        logger.info("Analyzing slice statistics...")
        
        slice_stats = {}
        
        for slice_name, slice_df in slices.items():
            stats = {
                'size': len(slice_df),
                'percentage': len(slice_df) / len(pd.concat(slices.values()).drop_duplicates()) * 100
            }
            
            # Calculate statistics for numerical features
            numerical_cols = slice_df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if col != 'target':  # Exclude target from feature statistics
                    stats[f'{col}_mean'] = float(slice_df[col].mean())
                    stats[f'{col}_std'] = float(slice_df[col].std())
            
            # Calculate target distribution if available
            if 'target' in slice_df.columns:
                target_dist = slice_df['target'].value_counts(normalize=True).to_dict()
                stats['target_distribution'] = {str(k): float(v) for k, v in target_dist.items()}
            
            slice_stats[slice_name] = stats
        
        return slice_stats
    
    def detect_representation_bias(self, slice_stats):
        """
        Detect representation bias (imbalanced slice sizes)
        
        Args:
            slice_stats (dict): Slice statistics
            
        Returns:
            list: Detected biases
        """
        logger.info("Detecting representation bias...")
        
        biases = []
        sizes = [stats['size'] for stats in slice_stats.values()]
        mean_size = np.mean(sizes)
        
        for slice_name, stats in slice_stats.items():
            deviation = abs(stats['size'] - mean_size) / mean_size
            
            if deviation > self.fairness_threshold:
                bias = {
                    'type': 'representation',
                    'slice': slice_name,
                    'size': stats['size'],
                    'percentage': stats['percentage'],
                    'deviation': float(deviation),
                    'severity': 'high' if deviation > 2 * self.fairness_threshold else 'medium'
                }
                biases.append(bias)
                logger.warning(f"Representation bias detected in '{slice_name}': {deviation:.2%} deviation")
        
        return biases
    
    def detect_statistical_bias(self, slice_stats):
        """
        Detect statistical bias in feature distributions
        
        Args:
            slice_stats (dict): Slice statistics
            
        Returns:
            list: Detected biases
        """
        logger.info("Detecting statistical bias...")
        
        biases = []
        
        # Get all numerical features (excluding size, percentage, target_distribution)
        numerical_features = set()
        for stats in slice_stats.values():
            for key in stats.keys():
                if key.endswith('_mean') or key.endswith('_std'):
                    feature = key.rsplit('_', 1)[0]
                    numerical_features.add(feature)
        
        for feature in numerical_features:
            means = [stats.get(f'{feature}_mean', 0) for stats in slice_stats.values()]
            if len(means) < 2:
                continue
            
            mean_of_means = np.mean(means)
            max_deviation = max([abs(m - mean_of_means) / (mean_of_means + 1e-10) for m in means])
            
            if max_deviation > self.fairness_threshold:
                bias = {
                    'type': 'statistical',
                    'feature': feature,
                    'max_deviation': float(max_deviation),
                    'severity': 'high' if max_deviation > 2 * self.fairness_threshold else 'medium'
                }
                biases.append(bias)
                logger.warning(f"Statistical bias detected in '{feature}': {max_deviation:.2%} deviation")
        
        return biases
    
    def detect_target_bias(self, slice_stats):
        """
        Detect bias in target distribution across slices
        
        Args:
            slice_stats (dict): Slice statistics
            
        Returns:
            list: Detected biases
        """
        logger.info("Detecting target distribution bias...")
        
        biases = []
        
        # Check if target information is available
        has_target = any('target_distribution' in stats for stats in slice_stats.values())
        if not has_target:
            logger.info("No target variable found, skipping target bias detection")
            return biases
        
        # Calculate overall target distribution
        all_targets = []
        for slice_name, stats in slice_stats.items():
            if 'target_distribution' in stats:
                for target_value, proportion in stats['target_distribution'].items():
                    all_targets.extend([target_value] * int(proportion * stats['size']))
        
        overall_dist = pd.Series(all_targets).value_counts(normalize=True).to_dict()
        
        # Compare each slice to overall distribution
        for slice_name, stats in slice_stats.items():
            if 'target_distribution' not in stats:
                continue
            
            slice_dist = stats['target_distribution']
            
            # Calculate maximum deviation from overall distribution
            max_deviation = 0
            for target_value in overall_dist.keys():
                overall_prop = overall_dist.get(str(target_value), 0)
                slice_prop = slice_dist.get(str(target_value), 0)
                deviation = abs(slice_prop - overall_prop)
                max_deviation = max(max_deviation, deviation)
            
            if max_deviation > self.fairness_threshold:
                bias = {
                    'type': 'target_distribution',
                    'slice': slice_name,
                    'deviation': float(max_deviation),
                    'overall_distribution': {str(k): float(v) for k, v in overall_dist.items()},
                    'slice_distribution': slice_dist,
                    'severity': 'high' if max_deviation > 2 * self.fairness_threshold else 'medium'
                }
                biases.append(bias)
                logger.warning(f"Target bias detected in '{slice_name}': {max_deviation:.2%} deviation")
        
        return biases
    
    def suggest_mitigations(self, biases):
        """
        Suggest mitigation strategies for detected biases
        
        Args:
            biases (list): List of detected biases
            
        Returns:
            dict: Mitigation suggestions
        """
        mitigations = defaultdict(list)
        
        for bias in biases:
            if bias['type'] == 'representation':
                mitigations[bias['slice']].append({
                    'strategy': 'oversampling',
                    'description': f"Oversample underrepresented slice to balance representation"
                })
                mitigations[bias['slice']].append({
                    'strategy': 'synthetic_data',
                    'description': f"Generate synthetic samples for underrepresented slice using SMOTE or similar"
                })
            
            elif bias['type'] == 'statistical':
                mitigations[bias['feature']].append({
                    'strategy': 'feature_transformation',
                    'description': f"Apply feature-specific normalization or transformation"
                })
            
            elif bias['type'] == 'target_distribution':
                mitigations[bias['slice']].append({
                    'strategy': 'stratified_sampling',
                    'description': f"Use stratified sampling to balance target distribution"
                })
                mitigations[bias['slice']].append({
                    'strategy': 'class_weights',
                    'description': f"Apply class weights during model training to account for imbalance"
                })
        
        return dict(mitigations)
    
    def detect_bias(self, df):
        """
        Run complete bias detection pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Bias detection results
        """
        if not self.bias_enabled:
            logger.info("Bias detection disabled in configuration")
            return {'enabled': False}
        
        logger.info("Starting bias detection...")
        
        # Create slices
        slices = self.create_slices(df)
        
        if not slices:
            logger.warning("No slices created, skipping bias detection")
            return {'enabled': True, 'slices_created': False}
        
        # Analyze slice statistics
        slice_stats = self.analyze_slice_statistics(slices)
        
        # Detect different types of bias
        representation_biases = self.detect_representation_bias(slice_stats)
        statistical_biases = self.detect_statistical_bias(slice_stats)
        target_biases = self.detect_target_bias(slice_stats)
        
        # Combine all biases
        all_biases = representation_biases + statistical_biases + target_biases
        
        # Suggest mitigations
        mitigations = self.suggest_mitigations(all_biases)
        
        # Compile results
        self.bias_results = {
            'enabled': True,
            'slices_created': True,
            'total_slices': len(slices),
            'slice_statistics': slice_stats,
            'biases_detected': len(all_biases),
            'biases': all_biases,
            'mitigation_suggestions': mitigations
        }
        
        logger.info(f"Bias detection completed. Found {len(all_biases)} biases")
        
        return self.bias_results
    
    def save_bias_report(self):
        """Save bias detection results to JSON"""
        output_path = self.reports_path / "bias_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.bias_results, f, indent=2)
        
        logger.info(f"Bias report saved to: {output_path}")
        
        return output_path
    
    def run(self, input_path="Data-Pipeline/data/features/features.csv"):
        """Execute bias detection pipeline"""
        try:
            df = self.load_data(input_path)
            self.detect_bias(df)
            self.save_bias_report()
            logger.info("Bias detection completed successfully")
            return self.bias_results
        except Exception as e:
            logger.error(f"Bias detection failed: {str(e)}")
            raise


if __name__ == "__main__":
    detector = BiasDetector()
    detector.run()
