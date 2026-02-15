"""
Preprocessor Module
Orchestrates all preprocessing operations for the MOMENT pipeline.
"""

from typing import Dict, Any
from datetime import datetime, UTC

from .text_cleaner import TextCleaner
from .text_validator import TextValidator
from .issue_detector import IssueDetector
from .metrics_calculator import MetricsCalculator


class Preprocessor:
    """
    Universal preprocessor that orchestrates all preprocessing operations.
    Works on any text input regardless of source.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Preprocessor with configuration.
        
        Args:
            config: Dictionary with preprocessing settings from config.yaml
        """
        self.config = config or {}
        
        # Initialize all preprocessing modules
        cleaning_config = self.config.get('text_cleaning', {})
        validation_config = self.config.get('validation', {})
        issue_config = self.config.get('issue_detection', {})
        
        self.cleaner = TextCleaner(cleaning_config)
        self.validator = TextValidator(validation_config)
        self.issue_detector = IssueDetector(issue_config)
        self.metrics_calculator = MetricsCalculator()
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Main preprocessing function - applies all operations in sequence.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Dictionary with complete preprocessing results:
            {
                'original_text': str,
                'cleaned_text': str,
                'is_valid': bool,
                'quality_score': float,
                'quality_issues': list,
                'detected_issues': dict,
                'metrics': dict,
                'timestamp': str
            }
        """
        # Store original text
        original_text = text
        
        # Step 1: Clean the text
        cleaned_text = self.cleaner.clean(text)
        
        # Step 2: Validate the cleaned text
        validation_result = self.validator.validate(cleaned_text)
        
        # Step 3: Detect issues (PII, profanity, spam)
        issue_result = self.issue_detector.detect(cleaned_text)
        
        # Step 4: Calculate metrics
        metrics = self.metrics_calculator.calculate(cleaned_text)
        
        # Compile complete result
        result = {
            'original_text': original_text,
            'cleaned_text': cleaned_text,
            'is_valid': validation_result['is_valid'],
            'quality_score': validation_result['quality_score'],
            'quality_issues': validation_result['quality_issues'],
            'detected_issues': {
                'has_pii': issue_result['has_pii'],
                'pii_types': issue_result['pii_types'],
                'has_profanity': issue_result['has_profanity'],
                'is_spam': issue_result['is_spam'],
                'spam_indicators': issue_result['spam_indicators']
            },
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat() + 'Z'

        }
        
        return result
    
    def process_batch(self, texts: list) -> list:
        """
        Process multiple texts at once.
        
        Args:
            texts: List of raw texts to preprocess
            
        Returns:
            List of preprocessing results
        """
        return [self.process(text) for text in texts]
    
    def process_with_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text with additional context (user_id, book_id, etc.).
        
        Args:
            text: Raw text to preprocess
            context: Additional context dictionary (e.g., user_id, passage_id)
            
        Returns:
            Dictionary with preprocessing results + context
        """
        # Get preprocessing results
        result = self.process(text)
        
        # Add context
        result['context'] = context
        
        return result


# Convenience function for quick usage
def preprocess_text(text: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Quick function to preprocess a single text.
    
    Args:
        text: Raw text to preprocess
        config: Optional preprocessing configuration
        
    Returns:
        Complete preprocessing results
    """
    preprocessor = Preprocessor(config)
    return preprocessor.process(text)


# Example usage and testing
if __name__ == "__main__":
    # Test case
    test_text = """
    He says   "catastrophe" before anything bad happens. Just... think about that. 
    The creature opened its eyes. That's it. Victor's already calling it disaster.
    Beautiful!—Great God! Right next to each other. His brain's breaking.
    """
    
    print("Testing Preprocessor:\n")
    print("="*70)
    print("Original Text:")
    print(test_text)
    print("="*70)
    
    preprocessor = Preprocessor()
    result = preprocessor.process(test_text)
    
    print("\nPreprocessing Results:")
    print("="*70)
    print(f"Cleaned Text: {result['cleaned_text']}\n")
    print(f"Is Valid: {result['is_valid']}")
    print(f"Quality Score: {result['quality_score']}")
    print(f"Quality Issues: {result['quality_issues']}\n")
    
    print("Detected Issues:")
    for key, value in result['detected_issues'].items():
        print(f"  {key}: {value}")
    
    print("\nMetrics:")
    for key, value in result['metrics'].items():
        print(f"  {key}: {value}")
    
    print(f"\nTimestamp: {result['timestamp']}")
    print("="*70)