"""
Metrics Calculator Module
Calculates text metrics for the MOMENT preprocessing pipeline.
"""

import re
from typing import Dict, Any
import textstat # type: ignore


class MetricsCalculator:
    """
    Universal metrics calculator for text analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize MetricsCalculator with configuration.
        
        Args:
            config: Dictionary with metrics settings from config.yaml
        """
        self.config = config or {}
    
    def calculate(self, text: str) -> Dict[str, Any]:
        """
        Calculate all metrics for given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with all calculated metrics:
            {
                'char_count': int,
                'word_count': int,
                'sentence_count': int,
                'avg_word_length': float,
                'avg_sentence_length': float,
                'readability_score': float
            }
        """
        if not text or not isinstance(text, str):
            return self._empty_metrics()
        
        # Basic counts
        char_count = self._count_characters(text)
        word_count = self._count_words(text)
        sentence_count = self._count_sentences(text)
        
        # Averages
        avg_word_length = self._calculate_avg_word_length(text, word_count)
        avg_sentence_length = self._calculate_avg_sentence_length(word_count, sentence_count)
        
        # Readability
        readability_score = self._calculate_readability(text)
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 1),
            'readability_score': round(readability_score, 2)
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics for invalid input."""
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0,
            'readability_score': 0.0
        }
    
    def _count_characters(self, text: str) -> int:
        """Count total characters (excluding whitespace)."""
        return len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
    
    def _count_words(self, text: str) -> int:
        """Count total words."""
        # Split by whitespace and filter empty strings
        words = text.split()
        return len(words)
    
    def _count_sentences(self, text: str) -> int:
        """Count total sentences."""
        # Split by sentence-ending punctuation
        # This is a simple heuristic - can be improved with NLTK
        sentences = re.split(r'[.!?]+', text)
        # Filter empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        return len(sentences)
    
    def _calculate_avg_word_length(self, text: str, word_count: int) -> float:
        """Calculate average word length."""
        if word_count == 0:
            return 0.0
        
        # Get all words (alphanumeric only)
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0
        
        total_length = sum(len(word) for word in words)
        return total_length / len(words)
    
    def _calculate_avg_sentence_length(self, word_count: int, sentence_count: int) -> float:
        """Calculate average sentence length (words per sentence)."""
        if sentence_count == 0:
            return 0.0
        return word_count / sentence_count
    
    def _calculate_readability(self, text: str) -> float:
        """
        Calculate readability score using Flesch Reading Ease.
        
        Score interpretation:
        90-100: Very easy (5th grade)
        80-90: Easy (6th grade)
        70-80: Fairly easy (7th grade)
        60-70: Standard (8th-9th grade)
        50-60: Fairly difficult (10th-12th grade)
        30-50: Difficult (college)
        0-30: Very difficult (college graduate)
        
        Returns score between 0-100 (higher = easier to read)
        """
        try:
            # Flesch Reading Ease score
            score = textstat.flesch_reading_ease(text)
            # Ensure score is within 0-100 range
            return max(0.0, min(100.0, score))
        except:
            # If calculation fails (e.g., text too short), return neutral score
            return 50.0
    
    def calculate_batch(self, texts: list) -> list:
        """
        Calculate metrics for multiple texts at once.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of metrics dictionaries
        """
        return [self.calculate(text) for text in texts]


# Convenience function for quick usage
def calculate_metrics(text: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Quick function to calculate metrics for a single text.
    
    Args:
        text: Text to analyze
        config: Optional metrics configuration
        
    Returns:
        Metrics dictionary
    """
    calculator = MetricsCalculator(config)
    return calculator.calculate(text)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Short text.",
        "This is a longer text with multiple sentences. It has better readability. The metrics should be more interesting.",
        "He says catastrophe before anything bad happens. Just think about that. The creature opened its eyes. That's it. Victor's already calling it disaster.",
        "Very technical jargon obfuscates comprehension substantially.",
        "Simple words make reading easy for everyone.",
    ]
    
    print("Testing MetricsCalculator:\n")
    calculator = MetricsCalculator()
    
    for i, text in enumerate(test_texts, 1):
        metrics = calculator.calculate(text)
        print(f"Test {i}: {text[:50]}...")
        print(f"  Characters: {metrics['char_count']}")
        print(f"  Words: {metrics['word_count']}")
        print(f"  Sentences: {metrics['sentence_count']}")
        print(f"  Avg Word Length: {metrics['avg_word_length']}")
        print(f"  Avg Sentence Length: {metrics['avg_sentence_length']}")
        print(f"  Readability: {metrics['readability_score']}")
        print()