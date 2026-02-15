"""
Text Validation Module
Validates text quality for the MOMENT preprocessing pipeline.
"""

import re
from typing import Dict, Any, Tuple
from langdetect import detect, LangDetectException # type: ignore


class TextValidator:
    """
    Universal text validator that checks quality of any text input.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize TextValidator with configuration.
        
        Args:
            config: Dictionary with validation settings from config.yaml
        """
        # Default settings if no config provided
        self.config = config or {
            'min_words': 10,
            'max_words': 500,
            'min_chars': 50,
            'max_chars': 3000,
            'allowed_languages': ['en'],
            'quality_threshold': 0.5
        }
    
    def validate(self, text: str) -> Dict[str, Any]:
        """
        Main validation function - performs all quality checks.
        
        Args:
            text: Text to validate
            
        Returns:
            Dictionary with validation results:
            {
                'is_valid': bool,
                'quality_score': float,
                'quality_issues': list,
                'checks': dict
            }
        """
        if not text or not isinstance(text, str):
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'quality_issues': ['empty_text'],
                'checks': {}
            }
        
        # Run all validation checks
        checks = {
            'length': self._check_length(text),
            'language': self._check_language(text),
            'gibberish': self._check_gibberish(text),
            'character_diversity': self._check_character_diversity(text)
        }
        
        # Collect quality issues
        quality_issues = []
        for check_name, result in checks.items():
            if not result['passed']:
                quality_issues.append(result.get('issue', check_name))
        
        # Calculate overall quality score (0-1)
        quality_score = self._calculate_quality_score(checks)
        
        # Determine if valid based on threshold
        is_valid = quality_score >= self.config.get('quality_threshold', 0.5)
        
        return {
            'is_valid': is_valid,
            'quality_score': round(quality_score, 2),
            'quality_issues': quality_issues,
            'checks': checks
        }
    
    def _check_length(self, text: str) -> Dict[str, Any]:
        """Check if text length is within acceptable range."""
        word_count = len(text.split())
        char_count = len(text)
        
        min_words = self.config.get('min_words', 10)
        max_words = self.config.get('max_words', 500)
        min_chars = self.config.get('min_chars', 50)
        max_chars = self.config.get('max_chars', 3000)
        
        issues = []
        if word_count < min_words:
            issues.append(f'too_short_{word_count}_words')
        if word_count > max_words:
            issues.append(f'too_long_{word_count}_words')
        if char_count < min_chars:
            issues.append(f'too_short_{char_count}_chars')
        if char_count > max_chars:
            issues.append(f'too_long_{char_count}_chars')
        
        passed = len(issues) == 0
        
        return {
            'passed': passed,
            'word_count': word_count,
            'char_count': char_count,
            'issue': issues[0] if issues else None
        }
    
    def _check_language(self, text: str) -> Dict[str, Any]:
        """Check if text is in an allowed language."""
        try:
            detected_lang = detect(text)
            allowed_langs = self.config.get('allowed_languages', ['en'])
            
            passed = detected_lang in allowed_langs
            
            return {
                'passed': passed,
                'detected_language': detected_lang,
                'issue': f'possibly_non_english_{detected_lang}' if not passed else None
            }
        except LangDetectException:
            # If language detection fails, assume it's gibberish or too short
            return {
                'passed': False,
                'detected_language': 'unknown',
                'issue': 'language_detection_failed'
            }
    
    def _check_gibberish(self, text: str) -> Dict[str, Any]:
        """Check if text appears to be gibberish."""
        # Simple heuristic: check ratio of vowels to consonants
        text_lower = text.lower()
        vowels = sum(1 for char in text_lower if char in 'aeiou')
        consonants = sum(1 for char in text_lower if char.isalpha() and char not in 'aeiou')
        
        total_alpha = vowels + consonants
        
        if total_alpha == 0:
            return {
                'passed': False,
                'vowel_ratio': 0.0,
                'issue': 'no_alphabetic_characters'
            }
        
        vowel_ratio = vowels / total_alpha
        
        # Normal English text has vowel ratio between 0.35-0.45
        # Flag as gibberish if outside 0.15-0.65 range (very permissive)
        is_gibberish = vowel_ratio < 0.15 or vowel_ratio > 0.65
        
        return {
            'passed': not is_gibberish,
            'vowel_ratio': round(vowel_ratio, 2),
            'issue': f'possible_gibberish_{vowel_ratio:.2f}' if is_gibberish else None
        }
    
    def _check_character_diversity(self, text: str) -> Dict[str, Any]:
        """Check if text has reasonable character diversity."""
        # Remove spaces and get unique characters
        text_no_space = text.replace(' ', '')
        
        if len(text_no_space) == 0:
            return {
                'passed': False,
                'diversity_ratio': 0.0,
                'issue': 'no_characters'
            }
        
        unique_chars = len(set(text_no_space.lower()))
        total_chars = len(text_no_space)
        diversity_ratio = unique_chars / total_chars
        
        # Low diversity might indicate spam (e.g., "aaaaaaa")
        # Flag if diversity is below 0.3 (very permissive threshold)
        low_diversity = diversity_ratio < 0.3
        
        return {
            'passed': not low_diversity,
            'diversity_ratio': round(diversity_ratio, 2),
            'unique_chars': unique_chars,
            'issue': f'low_character_diversity_{diversity_ratio:.2f}' if low_diversity else None
        }
    
    def _calculate_quality_score(self, checks: Dict[str, Any]) -> float:
        """
        Calculate overall quality score based on all checks.
        
        Returns score between 0.0 and 1.0
        """
        # Equal weight for each check
        scores = []
        
        # Length check: 1.0 if passed, 0.5 if failed (not critical)
        scores.append(1.0 if checks['length']['passed'] else 0.5)
        
        # Language check: 1.0 if passed, 0.0 if failed (critical)
        scores.append(1.0 if checks['language']['passed'] else 0.0)
        
        # Gibberish check: 1.0 if passed, 0.2 if failed (critical)
        scores.append(1.0 if checks['gibberish']['passed'] else 0.2)
        
        # Diversity check: 1.0 if passed, 0.5 if failed (not critical)
        scores.append(1.0 if checks['character_diversity']['passed'] else 0.5)
        
        # Average score
        return sum(scores) / len(scores)
    
    def validate_batch(self, texts: list) -> list:
        """
        Validate multiple texts at once.
        
        Args:
            texts: List of texts to validate
            
        Returns:
            List of validation results
        """
        return [self.validate(text) for text in texts]


# Convenience function for quick usage
def validate_text(text: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Quick function to validate a single text.
    
    Args:
        text: Text to validate
        config: Optional validation configuration
        
    Returns:
        Validation results dictionary
    """
    validator = TextValidator(config)
    return validator.validate(text)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "This is a perfectly normal interpretation about Frankenstein's creature and Victor's reaction to bringing it to life.",
        "Short.",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "Ceci est un texte en français qui devrait être détecté.",
        "Normal text that passes all validation checks with flying colors. It has good length, proper language, and reasonable character diversity throughout.",
    ]
    
    print("Testing TextValidator:\n")
    validator = TextValidator()
    
    for i, text in enumerate(test_texts, 1):
        result = validator.validate(text)
        print(f"Test {i}: {text[:50]}...")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Quality Score: {result['quality_score']}")
        print(f"  Issues: {result['quality_issues']}")
        print()