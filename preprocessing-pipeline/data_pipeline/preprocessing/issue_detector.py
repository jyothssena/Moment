"""
Issue Detection Module
Detects PII, profanity, spam in text for the MOMENT preprocessing pipeline.
"""

import re
from typing import Dict, Any, List


class IssueDetector:
    """
    Universal issue detector that checks for problematic content in text.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize IssueDetector with configuration.
        
        Args:
            config: Dictionary with issue detection settings from config.yaml
        """
        # Default settings if no config provided
        self.config = config or {
            'check_pii': True,
            'check_profanity': True,
            'check_spam': True,
            'profanity_threshold': 0.3
        }
        
        # Basic profanity word list (add more as needed)
        # Using mild examples for demonstration
        self.profanity_words = {
            'damn', 'hell', 'crap', 'stupid', 'idiot',
            # Add more comprehensive list in production
        }
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Main detection function - checks for all issues.
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with detection results:
            {
                'has_pii': bool,
                'pii_types': list,
                'has_profanity': bool,
                'profanity_count': int,
                'is_spam': bool,
                'spam_indicators': list
            }
        """
        if not text or not isinstance(text, str):
            return {
                'has_pii': False,
                'pii_types': [],
                'has_profanity': False,
                'profanity_count': 0,
                'is_spam': False,
                'spam_indicators': []
            }
        
        results = {}
        
        # PII Detection
        if self.config.get('check_pii', True):
            pii_result = self._detect_pii(text)
            results['has_pii'] = pii_result['has_pii']
            results['pii_types'] = pii_result['pii_types']
        else:
            results['has_pii'] = False
            results['pii_types'] = []
        
        # Profanity Detection
        if self.config.get('check_profanity', True):
            profanity_result = self._detect_profanity(text)
            results['has_profanity'] = profanity_result['has_profanity']
            results['profanity_count'] = profanity_result['count']
        else:
            results['has_profanity'] = False
            results['profanity_count'] = 0
        
        # Spam Detection
        if self.config.get('check_spam', True):
            spam_result = self._detect_spam(text)
            results['is_spam'] = spam_result['is_spam']
            results['spam_indicators'] = spam_result['indicators']
        else:
            results['is_spam'] = False
            results['spam_indicators'] = []
        
        return results
    
    def _detect_pii(self, text: str) -> Dict[str, Any]:
        """Detect Personal Identifiable Information."""
        pii_types = []
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, text):
            pii_types.append('email')
        
        # Phone number patterns (US format)
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890 or 1234567890
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',     # (123) 456-7890
            r'\+1[-.\s]?\d{3}[-.]?\d{3}[-.]?\d{4}'  # +1-123-456-7890
        ]
        for pattern in phone_patterns:
            if re.search(pattern, text):
                pii_types.append('phone')
                break
        
        # SSN pattern (US)
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        if re.search(ssn_pattern, text):
            pii_types.append('ssn')
        
        # Credit card pattern (basic check - 13-19 digits)
        cc_pattern = r'\b\d{13,19}\b'
        if re.search(cc_pattern, text):
            pii_types.append('potential_credit_card')
        
        # Remove duplicates
        pii_types = list(set(pii_types))
        
        return {
            'has_pii': len(pii_types) > 0,
            'pii_types': pii_types
        }
    
    def _detect_profanity(self, text: str) -> Dict[str, Any]:
        """Detect profanity and inappropriate language."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Count profanity words
        profanity_count = sum(1 for word in words if word in self.profanity_words)
        
        # Calculate profanity ratio
        total_words = len(words)
        profanity_ratio = profanity_count / total_words if total_words > 0 else 0
        
        # Flag if profanity ratio exceeds threshold
        threshold = self.config.get('profanity_threshold', 0.3)
        has_profanity = profanity_ratio > threshold
        
        return {
            'has_profanity': has_profanity,
            'count': profanity_count,
            'ratio': round(profanity_ratio, 3)
        }
    
    def _detect_spam(self, text: str) -> Dict[str, Any]:
        """Detect spam patterns."""
        indicators = []
        
        # Check for excessive capitalization
        if len(text) > 0:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.5:  # More than 50% caps
                indicators.append('excessive_caps')
        
        # Check for excessive punctuation
        punct_count = sum(1 for c in text if c in '!?.')
        if punct_count > 10:
            indicators.append('excessive_punctuation')
        
        # Check for repetitive characters (e.g., "!!!!!", "????")
        if re.search(r'(.)\1{4,}', text):
            indicators.append('repetitive_characters')
        
        # Check for repetitive words
        words = text.lower().split()
        if len(words) > 0:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # If any word appears more than 30% of the time
            max_repetition = max(word_counts.values()) / len(words)
            if max_repetition > 0.3:
                indicators.append('repetitive_words')
        
        # Check for common spam phrases
        spam_phrases = [
            'click here', 'buy now', 'limited time', 'act now',
            'free money', 'guaranteed', 'no risk'
        ]
        text_lower = text.lower()
        for phrase in spam_phrases:
            if phrase in text_lower:
                indicators.append(f'spam_phrase_{phrase.replace(" ", "_")}')
        
        return {
            'is_spam': len(indicators) > 0,
            'indicators': indicators
        }
    
    def detect_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect issues in multiple texts at once.
        
        Args:
            texts: List of texts to check
            
        Returns:
            List of detection results
        """
        return [self.detect(text) for text in texts]


# Convenience function for quick usage
def detect_issues(text: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Quick function to detect issues in a single text.
    
    Args:
        text: Text to check
        config: Optional detection configuration
        
    Returns:
        Detection results dictionary
    """
    detector = IssueDetector(config)
    return detector.detect(text)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "This is a clean interpretation about Frankenstein with no issues.",
        "Contact me at test@example.com or call 123-456-7890!",
        "THIS IS ALL CAPS AND LOOKS LIKE SPAM!!!!",
        "Buy now! Limited time offer! Click here! Guaranteed results!",
        "This damn interpretation is stupid but otherwise fine.",
    ]
    
    print("Testing IssueDetector:\n")
    detector = IssueDetector()
    
    for i, text in enumerate(test_texts, 1):
        result = detector.detect(text)
        print(f"Test {i}: {text[:60]}...")
        print(f"  PII: {result['has_pii']} - {result['pii_types']}")
        print(f"  Profanity: {result['has_profanity']} (count: {result['profanity_count']})")
        print(f"  Spam: {result['is_spam']} - {result['spam_indicators']}")
        print()