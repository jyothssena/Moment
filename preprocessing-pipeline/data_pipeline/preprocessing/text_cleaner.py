"""
Text Cleaning Module
Handles all text cleaning operations for the MOMENT preprocessing pipeline.
"""

import re
import unicodedata
from typing import Dict, Any


class TextCleaner:
    """
    Universal text cleaner that works on any text input.
    Configurable cleaning operations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize TextCleaner with configuration.
        
        Args:
            config: Dictionary with cleaning settings from config.yaml
        """
        # Default settings if no config provided
        self.config = config or {
            'remove_extra_whitespace': True,
            'normalize_unicode': True,
            'fix_encoding': True,
            'remove_urls': False,
            'remove_emails': False
        }
    
    def clean(self, text: str) -> str:
        """
        Main cleaning function - applies all enabled cleaning operations.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        cleaned = text
        
        # Apply cleaning operations based on config
        if self.config.get('fix_encoding', True):
            cleaned = self._fix_encoding(cleaned)
        
        if self.config.get('normalize_unicode', True):
            cleaned = self._normalize_unicode(cleaned)
        
        if self.config.get('remove_urls', False):
            cleaned = self._remove_urls(cleaned)
        
        if self.config.get('remove_emails', False):
            cleaned = self._remove_emails(cleaned)
        
        if self.config.get('remove_extra_whitespace', True):
            cleaned = self._remove_extra_whitespace(cleaned)
        
        return cleaned.strip()
    
    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Handle common encoding problems
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...',# Ellipsis
            '\xa0': ' ',    # Non-breaking space
            '\r\n': ' ',    # Windows newline
            '\r': ' ',      # Old Mac newline
            '\n': ' ',      # Unix newline
            '\t': ' ',      # Tab
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to standard form."""
        # NFKC normalization: compatibility decomposition, followed by canonical composition
        return unicodedata.normalize('NFKC', text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        # Pattern matches http://, https://, www., and basic domains
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        
        # Also remove www. patterns
        www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+'
        text = re.sub(www_pattern, '', text)
        
        return text
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace while preserving single spaces.
        Handles multiple spaces, tabs, newlines.
        """
        # Replace multiple whitespace characters with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Ensure space after punctuation (if not already there)
        text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
        
        return text.strip()
    
    def clean_batch(self, texts: list) -> list:
        """
        Clean multiple texts at once.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]


# Convenience function for quick usage
def clean_text(text: str, config: Dict[str, Any] = None) -> str:
    """
    Quick function to clean a single text.
    
    Args:
        text: Raw text to clean
        config: Optional cleaning configuration
        
    Returns:
        Cleaned text
    """
    cleaner = TextCleaner(config)
    return cleaner.clean(text)


# Example usage and testing
# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "This   has    extra     spaces.",
        "This\nhas\nmultiple\nnewlines.",
        'This has "smart quotes" and –dashes–.',
        "Remove this url: https://example.com and email: test@example.com",
        "Normal text with proper spacing. Nothing to clean here!",
    ]
    
    print("Testing TextCleaner:\n")
    cleaner = TextCleaner()
    
    for i, text in enumerate(test_texts, 1):
        cleaned = cleaner.clean(text)
        print(f"Test {i}:")
        print(f"  Original: {repr(text)}")
        print(f"  Cleaned:  {repr(cleaned)}\n")