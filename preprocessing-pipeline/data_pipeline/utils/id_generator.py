"""
ID Generator Module
Generates unique IDs for users, books, passages, and interpretations.
"""

import hashlib
import re
from typing import Dict, Any


class IDGenerator:
    """
    Universal ID generator for the MOMENT pipeline.
    Generates deterministic, unique IDs.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize IDGenerator with configuration.
        
        Args:
            config: Dictionary with ID generation settings from config.yaml
        """
        self.config = config or {
            'user_id_prefix': 'user',
            'book_id_prefix': 'gutenberg',
            'passage_id_prefix': 'passage',
            'interpretation_id_prefix': 'moment',
            'hash_length': 12
        }
    
    def generate_user_id(self, character_name: str) -> str:
        """
        Generate user_id from character name.
        
        Args:
            character_name: Name of the character (e.g., "Emma Chen")
            
        Returns:
            user_id (e.g., "user_emma_chen_a1b2c3d4")
        """
        prefix = self.config.get('user_id_prefix', 'user')
        hash_length = self.config.get('hash_length', 12)
        
        # Sanitize name: lowercase, replace spaces with underscores
        sanitized = self._sanitize_string(character_name)
        
        # Generate hash from original name
        hash_suffix = self._generate_hash(character_name, hash_length)
        
        return f"{prefix}_{sanitized}_{hash_suffix}"
    
    def generate_book_id(self, book_title: str = None, gutenberg_id: str = None) -> str:
        """
        Generate book_id.
        
        Args:
            book_title: Title of the book (e.g., "Frankenstein")
            gutenberg_id: Gutenberg ID if known (e.g., "84")
            
        Returns:
            book_id (e.g., "gutenberg_84" or "book_frankenstein_a1b2c3d4")
        """
        prefix = self.config.get('book_id_prefix', 'gutenberg')
        
        # If gutenberg_id provided, use it directly
        if gutenberg_id:
            return f"{prefix}_{gutenberg_id}"
        
        # Otherwise, generate from book title
        if book_title:
            hash_length = self.config.get('hash_length', 12)
            sanitized = self._sanitize_string(book_title)
            hash_suffix = self._generate_hash(book_title, hash_length)
            return f"book_{sanitized}_{hash_suffix}"
        
        raise ValueError("Either book_title or gutenberg_id must be provided")
    
    def generate_passage_id(self, book_id: str, passage_number: int) -> str:
        """
        Generate passage_id.
        
        Args:
            book_id: ID of the book (e.g., "gutenberg_84")
            passage_number: Passage number (1, 2, 3, etc.)
            
        Returns:
            passage_id (e.g., "gutenberg_84_passage_1")
        """
        return f"{book_id}_passage_{passage_number}"
    
    def generate_interpretation_id(self, user_id: str, passage_id: str, 
                                   interpretation_text: str) -> str:
        """
        Generate interpretation_id (unique for each interpretation).
        
        Args:
            user_id: ID of the user
            passage_id: ID of the passage
            interpretation_text: The interpretation text (for uniqueness)
            
        Returns:
            interpretation_id (e.g., "moment_a1b2c3d4e5f6")
        """
        prefix = self.config.get('interpretation_id_prefix', 'moment')
        hash_length = self.config.get('hash_length', 12)
        
        # Combine all inputs for unique hash
        combined = f"{user_id}_{passage_id}_{interpretation_text[:100]}"
        hash_suffix = self._generate_hash(combined, hash_length)
        
        return f"{prefix}_{hash_suffix}"
    
    def _sanitize_string(self, text: str) -> str:
        """
        Sanitize string for use in IDs.
        - Lowercase
        - Replace spaces with underscores
        - Remove special characters
        - Limit length to 30 chars
        """
        # Lowercase
        text = text.lower()
        
        # Replace spaces and hyphens with underscores
        text = text.replace(' ', '_').replace('-', '_')
        
        # Remove special characters, keep only alphanumeric and underscores
        text = re.sub(r'[^a-z0-9_]', '', text)
        
        # Remove multiple consecutive underscores
        text = re.sub(r'_+', '_', text)
        
        # Strip leading/trailing underscores
        text = text.strip('_')
        
        # Limit length
        text = text[:30]
        
        return text
    
    def _generate_hash(self, text: str, length: int = 12) -> str:
        """
        Generate a deterministic hash from text.
        
        Args:
            text: Input text to hash
            length: Length of hash to return
            
        Returns:
            Hexadecimal hash string
        """
        # Use SHA256 for deterministic hashing
        hash_object = hashlib.sha256(text.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        
        # Return first N characters
        return hash_hex[:length]
    
    def generate_all_ids(self, character_name: str, book_title: str = None,
                        gutenberg_id: str = None, passage_number: int = None,
                        interpretation_text: str = None) -> Dict[str, str]:
        """
        Convenience method to generate all IDs at once.
        
        Args:
            character_name: Name of the character
            book_title: Title of the book
            gutenberg_id: Gutenberg ID if known
            passage_number: Passage number
            interpretation_text: Interpretation text
            
        Returns:
            Dictionary with all IDs
        """
        user_id = self.generate_user_id(character_name)
        book_id = self.generate_book_id(book_title, gutenberg_id)
        passage_id = self.generate_passage_id(book_id, passage_number)
        interpretation_id = None
        
        if interpretation_text:
            interpretation_id = self.generate_interpretation_id(
                user_id, passage_id, interpretation_text
            )
        
        return {
            'user_id': user_id,
            'book_id': book_id,
            'passage_id': passage_id,
            'interpretation_id': interpretation_id
        }


# Convenience functions for quick usage
def generate_user_id(character_name: str, config: Dict[str, Any] = None) -> str:
    """Quick function to generate user_id."""
    generator = IDGenerator(config)
    return generator.generate_user_id(character_name)


def generate_book_id(book_title: str = None, gutenberg_id: str = None, 
                     config: Dict[str, Any] = None) -> str:
    """Quick function to generate book_id."""
    generator = IDGenerator(config)
    return generator.generate_book_id(book_title, gutenberg_id)


def generate_passage_id(book_id: str, passage_number: int, 
                       config: Dict[str, Any] = None) -> str:
    """Quick function to generate passage_id."""
    generator = IDGenerator(config)
    return generator.generate_passage_id(book_id, passage_number)


def generate_interpretation_id(user_id: str, passage_id: str, 
                              interpretation_text: str,
                              config: Dict[str, Any] = None) -> str:
    """Quick function to generate interpretation_id."""
    generator = IDGenerator(config)
    return generator.generate_interpretation_id(user_id, passage_id, interpretation_text)


# Example usage and testing
if __name__ == "__main__":
    print("Testing IDGenerator:\n")
    print("="*70)
    
    generator = IDGenerator()
    
    # Test case: Emma Chen interpreting Frankenstein Passage 1
    character_name = "Emma Chen"
    book_title = "Frankenstein"
    gutenberg_id = "84"
    passage_number = 1
    interpretation_text = "He says catastrophe before anything bad happens..."
    
    print("Inputs:")
    print(f"  Character: {character_name}")
    print(f"  Book: {book_title} (Gutenberg ID: {gutenberg_id})")
    print(f"  Passage: {passage_number}")
    print(f"  Interpretation: {interpretation_text[:50]}...\n")
    
    # Generate individual IDs
    user_id = generator.generate_user_id(character_name)
    book_id = generator.generate_book_id(gutenberg_id=gutenberg_id)
    passage_id = generator.generate_passage_id(book_id, passage_number)
    interpretation_id = generator.generate_interpretation_id(
        user_id, passage_id, interpretation_text
    )
    
    print("Generated IDs:")
    print(f"  user_id: {user_id}")
    print(f"  book_id: {book_id}")
    print(f"  passage_id: {passage_id}")
    print(f"  interpretation_id: {interpretation_id}")
    
    print("\n" + "="*70)
    
    # Test generate_all_ids convenience method
    print("\nUsing generate_all_ids():")
    all_ids = generator.generate_all_ids(
        character_name=character_name,
        gutenberg_id=gutenberg_id,
        passage_number=passage_number,
        interpretation_text=interpretation_text
    )
    
    for key, value in all_ids.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)