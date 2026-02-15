"""
CSV Lookup Module
Looks up book and user data from CSV files.
"""

import pandas as pd # type: ignore
from typing import Dict, Any, Optional, List
from .base_lookup import BaseLookup


class CSVLookup(BaseLookup):
    """
    Lookup implementation for CSV files.
    Works for both books (passages.csv) and users (characters.csv).
    """
    
    def __init__(self, csv_path: str, key_column: str):
        """
        Initialize CSV lookup.
        
        Args:
            csv_path: Path to CSV file
            key_column: Column name to use as lookup key
        """
        self.csv_path = csv_path
        self.key_column = key_column
        self.data = pd.read_csv(csv_path)
    
    def lookup(self, key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Look up a single row by key.
        
        Args:
            key: Value to search for in key_column
            **kwargs: Additional filter criteria (e.g., passage_id=1)
            
        Returns:
            Dictionary with row data or None if not found
        """
        # Start with key column filter
        mask = self.data[self.key_column] == key
        
        # Apply additional filters
        for col, val in kwargs.items():
            if col in self.data.columns:
                mask &= (self.data[col] == val)
        
        # Get matching rows
        result = self.data[mask]
        
        if len(result) == 0:
            return None
        
        # Return first match as dictionary
        return result.iloc[0].to_dict()
    
    def lookup_batch(self, keys: List[str], **kwargs) -> List[Optional[Dict[str, Any]]]:
        """
        Look up multiple rows by keys.
        
        Args:
            keys: List of values to search for
            **kwargs: Additional filter criteria
            
        Returns:
            List of dictionaries (None for keys not found)
        """
        return [self.lookup(key, **kwargs) for key in keys]
    
    def lookup_by_row_index(self, row_index: int) -> Optional[Dict[str, Any]]:
        """
        Look up by row index (useful for row-based book mapping).
        
        Args:
            row_index: Index of the row (0-based)
            
        Returns:
            Dictionary with row data or None if index out of bounds
        """
        if row_index < 0 or row_index >= len(self.data):
            return None
        
        return self.data.iloc[row_index].to_dict()


class BookLookup:
    """
    Specialized lookup for books using passages.csv with row-based mapping.
    """
    
    def __init__(self, passages_csv_path: str, config: Dict[str, Any]):
        """
        Initialize book lookup.
        
        Args:
            passages_csv_path: Path to passages.csv
            config: Configuration with row_ranges for book mapping
        """
        self.passages = pd.read_csv(passages_csv_path)
        self.config = config
        self.row_ranges = config.get('row_ranges', [])
    
    def get_book_by_row_index(self, row_index: int, passage_id: int) -> Optional[Dict[str, Any]]:
        """
        Get book info using row-based mapping strategy.
        
        Args:
            row_index: Row index from user_interpretations.csv
            passage_id: Passage ID (1, 2, or 3)
            
        Returns:
            Dictionary with book info
        """
        # Determine which book based on row_index
        book_info = None
        for book in self.row_ranges:
            if book['row_start'] <= row_index <= book['row_end']:
                book_info = book
                break
        
        if not book_info:
            return None
        
        # Look up passage in passages.csv
        book_title = book_info['book_title']
        
        # Create mask for finding the right passage
        mask = (self.passages['passage_id'] == passage_id)
        
        # Match book based on title
        if book_title == "Frankenstein":
            # Frankenstein passages have book_title="Unknown" in CSV
            mask &= (self.passages['book_title'] == 'Unknown')
        elif book_title == "Pride and Prejudice":
            # Match "PRIDE & PREJUDICE" in CSV
            mask &= (self.passages['book_title'].str.upper().str.contains('PRIDE', na=False))
        elif book_title == "The Great Gatsby":
            # Match "The Great Gatsby" in CSV
            mask &= (self.passages['book_title'].str.contains('Gatsby', case=False, na=False))
        else:
            # Generic match
            mask &= (self.passages['book_title'].str.contains(book_title, case=False, na=False))
        
        result = self.passages[mask]
        
        if len(result) == 0:
            return None
        
        passage = result.iloc[0]
        
        return {
            'book_title': book_info['book_title'],
            'book_author': book_info['book_author'],
            'book_id': f"gutenberg_{book_info['gutenberg_id']}",
            'passage_id': f"gutenberg_{book_info['gutenberg_id']}_passage_{passage_id}",
            'passage_text': passage['passage_text'],
            'chapter_number': passage.get('chapter_number', 'Unknown'),
            'passage_number': passage_id
        }


class UserLookup:
    """
    Specialized lookup for users using characters.csv.
    """
    
    def __init__(self, characters_csv_path: str):
        """
        Initialize user lookup.
        
        Args:
            characters_csv_path: Path to characters.csv
        """
        self.characters = pd.read_csv(characters_csv_path)
    
    def get_user_by_name(self, character_name: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile by character name.
        
        Args:
            character_name: Name of the character (e.g., "Emma Chen")
            
        Returns:
            Dictionary with user profile or None if not found
        """
        mask = self.characters['Name'] == character_name
        result = self.characters[mask]
        
        if len(result) == 0:
            return None
        
        return result.iloc[0].to_dict()


# Example usage and testing
if __name__ == "__main__":
    print("Testing CSV Lookup:\n")
    print("="*70)
    
    # Test BookLookup
    print("Testing BookLookup:")
    book_config = {
        'row_ranges': [
            {
                'book_title': 'Frankenstein',
                'book_author': 'Mary Shelley',
                'gutenberg_id': '84',
                'row_start': 0,
                'row_end': 149
            },
            {
                'book_title': 'Pride and Prejudice',
                'book_author': 'Jane Austen',
                'gutenberg_id': '1342',
                'row_start': 150,
                'row_end': 299
            },
            {
                'book_title': 'The Great Gatsby',
                'book_author': 'F. Scott Fitzgerald',
                'gutenberg_id': '64317',
                'row_start': 300,
                'row_end': 449
            }
        ]
    }
    
    book_lookup = BookLookup('data/raw/passages.csv', book_config)
    
    # Test row 0 (Frankenstein, passage 1)
    result = book_lookup.get_book_by_row_index(0, 1)
    if result:
        print(f"\nRow 0, Passage 1:")
        print(f"  Book: {result['book_title']} by {result['book_author']}")
        print(f"  Book ID: {result['book_id']}")
        print(f"  Passage ID: {result['passage_id']}")
        print(f"  Passage Text: {result['passage_text'][:80]}...")
    
    # Test row 150 (Pride & Prejudice, passage 1)
    result = book_lookup.get_book_by_row_index(150, 1)
    if result:
        print(f"\nRow 150, Passage 1:")
        print(f"  Book: {result['book_title']} by {result['book_author']}")
        print(f"  Book ID: {result['book_id']}")
        print(f"  Passage ID: {result['passage_id']}")
    
    # Test row 300 (Gatsby, passage 1)
    result = book_lookup.get_book_by_row_index(300, 1)
    if result:
        print(f"\nRow 300, Passage 1:")
        print(f"  Book: {result['book_title']} by {result['book_author']}")
        print(f"  Book ID: {result['book_id']}")
        print(f"  Passage ID: {result['passage_id']}")
    
    print("\n" + "="*70)
    
    # Test UserLookup
    print("\nTesting UserLookup:")
    user_lookup = UserLookup('data/raw/characters.csv')
    
    user = user_lookup.get_user_by_name('Emma Chen')
    if user:
        print(f"\nCharacter: {user['Name']}")
        print(f"  Age: {user['Age']}")
        print(f"  Profession: {user['Profession']}")
        print(f"  Gender: {user['Gender']}")
    
    print("\n" + "="*70)