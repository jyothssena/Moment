"""
CSV Input Adapter
Reads data from CSV files.
"""

import pandas as pd # type: ignore
from typing import Iterator, Dict, Any
from .base_adapter import BaseInputAdapter


class CSVInputAdapter(BaseInputAdapter):
    """
    Input adapter for CSV files.
    Reads user_interpretations.csv and yields one interpretation at a time.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize CSV input adapter.
        
        Args:
            csv_path: Path to CSV file
        """
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
    
    def read(self) -> Iterator[Dict[str, Any]]:
        """
        Read CSV and yield one row at a time.
        
        Yields:
            Dictionary with row data + row_index
        """
        for idx, row in self.data.iterrows():
            record = row.to_dict()
            record['row_index'] = idx  # Add row index for book mapping
            yield record
    
    def read_all(self) -> list:
        """
        Read entire CSV at once.
        
        Returns:
            List of dictionaries with all rows
        """
        return list(self.read())
    
    def get_total_rows(self) -> int:
        """Get total number of rows in CSV."""
        return len(self.data)


# Example usage and testing
if __name__ == "__main__":
    print("Testing CSVInputAdapter:\n")
    print("="*70)
    
    adapter = CSVInputAdapter('data/raw/user_interpretations.csv')
    
    print(f"Total rows: {adapter.get_total_rows()}\n")
    
    print("First 3 records:")
    for i, record in enumerate(adapter.read()):
        if i >= 3:
            break
        print(f"\nRecord {i}:")
        print(f"  Row index: {record['row_index']}")
        print(f"  Character: {record['character_name']}")
        print(f"  Passage ID: {record['passage_id']}")
        print(f"  Interpretation: {record['interpretation_text'][:60]}...")
    
    print("\n" + "="*70)