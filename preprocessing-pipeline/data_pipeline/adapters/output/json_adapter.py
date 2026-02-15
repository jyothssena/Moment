"""
JSON Output Adapter
Writes data to JSON files.
"""

import json
from typing import Dict, Any, List
from pathlib import Path
from .base_adapter import BaseOutputAdapter


class JSONOutputAdapter(BaseOutputAdapter):
    """
    Output adapter for JSON files.
    Accumulates records and writes to file at the end.
    """
    
    def __init__(self, output_path: str, pretty_print: bool = True):
        """
        Initialize JSON output adapter.
        
        Args:
            output_path: Path to output JSON file
            pretty_print: Whether to format JSON with indentation
        """
        self.output_path = output_path
        self.pretty_print = pretty_print
        self.records = []
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    def write(self, data: Dict[str, Any]) -> None:
        """
        Add a single record to buffer.
        
        Args:
            data: Dictionary with record data
        """
        self.records.append(data)
    
    def write_batch(self, data: List[Dict[str, Any]]) -> None:
        """
        Add multiple records to buffer.
        
        Args:
            data: List of dictionaries with record data
        """
        self.records.extend(data)
    
    def finalize(self) -> None:
        """
        Write all buffered records to JSON file.
        """
        with open(self.output_path, 'w', encoding='utf-8') as f:
            if self.pretty_print:
                json.dump(self.records, f, indent=2, ensure_ascii=False)
            else:
                json.dump(self.records, f, ensure_ascii=False)
        
        print(f"✅ Wrote {len(self.records)} records to {self.output_path}")
    
    def get_record_count(self) -> int:
        """Get number of buffered records."""
        return len(self.records)


# Example usage and testing
if __name__ == "__main__":
    print("Testing JSONOutputAdapter:\n")
    print("="*70)
    
    # Create test output adapter
    adapter = JSONOutputAdapter('data/processed/test_output.json')
    
    # Write some test records
    test_records = [
        {
            'interpretation_id': 'moment_test1',
            'user_id': 'user_test',
            'book_id': 'gutenberg_84',
            'cleaned_text': 'This is a test interpretation.'
        },
        {
            'interpretation_id': 'moment_test2',
            'user_id': 'user_test',
            'book_id': 'gutenberg_1342',
            'cleaned_text': 'This is another test interpretation.'
        }
    ]
    
    print("Writing records...")
    adapter.write_batch(test_records)
    
    print(f"Buffered records: {adapter.get_record_count()}")
    
    print("\nFinalizing...")
    adapter.finalize()
    
    print("\n" + "="*70)