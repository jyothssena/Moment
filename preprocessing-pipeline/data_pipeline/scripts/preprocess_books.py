"""
Preprocess Books (Passages)
Main entry point for preprocessing book passages.
Called by Airflow/DVC pipeline.
"""

import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any

from ..adapters.output.json_adapter import JSONOutputAdapter
from ..preprocessing.preprocessor import Preprocessor
from ..utils.id_generator import IDGenerator
import pandas as pd # type: ignore


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def preprocess_books(input_dir: str, output_dir: str) -> None:
    """
    Preprocess book passages.
    
    This function is called by Airflow with specific input/output directories.
    
    Args:
        input_dir: Directory containing input CSV files (data/raw)
        output_dir: Directory for output JSON files (data/processed)
    """
    print("="*70)
    print("MOMENT PREPROCESSING - BOOK PASSAGES")
    print("="*70)
    
    # Load configuration
    config = load_config()
    
    # Setup paths
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    
    passages_path = input_dir_path / "passages.csv"
    output_path = output_dir_path / "books_processed.json"
    
    print(f"\nInput file:")
    print(f"  - {passages_path}")
    print(f"\nOutput file:")
    print(f"  - {output_path}\n")
    
    # Read passages
    passages_df = pd.read_csv(passages_path)
    
    print("Initializing components...")
    
    # Preprocessor
    preprocessor = Preprocessor(config['preprocessing'])
    
    # ID generator
    id_generator = IDGenerator(config['id_generation'])
    
    # Output adapter
    output_adapter = JSONOutputAdapter(str(output_path), pretty_print=True)
    
    print(f"✅ Initialized. Processing {len(passages_df)} passages...\n")
    
    # Get book mapping from config
    row_ranges = config['book_lookup']['row_ranges']
    
    # Process each passage
    processed_count = 0
    
    for idx, row in passages_df.iterrows():
        passage_id = row['passage_id']
        book_title = row['book_title']
        passage_text = row['passage_text']
        
        # Determine actual book from passages.csv book_title
        actual_book = None
        
        if book_title == "Unknown":
            # Frankenstein passages
            actual_book = row_ranges[0]
        elif "PRIDE" in book_title.upper():
            # Pride & Prejudice
            actual_book = row_ranges[1]
        elif "GATSBY" in book_title.upper():
            # The Great Gatsby
            actual_book = row_ranges[2]
        
        if not actual_book:
            print(f"⚠️  Warning: Could not determine book for passage {idx}")
            continue
        
        # Preprocess passage text
        preprocessing_result = preprocessor.process(passage_text)
        
        # Generate IDs
        book_id = id_generator.generate_book_id(gutenberg_id=actual_book['gutenberg_id'])
        passage_id_full = id_generator.generate_passage_id(book_id, passage_id)
        
        # Compile output
        # Compile output
        output_record = {
            'passage_id': passage_id_full,
            'book_id': book_id,
            'book_title': actual_book['book_title'],
            'book_author': actual_book['book_author'],
            'chapter_number': row.get('chapter_number', 'Unknown'),
            'passage_number': passage_id,
            'passage_title': row.get('passage_title', ''),
            'cleaned_passage_text': preprocessing_result['cleaned_text'],  # Only cleaned, no original
            'is_valid': preprocessing_result['is_valid'],
            'quality_score': preprocessing_result['quality_score'],
            'quality_issues': preprocessing_result['quality_issues'],
            'detected_issues': preprocessing_result['detected_issues'],
            'metrics': preprocessing_result['metrics'],
            'num_interpretations': None,  # Will be calculated later
            'timestamp': preprocessing_result['timestamp']
        }
        
        # Write to output
        output_adapter.write(output_record)
        processed_count += 1
    
    # Finalize output
    print(f"\n Processed {processed_count} passages")
    output_adapter.finalize()
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)


# Allow running directly for testing
if __name__ == "__main__":
    # Use default paths from config
    preprocess_books("data/raw", "data/processed")