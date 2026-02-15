"""
Preprocess Moments (User Interpretations)
Main entry point for preprocessing user interpretations.
Called by Airflow/DVC pipeline.
"""

import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any

from ..adapters.input.csv_adapter import CSVInputAdapter
from ..adapters.output.json_adapter import JSONOutputAdapter
from ..lookup.csv_lookup import BookLookup, UserLookup
from ..preprocessing.preprocessor import Preprocessor
from ..utils.id_generator import IDGenerator


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def preprocess_moments(input_dir: str, output_dir: str) -> None:
    """
    Preprocess user interpretations (moments).
    
    This function is called by Airflow with specific input/output directories.
    
    Args:
        input_dir: Directory containing input CSV files (data/raw)
        output_dir: Directory for output JSON files (data/processed)
    """
    print("="*70)
    print("MOMENT PREPROCESSING - USER INTERPRETATIONS")
    print("="*70)
    
    # Load configuration
    config = load_config()
    
    # Setup paths
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    
    user_interp_path = input_dir_path / "user_interpretations.csv"
    passages_path = input_dir_path / "passages.csv"
    characters_path = input_dir_path / "characters.csv"
    output_path = output_dir_path / "moments_processed.json"
    
    print(f"\nInput files:")
    print(f"  - {user_interp_path}")
    print(f"  - {passages_path}")
    print(f"  - {characters_path}")
    print(f"\nOutput file:")
    print(f"  - {output_path}\n")
    
    # Initialize components
    print("Initializing components...")
    
    # Input adapter
    input_adapter = CSVInputAdapter(str(user_interp_path))
    
    # Lookup modules
    book_lookup = BookLookup(str(passages_path), config['book_lookup'])
    user_lookup = UserLookup(str(characters_path))
    
    # Preprocessor
    preprocessor = Preprocessor(config['preprocessing'])
    
    # ID generator
    id_generator = IDGenerator(config['id_generation'])
    
    # Output adapter
    output_adapter = JSONOutputAdapter(str(output_path), pretty_print=True)
    
    print(f" Initialized. Processing {input_adapter.get_total_rows()} interpretations...\n")
    
    # Process each interpretation
    processed_count = 0
    valid_count = 0
    
    for record in input_adapter.read():
        # Extract fields
        row_index = record['row_index']
        character_name = record['character_name']
        passage_id = record['passage_id']
        interpretation_text = record['interpretation_text']
        
        # Lookup book info
        book_info = book_lookup.get_book_by_row_index(row_index, passage_id)
        
        if not book_info:
            print(f"⚠️  Warning: Could not find book info for row {row_index}, passage {passage_id}")
            continue
        
        # Lookup user info (optional - for enrichment)
        user_info = user_lookup.get_user_by_name(character_name)
        
        # Preprocess the interpretation text
        preprocessing_result = preprocessor.process(interpretation_text)
        
        # Generate IDs
        user_id = id_generator.generate_user_id(character_name)
        interpretation_id = id_generator.generate_interpretation_id(
            user_id, 
            book_info['passage_id'], 
            preprocessing_result['cleaned_text']
        )
        
        # Compile final output
        # Compile final output
        output_record = {
            'interpretation_id': interpretation_id,
            'user_id': user_id,
            'book_id': book_info['book_id'],
            'passage_id': book_info['passage_id'],
            'book_title': book_info['book_title'],
            'book_author': book_info['book_author'],
            'passage_number': book_info['passage_number'],
            'character_name': character_name,
            'cleaned_interpretation': preprocessing_result['cleaned_text'],  # Only cleaned, no original
            'is_valid': preprocessing_result['is_valid'],
            'quality_score': preprocessing_result['quality_score'],
            'quality_issues': preprocessing_result['quality_issues'],
            'detected_issues': preprocessing_result['detected_issues'],
            'metrics': preprocessing_result['metrics'],
            'timestamp': preprocessing_result['timestamp']
        }
        
        # Write to output
        output_adapter.write(output_record)
        
        processed_count += 1
        if preprocessing_result['is_valid']:
            valid_count += 1
        
        # Progress indicator
        if processed_count % 50 == 0:
            print(f"  Processed {processed_count}/{input_adapter.get_total_rows()} interpretations...")
    
    # Finalize output
    print(f"\n Processed {processed_count} interpretations")
    print(f"   Valid: {valid_count} ({valid_count/processed_count*100:.1f}%)")
    print(f"   Invalid: {processed_count - valid_count}")
    
    output_adapter.finalize()
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)


# Allow running directly for testing
if __name__ == "__main__":
    # Use default paths from config
    preprocess_moments("data/raw", "data/processed")