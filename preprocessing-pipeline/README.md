## Architecture

The pipeline follows a modular design that separates data sources from processing logic, making it easy to swap CSV files for API calls or database queries in production.

1. Core Processing (Universal)
2. Input Adapters (Swappable: CSV -> API -> Database)
3. Lookup Modules (Swappable: CSV -> Gutenberg API -> Database)
4. Output Adapters (Swappable: JSON Files -> Database)

### Key Design Principles

1. **Modularity**: Each component (cleaning, validation, issue detection) is independent
2. **Swappable Adapters**: Input and output methods can be changed without modifying core logic
3. **Configuration-Driven**: All settings controlled via YAML config file
4. **Airflow-Compatible**: Main entry points follow Airflow DAG conventions
5. **DVC-Ready**: Functions accept input_dir and output_dir parameters for data versioning

## Project Structure
```
MOMENT-PREPROCESSING/
├── data/
│   ├── raw/                           # Input CSV files
│   │   ├── user_interpretations.csv   # 450 user interpretations
│   │   ├── passages.csv               # 9 literary passages
│   │   └── characters.csv             # 50 character profiles
│   ├── processed/                     # Preprocessed outputs
│   │   ├── moments_processed.json     # Clean interpretations
│   │   └── books_processed.json       # Clean passages
│   └── validation/                    # Quality reports (future)
│
├── data_pipeline/
│   ├── preprocessing/                 # Core preprocessing modules
│   │   ├── text_cleaner.py           # Text cleaning operations
│   │   ├── text_validator.py         # Quality validation
│   │   ├── issue_detector.py         # PII, profanity, spam detection
│   │   ├── metrics_calculator.py     # Readability & text metrics
│   │   └── preprocessor.py           # Orchestrator
│   │
│   ├── adapters/                      # Input/output adapters
│   │   ├── input/
│   │   │   ├── base_adapter.py       # Input interface
│   │   │   └── csv_adapter.py        # CSV reader
│   │   └── output/
│   │       ├── base_adapter.py       # Output interface
│   │       └── json_adapter.py       # JSON writer
│   │
│   ├── lookup/                        # Data lookup modules
│   │   ├── base_lookup.py            # Lookup interface
│   │   └── csv_lookup.py             # Book & user lookup
│   │
│   ├── utils/                         # Utilities
│   │   └── id_generator.py           # ID generation
│   │
│   └── scripts/                       # Airflow entry points
│       ├── preprocess_books.py       # Process passages
│       └── preprocess_moments.py     # Process interpretations
│
├── config/
│   └── config.yaml                    # Configuration file
│
├── tests/                             # Unit tests (future)
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone 
cd MOMENT-PREPROCESSING
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

4. Verify installation:
```bash
python -m data_pipeline.preprocessing.text_cleaner
```

## Usage

### Quick Start

Process all data with default settings:
```bash
# Preprocess user interpretations
python -m data_pipeline.scripts.preprocess_moments

# Preprocess book passages
python -m data_pipeline.scripts.preprocess_books
```

### Airflow Integration

The pipeline is designed to be called by Airflow DAGs:
```python
from data_pipeline.scripts.preprocess_books import preprocess_books
from data_pipeline.scripts.preprocess_moments import preprocess_moments

# In our Airflow DAG
preprocess_books_task = PythonOperator(
    task_id='preprocess_books',
    python_callable=preprocess_books,
    op_kwargs={
        'input_dir': 'data/raw',
        'output_dir': 'data/processed'
    }
)

preprocess_moments_task = PythonOperator(
    task_id='preprocess_moments',
    python_callable=preprocess_moments,
    op_kwargs={
        'input_dir': 'data/raw',
        'output_dir': 'data/processed'
    }
)
```

### Configuration

Edit `config/config.yaml` to customize preprocessing behavior:
```yaml
preprocessing:
  text_cleaning:
    remove_extra_whitespace: true
    normalize_unicode: true
    fix_encoding: true
  
  validation:
    min_words: 10
    max_words: 500
    quality_threshold: 0.5
  
  issue_detection:
    check_pii: true
    check_profanity: true
    check_spam: true
```

## Data Flow

### Input Data

**user_interpretations.csv** (450 rows):
- character_name: Name of the reader
- passage_id: Reference to passage (1, 2, or 3)
- interpretation_text: User's written interpretation

**passages.csv** (9 rows):
- passage_id: Passage identifier
- book_title: Title of the book
- passage_text: Actual passage text from literature

**characters.csv** (50 rows):
- Name: Character profile name
- Age, Profession, Gender: Demographics
- Reading preferences and style

### Preprocessing Steps

1. **Extraction**: Read CSV files, lookup book and user context
2. **Cleaning**: Remove extra whitespace, fix encoding, normalize text
3. **Validation**: Check length, language, coherence, quality score
4. **Issue Detection**: Scan for PII, profanity, spam patterns
5. **Metrics Calculation**: Count words, sentences, calculate readability
6. **ID Generation**: Create unique identifiers for all entities
7. **Output**: Write structured JSON with all metadata

### Output Format

**moments_processed.json** (450 records):
```json
{
  "interpretation_id": "moment_abc123",
  "user_id": "user_emma_chen_xyz",
  "book_id": "gutenberg_84",
  "passage_id": "gutenberg_84_passage_1",
  "book_title": "Frankenstein",
  "book_author": "Mary Shelley",
  "passage_number": 1,
  "character_name": "Emma Chen",
  "cleaned_interpretation": "He says catastrophe before...",
  "is_valid": true,
  "quality_score": 0.88,
  "quality_issues": [],
  "detected_issues": {
    "has_pii": false,
    "has_profanity": false,
    "is_spam": false
  },
  "metrics": {
    "word_count": 99,
    "char_count": 438,
    "readability_score": 66.72
  },
  "timestamp": "2026-02-15T10:30:00"
}
```

**books_processed.json** (9 records):
```json
{
  "passage_id": "gutenberg_84_passage_1",
  "book_id": "gutenberg_84",
  "book_title": "Frankenstein",
  "book_author": "Mary Shelley",
  "chapter_number": "Chapter 5",
  "passage_number": 1,
  "cleaned_passage_text": "It was on a dreary night...",
  "is_valid": true,
  "quality_score": 0.95,
  "metrics": { ... },
  "timestamp": "2026-02-15T10:30:00"
}
```

## Preprocessing Details

### Text Cleaning

- Removes extra whitespace and newlines
- Fixes common encoding issues (smart quotes, dashes)
- Normalizes Unicode characters
- Removes URLs and emails (optional)

### Text Validation

- Minimum length: 10 words, 50 characters
- Maximum length: 500 words, 3000 characters
- Language detection: English only
- Gibberish detection: vowel/consonant ratio
- Character diversity check: flags repetitive text
- Quality score: 0.0 to 1.0 (threshold: 0.5)

### Issue Detection

**PII (Personal Identifiable Information)**:
- Email addresses
- Phone numbers (US formats)
- Social Security Numbers
- Potential credit card numbers

**Profanity Detection**:
- Configurable word list
- Ratio-based flagging (threshold: 30%)

**Spam Detection**:
- Excessive capitalization (>50%)
- Excessive punctuation (>10 marks)
- Repetitive characters (4+ in a row)
- Repetitive words (>30% frequency)
- Common spam phrases

### Metrics Calculation

- Character count (excluding whitespace)
- Word count
- Sentence count
- Average word length
- Average sentence length
- Flesch Reading Ease score (0-100, higher = easier)

### ID Generation

All IDs are deterministic (same input produces same ID):

- **user_id**: `user_{sanitized_name}_{hash}`
- **book_id**: `gutenberg_{gutenberg_id}`
- **passage_id**: `{book_id}_passage_{number}`
- **interpretation_id**: `moment_{hash(user+passage+text)}`

## Book Mapping Strategy

### Current Implementation (CSV-based)

Uses row-based mapping for Assignment 1:
- Rows 0-149: Frankenstein (Gutenberg ID: 84)
- Rows 150-299: Pride and Prejudice (Gutenberg ID: 1342)
- Rows 300-449: The Great Gatsby (Gutenberg ID: 64317)

### Future Implementation (API-based)

For production, the pipeline will:
1. Check local passages database for known passages
2. If not found, query Gutenberg API with passage text
3. API returns book metadata (title, author, ID)
4. Save new passage to database for future lookups

**No code changes required** - just swap the lookup strategy in config.yaml:
```yaml
book_lookup:
  strategy: "gutenberg_api"  # Changed from "csv"
```

## Extending the Pipeline

### Adding a New Input Source

1. Create new adapter in `data_pipeline/adapters/input/`:
```python
from .base_adapter import BaseInputAdapter

class APIInputAdapter(BaseInputAdapter):
    def read(self):
        # Fetch from API
        # Yield records one at a time
        pass
```

2. Update scripts to use new adapter:
```python
from ..adapters.input.api_adapter import APIInputAdapter

input_adapter = APIInputAdapter(api_url)
```

### Adding a New Output Format

1. Create new adapter in `data_pipeline/adapters/output/`:
```python
from .base_adapter import BaseOutputAdapter

class DatabaseAdapter(BaseOutputAdapter):
    def write(self, data):
        # Insert into database
        pass
```

2. Update scripts to use new adapter:
```python
from ..adapters.output.db_adapter import DatabaseAdapter

output_adapter = DatabaseAdapter(db_connection)
```

### Adding a New Preprocessing Step

1. Create new module in `data_pipeline/preprocessing/`:
```python
class SentimentAnalyzer:
    def analyze(self, text):
        # Return sentiment score
        pass
```

2. Import and use in `preprocessor.py`:
```python
from .sentiment_analyzer import SentimentAnalyzer

sentiment = self.sentiment_analyzer.analyze(cleaned_text)
result['sentiment'] = sentiment
```

## Testing

Run individual module tests:
```bash
# Test text cleaner
python -m data_pipeline.preprocessing.text_cleaner

# Test text validator
python -m data_pipeline.preprocessing.text_validator

# Test issue detector
python -m data_pipeline.preprocessing.issue_detector

# Test metrics calculator
python -m data_pipeline.preprocessing.metrics_calculator

# Test ID generator
python -m data_pipeline.utils.id_generator

# Test adapters
python -m data_pipeline.adapters.input.csv_adapter
python -m data_pipeline.adapters.output.json_adapter

# Test lookup modules
python -m data_pipeline.lookup.csv_lookup
```

Run full pipeline tests:
```bash
# Process test data
python -m data_pipeline.scripts.preprocess_moments
python -m data_pipeline.scripts.preprocess_books
```

## Performance

Current benchmarks on MacBook Pro M1:
- Preprocessing speed: ~150 interpretations per second
- Memory usage: <100MB for 450 records
- Output file sizes: 
  - moments_processed.json: ~800KB
  - books_processed.json: ~50KB

## Troubleshooting

**Import errors with relative imports:**
```bash
# Use python -m instead of python
python -m data_pipeline.scripts.preprocess_moments
```

**NLTK data not found:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Config file not found:**
Ensure you're running from project root directory where config/ exists.

**Book matching fails:**
Check that passages.csv contains correct book_title values matching config.yaml.

## Contributing

When adding new features:
1. Follow the existing modular architecture
2. Create interfaces (base classes) for swappable components
3. Add configuration options to config.yaml
4. Include test code in `if __name__ == "__main__"` blocks
5. Update this README with new functionality

## Dependencies

Core dependencies:
- pandas: Data manipulation
- numpy: Numerical operations
- nltk: Text processing
- textstat: Readability scores
- langdetect: Language detection
- pyyaml: Configuration management

See `requirements.txt` for complete list with versions.

## License

This project is part of the MOMENT platform developed for IE7374 (MLOps) coursework at Northeastern University

## Authors

- MOMENT Team (Group 23)

## Acknowledgments

- Literary passages from Project Gutenberg
- Synthesized interpretation data for ML research
- Airflow integration guidance from course staff
