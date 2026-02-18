# MOMENT Preprocessing Pipeline

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Input Data](#input-data)
- [Output Data](#output-data)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Running Tests](#running-tests)
- [Production Deployment](#production-deployment)
- [Module Reference](#module-reference)

---

## Overview

The preprocessing pipeline takes raw interpretation data and produces three clean, JSON datasets ready for embedding and matching:

- **450** reader interpretations → `moments_processed.json`
- **9** literary passages → `books_processed.json`
- **50** reader profiles → `users_processed.json`

### What the pipeline does

| Phase | What happens |
|-------|-------------|
| 1 | Reads raw JSON and CSV files via input adapters |
| 2 | Looks up book metadata from the Gutenberg API |
| 3 | Cleans text (smart quotes, encoding, whitespace) |
| 4 | Validates text quality (length, language, gibberish) |
| 5 | Detects issues (PII, profanity, spam) |
| 6 | Calculates metrics (readability, word count, sentence count) |
| 7 | Detects anomalies (outliers, duplicates, style mismatches) |
| 8 | Generates deterministic IDs for all records |
| 9 | Writes processed JSON files + validation report |

---

## Project Structure

```
pipeline_moments_preprocessing/
│
├── data/
│   ├── raw/                          # Input files (committed to git)
│   │   ├── all_interpretations_450_FINAL_NO_BIAS.json
│   │   ├── passages.csv
│   │   └── characters.csv
│   ├── processed/                    # Pipeline outputs (gitignored)
│   │   ├── moments_processed.json
│   │   ├── books_processed.json
│   │   └── users_processed.json
│   └── validation/                   # Quality reports (gitignored)
│       └── validation_report.json
│
├── data_pipeline/
│   ├── adapters/
│   │   ├── input/
│   │   │   ├── base_adapter.py       # Input interface (ABC)
│   │   │   └── json_csv_adapter.py   # Reads JSON + CSV files
│   │   └── output/
│   │       ├── base_adapter.py       # Output interface (ABC)
│   │       └── json_adapter.py       # Writes JSON files
│   │
│   ├── lookup/
│   │   ├── base_lookup.py            # Lookup interface (ABC)
│   │   └── gutenberg_lookup.py       # Gutenberg API + cache + fallback
│   │
│   ├── preprocessing/
│   │   ├── text_cleaner.py           # Encoding, quotes, whitespace
│   │   ├── text_validator.py         # Length, language, quality score
│   │   ├── issue_detector.py         # PII, profanity, spam
│   │   ├── metrics_calculator.py     # Readability, word count, etc.
│   │   ├── anomaly_detector.py       # Outliers, duplicates, mismatches
│   │   └── preprocessor.py           # Main orchestrator
│   │
│   └── utils/
│       └── id_generator.py           # Deterministic ID generation
│
├── scripts/
│   ├── preprocess_moments.py         # Full pipeline runner (use this)
│   ├── preprocess_books.py           # Books only
│   └── preprocess_users.py           # Users only
│
├── tests/
│   ├── conftest.py                   # Shared fixtures
│   ├── test_id_generator.py
│   ├── test_text_cleaner.py
│   ├── test_text_validator.py
│   ├── test_issue_detector.py
│   ├── test_metrics_calculator.py
│   └── test_anomaly_detector.py
│
├── config/
│   └── config.yaml                   # All pipeline settings
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone and set up

```bash
git clone <repository-url>
cd pipeline_moments_preprocessing

# create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# install dependencies
pip install -r requirements.txt

# download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Run the full pipeline

```bash
python -m scripts.preprocess_moments
```

### 3. Check outputs

```bash
ls data/processed/
# moments_processed.json
# books_processed.json
# users_processed.json

cat data/validation/validation_report.json
```

---

## Pipeline Architecture

### Design Principles

**Modularity** — Every component (cleaning, validation, detection) is independent and can be tested, replaced, or extended without touching anything else.

**Swappable Adapters** — Input and output sources are behind abstract interfaces. Switch from JSON files to a database by creating a new adapter — zero changes to core logic.

**Configuration-Driven** — Every threshold, path, and flag lives in `config/config.yaml`. No magic numbers in code. Production just gets a different config file.

**Airflow-Compatible** — All three entry point scripts accept `input_dir` and `output_dir` parameters and follow Airflow `PythonOperator` conventions.

**DVC-Ready** — All functions use explicit path parameters. Deterministic IDs ensure same input always produces identical output across runs.

**Graceful Degradation** — One bad record never crashes the pipeline. Every record is wrapped in try/except — failures are logged and skipped.

### How components connect

```
Raw Files          Input Adapter       Preprocessing Modules
─────────         ─────────────       ────────────────────────
JSON  ──────────► JsonCsvAdapter ────► TextCleaner
CSV   ──────────► JsonCsvAdapter ────► TextValidator
                                  ────► IssueDetector
Gutenberg API                     ────► MetricsCalculator
─────────────────► GutenbergLookup ───► AnomalyDetector
                                        │
                                        ▼
                   ID Generator ────► Preprocessor (orchestrator)
                                        │
                                        ▼
                   JsonOutputAdapter ◄── Processed Records
                        │
                        ▼
               moments_processed.json
               books_processed.json
               users_processed.json
               validation_report.json
```

---

## Input Data

| File | Records | Description |
|------|---------|-------------|
| `all_interpretations_450_FINAL_NO_BIAS.json` | 450 | Reader interpretations across 3 books × 3 passages × 50 characters |
| `passages.csv` | 9 | Literary passages (3 per book) from Frankenstein, Pride and Prejudice, The Great Gatsby |
| `characters.csv` | 50 | Reader profiles with demographics, reading habits, and experience levels |

### Books covered

| Book | Author | Gutenberg ID |
|------|--------|-------------|
| Frankenstein | Mary Shelley | 84 |
| Pride and Prejudice | Jane Austen | 1342 |
| The Great Gatsby | F. Scott Fitzgerald | 64317 |

---

## Output Data

### `moments_processed.json` (450 records)

One record per interpretation:

```json
{
  "interpretation_id": "moment_a1b2c3d4",
  "user_id": "user_emma_chen_a1b2c3d4",
  "book_id": "gutenberg_84",
  "passage_id": "gutenberg_84_passage_1",
  "book_title": "Frankenstein",
  "passage_number": 1,
  "character_id": 1,
  "character_name": "Emma Chen",
  "cleaned_interpretation": "He says \"catastrophe\" before anything bad happens...",
  "original_word_count": 99,
  "is_valid": true,
  "quality_score": 0.95,
  "quality_issues": [],
  "detected_issues": {
    "has_pii": false,
    "pii_types": [],
    "has_profanity": false,
    "profanity_ratio": 0.0,
    "is_spam": false,
    "spam_reasons": []
  },
  "anomalies": {
    "word_count_outlier": false,
    "readability_outlier": false,
    "duplicate_risk": false,
    "duplicate_of": null,
    "style_mismatch": false,
    "anomaly_details": []
  },
  "metrics": {
    "word_count": 99,
    "char_count": 452,
    "sentence_count": 8,
    "avg_word_length": 4.8,
    "avg_sentence_length": 12.4,
    "readability_score": 72.5
  },
  "timestamp": "2026-02-17T10:30:00"
}
```

### `books_processed.json` (9 records)

One record per literary passage:

```json
{
  "book_id": "gutenberg_84",
  "passage_id": "gutenberg_84_passage_1",
  "book_title": "Frankenstein",
  "book_author": "Mary Shelley",
  "chapter_number": "Unknown",
  "passage_title": "C",
  "passage_number": 1,
  "cleaned_passage_text": "It was on a dreary night of November...",
  "is_valid": true,
  "quality_score": 0.95,
  "quality_issues": [],
  "metrics": { ... },
  "timestamp": "2026-02-17T10:30:00"
}
```

### `users_processed.json` (50 records)

One record per reader profile:

```json
{
  "user_id": "user_emma_chen_a1b2c3d4",
  "character_name": "Emma Chen",
  "gender": "Female",
  "age": 28,
  "profession": "Data Scientist",
  "distribution_category": "DELIBERATE",
  "personality": "Analytical",
  "interest": "Psych/Phil",
  "reading_intensity": "Heavy",
  "reading_count": 35,
  "experience_level": "Some classics",
  "experience_count": 5,
  "journey": "Wanted answers about whether life has inherent meaning...",
  "reading_styles": ["Academic", "Text-focused", "Deep", "Extended"],
  "total_interpretations": 3,
  "books_interpreted": ["Frankenstein", "Pride and Prejudice", "The Great Gatsby"],
  "timestamp": "2026-02-17T10:30:00"
}
```

### `validation_report.json`

Pipeline quality summary:

```json
{
  "pipeline": "MOMENT Preprocessing Pipeline",
  "processing_start": "2026-02-17T10:30:00",
  "processing_end": "2026-02-17T10:30:45",
  "interpretations": {
    "total": 450,
    "valid": 447,
    "invalid": 3,
    "validity_rate": 99.33
  },
  "passages": { "total": 9, "valid": 9 },
  "users": { "total": 50 },
  "anomalies_detected": 12,
  "issues_detected": { "pii": 0, "profanity": 0, "spam": 2 }
}
```

---

## Configuration

All settings live in `config/config.yaml`. Key sections:

### Text Cleaning
```yaml
text_cleaning:
  remove_extra_whitespace: true
  fix_smart_quotes: true      # " " → " "
  fix_dashes: true            # — → --
  fix_encoding: true          # fixes mojibake
  remove_emails: true         # PII protection
  remove_urls: false          # keep URLs in interpretations
```

### Validation Thresholds
```yaml
validation:
  interpretations:
    min_words: 10
    max_words: 600
    quality_threshold: 0.5
  passages:
    min_words: 20
    max_words: 1000
    quality_threshold: 0.6
```

### Anomaly Detection
```yaml
anomaly_detection:
  word_count:
    method: "iqr"
    iqr_multiplier: 1.5
  readability:
    method: "zscore"
    zscore_threshold: 2.5
  duplicate:
    similarity_threshold: 0.85
```

### Gutenberg Lookup
```yaml
gutenberg:
  strategy: "api"       # "api" or "config"
  cache_results: true   # cache to avoid repeat API calls
  timeout_seconds: 10
```

---

## Running the Pipeline

### Full pipeline (recommended)
```bash
python -m scripts.preprocess_moments
```

### Individual processors
```bash
python -m scripts.preprocess_books
python -m scripts.preprocess_users
```

### With custom paths
```bash
python -m scripts.preprocess_moments \
  --input-dir data/raw \
  --output-dir data/processed
```

### Test individual modules
```bash
python -m data_pipeline.utils.id_generator
python -m data_pipeline.preprocessing.text_cleaner
python -m data_pipeline.preprocessing.text_validator
python -m data_pipeline.preprocessing.issue_detector
python -m data_pipeline.preprocessing.metrics_calculator
python -m data_pipeline.preprocessing.anomaly_detector
python -m data_pipeline.adapters.input.json_csv_adapter
python -m data_pipeline.adapters.output.json_adapter
python -m data_pipeline.lookup.gutenberg_lookup
```

---

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run a single test file
```bash
pytest tests/test_text_cleaner.py -v
```

### Run a single test class
```bash
pytest tests/test_text_cleaner.py::TestSmartQuotes -v
```

### Run with short traceback on failure
```bash
pytest tests/ -v --tb=short
```

### Test coverage summary
| File | Tests | Coverage |
|------|-------|----------|
| `test_id_generator.py` | 28 | ID generation, determinism, uniqueness |
| `test_text_cleaner.py` | 24 | Smart quotes, dashes, whitespace, emails |
| `test_text_validator.py` | 28 | Length, language, gibberish, quality scores |
| `test_issue_detector.py` | 22 | PII, profanity, spam patterns |
| `test_metrics_calculator.py` | 26 | Word count, readability, dataset stats |
| `test_anomaly_detector.py` | 30 | Outliers, duplicates, style mismatch |
| **Total** | **179** | All preprocessing modules |

---

## Production Deployment

### Switching to a database input
1. Create `data_pipeline/adapters/input/db_adapter.py`
2. Inherit from `BaseInputAdapter`
3. Implement `read_interpretations()`, `read_passages()`, `read_characters()`
4. Update script imports — zero other changes needed

### Switching to a database output
1. Create `data_pipeline/adapters/output/db_adapter.py`
2. Inherit from `BaseOutputAdapter`
3. Implement `write_moments()`, `write_books()`, `write_users()`, `write_validation_report()`
4. Update script imports — zero other changes needed

### Airflow DAG example
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from scripts.preprocess_books import preprocess_books
from scripts.preprocess_users import preprocess_users
from scripts.preprocess_moments import preprocess_moments

with DAG("moment_preprocessing", schedule_interval="@daily") as dag:

    books_task = PythonOperator(
        task_id="preprocess_books",
        python_callable=preprocess_books,
        op_kwargs={"input_dir": "data/raw", "output_dir": "data/processed"}
    )

    users_task = PythonOperator(
        task_id="preprocess_users",
        python_callable=preprocess_users,
        op_kwargs={"input_dir": "data/raw", "output_dir": "data/processed"}
    )

    moments_task = PythonOperator(
        task_id="preprocess_moments",
        python_callable=preprocess_moments,
        op_kwargs={"input_dir": "data/raw", "output_dir": "data/processed"}
    )

    # books and users must complete before moments
    [books_task, users_task] >> moments_task
```

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `id_generator.py` | Deterministic ID generation for all record types |
| `text_cleaner.py` | Fixes encoding, smart quotes, dashes, whitespace |
| `text_validator.py` | Validates length, language, gibberish, quality score |
| `issue_detector.py` | Detects PII, profanity, spam patterns |
| `metrics_calculator.py` | Flesch readability, word count, sentence metrics |
| `anomaly_detector.py` | IQR/z-score outliers, TF-IDF duplicates, style mismatch |
| `preprocessor.py` | Orchestrates all modules in correct order |
| `json_csv_adapter.py` | Reads JSON + CSV input files |
| `json_adapter.py` | Writes processed JSON output files |
| `gutenberg_lookup.py` | Gutenberg API with caching and config fallback |

---

## Authors

MOMENT Team — Group 23
IE7374 MLOps, Northeastern University

---

## Acknowledgments

- Literary passages from [Project Gutenberg](https://www.gutenberg.org/)
- Book metadata API: [Gutendex](https://gutendex.com/)
- Synthesized interpretation data for ML research