# MOMENT Preprocessing Pipeline
**IE7374 MLOps Coursework — Group 23**

Preprocessing pipeline for the MOMENT reading platform. Takes raw reader interpretations, literary passages, and user profiles and produces three clean JSON datasets ready for ML use.

---

## Project Structure

```
moment_preprocessing/
├── data/
│   ├── raw/                          # Input files
│   │   ├── all_interpretations_450_FINAL_NO_BIAS.json
│   │   ├── passages.csv
│   │   └── characters.csv
│   └── processed/                    # Output files (generated)
│       ├── moments_processed.json
│       ├── books_processed.json
│       └── users_processed.json
│
├── pipeline/
│   ├── __init__.py
│   ├── preprocessor.py               # cleaning, validation, PII, metrics, IDs
│   └── anomalies.py                  # anomaly detection
│
├── run.py                            # entry point
├── config.yaml                       # all settings
├── requirements.txt
├── test_pipeline.py                  # all tests
└── README.md
```

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python run.py
```

That's it. Output files are written to `data/processed/`.

---

## What It Does

| Step | What happens |
|------|-------------|
| 1 | Reads 3 raw files (JSON + 2 CSVs) |
| 2 | Looks up book metadata from Gutenberg API |
| 3 | Cleans text (smart quotes, encoding, whitespace, emails) |
| 4 | Validates text quality (length, language, gibberish) |
| 5 | Detects issues (PII, profanity, spam) |
| 6 | Calculates metrics (readability, word count, sentence count) |
| 7 | Generates deterministic IDs for all records |
| 8 | Detects anomalies across full batch (outliers, duplicates, style mismatches) |
| 9 | Writes 3 output JSON files |

---

## Output Files

- **`moments_processed.json`** — 450 cleaned interpretation records
- **`books_processed.json`** — 9 cleaned literary passage records
- **`users_processed.json`** — 50 enriched reader profile records

---

## Running Tests

```bash
pytest test_pipeline.py -v
```

---

## Airflow Integration

```python
from run import run_pipeline

task = PythonOperator(
    task_id='preprocess',
    python_callable=run_pipeline,
    op_kwargs={
        'input_dir': 'data/raw',
        'output_dir': 'data/processed'
    }
)
```

---

## Configuration

All settings are in `config.yaml`. Key sections:

- `paths` — input/output file paths
- `books` — Gutenberg ID fallback if API is down
- `validation` — min/max word counts, quality threshold
- `issues` — profanity ratio, spam detection thresholds
- `anomalies` — IQR multiplier, z-score threshold, similarity threshold

---

*MOMENT Team — Group 23 — Northeastern University*