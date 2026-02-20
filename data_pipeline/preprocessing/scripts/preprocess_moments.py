# ============================================================
# preprocess_moments.py
# MOMENT Preprocessing Pipeline - Moments Entry Point
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Airflow-compatible entry point for processing
# reader interpretations (moments_processed.json).
#
# This is also the FULL PIPELINE RUNNER - it runs all three
# processors (books, users, moments) in the correct order
# and produces all 3 output files + validation report.
#
# Can be run:
#   1. Directly from command line (RECOMMENDED - runs full pipeline):
#      python -m scripts.preprocess_moments
#
#   2. As an Airflow PythonOperator:
#      from scripts.preprocess_moments import preprocess_moments
#      task = PythonOperator(
#          task_id='preprocess_moments',
#          python_callable=preprocess_moments,
#          op_kwargs={
#              'input_dir': 'data/raw',
#              'output_dir': 'data/processed'
#          }
#      )
#
# AIRFLOW DAG ORDERING:
#   preprocess_books >> preprocess_users >> preprocess_moments
#
# OUTPUT FILES PRODUCED:
#   data/processed/moments_processed.json  (450 records)
#   data/processed/books_processed.json    (9 records)
#   data/processed/users_processed.json    (50 records)
#   data/validation/validation_report.json
# ============================================================

import logging
import os
import sys
import yaml # type: ignore

# add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_pipeline.adapters.input.json_csv_adapter import JsonCsvInputAdapter
from data_pipeline.adapters.output.json_adapter import JsonOutputAdapter
from data_pipeline.lookup.gutenberg_lookup import GutenbergLookup
from data_pipeline.preprocessing.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


def preprocess_moments(input_dir: str = None,
                       output_dir: str = None,
                       config_path: str = None) -> bool:
    """
    Run the full MOMENT preprocessing pipeline.

    Processes all 3 datasets and writes all output files:
        - books_processed.json
        - users_processed.json
        - moments_processed.json
        - validation_report.json

    Args:
        input_dir: path to raw data directory (overrides config)
                   e.g. "data/raw"
        output_dir: path to processed data directory (overrides config)
                    e.g. "data/processed"
        config_path: path to config.yaml

    Returns:
        bool: True if all steps succeeded, False if any failed
    """
    # --- Load config ---
    config = _load_config(config_path)

    # --- Override paths if provided ---
    if input_dir:
        config["paths"]["raw_data"]["interpretations"] = os.path.join(
            input_dir, "all_interpretations_450_FINAL_NO_BIAS.json"
        )
        config["paths"]["raw_data"]["passages"] = os.path.join(
            input_dir, "passages.csv"
        )
        config["paths"]["raw_data"]["characters"] = os.path.join(
            input_dir, "characters.csv"
        )

    if output_dir:
        config["paths"]["processed_data"]["moments"] = os.path.join(
            output_dir, "moments_processed.json"
        )
        config["paths"]["processed_data"]["books"] = os.path.join(
            output_dir, "books_processed.json"
        )
        config["paths"]["processed_data"]["users"] = os.path.join(
            output_dir, "users_processed.json"
        )

    # --- Set up logging ---
    _setup_logging(config)

    logger.info("=" * 60)
    logger.info("MOMENT Preprocessing Pipeline - Full Run")
    logger.info("=" * 60)
    logger.info(f"Input dir:  {config['paths']['raw_data']}")
    logger.info(f"Output dir: {config['paths']['processed_data']}")

    try:
        # --- Initialize all components ---
        logger.info("Initializing pipeline components...")

        input_adapter = JsonCsvInputAdapter(config)
        output_adapter = JsonOutputAdapter(config)
        lookup = GutenbergLookup(config)
        preprocessor = Preprocessor(
            config, input_adapter, output_adapter, lookup
        )

        # --- Run full pipeline ---
        # preprocessor.run() handles all phases:
        # Phase 1: Read raw data
        # Phase 2: Look up book metadata
        # Phase 3: Process books
        # Phase 4: Process users
        # Phase 5: Process moments (2-pass with anomaly detection)
        # Phase 6: Write all outputs
        # Phase 7: Write validation report
        success = preprocessor.run()

        if success:
            logger.info("=" * 60)
            logger.info("Full pipeline completed successfully!")
            logger.info(
                f"Output files written to: "
                f"{os.path.dirname(config['paths']['processed_data']['moments'])}"
            )
            logger.info("=" * 60)
        else:
            logger.error("Pipeline completed with errors. Check logs.")

        return success

    except Exception as e:
        logger.error(
            f"Pipeline failed with critical error: {e}",
            exc_info=True
        )
        return False


def _load_config(config_path: str = None) -> dict:
    """Load config.yaml from default or specified path."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found at: {config_path}\n"
            f"Make sure you are running from the project root: "
            f"pipeline_moments_preprocessing/"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def _setup_logging(config: dict) -> None:
    """Set up logging from config settings."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get(
        "log_format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # configure root logger
    logging.basicConfig(level=level, format=log_format)

    # optionally write to file
    if log_config.get("log_to_file", False):
        log_file = log_config.get("log_file", "pipeline.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


# ============================================================
# COMMAND LINE ENTRY POINT
# Run full pipeline: python -m scripts.preprocess_moments
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the full MOMENT preprocessing pipeline. "
            "Processes all interpretations, passages, and user profiles. "
            "Produces moments_processed.json, books_processed.json, "
            "users_processed.json, and validation_report.json."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to raw data directory (overrides config.yaml paths)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory (overrides config.yaml paths)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (uses config/config.yaml if not specified)"
    )

    args = parser.parse_args()

    success = preprocess_moments(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config
    )

    # exit code 0 = success, 1 = failure
    # used by Airflow and CI/CD to detect pipeline failures
    sys.exit(0 if success else 1)