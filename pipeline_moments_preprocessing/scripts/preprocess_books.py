# ============================================================
# preprocess_books.py
# MOMENT Preprocessing Pipeline - Books Entry Point
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Airflow-compatible entry point for processing
# literary passages (books_processed.json).
#
# Can be run:
#   1. Directly from command line:
#      python -m scripts.preprocess_books
#
#   2. As an Airflow PythonOperator:
#      from scripts.preprocess_books import preprocess_books
#      task = PythonOperator(
#          task_id='preprocess_books',
#          python_callable=preprocess_books,
#          op_kwargs={
#              'input_dir': 'data/raw',
#              'output_dir': 'data/processed'
#          }
#      )
#
# NOTE: This script only processes books/passages.
# For full pipeline run, use preprocess_moments.py which
# calls all three processors in the correct order.
# ============================================================

import logging
import os
import sys
import yaml # type: ignore

# add project root to path so imports work
# when running as: python -m scripts.preprocess_books
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# import pipeline components
from data_pipeline.adapters.input.json_csv_adapter import JsonCsvInputAdapter
from data_pipeline.adapters.output.json_adapter import JsonOutputAdapter
from data_pipeline.lookup.gutenberg_lookup import GutenbergLookup
from data_pipeline.preprocessing.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


def preprocess_books(input_dir: str = None,
                     output_dir: str = None,
                     config_path: str = None) -> bool:
    """
    Process all literary passages and write books_processed.json.

    This is the Airflow-compatible entry point function.
    Accepts input_dir and output_dir as parameters so it
    works with DVC and Airflow conventions.

    Args:
        input_dir: path to raw data directory
                   overrides config if provided
                   e.g. "data/raw"
        output_dir: path to processed data directory
                    overrides config if provided
                    e.g. "data/processed"
        config_path: path to config.yaml
                     uses default if not provided

    Returns:
        bool: True if successful, False if failed
    """
    # --- Load config ---
    config = _load_config(config_path)

    # --- Override paths if provided ---
    # this allows Airflow/DVC to control paths externally
    if input_dir:
        config["paths"]["raw_data"]["passages"] = os.path.join(
            input_dir, "passages.csv"
        )
        config["paths"]["raw_data"]["characters"] = os.path.join(
            input_dir, "characters.csv"
        )
        config["paths"]["raw_data"]["interpretations"] = os.path.join(
            input_dir, "all_interpretations_450_FINAL_NO_BIAS.json"
        )

    if output_dir:
        config["paths"]["processed_data"]["books"] = os.path.join(
            output_dir, "books_processed.json"
        )

    # --- Set up logging ---
    _setup_logging(config)

    logger.info("=" * 50)
    logger.info("preprocess_books.py starting")
    logger.info("=" * 50)

    try:
        # initialize components
        input_adapter = JsonCsvInputAdapter(config)
        output_adapter = JsonOutputAdapter(config)
        lookup = GutenbergLookup(config)

        # read raw data
        passages = input_adapter.read_passages()

        # look up book metadata
        books_metadata = lookup.get_all_books_metadata()
        book_lookup = {
            book["book_title"]: book
            for book in books_metadata
            if book["found"]
        }

        # initialize preprocessor and process books only
        preprocessor = Preprocessor(config, input_adapter,
                                     output_adapter, lookup)
        books_processed = preprocessor.process_books(passages, book_lookup)

        # write output
        success = output_adapter.write_books(books_processed)

        if success:
            logger.info(
                f"preprocess_books.py complete. "
                f"{len(books_processed)} passages written to "
                f"{config['paths']['processed_data']['books']}"
            )
        else:
            logger.error("preprocess_books.py failed to write output.")

        return success

    except Exception as e:
        logger.error(
            f"preprocess_books.py failed: {e}",
            exc_info=True
        )
        return False


def _load_config(config_path: str = None) -> dict:
    """Load config.yaml from default or specified path."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found at: {config_path}"
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _setup_logging(config: dict) -> None:
    """Set up logging from config settings."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))

    # basic format
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
# Run directly: python -m scripts.preprocess_books
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process literary passages for MOMENT pipeline."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to raw data directory (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to processed data directory (overrides config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (uses default if not specified)"
    )

    args = parser.parse_args()

    success = preprocess_books(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config
    )

    # exit with appropriate code for Airflow/CI
    sys.exit(0 if success else 1)