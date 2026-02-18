# ============================================================
# preprocess_users.py
# MOMENT Preprocessing Pipeline - Users Entry Point
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Airflow-compatible entry point for processing
# character profiles (users_processed.json).
#
# Can be run:
#   1. Directly from command line:
#      python -m scripts.preprocess_users
#
#   2. As an Airflow PythonOperator:
#      from scripts.preprocess_users import preprocess_users
#      task = PythonOperator(
#          task_id='preprocess_users',
#          python_callable=preprocess_users,
#          op_kwargs={
#              'input_dir': 'data/raw',
#              'output_dir': 'data/processed'
#          }
#      )
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


def preprocess_users(input_dir: str = None,
                     output_dir: str = None,
                     config_path: str = None) -> bool:
    """
    Process all character profiles and write users_processed.json.

    Args:
        input_dir: path to raw data directory (overrides config)
        output_dir: path to processed data directory (overrides config)
        config_path: path to config.yaml

    Returns:
        bool: True if successful, False if failed
    """
    # --- Load config ---
    config = _load_config(config_path)

    # --- Override paths if provided ---
    if input_dir:
        config["paths"]["raw_data"]["characters"] = os.path.join(
            input_dir, "characters.csv"
        )
        config["paths"]["raw_data"]["interpretations"] = os.path.join(
            input_dir, "all_interpretations_450_FINAL_NO_BIAS.json"
        )

    if output_dir:
        config["paths"]["processed_data"]["users"] = os.path.join(
            output_dir, "users_processed.json"
        )

    # --- Set up logging ---
    _setup_logging(config)

    logger.info("=" * 50)
    logger.info("preprocess_users.py starting")
    logger.info("=" * 50)

    try:
        # initialize components
        input_adapter = JsonCsvInputAdapter(config)
        output_adapter = JsonOutputAdapter(config)
        lookup = GutenbergLookup(config)

        # read raw data
        # users processing needs both characters AND interpretations
        # (to count total_interpretations and books_interpreted per user)
        characters = input_adapter.read_characters()
        interpretations = input_adapter.read_interpretations()

        # initialize preprocessor and process users only
        preprocessor = Preprocessor(config, input_adapter,
                                     output_adapter, lookup)
        users_processed = preprocessor.process_users(
            characters, interpretations
        )

        # write output
        success = output_adapter.write_users(users_processed)

        if success:
            logger.info(
                f"preprocess_users.py complete. "
                f"{len(users_processed)} users written to "
                f"{config['paths']['processed_data']['users']}"
            )
        else:
            logger.error("preprocess_users.py failed to write output.")

        return success

    except Exception as e:
        logger.error(
            f"preprocess_users.py failed: {e}",
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
    log_format = log_config.get(
        "log_format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.basicConfig(level=level, format=log_format)

    if log_config.get("log_to_file", False):
        log_file = log_config.get("log_file", "pipeline.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


# ============================================================
# COMMAND LINE ENTRY POINT
# Run directly: python -m scripts.preprocess_users
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process character profiles for MOMENT pipeline."
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Path to raw data directory (overrides config)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Path to processed data directory (overrides config)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml"
    )

    args = parser.parse_args()

    success = preprocess_users(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config
    )

    sys.exit(0 if success else 1)