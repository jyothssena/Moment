# ============================================================
# run.py
# MOMENT Preprocessing Pipeline - Entry Point
#
# Usage:
#   python run.py
#   python run.py --config config.yaml
#
# Airflow usage:
#   from run import run_pipeline
#   PythonOperator(task_id='preprocess', python_callable=run_pipeline,
#                  op_kwargs={'input_dir': 'data/raw',
#                             'output_dir': 'data/processed'})
# ============================================================

import logging
import os
import sys
import yaml # type: ignore
import argparse
from datetime import datetime

from preprocessor import (
    read_raw_data,
    lookup_books,
    process_books,
    process_users,
    process_moments_pass1,
    write_outputs
)
from anomalies import detect_anomalies


def load_config(config_path="config.yaml"):
    """Load config.yaml from the given path."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(cfg):
    """Set up logging from config."""
    level = getattr(logging, cfg.get("log_level", "INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S"
    )


def run_pipeline(input_dir=None, output_dir=None, config_path="config.yaml"):
    """
    Run the full MOMENT preprocessing pipeline.

    Args:
        input_dir:   override raw data directory (for Airflow)
        output_dir:  override output directory (for Airflow)
        config_path: path to config.yaml

    Returns:
        bool: True if successful, False if failed
    """
    cfg = load_config(config_path)
    setup_logging(cfg)

    # override paths if provided (used by Airflow)
    if input_dir:
        cfg["paths"]["raw"]["interpretations"] = os.path.join(
            input_dir, "all_interpretations_450_FINAL_NO_BIAS.json"
        )
        cfg["paths"]["raw"]["passages"]        = os.path.join(input_dir, "passages.csv")
        cfg["paths"]["raw"]["characters"]      = os.path.join(input_dir, "characters.csv")
    if output_dir:
        cfg["paths"]["processed"]["moments"] = os.path.join(output_dir, "moments_processed.json")
        cfg["paths"]["processed"]["books"]   = os.path.join(output_dir, "books_processed.json")
        cfg["paths"]["processed"]["users"]   = os.path.join(output_dir, "users_processed.json")

    start = datetime.now()
    logger = logging.getLogger(__name__)

    logger.info("=" * 55)
    logger.info("  MOMENT Preprocessing Pipeline starting")
    logger.info("=" * 55)

    try:
        # Step 1: Read raw data
        logger.info("Step 1/6  Reading raw data...")
        interpretations, passages, characters = read_raw_data(cfg)

        # Step 2: Look up book metadata from Gutenberg API
        logger.info("Step 2/6  Looking up book metadata...")
        book_meta = lookup_books(cfg)

        # Step 3: Process books (passages)
        logger.info("Step 3/6  Processing passages...")
        books = process_books(passages, book_meta, cfg)

        # Step 4: Process users (character profiles)
        logger.info("Step 4/6  Processing users...")
        users = process_users(characters, interpretations, cfg)

        # Step 5: Process moments - pass 1 (individual records)
        logger.info("Step 5a/6  Processing moments (pass 1 - individual records)...")
        moments = process_moments_pass1(interpretations, book_meta, cfg)

        # Step 5b: Anomaly detection - pass 2 (full batch)
        logger.info("Step 5b/6  Running anomaly detection (pass 2 - full batch)...")
        moments = detect_anomalies(moments, characters, cfg)

        # Step 6: Write outputs
        logger.info("Step 6/6  Writing output files...")
        write_outputs(moments, books, users, cfg)

        elapsed = (datetime.now() - start).seconds
        logger.info("=" * 55)
        logger.info(f"  Pipeline complete in {elapsed}s")
        logger.info(f"  {len(moments)} moments  |  {len(books)} books  |  {len(users)} users")
        valid = sum(1 for m in moments if m["is_valid"])
        anomalous = sum(
            1 for m in moments
            if any([m["anomalies"].get("word_count_outlier"),
                    m["anomalies"].get("readability_outlier"),
                    m["anomalies"].get("duplicate_risk"),
                    m["anomalies"].get("style_mismatch")])
        )
        logger.info(f"  Valid: {valid}/{len(moments)}  |  Anomalies: {anomalous}")
        logger.info("=" * 55)
        return True

    except Exception as e:
        logging.getLogger(__name__).error(f"Pipeline failed: {e}", exc_info=True)
        return False


# ── Command line entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOMENT Preprocessing Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input-dir", default=None, help="Override raw data directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    success = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config
    )
    sys.exit(0 if success else 1)