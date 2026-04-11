"""
bq_loader.py — MOMENT BigQuery Data Acquisition
=================================================
Drop-in replacement for DataAcquisition — same interface,
reads from BigQuery instead of GCS.

Reads from:
  - moment-486719.moments_raw.interpretations_raw
  - moment-486719.moments_raw.passage_details_new
  - moment-486719.moments_raw.user_details_new

Usage:
    bq = BQLoader(config_path="data_pipeline/config/config.yaml")
    metadata = bq.run()
    dataframes = bq.get_dataframes()

    # or access individually
    dataframes["interpretations_raw"]
    dataframes["passage_details_new"]
    dataframes["user_details_new"]
"""

import logging
import os
import yaml # type: ignore
from datetime import datetime
from google.cloud import bigquery
import pandas as pd # type: ignore

logger = logging.getLogger('moment_pipeline')

# Configure logging directory
LOGS_DIR = os.environ.get('LOGS_DIR', '/opt/airflow/logs')
os.makedirs(LOGS_DIR, exist_ok=True)

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO
    
# Create separate handlers for each log level
info_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'team_pipeline_INFO.log'))
info_handler.setLevel(logging.INFO)
info_handler.addFilter(InfoFilter())


PROJECT_ID = "moment-486719"
DATASET    = "moments_raw"


class BQLoader:
    def __init__(self, config_path="data_pipeline/config/config.yaml"):
        """
        Initialize with config file path — same signature as DataAcquisition.
        """
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataframes = None  # populated by run()

        # BQ client — uses ADC (gcloud auth application-default login)
        project = self.config.get("acquisition", {}).get("project_id") \
                  or os.environ.get("GOOGLE_CLOUD_PROJECT") \
                  or PROJECT_ID
        logger.info(f"Initializing BigQuery client for project: {project}") 
        self.client = bigquery.Client(project=project)
        logger.info(f"Initialized BigQuery client for project: {project}") 
        # table names — overridable via config['bigquery'] block
        bq_cfg = self.config.get("bigquery", {})
        self.interpretations_table = bq_cfg.get(
            "interpretations_table",
            f"{PROJECT_ID}.{DATASET}.interpretations_train"
        )
        self.passages_table = bq_cfg.get(
            "passages_table",
            f"{PROJECT_ID}.{DATASET}.passage_details_new"
        )
        self.characters_table = bq_cfg.get(
            "characters_table",
            f"{PROJECT_ID}.{DATASET}.user_details_new"
        )

    def run(self):
        """
        Load all three tables from BigQuery into memory.
        Same return shape as DataAcquisition.run().
        """
        logger.info("Starting BQ data acquisition")

        self.dataframes = {
            "interpretations_train": self._load_interpretations(),
            "passage_details_new": self._load_passages(),
            "user_details_new":    self._load_characters(),
        }

        total_rows = sum(len(df) for df in self.dataframes.values())
        num_files  = len(self.dataframes)

        logger.info(f"Acquisition complete: {num_files} tables, {total_rows} total rows")

        return {
            "timestamp":  self.timestamp,
            "num_files":  num_files,
            "total_rows": total_rows,
            "files":      list(self.dataframes.keys()),
            "project":    PROJECT_ID,
            "dataset":    DATASET,
        }

    def get_dataframes(self):
        """
        Return the loaded dataframes dict.
        Same as DataAcquisition.get_dataframes().

        Keys:
            "interpretations_train"  — 450 rows
            "passage_details_new"  — 9 rows
            "user_details_new"     — 50 rows
        """
        if self.dataframes is None:
            raise ValueError("No data loaded. Call run() first.")
        return self.dataframes

    # ------------------------------------------------------------------
    # Private loaders — one per table
    # ------------------------------------------------------------------

    def _load_interpretations(self) -> pd.DataFrame:
        """
        Columns: character_id, character_name, passage_id,
                 book, interpretation, word_count
        """
        query = f"""
            SELECT
                character_id,
                character_name,
                passage_id,
                book,
                interpretation,
                word_count
            FROM `{self.interpretations_table}`
        """
        logger.info(f"Loading interpretations from {self.interpretations_table}")
        df = self._run_query(query)
        logger.info(f"  -> {len(df)} rows")
        return df

    def _load_passages(self) -> pd.DataFrame:
        """
        Columns: passage_id, book_title, passage_title, passage_text,
                 book_author, chapter_number, num_interpretations
        """
        query = f"""
            SELECT
                passage_id,
                book_title,
                passage_title,
                passage_text,
                book_author,
                chapter_number,
                num_interpretations
            FROM `{self.passages_table}`
        """
        logger.info(f"Loading passages from {self.passages_table}")
        df = self._run_query(query)
        logger.info(f"  -> {len(df)} rows")
        return df

    def _load_characters(self) -> pd.DataFrame:
        """
        Columns: Name, Age, Gender, Profession, Personality, Interest,
                 Reading_Intensity, Experience_Level, Experience_Count,
                 Reading_Count, Journey, Distribution_Category,
                 Style_1, Style_2, Style_3, Style_4
        """
        query = f"""
            SELECT
                Name,
                Age,
                Gender,
                Profession,
                Personality,
                Interest,
                Reading_Intensity,
                Experience_Level,
                Experience_Count,
                Reading_Count,
                Journey,
                Distribution_Category,
                Style_1,
                Style_2,
                Style_3,
                Style_4
            FROM `{self.characters_table}`
        """
        logger.info(f"Loading characters from {self.characters_table}")
        df = self._run_query(query)
        logger.info(f"  -> {len(df)} rows")
        return df

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _run_query(self, query: str) -> pd.DataFrame:
        try:
            return self.client.query(query).to_dataframe()
        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            raise


# ------------------------------------------------------------------
# Standalone entry point — mirrors DataAcquisition __main__ block
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MOMENT BQ Data Acquisition")
    parser.add_argument("--config", default="data_pipeline/config/config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    bq = BQLoader(config_path=args.config)
    metadata = bq.run()
    logger.info(f"Done: {metadata['num_files']} tables, {metadata['total_rows']} rows")