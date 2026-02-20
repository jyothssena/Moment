# ============================================================
# json_adapter.py
# MOMENT Preprocessing Pipeline - JSON Output Adapter
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Concrete implementation of BaseOutputAdapter that
# writes our 3 processed output files as JSON:
#   - moments_processed.json   (450 interpretation records)
#   - books_processed.json     (9 passage records)
#   - users_processed.json     (50 user profile records)
#   - validation_report.json   (pipeline quality summary)
#
# This is the Assignment 1 adapter. In production, this gets
# swapped for a database or API adapter - zero changes needed
# anywhere else in the pipeline.
#
# INHERITS FROM: BaseOutputAdapter
# IMPLEMENTS:
#   - write_moments()
#   - write_books()
#   - write_users()
#   - write_validation_report()
# ============================================================

import json        # for writing JSON files
import os          # for creating directories and building paths
import logging

# import the base class this adapter inherits from
from data_pipeline.adapters.output.base_adapter import BaseOutputAdapter

# set up logger for this module
logger = logging.getLogger(__name__)


class JsonOutputAdapter(BaseOutputAdapter):
    """
    Output adapter that writes processed data to JSON files.

    Writes:
        - moments_processed.json
        - books_processed.json
        - users_processed.json
        - validation_report.json

    All output paths are read from config/config.yaml so no
    paths are hardcoded here.
    """

    def __init__(self, config: dict):
        """
        Initialize the adapter and resolve all output file paths.

        Also creates the output directories if they don't exist yet.
        This way the pipeline never fails just because a folder
        hasn't been created manually.

        Args:
            config: full config dict loaded from config/config.yaml
        """
        # call parent __init__ to store config
        super().__init__(config)

        # resolve output file paths from config
        self.moments_path = config["paths"]["processed_data"]["moments"]
        self.books_path = config["paths"]["processed_data"]["books"]
        self.users_path = config["paths"]["processed_data"]["users"]
        self.validation_path = config["paths"]["validation"]["report"]

        # get output formatting settings from config
        self.indent = config["output"]["indent"]                   # 2
        self.ensure_ascii = config["output"]["ensure_ascii"]       # False

        # create output directories if they don't exist
        # this handles both data/processed/ and data/validation/
        self._create_output_dirs()

        logger.info(
            f"JsonOutputAdapter initialized with paths:\n"
            f"  moments:    {self.moments_path}\n"
            f"  books:      {self.books_path}\n"
            f"  users:      {self.users_path}\n"
            f"  validation: {self.validation_path}"
        )

    def _create_output_dirs(self) -> None:
        """
        Create output directories if they don't already exist.

        Private method (prefixed with _) - only called internally.
        Uses os.makedirs with exist_ok=True so it never fails
        if the directory already exists.
        """
        # get unique directories from all output paths
        output_dirs = set([
            os.path.dirname(self.moments_path),
            os.path.dirname(self.books_path),
            os.path.dirname(self.users_path),
            os.path.dirname(self.validation_path)
        ])

        for directory in output_dirs:
            if directory:  # skip if directory is empty string (root level)
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Ensured output directory exists: {directory}")

    def _write_json(self, data: list or dict, filepath: str) -> bool: # type: ignore
        """
        Core JSON writing utility used by all write methods.

        Private method - handles the actual file writing with
        proper encoding, formatting, and error handling.

        Args:
            data: list of records or dict to write as JSON
            filepath: where to write the file

        Returns:
            bool: True if successful, False if failed
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    indent=self.indent,           # pretty print with 2 spaces
                    ensure_ascii=self.ensure_ascii, # allow unicode (em-dashes etc)
                    default=str                   # convert any non-serializable
                                                  # types to string as fallback
                )
            logger.debug(f"Successfully wrote JSON to: {filepath}")
            return True

        except IOError as e:
            # file system error (permissions, disk full, etc.)
            logger.error(f"IOError writing to {filepath}: {e}")
            return False

        except TypeError as e:
            # JSON serialization error (non-serializable object)
            logger.error(f"Serialization error writing to {filepath}: {e}")
            return False

        except Exception as e:
            # catch-all for unexpected errors
            logger.error(f"Unexpected error writing to {filepath}: {e}")
            return False

    def write_moments(self, records: list) -> bool:
        """
        Write 450 processed interpretation records to JSON.

        Output file: data/processed/moments_processed.json

        Args:
            records: list of fully processed interpretation dicts

        Returns:
            bool: True if write succeeded, False if failed
        """
        logger.info(
            f"Writing {len(records)} moment records to: {self.moments_path}"
        )

        # write to file
        success = self._write_json(records, self.moments_path)

        # log summary using base class utility
        self.log_write_summary("moments", len(records), success)

        if success:
            # log file size for debugging
            file_size_kb = os.path.getsize(self.moments_path) / 1024
            logger.info(
                f"moments_processed.json size: {file_size_kb:.1f} KB"
            )

        return success

    def write_books(self, records: list) -> bool:
        """
        Write 9 processed passage records to JSON.

        Output file: data/processed/books_processed.json

        Args:
            records: list of fully processed passage dicts

        Returns:
            bool: True if write succeeded, False if failed
        """
        logger.info(
            f"Writing {len(records)} book records to: {self.books_path}"
        )

        success = self._write_json(records, self.books_path)

        self.log_write_summary("books", len(records), success)

        if success:
            file_size_kb = os.path.getsize(self.books_path) / 1024
            logger.info(
                f"books_processed.json size: {file_size_kb:.1f} KB"
            )

        return success

    def write_users(self, records: list) -> bool:
        """
        Write 50 processed user profile records to JSON.

        Output file: data/processed/users_processed.json

        Args:
            records: list of fully processed user profile dicts

        Returns:
            bool: True if write succeeded, False if failed
        """
        logger.info(
            f"Writing {len(records)} user records to: {self.users_path}"
        )

        success = self._write_json(records, self.users_path)

        self.log_write_summary("users", len(records), success)

        if success:
            file_size_kb = os.path.getsize(self.users_path) / 1024
            logger.info(
                f"users_processed.json size: {file_size_kb:.1f} KB"
            )

        return success

    def write_validation_report(self, report: dict) -> bool:
        """
        Write the pipeline validation/quality report to JSON.

        Output file: data/validation/validation_report.json

        Args:
            report: dict containing pipeline quality summary

        Returns:
            bool: True if write succeeded, False if failed
        """
        logger.info(
            f"Writing validation report to: {self.validation_path}"
        )

        success = self._write_json(report, self.validation_path)

        self.log_write_summary("validation report", 1, success)

        return success


# ============================================================
# TEST BLOCK
# Run this file directly to verify JSON writing works:
#   python -m data_pipeline.adapters.output.json_adapter
# ============================================================

if __name__ == "__main__":

    import yaml # type: ignore
    import sys
    from datetime import datetime

    # set up basic logging for the test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("Testing JsonOutputAdapter")
    print("=" * 60)

    # load config from default location
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )),
        "config", "config.yaml"
    )

    if not os.path.exists(config_path):
        print(f"ERROR: config.yaml not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # initialize the adapter
    adapter = JsonOutputAdapter(config)

    # create mock records to test writing
    timestamp = datetime.now().strftime(
        config["output"]["timestamp_format"]
    )

    # mock moment record
    mock_moments = [
        {
            "interpretation_id": "moment_a1b2c3d4",
            "user_id": "user_emma_chen_a1b2c3d4",
            "book_id": "gutenberg_84",
            "passage_id": "gutenberg_84_passage_1",
            "book_title": "Frankenstein",
            "passage_number": 1,
            "character_name": "Emma Chen",
            "cleaned_interpretation": "He says catastrophe before anything bad happens.",
            "is_valid": True,
            "quality_score": 0.88,
            "quality_issues": [],
            "detected_issues": {
                "has_pii": False,
                "has_profanity": False,
                "is_spam": False
            },
            "anomalies": {
                "word_count_outlier": False,
                "readability_outlier": False,
                "duplicate_risk": False,
                "style_mismatch": False
            },
            "metrics": {
                "word_count": 8,
                "char_count": 47,
                "sentence_count": 1,
                "readability_score": 72.5,
                "avg_word_length": 5.1,
                "avg_sentence_length": 8.0
            },
            "timestamp": timestamp
        }
    ]

    # mock book record
    mock_books = [
        {
            "book_id": "gutenberg_84",
            "passage_id": "gutenberg_84_passage_1",
            "book_title": "Frankenstein",
            "book_author": "Mary Shelley",
            "chapter_number": "Unknown",
            "passage_title": "C",
            "passage_number": 1,
            "cleaned_passage_text": "It was on a dreary night of November...",
            "is_valid": True,
            "quality_score": 0.95,
            "metrics": {
                "word_count": 185,
                "char_count": 890,
                "sentence_count": 6,
                "readability_score": 45.2
            },
            "timestamp": timestamp
        }
    ]

    # mock user record
    mock_users = [
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
            "journey": "Wanted answers about whether life has inherent meaning.",
            "reading_styles": ["Academic", "Text-focused", "Deep", "Extended"],
            "total_interpretations": 3,
            "books_interpreted": ["Frankenstein", "Pride and Prejudice",
                                  "The Great Gatsby"],
            "timestamp": timestamp
        }
    ]

    # mock validation report
    mock_report = {
        "total_interpretations": 450,
        "valid_interpretations": 448,
        "invalid_interpretations": 2,
        "total_passages": 9,
        "total_users": 50,
        "anomalies_detected": 12,
        "processing_timestamp": timestamp
    }

    # test all write methods
    print("\n--- Testing write_moments ---")
    success = adapter.write_moments(mock_moments)
    print(f"  Success: {success}")
    print(f"  File exists: {os.path.exists(adapter.moments_path)}")

    print("\n--- Testing write_books ---")
    success = adapter.write_books(mock_books)
    print(f"  Success: {success}")
    print(f"  File exists: {os.path.exists(adapter.books_path)}")

    print("\n--- Testing write_users ---")
    success = adapter.write_users(mock_users)
    print(f"  Success: {success}")
    print(f"  File exists: {os.path.exists(adapter.users_path)}")

    print("\n--- Testing write_validation_report ---")
    success = adapter.write_validation_report(mock_report)
    print(f"  Success: {success}")
    print(f"  File exists: {os.path.exists(adapter.validation_path)}")

    # verify output by reading back and printing
    print("\n--- Verifying output (reading back moments_processed.json) ---")
    with open(adapter.moments_path, "r") as f:
        written = json.load(f)
    print(f"  Records in file: {len(written)}")
    print(f"  First record keys: {list(written[0].keys())}")

    print("\n✓ JsonOutputAdapter tests complete")
    print(f"\nCheck your data/processed/ folder for the output files!")