# ============================================================
# base_adapter.py
# MOMENT Preprocessing Pipeline - Base Input Adapter Interface
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Defines the interface (contract) that ALL input
# adapters must follow. This is what makes the pipeline
# swappable - any input source (JSON file, CSV, API, database)
# just needs to implement these methods and the rest of the
# pipeline works without any changes.
#
# PATTERN: Abstract Base Class (ABC)
# - ABCs define methods that subclasses MUST implement
# - If a subclass doesn't implement a required method,
#   Python raises an error immediately when you try to
#   instantiate it - catching mistakes early
#
# CURRENT ADAPTERS:
#   json_csv_adapter.py  - reads JSON + CSV files (Assignment 1)
#
# FUTURE ADAPTERS (production):
#   api_adapter.py       - reads from REST API
#   db_adapter.py        - reads from database
#
# To add a new input source, just create a new file that
# inherits from BaseInputAdapter and implements all methods.
# Zero changes needed anywhere else in the pipeline.
# ============================================================

from abc import ABC, abstractmethod   # ABC = Abstract Base Class tools
import logging

# set up logger for this module
logger = logging.getLogger(__name__)


class BaseInputAdapter(ABC):
    """
    Abstract base class for all input adapters.

    Any class that reads data into the pipeline must inherit
    from this class and implement all @abstractmethod methods.

    This guarantees that regardless of WHERE data comes from
    (file, API, database), the pipeline always receives data
    in the same format and can call the same methods.
    """

    def __init__(self, config: dict):
        """
        Initialize the adapter with the pipeline config.

        All adapters receive the full config dict so they can
        read any settings they need (file paths, API URLs, etc.)

        Args:
            config: full config dict loaded from config/config.yaml
        """
        self.config = config
        logger.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def read_interpretations(self) -> list:
        """
        Read and return all 450 reader interpretations.

        MUST be implemented by every input adapter.

        Returns:
            list of dicts, where each dict represents one
            interpretation record with at minimum these fields:
            {
                "book": str,           # book title
                "passage_id": str,     # e.g. "passage_1"
                "character_id": int,   # 1-50
                "character_name": str, # e.g. "Emma Chen"
                "interpretation": str, # the interpretation text
                "word_count": int      # pre-computed word count
            }

        Raises:
            FileNotFoundError: if source file/API not found
            ValueError: if data format is unexpected
        """
        pass    # subclasses must implement this

    @abstractmethod
    def read_passages(self) -> list:
        """
        Read and return all 9 literary passages.

        MUST be implemented by every input adapter.

        Returns:
            list of dicts, where each dict represents one
            passage record with at minimum these fields:
            {
                "passage_id": str/int,   # 1, 2, or 3
                "book_title": str,       # e.g. "Frankenstein"
                "book_author": str,      # e.g. "Mary Shelley"
                "chapter_number": str,   # chapter reference
                "passage_title": str,    # single letter code
                "passage_text": str,     # the actual passage text
                "num_interpretations": int
            }

        Raises:
            FileNotFoundError: if source file/API not found
            ValueError: if data format is unexpected
        """
        pass    # subclasses must implement this

    @abstractmethod
    def read_characters(self) -> list:
        """
        Read and return all 50 character profiles.

        MUST be implemented by every input adapter.

        Returns:
            list of dicts, where each dict represents one
            character profile with at minimum these fields:
            {
                "Name": str,
                "Distribution_Category": str,
                "Gender": str,
                "Age": int,
                "Profession": str,
                "Personality": str,
                "Interest": str,
                "Reading_Intensity": str,
                "Reading_Count": int,
                "Experience_Level": str,
                "Experience_Count": int,
                "Journey": str,
                "Style_1": str,
                "Style_2": str,
                "Style_3": str,
                "Style_4": str
            }

        Raises:
            FileNotFoundError: if source file/API not found
            ValueError: if data format is unexpected
        """
        pass    # subclasses must implement this

    def validate_interpretations(self, records: list) -> list:
        """
        Basic structural check on interpretation records.

        This is NOT text quality validation (that happens in
        text_validator.py). This just checks that required
        fields exist and aren't empty.

        This method is NOT abstract - it's shared logic that
        all adapters can use without overriding.

        Args:
            records: list of interpretation dicts

        Returns:
            list: only the records that passed structural check

        Logs a warning for any record that fails.
        """
        required_fields = [
            "book", "passage_id", "character_id",
            "character_name", "interpretation", "word_count"
        ]

        valid_records = []

        for i, record in enumerate(records):
            # check all required fields exist and are not None/empty
            missing = [
                field for field in required_fields
                if field not in record or record[field] is None
            ]

            if missing:
                # log warning but don't crash - just skip this record
                logger.warning(
                    f"Interpretation record {i} missing fields: {missing}. "
                    f"Character: {record.get('character_name', 'unknown')}. "
                    f"Skipping."
                )
                continue

            # check interpretation text is not empty string
            if not str(record["interpretation"]).strip():
                logger.warning(
                    f"Interpretation record {i} has empty text. "
                    f"Character: {record.get('character_name', 'unknown')}. "
                    f"Skipping."
                )
                continue

            valid_records.append(record)

        # log summary
        skipped = len(records) - len(valid_records)
        if skipped > 0:
            logger.warning(
                f"Structural validation: {skipped} interpretation records "
                f"skipped out of {len(records)} total."
            )
        else:
            logger.info(
                f"Structural validation: all {len(records)} interpretation "
                f"records passed."
            )

        return valid_records

    def validate_passages(self, records: list) -> list:
        """
        Basic structural check on passage records.

        Checks required fields exist and passage text is not empty.
        Shared logic available to all adapters.

        Args:
            records: list of passage dicts

        Returns:
            list: only the records that passed structural check
        """
        required_fields = [
            "passage_id", "book_title", "passage_text"
        ]

        valid_records = []

        for i, record in enumerate(records):
            missing = [
                field for field in required_fields
                if field not in record or record[field] is None
            ]

            if missing:
                logger.warning(
                    f"Passage record {i} missing fields: {missing}. "
                    f"Skipping."
                )
                continue

            # check passage text is not empty
            if not str(record["passage_text"]).strip():
                logger.warning(
                    f"Passage record {i} has empty text. "
                    f"Book: {record.get('book_title', 'unknown')}. "
                    f"Skipping."
                )
                continue

            valid_records.append(record)

        skipped = len(records) - len(valid_records)
        if skipped > 0:
            logger.warning(
                f"Structural validation: {skipped} passage records "
                f"skipped out of {len(records)} total."
            )
        else:
            logger.info(
                f"Structural validation: all {len(records)} passage "
                f"records passed."
            )

        return valid_records

    def validate_characters(self, records: list) -> list:
        """
        Basic structural check on character records.

        Checks required fields exist.
        Shared logic available to all adapters.

        Args:
            records: list of character dicts

        Returns:
            list: only the records that passed structural check
        """
        required_fields = ["Name"]

        valid_records = []

        for i, record in enumerate(records):
            missing = [
                field for field in required_fields
                if field not in record or record[field] is None
            ]

            if missing:
                logger.warning(
                    f"Character record {i} missing fields: {missing}. "
                    f"Skipping."
                )
                continue

            valid_records.append(record)

        skipped = len(records) - len(valid_records)
        if skipped > 0:
            logger.warning(
                f"Structural validation: {skipped} character records "
                f"skipped out of {len(records)} total."
            )
        else:
            logger.info(
                f"Structural validation: all {len(records)} character "
                f"records passed."
            )

        return valid_records

    def __repr__(self) -> str:
        """String representation for logging/debugging."""
        return f"{self.__class__.__name__}(config loaded)"