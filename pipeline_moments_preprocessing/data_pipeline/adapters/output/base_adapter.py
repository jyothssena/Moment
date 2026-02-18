# ============================================================
# base_adapter.py
# MOMENT Preprocessing Pipeline - Base Output Adapter Interface
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Defines the interface (contract) that ALL output
# adapters must follow. Just like the input base adapter,
# this makes the output destination fully swappable.
#
# CURRENT ADAPTERS:
#   json_adapter.py  - writes JSON files (Assignment 1)
#
# FUTURE ADAPTERS (production):
#   db_adapter.py    - writes to database
#   api_adapter.py   - pushes to REST API
#
# To add a new output destination, just create a new file
# that inherits from BaseOutputAdapter and implements all
# methods. Zero changes needed anywhere else in the pipeline.
# ============================================================

from abc import ABC, abstractmethod   # ABC = Abstract Base Class tools
import logging

# set up logger for this module
logger = logging.getLogger(__name__)


class BaseOutputAdapter(ABC):
    """
    Abstract base class for all output adapters.

    Any class that writes processed data out of the pipeline
    must inherit from this class and implement all
    @abstractmethod methods.

    This guarantees that regardless of WHERE data goes
    (JSON file, database, API), the pipeline always calls
    the same methods in the same way.
    """

    def __init__(self, config: dict):
        """
        Initialize the adapter with the pipeline config.

        Args:
            config: full config dict loaded from config/config.yaml
        """
        self.config = config
        logger.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def write_moments(self, records: list) -> bool:
        """
        Write the 450 processed interpretation records.

        MUST be implemented by every output adapter.

        Args:
            records: list of fully processed interpretation dicts,
                     each containing all fields defined in the
                     moments_processed.json schema:
                     {
                         "interpretation_id": str,
                         "user_id": str,
                         "book_id": str,
                         "passage_id": str,
                         "book_title": str,
                         "passage_number": int,
                         "character_name": str,
                         "cleaned_interpretation": str,
                         "is_valid": bool,
                         "quality_score": float,
                         "quality_issues": list,
                         "detected_issues": dict,
                         "anomalies": dict,
                         "metrics": dict,
                         "timestamp": str
                     }

        Returns:
            bool: True if write succeeded, False if it failed

        Raises:
            IOError: if write operation fails
        """
        pass    # subclasses must implement this

    @abstractmethod
    def write_books(self, records: list) -> bool:
        """
        Write the 9 processed passage records.

        MUST be implemented by every output adapter.

        Args:
            records: list of fully processed passage dicts,
                     each containing all fields defined in the
                     books_processed.json schema:
                     {
                         "book_id": str,
                         "passage_id": str,
                         "book_title": str,
                         "book_author": str,
                         "chapter_number": str,
                         "passage_title": str,
                         "passage_number": int,
                         "cleaned_passage_text": str,
                         "is_valid": bool,
                         "quality_score": float,
                         "metrics": dict,
                         "timestamp": str
                     }

        Returns:
            bool: True if write succeeded, False if it failed

        Raises:
            IOError: if write operation fails
        """
        pass    # subclasses must implement this

    @abstractmethod
    def write_users(self, records: list) -> bool:
        """
        Write the 50 processed user profile records.

        MUST be implemented by every output adapter.

        Args:
            records: list of fully processed user dicts,
                     each containing all fields defined in the
                     users_processed.json schema:
                     {
                         "user_id": str,
                         "character_name": str,
                         "gender": str,
                         "age": int,
                         "profession": str,
                         "distribution_category": str,
                         "personality": str,
                         "interest": str,
                         "reading_intensity": str,
                         "reading_count": int,
                         "experience_level": str,
                         "experience_count": int,
                         "journey": str,
                         "reading_styles": list,
                         "total_interpretations": int,
                         "books_interpreted": list,
                         "timestamp": str
                     }

        Returns:
            bool: True if write succeeded, False if it failed

        Raises:
            IOError: if write operation fails
        """
        pass    # subclasses must implement this

    @abstractmethod
    def write_validation_report(self, report: dict) -> bool:
        """
        Write the validation/quality report.

        MUST be implemented by every output adapter.

        Args:
            report: dict containing pipeline quality summary:
                    {
                        "total_interpretations": int,
                        "valid_interpretations": int,
                        "invalid_interpretations": int,
                        "total_passages": int,
                        "total_users": int,
                        "anomalies_detected": int,
                        "issues_detected": dict,
                        "processing_timestamp": str
                    }

        Returns:
            bool: True if write succeeded, False if it failed
        """
        pass    # subclasses must implement this

    def log_write_summary(self, record_type: str,
                          count: int, success: bool) -> None:
        """
        Log a summary after each write operation.

        Shared utility available to all output adapters.
        NOT abstract - adapters inherit this without overriding.

        Args:
            record_type: what was written e.g. "moments", "books"
            count: how many records were written
            success: whether the write succeeded
        """
        if success:
            logger.info(
                f"Successfully wrote {count} {record_type} records."
            )
        else:
            logger.error(
                f"Failed to write {record_type} records. "
                f"Check output adapter for details."
            )

    def __repr__(self) -> str:
        """String representation for logging/debugging."""
        return f"{self.__class__.__name__}(config loaded)"