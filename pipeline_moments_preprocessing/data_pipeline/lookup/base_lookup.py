# ============================================================
# base_lookup.py
# MOMENT Preprocessing Pipeline - Base Lookup Interface
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Defines the interface (contract) that ALL lookup
# modules must follow. A "lookup" is anything that resolves
# book metadata - given a book title, return its Gutenberg ID,
# author, book_id etc.
#
# CURRENT IMPLEMENTATIONS:
#   gutenberg_lookup.py  - hits the Gutenberg API (production)
#
# FUTURE IMPLEMENTATIONS:
#   db_lookup.py         - looks up from local database
#   cache_lookup.py      - looks up from local cache only
#
# To swap lookup strategy: change gutenberg.strategy in
# config.yaml - zero code changes needed anywhere else.
# ============================================================

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseLookup(ABC):
    """
    Abstract base class for all book metadata lookup modules.

    Any class that resolves book metadata must inherit from
    this class and implement all @abstractmethod methods.
    """

    def __init__(self, config: dict):
        """
        Initialize with pipeline config.

        Args:
            config: full config dict from config/config.yaml
        """
        self.config = config
        logger.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def get_book_metadata(self, book_title: str) -> dict:
        """
        Look up metadata for a single book by title.

        MUST be implemented by every lookup module.

        Args:
            book_title: the book title to look up
                        e.g. "Frankenstein"

        Returns:
            dict with at minimum these fields:
            {
                "book_title": str,      # e.g. "Frankenstein"
                "book_id": str,         # e.g. "gutenberg_84"
                "gutenberg_id": int,    # e.g. 84
                "author": str,          # e.g. "Mary Shelley"
                "found": bool           # True if lookup succeeded
            }

            If book not found, returns:
            {
                "book_title": book_title,
                "book_id": None,
                "gutenberg_id": None,
                "author": "Unknown",
                "found": False
            }
        """
        pass

    @abstractmethod
    def get_all_books_metadata(self) -> list:
        """
        Look up metadata for all books in the dataset.

        MUST be implemented by every lookup module.

        Returns:
            list of metadata dicts, one per book
            (same format as get_book_metadata return value)
        """
        pass

    def get_not_found_response(self, book_title: str) -> dict:
        """
        Standard response when a book lookup fails.

        Shared utility - all lookup modules inherit this.
        Returns a consistent "not found" dict so the pipeline
        can handle missing metadata gracefully.

        Args:
            book_title: the title that wasn't found

        Returns:
            dict: standard not-found response
        """
        logger.warning(f"Book not found in lookup: {book_title!r}")
        return {
            "book_title": book_title,
            "book_id": None,
            "gutenberg_id": None,
            "author": "Unknown",
            "found": False
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config loaded)"