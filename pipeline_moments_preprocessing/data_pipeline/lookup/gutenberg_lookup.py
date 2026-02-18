# ============================================================
# gutenberg_lookup.py
# MOMENT Preprocessing Pipeline - Gutenberg API Lookup
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Looks up book metadata from the Gutenberg API
# (gutendex.com). Given a book title, returns Gutenberg ID,
# author, and our internal book_id.
#
# KEY FEATURE - CACHING:
# We only have 3 books but 450 interpretations. Without
# caching, we'd hit the API 450 times for the same 3 books.
# With caching, we hit it exactly 3 times and reuse results.
# Cache is in-memory (dict) - lives for the pipeline run.
#
# FALLBACK:
# If API call fails, falls back to config.yaml book metadata.
# This means the pipeline never fails just because the API
# is temporarily down.
#
# INHERITS FROM: BaseLookup
# ============================================================

import requests    # type: ignore # for HTTP calls to Gutenberg API
import logging
import time        # for retry delays

# import base class
from data_pipeline.lookup.base_lookup import BaseLookup
# import ID generator for generating book_id from gutenberg_id
from data_pipeline.utils.id_generator import generate_book_id

logger = logging.getLogger(__name__)


class GutenbergLookup(BaseLookup):
    """
    Looks up book metadata from the Gutenberg API.

    Uses gutendex.com - a JSON web API for Project Gutenberg.
    API docs: https://gutendex.com/

    Example API call:
        GET https://gutendex.com/books?search=Frankenstein
    
    Example response (simplified):
        {
            "results": [
                {
                    "id": 84,
                    "title": "Frankenstein",
                    "authors": [{"name": "Shelley, Mary Wollstonecraft"}]
                }
            ]
        }
    """

    def __init__(self, config: dict):
        """
        Initialize with config and set up in-memory cache.

        Args:
            config: full config dict from config/config.yaml
        """
        super().__init__(config)

        # get Gutenberg API settings from config
        self.api_base_url = config["gutenberg"]["api_base_url"]
        self.cache_enabled = config["gutenberg"]["cache_results"]
        self.timeout = config["gutenberg"]["timeout_seconds"]

        # in-memory cache: maps book_title → metadata dict
        # e.g. {"Frankenstein": {"book_id": "gutenberg_84", ...}}
        # lives for the duration of this pipeline run
        self._cache = {}

        # build a fallback lookup from config books section
        # used if API call fails
        # maps book_title → config metadata
        self._config_fallback = {
            book["book_title"]: book
            for book in config.get("books", [])
        }

        logger.info(
            f"GutenbergLookup initialized.\n"
            f"  API URL: {self.api_base_url}\n"
            f"  Cache enabled: {self.cache_enabled}\n"
            f"  Timeout: {self.timeout}s\n"
            f"  Config fallback books: "
            f"{list(self._config_fallback.keys())}"
        )

    def get_book_metadata(self, book_title: str) -> dict:
        """
        Look up metadata for a single book by title.

        Strategy:
        1. Check in-memory cache first (fastest)
        2. If not cached, hit Gutenberg API
        3. If API fails, fall back to config.yaml metadata
        4. If config fallback also fails, return not-found response

        Args:
            book_title: book title to look up e.g. "Frankenstein"

        Returns:
            dict with book_title, book_id, gutenberg_id, author, found
        """
        # --- Step 1: Check cache ---
        if self.cache_enabled and book_title in self._cache:
            logger.debug(f"Cache hit for: {book_title!r}")
            return self._cache[book_title]

        logger.info(f"Looking up book metadata for: {book_title!r}")

        # --- Step 2: Try Gutenberg API ---
        metadata = self._fetch_from_api(book_title)

        # --- Step 3: Fall back to config if API failed ---
        if not metadata["found"]:
            logger.warning(
                f"API lookup failed for {book_title!r}. "
                f"Trying config fallback..."
            )
            metadata = self._fetch_from_config(book_title)

        # --- Step 4: Cache the result if caching enabled ---
        if self.cache_enabled and metadata["found"]:
            self._cache[book_title] = metadata
            logger.debug(f"Cached metadata for: {book_title!r}")

        return metadata

    def get_all_books_metadata(self) -> list:
        """
        Look up metadata for all 3 books in the dataset.

        Reads book titles from config.yaml books section and
        looks up each one. Results are cached so subsequent
        calls per book are instant.

        Returns:
            list of metadata dicts, one per book
        """
        logger.info("Looking up metadata for all books in dataset...")

        # get book titles from config
        book_titles = [
            book["book_title"]
            for book in self.config.get("books", [])
        ]

        results = []
        for title in book_titles:
            metadata = self.get_book_metadata(title)
            results.append(metadata)
            logger.info(
                f"  {title!r}: "
                f"found={metadata['found']}, "
                f"book_id={metadata.get('book_id')}, "
                f"author={metadata.get('author')}"
            )

        # summary
        found_count = sum(1 for r in results if r["found"])
        logger.info(
            f"Book lookup complete: {found_count}/{len(results)} found."
        )

        return results

    def _fetch_from_api(self, book_title: str) -> dict:
        """
        Hit the Gutenberg API to look up a book.

        Private method - only called by get_book_metadata().

        Uses the gutendex search endpoint:
            GET {api_base_url}?search={book_title}

        Includes retry logic - tries up to 3 times before
        giving up, with a 2-second delay between retries.

        Args:
            book_title: book title to search for

        Returns:
            dict: metadata if found, not-found response if failed
        """
        max_retries = 3
        retry_delay = 2  # seconds between retries

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    f"API attempt {attempt}/{max_retries} "
                    f"for: {book_title!r}"
                )

                # build the API request URL
                # gutendex search endpoint: ?search=book+title
                params = {"search": book_title}

                # make the HTTP GET request
                response = requests.get(
                    self.api_base_url,
                    params=params,
                    timeout=self.timeout
                )

                # raise exception for HTTP error status codes
                # (4xx, 5xx responses)
                response.raise_for_status()

                # parse the JSON response
                data = response.json()

                # extract the best matching result
                metadata = self._parse_api_response(data, book_title)

                if metadata["found"]:
                    logger.info(
                        f"API found {book_title!r}: "
                        f"ID={metadata['gutenberg_id']}, "
                        f"author={metadata['author']!r}"
                    )
                    return metadata

                else:
                    # API responded but book not found in results
                    logger.warning(
                        f"API returned no results for: {book_title!r}"
                    )
                    return self.get_not_found_response(book_title)

            except requests.exceptions.Timeout:
                logger.warning(
                    f"API timeout on attempt {attempt}/{max_retries} "
                    f"for {book_title!r}."
                )

            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"API connection error on attempt "
                    f"{attempt}/{max_retries} for {book_title!r}."
                )

            except requests.exceptions.HTTPError as e:
                logger.warning(
                    f"API HTTP error on attempt {attempt}/{max_retries} "
                    f"for {book_title!r}: {e}"
                )

            except Exception as e:
                logger.error(
                    f"Unexpected API error on attempt "
                    f"{attempt}/{max_retries} for {book_title!r}: {e}"
                )

            # wait before retrying (except on last attempt)
            if attempt < max_retries:
                logger.debug(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        # all retries exhausted
        logger.error(
            f"All {max_retries} API attempts failed for: {book_title!r}"
        )
        return self.get_not_found_response(book_title)

    def _parse_api_response(self, data: dict, book_title: str) -> dict:
        """
        Parse the Gutenberg API response and extract metadata.

        The API returns a list of results. We find the best match
        by looking for a result whose title contains our search term.

        Gutenberg API response format:
        {
            "count": 1,
            "results": [
                {
                    "id": 84,
                    "title": "Frankenstein; Or, The Modern Prometheus",
                    "authors": [
                        {
                            "name": "Shelley, Mary Wollstonecraft",
                            "birth_year": 1797,
                            "death_year": 1851
                        }
                    ]
                }
            ]
        }

        Note: Gutenberg stores author names as "Last, First"
        We normalize this to "First Last" format.

        Args:
            data: parsed JSON response from API
            book_title: the title we searched for (for matching)

        Returns:
            dict: metadata if best match found, not-found if not
        """
        results = data.get("results", [])

        if not results:
            return self.get_not_found_response(book_title)

        # find best match: result whose title contains our search term
        # we check case-insensitively
        book_title_lower = book_title.lower()
        best_match = None

        for result in results:
            result_title = result.get("title", "").lower()

            # check if our search term appears in the result title
            # e.g. "frankenstein" in "frankenstein; or, the modern prometheus"
            if book_title_lower in result_title:
                best_match = result
                break  # take first match

        # if no title match, just take the first result
        # (API already ranked by relevance)
        if best_match is None and results:
            best_match = results[0]
            logger.debug(
                f"No exact title match for {book_title!r}, "
                f"using first result: {results[0].get('title')!r}"
            )

        if best_match is None:
            return self.get_not_found_response(book_title)

        # extract gutenberg ID
        gutenberg_id = best_match.get("id")

        # extract and normalize author name
        # Gutenberg format: "Shelley, Mary Wollstonecraft"
        # We want: "Mary Wollstonecraft Shelley"
        authors = best_match.get("authors", [])
        author = self._normalize_author_name(
            authors[0].get("name", "Unknown") if authors else "Unknown"
        )

        # generate our internal book_id
        book_id = generate_book_id(gutenberg_id)

        return {
            "book_title": book_title,
            "book_id": book_id,
            "gutenberg_id": gutenberg_id,
            "author": author,
            "found": True,
            # keep the full Gutenberg title for reference
            "gutenberg_title": best_match.get("title", book_title)
        }

    def _normalize_author_name(self, gutenberg_name: str) -> str:
        """
        Convert Gutenberg author name format to normal format.

        Gutenberg stores names as: "Last, First Middle"
        We want: "First Middle Last"

        Examples:
            "Shelley, Mary Wollstonecraft" → "Mary Wollstonecraft Shelley"
            "Austen, Jane"                 → "Jane Austen"
            "Fitzgerald, F. Scott"         → "F. Scott Fitzgerald"
            "Unknown"                      → "Unknown"

        Args:
            gutenberg_name: author name in Gutenberg format

        Returns:
            str: author name in normal "First Last" format
        """
        if not gutenberg_name or gutenberg_name == "Unknown":
            return "Unknown"

        # check if name is in "Last, First" format
        if "," in gutenberg_name:
            parts = gutenberg_name.split(",", 1)  # split on first comma only
            last_name = parts[0].strip()
            first_name = parts[1].strip()
            return f"{first_name} {last_name}"

        # already in normal format
        return gutenberg_name

    def _fetch_from_config(self, book_title: str) -> dict:
        """
        Fall back to config.yaml metadata if API fails.

        Private method - only called by get_book_metadata()
        when API lookup fails.

        Args:
            book_title: book title to look up

        Returns:
            dict: metadata from config if found, not-found if not
        """
        if book_title in self._config_fallback:
            config_book = self._config_fallback[book_title]
            gutenberg_id = config_book["gutenberg_id"]
            book_id = generate_book_id(gutenberg_id)

            logger.info(
                f"Config fallback found {book_title!r}: "
                f"ID={gutenberg_id}"
            )

            return {
                "book_title": book_title,
                "book_id": book_id,
                "gutenberg_id": gutenberg_id,
                "author": config_book.get("author", "Unknown"),
                "found": True,
                "gutenberg_title": book_title  # use our title as fallback
            }

        # not found in config either
        return self.get_not_found_response(book_title)

    def get_cache_stats(self) -> dict:
        """
        Return cache statistics for debugging/logging.

        Returns:
            dict: cache size and cached book titles
        """
        return {
            "cache_enabled": self.cache_enabled,
            "cached_books": list(self._cache.keys()),
            "cache_size": len(self._cache)
        }


# ============================================================
# TEST BLOCK
# Run this file directly to verify API lookup works:
#   python -m data_pipeline.lookup.gutenberg_lookup
# ============================================================

if __name__ == "__main__":

    import yaml # type: ignore
    import os
    import sys

    # set up logging for the test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("Testing GutenbergLookup")
    print("=" * 60)

    # load config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )),
        "config", "config.yaml"
    )

    if not os.path.exists(config_path):
        print(f"ERROR: config.yaml not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # initialize lookup
    lookup = GutenbergLookup(config)

    # test looking up all 3 books
    print("\n--- Looking up all books ---")
    all_books = lookup.get_all_books_metadata()

    for book in all_books:
        print(f"\n  Book: {book['book_title']!r}")
        print(f"    found:        {book['found']}")
        print(f"    book_id:      {book.get('book_id')}")
        print(f"    gutenberg_id: {book.get('gutenberg_id')}")
        print(f"    author:       {book.get('author')}")
        if "gutenberg_title" in book:
            print(f"    full title:   {book.get('gutenberg_title')}")

    # test cache stats
    print(f"\n--- Cache stats ---")
    stats = lookup.get_cache_stats()
    print(f"  Cache enabled: {stats['cache_enabled']}")
    print(f"  Cached books:  {stats['cached_books']}")
    print(f"  Cache size:    {stats['cache_size']}")

    # test that second lookup uses cache (should be instant)
    print(f"\n--- Testing cache (second lookup should be instant) ---")
    import time
    start = time.time()
    result = lookup.get_book_metadata("Frankenstein")
    elapsed = time.time() - start
    print(f"  Second lookup for 'Frankenstein': {elapsed:.4f}s")
    print(f"  (Should be near 0.0000s if cached)")

    # test fallback with unknown book
    print(f"\n--- Testing not-found response ---")
    result = lookup.get_book_metadata("This Book Does Not Exist")
    print(f"  found: {result['found']}")
    print(f"  book_id: {result['book_id']}")

    print("\n✓ GutenbergLookup tests complete")