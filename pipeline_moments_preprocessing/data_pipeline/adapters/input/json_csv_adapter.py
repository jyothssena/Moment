# ============================================================
# json_csv_adapter.py
# MOMENT Preprocessing Pipeline - JSON/CSV Input Adapter
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Concrete implementation of BaseInputAdapter that
# reads our 3 raw input files:
#   - all_interpretations_450_FINAL_NO_BIAS.json  (JSON)
#   - passages.csv                                 (CSV)
#   - characters.csv                               (CSV)
#
# This is the Assignment 1 adapter. In production, this gets
# swapped for an API or database adapter - zero changes needed
# anywhere else in the pipeline.
#
# INHERITS FROM: BaseInputAdapter
# IMPLEMENTS:
#   - read_interpretations()
#   - read_passages()
#   - read_characters()
# ============================================================

import json        # for reading the interpretations JSON file
import pandas as pd  # type: ignore # for reading CSV files (passages, characters)
import logging
import os

# import the base class this adapter inherits from
from data_pipeline.adapters.input.base_adapter import BaseInputAdapter

# set up logger for this module
logger = logging.getLogger(__name__)


class JsonCsvInputAdapter(BaseInputAdapter):
    """
    Input adapter that reads from local JSON and CSV files.

    Reads:
        - interpretations from a JSON file
        - passages from a CSV file
        - characters from a CSV file

    All file paths are read from config/config.yaml so no
    paths are hardcoded here.
    """

    def __init__(self, config: dict):
        """
        Initialize the adapter and resolve all file paths.

        Args:
            config: full config dict loaded from config/config.yaml
        """
        # call parent __init__ to store config
        super().__init__(config)

        # resolve file paths from config
        # all paths in config are relative to project root
        self.interpretations_path = config["paths"]["raw_data"]["interpretations"]
        self.passages_path = config["paths"]["raw_data"]["passages"]
        self.characters_path = config["paths"]["raw_data"]["characters"]

        logger.info(
            f"JsonCsvInputAdapter initialized with paths:\n"
            f"  interpretations: {self.interpretations_path}\n"
            f"  passages:        {self.passages_path}\n"
            f"  characters:      {self.characters_path}"
        )

    def read_interpretations(self) -> list:
        """
        Read all 450 interpretations from the JSON file.

        Reads: all_interpretations_450_FINAL_NO_BIAS.json

        Each record in the JSON looks like:
        {
            "book": "Frankenstein",
            "passage_id": "passage_1",
            "character_id": 1,
            "character_name": "Emma Chen",
            "interpretation": "He says catastrophe...",
            "word_count": 99
        }

        Returns:
            list of dicts, one per interpretation (450 total)

        Raises:
            FileNotFoundError: if JSON file doesn't exist
            json.JSONDecodeError: if JSON file is malformed
        """
        logger.info(f"Reading interpretations from: {self.interpretations_path}")

        # check file exists before trying to open it
        if not os.path.exists(self.interpretations_path):
            raise FileNotFoundError(
                f"Interpretations file not found: {self.interpretations_path}\n"
                f"Make sure the file is in data/raw/ and the path in "
                f"config.yaml is correct."
            )

        # read the JSON file
        with open(self.interpretations_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        logger.info(f"Loaded {len(records)} raw interpretation records.")

        # run structural validation from base class
        # this checks required fields exist - not text quality
        valid_records = self.validate_interpretations(records)

        logger.info(
            f"After structural validation: {len(valid_records)} valid "
            f"interpretation records."
        )

        return valid_records

    def read_passages(self) -> list:
        """
        Read all 9 passages from the CSV file.

        Reads: passages.csv

        The CSV has columns:
            passage_id, book_title, book_author, chapter_number,
            passage_title, passage_text, num_interpretations

        NOTE: passage_id resets 1-3 per book, and book_title is
        "Unknown" for Frankenstein and "PRIDE & PREJUDICE" for
        Pride and Prejudice. These are normalized here using the
        passage_title_mapping from config.yaml.

        Returns:
            list of dicts, one per passage (9 total)

        Raises:
            FileNotFoundError: if CSV file doesn't exist
        """
        logger.info(f"Reading passages from: {self.passages_path}")

        # check file exists
        if not os.path.exists(self.passages_path):
            raise FileNotFoundError(
                f"Passages file not found: {self.passages_path}\n"
                f"Make sure passages.csv is in data/raw/."
            )

        # read CSV with pandas
        # keep_default_na=False prevents pandas from converting
        # "Unknown" and empty strings to NaN values
        df = pd.read_csv(self.passages_path, keep_default_na=False)

        logger.info(f"Loaded {len(df)} raw passage records from CSV.")
        logger.debug(f"Passage CSV columns: {list(df.columns)}")

        # get the title mapping from config to normalize book names
        # e.g. "PRIDE & PREJUDICE" → "Pride and Prejudice"
        # e.g. "Unknown" → "Frankenstein"
        title_mapping = self.config.get("passage_title_mapping", {})

        # convert dataframe rows to list of dicts
        records = []
        for _, row in df.iterrows():
            record = row.to_dict()

            # normalize book_title using the mapping from config
            # if book_title is in mapping, replace it; otherwise keep as-is
            original_title = str(record.get("book_title", "")).strip()
            if original_title in title_mapping:
                normalized_title = title_mapping[original_title]
                logger.debug(
                    f"Normalized book title: {original_title!r} → "
                    f"{normalized_title!r}"
                )
                record["book_title"] = normalized_title

            # convert passage_id to int if it's a string number
            # passages.csv stores it as 1, 2, 3 (int)
            try:
                record["passage_id"] = int(record["passage_id"])
            except (ValueError, TypeError):
                # if it can't be converted, leave it as-is
                pass

            # convert num_interpretations to int
            try:
                record["num_interpretations"] = int(
                    record.get("num_interpretations", 0)
                )
            except (ValueError, TypeError):
                record["num_interpretations"] = 0

            records.append(record)

        # run structural validation from base class
        valid_records = self.validate_passages(records)

        logger.info(
            f"After structural validation: {len(valid_records)} valid "
            f"passage records."
        )

        return valid_records

    def read_characters(self) -> list:
        """
        Read all 50 character profiles from the CSV file.

        Reads: characters.csv

        The CSV has columns:
            Name, Distribution_Category, Gender, Age, Profession,
            Personality, Interest, Reading_Intensity, Reading_Count,
            Experience_Level, Experience_Count, Journey,
            Style_1, Style_2, Style_3, Style_4

        Returns:
            list of dicts, one per character (50 total)

        Raises:
            FileNotFoundError: if CSV file doesn't exist
        """
        logger.info(f"Reading characters from: {self.characters_path}")

        # check file exists
        if not os.path.exists(self.characters_path):
            raise FileNotFoundError(
                f"Characters file not found: {self.characters_path}\n"
                f"Make sure characters.csv is in data/raw/."
            )

        # read CSV with pandas
        # keep_default_na=False prevents empty fields becoming NaN
        df = pd.read_csv(self.characters_path, keep_default_na=False)

        logger.info(f"Loaded {len(df)} raw character records from CSV.")
        logger.debug(f"Characters CSV columns: {list(df.columns)}")

        # convert dataframe to list of dicts
        records = []
        for _, row in df.iterrows():
            record = row.to_dict()

            # convert numeric fields to proper types
            # Age, Reading_Count, Experience_Count should be integers
            for int_field in ["Age", "Reading_Count", "Experience_Count"]:
                try:
                    record[int_field] = int(record[int_field])
                except (ValueError, TypeError):
                    # if conversion fails, set to 0 and log warning
                    logger.warning(
                        f"Could not convert {int_field} to int for "
                        f"character {record.get('Name', 'unknown')}. "
                        f"Setting to 0."
                    )
                    record[int_field] = 0

            records.append(record)

        # run structural validation from base class
        valid_records = self.validate_characters(records)

        logger.info(
            f"After structural validation: {len(valid_records)} valid "
            f"character records."
        )

        return valid_records


# ============================================================
# TEST BLOCK
# Run this file directly to verify all 3 files are read:
#   python -m data_pipeline.adapters.input.json_csv_adapter
# ============================================================

if __name__ == "__main__":

    import yaml # type: ignore
    import sys

    # set up basic logging for the test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("Testing JsonCsvInputAdapter")
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
    adapter = JsonCsvInputAdapter(config)

    # test reading interpretations
    print("\n--- Reading Interpretations ---")
    try:
        interpretations = adapter.read_interpretations()
        print(f"  Total records loaded: {len(interpretations)}")
        print(f"  First record:")
        first = interpretations[0]
        for key, value in first.items():
            # truncate long text for display
            display_val = str(value)[:80] + "..." if len(str(value)) > 80 else value
            print(f"    {key:20} : {display_val}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # test reading passages
    print("\n--- Reading Passages ---")
    try:
        passages = adapter.read_passages()
        print(f"  Total records loaded: {len(passages)}")
        print(f"  Book titles found:")
        # show unique book titles after normalization
        titles = set(p["book_title"] for p in passages)
        for title in sorted(titles):
            count = sum(1 for p in passages if p["book_title"] == title)
            print(f"    {title!r} ({count} passages)")
        print(f"  First record:")
        first = passages[0]
        for key, value in first.items():
            display_val = str(value)[:80] + "..." if len(str(value)) > 80 else value
            print(f"    {key:20} : {display_val}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # test reading characters
    print("\n--- Reading Characters ---")
    try:
        characters = adapter.read_characters()
        print(f"  Total records loaded: {len(characters)}")
        print(f"  Distribution categories:")
        # show distribution category breakdown
        from collections import Counter
        categories = Counter(c["Distribution_Category"] for c in characters)
        for cat, count in sorted(categories.items()):
            print(f"    {cat:15} : {count} characters")
        print(f"  First record:")
        first = characters[0]
        for key, value in first.items():
            display_val = str(value)[:80] + "..." if len(str(value)) > 80 else value
            print(f"    {key:20} : {display_val}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n✓ JsonCsvInputAdapter tests complete")