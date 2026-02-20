# ============================================================
# id_generator.py
# MOMENT Preprocessing Pipeline - ID Generation Utility
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Generates all unique IDs used across the pipeline.
#
# KEY PRINCIPLE: All IDs are DETERMINISTIC - meaning the same
# input always produces the same ID. This is critical for:
#   - Airflow idempotency: re-running a failed task produces
#     identical output, no duplicate records
#   - DVC reproducibility: same data = same pipeline output
#   - Deduplication: same record detected across multiple runs
#
# ID FORMATS:
#   user_id:            user_emma_chen_a1b2c3d4
#   interpretation_id:  moment_a1b2c3d4
#   book_id:            gutenberg_84
#   passage_id:         gutenberg_84_passage_1
# ============================================================

import hashlib    # for generating deterministic hashes
import re         # for sanitizing strings (removing special chars)
import logging    # for logging warnings and errors
import yaml       # type: ignore # for reading config/config.yaml
import os         # for building file paths

# set up logger for this module
# __name__ gives us "data_pipeline.utils.id_generator" as the logger name
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """
    Load the config.yaml file.

    Args:
        config_path: path to config.yaml. If None, looks for
                     config/config.yaml relative to project root.

    Returns:
        dict: the full config as a Python dictionary
    """
    # if no path provided, build path relative to this file's location
    if config_path is None:
        # this file is at data_pipeline/utils/id_generator.py
        # so we go up 3 levels to reach project root, then into config/
        base_dir = os.path.dirname(                  # data_pipeline/utils/
                       os.path.dirname(              # data_pipeline/
                           os.path.dirname(          # project root
                               os.path.abspath(__file__)
                           )
                       )
                   )
        config_path = os.path.join(base_dir, "config", "config.yaml")

    # read and parse the YAML file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def sanitize_name(name: str) -> str:
    """
    Convert a human-readable name into a clean string safe for use in IDs.

    Examples:
        "Emma Chen"      → "emma_chen"
        "Dr. James Fletcher" → "dr_james_fletcher"
        "Ryan O'Connor"  → "ryan_oconnor"

    Args:
        name: raw name string (e.g. character name)

    Returns:
        str: lowercase, underscored, alphanumeric-only string
    """
    if not name:
        return "unknown"

    # convert to lowercase
    clean = name.lower()

    # remove apostrophes and hyphens without replacing with underscore
    # e.g. "O'Connor" → "oconnor", "well-read" → "wellread"
    clean = re.sub(r"['\-]", "", clean)

    # replace any remaining non-alphanumeric characters with underscore
    # e.g. "Dr. James" → "dr__james" (double underscore handled next)
    clean = re.sub(r"[^a-z0-9]", "_", clean)

    # collapse multiple consecutive underscores into one
    # e.g. "dr__james" → "dr_james"
    clean = re.sub(r"_+", "_", clean)

    # strip leading/trailing underscores
    clean = clean.strip("_")

    return clean


def generate_hash(input_string: str, length: int = 8) -> str:
    """
    Generate a short deterministic hash from any input string.

    Uses MD5 (not for security - purely for deterministic short IDs).
    Same input string ALWAYS produces the same hash.

    Examples:
        "Emma Chen_passage_1_He says catastrophe..." → "a1b2c3d4"

    Args:
        input_string: any string to hash
        length: how many characters of the hash to return (default 8)

    Returns:
        str: short hex hash string of specified length
    """
    # encode to bytes, hash with MD5, get hex string, take first N chars
    hash_value = hashlib.md5(input_string.encode("utf-8")).hexdigest()[:length]
    return hash_value


def generate_user_id(character_name: str, config: dict = None) -> str:
    """
    Generate a deterministic user ID from a character name.

    Format: user_{sanitized_name}_{hash}
    Example: "Emma Chen" → "user_emma_chen_a1b2c3d4"

    The hash is based on the name itself, ensuring:
    - Same name always gets same ID
    - Different names with similar sanitized forms still get unique IDs
      e.g. "Dr. James" and "Dr James" would hash differently

    Args:
        character_name: the reader's name (e.g. "Emma Chen")
        config: config dict (loaded from config.yaml if not provided)

    Returns:
        str: deterministic user ID
    """
    # load config if not provided
    if config is None:
        config = load_config()

    # get settings from config
    prefix = config["id_generation"]["user_prefix"]           # "user"
    hash_length = config["id_generation"]["hash_length"]       # 8

    # sanitize the name for use in ID
    sanitized = sanitize_name(character_name)

    # generate hash from the original name (before sanitizing)
    # this ensures uniqueness even if two names sanitize to the same string
    name_hash = generate_hash(character_name, hash_length)

    # combine into final ID
    user_id = f"{prefix}_{sanitized}_{name_hash}"

    logger.debug(f"Generated user_id: {user_id} for character: {character_name}")

    return user_id


def generate_interpretation_id(character_name: str,
                                passage_id: str,
                                interpretation_text: str,
                                config: dict = None) -> str:
    """
    Generate a deterministic interpretation (moment) ID.

    Format: moment_{hash}
    The hash is based on: character_name + passage_id + first 100 chars of text
    This ensures:
    - Same interpretation always gets the same ID
    - Different interpretations always get different IDs

    Example: "moment_a1b2c3d4"

    Args:
        character_name: the reader's name
        passage_id: the passage ID (e.g. "gutenberg_84_passage_1")
        interpretation_text: the full interpretation text
        config: config dict (loaded from config.yaml if not provided)

    Returns:
        str: deterministic interpretation ID
    """
    # load config if not provided
    if config is None:
        config = load_config()

    # get settings from config
    prefix = config["id_generation"]["interpretation_prefix"]  # "moment"
    hash_length = config["id_generation"]["hash_length"]        # 8

    # build the string to hash from:
    # we use first 100 chars of text to keep the hash input manageable
    # while still being unique enough to avoid collisions
    hash_input = f"{character_name}_{passage_id}_{interpretation_text[:100]}"

    # generate the hash
    id_hash = generate_hash(hash_input, hash_length)

    # combine into final ID
    interpretation_id = f"{prefix}_{id_hash}"

    logger.debug(f"Generated interpretation_id: {interpretation_id} "
                 f"for {character_name} on {passage_id}")

    return interpretation_id


def generate_book_id(gutenberg_id: int) -> str:
    """
    Generate a book ID from a Gutenberg numeric ID.

    Format: gutenberg_{gutenberg_id}
    Example: 84 → "gutenberg_84"

    This is NOT hashed - it's a direct, human-readable ID
    because Gutenberg IDs are already unique and stable.

    Args:
        gutenberg_id: the numeric Project Gutenberg ID

    Returns:
        str: book ID string
    """
    book_id = f"gutenberg_{gutenberg_id}"

    logger.debug(f"Generated book_id: {book_id}")

    return book_id


def generate_passage_id(book_id: str, passage_number: int) -> str:
    """
    Generate a passage ID from a book ID and passage number.

    Format: {book_id}_passage_{passage_number}
    Example: "gutenberg_84", 1 → "gutenberg_84_passage_1"

    Args:
        book_id: the book ID (e.g. "gutenberg_84")
        passage_number: the passage number (1, 2, or 3)

    Returns:
        str: passage ID string
    """
    passage_id = f"{book_id}_passage_{passage_number}"

    logger.debug(f"Generated passage_id: {passage_id}")

    return passage_id


# ============================================================
# CONVENIENCE FUNCTION
# Generates all IDs for a single interpretation record at once
# Used by preprocess_moments.py to avoid calling each function
# separately for every record
# ============================================================

def generate_all_ids(character_name: str,
                     book_title: str,
                     gutenberg_id: int,
                     passage_number: int,
                     interpretation_text: str,
                     config: dict = None) -> dict:
    """
    Generate all IDs needed for a single interpretation record.

    Calls all the individual ID generation functions and returns
    everything in one dict so the preprocessor can use them easily.

    Args:
        character_name: reader's name (e.g. "Emma Chen")
        book_title: book name (e.g. "Frankenstein")
        gutenberg_id: Gutenberg numeric ID (e.g. 84)
        passage_number: passage number 1, 2, or 3
        interpretation_text: the full interpretation text
        config: config dict (loaded automatically if not provided)

    Returns:
        dict: {
            "user_id": "user_emma_chen_a1b2c3d4",
            "book_id": "gutenberg_84",
            "passage_id": "gutenberg_84_passage_1",
            "interpretation_id": "moment_a1b2c3d4"
        }
    """
    # load config once and pass it to all functions
    # avoids re-reading the file for every record
    if config is None:
        config = load_config()

    # generate each ID
    user_id = generate_user_id(character_name, config)
    book_id = generate_book_id(gutenberg_id)
    passage_id = generate_passage_id(book_id, passage_number)
    interpretation_id = generate_interpretation_id(
        character_name,
        passage_id,
        interpretation_text,
        config
    )

    return {
        "user_id": user_id,
        "book_id": book_id,
        "passage_id": passage_id,
        "interpretation_id": interpretation_id
    }


# ============================================================
# TEST BLOCK
# Run this file directly to verify ID generation is working:
#   python -m data_pipeline.utils.id_generator
# ============================================================

if __name__ == "__main__":

    # set up basic logging for the test
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("Testing ID Generator")
    print("=" * 60)

    # test sanitize_name
    print("\n--- sanitize_name ---")
    test_names = ["Emma Chen", "Dr. James Fletcher", "Ryan O'Connor",
                  "Marcus Williams", "Sophia Patel"]
    for name in test_names:
        print(f"  {name!r:30} → {sanitize_name(name)!r}")

    # test generate_user_id
    print("\n--- generate_user_id ---")
    for name in test_names:
        # load config from default path
        try:
            config = load_config()
            uid = generate_user_id(name, config)
            print(f"  {name!r:30} → {uid!r}")
        except FileNotFoundError:
            # if running outside project, use mock config
            mock_config = {
                "id_generation": {
                    "user_prefix": "user",
                    "interpretation_prefix": "moment",
                    "passage_prefix": "passage",
                    "hash_length": 8
                }
            }
            uid = generate_user_id(name, mock_config)
            print(f"  {name!r:30} → {uid!r}")

    # test generate_book_id
    print("\n--- generate_book_id ---")
    for gid in [84, 1342, 64317]:
        print(f"  Gutenberg ID {gid:6} → {generate_book_id(gid)!r}")

    # test generate_passage_id
    print("\n--- generate_passage_id ---")
    for gid, pnum in [(84, 1), (84, 2), (1342, 1), (64317, 3)]:
        book_id = generate_book_id(gid)
        print(f"  book_id={book_id!r}, passage={pnum} → "
              f"{generate_passage_id(book_id, pnum)!r}")

    # test generate_all_ids - full example
    print("\n--- generate_all_ids (full example) ---")
    mock_config = {
        "id_generation": {
            "user_prefix": "user",
            "interpretation_prefix": "moment",
            "passage_prefix": "passage",
            "hash_length": 8
        }
    }
    result = generate_all_ids(
        character_name="Emma Chen",
        book_title="Frankenstein",
        gutenberg_id=84,
        passage_number=1,
        interpretation_text="He says catastrophe before anything bad happens.",
        config=mock_config
    )
    for key, value in result.items():
        print(f"  {key:20} → {value!r}")

    # test determinism - same input should give same output every time
    print("\n--- Determinism check (run twice, must be identical) ---")
    ids_run1 = generate_all_ids(
        character_name="Maya Singh",
        book_title="Frankenstein",
        gutenberg_id=84,
        passage_number=2,
        interpretation_text="Shelley uses musical metaphors",
        config=mock_config
    )
    ids_run2 = generate_all_ids(
        character_name="Maya Singh",
        book_title="Frankenstein",
        gutenberg_id=84,
        passage_number=2,
        interpretation_text="Shelley uses musical metaphors",
        config=mock_config
    )
    print(f"  Run 1 == Run 2: {ids_run1 == ids_run2} ✓" if ids_run1 == ids_run2
          else "  FAILED: IDs are not deterministic!")

    print("\n✓ ID Generator tests complete")