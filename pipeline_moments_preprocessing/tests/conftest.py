# ============================================================
# conftest.py
# MOMENT Preprocessing Pipeline - Shared Test Fixtures
# IE7374 MLOps Coursework - Group 23
#
# PURPOSE: Defines shared pytest fixtures used across all
# test files. Fixtures are reusable setup code that pytest
# automatically injects into test functions.
#
# HOW FIXTURES WORK:
#   - Define a function with @pytest.fixture decorator
#   - Any test function that lists the fixture name as a
#     parameter gets the fixture's return value injected
#   - Fixtures with scope="session" are created once per
#     test run (expensive setup like config loading)
#   - Fixtures with scope="function" (default) are created
#     fresh for each test (ensures test isolation)
#
# USAGE IN TEST FILES:
#   def test_something(mock_config, sample_interpretation):
#       # mock_config and sample_interpretation are auto-injected
#       cleaner = TextCleaner(mock_config)
#       result = cleaner.clean(sample_interpretation)
#       assert result is not None
# ============================================================

import pytest # type: ignore
import os
import sys

# add project root to path so all imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# CONFIG FIXTURES
# ============================================================

@pytest.fixture(scope="session")
def mock_config():
    """
    Provide a complete mock config dict for all tests.

    scope="session" means this is created once for the entire
    test run - config doesn't change between tests so no need
    to recreate it every time.

    Returns:
        dict: complete mock config matching config.yaml structure
    """
    return {
        "paths": {
            "raw_data": {
                "interpretations": "data/raw/all_interpretations_450_FINAL_NO_BIAS.json",
                "passages": "data/raw/passages.csv",
                "characters": "data/raw/characters.csv"
            },
            "processed_data": {
                "moments": "data/processed/moments_processed.json",
                "books": "data/processed/books_processed.json",
                "users": "data/processed/users_processed.json"
            },
            "validation": {
                "report": "data/validation/validation_report.json"
            }
        },
        "books": [
            {
                "book_title": "Frankenstein",
                "gutenberg_id": 84,
                "book_id": "gutenberg_84",
                "author": "Mary Shelley",
                "passage_count": 3
            },
            {
                "book_title": "Pride and Prejudice",
                "gutenberg_id": 1342,
                "book_id": "gutenberg_1342",
                "author": "Jane Austen",
                "passage_count": 3
            },
            {
                "book_title": "The Great Gatsby",
                "gutenberg_id": 64317,
                "book_id": "gutenberg_64317",
                "author": "F. Scott Fitzgerald",
                "passage_count": 3
            }
        ],
        "passage_title_mapping": {
            "PRIDE & PREJUDICE": "Pride and Prejudice",
            "Unknown": "Frankenstein"
        },
        "id_generation": {
            "user_prefix": "user",
            "interpretation_prefix": "moment",
            "passage_prefix": "passage",
            "hash_length": 8
        },
        "text_cleaning": {
            "remove_extra_whitespace": True,
            "normalize_unicode": True,
            "fix_encoding": True,
            "fix_smart_quotes": True,
            "fix_dashes": True,
            "remove_urls": False,
            "remove_emails": True,
            "lowercase": False
        },
        "validation": {
            "interpretations": {
                "min_words": 10,
                "max_words": 600,
                "min_chars": 50,
                "max_chars": 4000,
                "quality_threshold": 0.5,
                "language": "en"
            },
            "passages": {
                "min_words": 20,
                "max_words": 1000,
                "min_chars": 100,
                "max_chars": 6000,
                "quality_threshold": 0.6,
                "language": "en"
            }
        },
        "issue_detection": {
            "pii": {
                "check_emails": True,
                "check_phone_numbers": True,
                "check_ssn": True,
                "check_credit_cards": True
            },
            "profanity": {
                "enabled": True,
                "ratio_threshold": 0.30
            },
            "spam": {
                "enabled": True,
                "caps_threshold": 0.50,
                "punctuation_threshold": 0.10,
                "repetitive_chars": 4,
                "repetitive_words_threshold": 0.30
            }
        },
        "anomaly_detection": {
            "enabled": True,
            "word_count": {
                "method": "iqr",
                "iqr_multiplier": 1.5
            },
            "readability": {
                "method": "zscore",
                "zscore_threshold": 2.5
            },
            "duplicate": {
                "enabled": True,
                "similarity_threshold": 0.85
            },
            "style_mismatch": {
                "enabled": True,
                "new_reader_readability_ceiling": 70,
                "well_read_readability_floor": 30
            }
        },
        "metrics": {
            "calculate_readability": True,
            "calculate_word_count": True,
            "calculate_char_count": True,
            "calculate_sentence_count": True,
            "calculate_avg_word_length": True,
            "calculate_avg_sentence_length": True
        },
        "gutenberg": {
            "strategy": "config",
            "api_base_url": "https://gutendex.com/books",
            "cache_results": True,
            "timeout_seconds": 10
        },
        "output": {
            "indent": 2,
            "ensure_ascii": False,
            "include_timestamp": True,
            "timestamp_format": "%Y-%m-%dT%H:%M:%S"
        },
        "logging": {
            "level": "WARNING",  # keep tests quiet
            "log_to_file": False,
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


# ============================================================
# TEXT SAMPLE FIXTURES
# Real examples from our dataset
# ============================================================

@pytest.fixture(scope="session")
def sample_interpretation_valid():
    """
    Valid interpretation from Emma Chen (99 words).
    Should pass all validation checks.
    """
    return (
        'He says "catastrophe" before anything bad happens. '
        "Just think about that. The creature opened its eyes. "
        "That's it. Victor's already calling it disaster. "
        '"Beautiful!--Great God!" Right next to each other. '
        "His brain's breaking. He built this with specific features "
        "and now he can't handle that it's real. "
        "That yellow eye. Why does he fixate on that one detail? "
        "Reducing the whole being to one gross feature so he "
        "doesn't have to see it as alive. "
        "The creature reaches for him later. Newborns do that. "
        "But Victor sees threat. He created the monster by "
        "interpreting everything wrong."
    )


@pytest.fixture(scope="session")
def sample_interpretation_short():
    """
    Very short interpretation from Ryan O'Connor (27 words).
    May fail word count validation depending on threshold.
    """
    return (
        "Dude builds monster. Monster opens eyes. "
        "Dude runs away screaming. How did he not see this "
        "coming? Like bro you literally assembled the parts yourself."
    )


@pytest.fixture(scope="session")
def sample_interpretation_with_smart_quotes():
    """
    Interpretation with unicode smart quotes and em dashes.
    Used to test text_cleaner smart quote and dash fixing.
    """
    return (
        "\u201cBeautiful!\u2014Great God!\u201d Right next to each other. "
        "His brain\u2019s breaking. He built this with specific "
        "features and now he can\u2019t handle that it\u2019s real. "
        "That yellow eye\u2026 why does he fixate on that one detail?"
    )


@pytest.fixture(scope="session")
def sample_interpretation_with_email():
    """
    Interpretation containing an email address (PII).
    Used to test email detection in issue_detector.
    """
    return (
        "Contact me at test.user@example.com for my analysis. "
        "He says catastrophe before anything bad happens. "
        "The creature opened its eyes and Victor ran away immediately. "
        "This shows how fear drives irrational behavior in humans."
    )


@pytest.fixture(scope="session")
def sample_interpretation_excessive_caps():
    """
    Interpretation with excessive capitalization (spam pattern).
    Used to test spam detection in issue_detector.
    """
    return (
        "THIS IS THE MOST AMAZING BOOK I HAVE EVER READ. "
        "VICTOR IS SO STUPID FOR RUNNING AWAY FROM HIS CREATION. "
        "THE CREATURE IS INNOCENT AND BEAUTIFUL AND DESERVES LOVE."
    )


@pytest.fixture(scope="session")
def sample_passage_valid():
    """
    Valid passage from Frankenstein.
    Should pass all passage validation checks.
    """
    return (
        "It was on a dreary night of November that I beheld the "
        "accomplishment of my toils. With an anxiety that almost "
        "amounted to agony, I collected the instruments of life "
        "around me, that I might infuse a spark of being into the "
        "lifeless thing that lay at my feet. It was already one in "
        "the morning; the rain pattered dismally against the panes, "
        "and my candle was nearly burnt out, when, by the glimmer "
        "of the half-extinguished light, I saw the dull yellow eye "
        "of the creature open; it breathed hard, and a convulsive "
        "motion agitated its limbs."
    )


@pytest.fixture(scope="session")
def sample_empty_text():
    """Empty string - should fail all validation checks."""
    return ""


@pytest.fixture(scope="session")
def sample_gibberish_text():
    """
    Gibberish text with abnormal vowel/consonant ratio.
    Used to test gibberish detection in text_validator.
    """
    return "asdfghjkl qwerty zxcvbnm poiuyt rewq asdfgh jklzxc vbnmqw"


# ============================================================
# CHARACTER/USER FIXTURES
# ============================================================

@pytest.fixture(scope="session")
def sample_character_new_reader():
    """
    Character profile for a NEW READER.
    Used to test style mismatch detection.
    """
    return {
        "Name": "Zoe Anderson",
        "Distribution_Category": "NEW READER",
        "Gender": "Female",
        "Age": 18,
        "Profession": "Freshman",
        "Personality": "Narrative",
        "Interest": "Romance",
        "Reading_Intensity": "Moderate",
        "Reading_Count": 17,
        "Experience_Level": "New",
        "Experience_Count": 1,
        "Journey": "Roommate kept laughing while reading. Got curious.",
        "Style_1": "Conversational",
        "Style_2": "Text",
        "Style_3": "Surface",
        "Style_4": "Brief"
    }


@pytest.fixture(scope="session")
def sample_character_well_read():
    """
    Character profile for a well-read reader.
    Used to test style mismatch detection.
    """
    return {
        "Name": "Dr. James Fletcher",
        "Distribution_Category": "DELIBERATE",
        "Gender": "Male",
        "Age": 48,
        "Profession": "Philosophy Professor",
        "Personality": "Philosophical",
        "Interest": "Literary",
        "Reading_Intensity": "Heavy",
        "Reading_Count": 50,
        "Experience_Level": "Well-read",
        "Experience_Count": 25,
        "Journey": "Read to answer philosophical questions.",
        "Style_1": "Academic",
        "Style_2": "World",
        "Style_3": "Deep",
        "Style_4": "Extended"
    }


# ============================================================
# METRICS FIXTURES
# Pre-computed metrics for anomaly detection tests
# ============================================================

@pytest.fixture(scope="session")
def sample_metrics_normal():
    """Normal metrics for a typical interpretation."""
    return {
        "word_count": 75,
        "char_count": 380,
        "sentence_count": 6,
        "avg_word_length": 4.8,
        "avg_sentence_length": 12.5,
        "readability_score": 62.5
    }


@pytest.fixture(scope="session")
def sample_metrics_short():
    """Metrics for a very short interpretation (potential outlier)."""
    return {
        "word_count": 8,
        "char_count": 40,
        "sentence_count": 1,
        "avg_word_length": 4.2,
        "avg_sentence_length": 8.0,
        "readability_score": 85.0
    }


@pytest.fixture(scope="session")
def sample_metrics_dataset():
    """
    A list of metrics representing a small dataset.
    Used to test fit() in anomaly_detector.
    Covers a range of word counts and readability scores.
    """
    return [
        {"word_count": 99,  "readability_score": 65.2},
        {"word_count": 59,  "readability_score": 72.1},
        {"word_count": 73,  "readability_score": 58.4},
        {"word_count": 81,  "readability_score": 55.3},
        {"word_count": 46,  "readability_score": 68.7},
        {"word_count": 27,  "readability_score": 85.0},
        {"word_count": 51,  "readability_score": 61.2},
        {"word_count": 68,  "readability_score": 63.8},
        {"word_count": 52,  "readability_score": 70.4},
        {"word_count": 71,  "readability_score": 57.9},
    ]