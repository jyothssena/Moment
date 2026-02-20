# ============================================================
# test_pipeline.py
# MOMENT Preprocessing Pipeline - Tests
#
# Run with: pytest test_pipeline.py -v
# ============================================================

import pytest # type: ignore
import sys
import os

# make sure pipeline package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.preprocessor import (
    clean_text, validate_text, detect_issues,
    calculate_metrics, make_book_id, make_passage_id,
    make_user_id, make_interpretation_id
)
from pipeline.anomalies import (
    _iqr_bounds, _mean_std, _check_wc_outlier,
    _check_read_outlier, _check_style_mismatch, detect_anomalies
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    """Minimal config for tests."""
    return {
        "validation": {
            "min_words": 10,
            "max_words": 600,
            "min_chars": 50,
            "quality_threshold": 0.5
        },
        "issues": {
            "profanity_ratio_threshold": 0.30,
            "caps_threshold": 0.50,
            "punct_threshold": 0.10,
            "repetitive_chars": 4,
            "repetitive_words_threshold": 0.30
        },
        "anomalies": {
            "iqr_multiplier": 1.5,
            "zscore_threshold": 2.5,
            "similarity_threshold": 0.85,
            "new_reader_ceiling": 70,
            "well_read_floor": 30
        },
        "timestamp_format": "%Y-%m-%dT%H:%M:%S"
    }


@pytest.fixture
def valid_text():
    """A valid interpretation that should pass all checks."""
    return (
        'He says "catastrophe" before anything bad happens. '
        "Just think about that. The creature opened its eyes. "
        "That's it. Victor's already calling it disaster. "
        "Beautiful Great God right next to each other. "
        "His brain is breaking. He built this with specific features "
        "and now he can't handle that it's real. "
        "That yellow eye. Why does he fixate on that one detail? "
        "Reducing the whole being to one gross feature."
    )


@pytest.fixture
def short_text():
    """A very short text that should fail word count validation."""
    return "Dude builds monster. Monster opens eyes. Runs away."


@pytest.fixture
def new_reader_character():
    return {
        "Name": "Zoe Anderson",
        "Distribution_Category": "NEW READER",
        "Experience_Level": "New"
    }


@pytest.fixture
def well_read_character():
    return {
        "Name": "Dr. James Fletcher",
        "Distribution_Category": "DELIBERATE",
        "Experience_Level": "Well-read"
    }


# ── ID Generation Tests ───────────────────────────────────────────────────────

class TestIDGeneration:

    def test_book_id_format(self):
        assert make_book_id(84) == "gutenberg_84"

    def test_book_id_all_books(self):
        assert make_book_id(1342) == "gutenberg_1342"
        assert make_book_id(64317) == "gutenberg_64317"

    def test_passage_id_format(self):
        assert make_passage_id("gutenberg_84", 1) == "gutenberg_84_passage_1"

    def test_passage_id_unique_across_books(self):
        id1 = make_passage_id("gutenberg_84", 1)
        id2 = make_passage_id("gutenberg_1342", 1)
        assert id1 != id2

    def test_user_id_format(self):
        uid = make_user_id("Emma Chen")
        assert uid.startswith("user_")
        assert "emma_chen" in uid

    def test_user_id_deterministic(self):
        assert make_user_id("Emma Chen") == make_user_id("Emma Chen")

    def test_user_id_unique_per_person(self):
        assert make_user_id("Emma Chen") != make_user_id("Marcus Williams")

    def test_user_id_handles_special_chars(self):
        uid = make_user_id("Ryan O'Connor")
        assert "'" not in uid

    def test_interpretation_id_format(self):
        iid = make_interpretation_id("Emma Chen", "gutenberg_84_passage_1", "some text")
        assert iid.startswith("moment_")

    def test_interpretation_id_deterministic(self):
        iid1 = make_interpretation_id("Emma Chen", "gutenberg_84_passage_1", "some text")
        iid2 = make_interpretation_id("Emma Chen", "gutenberg_84_passage_1", "some text")
        assert iid1 == iid2

    def test_interpretation_id_unique_per_character(self):
        iid1 = make_interpretation_id("Emma Chen", "gutenberg_84_passage_1", "some text")
        iid2 = make_interpretation_id("Marcus Williams", "gutenberg_84_passage_1", "some text")
        assert iid1 != iid2


# ── Text Cleaning Tests ───────────────────────────────────────────────────────

class TestTextCleaning:

    def test_returns_string(self):
        assert isinstance(clean_text("hello"), str)

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_empty_returns_empty(self):
        assert clean_text("") == ""

    def test_smart_quotes_removed(self):
        result = clean_text("\u201cBeautiful\u201d")
        assert "\u201c" not in result
        assert "\u201d" not in result
        assert '"' in result

    def test_smart_apostrophe_fixed(self):
        result = clean_text("Victor\u2019s creation")
        assert "\u2019" not in result
        assert "Victor" in result

    def test_em_dash_normalized(self):
        result = clean_text("before\u2014after")
        assert "\u2014" not in result
        assert "--" in result

    def test_en_dash_normalized(self):
        result = clean_text("before\u2013after")
        assert "\u2013" not in result

    def test_extra_spaces_collapsed(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result

    def test_leading_trailing_stripped(self):
        result = clean_text("  hello world  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_email_removed(self):
        result = clean_text("email me at test@example.com please")
        assert "test@example.com" not in result
        assert "[EMAIL REMOVED]" in result

    def test_ellipsis_normalized(self):
        result = clean_text("Just\u2026 think about that")
        assert "\u2026" not in result
        assert "..." in result

    def test_content_preserved(self):
        text = "Victor ran away from the creature immediately."
        result = clean_text(text)
        assert "Victor" in result
        assert "creature" in result


# ── Validation Tests ──────────────────────────────────────────────────────────

class TestValidation:

    def test_returns_dict(self, valid_text, cfg):
        result = validate_text(valid_text, cfg)
        assert isinstance(result, dict)

    def test_required_keys(self, valid_text, cfg):
        result = validate_text(valid_text, cfg)
        for key in ["is_valid", "quality_score", "quality_issues", "word_count", "language"]:
            assert key in result

    def test_valid_text_passes(self, valid_text, cfg):
        result = validate_text(valid_text, cfg)
        assert result["is_valid"] is True

    def test_valid_text_no_issues(self, valid_text, cfg):
        result = validate_text(valid_text, cfg)
        assert len(result["quality_issues"]) == 0

    def test_empty_text_fails(self, cfg):
        result = validate_text("", cfg)
        assert result["is_valid"] is False
        assert result["quality_score"] == 0.0

    def test_too_short_fails(self, short_text, cfg):
        result = validate_text(short_text, cfg)
        assert result["is_valid"] is False
        assert any("too_short" in i for i in result["quality_issues"])

    def test_too_long_fails(self, cfg):
        long_text = " ".join(["word"] * 650)
        result = validate_text(long_text, cfg)
        assert result["is_valid"] is False

    def test_gibberish_fails(self, cfg):
        result = validate_text("asdfghjkl qwerty zxcvbnm poiuyt rewq asdfgh", cfg)
        assert result["is_valid"] is False

    def test_quality_score_between_0_and_1(self, valid_text, cfg):
        result = validate_text(valid_text, cfg)
        assert 0.0 <= result["quality_score"] <= 1.0

    def test_quality_score_never_negative(self, cfg):
        result = validate_text("asdfghjkl", cfg)
        assert result["quality_score"] >= 0.0

    def test_word_count_correct(self, cfg):
        text = "one two three four five six seven eight nine ten eleven"
        result = validate_text(text, cfg)
        assert result["word_count"] == 11

    def test_none_text_fails(self, cfg):
        result = validate_text(None, cfg)
        assert result["is_valid"] is False


# ── Issue Detection Tests ─────────────────────────────────────────────────────

class TestIssueDetection:

    def test_returns_dict(self, valid_text, cfg):
        result = detect_issues(valid_text, cfg)
        assert isinstance(result, dict)

    def test_required_keys(self, valid_text, cfg):
        result = detect_issues(valid_text, cfg)
        for key in ["has_pii", "pii_types", "has_profanity",
                    "profanity_ratio", "is_spam", "spam_reasons"]:
            assert key in result

    def test_clean_text_no_issues(self, valid_text, cfg):
        result = detect_issues(valid_text, cfg)
        assert result["has_pii"] is False
        assert result["has_profanity"] is False
        assert result["is_spam"] is False

    def test_email_detected(self, cfg):
        result = detect_issues("Contact me at test@example.com for analysis.", cfg)
        assert result["has_pii"] is True
        assert "email" in result["pii_types"]

    def test_phone_detected(self, cfg):
        text = (
            "Call me at (555) 123-4567 to discuss. "
            "The creature opened its eyes and Victor ran away."
        )
        result = detect_issues(text, cfg)
        assert result["has_pii"] is True
        assert "phone_number" in result["pii_types"]

    def test_excessive_caps_spam(self, cfg):
        result = detect_issues(
            "THIS IS AMAZING I LOVE THIS BOOK SO MUCH IT IS GREAT AND WONDERFUL",
            cfg
        )
        assert result["is_spam"] is True
        assert any("excessive_caps" in r for r in result["spam_reasons"])

    def test_repetitive_chars_spam(self, cfg):
        result = detect_issues(
            "This passage is sooooo good and amazinggggg to read indeed.",
            cfg
        )
        assert result["is_spam"] is True
        assert any("repetitive_chars" in r for r in result["spam_reasons"])

    def test_repetitive_words_spam(self, cfg):
        result = detect_issues(
            "the the the the the book the the the the passage the the",
            cfg
        )
        assert result["is_spam"] is True

    def test_pii_types_is_list(self, valid_text, cfg):
        result = detect_issues(valid_text, cfg)
        assert isinstance(result["pii_types"], list)

    def test_empty_text_returns_clean(self, cfg):
        result = detect_issues("", cfg)
        assert result["has_pii"] is False
        assert result["is_spam"] is False


# ── Metrics Tests ─────────────────────────────────────────────────────────────

class TestMetrics:

    def test_returns_dict(self, valid_text):
        result = calculate_metrics(valid_text)
        assert isinstance(result, dict)

    def test_required_keys(self, valid_text):
        result = calculate_metrics(valid_text)
        for key in ["word_count", "char_count", "sentence_count",
                    "avg_word_length", "avg_sentence_length", "readability_score"]:
            assert key in result

    def test_empty_text_zeros(self):
        result = calculate_metrics("")
        assert result["word_count"] == 0
        assert result["readability_score"] == 0.0

    def test_word_count_correct(self):
        result = calculate_metrics("one two three four five")
        assert result["word_count"] == 5

    def test_char_count_excludes_spaces(self):
        result = calculate_metrics("hello world")
        assert result["char_count"] == 10

    def test_readability_in_range(self, valid_text):
        result = calculate_metrics(valid_text)
        assert 0.0 <= result["readability_score"] <= 100.0

    def test_simple_text_higher_readability(self):
        simple = "The cat sat on the mat. It was a big cat. He ran fast."
        complex_text = (
            "The epistemological ramifications of anthropomorphic attribution "
            "in phenomenological discourse necessitate philosophical reexamination."
        )
        r1 = calculate_metrics(simple)
        r2 = calculate_metrics(complex_text)
        assert r1["readability_score"] > r2["readability_score"]

    def test_none_returns_zeros(self):
        result = calculate_metrics(None)
        assert result["word_count"] == 0


# ── Anomaly Detection Tests ───────────────────────────────────────────────────

class TestAnomalyDetection:

    def test_iqr_bounds_calculated(self):
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        bounds = _iqr_bounds(values, 1.5)
        assert "lower" in bounds
        assert "upper" in bounds
        assert bounds["lower"] < bounds["upper"]

    def test_iqr_lower_less_than_upper(self):
        bounds = _iqr_bounds([50, 60, 70, 80, 90], 1.5)
        assert bounds["lower"] < bounds["upper"]

    def test_mean_std_calculated(self):
        stats = _mean_std([10, 20, 30, 40, 50])
        assert abs(stats["mean"] - 30.0) < 0.01
        assert stats["std"] > 0

    def test_wc_outlier_short(self):
        bounds = {"lower": 20.0, "upper": 150.0}
        details = []
        assert _check_wc_outlier(5, bounds, details) is True
        assert len(details) > 0

    def test_wc_outlier_normal(self):
        bounds = {"lower": 20.0, "upper": 150.0}
        details = []
        assert _check_wc_outlier(75, bounds, details) is False

    def test_read_outlier_extreme(self):
        stats = {"mean": 65.0, "std": 5.0}
        details = []
        # score of 90 = z-score of (90-65)/5 = 5.0, above threshold 2.5
        assert _check_read_outlier(90.0, stats, 2.5, details) is True

    def test_read_outlier_normal(self):
        stats = {"mean": 65.0, "std": 5.0}
        details = []
        # score of 67 = z-score of 0.4, below threshold
        assert _check_read_outlier(67.0, stats, 2.5, details) is False

    def test_style_mismatch_new_reader_complex(self, new_reader_character):
        details = []
        # readability 15 = very complex, floor is 30
        result = _check_style_mismatch(15.0, new_reader_character, 70, 30, details)
        assert result is True
        assert len(details) > 0

    def test_style_mismatch_new_reader_simple(self, new_reader_character):
        details = []
        # readability 80 = very simple, fine for new reader
        result = _check_style_mismatch(80.0, new_reader_character, 70, 30, details)
        assert result is False

    def test_style_mismatch_well_read_simple(self, well_read_character):
        details = []
        # readability 85 = very simple, above ceiling 70 for well-read
        result = _check_style_mismatch(85.0, well_read_character, 70, 30, details)
        assert result is True

    def test_style_mismatch_well_read_complex(self, well_read_character):
        details = []
        # readability 35 = complex writing, fine for well-read
        result = _check_style_mismatch(35.0, well_read_character, 70, 30, details)
        assert result is False

    def test_no_mismatch_without_character(self):
        details = []
        result = _check_style_mismatch(15.0, None, 70, 30, details)
        assert result is False

    def test_detect_anomalies_adds_anomalies_field(self, cfg):
        """Full detect_anomalies run populates anomalies on all records."""
        records = [
            {
                "interpretation_id": f"moment_{i:03d}",
                "character_name": "Emma Chen",
                "cleaned_interpretation": f"This is interpretation number {i} about the passage.",
                "metrics": {"word_count": 50 + i, "readability_score": 60.0 + i},
                "anomalies": {}
            }
            for i in range(10)
        ]
        characters = [{"Name": "Emma Chen", "Experience_Level": "Some", "Distribution_Category": "DELIBERATE"}]
        result = detect_anomalies(records, characters, cfg)

        # every record should now have a populated anomalies dict
        for r in result:
            assert "word_count_outlier" in r["anomalies"]
            assert "readability_outlier" in r["anomalies"]
            assert "duplicate_risk" in r["anomalies"]
            assert "style_mismatch" in r["anomalies"]
            assert "anomaly_details" in r["anomalies"]

    def test_detect_anomalies_returns_same_count(self, cfg):
        """detect_anomalies returns same number of records as input."""
        records = [
            {
                "interpretation_id": f"moment_{i:03d}",
                "character_name": "Emma Chen",
                "cleaned_interpretation": f"Interpretation {i} about the passage themes.",
                "metrics": {"word_count": 50, "readability_score": 65.0},
                "anomalies": {}
            }
            for i in range(5)
        ]
        characters = [{"Name": "Emma Chen", "Experience_Level": "Some", "Distribution_Category": "SOCIAL"}]
        result = detect_anomalies(records, characters, cfg)
        assert len(result) == 5