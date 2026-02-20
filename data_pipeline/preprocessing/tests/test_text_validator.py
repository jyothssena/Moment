# ============================================================
# test_text_validator.py
# MOMENT Preprocessing Pipeline - Text Validator Tests
# IE7374 MLOps Coursework - Group 23
#
# Run with: pytest tests/test_text_validator.py -v
# ============================================================

import pytest # type: ignore
from data_pipeline.preprocessing.text_validator import TextValidator


@pytest.fixture
def validator(mock_config):
    """Create a TextValidator instance for each test."""
    return TextValidator(mock_config)


class TestValidateBasic:
    """Basic validate() method tests."""

    def test_returns_dict(self, validator, sample_interpretation_valid):
        """validate() returns a dict."""
        result = validator.validate(sample_interpretation_valid)
        assert isinstance(result, dict)

    def test_returns_required_keys(self, validator, sample_interpretation_valid):
        """Result contains all required keys."""
        result = validator.validate(sample_interpretation_valid)
        assert "is_valid" in result
        assert "quality_score" in result
        assert "quality_issues" in result
        assert "word_count" in result
        assert "char_count" in result
        assert "language" in result

    def test_is_valid_is_bool(self, validator, sample_interpretation_valid):
        """is_valid is always a bool."""
        result = validator.validate(sample_interpretation_valid)
        assert isinstance(result["is_valid"], bool)

    def test_quality_score_is_float(self, validator, sample_interpretation_valid):
        """quality_score is always a float."""
        result = validator.validate(sample_interpretation_valid)
        assert isinstance(result["quality_score"], float)

    def test_quality_score_range(self, validator, sample_interpretation_valid):
        """quality_score is always between 0.0 and 1.0."""
        result = validator.validate(sample_interpretation_valid)
        assert 0.0 <= result["quality_score"] <= 1.0

    def test_quality_issues_is_list(self, validator, sample_interpretation_valid):
        """quality_issues is always a list."""
        result = validator.validate(sample_interpretation_valid)
        assert isinstance(result["quality_issues"], list)


class TestEmptyText:
    """Tests for empty/None text handling."""

    def test_empty_string_is_invalid(self, validator, sample_empty_text):
        """Empty string is not valid."""
        result = validator.validate(sample_empty_text)
        assert result["is_valid"] is False

    def test_empty_string_score_is_zero(self, validator, sample_empty_text):
        """Empty string gets quality score of 0.0."""
        result = validator.validate(sample_empty_text)
        assert result["quality_score"] == 0.0

    def test_empty_string_word_count_is_zero(self, validator, sample_empty_text):
        """Empty string has word count of 0."""
        result = validator.validate(sample_empty_text)
        assert result["word_count"] == 0

    def test_empty_string_has_issue(self, validator, sample_empty_text):
        """Empty string has 'empty_text' in quality_issues."""
        result = validator.validate(sample_empty_text)
        assert any("empty_text" in issue for issue in result["quality_issues"])

    def test_whitespace_only_is_invalid(self, validator):
        """Whitespace-only text is not valid."""
        result = validator.validate("   \n\n\t   ")
        assert result["is_valid"] is False


class TestValidInterpretation:
    """Tests for valid interpretations."""

    def test_valid_interpretation_passes(
        self, validator, sample_interpretation_valid
    ):
        """A valid interpretation passes validation."""
        result = validator.validate(
            sample_interpretation_valid, text_type="interpretation"
        )
        assert result["is_valid"] is True

    def test_valid_interpretation_no_issues(
        self, validator, sample_interpretation_valid
    ):
        """A valid interpretation has no quality issues."""
        result = validator.validate(
            sample_interpretation_valid, text_type="interpretation"
        )
        assert len(result["quality_issues"]) == 0

    def test_valid_interpretation_high_score(
        self, validator, sample_interpretation_valid
    ):
        """A valid interpretation has quality score >= threshold."""
        result = validator.validate(
            sample_interpretation_valid, text_type="interpretation"
        )
        # threshold is 0.5 from config
        assert result["quality_score"] >= 0.5

    def test_valid_passage_passes(self, validator, sample_passage_valid):
        """A valid passage passes validation."""
        result = validator.validate(
            sample_passage_valid, text_type="passage"
        )
        assert result["is_valid"] is True


class TestWordCountValidation:
    """Tests for word count threshold validation."""

    def test_too_short_is_invalid(self, validator):
        """Text below min_words threshold is invalid."""
        # min_words is 10 in mock_config
        short_text = "Too short text here."
        result = validator.validate(short_text, text_type="interpretation")
        assert result["is_valid"] is False

    def test_too_short_has_issue(self, validator):
        """Too short text has 'too_short' in quality_issues."""
        short_text = "Too short."
        result = validator.validate(short_text, text_type="interpretation")
        assert any("too_short" in issue for issue in result["quality_issues"])

    def test_too_long_is_invalid(self, validator):
        """Text above max_words threshold is invalid."""
        # max_words is 600 in mock_config
        long_text = " ".join(["word"] * 650)
        result = validator.validate(long_text, text_type="interpretation")
        assert result["is_valid"] is False

    def test_too_long_has_issue(self, validator):
        """Too long text has 'too_long' in quality_issues."""
        long_text = " ".join(["word"] * 650)
        result = validator.validate(long_text, text_type="interpretation")
        assert any("too_long" in issue for issue in result["quality_issues"])

    def test_word_count_correct(self, validator):
        """Word count in result matches actual word count."""
        text = "one two three four five six seven eight nine ten"
        result = validator.validate(text, text_type="interpretation")
        assert result["word_count"] == 10


class TestGibberishDetection:
    """Tests for gibberish detection."""

    def test_gibberish_is_invalid(self, validator, sample_gibberish_text):
        """Gibberish text is marked invalid."""
        result = validator.validate(
            sample_gibberish_text, text_type="interpretation"
        )
        assert result["is_valid"] is False

    def test_gibberish_has_issue(self, validator, sample_gibberish_text):
        """Gibberish text has 'gibberish' in quality_issues."""
        result = validator.validate(
            sample_gibberish_text, text_type="interpretation"
        )
        assert any("gibberish" in issue for issue in result["quality_issues"])

    def test_real_english_not_gibberish(
        self, validator, sample_interpretation_valid
    ):
        """Real English text is not flagged as gibberish."""
        result = validator.validate(
            sample_interpretation_valid, text_type="interpretation"
        )
        assert not any(
            "gibberish" in issue for issue in result["quality_issues"]
        )


class TestTextTypes:
    """Tests for interpretation vs passage text type thresholds."""

    def test_passage_uses_different_thresholds(self, validator):
        """Passage validation uses passage-specific thresholds."""
        # a text that passes interpretation min (10 words) but
        # fails passage min (20 words)
        # using 15 words - clearly above 10 but below 20
        # long enough for langdetect to identify as English
        text = (
            "Victor ran away from his creation immediately after it opened "
            "its dull yellow eyes and breathed."
        )
        interp_result = validator.validate(text, text_type="interpretation")
        passage_result = validator.validate(text, text_type="passage")

        # interpretation should pass (>10 words, clearly English)
        # passage should fail (<20 words)
        assert interp_result["is_valid"] is True
        assert passage_result["is_valid"] is False

    def test_default_type_is_interpretation(self, validator):
        """Default text_type is 'interpretation'."""
        text = " ".join(["word"] * 15)
        result_default = validator.validate(text)
        result_explicit = validator.validate(text, text_type="interpretation")
        assert result_default["is_valid"] == result_explicit["is_valid"]


class TestQualityScore:
    """Tests for quality score calculation."""

    def test_score_decreases_with_issues(self, validator):
        """Quality score is lower when issues are found."""
        valid_result = validator.validate(
            "word " * 15,  # just enough words
            text_type="interpretation"
        )
        short_result = validator.validate(
            "too short",
            text_type="interpretation"
        )
        assert short_result["quality_score"] < valid_result["quality_score"]

    def test_score_never_below_zero(self, validator, sample_gibberish_text):
        """Quality score never goes below 0.0."""
        result = validator.validate(
            sample_gibberish_text, text_type="interpretation"
        )
        assert result["quality_score"] >= 0.0

    def test_score_never_above_one(
        self, validator, sample_interpretation_valid
    ):
        """Quality score never goes above 1.0."""
        result = validator.validate(
            sample_interpretation_valid, text_type="interpretation"
        )
        assert result["quality_score"] <= 1.0


class TestValidateBatch:
    """Tests for validate_batch() method."""

    def test_batch_returns_list(self, validator, sample_interpretation_valid):
        """validate_batch returns a list."""
        texts = [sample_interpretation_valid, sample_interpretation_valid]
        result = validator.validate_batch(texts)
        assert isinstance(result, list)

    def test_batch_length_matches_input(
        self, validator, sample_interpretation_valid, sample_empty_text
    ):
        """Batch result length matches input length."""
        texts = [
            sample_interpretation_valid,
            sample_empty_text,
            sample_interpretation_valid
        ]
        result = validator.validate_batch(texts)
        assert len(result) == 3

    def test_batch_each_item_is_dict(
        self, validator, sample_interpretation_valid
    ):
        """Each item in batch result is a dict."""
        texts = [sample_interpretation_valid, sample_interpretation_valid]
        result = validator.validate_batch(texts)
        for item in result:
            assert isinstance(item, dict)