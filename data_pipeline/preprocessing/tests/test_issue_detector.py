# ============================================================
# test_issue_detector.py
# MOMENT Preprocessing Pipeline - Issue Detector Tests
# IE7374 MLOps Coursework - Group 23
#
# Run with: pytest tests/test_issue_detector.py -v
# ============================================================

import pytest # type: ignore
from data_pipeline.preprocessing.issue_detector import IssueDetector


@pytest.fixture
def detector(mock_config):
    """Create an IssueDetector instance for each test."""
    return IssueDetector(mock_config)


class TestDetectBasic:
    """Basic detect() method tests."""

    def test_returns_dict(self, detector, sample_interpretation_valid):
        """detect() returns a dict."""
        result = detector.detect(sample_interpretation_valid)
        assert isinstance(result, dict)

    def test_returns_required_keys(self, detector, sample_interpretation_valid):
        """Result contains all required keys."""
        result = detector.detect(sample_interpretation_valid)
        assert "has_pii" in result
        assert "pii_types" in result
        assert "has_profanity" in result
        assert "profanity_ratio" in result
        assert "is_spam" in result
        assert "spam_reasons" in result

    def test_empty_text_returns_clean(self, detector, sample_empty_text):
        """Empty text returns all-false result."""
        result = detector.detect(sample_empty_text)
        assert result["has_pii"] is False
        assert result["has_profanity"] is False
        assert result["is_spam"] is False

    def test_clean_text_has_no_issues(
        self, detector, sample_interpretation_valid
    ):
        """Clean interpretation has no detected issues."""
        result = detector.detect(sample_interpretation_valid)
        assert result["has_pii"] is False
        assert result["has_profanity"] is False
        assert result["is_spam"] is False


class TestPIIDetection:
    """Tests for PII detection."""

    def test_email_detected(self, detector, sample_interpretation_with_email):
        """Email address is detected as PII."""
        result = detector.detect(sample_interpretation_with_email)
        assert result["has_pii"] is True
        assert "email" in result["pii_types"]

    def test_phone_number_detected(self, detector):
        """US phone number is detected as PII."""
        text = (
            "Call me at (555) 123-4567 to discuss this analysis. "
            "The passage shows how Victor responds to his creation. "
            "His immediate rejection is the key theme here."
        )
        result = detector.detect(text)
        assert result["has_pii"] is True
        assert "phone_number" in result["pii_types"]

    def test_ssn_detected(self, detector):
        """SSN pattern is detected as PII."""
        text = (
            "My SSN is 123-45-6789 and I think Victor is irresponsible. "
            "The creature deserved better treatment from its creator. "
            "This passage shows the origin of the monster narrative."
        )
        result = detector.detect(text)
        assert result["has_pii"] is True
        assert "ssn" in result["pii_types"]

    def test_pii_types_is_list(self, detector, sample_interpretation_valid):
        """pii_types is always a list."""
        result = detector.detect(sample_interpretation_valid)
        assert isinstance(result["pii_types"], list)

    def test_no_pii_types_empty(self, detector, sample_interpretation_valid):
        """pii_types is empty when no PII found."""
        result = detector.detect(sample_interpretation_valid)
        assert result["pii_types"] == []

    def test_text_without_pii_clean(self, detector):
        """Text without PII is not flagged."""
        text = (
            "Victor's reaction to the creature is fascinating. "
            "He built something beautiful and was horrified by it. "
            "The yellow eye detail is the most striking image. "
            "His prejudice precedes any action by the creature."
        )
        result = detector.detect(text)
        assert result["has_pii"] is False


class TestProfanityDetection:
    """Tests for profanity detection."""

    def test_high_profanity_ratio_detected(self, detector):
        """Text with high profanity ratio is flagged."""
        # over 30% profanity ratio threshold
        text = "damn this shit is crap hell what the damn crap"
        result = detector.detect(text)
        assert result["has_profanity"] is True

    def test_single_mild_word_not_flagged(self, detector):
        """Single mild word below ratio threshold is not flagged."""
        # one word out of many = well below 30% threshold
        text = (
            "This is a damn shame that Victor abandoned his creation. "
            "The creature only wanted love and acceptance from its maker. "
            "Victor's pride and fear drove him to reject his own work. "
            "The passage shows how prejudice precedes any harmful action."
        )
        result = detector.detect(text)
        # one word / ~50 words = 2% ratio, below 30% threshold
        assert result["has_profanity"] is False

    def test_profanity_ratio_is_float(
        self, detector, sample_interpretation_valid
    ):
        """profanity_ratio is always a float."""
        result = detector.detect(sample_interpretation_valid)
        assert isinstance(result["profanity_ratio"], float)

    def test_clean_text_zero_ratio(
        self, detector, sample_interpretation_valid
    ):
        """Clean text has profanity ratio of 0.0."""
        result = detector.detect(sample_interpretation_valid)
        assert result["profanity_ratio"] == 0.0


class TestSpamDetection:
    """Tests for spam pattern detection."""

    def test_excessive_caps_detected(
        self, detector, sample_interpretation_excessive_caps
    ):
        """Excessive capitalization is detected as spam."""
        result = detector.detect(sample_interpretation_excessive_caps)
        assert result["is_spam"] is True
        assert any(
            "excessive_caps" in reason
            for reason in result["spam_reasons"]
        )

    def test_repetitive_chars_detected(self, detector):
        """Repetitive characters (4+) are detected as spam."""
        text = (
            "This is sooooo amazing and the passage is greatttt. "
            "I loved reading about how Victor reacts to his creation. "
            "The yellow eye detail is really striking and memorable."
        )
        result = detector.detect(text)
        assert result["is_spam"] is True
        assert any(
            "repetitive_chars" in reason
            for reason in result["spam_reasons"]
        )

    def test_repetitive_words_detected(self, detector):
        """Repetitive words (>30%) are detected as spam."""
        # "the" repeated many times
        text = "the the the the the book the the the the passage the the"
        result = detector.detect(text)
        assert result["is_spam"] is True
        assert any(
            "repetitive_words" in reason
            for reason in result["spam_reasons"]
        )

    def test_spam_reasons_is_list(
        self, detector, sample_interpretation_valid
    ):
        """spam_reasons is always a list."""
        result = detector.detect(sample_interpretation_valid)
        assert isinstance(result["spam_reasons"], list)

    def test_clean_text_no_spam_reasons(
        self, detector, sample_interpretation_valid
    ):
        """Clean text has empty spam_reasons."""
        result = detector.detect(sample_interpretation_valid)
        assert result["spam_reasons"] == []

    def test_normal_caps_not_flagged(self, detector):
        """Normal capitalization is not flagged."""
        text = (
            "Victor's reaction reveals his deep prejudice. "
            "The creature is innocent at this point in the story. "
            "Shelley uses Gothic atmosphere to emphasize the horror. "
            "The yellow eye becomes a symbol of Victor's rejection."
        )
        result = detector.detect(text)
        # normal text should not be flagged for caps
        assert not any(
            "excessive_caps" in reason
            for reason in result["spam_reasons"]
        )


class TestDetectBatch:
    """Tests for detect_batch() method."""

    def test_batch_returns_list(self, detector, sample_interpretation_valid):
        """detect_batch returns a list."""
        texts = [sample_interpretation_valid, sample_interpretation_valid]
        result = detector.detect_batch(texts)
        assert isinstance(result, list)

    def test_batch_length_matches_input(
        self, detector,
        sample_interpretation_valid,
        sample_interpretation_with_email
    ):
        """Batch result length matches input length."""
        texts = [
            sample_interpretation_valid,
            sample_interpretation_with_email,
            sample_interpretation_valid
        ]
        result = detector.detect_batch(texts)
        assert len(result) == 3

    def test_batch_detects_issues_correctly(
        self, detector,
        sample_interpretation_valid,
        sample_interpretation_with_email
    ):
        """Batch correctly identifies which texts have issues."""
        texts = [
            sample_interpretation_valid,      # no issues
            sample_interpretation_with_email  # has PII
        ]
        result = detector.detect_batch(texts)
        assert result[0]["has_pii"] is False
        assert result[1]["has_pii"] is True