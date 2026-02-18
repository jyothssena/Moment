# ============================================================
# test_metrics_calculator.py
# MOMENT Preprocessing Pipeline - Metrics Calculator Tests
# IE7374 MLOps Coursework - Group 23
#
# Run with: pytest tests/test_metrics_calculator.py -v
# ============================================================

import pytest # type: ignore
from data_pipeline.preprocessing.metrics_calculator import MetricsCalculator


@pytest.fixture
def calculator(mock_config):
    """Create a MetricsCalculator instance for each test."""
    return MetricsCalculator(mock_config)


class TestCalculateBasic:
    """Basic calculate() method tests."""

    def test_returns_dict(self, calculator, sample_interpretation_valid):
        """calculate() returns a dict."""
        result = calculator.calculate(sample_interpretation_valid)
        assert isinstance(result, dict)

    def test_returns_required_keys(
        self, calculator, sample_interpretation_valid
    ):
        """Result contains all required metric keys."""
        result = calculator.calculate(sample_interpretation_valid)
        assert "word_count" in result
        assert "char_count" in result
        assert "sentence_count" in result
        assert "avg_word_length" in result
        assert "avg_sentence_length" in result
        assert "readability_score" in result

    def test_empty_text_returns_zeros(self, calculator, sample_empty_text):
        """Empty text returns all zero metrics."""
        result = calculator.calculate(sample_empty_text)
        assert result["word_count"] == 0
        assert result["char_count"] == 0
        assert result["sentence_count"] == 0
        assert result["avg_word_length"] == 0.0
        assert result["avg_sentence_length"] == 0.0
        assert result["readability_score"] == 0.0

    def test_none_returns_zeros(self, calculator):
        """None input returns all zero metrics."""
        result = calculator.calculate(None)
        assert result["word_count"] == 0


class TestWordCount:
    """Tests for word count calculation."""

    def test_simple_word_count(self, calculator):
        """Word count is correct for simple text."""
        text = "one two three four five"
        result = calculator.calculate(text)
        assert result["word_count"] == 5

    def test_word_count_with_punctuation(self, calculator):
        """Word count handles punctuation correctly."""
        text = "Hello, world! How are you?"
        result = calculator.calculate(text)
        assert result["word_count"] == 5

    def test_word_count_multiline(self, calculator):
        """Word count handles multiline text."""
        text = "line one\nline two\nline three"
        result = calculator.calculate(text)
        assert result["word_count"] == 6

    def test_word_count_matches_raw(self, calculator):
        """Word count matches simple split() count."""
        text = "The creature opened its eyes and Victor ran away."
        result = calculator.calculate(text)
        assert result["word_count"] == len(text.split())


class TestCharCount:
    """Tests for character count calculation."""

    def test_char_count_excludes_spaces(self, calculator):
        """Character count excludes whitespace."""
        text = "hello world"
        result = calculator.calculate(text)
        # "helloworld" = 10 chars without space
        assert result["char_count"] == 10

    def test_char_count_positive(self, calculator, sample_interpretation_valid):
        """Character count is positive for valid text."""
        result = calculator.calculate(sample_interpretation_valid)
        assert result["char_count"] > 0

    def test_char_count_less_than_total_length(
        self, calculator, sample_interpretation_valid
    ):
        """Char count (no spaces) < total text length."""
        result = calculator.calculate(sample_interpretation_valid)
        assert result["char_count"] < len(sample_interpretation_valid)


class TestSentenceCount:
    """Tests for sentence count calculation."""

    def test_single_sentence(self, calculator):
        """Single sentence is counted correctly."""
        text = "This is one sentence with enough words in it."
        result = calculator.calculate(text)
        assert result["sentence_count"] >= 1

    def test_multiple_sentences(self, calculator):
        """Multiple sentences are counted."""
        text = "First sentence. Second sentence. Third sentence."
        result = calculator.calculate(text)
        assert result["sentence_count"] >= 2

    def test_sentence_count_positive(
        self, calculator, sample_interpretation_valid
    ):
        """Sentence count is positive for valid text."""
        result = calculator.calculate(sample_interpretation_valid)
        assert result["sentence_count"] > 0


class TestAvgWordLength:
    """Tests for average word length calculation."""

    def test_avg_word_length_positive(
        self, calculator, sample_interpretation_valid
    ):
        """Average word length is positive for valid text."""
        result = calculator.calculate(sample_interpretation_valid)
        assert result["avg_word_length"] > 0.0

    def test_avg_word_length_reasonable(
        self, calculator, sample_interpretation_valid
    ):
        """Average word length is within reasonable English range."""
        result = calculator.calculate(sample_interpretation_valid)
        # average English word length is 4-6 characters
        assert 2.0 <= result["avg_word_length"] <= 12.0

    def test_avg_word_length_is_float(
        self, calculator, sample_interpretation_valid
    ):
        """Average word length is a float."""
        result = calculator.calculate(sample_interpretation_valid)
        assert isinstance(result["avg_word_length"], float)


class TestReadabilityScore:
    """Tests for Flesch Reading Ease score."""

    def test_readability_in_range(
        self, calculator, sample_interpretation_valid
    ):
        """Readability score is between 0.0 and 100.0."""
        result = calculator.calculate(sample_interpretation_valid)
        assert 0.0 <= result["readability_score"] <= 100.0

    def test_readability_is_float(
        self, calculator, sample_interpretation_valid
    ):
        """Readability score is a float."""
        result = calculator.calculate(sample_interpretation_valid)
        assert isinstance(result["readability_score"], float)

    def test_simple_text_higher_readability(self, calculator):
        """Simple text has higher readability than complex text."""
        simple = "The cat sat on the mat. It was a big cat."
        complex_text = (
            "The epistemological ramifications of anthropomorphic "
            "attribution in contemporary phenomenological discourse "
            "necessitate comprehensive philosophical reexamination."
        )
        simple_result = calculator.calculate(simple)
        complex_result = calculator.calculate(complex_text)
        assert (
            simple_result["readability_score"] >
            complex_result["readability_score"]
        )

    def test_empty_text_zero_readability(
        self, calculator, sample_empty_text
    ):
        """Empty text has readability score of 0.0."""
        result = calculator.calculate(sample_empty_text)
        assert result["readability_score"] == 0.0


class TestGetDatasetStats:
    """Tests for get_dataset_stats() method."""

    def test_returns_dict(self, calculator, sample_metrics_dataset):
        """get_dataset_stats returns a dict."""
        result = calculator.get_dataset_stats(sample_metrics_dataset)
        assert isinstance(result, dict)

    def test_contains_word_count_stats(
        self, calculator, sample_metrics_dataset
    ):
        """Stats contain word_count metrics."""
        result = calculator.get_dataset_stats(sample_metrics_dataset)
        assert "word_count" in result

    def test_stats_contain_required_keys(
        self, calculator, sample_metrics_dataset
    ):
        """Each metric's stats contain required statistical measures."""
        result = calculator.get_dataset_stats(sample_metrics_dataset)
        for metric_stats in result.values():
            assert "mean" in metric_stats
            assert "std" in metric_stats
            assert "min" in metric_stats
            assert "max" in metric_stats
            assert "q1" in metric_stats
            assert "q3" in metric_stats
            assert "iqr" in metric_stats

    def test_iqr_calculated_correctly(
        self, calculator, sample_metrics_dataset
    ):
        """IQR = Q3 - Q1."""
        result = calculator.get_dataset_stats(sample_metrics_dataset)
        wc_stats = result["word_count"]
        expected_iqr = wc_stats["q3"] - wc_stats["q1"]
        assert abs(wc_stats["iqr"] - expected_iqr) < 0.01

    def test_empty_dataset_returns_empty(self, calculator):
        """Empty dataset returns empty dict."""
        result = calculator.get_dataset_stats([])
        assert result == {}


class TestCalculateBatch:
    """Tests for calculate_batch() method."""

    def test_batch_returns_list(
        self, calculator, sample_interpretation_valid
    ):
        """calculate_batch returns a list."""
        texts = [sample_interpretation_valid, sample_interpretation_valid]
        result = calculator.calculate_batch(texts)
        assert isinstance(result, list)

    def test_batch_length_matches_input(
        self, calculator,
        sample_interpretation_valid,
        sample_empty_text
    ):
        """Batch result length matches input length."""
        texts = [
            sample_interpretation_valid,
            sample_empty_text,
            sample_interpretation_valid
        ]
        result = calculator.calculate_batch(texts)
        assert len(result) == 3

    def test_batch_handles_empty_text(
        self, calculator,
        sample_interpretation_valid,
        sample_empty_text
    ):
        """Batch handles mix of valid and empty texts."""
        texts = [sample_interpretation_valid, sample_empty_text]
        result = calculator.calculate_batch(texts)
        assert result[0]["word_count"] > 0
        assert result[1]["word_count"] == 0