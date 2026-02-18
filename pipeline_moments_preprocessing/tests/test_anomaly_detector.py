# ============================================================
# test_anomaly_detector.py
# MOMENT Preprocessing Pipeline - Anomaly Detector Tests
# IE7374 MLOps Coursework - Group 23
#
# Run with: pytest tests/test_anomaly_detector.py -v
# ============================================================

import pytest
from data_pipeline.preprocessing.anomaly_detector import AnomalyDetector


@pytest.fixture
def detector(mock_config):
    """Create an AnomalyDetector instance for each test."""
    return AnomalyDetector(mock_config)


@pytest.fixture
def fitted_detector(mock_config, sample_metrics_dataset):
    """
    Create an AnomalyDetector that has been fitted on sample data.
    Used for tests that need baselines to be established first.
    """
    detector = AnomalyDetector(mock_config)

    # create sample texts matching metrics dataset length
    texts = [
        f"This is sample interpretation number {i} with enough words "
        f"to be meaningful and pass basic validation checks for testing."
        for i in range(len(sample_metrics_dataset))
    ]
    ids = [f"moment_{i:03d}" for i in range(len(sample_metrics_dataset))]

    detector.fit(sample_metrics_dataset, texts, ids)
    return detector


class TestInitialization:
    """Tests for AnomalyDetector initialization."""

    def test_not_fitted_initially(self, detector):
        """Detector is not fitted before fit() is called."""
        assert detector._fitted is False

    def test_cache_empty_initially(self, detector):
        """Stats cache is empty before fit()."""
        assert detector._wc_stats is None
        assert detector._read_stats is None

    def test_enabled_from_config(self, detector):
        """Detector respects enabled flag from config."""
        assert detector.enabled is True


class TestFit:
    """Tests for fit() method."""

    def test_fitted_after_fit(self, fitted_detector):
        """Detector is fitted after fit() is called."""
        assert fitted_detector._fitted is True

    def test_wc_stats_set_after_fit(self, fitted_detector):
        """Word count stats are set after fit()."""
        assert fitted_detector._wc_stats is not None

    def test_read_stats_set_after_fit(self, fitted_detector):
        """Readability stats are set after fit()."""
        assert fitted_detector._read_stats is not None

    def test_wc_stats_contain_bounds(self, fitted_detector):
        """Word count stats contain lower and upper bounds."""
        assert "lower_bound" in fitted_detector._wc_stats
        assert "upper_bound" in fitted_detector._wc_stats

    def test_wc_bounds_make_sense(self, fitted_detector):
        """Lower bound is less than upper bound."""
        assert (
            fitted_detector._wc_stats["lower_bound"] <
            fitted_detector._wc_stats["upper_bound"]
        )

    def test_read_stats_contain_mean_std(self, fitted_detector):
        """Readability stats contain mean and std."""
        assert "mean" in fitted_detector._read_stats
        assert "std" in fitted_detector._read_stats

    def test_tfidf_built_after_fit(self, fitted_detector):
        """TF-IDF matrix is built after fit()."""
        assert fitted_detector._tfidf_matrix is not None


class TestDetectWordCountOutlier:
    """Tests for word count outlier detection."""

    def test_normal_word_count_not_outlier(
        self, fitted_detector, sample_metrics_normal
    ):
        """Normal word count is not flagged."""
        result = fitted_detector.detect(
            sample_metrics_normal,
            "This is a normal interpretation with typical word count.",
            "moment_normal"
        )
        assert result["word_count_outlier"] is False

    def test_very_short_is_outlier(self, fitted_detector, sample_metrics_short):
        """Very short word count is flagged as outlier."""
        result = fitted_detector.detect(
            sample_metrics_short,
            "Short text.",
            "moment_short"
        )
        assert result["word_count_outlier"] is True

    def test_outlier_has_detail(self, fitted_detector, sample_metrics_short):
        """Word count outlier has explanation in anomaly_details."""
        result = fitted_detector.detect(
            sample_metrics_short,
            "Short text.",
            "moment_short"
        )
        assert len(result["anomaly_details"]) > 0
        assert any(
            "word_count" in detail
            for detail in result["anomaly_details"]
        )


class TestDetectReadabilityOutlier:
    """Tests for readability outlier detection."""

    def test_normal_readability_not_outlier(
        self, fitted_detector, sample_metrics_normal
    ):
        """Normal readability score is not flagged."""
        result = fitted_detector.detect(
            sample_metrics_normal,
            "Normal text with average readability.",
            "moment_normal_read"
        )
        assert result["readability_outlier"] is False

    def test_extreme_readability_is_outlier(self, fitted_detector):
        """Extremely high readability score is flagged."""
        extreme_metrics = {
            "word_count": 50,
            "readability_score": 99.9  # extreme outlier
        }
        result = fitted_detector.detect(
            extreme_metrics,
            "Very simple text.",
            "moment_extreme"
        )
        assert result["readability_outlier"] is True


class TestDetectDuplicates:
    """Tests for near-duplicate detection."""

    def test_unique_text_not_duplicate(
        self, fitted_detector, sample_metrics_normal
    ):
        """Unique text is not flagged as duplicate."""
        unique_text = (
            "This completely unique interpretation discusses themes "
            "of identity and transformation in the literary work. "
            "The author explores profound philosophical questions."
        )
        result = fitted_detector.detect(
            sample_metrics_normal,
            unique_text,
            "moment_unique_xyz"
        )
        assert result["duplicate_risk"] is False

    def test_duplicate_of_is_none_for_unique(
        self, fitted_detector, sample_metrics_normal
    ):
        """duplicate_of is None when no duplicate found."""
        result = fitted_detector.detect(
            sample_metrics_normal,
            "Completely unique text that matches nothing else.",
            "moment_no_dup"
        )
        assert result["duplicate_of"] is None


class TestDetectStyleMismatch:
    """Tests for style vs experience mismatch detection."""

    def test_new_reader_simple_writing_no_mismatch(
        self, fitted_detector,
        sample_metrics_short,
        sample_character_new_reader
    ):
        """NEW READER with simple writing has no mismatch."""
        # high readability score = simple writing
        simple_metrics = {
            "word_count": 30,
            "readability_score": 85.0  # above new_reader_ceiling of 70
        }
        result = fitted_detector.detect(
            simple_metrics,
            "Simple text by new reader.",
            "moment_new_simple",
            character=sample_character_new_reader
        )
        assert result["style_mismatch"] is False

    def test_new_reader_complex_writing_is_mismatch(
        self, fitted_detector,
        sample_character_new_reader
    ):
        """NEW READER with very complex writing is flagged."""
        # very low readability = complex writing (below well_read_floor 30)
        complex_metrics = {
            "word_count": 80,
            "readability_score": 15.0  # below well_read_floor of 30
        }
        result = fitted_detector.detect(
            complex_metrics,
            "Extremely complex philosophical text.",
            "moment_new_complex",
            character=sample_character_new_reader
        )
        assert result["style_mismatch"] is True

    def test_well_read_complex_writing_no_mismatch(
        self, fitted_detector,
        sample_character_well_read
    ):
        """Well-read reader with complex writing has no mismatch."""
        complex_metrics = {
            "word_count": 120,
            "readability_score": 35.0  # between floors - no mismatch
        }
        result = fitted_detector.detect(
            complex_metrics,
            "Complex academic text by well-read reader.",
            "moment_wellread_complex",
            character=sample_character_well_read
        )
        assert result["style_mismatch"] is False

    def test_no_character_skips_mismatch_check(
        self, fitted_detector, sample_metrics_normal
    ):
        """When no character provided, style mismatch is not checked."""
        result = fitted_detector.detect(
            sample_metrics_normal,
            "Some interpretation text.",
            "moment_no_char",
            character=None  # no character profile
        )
        assert result["style_mismatch"] is False


class TestDetectReturnStructure:
    """Tests for detect() return value structure."""

    def test_returns_dict(self, fitted_detector, sample_metrics_normal):
        """detect() returns a dict."""
        result = fitted_detector.detect(
            sample_metrics_normal,
            "Some text for testing.",
            "moment_test"
        )
        assert isinstance(result, dict)

    def test_contains_required_keys(
        self, fitted_detector, sample_metrics_normal
    ):
        """Result contains all required keys."""
        result = fitted_detector.detect(
            sample_metrics_normal,
            "Some text for testing.",
            "moment_test"
        )
        assert "word_count_outlier" in result
        assert "readability_outlier" in result
        assert "duplicate_risk" in result
        assert "duplicate_of" in result
        assert "style_mismatch" in result
        assert "anomaly_details" in result

    def test_anomaly_details_is_list(
        self, fitted_detector, sample_metrics_normal
    ):
        """anomaly_details is always a list."""
        result = fitted_detector.detect(
            sample_metrics_normal,
            "Some text for testing.",
            "moment_test"
        )
        assert isinstance(result["anomaly_details"], list)

    def test_boolean_fields_are_bool(
        self, fitted_detector, sample_metrics_normal
    ):
        """All boolean fields are actually bools."""
        result = fitted_detector.detect(
            sample_metrics_normal,
            "Some text for testing.",
            "moment_test"
        )
        assert isinstance(result["word_count_outlier"], bool)
        assert isinstance(result["readability_outlier"], bool)
        assert isinstance(result["duplicate_risk"], bool)
        assert isinstance(result["style_mismatch"], bool)


class TestDisabledDetector:
    """Tests for disabled anomaly detection."""

    def test_disabled_returns_empty(self, mock_config):
        """Disabled detector returns empty anomalies."""
        # create config with anomaly detection disabled
        disabled_config = dict(mock_config)
        disabled_config["anomaly_detection"] = {"enabled": False}

        detector = AnomalyDetector(disabled_config)
        result = detector.detect(
            {"word_count": 5, "readability_score": 85.0},
            "Short text.",
            "moment_disabled"
        )
        assert result["word_count_outlier"] is False
        assert result["readability_outlier"] is False
        assert result["duplicate_risk"] is False
        assert result["style_mismatch"] is False

    def test_detect_before_fit_returns_empty(self, detector):
        """Calling detect() before fit() returns empty anomalies."""
        result = detector.detect(
            {"word_count": 5, "readability_score": 85.0},
            "Short text.",
            "moment_unfitted"
        )
        # should not crash, returns empty
        assert result["word_count_outlier"] is False


class TestGetFitStats:
    """Tests for get_fit_stats() method."""

    def test_empty_stats_before_fit(self, detector):
        """Stats are empty before fit()."""
        result = detector.get_fit_stats()
        assert result == {}

    def test_stats_populated_after_fit(self, fitted_detector):
        """Stats are populated after fit()."""
        result = fitted_detector.get_fit_stats()
        assert result != {}
        assert result["fitted"] is True

    def test_stats_contain_tfidf_info(self, fitted_detector):
        """Stats contain TF-IDF information."""
        result = fitted_detector.get_fit_stats()
        assert "tfidf_built" in result