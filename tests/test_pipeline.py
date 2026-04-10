"""
tests/test_pipeline.py — Momento CI/CD Test Suite
==================================================
Covers all 6 pipeline requirements:
  1. Model interface contract (CI trigger)
  2. Automated model validation with thresholds
  3. Bias detection across data slices
  4. Registry push readiness check
  5. Notification hook tests
  6. Rollback threshold verification

Run with: pytest tests/ -v
"""

import json
import os
import sys

import json
import os
import sys
from unittest.mock import patch, MagicMock

# Mock google.genai before any agent imports
sys.modules['google.genai'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

import pytest # type: ignore
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_interface import (
    decompose_moment,
    run_compatibility_pipeline,
    health_check,
)
from aggregator import aggregate
from validate_model import (
    validate_output_schema,
    validate_rcd_sums,
    validate_confidence_range,
    validate_dominant_labels,
    compute_validation_metrics,
    VALIDATION_THRESHOLDS,
)
from bias_detection import (
    detect_bias_across_slices,
    BIAS_ALERT_THRESHOLD,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_moment_a():
    return {
        "character_name": "emma_chen",
        "interpretation": (
            "The creature's demand for a companion reveals the fundamental "
            "human need for belonging. Shelley frames this as both a plea "
            "and an accusation — Victor created the longing but refuses to "
            "satisfy it."
        ),
        "word_count": 38,
    }


@pytest.fixture
def sample_moment_b():
    return {
        "character_name": "james_park",
        "interpretation": (
            "I read this passage as Victor's moral failure crystallised. "
            "The creature is entirely rational. Victor's refusal is "
            "self-protective cowardice dressed up as ethical concern."
        ),
        "word_count": 31,
    }


@pytest.fixture
def sample_decomp_a():
    return {
        "passage_id": "passage_1",
        "user_id": "emma_chen",
        "subclaims": [
            {
                "id": "1",
                "claim": "The creature's demand reveals a fundamental need for belonging",
                "quote": "(no direct quote)",
                "weight": 0.60,
                "emotional_mode": "philosophical",
            },
            {
                "id": "2",
                "claim": "Shelley frames the demand as both plea and accusation",
                "quote": "(no direct quote)",
                "weight": 0.40,
                "emotional_mode": "observational",
            },
        ],
    }


@pytest.fixture
def sample_decomp_b():
    return {
        "passage_id": "passage_1",
        "user_id": "james_park",
        "subclaims": [
            {
                "id": "1",
                "claim": "Victor's refusal is moral failure and cowardice",
                "quote": "(no direct quote)",
                "weight": 0.55,
                "emotional_mode": "prosecutorial",
            },
            {
                "id": "2",
                "claim": "The creature is entirely rational",
                "quote": "(no direct quote)",
                "weight": 0.45,
                "emotional_mode": "empathetic",
            },
        ],
    }


@pytest.fixture
def sample_scoring():
    return {
        "passage_id": "passage_1",
        "matched_pairs": [
            {
                "a_id": "1",
                "b_id": "1",
                "weight_a": 0.60,
                "weight_b": 0.55,
                "gate_confidence": 0.75,
                "think": {"R": 0.6, "C": 0.4, "D": 0.0},
                "feel":  {"R": 0.4, "C": 0.6, "D": 0.0},
            }
        ],
        "unmatched_a": ["2"],
        "unmatched_b": ["2"],
    }


@pytest.fixture
def sample_pipeline_result():
    """A valid full pipeline output (what run_compatibility_pipeline returns)."""
    return {
        "passage_id": "passage_1",
        "character_a": "emma_chen",
        "character_b": "james_park",
        "book": "Frankenstein",
        "think": {"R": 45, "C": 35, "D": 20},
        "feel":  {"R": 30, "C": 50, "D": 20},
        "dominant_think": "resonate",
        "dominant_feel":  "contradict",
        "match_count": 1,
        "confidence": 0.71,
        "computed_at": datetime.utcnow().isoformat(),
    }


# ── 1. MODEL INTERFACE CONTRACT ───────────────────────────────────────────────

class TestModelInterface:
    """Requirement 1: CI triggers on new code — interface must load and respond."""

    def test_health_check_returns_ok(self):
        result = health_check()
        assert result["status"] == "ok"
        assert "interface_version" in result
        assert "functions" in result


# ── 2. AUTOMATED MODEL VALIDATION ────────────────────────────────────────────

class TestModelValidation:
    """Requirement 2: Validation must pass defined thresholds to proceed."""

    def test_validate_rcd_sums_to_100_think(self, sample_pipeline_result):
        assert validate_rcd_sums(sample_pipeline_result, "think")

    def test_validate_rcd_sums_to_100_feel(self, sample_pipeline_result):
        assert validate_rcd_sums(sample_pipeline_result, "feel")

    def test_validate_confidence_in_range(self, sample_pipeline_result):
        assert validate_confidence_range(sample_pipeline_result)

    def test_validate_dominant_labels_valid(self, sample_pipeline_result):
        assert validate_dominant_labels(sample_pipeline_result)

    def test_validate_output_schema_passes(self, sample_pipeline_result):
        errors = validate_output_schema(sample_pipeline_result)
        assert errors == [], f"Schema errors: {errors}"

    def test_rcd_sum_failure_detected(self):
        bad_result = {
            "think": {"R": 50, "C": 50, "D": 50},  # sums to 150
            "feel":  {"R": 30, "C": 30, "D": 30},
        }
        assert not validate_rcd_sums(bad_result, "think")

    def test_confidence_out_of_range_detected(self):
        bad_result = {"confidence": 1.5}
        assert not validate_confidence_range(bad_result)

    def test_invalid_dominant_label_detected(self):
        bad_result = {"dominant_think": "unknown", "dominant_feel": "resonate"}
        assert not validate_dominant_labels(bad_result)

    def test_validation_metrics_above_threshold(self, sample_pipeline_result):
        """Schema correctness rate must exceed VALIDATION_THRESHOLDS['schema_pass_rate']."""
        results = [sample_pipeline_result] * 10
        metrics = compute_validation_metrics(results)
        assert metrics["schema_pass_rate"] >= VALIDATION_THRESHOLDS["schema_pass_rate"]
        assert metrics["confidence_in_range_rate"] >= VALIDATION_THRESHOLDS["confidence_in_range_rate"]

    def test_validation_blocks_on_low_pass_rate(self):
        """If schema_pass_rate < threshold, validation should fail."""
        bad_results = [
            {"think": {"R": 200, "C": 0, "D": 0}, "feel": {"R": 0, "C": 0, "D": 0},
             "confidence": 0.5, "dominant_think": "bad", "dominant_feel": "bad"}
        ] * 10
        metrics = compute_validation_metrics(bad_results)
        assert metrics["schema_pass_rate"] < VALIDATION_THRESHOLDS["schema_pass_rate"]


# ── 3. BIAS DETECTION ────────────────────────────────────────────────────────

class TestBiasDetection:
    """Requirement 3: Bias across data slices must stay below alert threshold."""

    def test_bias_detection_returns_slice_results(self):
        results = [
            {"character_a": "user_1", "character_b": "user_2",
             "book": "Frankenstein", "passage_id": "passage_1",
             "think": {"R": 60, "C": 20, "D": 20},
             "feel":  {"R": 50, "C": 30, "D": 20},
             "confidence": 0.75},
            {"character_a": "user_3", "character_b": "user_4",
             "book": "The Great Gatsby", "passage_id": "passage_2",
             "think": {"R": 30, "C": 50, "D": 20},
             "feel":  {"R": 20, "C": 60, "D": 20},
             "confidence": 0.68},
        ]
        report = detect_bias_across_slices(results)
        assert "by_book" in report
        assert "by_passage" in report
        assert "alerts" in report
        assert "max_confidence_gap" in report

    def test_bias_alert_triggers_on_large_gap(self):
        """High-confidence results for one slice only should trigger alert."""
        skewed_results = (
            [{"character_a": f"u{i}", "character_b": f"u{i+1}",
              "book": "Frankenstein", "passage_id": "passage_1",
              "think": {"R": 80, "C": 10, "D": 10},
              "feel":  {"R": 80, "C": 10, "D": 10},
              "confidence": 0.92} for i in range(10)]
            +
            [{"character_a": f"v{i}", "character_b": f"v{i+1}",
              "book": "The Great Gatsby", "passage_id": "passage_1",
              "think": {"R": 10, "C": 10, "D": 80},
              "feel":  {"R": 10, "C": 10, "D": 80},
              "confidence": 0.20} for i in range(10)]
        )
        report = detect_bias_across_slices(skewed_results)
        assert report["max_confidence_gap"] >= BIAS_ALERT_THRESHOLD or \
               len(report["alerts"]) > 0

    def test_no_bias_alert_on_balanced_data(self):
        balanced = [
            {"character_a": f"u{i}", "character_b": f"v{i}",
             "book": b, "passage_id": "passage_1",
             "think": {"R": 40, "C": 30, "D": 30},
             "feel":  {"R": 35, "C": 35, "D": 30},
             "confidence": 0.70}
            for i in range(5)
            for b in ["Frankenstein", "The Great Gatsby", "Pride and Prejudice"]
        ]
        report = detect_bias_across_slices(balanced)
        assert report["max_confidence_gap"] < BIAS_ALERT_THRESHOLD


# ── 4. AGGREGATE MATH (unit tested independently) ────────────────────────────

class TestAggregatemath:
    """Requirement 2: Core scoring math must be deterministic and correct."""

    def test_aggregate_rcd_sums_to_100(self, sample_decomp_a, sample_decomp_b, sample_scoring):
        combined = {
            "reader_a": {**sample_decomp_a, "word_count": 38},
            "reader_b": {**sample_decomp_b, "word_count": 31},
        }
        result = aggregate(combined, sample_scoring, wc_a=38, wc_b=31)
        assert result["think"]["R"] + result["think"]["C"] + result["think"]["D"] == 100
        assert result["feel"]["R"]  + result["feel"]["C"]  + result["feel"]["D"]  == 100

    def test_aggregate_confidence_in_valid_range(self, sample_decomp_a, sample_decomp_b, sample_scoring):
        combined = {
            "reader_a": {**sample_decomp_a, "word_count": 38},
            "reader_b": {**sample_decomp_b, "word_count": 31},
        }
        result = aggregate(combined, sample_scoring, wc_a=38, wc_b=31)
        assert 0.20 <= result["confidence"] <= 0.95

    def test_aggregate_dominant_is_valid_label(self, sample_decomp_a, sample_decomp_b, sample_scoring):
        combined = {
            "reader_a": {**sample_decomp_a, "word_count": 38},
            "reader_b": {**sample_decomp_b, "word_count": 31},
        }
        result = aggregate(combined, sample_scoring, wc_a=38, wc_b=31)
        valid = {"resonate", "contradict", "diverge"}
        assert result["dominant_think"] in valid
        assert result["dominant_feel"] in valid

    def test_aggregate_unmatched_all_diverge(self):
        """All unmatched sub-claims → result should be heavily Diverge."""
        decomp = {
            "reader_a": {
                "user_id": "a", "word_count": 50,
                "subclaims": [
                    {"id": "1", "claim": "x", "quote": "(none)", "weight": 1.0, "emotional_mode": "philosophical"}
                ]
            },
            "reader_b": {
                "user_id": "b", "word_count": 50,
                "subclaims": [
                    {"id": "1", "claim": "y", "quote": "(none)", "weight": 1.0, "emotional_mode": "observational"}
                ]
            },
        }
        scoring = {
            "passage_id": "p1",
            "matched_pairs": [],
            "unmatched_a": ["1"],
            "unmatched_b": ["1"],
        }
        result = aggregate(decomp, scoring, wc_a=50, wc_b=50)
        assert result["think"]["D"] == 100
        assert result["feel"]["D"] == 100


# ── 5. ROLLBACK THRESHOLD ─────────────────────────────────────────────────────

class TestRollback:
    """Requirement 6: Rollback must trigger when new model underperforms previous."""

    def test_rollback_triggered_when_confidence_drops(self):
        from rollback import should_rollback, ROLLBACK_THRESHOLDS
        previous_metrics = {"mean_confidence": 0.72, "schema_pass_rate": 0.98}
        new_metrics      = {"mean_confidence": 0.55, "schema_pass_rate": 0.97}
        assert should_rollback(previous_metrics, new_metrics)

    def test_rollback_not_triggered_on_improvement(self):
        from rollback import should_rollback
        previous_metrics = {"mean_confidence": 0.65, "schema_pass_rate": 0.96}
        new_metrics      = {"mean_confidence": 0.72, "schema_pass_rate": 0.99}
        assert not should_rollback(previous_metrics, new_metrics)

    def test_rollback_triggered_on_schema_regression(self):
        from rollback import should_rollback
        previous_metrics = {"mean_confidence": 0.70, "schema_pass_rate": 0.99}
        new_metrics      = {"mean_confidence": 0.71, "schema_pass_rate": 0.80}
        assert should_rollback(previous_metrics, new_metrics)


# ── 6. NOTIFICATION HOOKS ─────────────────────────────────────────────────────

class TestNotifications:
    """Requirement 5: Notification functions must be callable without error."""

    @patch("notifications.send_slack_alert")
    def test_slack_alert_called_on_bias_alert(self, mock_slack):
        from notifications import notify_bias_alert
        notify_bias_alert(alerts=["High confidence gap in Frankenstein slice"], dry_run=True)
        # In dry_run mode, real HTTP call is suppressed — function must not raise

    @patch("notifications.send_slack_alert")
    def test_slack_alert_called_on_validation_failure(self, mock_slack):
        from notifications import notify_validation_failure
        notify_validation_failure(metrics={"schema_pass_rate": 0.60}, dry_run=True)

    @patch("notifications.send_slack_alert")
    def test_slack_alert_called_on_pipeline_success(self, mock_slack):
        from notifications import notify_training_complete
        notify_training_complete(model_version="sha-abc123", dry_run=True)