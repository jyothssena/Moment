"""
api/metrics.py — Prometheus Metrics for Momento API
=====================================================
Exposes /metrics endpoint with:
  - Request counts by endpoint and status
  - Request latency histograms
  - Model confidence distribution
  - Decomposition and compatibility call counts
  - Error counts by type
  - Active model mode (stub vs real)

Usage in main.py:
    from api.metrics import setup_metrics, track_request, track_model_call
"""

import time
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ── Request-level metrics ─────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "momento_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "momento_http_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# ── Model-level metrics ──────────────────────────────────────────────────────

MODEL_CALLS = Counter(
    "momento_model_calls_total",
    "Total model pipeline calls",
    ["pipeline"],  # decompose, compatibility, batch
)

MODEL_ERRORS = Counter(
    "momento_model_errors_total",
    "Model call errors",
    ["pipeline", "error_type"],
)

CONFIDENCE_HISTOGRAM = Histogram(
    "momento_confidence_score",
    "Distribution of compatibility confidence scores",
    buckets=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
)

DOMINANT_VERDICT = Counter(
    "momento_verdict_total",
    "Count of dominant_think verdicts",
    ["verdict"],  # resonate, contradict, diverge
)

# ── System metrics ────────────────────────────────────────────────────────────

MODEL_INFO = Info(
    "momento_model",
    "Model configuration info",
)

ACTIVE_REQUESTS = Gauge(
    "momento_active_requests",
    "Currently in-flight requests",
)


# ── Helper functions called from main.py ──────────────────────────────────────

def setup_metrics(model_mode: str, git_sha: str):
    """Call once at startup to set static model info."""
    MODEL_INFO.info({
        "mode": model_mode,
        "git_sha": git_sha,
        "model_name": "gemini-2.5-flash",
    })


def track_request(method: str, endpoint: str, status: int, duration: float):
    """Track an HTTP request."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def track_model_call(pipeline: str):
    """Track a successful model pipeline call."""
    MODEL_CALLS.labels(pipeline=pipeline).inc()


def track_model_error(pipeline: str, error_type: str):
    """Track a model pipeline error."""
    MODEL_ERRORS.labels(pipeline=pipeline, error_type=error_type).inc()


def track_confidence(score: float):
    """Track a confidence score from compatibility results."""
    CONFIDENCE_HISTOGRAM.observe(score)


def track_verdict(verdict: str):
    """Track a dominant_think verdict."""
    if verdict in ("resonate", "contradict", "diverge"):
        DOMINANT_VERDICT.labels(verdict=verdict).inc()


def get_metrics_response():
    """Generate Prometheus metrics text for /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
