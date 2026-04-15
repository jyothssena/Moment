"""
metrics.py — Google Cloud Monitoring custom metrics for MOMENT.

Replaces the prometheus_client implementation. Provides the same
.inc() / .observe() / .set() / .labels() interface so every call site
in compatibility_agent.py, preprocessor_fastapi.py, run_rankings.py,
and main.py requires zero changes.

How it works:
  - All metric values accumulate in memory (thread-safe dicts).
  - A daemon thread pushes everything to Cloud Monitoring every 60 s.
  - push_metrics_now() forces an immediate push (called at pipeline end).
  - On Cloud Run, GCP auth comes from the service's attached service account.
  - Locally, auth comes from ADC (gcloud auth application-default login).
  - If credentials are unavailable the push fails silently — the pipeline
    is never affected.
"""

import os
import threading
import time
from collections import defaultdict

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "moment-486719")
_METRIC_PREFIX = "custom.googleapis.com/moment"
_project_name  = f"projects/{GOOGLE_CLOUD_PROJECT}"

# Minimum seconds between consecutive pushes (Cloud Monitoring rate limit guard).
_MIN_PUSH_INTERVAL = 30
_last_push: float  = 0.0
_push_lock         = threading.Lock()

# Registry of every metric object — collected at push time.
_REGISTRY: list = []


# ── Counter ───────────────────────────────────────────────────────────────────

class _Counter:
    """
    Monotonically increasing counter.
    Identical API to prometheus_client.Counter:
      counter.inc()
      counter.labels("value").inc(amount)
    """

    def __init__(self, name: str, description: str, label_names=()):
        self._type   = f"{_METRIC_PREFIX}/{name}"
        self._names  = tuple(label_names)
        self._vals: dict = defaultdict(float)   # label_key → total
        self._lock   = threading.Lock()
        self._labels: dict = {}
        _REGISTRY.append(self)

    def labels(self, *args, **kwargs):
        c = _Counter.__new__(_Counter)
        c._type   = self._type
        c._names  = self._names
        c._vals   = self._vals          # shared — accumulates centrally
        c._lock   = self._lock
        c._labels = dict(zip(self._names, args)) if args else dict(kwargs)
        return c

    def inc(self, amount: float = 1):
        key = tuple(sorted(self._labels.items()))
        with self._lock:
            self._vals[key] += float(amount)

    # Called by the push thread.
    def _collect(self):
        with self._lock:
            return [(self._type, dict(k), v) for k, v in self._vals.items()]


# ── Gauge ─────────────────────────────────────────────────────────────────────

class _Gauge:
    """
    Point-in-time value.
    Identical API to prometheus_client.Gauge:
      gauge.set(value)
      gauge.inc(amount)          # works the same as Prometheus Gauge.inc()
      gauge.labels("v").set(x)
    """

    def __init__(self, name: str, description: str, label_names=()):
        self._type   = f"{_METRIC_PREFIX}/{name}"
        self._names  = tuple(label_names)
        self._vals: dict = defaultdict(float)
        self._lock   = threading.Lock()
        self._labels: dict = {}
        _REGISTRY.append(self)

    def labels(self, *args, **kwargs):
        g = _Gauge.__new__(_Gauge)
        g._type   = self._type
        g._names  = self._names
        g._vals   = self._vals
        g._lock   = self._lock
        g._labels = dict(zip(self._names, args)) if args else dict(kwargs)
        return g

    def set(self, value: float):
        key = tuple(sorted(self._labels.items()))
        with self._lock:
            self._vals[key] = float(value)

    def inc(self, amount: float = 1):
        key = tuple(sorted(self._labels.items()))
        with self._lock:
            self._vals[key] = self._vals.get(key, 0.0) + float(amount)

    def _collect(self):
        with self._lock:
            return [(self._type, dict(k), v) for k, v in self._vals.items()]


# ── Histogram ─────────────────────────────────────────────────────────────────

class _Histogram:
    """
    Observation distribution — stores last 1 000 values per label set.
    Pushes mean and P95 as separate GAUGE time series (GCM does not have
    a native histogram type; mean + P95 satisfy all dashboard and alerting needs).
    Identical API to prometheus_client.Histogram:
      hist.observe(value)
      hist.labels("v").observe(value)
    The buckets= argument is accepted but ignored (kept for API compatibility).
    """

    def __init__(self, name: str, description: str, label_names=(), buckets=None):
        self._type   = f"{_METRIC_PREFIX}/{name}"
        self._names  = tuple(label_names)
        self._obs: dict  = defaultdict(list)
        self._lock   = threading.Lock()
        self._labels: dict = {}
        _REGISTRY.append(self)

    def labels(self, *args, **kwargs):
        h = _Histogram.__new__(_Histogram)
        h._type   = self._type
        h._names  = self._names
        h._obs    = self._obs           # shared
        h._lock   = self._lock
        h._labels = dict(zip(self._names, args)) if args else dict(kwargs)
        return h

    def observe(self, value: float):
        key = tuple(sorted(self._labels.items()))
        with self._lock:
            self._obs[key].append(float(value))
            if len(self._obs[key]) > 1000:
                self._obs[key] = self._obs[key][-1000:]

    def _collect(self):
        result = []
        with self._lock:
            for key, vals in self._obs.items():
                if not vals:
                    continue
                labels   = dict(key)
                mean_val = sum(vals) / len(vals)
                p95_val  = sorted(vals)[int(0.95 * len(vals))]
                result.append((f"{self._type}/mean", labels, mean_val))
                result.append((f"{self._type}/p95",  labels, p95_val))
        return result


# ── Push to Google Cloud Monitoring ──────────────────────────────────────────

def _build_time_series(mtype: str, labels: dict, value: float, now: float):
    """Build one Cloud Monitoring TimeSeries for a single GAUGE point."""
    from google.cloud import monitoring_v3  # lazy — not loaded at import time
    seconds = int(now)
    nanos   = int((now - seconds) * 10 ** 9)
    interval = monitoring_v3.TimeInterval(
        {"end_time": {"seconds": seconds, "nanos": nanos}}
    )
    point = monitoring_v3.Point(
        {"interval": interval, "value": {"double_value": float(value)}}
    )
    ts = monitoring_v3.TimeSeries()
    ts.metric.type = mtype
    ts.metric.labels.update({k: str(v) for k, v in labels.items()})
    ts.resource.type = "global"
    ts.resource.labels["project_id"] = GOOGLE_CLOUD_PROJECT
    # Do NOT set metric_kind / value_type — GCM infers them from the descriptor.
    # Use list assignment (not .append) — works across all protobuf versions.
    ts.points = [point]
    return ts


def _do_push():
    """Collect all metrics and write to Cloud Monitoring. Silently skips on error."""
    try:
        from google.cloud import monitoring_v3  # noqa: F401  (triggers lazy import check)
        client    = monitoring_v3.MetricServiceClient()
        now       = time.time()
        all_ts    = []

        for metric in _REGISTRY:
            for (mtype, labels, value) in metric._collect():
                all_ts.append(_build_time_series(mtype, labels, value, now))

        # Cloud Monitoring accepts at most 200 time series per request.
        for i in range(0, len(all_ts), 200):
            client.create_time_series(
                name=_project_name,
                time_series=all_ts[i : i + 200],
            )
        print(f"[Metrics] pushed {len(all_ts)} series to Cloud Monitoring")

    except Exception as exc:
        # Credentials absent locally or transient GCP error — never crash the app.
        print(f"[Metrics] Cloud Monitoring push skipped: {exc}")


def push_metrics_now():
    """
    Force an immediate metric push to Cloud Monitoring.
    Called at the end of each pipeline run so metrics are not lost if
    Cloud Run shuts down the instance before the background thread fires.
    Respects a 30-second minimum interval to avoid rate-limit errors.
    """
    global _last_push
    with _push_lock:
        now = time.time()
        if now - _last_push < _MIN_PUSH_INTERVAL:
            return
        _last_push = now
    _do_push()


def _background_push():
    """Daemon thread: push all metrics every 60 seconds."""
    while True:
        time.sleep(60)
        push_metrics_now()


threading.Thread(target=_background_push, daemon=True, name="gcm-push").start()


# ── Metric definitions ────────────────────────────────────────────────────────
# Names, label sets, and types mirror the original prometheus_client definitions
# exactly — every call site in the codebase works without modification.

# A. Pipeline ─────────────────────────────────────────────────────────────────

pipeline_runs = _Counter(
    "pipeline_runs_total",
    "Total pipeline run outcomes",
    ["status"],           # success | skipped | error | retrain_triggered
)

pipeline_duration = _Histogram(
    "pipeline_duration_seconds",
    "Duration of each pipeline phase in seconds",
    ["phase"],            # full | preprocess | bq_write
    buckets=[1, 2, 5, 10, 30, 60, 120, 300, 600],
)

bq_write_errors = _Counter(
    "pipeline_bq_write_errors_total",
    "BigQuery write errors per table",
    ["table"],
)

bq_writes = _Counter(
    "pipeline_bq_writes_total",
    "Successful BigQuery write operations per table",
    ["table"],
)

# B. Preprocessing — data quality / drift signals ─────────────────────────────

moments_processed = _Counter(
    "pipeline_moments_processed_total",
    "Moment processing outcomes",
    ["outcome"],          # valid | skipped_injection | skipped_pii | skipped_profanity | skipped_spam | skipped_invalid
)

# Per-run distribution gauges — data drift indicators.
valid_ratio        = _Gauge("valid_ratio",        "Fraction of moments passing preprocessing",  ["pipeline_run_id"])
word_count_mean    = _Gauge("word_count_mean",     "Mean interpretation word count per run",     ["pipeline_run_id"])
word_count_p50     = _Gauge("word_count_p50",      "Median interpretation word count per run",   ["pipeline_run_id"])
word_count_p95     = _Gauge("word_count_p95",      "P95 interpretation word count per run",      ["pipeline_run_id"])
quality_score_mean = _Gauge("quality_score_mean",  "Mean quality score per run",                 ["pipeline_run_id"])
quality_score_p10  = _Gauge("quality_score_p10",   "P10 quality score per run",                  ["pipeline_run_id"])
readability_mean   = _Gauge("readability_mean",    "Mean Flesch readability score per run",      ["pipeline_run_id"])

# C. Compatibility / Gemini — model decay signals ─────────────────────────────

compat_runs = _Counter(
    "compat_runs_total",
    "Compatibility agent run outcomes",
    ["outcome"],          # success | cached | error_decomp | error_scoring | error_agg
)

# Primary model decay indicator: sustained low confidence → retraining.
compat_confidence_hist = _Histogram(
    "compat_confidence",
    "Distribution of compatibility confidence scores",
    ["dominant_think"],   # resonate | contradict | diverge
    buckets=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0],
)

compat_confidence_gauge = _Gauge(
    "compat_confidence_gauge",
    "Most recently observed compatibility confidence score",
)

# Verdict distribution — behavioral drift signal.
compat_think_ratio = _Gauge(
    "compat_think_ratio",
    "Cumulative think-dimension verdict counts",
    ["verdict"],          # resonate | contradict | diverge
)
compat_feel_ratio = _Gauge(
    "compat_feel_ratio",
    "Cumulative feel-dimension verdict counts",
    ["verdict"],
)

gemini_latency = _Histogram(
    "gemini_latency_seconds",
    "Gemini API call latency in seconds",
    ["call_type"],        # decompose | score
    buckets=[0.5, 1, 2, 3, 5, 10, 20, 30, 60],
)

gemini_errors = _Counter(
    "gemini_errors_total",
    "Gemini API call errors",
    ["call_type", "error_type"],  # error_type: json_parse | missing_keys | exception
)

decomp_cache_hits   = _Counter("decomp_cache_hits_total",   "Decomposition BQ cache hits")
decomp_cache_misses = _Counter("decomp_cache_misses_total", "Decomposition BQ cache misses")
compat_cache_hits   = _Counter("compat_cache_hits_total",   "Full compat result BQ cache hits")

# D. Bradley-Terry rankings ───────────────────────────────────────────────────

bt_refits = _Counter(
    "bt_refits_total",
    "Bradley-Terry model refit outcomes",
    ["outcome"],          # success | no_data | error
)

bt_refit_duration = _Histogram(
    "bt_refit_duration_seconds",
    "Time to refit BT model and write rankings for one user",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

bt_n_comparisons = _Histogram(
    "bt_n_comparisons",
    "Comparison count available at BT refit time",
    buckets=[0, 1, 2, 5, 10, 20, 50, 100, 200],
)

# Sustained near-zero → users have given minimal explicit feedback (cold start).
bt_blend_weight = _Gauge(
    "bt_blend_weight_bt",
    "BT contribution to final blend score for the most recently refit user",
)

rankings_written = _Counter(
    "rankings_written_total",
    "Ranking rows successfully inserted to BigQuery",
)