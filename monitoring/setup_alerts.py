#!/usr/bin/env python3
"""
monitoring/setup_alerts.py — Create Cloud Monitoring alert policies for MOMENT.

Run once after deploying moment-api to Cloud Run:

    pip install google-cloud-monitoring
    export GOOGLE_CLOUD_PROJECT=moment-486719
    export ALERTING_EMAIL=your@email.com
    export MOMENT_API_URL=https://YOUR-CLOUD-RUN-URL.run.app
    python monitoring/setup_alerts.py

What this creates:
  1. Email notification channel   → alerts stakeholders
  2. Webhook notification channel → POSTs to /admin/retrain-trigger on Cloud Run
  3. Alert policies (thresholds matching the original Prometheus rules):
     - CompatConfidenceLow        → confidence < 0.40 for 15 min
     - CompatConfidenceCritical   → confidence < 0.30 for 10 min  (triggers retraining)
     - ValidMomentRatioCritical   → valid_ratio  < 0.30 for 10 min (triggers retraining)
     - ValidMomentRatioLow        → valid_ratio  < 0.50 for 15 min
     - GeminiErrorRateHigh        → gemini_errors > 5 total in 15 min
     - BTModelColdStart           → bt_blend_weight < 0.05 for 30 min
     - PipelineRunFailed          → pipeline_runs{status=error} > 0 in 30 min
"""

import os
import sys
from google.cloud import monitoring_v3
from google.protobuf.duration_pb2 import Duration

PROJECT_ID   = os.environ.get("GOOGLE_CLOUD_PROJECT", "moment-486719")
ALERT_EMAIL  = os.environ.get("ALERTING_EMAIL", "")
RETRAIN_URL  = os.environ.get("MOMENT_API_URL", "https://YOUR-CLOUD-RUN-URL.run.app")

project_name = f"projects/{PROJECT_ID}"
nc_client    = monitoring_v3.NotificationChannelServiceClient()
ap_client    = monitoring_v3.AlertPolicyServiceClient()


# ── Step 1: Create notification channels ─────────────────────────────────────

def create_email_channel(email: str) -> str:
    """Create an email notification channel. Returns channel name."""
    channel = monitoring_v3.NotificationChannel(
        type_="email",
        display_name="MOMENT Alerts — Email",
        labels={"email_address": email},
    )
    result = nc_client.create_notification_channel(
        name=project_name, notification_channel=channel
    )
    print(f"  Created email channel: {result.name}")
    return result.name


def create_webhook_channel(url: str) -> str:
    """Create a webhook notification channel pointing at /admin/retrain-trigger."""
    channel = monitoring_v3.NotificationChannel(
        type_="webhook_basicauth",
        display_name="MOMENT Retraining Webhook",
        labels={"url": f"{url}/admin/retrain-trigger"},
    )
    result = nc_client.create_notification_channel(
        name=project_name, notification_channel=channel
    )
    print(f"  Created webhook channel: {result.name}")
    return result.name


# ── Step 2: Alert policy builder ─────────────────────────────────────────────

def _duration(seconds: int) -> Duration:
    d = Duration()
    d.seconds = seconds
    return d


def create_threshold_policy(
    display_name: str,
    metric_name: str,                # e.g. "compat_confidence_gauge"
    threshold: float,
    comparison: str,                 # "COMPARISON_LT" or "COMPARISON_GT"
    duration_seconds: int,
    notification_channels: list[str],
    severity: str = "WARNING",       # WARNING | CRITICAL
    extra_filter: str = "",
) -> str:
    """Create a simple threshold alert policy. Returns policy name."""
    metric_type = f"custom.googleapis.com/moment/{metric_name}"
    filter_str  = f'metric.type="{metric_type}" resource.type="global"'
    if extra_filter:
        filter_str += f' {extra_filter}'

    condition = monitoring_v3.AlertPolicy.Condition(
        display_name=display_name,
        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
            filter=filter_str,
            comparison=monitoring_v3.ComparisonType[comparison],
            threshold_value=threshold,
            duration=_duration(duration_seconds),
            aggregations=[
                monitoring_v3.Aggregation(
                    alignment_period=_duration(300),   # 5-minute windows
                    per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                    cross_series_reducer=monitoring_v3.Aggregation.Reducer.REDUCE_MEAN,
                )
            ],
        ),
    )

    policy = monitoring_v3.AlertPolicy(
        display_name=display_name,
        conditions=[condition],
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
        notification_channels=notification_channels,
        alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
            notification_rate_limit=monitoring_v3.AlertPolicy.AlertStrategy.NotificationRateLimit(
                period=_duration(3600)   # at most one notification per hour
            )
        ),
        user_labels={"severity": severity.lower(), "project": "moment"},
    )

    result = ap_client.create_alert_policy(name=project_name, alert_policy=policy)
    print(f"  Created policy: {result.display_name}  →  {result.name}")
    return result.name


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\nSetting up Cloud Monitoring alerts for project: {PROJECT_ID}\n")

    # ── Notification channels ─────────────────────────────────────────────────
    channels_all      = []   # email only
    channels_retrain  = []   # email + webhook (triggers retraining)

    if ALERT_EMAIL:
        print("Creating notification channels...")
        email_ch   = create_email_channel(ALERT_EMAIL)
        webhook_ch = create_webhook_channel(RETRAIN_URL)
        channels_all     = [email_ch]
        channels_retrain = [email_ch, webhook_ch]
    else:
        print("ALERTING_EMAIL not set — policies will have no notification channels.")
        print("Set the env var and rerun, or add channels manually in GCP Console.\n")

    # ── Alert policies ────────────────────────────────────────────────────────
    print("\nCreating alert policies...")

    # Model decay — confidence dropping
    create_threshold_policy(
        display_name="CompatConfidenceLow — model decay warning",
        metric_name="compat_confidence_gauge",
        threshold=0.40,
        comparison="COMPARISON_LT",
        duration_seconds=900,   # 15 minutes
        notification_channels=channels_all,
        severity="WARNING",
    )

    create_threshold_policy(
        display_name="CompatConfidenceCritical — retraining triggered",
        metric_name="compat_confidence_gauge",
        threshold=0.30,
        comparison="COMPARISON_LT",
        duration_seconds=600,   # 10 minutes
        notification_channels=channels_retrain,
        severity="CRITICAL",
    )

    # Data drift — valid moment ratio
    create_threshold_policy(
        display_name="ValidMomentRatioLow — input quality warning",
        metric_name="valid_ratio",
        threshold=0.50,
        comparison="COMPARISON_LT",
        duration_seconds=900,
        notification_channels=channels_all,
        severity="WARNING",
    )

    create_threshold_policy(
        display_name="ValidMomentRatioCritical — retraining triggered",
        metric_name="valid_ratio",
        threshold=0.30,
        comparison="COMPARISON_LT",
        duration_seconds=600,
        notification_channels=channels_retrain,
        severity="CRITICAL",
    )

    # Gemini API reliability
    create_threshold_policy(
        display_name="GeminiErrorRateHigh — API reliability issue",
        metric_name="gemini_errors_total",
        threshold=5.0,
        comparison="COMPARISON_GT",
        duration_seconds=900,
        notification_channels=channels_all,
        severity="WARNING",
    )

    # Bradley-Terry cold start
    create_threshold_policy(
        display_name="BTModelColdStart — insufficient feedback data",
        metric_name="bt_blend_weight_bt",
        threshold=0.05,
        comparison="COMPARISON_LT",
        duration_seconds=1800,  # 30 minutes
        notification_channels=channels_all,
        severity="WARNING",
    )

    # Pipeline failures
    create_threshold_policy(
        display_name="PipelineRunFailed — pipeline error detected",
        metric_name="pipeline_runs_total",
        threshold=0.0,
        comparison="COMPARISON_GT",
        duration_seconds=1800,
        notification_channels=channels_all,
        severity="WARNING",
        extra_filter='metric.label.status="error"',
    )

    print("\nDone. View policies at:")
    print(f"  https://console.cloud.google.com/monitoring/alerting?project={PROJECT_ID}\n")


if __name__ == "__main__":
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("ERROR: GOOGLE_CLOUD_PROJECT env var is required.")
        sys.exit(1)
    main()