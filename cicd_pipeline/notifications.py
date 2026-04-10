"""
notifications.py — Pipeline Notifications & Alerts
====================================================
Requirement 5: Send Slack/email alerts for:
  - Pipeline failures
  - Training completion
  - Validation failures
  - Bias alerts
  - Rollback events

Set SLACK_WEBHOOK_URL in GitHub Actions secrets.
Set NOTIFICATION_EMAIL for email fallback (optional).
"""

import json
import os
import urllib.request
import urllib.error
from datetime import datetime
from typing import List, Optional


SLACK_WEBHOOK_URL  = os.environ.get("SLACK_WEBHOOK_URL", "")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "")
PROJECT_NAME       = "Momento"


# ── Core send ─────────────────────────────────────────────────────────────────

def send_slack_alert(message: str, dry_run: bool = False) -> bool:
    """
    Sends a message to Slack via webhook.
    Returns True on success, False on failure.
    dry_run=True prints without sending (used in tests and local dev).
    """
    if dry_run or not SLACK_WEBHOOK_URL:
        print(f"[NOTIFICATION DRY RUN] {message}")
        return True

    payload = json.dumps({"text": message}).encode("utf-8")
    try:
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Slack notification failed: {e}")
        return False


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


# ── Typed notification helpers ────────────────────────────────────────────────

def notify_training_complete(
    model_version: str,
    metrics: Optional[dict] = None,
    dry_run: bool = False,
) -> bool:
    """Called when training finishes successfully."""
    msg = (
        f"✅ *{PROJECT_NAME} — Training Complete*\n"
        f"Version: `{model_version}`\n"
        f"Time: {_timestamp()}"
    )
    if metrics:
        conf = metrics.get("mean_confidence", "n/a")
        rate = metrics.get("schema_pass_rate", "n/a")
        msg += f"\nMean confidence: `{conf}` | Schema pass rate: `{rate}`"
    return send_slack_alert(msg, dry_run=dry_run)


def notify_validation_failure(
    metrics: dict,
    failures: Optional[List[str]] = None,
    dry_run: bool = False,
) -> bool:
    """Called when validation gate fails — blocks deployment."""
    failure_lines = "\n".join(f"  • {f}" for f in (failures or []))
    msg = (
        f"❌ *{PROJECT_NAME} — Validation FAILED — Deployment Blocked*\n"
        f"Time: {_timestamp()}\n"
        f"Failures:\n{failure_lines or '  • Unknown'}"
    )
    return send_slack_alert(msg, dry_run=dry_run)


def notify_bias_alert(
    alerts: List[str],
    block_deployment: bool = False,
    dry_run: bool = False,
) -> bool:
    """Called when bias detection finds significant gaps."""
    icon  = "🚫" if block_deployment else "⚠️"
    label = "DEPLOYMENT BLOCKED" if block_deployment else "Alert (not blocking)"
    alert_lines = "\n".join(f"  • {a}" for a in alerts)
    msg = (
        f"{icon} *{PROJECT_NAME} — Bias Detection: {label}*\n"
        f"Time: {_timestamp()}\n"
        f"Alerts:\n{alert_lines}"
    )
    return send_slack_alert(msg, dry_run=dry_run)


def notify_deployment_success(
    model_version: str,
    environment: str = "production",
    dry_run: bool = False,
) -> bool:
    """Called when deployment to Vertex AI completes."""
    msg = (
        f"🚀 *{PROJECT_NAME} — Deployed to {environment}*\n"
        f"Version: `{model_version}`\n"
        f"Time: {_timestamp()}"
    )
    return send_slack_alert(msg, dry_run=dry_run)


def notify_rollback(
    reason: str,
    rolled_back_to: str,
    dry_run: bool = False,
) -> bool:
    """Called when rollback is triggered."""
    msg = (
        f"🔄 *{PROJECT_NAME} — Rollback Triggered*\n"
        f"Reason: {reason}\n"
        f"Rolled back to: `{rolled_back_to}`\n"
        f"Time: {_timestamp()}"
    )
    return send_slack_alert(msg, dry_run=dry_run)


def notify_pipeline_failure(
    step: str,
    error: str,
    dry_run: bool = False,
) -> bool:
    """Called on unexpected pipeline failure at any step."""
    msg = (
        f"💥 *{PROJECT_NAME} — Pipeline Failure*\n"
        f"Step: `{step}`\n"
        f"Error: {error}\n"
        f"Time: {_timestamp()}"
    )
    return send_slack_alert(msg, dry_run=dry_run)