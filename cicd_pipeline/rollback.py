"""
rollback.py — Rollback Mechanism
=================================
Requirement 6: If the new model performs worse than the previous
model, block deployment and optionally re-deploy the previous version.

The CI pipeline calls should_rollback() after computing metrics on
the new model. If it returns True, the pipeline re-tags the previous
image as :production in Artifact Registry and re-deploys it.
"""

import json
import os
import sys
from typing import Dict

# ── Rollback thresholds ───────────────────────────────────────────────────────
# A regression on ANY of these metrics triggers rollback.
ROLLBACK_THRESHOLDS = {
    "mean_confidence":   0.05,   # new model may not be more than 0.05 lower
    "schema_pass_rate":  0.03,   # new model may not be more than 3% lower
}


def should_rollback(
    previous_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
) -> bool:
    """
    Returns True if the new model has regressed beyond the allowed threshold
    on any monitored metric.

    Parameters
    ----------
    previous_metrics : metrics from the currently deployed model
    new_metrics      : metrics from the newly trained model
    """
    for metric, max_drop in ROLLBACK_THRESHOLDS.items():
        prev = previous_metrics.get(metric)
        new  = new_metrics.get(metric)

        if prev is None or new is None:
            continue  # skip if metric not available

        drop = prev - new
        if drop > max_drop:
            print(
                f"✗ Rollback triggered: {metric} dropped by {drop:.4f} "
                f"(prev={prev:.4f}, new={new:.4f}, max_allowed_drop={max_drop})"
            )
            return True

    return False


def load_previous_metrics(path: str = "metrics_baseline.json") -> dict:
    """
    Loads the metrics from the previously deployed model.
    Returns empty dict if no baseline exists (first deploy always proceeds).
    """
    if not os.path.exists(path):
        print(f"No baseline metrics at {path} — first deploy, skipping rollback check")
        return {}
    with open(path) as f:
        return json.load(f)


def save_metrics_baseline(metrics: dict, path: str = "metrics_baseline.json"):
    """Saves current metrics as the new baseline after a successful deploy."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved new baseline metrics to {path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage: python rollback.py new_metrics.json [baseline_metrics.json]
    Exits 0 if no rollback needed, 1 if rollback should be triggered.
    """
    if len(sys.argv) < 2:
        print("Usage: python rollback.py <new_metrics.json> [baseline_metrics.json]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        new_metrics = json.load(f)

    baseline_path = sys.argv[2] if len(sys.argv) > 2 else "metrics_baseline.json"
    previous_metrics = load_previous_metrics(baseline_path)

    if not previous_metrics:
        print("✓ No baseline — proceeding with deployment")
        sys.exit(0)

    if should_rollback(previous_metrics, new_metrics):
        print("✗ ROLLBACK TRIGGERED — re-deploying previous model version")
        sys.exit(1)
    else:
        print("✓ No rollback needed — new model meets or exceeds baseline")
        sys.exit(0)