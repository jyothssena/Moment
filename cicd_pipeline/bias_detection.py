"""
bias_detection.py — Automated Model Bias Detection
===================================================
Requirement 3: Detect bias across data slices (book, passage).
Significant bias triggers an alert and blocks deployment.

Bias definition for Momento:
  A significant confidence gap between slices indicates the model
  performs much better on certain books/passages than others —
  which would skew compatibility scores unfairly.

Called by CI pipeline after validation passes.
Exits 0 if clean, 1 if alerts are triggered.
"""

import json
import sys
from collections import defaultdict
from typing import List, Dict, Any

# ── Thresholds ────────────────────────────────────────────────────────────────
BIAS_ALERT_THRESHOLD = 0.30   # max allowed confidence gap between any two slices
BIAS_BLOCK_THRESHOLD = 0.50   # gap above this blocks deployment entirely

SLICE_KEYS = ["book", "passage_id"]   # dimensions to check for bias


# ── Core detection ────────────────────────────────────────────────────────────

def compute_slice_stats(results: List[Dict], slice_key: str) -> dict:
    """
    Groups results by slice_key and computes mean confidence per group.
    Returns {slice_value: {"mean_confidence": float, "count": int}}
    """
    groups = defaultdict(list)
    for r in results:
        key_val = r.get(slice_key, "unknown")
        conf    = r.get("confidence")
        if conf is not None:
            groups[key_val].append(conf)

    stats = {}
    for key_val, confs in groups.items():
        stats[key_val] = {
            "mean_confidence": round(sum(confs) / len(confs), 4),
            "count": len(confs),
        }
    return stats


def compute_confidence_gap(stats: dict) -> float:
    """Returns the max difference in mean_confidence across slices."""
    if len(stats) < 2:
        return 0.0
    confs = [v["mean_confidence"] for v in stats.values()]
    return round(max(confs) - min(confs), 4)


def detect_bias_across_slices(results: List[Dict[str, Any]]) -> dict:
    """
    Full bias report across all slice dimensions.

    Returns
    -------
    {
      "by_book":             {book: {mean_confidence, count}},
      "by_passage":          {passage_id: {mean_confidence, count}},
      "max_confidence_gap":  float,
      "alerts":              [str],   # human-readable alerts
      "block_deployment":    bool,
    }
    """
    report = {
        "by_book":            compute_slice_stats(results, "book"),
        "by_passage":         compute_slice_stats(results, "passage_id"),
        "max_confidence_gap": 0.0,
        "alerts":             [],
        "block_deployment":   False,
    }

    gaps = {}
    for key in SLICE_KEYS:
        label = f"by_{key}" if key != "passage_id" else "by_passage"
        stats = report[label]
        gap   = compute_confidence_gap(stats)
        gaps[key] = gap

        if gap >= BIAS_ALERT_THRESHOLD:
            worst = min(stats, key=lambda k: stats[k]["mean_confidence"])
            best  = max(stats, key=lambda k: stats[k]["mean_confidence"])
            report["alerts"].append(
                f"Confidence gap by {key}: {gap:.3f} "
                f"(best: {best} @ {stats[best]['mean_confidence']:.3f}, "
                f"worst: {worst} @ {stats[worst]['mean_confidence']:.3f})"
            )

    report["max_confidence_gap"] = round(max(gaps.values()), 4) if gaps else 0.0

    if report["max_confidence_gap"] >= BIAS_BLOCK_THRESHOLD:
        report["block_deployment"] = True

    return report


def run_bias_gate(results: List[Dict[str, Any]]) -> dict:
    """
    Runs bias detection and returns a gate decision.
    Called by CI pipeline after validation passes.

    Returns
    -------
    {
      "passed": bool,
      "report": dict,
      "block_reason": str | None
    }
    """
    report = detect_bias_across_slices(results)

    if report["block_deployment"]:
        return {
            "passed":       False,
            "report":       report,
            "block_reason": (
                f"Confidence gap {report['max_confidence_gap']:.3f} "
                f"exceeds block threshold {BIAS_BLOCK_THRESHOLD}"
            ),
        }

    if report["alerts"]:
        # Alerts are logged but don't block (gap is between ALERT and BLOCK threshold)
        print("⚠  Bias alerts (logged, not blocking):")
        for alert in report["alerts"]:
            print(f"   • {alert}")

    return {
        "passed":       True,
        "report":       report,
        "block_reason": None,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage: python bias_detection.py results.json
    Exits 0 if clean (or alerts-only), 1 if deployment should be blocked.
    """
    if len(sys.argv) < 2:
        print("Usage: python bias_detection.py <results.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    results = data if isinstance(data, list) else data.get("pairs", [])
    gate = run_bias_gate(results)

    print(json.dumps(gate["report"], indent=2))

    if gate["passed"]:
        if gate["report"]["alerts"]:
            print("\n⚠  Bias alerts logged — proceeding to registry push")
        else:
            print("\n✓ Bias check PASSED — no significant bias detected")
        sys.exit(0)
    else:
        print(f"\n✗ Bias check BLOCKED: {gate['block_reason']}")
        sys.exit(1)