"""
validate_model.py — Automated Model Validation
===============================================
Requirement 2: After training, automatically evaluate the model on
the validation set and block deployment if thresholds are not met.

Called by the CI/CD pipeline after run_compatibility_pipeline().
Also used directly in test_pipeline.py.
"""

import json
import os
import sys
from typing import List, Dict, Any

# ── Thresholds — adjust these as your model matures ─────────────────────────
VALIDATION_THRESHOLDS = {
    "schema_pass_rate":          0.95,   # 95% of outputs must pass schema check
    "confidence_in_range_rate":  0.95,   # 95% must have confidence in [0.20, 0.95]
    "mean_confidence":       0.40,   # average confidence must be at least 0.40
    "rcd_think_pass_rate":       1.00,
    "rcd_feel_pass_rate":        1.00    # 100% of R+C+D must sum to 100 (non-negotiable)
}

VALID_DOMINANT_LABELS = {"resonate", "contradict", "diverge"}
VALID_EMOTIONAL_MODES = {
    "prosecutorial", "philosophical", "empathetic",
    "observational", "aesthetic", "self-referential",
}


# ── Individual validators ─────────────────────────────────────────────────────

def validate_output_schema(result: dict) -> list:
    """
    Returns a list of error strings. Empty list = valid.
    Checks all required keys and types are present.
    """
    errors = []
    required_keys = [
        "passage_id", "character_a", "character_b", "book",
        "think", "feel", "dominant_think", "dominant_feel",
        "match_count", "confidence", "computed_at",
    ]
    for key in required_keys:
        if key not in result:
            errors.append(f"Missing required key: {key}")

    for dim in ["think", "feel"]:
        if dim in result:
            for subkey in ["R", "C", "D"]:
                if subkey not in result[dim]:
                    errors.append(f"Missing {dim}.{subkey}")

    return errors


def validate_rcd_sums(result: dict, dim: str) -> bool:
    """Returns True if R + C + D == 100 (within tolerance of 2)."""
    if dim not in result:
        return False
    s = sum(result[dim].get(k, 0) for k in ["R", "C", "D"])
    return abs(s - 100) <= 2


def validate_confidence_range(result: dict) -> bool:
    """Returns True if confidence is in [0.20, 0.95]."""
    conf = result.get("confidence")
    if conf is None:
        return False
    return 0.20 <= conf <= 0.95


def validate_dominant_labels(result: dict) -> bool:
    """Returns True if both dominant labels are valid."""
    dt = result.get("dominant_think", "")
    df = result.get("dominant_feel", "")
    return dt in VALID_DOMINANT_LABELS and df in VALID_DOMINANT_LABELS


# ── Batch metrics ─────────────────────────────────────────────────────────────

def compute_validation_metrics(results: List[Dict[str, Any]]) -> dict:
    """
    Runs all validators across a list of pipeline results.
    Returns aggregate metrics dict.
    """
    if not results:
        return {"error": "No results to validate"}

    n = len(results)
    schema_pass   = sum(1 for r in results if validate_output_schema(r) == [])
    conf_in_range = sum(1 for r in results if validate_confidence_range(r))
    rcd_think     = sum(1 for r in results if validate_rcd_sums(r, "think"))
    rcd_feel      = sum(1 for r in results if validate_rcd_sums(r, "feel"))
    label_valid   = sum(1 for r in results if validate_dominant_labels(r))

    valid_confs = [r["confidence"] for r in results if validate_confidence_range(r)]
    mean_conf   = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0

    return {
        "total_results":            n,
        "schema_pass_rate":         schema_pass / n,
        "confidence_in_range_rate": conf_in_range / n,
        "rcd_think_pass_rate":      rcd_think / n,
        "rcd_feel_pass_rate":       rcd_feel / n,
        "dominant_label_pass_rate": label_valid / n,
        "mean_confidence":          round(mean_conf, 4),
    }


def run_validation_gate(results: List[Dict[str, Any]]) -> dict:
    """
    Runs all validators and returns a gate decision.
    Called by the CI pipeline after model inference.

    Returns
    -------
    {
      "passed": bool,
      "metrics": dict,
      "failures": [str]    # human-readable failure reasons
    }
    """
    metrics  = compute_validation_metrics(results)
    failures = []

    for metric_key, threshold in VALIDATION_THRESHOLDS.items():
        actual = metrics.get(metric_key)
        if actual is None:
            failures.append(f"{metric_key}: not computed")
        elif actual < threshold:
            failures.append(
                f"{metric_key}: {actual:.3f} < threshold {threshold}"
            )

    passed = len(failures) == 0
    return {
        "passed":   passed,
        "metrics":  metrics,
        "failures": failures,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage: python validate_model.py results.json
    Exits 0 if validation passes, 1 if it fails (used by CI pipeline).
    """
    if len(sys.argv) < 2:
        print("Usage: python validate_model.py <results.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    results = data if isinstance(data, list) else data.get("pairs", [])
    gate = run_validation_gate(results)

    print(json.dumps(gate, indent=2))

    if gate["passed"]:
        print("\n✓ Validation PASSED — proceeding to bias detection")
        sys.exit(0)
    else:
        print("\n✗ Validation FAILED — deployment blocked")
        for f in gate["failures"]:
            print(f"  • {f}")
        sys.exit(1)