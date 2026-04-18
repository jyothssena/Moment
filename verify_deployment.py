"""
verify_deployment.py
====================
Smoke-test the deployed Moment API.

Usage:
  # Test local dev server
  python verify_deployment.py

  # Test a deployed endpoint (Kubernetes / Cloud Run / Vertex AI)
  python verify_deployment.py --url https://your-endpoint.run.app

  # Test and fail fast (exit 1) if any check fails — used in CI/CD
  python verify_deployment.py --url $ENDPOINT_URL --strict

What it checks:
  1. GET  /health        — API is up, model backend responding
  2. GET  /info          — version and mode
  3. POST /decompose     — decompose a moment into sub-claims
  4. POST /compatibility — score a reader pair
  5. POST /batch         — batch score an anchor user vs others
  6. POST /validate      — validate a result list against thresholds
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests")
    sys.exit(1)

# ── Sample test payloads (match the API's example schemas) ────────────────────

MOMENT_A = {
    "interpretation": (
        "Victor's immediate reaction of horror reveals his inability to take "
        "responsibility for what he created. The yellow eye feels like a symbol "
        "of everything he refused to confront in himself."
    ),
    "passage_id": "gutenberg_84_passage_1",
}

MOMENT_B = {
    "interpretation": (
        "The creature opening its eyes is the precise moment the text shifts "
        "from ambition to consequence. The dull yellow eye is not monstrous — "
        "it is simply alive, which is exactly what Victor cannot accept."
    ),
    "passage_id": "gutenberg_84_passage_1",
}

MOMENTS_MAP = {
    "user_emma_chen":    MOMENT_A,
    "user_marcus_will":  MOMENT_B,
    "user_sofia_reyes":  {
        "interpretation": "Frankenstein's creation narrative mirrors the anxieties of industrial progress — "
                          "the creature is the machine that outlives its maker's intentions.",
        "passage_id": "gutenberg_84_passage_1",
    },
}

# ── Printer helpers ───────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {BLUE}·{RESET}  {msg}")
def section(msg): print(f"\n{BOLD}{msg}{RESET}")


def check(label: str, resp, expected_status: int = 200, strict: bool = False):
    """Print result of one HTTP check. Returns True on pass."""
    elapsed = resp.elapsed.total_seconds() * 1000
    if resp.status_code == expected_status:
        ok(f"{label}  →  {resp.status_code}  ({elapsed:.0f}ms)")
        return True
    else:
        fail(f"{label}  →  {resp.status_code} (expected {expected_status})  ({elapsed:.0f}ms)")
        try:
            info(f"Response: {json.dumps(resp.json(), indent=2)[:400]}")
        except Exception:
            info(f"Body: {resp.text[:200]}")
        if strict:
            sys.exit(1)
        return False


# ── Test suite ────────────────────────────────────────────────────────────────

def run_checks(base_url: str, strict: bool) -> bool:
    base_url = base_url.rstrip("/")
    passed = 0
    total  = 0

    print(f"\n{BOLD}Moment API — Deployment Verification{RESET}")
    print(f"Target: {BLUE}{base_url}{RESET}")
    print(f"Time:   {datetime.now(timezone.utc).isoformat()}Z")
    print("─" * 50)

    # ── 1. Health ────────────────────────────────────────────────────────────
    section("1 · Health check")
    try:
        resp = requests.get(f"{base_url}/health", timeout=15)
        total += 1
        if check("GET /health", resp, strict=strict):
            passed += 1
            body = resp.json()
            info(f"Mode:   {body.get('mode', '?')}")
            info(f"Uptime: {body.get('uptime_seconds', '?')}s")
            model = body.get("model", {})
            info(f"Model:  {model.get('status', '?')} — {model.get('message', '')}")
    except requests.exceptions.ConnectionError:
        fail(f"Cannot connect to {base_url}")
        print(f"\n{RED}API is not reachable. Is the server running?{RESET}")
        if strict:
            sys.exit(1)
        return False

    # ── 2. Info ──────────────────────────────────────────────────────────────
    section("2 · Info endpoint")
    resp = requests.get(f"{base_url}/info", timeout=10)
    total += 1
    if check("GET /info", resp, strict=strict):
        passed += 1
        body = resp.json()
        info(f"Service: {body.get('service')}  v{body.get('version')}")
        info(f"Mode:    {body.get('mode')}")

    # ── 3. Decompose ─────────────────────────────────────────────────────────
    section("3 · Decompose moment")
    payload = {
        "user_id":     "user_emma_chen",
        "passage_id":  "gutenberg_84_passage_1",
        "book_id":     "gutenberg_84",
        "moment_text": MOMENT_A["interpretation"],
        "word_count":  len(MOMENT_A["interpretation"].split()),
    }
    resp = requests.post(f"{base_url}/decompose", json=payload, timeout=60)
    total += 1
    if check("POST /decompose", resp, strict=strict):
        passed += 1
        body = resp.json()
        subclaims = body.get("subclaims", [])
        info(f"Sub-claims: {len(subclaims)}")
        for sc in subclaims:
            info(f"  [{sc['id']}] weight={sc['weight']}  mode={sc['emotional_mode']}")
            info(f"       {sc['claim'][:80]}...")

    # ── 4. Compatibility ─────────────────────────────────────────────────────
    section("4 · Compatibility scoring")
    payload = {
        "user_a":     "user_emma_chen",
        "user_b":     "user_marcus_will",
        "book":       "Frankenstein",
        "passage_id": "gutenberg_84_passage_1",
        "moment_a":   MOMENT_A,
        "moment_b":   MOMENT_B,
    }
    resp = requests.post(f"{base_url}/compatibility", json=payload, timeout=120)
    total += 1
    if check("POST /compatibility", resp, strict=strict):
        passed += 1
        body = resp.json()
        think = body.get("think", {})
        feel  = body.get("feel",  {})
        conf  = body.get("confidence", "?")
        info(f"Think:      R={think.get('R')}%  C={think.get('C')}%  D={think.get('D')}%  → {body.get('dominant_think')}")
        info(f"Feel:       R={feel.get('R')}%   C={feel.get('C')}%   D={feel.get('D')}%  → {body.get('dominant_feel')}")
        info(f"Confidence: {conf}")
        info(f"Mode:       {'STUB' if body.get('_stub') else 'REAL'}")

        # Verify R+C+D = 100 (the most critical validation rule)
        think_sum = sum(think.values()) if think else 0
        feel_sum  = sum(feel.values())  if feel  else 0
        if think_sum == 100 and feel_sum == 100:
            ok(f"R+C+D validation: think={think_sum} feel={feel_sum} ✓")
        else:
            fail(f"R+C+D validation FAILED: think={think_sum} feel={feel_sum} (must be 100)")
            if strict:
                sys.exit(1)

    # ── 5. Batch ─────────────────────────────────────────────────────────────
    section("5 · Batch scoring")
    payload = {
        "user_a_id":   "user_emma_chen",
        "book_id":     "gutenberg_84",
        "passage_id":  "gutenberg_84_passage_1",
        "moments_map": MOMENTS_MAP,
    }
    resp = requests.post(f"{base_url}/batch", json=payload, timeout=180)
    total += 1
    if check("POST /batch", resp, strict=strict):
        passed += 1
        body = resp.json()
        results = body.get("results", [])
        info(f"Results: {len(results)} matches")
        for r in results:
            info(f"  {r.get('character_b', '?'):<25}  confidence={r.get('confidence', '?')}")

    # ── 6. Validate ──────────────────────────────────────────────────────────
    section("6 · Validation gate")
    # Use the result from the compatibility check above as a sample result list
    if resp.status_code == 200:
        sample_results = body.get("results", [])
    else:
        sample_results = []

    if sample_results:
        # Add required keys that validate_model.py checks for
        for r in sample_results:
            r.setdefault("computed_at", datetime.now(timezone.utc).isoformat())
            r.setdefault("match_count", 2)

        val_payload = {"results": sample_results}
        val_resp = requests.post(f"{base_url}/validate", json=val_payload, timeout=30)
        total += 1
        if check("POST /validate", val_resp, strict=strict):
            passed += 1
            val_body = val_resp.json()
            gate_passed = val_body.get("passed", False)
            if gate_passed:
                ok(f"Validation gate: PASSED")
            else:
                fail(f"Validation gate: FAILED — {val_body.get('failures', [])}")
            metrics = val_body.get("metrics", {})
            for k, v in metrics.items():
                info(f"  {k}: {v}")
    else:
        info("Skipping /validate — no results from batch check")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    color = GREEN if passed == total else (YELLOW if passed > 0 else RED)
    print(f"{BOLD}{color}{passed}/{total} checks passed{RESET}")

    if passed == total:
        print(f"\n{GREEN}{BOLD}Deployment verified successfully.{RESET}")
        print(f"Endpoint is live and responding correctly at:\n  {base_url}\n")
    else:
        print(f"\n{YELLOW}Some checks failed. Review output above.{RESET}\n")

    return passed == total


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Moment API deployment")
    parser.add_argument(
        "--url", default="http://localhost:8080",
        help="Base URL of the API (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 on first failure (for CI use)"
    )
    args = parser.parse_args()

    success = run_checks(args.url, args.strict)
    sys.exit(0 if success else 1)
