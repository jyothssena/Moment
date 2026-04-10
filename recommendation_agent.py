import json
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

COMPAT_LOG_FILE = "data/processed/compatibility_runs.json"
RECO_LOG_FILE   = "data/processed/recommendation_runs.json"

TOP_K = 3  # number of results to return per category

# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: str, default):
    try:
        with open(path) as f:
            content = f.read()
            return json.loads(content) if content.strip() else default
    except FileNotFoundError:
        return default

def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _get_runs_for_user(user_id: str) -> list[dict]:
    """
    Return all compatibility runs involving user_id (either side).
    Normalises so user_id is always 'user_a' in the returned records.
    """
    runs = _read_json(COMPAT_LOG_FILE, [])
    results = []
    for r in runs:
        if r.get("user_a") == user_id:
            results.append(r)
        elif r.get("user_b") == user_id:
            # flip so caller always reads from user_a's perspective
            flipped = dict(r)
            flipped["user_a"] = r["user_b"]
            flipped["user_b"] = r["user_a"]
            flipped["portrait_a"] = r.get("portrait_b")
            flipped["portrait_b"] = r.get("portrait_a")
            results.append(flipped)
    return results

# ── Top-k selector ────────────────────────────────────────────────────────────

def get_top_k_recommendations(user_id: str, k: int = TOP_K) -> dict:
    """
    Select the top-k compatibility runs for user_id in each verdict category:
      - resonance
      - contradiction
      - divergence

    Within each category, ranks by confidence descending.
    Excludes no_match and error results.

    Returns:
    {
        "user_id": ...,
        "resonance":     [ top-k runs ],
        "contradiction": [ top-k runs ],
        "divergence":    [ top-k runs ],
        "timestamp": ...
    }
    """
    runs = _get_runs_for_user(user_id)
    # bucket by verdict
    buckets: dict[str, list] = {
        "resonance":     [],
        "contradiction": [],
        "divergence":    [],
    }

    for r in runs:
        verdict = r.get("verdict")
        if verdict in buckets:
            buckets[verdict].append(r)

    # sort each bucket by confidence descending, take top k
    result = {"user_id": user_id, "timestamp": datetime.utcnow().isoformat()}
    for category, candidates in buckets.items():
        top = sorted(
            candidates,
            key=lambda r: r.get("confidence", 0.0),
            reverse=True
        )[:k]

        # slim down each entry — drop portrait snapshots for display
        result[category] = [
            {
                "match_user_id":  r["user_b"],
                "book_id":        r.get("book_id"),
                "think_dimension": r.get("think_dimension"),
                "feel_dimension":  r.get("feel_dimension"),
                "confidence":     r.get("confidence"),
                "insight":        r.get("insight"),
                "timestamp":      r.get("timestamp"),
            }
            for r in top
        ]

    # log the recommendation run
    _write_json(
        RECO_LOG_FILE,
        _read_json(RECO_LOG_FILE, []) + [result]
    )

    # console summary
    print(f"\n── Recommendations for {user_id} (top {k} per category) ──")
    for category in ("resonance", "contradiction", "divergence"):
        matches = result[category]
        print(f"\n  {category.upper()} ({len(matches)})")
        for m in matches:
            print(f"    {m['match_user_id']}  "
                  f"conf={m['confidence']}  "
                  f"think={m['think_dimension']}  "
                  f"feel={m['feel_dimension']}")
            print(f"    {m['insight']}")

    return result

# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    USER_ID = "Emma Chen"

    result = get_top_k_recommendations(USER_ID, k=TOP_K)
    print("\n── Full result (JSON) ───────────────────────────")
    print(json.dumps(result, indent=2))