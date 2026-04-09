"""
aggregator.py — Moment Compatibility Aggregator with Bradley-Terry Reranking

Responsibilities:
  1. Load compatibility runs from output.jsonl
  2. Fit per-user Bradley-Terry models from implicit engagement signal
     (sessions + conversation depth)
  3. Rerank top-k recommendations using blended confidence + BT score
  4. Expose inferred verdict preferences per user for analysis

Signal hierarchy:
  - Run shown but ignored       → loss (loser in comparison)
  - Run opened (no chat)        → weight 1.0
  - Run opened + chat started   → weight 1.0 + engagement_score (up to 2x)
"""

import json
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from pathlib import Path
from typing import Optional
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR              = Path("data/processed")
RUNS_FILE             = DATA_DIR / "compatibility_runs.jsonl"
COMPARISONS_FILE      = DATA_DIR / "comparisons.jsonl"
CONVERSATIONS_FILE    = DATA_DIR / "conversations.jsonl"


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_runs() -> dict[str, dict]:
    """Load all compatibility runs, indexed by run_id."""
    runs = {}
    with open(RUNS_FILE) as f:
        for line in f:
            r = json.loads(line)
            runs[r["run_id"]] = r
    return runs


def load_comparisons() -> list[dict]:
    """Load all implicit pairwise comparisons."""
    records = []
    with open(COMPARISONS_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_conversation_weights() -> dict[str, dict[str, float]]:
    """
    Build a per-user index of run_id → engagement_score from conversations.
    engagement_score is derived from message_count and unique_sessions.
    """
    weights: dict[str, dict[str, float]] = defaultdict(dict)
    with open(CONVERSATIONS_FILE) as f:
        for line in f:
            c = json.loads(line)
            weights[c["user_id"]][c["match_run_id"]] = c["engagement_score"]
    return weights


# ── Core Bradley-Terry fitter ─────────────────────────────────────────────────

def fit_bradley_terry(
    comparisons_list: list[tuple[str, str]],
    weights: Optional[list[float]] = None,
) -> dict[str, float]:
    """
    Fit a Bradley-Terry model from pairwise comparisons.

    Args:
        comparisons_list: list of (winner_run_id, loser_run_id)
        weights: optional per-comparison weights (e.g. boosted by conversation depth)

    Returns:
        dict mapping run_id → normalised quality score

    The probability that run A beats run B is:
        P(A > B) = score_A / (score_A + score_B)
    """
    if not comparisons_list:
        return {}

    items = list({run_id for pair in comparisons_list for run_id in pair})
    idx   = {item: i for i, item in enumerate(items)}
    n     = len(items)

    if weights is None:
        weights = [1.0] * len(comparisons_list)

    def neg_log_likelihood(log_scores: np.ndarray) -> float:
        scores_exp = np.exp(log_scores)
        loss = 0.0
        for (winner, loser), w in zip(comparisons_list, weights):
            s_w = scores_exp[idx[winner]]
            s_l = scores_exp[idx[loser]]
            loss -= w * np.log(s_w / (s_w + s_l) + 1e-10)
        return loss

    result   = minimize(neg_log_likelihood, x0=np.zeros(n), method="L-BFGS-B")
    scores   = np.exp(result.x)
    scores  /= scores.sum()   # normalise to sum to 1

    return {item: round(float(scores[idx[item]]), 6) for item in items}


# ── Per-user BT model ─────────────────────────────────────────────────────────

MIN_COMPARISONS = 5   # fall back to global model below this threshold
BT_CACHE: dict[str, dict[str, float]] = {}   # in-memory cache per session


def fit_user_bt(
    user_id: str,
    comparisons: list[dict],
    conv_weights: dict[str, dict[str, float]],
    use_cache: bool = True,
) -> Optional[dict[str, float]]:
    """
    Fit a Bradley-Terry model for a single user using their implicit engagement history.

    Comparison weights are boosted by conversation depth:
      - plain tap (no conversation):  weight = 1.0
      - chat started:                 weight = 1.0 + engagement_score  (up to ~2.0)

    Returns None if the user has fewer than MIN_COMPARISONS (caller should
    fall back to the global model).
    """
    if use_cache and user_id in BT_CACHE:
        return BT_CACHE[user_id]

    user_cmps = [
        (c["winner_run_id"], c["loser_run_id"])
        for c in comparisons
        if c["user_id"] == user_id
    ]

    if len(user_cmps) < MIN_COMPARISONS:
        return None   # not enough signal yet

    user_conv = conv_weights.get(user_id, {})
    weights   = [1.0 + user_conv.get(winner_id, 0.0) for winner_id, _ in user_cmps]

    scores = fit_bradley_terry(user_cmps, weights)
    BT_CACHE[user_id] = scores
    return scores


def fit_global_bt(
    comparisons: list[dict],
    conv_weights: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Fit a single global Bradley-Terry model across all users.
    Used as a fallback for new users with insufficient comparison history.
    """
    all_cmps = [(c["winner_run_id"], c["loser_run_id"]) for c in comparisons]

    # Global weights: use the max engagement score across users for each winner
    all_conv_scores: dict[str, float] = {}
    for user_conv in conv_weights.values():
        for run_id, score in user_conv.items():
            all_conv_scores[run_id] = max(all_conv_scores.get(run_id, 0.0), score)

    weights = [1.0 + all_conv_scores.get(winner_id, 0.0) for winner_id, _ in all_cmps]
    return fit_bradley_terry(all_cmps, weights)


# ── Verdict preference inference ──────────────────────────────────────────────

def infer_verdict_preference(
    user_id: str,
    bt_scores: dict[str, float],
    runs: dict[str, dict],
) -> dict[str, float]:
    """
    Derive a user's average BT score per verdict type (resonate / contradict / diverge).

    A user whose BT scores are consistently higher on 'contradict' runs
    prefers contradiction matches. Feed this back into prompt weighting
    or UI personalisation.

    Returns:
        dict like {"resonate": 0.28, "contradict": 0.51, "diverge": 0.21}
        Values are None if no scored runs exist for that verdict type.
    """
    by_verdict: dict[str, list[float]] = defaultdict(list)

    for run_id, score in bt_scores.items():
        run     = runs.get(run_id, {})
        verdict = run.get("dominant_think")
        if verdict:
            by_verdict[verdict].append(score)

    return {
        verdict: round(float(np.mean(scores)), 6) if scores else 0.0
        for verdict, scores in by_verdict.items()
    }

def blend_weights(n_comparisons: int) -> tuple[float, float]:
    """
    Dynamically compute blend weights based on how many comparisons
    a user has. BT weight grows as confidence in the model increases.
    
    Returns (conf_weight, bt_weight)
    """
    # BT weight grows from 0.1 to 0.7 as comparisons increase
    # Plateaus after ~50 comparisons
    bt_weight   = min(0.7, 0.1 + (n_comparisons / 50) * 0.6)
    conf_weight = round(1.0 - bt_weight, 2)
    bt_weight   = round(bt_weight, 2)
    return conf_weight, bt_weight

# ── Top-k reranking ───────────────────────────────────────────────────────────

def rerank_topk(
    candidate_run_ids: list[str],
    user_id: str,
    runs: dict[str, dict],
    comparisons: list[dict],
    conv_weights: dict[str, dict[str, float]],
    global_bt: Optional[dict[str, float]] = None,
    k: int = 5,
    passage_id: Optional[str] = None, 
    book_id: Optional[str] = None,
) -> list[dict]:
    """
    Rerank a list of candidate compatibility runs for a user using a blended
    score of confidence (from the compatibility agent) and personalised
    Bradley-Terry quality score.
    
    blend = conf_weight * confidence + bt_weight * bt_score_normalised
    where weights are dynamically set by blend_weights(n_comparisons)

    Falls back to pure confidence ranking if no BT scores are available.

    Args:
        candidate_run_ids: run_ids to rank (e.g. all runs for this user)
        user_id:           the anchor user we are recommending matches for
        runs:              full run index (from load_runs)
        comparisons:       all comparisons (from load_comparisons)
        conv_weights:      conversation engagement weights
        global_bt:         pre-fitted global BT model for fallback
        k:                 number of results to return

    Returns:
        List of dicts with run metadata + blend_score, sorted best first.
    """
    # Filter to specific passage if requested
    if passage_id and book_id:
        candidate_run_ids = [
            rid for rid in candidate_run_ids
            if runs.get(rid, {}).get("passage_id") == passage_id
            and runs.get(rid, {}).get("book_id")    == book_id
        ]
    
    # Fit or retrieve per-user BT model
    bt_scores = fit_user_bt(user_id, comparisons, conv_weights)

    # Fall back to global BT if not enough personal data
    if bt_scores is None:
        bt_scores = global_bt or {}

    # Normalise BT scores within the candidate set so they are on a 0-1 scale
    candidate_bt = {rid: bt_scores.get(rid, 0.0) for rid in candidate_run_ids}
    max_bt       = max(candidate_bt.values()) if candidate_bt else 1.0
    if max_bt == 0:
        max_bt = 1.0   # avoid division by zero when no BT signal exists yet

    user_cmps = [
    c for c in comparisons
    if c["user_id"] == user_id
    ]
    conf_weight, bt_weight = blend_weights(len(user_cmps))
    ranked = []
    for run_id in candidate_run_ids:
        run      = runs.get(run_id)
        if not run:
            continue

        confidence             = run.get("confidence", 0.0)
        bt_norm                = candidate_bt.get(run_id, 0.0) / max_bt
        blend                  = conf_weight * confidence + bt_weight * bt_norm
        ranked.append({
            "run_id":        run_id,
            "user_a":        run.get("user_a"),
            "user_b":        run.get("user_b"),
            "book_id":       run.get("book_id"),
            "passage_id":    run.get("passage_id"),
            "verdict":       run.get("dominant_think"),
            "confidence":    confidence,
            "bt_score":      round(candidate_bt.get(run_id, 0.0), 6),
            "bt_score_norm": round(bt_norm, 4),
            "blend_score":   round(blend, 4),
            "weights_used":  {
                "conf": conf_weight,
                "bt":   bt_weight,
                "n_comparisons": len(user_cmps),
            },
        })

    ranked.sort(key=lambda x: x["blend_score"], reverse=True)
    return ranked[:k]


# ── Record new comparison ─────────────────────────────────────────────────────

def record_comparison(
    user_id: str,
    winner_run_id: str,
    loser_run_id: str,
    session_id: str,
    winner_run: Optional[dict] = None,
) -> dict:
    """
    Persist a new implicit comparison to comparisons.jsonl.
    Call this whenever a user engages with one match and ignores another
    shown in the same session.

    Session-scoping rule: only record when both runs were shown in the
    same session — this filters accidental skips from genuine preferences.
    """
    from datetime import datetime
    
    record = {
        "comparison_id":      f"cmp_{abs(hash((user_id, winner_run_id, loser_run_id, session_id))):08x}",
        "user_id":            user_id,
        "session_id":         session_id,
        "winner_run_id":      winner_run_id,
        "loser_run_id":       loser_run_id,
        "winner_confidence":  winner_run.get("confidence") if winner_run else None,
        "winner_verdict":     winner_run.get("dominant_think") if winner_run else None,
        "timestamp":          datetime.utcnow().isoformat(),
    }

    with open(COMPARISONS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Invalidate cache for this user so next call refits
    BT_CACHE.pop(user_id, None)

    return record

def save_results(all_results: dict) -> None:
    out_dir  = DATA_DIR / "rankings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "rankings.json"

    with open(out_file, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "results":      all_results
        }, f, indent=2)
              
# ── Main: demo run ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Bradley-Terry reranking.")
    parser.add_argument("--user_id", type=str, help="Specific user_id to run for", default=None)
    args = parser.parse_args()

    print("Loading data...")
    runs         = load_runs()
    comparisons  = load_comparisons()
    conv_weights = load_conversation_weights()

    print(f"  {len(runs):,} runs  |  {len(comparisons):,} comparisons  |  "
          f"{sum(len(v) for v in conv_weights.values()):,} conversations\n")

    # Fit global BT once (fallback for cold-start users)
    print("Fitting global Bradley-Terry model...")
    global_bt = fit_global_bt(comparisons, conv_weights)
    print(f"  Global model covers {len(global_bt):,} runs\n")

    all_users = set()
    for r in runs.values():
        all_users.add(r["user_a"])
        all_users.add(r["user_b"])
    
    if args.user_id:
        if args.user_id not in all_users:
            print(f"Warning: user {args.user_id} not found in runs.")
        all_users = {args.user_id}

    # Get all unique (book, passage) combinations
    passages = set()
    for r in runs.values():
        passages.add((r["book_id"], r["passage_id"]))

    print(f"Running reranking for {len(all_users)} users "
          f"across {len(passages)} passages...\n")
    all_results = {}
    for user in sorted(all_users):
        candidate_ids = [
            rid for rid, r in runs.items()
            if r.get("user_a") == user or r.get("user_b") == user
        ]
        # Group passages by book
        all_results[user] = {}
        books = set(book_id for book_id, _ in passages)

        for book_id in sorted(books):
            book_passages = [p for b, p in passages if b == book_id]
            all_results[user][book_id] = {}

            for passage_id in sorted(book_passages):
                results = rerank_topk(
                    candidate_run_ids=candidate_ids,
                    user_id=user,
                    runs=runs,
                    comparisons=comparisons,
                    conv_weights=conv_weights,
                    global_bt=global_bt,
                    k=5,
                    book_id=book_id,
                    passage_id=passage_id,
                )
                if results:
                    all_results[user][book_id][passage_id] = results

    save_results(all_results)
    print(f"\nDone. Results saved to {DATA_DIR / 'rankings' / 'rankings.json'}")

if __name__ == "__main__":
    main()