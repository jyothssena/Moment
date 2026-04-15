"""
run_rankings.py — Standalone BT reranking script.

Reads compatibility_results, comparisons, and conversations from BQ,
fits Bradley-Terry models per user, and writes ranked results to BQ rankings table.

Run once to populate rankings from existing data:
  python run_rankings.py

Env vars required:
  MOMENT_GCP_PROJECTID  — your GCP project ID
  BQ_DATASET            — your BQ dataset name (default: new_moments_process)
"""

import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from scipy.optimize import minimize
from google.cloud import bigquery

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT = os.environ.get("MOMENT_GCP_PROJECTID",  "your-gcp-project")
DATASET = os.environ.get("BQ_DATASET", "new_moments_processed")
K       = 5

client = bigquery.Client(project=PROJECT)


# ── BQ loaders ────────────────────────────────────────────────────────────────

def _query(sql: str) -> list[dict]:
    rows = client.query(sql).result()
    return [dict(row) for row in rows]


def load_compat_runs() -> dict[str, dict]:
    """Load compatibility_results from BQ, indexed by run_id (string)."""
    rows = _query(f"""
        SELECT
            CAST(run_id AS STRING)      AS run_id,
            CAST(user_a AS STRING)      AS user_a,
            CAST(user_b AS STRING)      AS user_b,
            book_id,
            passage_id,
            dominant_think,
            confidence
        FROM `{PROJECT}.{DATASET}.compatibility_results`
    """)
    runs = {r["run_id"]: r for r in rows}
    print(f"  Loaded {len(runs):,} compat runs from BQ")
    return runs


def load_comparisons() -> list[dict]:
    """Load comparisons from BQ."""
    rows = _query(f"""
        SELECT
            CAST(user_id AS STRING)       AS user_id,
            CAST(winner_run_id AS STRING) AS winner_run_id,
            CAST(loser_run_id AS STRING)  AS loser_run_id,
            winner_confidence,
            winner_verdict,
            session_id,
            timestamp
        FROM `{PROJECT}.{DATASET}.comparisons`
    """)
    print(f"  Loaded {len(rows):,} comparisons from BQ")
    return rows


def load_conv_weights() -> dict[str, dict[str, float]]:
    """{ user_id: { run_id: engagement_score } } from BQ conversations."""
    rows = _query(f"""
        SELECT
            CAST(user_id AS STRING)       AS user_id,
            CAST(match_run_id AS STRING)  AS match_run_id,
            engagement_score
        FROM `{PROJECT}.{DATASET}.conversations`
    """)
    weights: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        weights[r["user_id"]][r["match_run_id"]] = float(r["engagement_score"])
    print(f"  Loaded conversation weights for {len(weights):,} users from BQ")
    return weights


# ── Bradley-Terry ─────────────────────────────────────────────────────────────

MIN_COMPARISONS = 1


def fit_bradley_terry(comparisons_list: list[tuple], weights: list[float] = None) -> dict[str, float]:
    if not comparisons_list:
        return {}
    items = list({r for pair in comparisons_list for r in pair})
    idx   = {item: i for i, item in enumerate(items)}
    n     = len(items)
    if weights is None:
        weights = [1.0] * len(comparisons_list)

    def neg_log_likelihood(log_scores):
        scores = np.exp(log_scores)
        loss = 0.0
        for (w, l), wt in zip(comparisons_list, weights):
            sw, sl = scores[idx[w]], scores[idx[l]]
            loss -= wt * np.log(sw / (sw + sl) + 1e-10)
        return loss

    result = minimize(neg_log_likelihood, x0=np.zeros(n), method="L-BFGS-B")
    scores = np.exp(result.x)
    scores /= scores.sum()
    return {item: round(float(scores[idx[item]]), 6) for item in items}


def blend_weights(n_comparisons: int) -> tuple[float, float]:
    if n_comparisons == 0:
        return 1.0, 0.0  # pure confidence, no BT
    # Even 1 comparison gives meaningful BT weight
    bt_weight   = min(0.8, 0.3 + (n_comparisons / 50) * 0.5)
    conf_weight = round(1.0 - bt_weight, 2)
    return conf_weight, round(bt_weight, 2)


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank_for_user(
    user_id: str,
    candidate_run_ids: list[str],
    runs: dict[str, dict],
    comparisons: list[dict],
    conv_weights: dict[str, dict[str, float]],
    global_bt: dict[str, float],
    book_id: str,
    passage_id: str,
    k: int = 5,
) -> list[dict]:
    passage_candidates = [
        rid for rid in candidate_run_ids
        if runs.get(rid, {}).get("passage_id") == passage_id
        and runs.get(rid, {}).get("book_id") == book_id
    ]
    if not passage_candidates:
        return []

    # Use all comparisons for this user across all passages for BT fitting
    # (not just passage_candidates) so we get broader signal
    all_user_cmps = [c for c in comparisons if c["user_id"] == user_id]
    user_cmps = [
        (c["winner_run_id"], c["loser_run_id"])
        for c in all_user_cmps
    ]
    if len(user_cmps) >= MIN_COMPARISONS:
        user_conv = conv_weights.get(user_id, {})
        wts       = [1.0 + user_conv.get(w, 0.0) for w, _ in user_cmps]
        bt_scores = fit_bradley_terry(user_cmps, wts)
    else:
        bt_scores = global_bt

    # Only use BT signal for candidates that actually have a score
    # For candidates with no BT score, blend weight shifts fully to confidence
    raw_bt = [bt_scores.get(rid, 0.0) for rid in passage_candidates]
    max_bt = max(raw_bt) if max(raw_bt) > 0 else 1.0
    has_bt_signal = max_bt > 0 and any(bt_scores.get(rid, 0.0) > 0 for rid in passage_candidates)

    conf_w, bt_w = blend_weights(len(all_user_cmps))
    # If no BT signal for these candidates, use confidence only
    if not has_bt_signal:
        conf_w, bt_w = 1.0, 0.0

    ranked = []
    for rid in passage_candidates:
        run        = runs.get(rid, {})
        confidence = float(run.get("confidence", 0.0))
        bt_norm    = bt_scores.get(rid, 0.0) / max_bt if has_bt_signal else 0.0
        blend      = round(conf_w * confidence + bt_w * bt_norm, 4)

        user_a     = str(run.get("user_a", ""))
        user_b     = str(run.get("user_b", ""))
        match_user = user_b if user_a == str(user_id) else user_a

        ranked.append({
            "run_id":       int(rid),
            "match_user":   int(match_user) if match_user else 0,
            "verdict":      run.get("dominant_think", ""),
            "confidence":   confidence,
            "bt_score":     round(bt_scores.get(rid, 0.0), 6),
            "blend_score":  blend,
            "weights_used": {"conf": conf_w, "bt": bt_w, "n_comparisons": len(all_user_cmps)},
        })

    ranked.sort(key=lambda x: x["blend_score"], reverse=True)
    return ranked[:k]


# ── BQ writer ─────────────────────────────────────────────────────────────────

def write_rankings_to_bq(all_rows: list[dict]) -> None:
    if not all_rows:
        print("  No rows to insert")
        return

    table_ref = f"{PROJECT}.{DATASET}.rankings"

    # Note: no DELETE here — BQ streaming buffer blocks DML.
    # get_rankings() deduplicates on read using ROW_NUMBER().

    chunk_size = 500
    for i in range(0, len(all_rows), chunk_size):
        chunk  = all_rows[i:i + chunk_size]
        errors = client.insert_rows_json(table_ref, chunk)
        if errors:
            print(f"  INSERT errors: {errors[:3]}")
        else:
            print(f"  Inserted {min(i + chunk_size, len(all_rows))}/{len(all_rows)} rows")


# ── Single-user refit (called from main.py background task) ──────────────────

def refit_user(
    user_id: str,
    book_id: str = None,
    passage_id: str = None,
    k: int = 5,
) -> None:
    """
    Refit BT model for a single user and write fresh rankings to BQ.
    Called as a background task from main.py after feedback is recorded.
    Reads compat runs, comparisons, and conversations from BQ.
    """
    print(f"[BT] refitting rankings for {user_id}")

    # Load only runs involving this user
    uid_filter = f"(CAST(user_a AS STRING) = '{user_id}' OR CAST(user_b AS STRING) = '{user_id}')"
    if book_id:
        uid_filter += f" AND book_id = '{book_id}'"
    if passage_id:
        uid_filter += f" AND passage_id = '{passage_id}'"

    rows = _query(f"""
        SELECT
            CAST(run_id AS STRING)  AS run_id,
            CAST(user_a AS STRING)  AS user_a,
            CAST(user_b AS STRING)  AS user_b,
            book_id, passage_id, dominant_think, confidence
        FROM `{PROJECT}.{DATASET}.compatibility_results`
        WHERE {uid_filter}
    """)

    if not rows:
        print(f"[BT] no runs found for {user_id}, skipping")
        return

    runs_dict = {r["run_id"]: r for r in rows}
    candidate_ids = list(runs_dict.keys())

    comparisons  = load_comparisons()
    conv_weights = load_conv_weights()

    # Global BT as cold-start fallback
    all_cmps   = [(c["winner_run_id"], c["loser_run_id"]) for c in comparisons]
    all_conv: dict[str, float] = {}
    for user_conv in conv_weights.values():
        for rid, score in user_conv.items():
            all_conv[rid] = max(all_conv.get(rid, 0.0), score)
    global_bt = fit_bradley_terry(all_cmps, [1.0 + all_conv.get(w, 0.0) for w, _ in all_cmps])

    # Get unique passages in candidate set
    passages: set[tuple[str, str]] = set()
    for r in runs_dict.values():
        passages.add((r["book_id"], r["passage_id"]))

    table_ref = f"{PROJECT}.{DATASET}.rankings"
    for b_id, p_id in passages:
        ranked = rerank_for_user(
            user_id=user_id,
            candidate_run_ids=candidate_ids,
            runs=runs_dict,
            comparisons=comparisons,
            conv_weights=conv_weights,
            global_bt=global_bt,
            book_id=b_id,
            passage_id=p_id,
            k=k,
        )
        if not ranked:
            continue

        # Delete stale rows for this user+book+passage before inserting fresh ones.
        # If rows were just inserted (streaming buffer), BQ blocks DELETE —
        # in that case we skip and let get_rankings() deduplicate via ROW_NUMBER.
        try:
            client.query(
                f"DELETE FROM `{PROJECT}.{DATASET}.rankings` "
                f"WHERE CAST(user_id AS STRING) = '{str(user_id)}' "
                f"AND book_id = '{b_id}' AND passage_id = '{p_id}'"
            ).result()
        except Exception as e:
            if "streaming buffer" in str(e).lower():
                print(f"[BT] streaming buffer active for {user_id}/{b_id}/{p_id}, skipping delete")
            else:
                raise

        rows_to_insert = []
        for pos, r in enumerate(ranked, 1):
            wu = r.pop("weights_used", {})
            rows_to_insert.append({
                "user_id":       str(user_id),
                "book_id":       b_id,
                "passage_id":    p_id,
                "rank_position": pos,
                "run_id":        r["run_id"],
                "match_user":    r["match_user"],
                "verdict":       r["verdict"],
                "confidence":    r["confidence"],
                "bt_score":      r["bt_score"],
                "blend_score":   r["blend_score"],
                "conf_weight":   wu.get("conf", 0.4),
                "bt_weight":     wu.get("bt", 0.6),
                "n_comparisons": wu.get("n_comparisons", 0),
                "generated_at":  datetime.utcnow().isoformat(),
            })

        errors = client.insert_rows_json(table_ref, rows_to_insert)
        if errors:
            print(f"[BT] insert errors for {user_id}/{b_id}/{p_id}: {errors[:2]}")
        else:
            print(f"[BT] saved {len(rows_to_insert)} rankings for {user_id} / {b_id} / {p_id}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading data from BQ ({PROJECT}.{DATASET})...")
    runs         = load_compat_runs()
    comparisons  = load_comparisons()
    conv_weights = load_conv_weights()

    users: set[str] = set()
    for r in runs.values():
        users.add(str(r["user_a"]))
        users.add(str(r["user_b"]))

    passages: set[tuple[str, str]] = set()
    for r in runs.values():
        passages.add((r["book_id"], r["passage_id"]))

    print(f"\n  {len(users)} users | {len(passages)} passages\n")

    print("Fitting global Bradley-Terry model...")
    all_cmps = [(c["winner_run_id"], c["loser_run_id"]) for c in comparisons]
    all_conv: dict[str, float] = {}
    for user_conv in conv_weights.values():
        for rid, score in user_conv.items():
            all_conv[rid] = max(all_conv.get(rid, 0.0), score)
    global_wts = [1.0 + all_conv.get(w, 0.0) for w, _ in all_cmps]
    global_bt  = fit_bradley_terry(all_cmps, global_wts)
    print(f"  Global BT covers {len(global_bt):,} runs\n")

    user_runs: dict[str, list[str]] = defaultdict(list)
    for rid, r in runs.items():
        user_runs[str(r["user_a"])].append(rid)
        user_runs[str(r["user_b"])].append(rid)

    print("Reranking...")
    all_rows = []
    for user_id in sorted(users):
        candidate_ids = user_runs[user_id]
        for book_id, passage_id in sorted(passages):
            ranked = rerank_for_user(
                user_id=user_id,
                candidate_run_ids=candidate_ids,
                runs=runs,
                comparisons=comparisons,
                conv_weights=conv_weights,
                global_bt=global_bt,
                book_id=book_id,
                passage_id=passage_id,
                k=K,
            )
            for pos, r in enumerate(ranked, 1):
                wu = r.pop("weights_used", {})
                all_rows.append({
                    "user_id":       str(user_id),
                    "book_id":       book_id,
                    "passage_id":    passage_id,
                    "rank_position": pos,
                    "run_id":        r["run_id"],
                    "match_user":    r["match_user"],
                    "verdict":       r["verdict"],
                    "confidence":    r["confidence"],
                    "bt_score":      r["bt_score"],
                    "blend_score":   r["blend_score"],
                    "conf_weight":   wu.get("conf", 0.4),
                    "bt_weight":     wu.get("bt", 0.6),
                    "n_comparisons": wu.get("n_comparisons", 0),
                    "generated_at":  datetime.utcnow().isoformat(),
                })

    print(f"\n  Generated {len(all_rows):,} ranking rows")
    print("\nWriting to BQ...")
    write_rankings_to_bq(all_rows)
    print("\nDone.")


if __name__ == "__main__":
    main()