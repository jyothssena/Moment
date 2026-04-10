"""
run_experiment.py
─────────────────
Entry point for running the full Momento compatibility pipeline
with MLflow tracking.

Usage
─────
  # Run everything (all books × passages × user pairs)
  python experiment_tracking/run_experiment.py

  # Run for a single book and passage
  python experiment_tracking/run_experiment.py \
      --book "Frankenstein" --passage "passage_1"

  # Replay from existing compatibility_runs.json (no Gemini calls)
  python experiment_tracking/run_experiment.py --replay

How it works
────────────
  Normal mode  → calls run_compatibility_agent() for each pair, which
                 triggers the decomposer + scorer + aggregator, then
                 logs everything to MLflow.

  Replay mode  → reads your existing compatibility_runs.json +
                 decompositions.json and logs them to MLflow without
                 making any new Gemini calls. Useful for logging runs
                 you already have.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# ── allow imports from repo root ─────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiment_tracking.MLflow_logger import setup_mlflow, load_config, log_compatibility_run, log_batch_artifacts


# ── Replay mode ───────────────────────────────────────────────────────────────

def replay_from_existing(config: dict, book_filter: str | None, passage_filter: str | None) -> None:
    """
    Log all runs already saved in compatibility_runs.json + decompositions.json
    to MLflow — no Gemini calls made.
    """
    data_cfg = config["data"]

    compat_path = REPO_ROOT / data_cfg["compat_runs"]
    decomp_path = REPO_ROOT / data_cfg["decompositions"]

    if not compat_path.exists():
        print(f"[replay] compatibility_runs not found at {compat_path}")
        return

    with open(compat_path) as f:
        compat_runs = json.load(f)

    decompositions = []
    if decomp_path.exists():
        with open(decomp_path) as f:
            decompositions = json.load(f)

    # index decompositions by (user_id, passage_id, book_id) for fast lookup
    decomp_index = {
        (d["user_id"], d["passage_id"], d.get("book_id", "")): d
        for d in decompositions
    }

    total = 0
    skipped = 0

    for run in compat_runs:
        book_id    = run.get("book_id", "")
        passage_id = run.get("passage_id", "")

        if book_filter    and book_id    != book_filter:    continue
        if passage_filter and passage_id != passage_filter: continue

        user_a = run.get("user_a", "")
        user_b = run.get("user_b", "")

        decomp_a = decomp_index.get((user_a, passage_id, book_id))
        decomp_b = decomp_index.get((user_b, passage_id, book_id))

        if not decomp_a or not decomp_b:
            print(f"[replay] skipping {user_a} × {user_b} / {passage_id} — "
                  f"missing decomposition(s)")
            skipped += 1
            continue

        run_id = log_compatibility_run(run, decomp_a, decomp_b, config)
        print(f"[replay] logged {user_a} × {user_b} / {passage_id} → run {run_id}")
        total += 1

    print(f"\n[replay] done — {total} runs logged, {skipped} skipped")

    # attach raw file snapshots to a summary artifact run
    log_batch_artifacts(str(compat_path), str(decomp_path))


# ── Live mode ─────────────────────────────────────────────────────────────────

def run_live(config: dict, book_filter: str | None, passage_filter: str | None) -> None:
    """
    Run the full pipeline (decompose → score → aggregate) for each pair
    and log results to MLflow as they complete.
    """
    from models.compatibility_agent import run_compatibility_agent
    from models.decomposing_agent import DECOMPOSITIONS_FILE
    from tools import _read_json_file

    data_cfg = config["data"]
    interp_path = REPO_ROOT / data_cfg["interpretations"]

    if not interp_path.exists():
        print(f"[live] interpretations file not found at {interp_path}")
        return

    with open(interp_path) as f:
        moments = json.load(f)

    books    = [book_filter]    if book_filter    else list({m["book_id"]    for m in moments if "book_id"    in m})
    passages = [passage_filter] if passage_filter else ["passage_1", "passage_2", "passage_3"]

    for book in books:
        for passage_id in passages:
            moments_map = {}
            for m in moments:
                uid = m["character_name"]
                if m.get("passage_id") == passage_id and uid not in moments_map:
                    moments_map[uid] = m

            users   = list(moments_map.keys())
            checked = set()

            for i, user_a in enumerate(users):
                for user_b in users[i + 1:]:
                    if (user_a, user_b) in checked:
                        continue
                    checked.add((user_a, user_b))

                    print(f"\n── {user_a} × {user_b} | {book} / {passage_id} ──")
                    result = run_compatibility_agent(
                        user_a, user_b, book,
                        moments_map[user_a], moments_map[user_b],
                    )

                    if "error" in result:
                        print(f"  [live] error: {result['error']} — skipping MLflow log")
                        continue

                    # fetch decompositions that were just written
                    all_decomps = _read_json_file(str(REPO_ROOT / data_cfg["decompositions"]), []) or []
                    decomp_index = {
                        (d["user_id"], d["passage_id"], d.get("book_id", "")): d
                        for d in all_decomps
                    }
                    decomp_a = decomp_index.get((user_a, passage_id, book))
                    decomp_b = decomp_index.get((user_b, passage_id, book))

                    if not decomp_a or not decomp_b:
                        print(f"  [live] decompositions not found after run — skipping MLflow log")
                        continue

                    run_id = log_compatibility_run(result, decomp_a, decomp_b, config)
                    print(f"  [live] MLflow run logged: {run_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Momento compatibility pipeline with MLflow tracking"
    )
    parser.add_argument("--book",    type=str, default=None,
                        help="Filter to a single book (e.g. 'Frankenstein')")
    parser.add_argument("--passage", type=str, default=None,
                        help="Filter to a single passage (e.g. 'passage_1')")
    parser.add_argument("--replay",  action="store_true",
                        help="Replay from existing JSON files — no Gemini calls")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config()

    setup_mlflow(config)

    if args.replay:
        print("[run_experiment] REPLAY mode — reading existing JSON files")
        replay_from_existing(config, args.book, args.passage)
    else:
        print("[run_experiment] LIVE mode — running full pipeline")
        run_live(config, args.book, args.passage)

    print("\n[run_experiment] done. Open the UI with:")
    print("  mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    main()