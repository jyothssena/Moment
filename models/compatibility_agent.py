import json
import re
import os
import textwrap
from datetime import datetime
from google import genai
from google.genai import types # type: ignore

from tools import (
    COMPAT_LOG_FILE,
    _read_json_file,
    _write_json_file,
    get_user_interpretations,
    get_user_profile,
    count_new_moments,
    log_compatibility_run,
    extract_json,
)
from decomposing_agent import (
    _gemini_client,
    _GEMINI_MODEL,
    _get_response_text,
    run_decomposer,
    DECOMPOSITIONS_FILE
)
from aggregator import aggregate

# ── Compatibility Agent ───────────────────────────────────────────────────────

_COMPAT_SYSTEM_PROMPT = """
You are the Moment Compatibility Scorer.
You receive pre-decomposed sub-claims from two readers (A and B) for the same
passage. Map them, score each pair, return JSON.

════════════════════════════════════════════════════════════
CRITICAL — THINK AND FEEL ARE INDEPENDENT
════════════════════════════════════════════════════════════

Score them separately. Two readers can share a conclusion but feel differently.
Two readers can disagree intellectually but share the same emotional response.
Never let Think scores contaminate Feel scores.

════════════════════════════════════════════════════════════
STEP 1 — MAP SUB-CLAIMS 
════════════════════════════════════════════════════════════

Match each A sub-claim to the best B candidate.
Two sub-claims match if they respond to the same passage phrase or the same
specific moment — even if worded differently.
If no genuine match exists, mark UNMATCHED.
Each sub-claim matched at most once. Unclaimed B sub-claims → UNMATCHED.

════════════════════════════════════════════════════════════
STEP 2 — SCORE EACH MATCHED PAIR
════════════════════════════════════════════════════════════

For each matched pair answer 10 booleans — 5 THINK then 5 FEEL.

THINK — T1. Same subject (always true) | T2. Positions same direction? |
  T3. Lenses compatible? | T4. Mutually exclusive conclusions? | T5. Would A agree with B?

FEEL  — F1. Same emotional subject (always true) | F2. Same mode label? |
  F3. Same trigger? | F4. Same experience? | F5. Would A recognise B's response as valid?


════════════════════════════════════════════════════════════
UNMATCHED SUB-CLAIMS
════════════════════════════════════════════════════════════

B engages the same subject but not the same specific phrase or moment → divergence: true
B has nothing on this subject                                         → divergence: false

════════════════════════════════════════════════════════════
OUTPUT — raw JSON only, no markdown, no preamble
════════════════════════════════════════════════════════════

{
  "passage_id": "<id>",
  "matched_pairs": [
    {
      "a_id": "<id>", "b_id": "<id>",
      "weight_a": <float>, "weight_b": <float>,
      "gate_confidence": 1.0 or 0.5,
      "think_q": [<bool>, <bool>, <bool>, <bool>, <bool>],
      "feel_q":  [<bool>, <bool>, <bool>, <bool>, <bool>]
    }
  ],
  "unmatched_a": [{"id": "<id>", "divergence": <bool>],
  "unmatched_b": [{"id": "<id>", "divergence": <bool>],
  "think_rationale": "<1-2 sentences>",
  "feel_rationale": "<1-2 sentences>"
}
"""

_SCORER_REQUIRED_KEYS = {"matched_pairs", "unmatched_a", "unmatched_b"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_existing_scoring(user_a_id: str, user_b_id: str, passage_id: str,book_id: str) -> dict | None:
    data = _read_json_file(SCORING_FILE, []) or []
    match = next(
        (d for d in data
         if d["passage_id"] == passage_id and d["book_id"] == book_id
         and (
             (d["user_a_id"] == user_a_id and d["user_b_id"] == user_b_id)
             or
             (d["user_a_id"] == user_b_id and d["user_b_id"] == user_a_id)
         )),
        None
    )
    return match["scoring"] if match else None

def _get_or_run_decomposition(user_id: str, passage_id: str, book_id: str, moment_text: str) -> dict:
    """
    Return cached decomposition for this user+passage+book if it exists,
    otherwise run the decomposer and return the result.
    """
    data = _read_json_file(DECOMPOSITIONS_FILE,[]) or []
    cached = next(
        (d for d in data
         if d["user_id"] == user_id
         and d["passage_id"] == passage_id
         and d.get("book_id") == book_id),
        None
    )
    if cached:
        print(f"[CompatAgent] using cached decomposition for {user_id} / {passage_id} / {book_id}")
        return cached
    return run_decomposer(user_id, passage_id, book_id, moment_text)

def _build_scorer_prompt(decomp_a: dict, decomp_b: dict) -> str:
    """Build the user-turn message for the scorer from two decompositions."""
    return textwrap.dedent(f"""
        Reader A:
        {json.dumps(decomp_a, indent=2)}

        Reader B:
        {json.dumps(decomp_b, indent=2)}

        Map and score these two decomposed moments.
        Return a raw JSON object.
    """)


def _call_scorer(prompt: str) -> dict:
    """Send decomposed profiles to the scorer and return parsed result."""
    response = _gemini_client.models.generate_content(
        model=_GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=_COMPAT_SYSTEM_PROMPT,
            temperature=0.1,
        ),
        contents=prompt,
    )
    result = extract_json(_get_response_text(response))

    if "error" in result:
        raw = _get_response_text(response) or ""
        print(f"[CompatAgent] JSON extraction failed. raw={raw[:200]}")
        return {"error": "invalid JSON from Compatibility Scorer", "raw": raw}

    missing = _SCORER_REQUIRED_KEYS - result.keys()
    if missing:
        print(f"[CompatAgent] incomplete result, missing keys: {missing}")
        result["error"] = f"incomplete result, missing: {missing}"

    return result



# ── Agent ─────────────────────────────────────────────────────────────────────

def run_compatibility_agent(user_a_id: str,
                             user_b_id: str,
                             book_id: str,
                             moment_a: dict,
                             moment_b: dict) -> dict:
    """
    Evaluate compatibility between two readers on the same passage.

    Decomposes each moment (or uses cache), scores matched pairs,
    then aggregates into final R/C/D percentages + confidence.

    Returns a dict with keys: think, feel, dominant_think, dominant_feel,
    verdict, confidence, user_a, user_b, book_id, timestamp.
    """
    print(f"[CompatAgent] evaluating {user_a_id} × {user_b_id} on book={book_id}")

    passage_id   = moment_a.get("passage_id", moment_b.get("passage_id", "unknown"))
    moment_a_txt = moment_a.get("interpretation", moment_a.get("text", ""))
    moment_b_txt = moment_b.get("interpretation", moment_b.get("text", ""))

    # ── Decompose ─────────────────────────────────────────────────────────────
    decomp_a = _get_or_run_decomposition(user_a_id, passage_id, book_id, moment_a_txt)
    decomp_b = _get_or_run_decomposition(user_b_id, passage_id, book_id, moment_b_txt)

    if "error" in decomp_a or "error" in decomp_b:
        failed = user_a_id if "error" in decomp_a else user_b_id
        print(f"[CompatAgent] decomposition failed for {failed}")
        return {"error": f"decomposition failed for {failed}",
                "user_a": user_a_id, "user_b": user_b_id}

    results = []
    existing = _get_existing_compat_run(user_a_id, user_b_id, book_id,passage_id)
    if existing:
        route = route_compatibility_result(existing)
        existing["route"] = route
        print(f"  {user_b_id} [cached] verdict={existing.get('verdict')} "
                  f"confidence={existing.get('confidence')}")
        if route != "discard":
            results.append(existing)
        return results
        
    # ── Score ─────────────────────────────────────────────────────────────────
    scoring = _get_existing_scoring(user_a_id, user_b_id, passage_id,book_id)
    if scoring:
        print(f"[CompatAgent] using cached scoring for {user_a_id} × {user_b_id} / {passage_id}")
    else:
        prompt  = _build_scorer_prompt(decomp_a, decomp_b)
        scoring = _call_scorer(prompt)
        _save_scoring(user_a_id, user_b_id, passage_id, scoring,book_id)

    if "error" in scoring:
        return {"error": scoring["error"],
                "user_a": user_a_id, "user_b": user_b_id}

    # ── Aggregate (pure Python) ───────────────────────────────────────────────
    # aggregator expects: aggregate(decomp, scoring)
    #   decomp  = {"reader_a": <decomp_a>, "reader_b": <decomp_b>}
    #   scoring = {"matched_pairs": [...], "unmatched_a": [...], "unmatched_b": [...], "passage_id": ...}
    combined_decomp = {"reader_a": decomp_a, "reader_b": decomp_b}
    scoring.setdefault("passage_id", passage_id)
    # Normalize unmatched lists — scorer returns [{"id": "1", ...}], aggregator expects ["1", ...]
    scoring["unmatched_a"] = [
        u["id"] if isinstance(u, dict) else u for u in scoring.get("unmatched_a", [])
    ]
    scoring["unmatched_b"] = [
        u["id"] if isinstance(u, dict) else u for u in scoring.get("unmatched_b", [])
    ]
    result = aggregate(combined_decomp, scoring)
    print(f"[CompatAgent] confidence={result.get('confidence', 0.0)}")

    # ── Attach metadata and log ───────────────────────────────────────────────
    result["user_a"]    = user_a_id
    result["user_b"]    = user_b_id
    result["book_id"]   = book_id
    result["timestamp"] = datetime.utcnow().isoformat()

    log_compatibility_run({
        "user_a":          user_a_id,
        "user_b":          user_b_id,
        "book_id":         book_id,
        "passage_id":      passage_id,
        "think":           result.get("think"),
        "feel":            result.get("feel"),
        "dominant_think":  result.get("dominant_think"),
        "dominant_feel":   result.get("dominant_feel"),
        "verdict":         result.get("verdict"),
        "confidence":      result.get("confidence"),
        "think_rationale": result.get("think_rationale"),
        "feel_rationale":  result.get("feel_rationale"),
        "timestamp":       result["timestamp"],
    })

    print(
        f"[CompatAgent] dominant_think={result.get('dominant_think')} "
        f"dominant_feel={result.get('dominant_feel')} "
        f"confidence={result.get('confidence')} "
        f"for {user_a_id} × {user_b_id}"
    ) 
    return result


# ── Uncertainty Router ────────────────────────────────────────────────────────

def route_compatibility_result(result: dict) -> str:
    if "error" in result:
        return "discard"
    return "display"


# ── Existing run cache ────────────────────────────────────────────────────────

def _get_existing_compat_run(user_a_id: str,
                              user_b_id: str,
                              book_id: str,passage_id: str) -> dict | None:
    runs = _read_json_file(COMPAT_LOG_FILE,[]) or []
    matches = [
        r for r in runs
        if r.get("book_id") == book_id and r.get("passage_id") == passage_id
        and (
            (r.get("user_a") == user_a_id and r.get("user_b") == user_b_id)
            or
            (r.get("user_a") == user_b_id and r.get("user_b") == user_a_id)
        )
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda r: r.get("timestamp", ""), reverse=True)[0]


# ── Batch runner ──────────────────────────────────────────────────────────────

COMPAT_RESULTS_FILE = "data/processed/compatibility_results.json"
SCORING_FILE        = "data/processed/scoring_runs.json"


def _save_scoring(user_a_id: str, user_b_id: str, passage_id: str, scoring: dict, book_id: str) -> None:
    """Upsert raw scorer output keyed by user_a + user_b + passage_id."""
    data = _read_json_file(SCORING_FILE,[]) or []
    key  = (user_a_id, user_b_id, passage_id)
    data = [
        d for d in data
        if (d["user_a_id"], d["user_b_id"], d["passage_id"]) != key
    ]
    data.append({
        "user_a_id":  user_a_id,
        "user_b_id":  user_b_id,
        "book_id":    book_id,
        "passage_id": passage_id,
        "timestamp":  datetime.utcnow().isoformat(),
        "scoring":    scoring,
    })
    _write_json_file(SCORING_FILE, data)
    print(f"[CompatAgent] scoring saved for {user_a_id} × {user_b_id} / {passage_id}")

def _save_compatibility_results(user_a_id: str, book_id: str, results: list[dict]) -> None:
    """Upsert batch results for this anchor user + book into compatibility_results.json."""
    data = _read_json_file(COMPAT_RESULTS_FILE,[]) or []

    # remove any previous batch for this anchor + book
    data = [
        d for d in data
        if not (d.get("user_a_id") == user_a_id and d.get("book_id") == book_id)
    ]
    data.append({
        "user_a_id": user_a_id,
        "book_id":   book_id,
        "timestamp": datetime.utcnow().isoformat(),
        "results":   results,
    })
    _write_json_file(COMPAT_RESULTS_FILE, data)
    print(f"[main] saved {len(results)} results to {COMPAT_RESULTS_FILE}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("interpretations_train.json") as f:
        moments = json.load(f)

    PASSAGE_IDS=['passage_1','passage_2','passage_3']
    BOOKS       = ["Frankenstein","Pride and Prejudice","The Great Gatsby"]
    for BOOK in BOOKS:
        for PASSAGE_ID in PASSAGE_IDS:
            moments_map = {}
            for m in moments:
                uid = m["character_name"]
                if m["passage_id"] == PASSAGE_ID and uid not in moments_map:
                    moments_map[uid] = m

            users = list(moments_map.keys())
            checked = set()

            for i, user_a in enumerate(users):
                for user_b in users[i+1:]:
                    pair = (user_a, user_b)
                    if pair in checked:
                        continue
                    checked.add(pair)

                    print(f"\n── {user_a} × {user_b} ──")
                    moment_a = moments_map[user_a]
                    moment_b = moments_map[user_b]

                    result = run_compatibility_agent(user_a, user_b, BOOK, moment_a, moment_b)
                    print(json.dumps(result, indent=2))