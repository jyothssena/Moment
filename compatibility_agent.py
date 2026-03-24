import json
import re
import os
import textwrap
from datetime import datetime
from google import genai
from google.genai import types

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

_COMPAT_SYSTEM_PROMPT = textwrap.dedent("""
You are the Moment Compatibility Scorer for Momento.
You will be given two decomposed reader profiles — sub-claims with
weights and emotional modes already assigned — along with the
original moment text for each reader.
Use the original moment texts as context when scoring matched pairs,
especially when judging whether positions point in the same direction
(Think Q2) or whether readers share the same emotional experience
(Feel Q4). The decomposed sub-claims define the structure.
The original texts provide the interpretive framing.
Map sub-claims across readers, then score each matched pair.
Output only the final JSON — no preamble, no markdown fences.

════════════════════════════════════════════════════════════
STEP 1 — MAP USING THREE GATES
════════════════════════════════════════════════════════════

For each sub-claim in A, find the best candidate in B.
All three gates must pass for a match.

GATE 1 — Shared subject
Both sub-claims make a claim about the same specific subject?
(same character, same named aspect, same event in the passage)

AUTOMATIC YES: if both reference the same specific phrase from
the passage text — different angles on the same phrase are Gate 2
and Gate 3 questions, not different subjects.

Note: character evasion ≠ authorial choice even if both about language
Note: pre-creation moment ≠ post-creation moment even if same character
YES → Gate 2   |   NO → UNMATCHED

GATE 2 — Shared textual anchor
At least one applies:
  a. Both reference the same quoted phrase from the passage
  b. Both describe the same specific moment or event in the passage
     (same action, same beat — even if described in different words
     or anchored to different passage phrases)
  c. Both sub-claims respond to the same specific character behaviour
     or authorial choice AND make claims that could directly agree or
     disagree with each other.
     Note: "same character" alone is not sufficient for 2c — must be
     the same specific behaviour or action of that character.
     Passes: both about whether Victor's reaction to the creature was
     justified (same action, opposing verdicts possible).
     Fails: one about Victor's language, one about Victor's emotional
     state — different aspects, not the same behaviour.
YES → Gate 3   |   NO → UNMATCHED

GATE 3 — Meaningful comparison
Formulate: "Is it true that [A's claim]?"
Would B's claim answer YES or NO?
YES or NO → MATCHED   |   Irrelevant → UNMATCHED

Each sub-claim can only be matched once.
Any B sub-claim not claimed by a matched A sub-claim → UNMATCHED.

════════════════════════════════════════════════════════════
STEP 2 — SCORE EACH MATCHED PAIR
════════════════════════════════════════════════════════════

For each matched pair, answer 5 Think questions then 5 Feel questions.
Each YES or NO maps to 1R or 1C as specified.
score = points / 5  (D = 0.0 for matched pairs)

THINK QUESTIONS:
  Q1. Same subject? Always YES for matched pairs → 1R
  Q2. Positions same direction? YES=1R  NO=1C
  Q3. Interpretive lenses compatible? YES=1R  NO=1C
  Q4. Conclusions mutually exclusive? YES=1C  NO=1R
  Q5. Would A agree with B's conclusion? YES=1R  NO=1C

FEEL QUESTIONS:
  Q1. Same emotional subject? Always YES for matched pairs → 1R
  Q2. Same emotional mode TYPE? Compare the assigned modes —
      same label → YES (1R)  |  different labels → NO (1C)
  Q3. Same specific trigger? YES=1R  NO=1C
  Q4. Same emotional experience? YES=1R  NO=1C
  Q5. Would A recognise B's response as valid? YES=1R  NO=1C

For UNMATCHED sub-claims:
  think: R=0.0  C=0.0  D=1.0
  feel:  R=0.0  C=0.0  D=1.0

════════════════════════════════════════════════════════════
GATE CONFIDENCE — assign for every matched pair
════════════════════════════════════════════════════════════

You must assign gate_confidence for every matched pair.
  1.00 = both gates passed clearly
  0.75 = one gate required interpretation
  0.50 = borderline match
  If zero matched pairs → omit gate_confidence entirely.

════════════════════════════════════════════════════════════
OUTPUT — return ONLY this JSON, no preamble, no markdown fences
════════════════════════════════════════════════════════════

{
  "passage_id": "<passage identifier>",
  "matched_pairs": [
    {
      "a_id": "<sub-claim id from A>",
      "b_id": "<sub-claim id from B>",
      "weight_a": <float>,
      "weight_b": <float>,
      "gate_confidence": <1.00|0.75|0.50>,
      "think": {"R": <float>, "C": <float>, "D": 0.0},
      "feel":  {"R": <float>, "C": <float>, "D": 0.0}
    }
  ],
  "unmatched_a": ["<sub-claim id>"],
  "unmatched_b": ["<sub-claim id>"]
}
""")

_SCORER_REQUIRED_KEYS = {"matched_pairs", "unmatched_a", "unmatched_b"}

# ── Helpers ───────────────────────────────────────────────────────────────────

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
    prompt  = _build_scorer_prompt(decomp_a, decomp_b)
    scoring = _call_scorer(prompt)
    _save_scoring(user_a_id, user_b_id, passage_id, scoring)

    if "error" in scoring:
        return {"error": scoring["error"],
                "user_a": user_a_id, "user_b": user_b_id}

    # ── Aggregate (pure Python) ───────────────────────────────────────────────
    # wrap the two separate decompositions into the shape aggregate() expects
    combined_decomp = {
        "reader_a": decomp_a,
        "reader_b": decomp_b,
    }
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

def run_compatibility_for_all(user_a_id: str,
                               book_id: str,
                               passage_id: str,
                               moments_map: dict[str, dict]) -> list[dict]:
    moment_a    = moments_map.get(user_a_id)
    other_users = [uid for uid in moments_map if uid != user_a_id]

    if not moment_a:
        print(f"[main] no moment provided for anchor user {user_a_id}")
        return []
    if not other_users:
        print(f"[main] no other users found in moments_map for book_id={book_id}")
        return []

    print(f"[main] found {len(other_users)} other readers of {book_id}")

    results = []
    for user_b_id in other_users:
        moment_b = moments_map.get(user_b_id)
        if not moment_b:
            print(f"  {user_b_id} — skipped (no moment provided)")
            continue

        existing = _get_existing_compat_run(user_a_id, user_b_id, book_id,passage_id)
        if existing:
            route = route_compatibility_result(existing)
            existing["route"] = route
            print(f"  {user_b_id} [cached] verdict={existing.get('verdict')} "
                  f"confidence={existing.get('confidence')}")
            if route != "discard":
                results.append(existing)
            continue

        result        = run_compatibility_agent(user_a_id, user_b_id, book_id, moment_a, moment_b)
        route         = route_compatibility_result(result)
        result["route"] = route

        print(f"  {user_b_id} — dominant_think={result.get('dominant_think')} "
              f"dominant_feel={result.get('dominant_feel')} "
              f"confidence={result.get('confidence')} route={route}")

        if route != "discard":
            results.append(result)

    results.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
    print(f"\n── Match pool for {user_a_id}: {len(results)} / {len(other_users)} ──")

    _save_compatibility_results(user_a_id, book_id, results)
    return results


COMPAT_RESULTS_FILE = "data/processed/compatibility_results.json"
SCORING_FILE        = "data/processed/scoring_runs.json"


def _save_scoring(user_a_id: str, user_b_id: str, passage_id: str, scoring: dict) -> None:
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
    with open("interpretations_10_users.json") as f:
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