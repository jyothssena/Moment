import json
import re
import os
import textwrap
from datetime import datetime
from google import genai
from google.genai import types
from tools import extract_json as _extract_json, RECO_TOOLS

# ── Config ────────────────────────────────────────────────────────────────────

COMPAT_LOG_FILE  = "data/processed/compatibility_runs.json"
PROFILE_FILE     = "data/processed/profiles.json"
RECO_LOG_FILE    = "data/processed/recommendation_runs.json"
TOOL_LOG_FILE    = "data/processed/tool_call_log.json"

_api_key = os.environ.get("GEMINI_API_KEY_MOMENT")
if not _api_key:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        _api_key = os.environ.get("GEMINI_API_KEY_MOMENT")
    except ImportError:
        pass

if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY_MOMENT is not set. Export it before importing this module."
    )

_gemini_client = genai.Client(api_key=_api_key)
_GEMINI_MODEL  = "gemini-2.5-flash"

# ── Recommendation Agent ──────────────────────────────────────────────────────

_RECO_SYSTEM_PROMPT = textwrap.dedent("""
    You are the Recommendation Agent for Moment, a reading platform that
    connects intellectually compatible readers.

    Your job: produce a ranked list of recommended matches for a target user,
    drawing on compatibility verdicts and reader portraits.

    Workflow:
    1. Call get_user_profile to load the target user's portrait.
       If none exists, return {"error": "no profile for user"}.

    2. Call get_compatibility_runs(user_id, min_confidence=0.5) to load all
       evaluated pairings for this user. These are your highest-signal candidates
       — they have already been scored.

    3. Call get_all_profiles to load portraits of users not yet evaluated.
       These are cold candidates. Include them only if the evaluated pool has
       fewer than 3 strong matches (confidence >= 0.75).

    4. Rank all candidates using this priority order:
       a. Evaluated matches with confidence >= 0.75 — rank by confidence desc.
       b. Evaluated matches with 0.5 <= confidence < 0.75 — rank after (a).
       c. Cold candidates — infer rough compatibility from portrait similarity
          alone; flag these with "source": "portrait_inference".

    5. For each recommended match write:
       - match_user_id: the other user's ID
       - book_id: the book that generated the connection (null for cold candidates)
       - verdict: from the compatibility run (null for cold candidates)
       - confidence: float from the run, or your portrait-inferred estimate
       - source: "compatibility_run" | "portrait_inference"
       - reason: 1-2 sentences grounded in specific evidence from portraits
         or moments. Never be generic.

    6. Cap the list at 15 recommendations. Call save_recommendations with the
       final list.

    Return ONLY a raw JSON object — no markdown fences, no preamble.
    Keys: user_id, recommendations (list), summary (1 sentence about this
    user's match landscape).
""")

_RECO_REQUIRED_KEYS = {"user_id", "recommendations", "summary"}

def run_recommendation_agent(user_id: str) -> dict:
    """
    Produce a ranked list of recommended reader matches for user_id.
    Returns a dict with keys: user_id, recommendations, summary.
    On failure returns {"error": ..., "user_id": user_id}.
    """
    print(f"[RecoAgent] running for user_id={user_id}")

    chat = _gemini_client.chats.create(
        model=_GEMINI_MODEL,
        config=types.GenerateContentConfig(
            tools=RECO_TOOLS,
            temperature=0.2,
            system_instruction=_RECO_SYSTEM_PROMPT,
        )
    )

    response = chat.send_message(
        f"Generate recommendations for user_id: {user_id}. "
        "Follow the workflow in your instructions exactly. "
        "Final output must be a raw JSON object."
    )

    result = _extract_json(response.text)

    if "error" in result:
        print(f"[RecoAgent] JSON extraction failed. raw={response.text[:200]}")
        return {"error": "invalid JSON from Recommendation Agent",
                "raw": response.text, "user_id": user_id}

    missing = _RECO_REQUIRED_KEYS - result.keys()
    if missing:
        print(f"[RecoAgent] incomplete result, missing keys: {missing}")
        return {"error": f"incomplete result, missing: {missing}",
                "raw": result, "user_id": user_id}

    result["user_id"] = user_id
    print(f"[RecoAgent] {len(result.get('recommendations', []))} recommendations for {user_id}")
    return result


# ── Router extension ──────────────────────────────────────────────────────────

def route_recommendation_result(result: dict) -> str:
    """
    Decide what to do with a recommendation result.

    Returns one of:
      "display"     — has at least one high-confidence match
      "deep_review" — only cold/portrait-inferred candidates
      "discard"     — empty list or agent error
    """
    if "error" in result:
        return "discard"

    recommendations = result.get("recommendations", [])
    if not recommendations:
        return "discard"

    has_strong = any(
        r.get("source") == "compatibility_run" and r.get("confidence", 0) >= 0.75
        for r in recommendations
    )
    if has_strong:
        return "display"

    has_inferred = any(r.get("source") == "portrait_inference" for r in recommendations)
    if has_inferred:
        return "deep_review"

    return "deep_review"


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    USER_ID = "user_emma_chen_fd5e3def"

    result = run_recommendation_agent(USER_ID)
    route  = route_recommendation_result(result)

    print("\n── Result ───────────────────────────────────────")
    print(json.dumps(result, indent=2))
    print(f"\n── Router decision: {route} ──────────────────────")