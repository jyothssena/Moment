import json
import re
import os
import textwrap
from datetime import datetime
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE       = "data/processed/moments_processed.json"
PROFILE_FILE     = "data/processed/profiles.json"
COMPAT_LOG_FILE  = "data/processed/compatibility_runs.json"
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

try:
    with open(INPUT_FILE) as f:
        _interpretations = json.load(f)
except FileNotFoundError:
    _interpretations = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json_file(path: str, default):
    try:
        with open(path) as f:
            content = f.read()
            return json.loads(content) if content.strip() else default
    except FileNotFoundError:
        return default

def _write_json_file(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _extract_json(text: str | None) -> dict:
    """Robustly extract a JSON object from LLM output."""
    if not text or not text.strip():
        return {"error": "empty response from model", "raw": text}
    text = text.strip()
    # direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find outermost {...} block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "could not extract JSON", "raw": text}

def _get_response_text(response) -> str | None:
    """
    Safely extract text from a Gemini response.
    Returns None if the model's final action was a tool call with no text,
    or if the response structure is unexpected.
    """
    try:
        # response.text raises if no text part exists
        return response.text
    except Exception:
        pass
    # walk parts manually and collect any text blocks
    try:
        parts = response.candidates[0].content.parts
        texts = [p.text for p in parts if hasattr(p, "text") and p.text]
        return "\n".join(texts) if texts else None
    except Exception:
        return None

def _log_tool_call(tool_name: str, args: tuple, kwargs: dict, result) -> None:
    """Append a tool call record to the tool log file."""
    entry = {
        "tool":      tool_name,
        "args":      args,
        "kwargs":    kwargs,
        "result":    result,
        "timestamp": datetime.utcnow().isoformat()
    }
    log = _read_json_file(TOOL_LOG_FILE, [])
    log.append(entry)
    _write_json_file(TOOL_LOG_FILE, log)

def _logged(fn):
    """Decorator — logs every tool call to TOOL_LOG_FILE."""
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        try:
            _log_tool_call(fn.__name__, args, kwargs, result)
        except Exception as e:
            print(f"[warn] tool log failed for {fn.__name__}: {e}")
        return result
    wrapper.__name__ = fn.__name__
    wrapper.__doc__  = fn.__doc__
    return wrapper


# ── Tools ─────────────────────────────────────────────────────────────────────
# All tools are decorated with @_logged so every call is written to TOOL_LOG_FILE.
# This is your instrumentation layer and future distillation training data.

@_logged
def get_user_interpretations(user_id: str, book_id: str = None) -> list[dict]:
    """Retrieve all book annotations/moments for a user. Optionally filter by book_id."""
    results = [i for i in _interpretations if i.get("user_id") == user_id]
    if book_id:
        results = [i for i in results if i.get("book_id") == book_id]
    return results

@_logged
def get_user_profile(user_id: str) -> dict | None:
    """Retrieve the saved reader portrait for a user. Returns None if not found."""
    profiles = _read_json_file(PROFILE_FILE, [])
    for p in profiles:
        if p.get("user_id") == user_id:
            return p
    return None

@_logged
def save_user_profile(user_id: str, profile_data: dict) -> str:
    """Save or overwrite the reader portrait for a user."""
    profile_data["user_id"]      = user_id
    profile_data["last_updated"] = datetime.utcnow().isoformat()

    profiles = _read_json_file(PROFILE_FILE, [])
    for i, p in enumerate(profiles):
        if p.get("user_id") == user_id:
            profiles[i] = profile_data
            _write_json_file(PROFILE_FILE, profiles)
            return "Profile updated successfully."

    profiles.append(profile_data)
    _write_json_file(PROFILE_FILE, profiles)
    return "Profile created successfully."

@_logged
def count_new_moments(user_id: str, since_iso: str) -> int:
    """Count how many moments a user has saved after a given ISO timestamp.
    Used by the Profile Agent to decide whether an update is warranted."""
    moments = get_user_interpretations(user_id)
    return sum(1 for m in moments if m.get("timestamp", "") > since_iso)


# ── Profile Agent ─────────────────────────────────────────────────────────────

_PROFILE_SYSTEM_PROMPT = textwrap.dedent("""
    You are the Profile Agent for Moment, a reading platform that matches
    intellectually compatible readers.

    Your job: build or incrementally update a reader portrait from a user's
    book annotations (called moments or interpretations).

    Workflow:
    1. Call get_user_profile to check whether a portrait already exists.
    2. Call get_user_interpretations to retrieve the user's moments.
    3. If no portrait exists — build one from scratch using all moments.
    4. If a portrait exists:
       a. Call count_new_moments with the portrait's last_updated timestamp.
       b. If fewer than 3 new moments exist, return the existing portrait unchanged.
       c. If 3 or more new moments exist, revise only the dimensions that the
          new moments change. Keep everything else.
    5. Call save_user_profile with the final portrait dict.

    The portrait must cover exactly these four dimensions.
    Each dimension must be 2-3 sentences grounded in specific evidence
    from the user's actual reflections — not generic observations.

    Dimensions:
    - themes: what subjects and questions does this reader keep returning to?
    - emotional_register: do their reflections tend toward analysis, empathy,
      argumentation, or surrender? Give a brief example.
    - engagement_style: do they read for structure, feeling, identification
      with characters, or friction with ideas?
    - reflection_density: how precisely and deeply do they articulate responses?
      Quote a short phrase to illustrate.

    For updates, include a "revision_notes" key explaining what changed and why.

    Return ONLY a raw JSON object — no markdown fences, no preamble, no explanation.
    Keys: themes, emotional_register, engagement_style, reflection_density,
    and optionally revision_notes.
""")
_PROFILE_REQUIRED_KEYS = {
    "themes", "emotional_register", "engagement_style", "reflection_density"
}

def run_profile_agent(user_id: str) -> dict:
    """
    Build or incrementally update a reader portrait for user_id.
    Returns the portrait dict. On failure returns {"error": ..., "user_id": user_id}.
    """
    print(f"[ProfileAgent] running for user_id={user_id}")

    chat = _gemini_client.chats.create(
        model=_GEMINI_MODEL,
        config=types.GenerateContentConfig(
            tools=[
                get_user_interpretations,
                get_user_profile,
                save_user_profile,
                count_new_moments,
            ],
            temperature=0.2,
            system_instruction=_PROFILE_SYSTEM_PROMPT,
        )
    )

    response = chat.send_message(
        f"Build or update the portrait for user_id: {user_id}. "
        "Follow the workflow in your instructions exactly. "
        "Final output must be a raw JSON object."
    )

    portrait = _extract_json(_get_response_text(response))

    if "error" in portrait:
        raw = _get_response_text(response) or ""
        print(f"[ProfileAgent] JSON extraction failed. raw={raw[:200]}")
        return {"error": "invalid JSON from Profile Agent",
                "raw": response.text, "user_id": user_id}

    missing = _PROFILE_REQUIRED_KEYS - portrait.keys()
    if missing:
        print(f"[ProfileAgent] incomplete portrait, missing keys: {missing}")
        return {"error": f"incomplete portrait, missing: {missing}",
                "raw": portrait, "user_id": user_id}

    portrait["user_id"] = user_id
    print(f"[ProfileAgent] portrait ready for user_id={user_id}")
    return portrait


# ── Compatibility Agent ───────────────────────────────────────────────────────

_COMPAT_SYSTEM_PROMPT = textwrap.dedent("""
    You are the Compatibility Investigator for Moment, a platform that matches
    readers of the same book based on how they intellectually engage with it.

    You will be given:
    - Two pre-built reader portraits (Portrait A and Portrait B)
    - The book_id being compared

    Your job is to investigate whether these two readers would have a meaningful
    intellectual connection. Use get_user_interpretations to examine their actual
    moments for this specific book before forming any verdict.

    Reasoning steps — follow in order:
    1. Read both portraits carefully. Are they fundamentally incompatible?
       If yes, set verdict to "no_match" and stop.

    2. Retrieve both users' moments for the book using get_user_interpretations.
       Look for passages they both engaged with, or passages where their
       reflections reveal a structural relationship.

    3. Determine the nature of the connection:
       - resonance: similar emotional and intellectual stance toward the book
       - mirror: opposite sides of the same dynamic (e.g. one identifies with
         character A, the other with character B in the same conflict)
       - contradiction: genuinely different interpretive frameworks that could
         produce productive tension
       - no_match: no meaningful basis for connection

    4. Assign a confidence score (0.0–1.0) reflecting how strongly the evidence
       supports your verdict.

    5. Write an insight — 2-3 sentences a real user would see. Be specific.
       Reference actual passages or reflections. Do not be generic.
       Set to null if verdict is no_match.

    Return ONLY a raw JSON object — no markdown fences, no preamble.
    Keys: verdict, confidence, reasoning, insight.
""")

_COMPAT_REQUIRED_KEYS = {"verdict", "confidence", "reasoning", "insight"}

def _log_compatibility_run(record: dict) -> None:
    runs = _read_json_file(COMPAT_LOG_FILE, [])
    runs.append(record)
    _write_json_file(COMPAT_LOG_FILE, runs)

def _ensure_portrait(user_id: str) -> dict:
    """
    Return a portrait for user_id. Uses the cached portrait if it exists
    and is fresh (fewer than 3 new moments since last update).
    Only calls run_profile_agent if no portrait exists or it is stale.
    """
    existing = get_user_profile(user_id)

    if existing:
        last_updated = existing.get("last_updated", "")
        new_count    = count_new_moments(user_id, last_updated)
        if new_count < 3:
            print(f"[ProfileAgent] using cached portrait for {user_id} "
                  f"({new_count} new moments, threshold=3)")
            return existing
        print(f"[ProfileAgent] portrait stale for {user_id} "
              f"({new_count} new moments) — rebuilding")
        return existing

    return run_profile_agent(user_id)


def run_compatibility_agent(user_a_id: str,
                             user_b_id: str,
                             book_id: str) -> dict:
    """
    Evaluate intellectual compatibility between two readers of the same book.
    Returns a dict with keys: verdict, confidence, reasoning, insight,
    user_a, user_b, book_id, timestamp.
    """
    print(f"[CompatAgent] evaluating {user_a_id} × {user_b_id} on book={book_id}")

    # ── Step 1: get portraits without triggering unnecessary agent runs ───────
    portrait_a = _ensure_portrait(user_a_id)
    portrait_b = _ensure_portrait(user_b_id)

    if "error" in portrait_a:
        return {"error": f"Portrait failed for user_a: {portrait_a['error']}",
                "user_a": user_a_id, "user_b": user_b_id, "book_id": book_id}

    if "error" in portrait_b:
        return {"error": f"Portrait failed for user_b: {portrait_b['error']}",
                "user_a": user_a_id, "user_b": user_b_id, "book_id": book_id}

    # ── Step 2: run the Compatibility Investigator ────────────────────────────
    # Portraits are passed in the prompt — the agent uses get_user_interpretations
    # to fetch moment-level evidence independently.
    chat = _gemini_client.chats.create(
        model=_GEMINI_MODEL,
        config=types.GenerateContentConfig(
            tools=[get_user_interpretations],
            temperature=0.2,
            system_instruction=_COMPAT_SYSTEM_PROMPT,
        )
    )

    prompt = textwrap.dedent(f"""
        Investigate compatibility between these two readers for book_id: {book_id}.

        User A ID: {user_a_id}
        Portrait A:
        {json.dumps(portrait_a, indent=2)}

        User B ID: {user_b_id}
        Portrait B:
        {json.dumps(portrait_b, indent=2)}

        Follow the reasoning steps in your instructions.
        Retrieve their actual moments before forming a verdict.
        Return a raw JSON object.
    """)

    response = chat.send_message(prompt)
    result   = _extract_json(_get_response_text(response))

    if "error" in result:
        print(f"[CompatAgent] JSON extraction failed. raw={response.text[:200]}")
        result = {"error": "invalid JSON from Compatibility Agent",
                  "raw": response.text}

    missing = _COMPAT_REQUIRED_KEYS - result.keys()
    if missing:
        print(f"[CompatAgent] incomplete result, missing keys: {missing}")
        result["error"] = f"incomplete result, missing: {missing}"

    # ── Step 3: attach metadata and log ──────────────────────────────────────
    result["user_a"]    = user_a_id
    result["user_b"]    = user_b_id
    result["book_id"]   = book_id
    result["timestamp"] = datetime.utcnow().isoformat()

    _log_compatibility_run({
        "user_a":      user_a_id,
        "user_b":      user_b_id,
        "book_id":     book_id,
        "portrait_a":  portrait_a,
        "portrait_b":  portrait_b,
        "verdict":     result.get("verdict"),
        "confidence":  result.get("confidence"),
        "reasoning":   result.get("reasoning"),
        "insight":     result.get("insight"),
        "timestamp":   result["timestamp"],
    })

    print(
        f"[CompatAgent] verdict={result.get('verdict')} "
        f"confidence={result.get('confidence')} "
        f"for {user_a_id} × {user_b_id}"
    )
    return result


# ── Uncertainty Router ────────────────────────────────────────────────────────

def route_compatibility_result(result: dict) -> str:
    """
    Decide what to do with a compatibility verdict.

    Returns one of:
      "display"     — high confidence match, show to users
      "deep_review" — low confidence or uncertain, flag for review
      "discard"     — no match or agent error
    """
    if "error" in result:
        return "discard"

    verdict    = result.get("verdict")
    confidence = result.get("confidence", 0.0)

    if verdict == "no_match":
        return "discard"

    if verdict == "uncertain":
        return "deep_review"

    if confidence >= 0.75:
        return "display"

    if confidence >= 0.5:
        return "deep_review"

    return "discard"


# ── Backwards compatibility aliases ──────────────────────────────────────────

run_profile_agent_v1       = run_profile_agent
run_profile_agent_v2       = run_profile_agent
run_compatibility_agent_v1 = run_compatibility_agent
run_compatibility_agent_v2 = run_compatibility_agent


# ── Entrypoint ────────────────────────────────────────────────────────────────

def _get_existing_compat_run(user_a_id: str,
                              user_b_id: str,
                              book_id: str) -> dict | None:
    """
    Return the most recent logged compatibility run for this pair and book,
    or None if it doesn't exist yet.
    Checks both orderings (A×B and B×A) since the pair is symmetric.
    """
    runs = _read_json_file(COMPAT_LOG_FILE, [])
    matches = [
        r for r in runs
        if r.get("book_id") == book_id
        and (
            (r.get("user_a") == user_a_id and r.get("user_b") == user_b_id)
            or
            (r.get("user_a") == user_b_id and r.get("user_b") == user_a_id)
        )
    ]
    if not matches:
        return None
    # return the most recent run
    return sorted(matches, key=lambda r: r.get("timestamp", ""), reverse=True)[0]


def run_compatibility_for_all(user_a_id: str, book_id: str) -> list[dict]:
    """
    Run the compatibility agent between user_a_id and every other user
    who has moments for the given book. Skips user_a_id itself.

    Skips pairs where a fresh compatibility run already exists and
    neither portrait has been updated since. Re-runs if portraits changed.

    Returns a list of results sorted by confidence descending,
    filtered to display/deep_review only (no_match discarded).
    """
    # find all other users with moments on this book
    other_users = list({
        m["user_id"]
        for m in _interpretations
        if m.get("book_id") == book_id and m.get("user_id") != user_a_id
    })

    if not other_users:
        print(f"[main] no other users found for book_id={book_id}")
        return []

    print(f"[main] found {len(other_users)} other readers of {book_id}")
    print(f"[main] running compatibility for {user_a_id} against all\n")

    results = []
    for user_b_id in other_users:

        # ── cache check ───────────────────────────────────────────────────────
        existing = _get_existing_compat_run(user_a_id, user_b_id, book_id)
        if existing:
            print(f"  {user_b_id}")
            print(f"    [cached] verdict={existing.get('verdict')}  "
                  f"confidence={existing.get('confidence')}")
            route = route_compatibility_result(existing)
            existing["route"] = route
            if route != "discard":
                results.append(existing)
            continue

        result = run_compatibility_agent(user_a_id, user_b_id, book_id)
        route  = route_compatibility_result(result)
        result["route"] = route

        print(f"  {user_b_id}")
        print(f"    verdict={result.get('verdict')}  "
              f"confidence={result.get('confidence')}  "
              f"route={route}")
        if route != "discard":
            print(f"    insight: {result.get('insight')}")
        print()

        if route != "discard":
            results.append(result)

    # sort by confidence descending
    results.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)

    print(f"── Match pool for {user_a_id} ───────────────────────────────")
    print(f"   {len(results)} matches from {len(other_users)} candidates\n")
    for r in results:
        print(f"  [{r['route'].upper()}] {r['user_b']} "
              f"— {r['verdict']} ({r['confidence']})")
        print(f"    {r.get('insight')}\n")

    return results

if __name__ == "__main__":
    USER_A  = "user_emma_chen_fd5e3def"
    BOOK_ID = "gutenberg_1342"

    matches = run_compatibility_for_all(USER_A, BOOK_ID)

    print("\n── Full results (JSON) ──────────────────────────")
    print(json.dumps(matches, indent=2))