import json
import re
import os
import functools
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE      = "data/processed/moments_processed.json"
PROFILE_FILE    = "data/processed/profiles.json"
COMPAT_LOG_FILE = "data/processed/compatibility_runs.json"
RECO_LOG_FILE   = "data/processed/recommendation_runs.json"
TOOL_LOG_FILE   = "data/processed/tool_call_log.json"

try:
    with open(INPUT_FILE) as f:
        _interpretations = json.load(f)
except FileNotFoundError:
    _interpretations = []


# ── File I/O ──────────────────────────────────────────────────────────────────

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


# ── Logging ───────────────────────────────────────────────────────────────────

def _log_tool_call(tool_name: str, args: tuple, kwargs: dict, result) -> None:
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
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        try:
            _log_tool_call(fn.__name__, args, kwargs, result)
        except Exception as e:
            print(f"[warn] tool log failed for {fn.__name__}: {e}")
        return result
    return wrapper


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """Robustly extract a JSON object from LLM output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "could not extract JSON", "raw": text}


# ── Reader tools ──────────────────────────────────────────────────────────────

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
def get_all_profiles(exclude_user_id: str) -> list[dict]:
    """Retrieve all reader portraits except for the given user.
    Used to find candidate matches not yet evaluated."""
    profiles = _read_json_file(PROFILE_FILE, [])
    return [p for p in profiles if p.get("user_id") != exclude_user_id]

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


# ── Compatibility tools ───────────────────────────────────────────────────────

@_logged
def get_compatibility_runs(user_id: str, min_confidence: float = 0.75) -> list[dict]:
    """Retrieve all logged compatibility results for a user above a confidence
    threshold. Returns runs where the user appears as user_a or user_b."""
    runs = _read_json_file(COMPAT_LOG_FILE, [])
    return [
        r for r in runs
        if (r.get("user_a") == user_id or r.get("user_b") == user_id)
        and r.get("confidence", 0.0) >= min_confidence
        and r.get("verdict") != "no_match"
    ]

def log_compatibility_run(record: dict) -> None:
    """Append a compatibility run record to the log. Not a tool — called internally."""
    runs = _read_json_file(COMPAT_LOG_FILE, [])
    runs.append(record)
    _write_json_file(COMPAT_LOG_FILE, runs)


# ── Recommendation tools ──────────────────────────────────────────────────────

@_logged
def save_recommendations(user_id: str, recommendations: list[dict]) -> str:
    """Persist the final ranked recommendation list for a user."""
    record = {
        "user_id":         user_id,
        "recommendations": recommendations,
        "last_updated":    datetime.utcnow().isoformat()
    }
    runs = _read_json_file(RECO_LOG_FILE, [])
    for i, r in enumerate(runs):
        if r.get("user_id") == user_id:
            runs[i] = record
            _write_json_file(RECO_LOG_FILE, runs)
            return "Recommendations updated."
    runs.append(record)
    _write_json_file(RECO_LOG_FILE, runs)
    return "Recommendations saved."


# ── Tool registries (passed to GenerateContentConfig) ─────────────────────────
# Import these in your agent files instead of listing tools manually.

PROFILE_TOOLS = [
    get_user_interpretations,
    get_user_profile,
    save_user_profile,
    count_new_moments,
]

COMPAT_TOOLS = [
    get_user_interpretations,
]

RECO_TOOLS = [
    get_compatibility_runs,
    get_user_profile,
    get_all_profiles,
    save_recommendations,
]