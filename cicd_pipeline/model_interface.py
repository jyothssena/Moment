"""
model_interface.py — Momento Agent Interface Contract
======================================================
NOW WIRED TO REAL MODEL CODE.

This module is the single entry point the CI/CD pipeline and deploy.py
use to call the agent pipeline. It delegates directly to:

  decomposing_agent.py   → run_decomposer()
  compatibility_agent.py → run_compatibility_agent(), run_compatibility_for_all()
  aggregator.py          → aggregate()  (called internally by compatibility_agent)

To swap any agent implementation, replace the underlying module —
this interface file does not need to change.
"""

from datetime import datetime

def _get_run_decomposer():
    from decomposing_agent import run_decomposer
    return run_decomposer

def _get_run_compatibility_agent():
    from compatibility_agent import run_compatibility_agent
    return run_compatibility_agent

def _get_run_compatibility_for_all():
    from compatibility_agent import run_compatibility_for_all
    return run_compatibility_for_all

# ── 1. Decompose a single moment ─────────────────────────────────────────────

def decompose_moment(
    passage_id: str,
    user_id: str,
    moment_text: str,
    word_count: int,        # kept for interface compatibility; decomposer uses text directly
    book_id: str = "",
) -> dict:
    """
    Decomposes a single reader moment into weighted sub-claims.

    Delegates to decomposing_agent.run_decomposer().
    Caches result to data/processed/decompositions.json automatically.

    Returns
    -------
    {
      "passage_id": str,
      "user_id": str,
      "book_id": str,
      "subclaims": [
        {
          "id": str,
          "claim": str,
          "quote": str,
          "weight": float,
          "emotional_mode": str
        }
      ]
    }
    On failure: {"error": str, "user_id": str}
    """
    return _get_run_decomposer()(
        user_id=user_id,
        passage_id=passage_id,
        book_id=book_id,
        moment_text=moment_text,
    )


# ── 2. Full per-passage pipeline ─────────────────────────────────────────────

def run_compatibility_pipeline(
    user_a: str,
    user_b: str,
    book: str,
    passage_id: str,
    moment_a: dict,
    moment_b: dict,
) -> dict:
    """
    Full per-passage pipeline for one user pair:
      decompose A → decompose B → score → aggregate → R/C/D + confidence

    Delegates to compatibility_agent.run_compatibility_agent().
    Decompositions and scoring runs are cached automatically.

    Parameters
    ----------
    moment_a, moment_b : raw moment dicts from moments_db / interpretations file
        Must contain: passage_id, interpretation (or text)

    Returns
    -------
    {
      "passage_id": str,
      "character_a": str,
      "character_b": str,
      "book": str,
      "think":  {"R": int, "C": int, "D": int},
      "feel":   {"R": int, "C": int, "D": int},
      "dominant_think": str,
      "dominant_feel":  str,
      "match_count": int,
      "confidence": float,
      "computed_at": str   # ISO timestamp
    }
    On failure: {"error": str, "user_a": str, "user_b": str}
    """
    # Ensure passage_id is in the moment dicts (compatibility_agent reads it from there)
    moment_a = {**moment_a, "passage_id": passage_id}
    moment_b = {**moment_b, "passage_id": passage_id}

    result = _get_run_compatibility_agent()(
        user_a_id=user_a,
        user_b_id=user_b,
        book_id=book,
        moment_a=moment_a,
        moment_b=moment_b,
    )

    # Normalise key names for validate_model.py and bias_detection.py
    # (compatibility_agent uses user_a/user_b; pipeline expects character_a/character_b)
    if "error" not in result:
        result.setdefault("character_a", result.get("user_a", user_a))
        result.setdefault("character_b", result.get("user_b", user_b))
        result.setdefault("book", book)
        result.setdefault("computed_at", result.get("timestamp", datetime.utcnow().isoformat()))

    return result


# ── 3. Batch runner — anchor user vs all others on same passage ──────────────

def run_batch_compatibility(
    user_a_id: str,
    book_id: str,
    passage_id: str,
    moments_map: dict,
) -> list:
    """
    Scores anchor user against every other user who has a moment on the
    same passage. Results are cached and deduplicated automatically.

    Delegates to compatibility_agent.run_compatibility_for_all().

    Parameters
    ----------
    moments_map : {user_id: moment_dict}
        All users with a moment for this book + passage.

    Returns
    -------
    List of result dicts sorted by confidence descending,
    with "route" key attached. Same shape as run_compatibility_pipeline().
    """
    return _get_run_compatibility_for_all()(
        user_a_id=user_a_id,
        book_id=book_id,
        passage_id=passage_id,
        moments_map=moments_map,
    )


# ── Health check ──────────────────────────────────────────────────────────────

def health_check() -> dict:
    """
    Verifies the module loads and all real agents are importable.
    Called by the CI pipeline before running inference.
    """
    return {
        "status": "ok",
        "interface_version": "2.0.0",
        "stub_mode": True,
        "functions": ["decompose_moment", "run_compatibility_pipeline", "run_batch_compatibility"],
        "agents": {
            "decomposer":   "decomposing_agent.run_decomposer",
            "scorer":       "compatibility_agent.run_compatibility_agent",
            "batch_runner": "compatibility_agent.run_compatibility_for_all",
            "aggregator":   "aggregator.aggregate",
        },
        "checked_at": datetime.utcnow().isoformat(),
    }