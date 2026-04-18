"""
model_interface_stub.py
=======================
Drop-in replacement for model_interface.py used during infrastructure
development while the real model is being fixed.

Matches the EXACT same function signatures and return shapes as
model_interface.py so the API, CI/CD, validate_model.py, and
verify_deployment.py all work without a live Gemini key.

Swap: set MODEL_MODE=stub (default) or MODEL_MODE=real in the environment.
The api/main.py entrypoint reads this and imports accordingly.

Returns deterministic-but-realistic outputs:
  - confidence always in [0.45, 0.80] — passes validate_model.py thresholds
  - R + C + D always sum to 100 — passes rcd validation
  - All required keys always present — passes schema validation
"""

import hashlib
import random
from datetime import datetime, timezone


def _seed_from(*args) -> random.Random:
    """Deterministic RNG seeded from input args — same inputs always give same output."""
    seed_str = "|".join(str(a) for a in args)
    seed_int = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed_int)


def _make_rcd(rng: random.Random) -> tuple[int, int, int]:
    """Return R, C, D integers that sum to exactly 100."""
    r = rng.randint(30, 65)
    c = rng.randint(10, 40)
    d = 100 - r - c
    if d < 0:
        c = 100 - r - 5
        d = 5
    dominant = max(("resonate", r), ("contradict", c), ("diverge", d), key=lambda x: x[1])[0]
    return r, c, d, dominant


# ── 1. Decompose a single moment ─────────────────────────────────────────────

def decompose_moment(
    passage_id: str,
    user_id: str,
    moment_text: str,
    word_count: int = 0,
    book_id: str = "",
) -> dict:
    """Stub: returns a valid decomposition without calling Gemini."""
    rng = _seed_from(user_id, passage_id, book_id)
    n_subclaims = rng.randint(2, 4)

    modes = ["prosecutorial", "philosophical", "empathetic",
             "observational", "aesthetic", "self-referential"]
    words = (moment_text or "the passage reflects on themes of identity").split()

    # Build weights that sum to 1.0
    raw = [rng.random() for _ in range(n_subclaims)]
    total = sum(raw)
    weights = [round(w / total, 2) for w in raw]
    weights[-1] = round(1.0 - sum(weights[:-1]), 2)  # fix rounding

    subclaims = []
    for i in range(n_subclaims):
        start = rng.randint(0, max(0, len(words) - 6))
        quote_words = words[start:start + 4]
        subclaims.append({
            "id": str(i + 1),
            "claim": f"The reader identifies a {rng.choice(['thematic', 'structural', 'emotional', 'symbolic'])} "
                     f"dimension related to {rng.choice(['identity', 'power', 'isolation', 'connection', 'ambition'])}.",
            "quote": " ".join(quote_words) if quote_words else "(no direct quote)",
            "weight": weights[i],
            "emotional_mode": rng.choice(modes),
        })

    return {
        "passage_id": passage_id,
        "user_id": user_id,
        "book_id": book_id,
        "subclaims": subclaims,
        "_stub": True,
    }


# ── 2. Full compatibility pipeline ───────────────────────────────────────────

def run_compatibility_pipeline(
    user_a: str,
    user_b: str,
    book: str,
    passage_id: str,
    moment_a: dict,
    moment_b: dict,
) -> dict:
    """Stub: returns a valid compatibility result without calling Gemini."""
    rng = _seed_from(user_a, user_b, passage_id)
    tr, tc, td, dominant_think = _make_rcd(rng)
    fr, fc, fd, dominant_feel  = _make_rcd(rng)
    confidence = round(rng.uniform(0.45, 0.80), 2)

    return {
        "passage_id":     passage_id,
        "character_a":    user_a,
        "character_b":    user_b,
        "book":           book,
        "think":          {"R": tr, "C": tc, "D": td},
        "feel":           {"R": fr, "C": fc, "D": fd},
        "dominant_think": dominant_think,
        "dominant_feel":  dominant_feel,
        "match_count":    rng.randint(1, 4),
        "confidence":     confidence,
        "computed_at":    datetime.now(timezone.utc).isoformat(),
        "_stub":          True,
    }


# ── 3. Batch runner ───────────────────────────────────────────────────────────

def run_batch_compatibility(
    user_a_id: str,
    book_id: str,
    passage_id: str,
    moments_map: dict,
) -> list:
    """Stub: returns a sorted list of compatibility results for all users."""
    results = []
    other_users = [uid for uid in moments_map if uid != user_a_id]

    for user_b_id in other_users:
        moment_a = moments_map.get(user_a_id, {})
        moment_b = moments_map.get(user_b_id, {})
        result = run_compatibility_pipeline(
            user_a=user_a_id,
            user_b=user_b_id,
            book=book_id,
            passage_id=passage_id,
            moment_a=moment_a,
            moment_b=moment_b,
        )
        result["route"] = "display"
        results.append(result)

    results.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
    return results


# ── Health check ──────────────────────────────────────────────────────────────

def health_check() -> dict:
    return {
        "status": "ok",
        "mode": "stub",
        "interface_version": "2.0.0",
        "stub_mode": True,
        "message": "Stub model active — real model not required",
        "functions": ["decompose_moment", "run_compatibility_pipeline", "run_batch_compatibility"],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
