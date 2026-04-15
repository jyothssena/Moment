import os
import json
from google import genai
from google.genai import types # type: ignore

from tools import extract_json, save_decomposition, get_decomposition

# ── Config ────────────────────────────────────────────────────────────────────

_api_key = os.environ.get("GEMINI_API_KEY_MOMENT")
if not _api_key:
    try:
        from dotenv import load_dotenv # type: ignore
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

# ── System prompt ─────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM_PROMPT = """\
You are the Moment Decomposition Agent for Momento.

You will be given a single reader moment on a passage.
Decompose it into sub-claims and assign an emotional mode to each.
Output only the final JSON — no preamble, no markdown fences.

════════════════════════════════════════════════════════════
STEP 1 — DECOMPOSE
════════════════════════════════════════════════════════════

Identify sub-claims from the moment.

Rules:
- Each sub-claim must be a distinct intellectual claim
- Each sub-claim must be grounded in a direct quote from the moment
  (words the reader wrote, not the passage text)
  If no direct quote exists, note "(no direct quote)"
- Do not split one claim into multiple because the reader repeated
  it about different phrases
- A sub-claim must represent a full independent intellectual position,
  not a supporting detail or restatement of another sub-claim

Count guideline:
  Aim for 2–4 sub-claims for most moments.
  Only exceed 4 if the moment contains clearly distinct ideas
  that cannot be merged without losing meaning.
  When in doubt, merge — do not split.

Weight each sub-claim:
  weight = words spent on this sub-claim / total words in moment
  Weights must sum to 1.0
  Minimum weight per sub-claim: 0.10
  If any sub-claim would be < 0.10, merge it into the nearest.

PROTECTED SUB-CLAIMS — never merge these:
  If the reader directly quotes a specific phrase from the passage
  (words placed in quotation marks in their moment), that quoted
  phrase must anchor its own sub-claim and cannot be merged into
  another, even if its weight would fall below 0.10.
  Assign it a minimum weight of 0.10 and adjust other weights
  proportionally so all weights still sum to 1.0.

  One sub-claim per quoted phrase — strictly:
  If the reader quotes N distinct phrases from the passage, the
  decomposition must contain at least N sub-claims, each anchored
  to one of those quoted phrases. A sub-claim can only be anchored
  to one quoted phrase.

════════════════════════════════════════════════════════════
STEP 2 — ASSIGN EMOTIONAL MODE (per sub-claim)
════════════════════════════════════════════════════════════

Assign one mode from this list to each sub-claim:

  prosecutorial    — controlled frustration, holds accountable,
                     finds the evidence and presents it
  philosophical    — intellectual distance, sits with the paradox,
                     frames as a question or pattern
  empathetic       — feels with the character or subject,
                     stays inside their experience
  observational    — notices and describes without taking sides,
                     atmospheric or craft-level attention
  aesthetic        — responds to language or craft execution,
                     moved by how something is written
  self-referential — maps the text explicitly to personal experience

════════════════════════════════════════════════════════════
OUTPUT
════════════════════════════════════════════════════════════

{
  "book_id":<book_id>
  "passage_id": "<passage identifier>",
  "user_id": "<id>",
  "subclaims": [
    {
      "id": "1",
      "claim": "<claim text>",
      "quote": "<direct quote from moment or (no direct quote)>",
      "weight": <float>,
      "emotional_mode": "<mode>"
    }
  ]
}

"""

# ── Required keys ─────────────────────────────────────────────────────────────

_DECOMPOSE_REQUIRED_KEYS = {"passage_id", "user_id", "subclaims"}
_SUBCLAIM_REQUIRED_KEYS  = {"id", "claim", "quote", "weight", "emotional_mode"}

# ── Main function ─────────────────────────────────────────────────────────────

def run_decomposer(user_id: str, passage_id: str, book_id: str, moment_text: str) -> dict:
    """
    Decompose a single reader moment into weighted sub-claims with emotional modes.
    Saves result to BQ via tools.save_decomposition().

    Args:
        user_id:      reader identifier
        passage_id:   passage identifier
        book_id:      book identifier
        moment_text:  the raw moment text written by the reader

    Returns:
        decomposition dict on success, {"error": ..., "user_id": user_id} on failure
    """
    print(f"[Decomposer] running for user_id={user_id}, passage_id={passage_id}")
    user_message = (
        f"user_id: {user_id}\n"
        f"passage_id: {passage_id}\n\n"
        f"book_id: {book_id}\n\n"
        f"Moment:\n{moment_text}"
    )
    response = _gemini_client.models.generate_content(
        model=_GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=_DECOMPOSE_SYSTEM_PROMPT,
            temperature=0.1,
        ),
        contents=user_message,
    )

    raw_text = _get_response_text(response)
    result   = extract_json(raw_text)
    if "error" in result:
        print(f"[Decomposer] JSON extraction failed. raw={raw_text[:200] if raw_text else 'None'}")
        return {"error": "invalid JSON from Decomposer", "raw": raw_text, "user_id": user_id}

    missing = _DECOMPOSE_REQUIRED_KEYS - result.keys()
    if missing:
        print(f"[Decomposer] incomplete output, missing keys: {missing}")
        return {"error": f"incomplete decomposition, missing: {missing}", "raw": result, "user_id": user_id}

    for i, sc in enumerate(result.get("subclaims", [])):
        missing_sc = _SUBCLAIM_REQUIRED_KEYS - sc.keys()
        if missing_sc:
            print(f"[Decomposer] subclaim {i} missing keys: {missing_sc}")
            return {"error": f"subclaim {i} missing: {missing_sc}", "raw": result, "user_id": user_id}

    weights = [sc["weight"] for sc in result["subclaims"]]
    if abs(sum(weights) - 1.0) > 0.02:
        print(f"[Decomposer] weights do not sum to 1.0: {weights}")
        return {"error": f"subclaim weights sum to {sum(weights):.2f}", "raw": result, "user_id": user_id}

    print(f"[Decomposer] decomposition ready: {len(result['subclaims'])} subclaims")
    # Save to BQ — main.py also calls this but safe to call here too (upsert)
    save_decomposition(result)
    return result


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_response_text(response) -> str | None:
    try:
        return response.text
    except Exception:
        pass
    try:
        parts = response.candidates[0].content.parts
        texts = [p.text for p in parts if hasattr(p, "text") and p.text]
        return "\n".join(texts) if texts else None
    except Exception:
        return None