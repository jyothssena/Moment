"""
eval.py — Moment Agent Evaluation Script

Scores portrait and compatibility output quality across three dimensions:
  - Portrait specificity   (is it grounded in actual quotes?)
  - Verdict defensibility  (does the reasoning go beyond surface agreement?)
  - Insight quality        (would a real user find this compelling?)

Run:
    python eval.py

Output:
    eval_report.json  — full scored results
    Console summary   — per-pair scores with flags
"""

import json
import os
import textwrap
from datetime import datetime
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────

PROFILE_FILE    = "data/processed/profiles.json"
COMPAT_LOG_FILE = "data/processed/compatibility_runs.json"
EVAL_OUTPUT     = "data/processed/eval_report.json"

_api_key = os.environ.get("GEMINI_API_KEY_MOMENT")
if not _api_key:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        _api_key = os.environ.get("GEMINI_API_KEY_MOMENT")
    except ImportError:
        pass

if not _api_key:
    raise EnvironmentError("GEMINI_API_KEY_MOMENT is not set.")

_client = genai.Client(api_key=_api_key)
_MODEL  = "gemini-2.5-flash"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: str, default):
    try:
        with open(path) as f:
            content = f.read()
            return json.loads(content) if content.strip() else default
    except FileNotFoundError:
        return default

def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _extract_json(text: str) -> dict:
    import re
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

# ── Portrait Evaluator ────────────────────────────────────────────────────────

_PORTRAIT_EVAL_PROMPT = textwrap.dedent("""
    You are evaluating the quality of an AI-generated reader portrait.
    A good portrait is specific, grounded in actual quotes from the reader's
    reflections, and would feel accurate to someone who read the same annotations.
    A bad portrait is generic — it could describe any thoughtful reader.

    Score each dimension from 1-3:
    1 = generic, could describe anyone
    2 = somewhat specific, some evidence but could go deeper
    3 = highly specific, clearly grounded in this reader's actual language

    Dimensions to score: themes, emotional_register, engagement_style,
    reflection_density.

    Also flag:
    - contains_quote: true if at least one dimension quotes the reader directly
    - feels_accurate: true if the portrait feels like a real characterization
      rather than a template

    Return ONLY raw JSON with keys:
    themes_score, emotional_register_score, engagement_style_score,
    reflection_density_score, overall_score (average of four),
    contains_quote, feels_accurate, notes (1-2 sentences on biggest weakness).
""")

def eval_portrait(portrait: dict) -> dict:
    """Score a single reader portrait on specificity and grounding."""
    prompt = f"""
    Evaluate this reader portrait:

    {json.dumps(portrait, indent=2)}

    Score each dimension and return raw JSON.
    """
    response = _client.models.generate_content(
        model=_MODEL,
        config=types.GenerateContentConfig(
            temperature=0.1,
            system_instruction=_PORTRAIT_EVAL_PROMPT,
        ),
        contents=prompt
    )
    result = _extract_json(response.text)
    result["user_id"] = portrait.get("user_id")
    return result

# ── Compatibility Run Evaluator ───────────────────────────────────────────────

_COMPAT_EVAL_PROMPT = textwrap.dedent("""
    You are evaluating the quality of an AI-generated compatibility assessment
    between two readers of the same book.

    A good compatibility result:
    - Has reasoning that goes beyond surface agreement ("both liked X")
      to identify a structural relationship between the readers
    - Has an insight the users would actually find compelling and specific
    - Has a verdict that is defensible given the evidence
    - References actual moments or passages, not just portrait summaries

    A weak compatibility result:
    - Lists similarities without explaining why they make a meaningful connection
    - Produces a generic insight that could apply to any two analytical readers
    - Ignores the possibility of mirror or contradiction matches in favor of
      easy resonance verdicts

    Score each dimension from 1-3:
    1 = weak / generic
    2 = adequate / some depth
    3 = strong / genuinely insightful

    Dimensions: reasoning_depth, insight_quality, verdict_defensibility,
    evidence_grounding.

    Also flag:
    - verdict_type_appropriate: true if resonance/mirror/contradiction was
      chosen correctly given the portraits (not defaulting to resonance lazily)
    - insight_would_compel_user: true if a real user would be intrigued by
      this insight

    Return ONLY raw JSON with keys:
    reasoning_depth_score, insight_quality_score, verdict_defensibility_score,
    evidence_grounding_score, overall_score (average of four),
    verdict_type_appropriate, insight_would_compel_user,
    notes (2-3 sentences on what would most improve this result).
""")

def eval_compatibility_run(run: dict) -> dict:
    """Score a single compatibility run on reasoning depth and insight quality."""
    prompt = f"""
    Evaluate this compatibility assessment:

    User A portrait:
    {json.dumps(run.get('portrait_a', {}), indent=2)}

    User B portrait:
    {json.dumps(run.get('portrait_b', {}), indent=2)}

    Verdict: {run.get('verdict')}
    Confidence: {run.get('confidence')}
    Reasoning: {run.get('reasoning')}
    Insight: {run.get('insight')}

    Score the quality of this assessment and return raw JSON.
    """
    response = _client.models.generate_content(
        model=_MODEL,
        config=types.GenerateContentConfig(
            temperature=0.1,
            system_instruction=_COMPAT_EVAL_PROMPT,
        ),
        contents=prompt
    )
    result = _extract_json(response.text)
    result["user_a"]  = run.get("user_a")
    result["user_b"]  = run.get("user_b")
    result["book_id"] = run.get("book_id")
    result["verdict"] = run.get("verdict")
    return result

# ── Batch Runner ──────────────────────────────────────────────────────────────

def run_eval(
    max_portraits: int = 10,
    max_compat_runs: int = 10
) -> dict:
    """
    Evaluate all portraits and compatibility runs up to the given limits.
    Writes a full report to EVAL_OUTPUT and prints a console summary.
    """
    profiles   = _read_json(PROFILE_FILE, [])
    compat_runs = _read_json(COMPAT_LOG_FILE, [])

    portrait_results = []
    compat_results   = []

    # ── Evaluate portraits ────────────────────────────────────────────────────
    print(f"\n── Evaluating {min(len(profiles), max_portraits)} portraits ──")
    for p in profiles[:max_portraits]:
        uid = p.get("user_id", "unknown")
        print(f"  scoring portrait: {uid}")
        score = eval_portrait(p)
        portrait_results.append(score)

        flag = "✓" if score.get("overall_score", 0) >= 2.5 else "⚠"
        print(
            f"  {flag} overall={score.get('overall_score')} "
            f"quote={score.get('contains_quote')} "
            f"accurate={score.get('feels_accurate')}"
        )
        if score.get("notes"):
            print(f"    note: {score['notes']}")

    # ── Evaluate compatibility runs ───────────────────────────────────────────
    print(f"\n── Evaluating {min(len(compat_runs), max_compat_runs)} compatibility runs ──")
    for run in compat_runs[:max_compat_runs]:
        a = run.get("user_a", "?")
        b = run.get("user_b", "?")
        print(f"  scoring: {a} × {b}")
        score = eval_compatibility_run(run)
        compat_results.append(score)

        flag = "✓" if score.get("overall_score", 0) >= 2.5 else "⚠"
        print(
            f"  {flag} overall={score.get('overall_score')} "
            f"verdict={score.get('verdict')} "
            f"appropriate={score.get('verdict_type_appropriate')} "
            f"compelling={score.get('insight_would_compel_user')}"
        )
        if score.get("notes"):
            print(f"    note: {score['notes']}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def avg(results, key):
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "portrait_summary": {
            "count":                    len(portrait_results),
            "avg_overall":              avg(portrait_results, "overall_score"),
            "avg_themes":               avg(portrait_results, "themes_score"),
            "avg_emotional_register":   avg(portrait_results, "emotional_register_score"),
            "avg_engagement_style":     avg(portrait_results, "engagement_style_score"),
            "avg_reflection_density":   avg(portrait_results, "reflection_density_score"),
            "pct_contains_quote":       round(
                sum(1 for r in portrait_results if r.get("contains_quote")) /
                max(len(portrait_results), 1) * 100, 1
            ),
            "pct_feels_accurate":       round(
                sum(1 for r in portrait_results if r.get("feels_accurate")) /
                max(len(portrait_results), 1) * 100, 1
            ),
        },
        "compat_summary": {
            "count":                        len(compat_results),
            "avg_overall":                  avg(compat_results, "overall_score"),
            "avg_reasoning_depth":          avg(compat_results, "reasoning_depth_score"),
            "avg_insight_quality":          avg(compat_results, "insight_quality_score"),
            "avg_verdict_defensibility":    avg(compat_results, "verdict_defensibility_score"),
            "avg_evidence_grounding":       avg(compat_results, "evidence_grounding_score"),
            "pct_verdict_appropriate":      round(
                sum(1 for r in compat_results if r.get("verdict_type_appropriate")) /
                max(len(compat_results), 1) * 100, 1
            ),
            "pct_insight_compelling":       round(
                sum(1 for r in compat_results if r.get("insight_would_compel_user")) /
                max(len(compat_results), 1) * 100, 1
            ),
        },
        "portrait_scores":  portrait_results,
        "compat_scores":    compat_results,
    }

    _write_json(EVAL_OUTPUT, summary)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────")
    ps = summary["portrait_summary"]
    cs = summary["compat_summary"]

    print(f"  Portraits   ({ps['count']})")
    print(f"    avg overall:       {ps['avg_overall']} / 3")
    print(f"    contains quote:    {ps['pct_contains_quote']}%")
    print(f"    feels accurate:    {ps['pct_feels_accurate']}%")

    print(f"\n  Compatibility runs  ({cs['count']})")
    print(f"    avg overall:       {cs['avg_overall']} / 3")
    print(f"    reasoning depth:   {cs['avg_reasoning_depth']} / 3")
    print(f"    insight quality:   {cs['avg_insight_quality']} / 3")
    print(f"    verdict correct:   {cs['pct_verdict_appropriate']}%")
    print(f"    insight compelling:{cs['pct_insight_compelling']}%")

    print(f"\n  Full report written to {EVAL_OUTPUT}")

    # ── Flags ─────────────────────────────────────────────────────────────────
    print("\n── Flags (items scoring below 2.0) ─────────────────────────")
    flagged = False
    for r in portrait_results:
        if r.get("overall_score", 3) < 2.0:
            print(f"  ⚠ portrait {r.get('user_id')}: {r.get('notes')}")
            flagged = True
    for r in compat_results:
        if r.get("overall_score", 3) < 2.0:
            print(
                f"  ⚠ compat {r.get('user_a')} × {r.get('user_b')}: "
                f"{r.get('notes')}"
            )
            flagged = True
    if not flagged:
        print("  none — all items scored 2.0 or above")

    return summary


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_eval(max_portraits=10, max_compat_runs=10)