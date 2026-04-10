"""
run_validation_set.py — Validation Set Runner
==============================================
Called by the CI pipeline in the validate job:
  python run_validation_set.py --output validation_results.json

Runs the full real pipeline on a small held-out validation set:
  decompose_moment (Gemini API call per reader)
  → run_compatibility_pipeline (score + aggregate per pair)
  → writes results to JSON for validate_model.py + bias_detection.py

The validation set below uses real data from interpretations_10_users.json.
To expand it, add more pairs to VALIDATION_PAIRS — no other changes needed.

Requires GEMINI_API_KEY_MOMENT env var (set in GitHub Actions secrets).
"""

import argparse
import json
import os
import sys
from datetime import datetime

from cicd_pipeline.model_interface import run_compatibility_pipeline

# ── Held-out validation set ───────────────────────────────────────────────────
# Drawn from interpretations_10_users.json.
# Covers all 3 books and multiple passages to give bias_detection.py
# enough slice diversity to compute meaningful confidence gaps.

VALIDATION_PAIRS = [
    # ── Frankenstein ──────────────────────────────────────────────────────────
    {
        "user_a": "Emma Chen",
        "user_b": "James Park",
        "book": "Frankenstein",
        "passage_id": "passage_1",
        "moment_a": {
            "character_name": "Emma Chen",
            "interpretation": (
                "The creature's demand for a companion reveals the fundamental "
                "human need for belonging. Shelley frames this as both a plea "
                "and an accusation — Victor created the longing but refuses to "
                "satisfy it. There is something devastating about a being who "
                "understands exactly what he lacks."
            ),
            "word_count": 50,
            "passage_id": "passage_1",
            "book_id": "Frankenstein",
        },
        "moment_b": {
            "character_name": "James Park",
            "interpretation": (
                "Victor's refusal is his moral failure crystallised. The creature "
                "is entirely rational — his argument is airtight. Victor's refusal "
                "is self-protective cowardice dressed up as ethical concern. "
                "Shelley makes it impossible to sympathise with him here."
            ),
            "word_count": 43,
            "passage_id": "passage_1",
            "book_id": "Frankenstein",
        },
    },
    {
        "user_a": "Sofia Ali",
        "user_b": "Priya Nair",
        "book": "Frankenstein",
        "passage_id": "passage_2",
        "moment_a": {
            "character_name": "Sofia Ali",
            "interpretation": (
                "What strikes me is Victor's language when he destroys the female "
                "creature. He describes it clinically — dismemberment as procedure. "
                "The detachment is its own horror. Shelley is showing us how "
                "violence hides inside the vocabulary of reason."
            ),
            "word_count": 44,
            "passage_id": "passage_2",
            "book_id": "Frankenstein",
        },
        "moment_b": {
            "character_name": "Priya Nair",
            "interpretation": (
                "I keep returning to the creature watching Victor destroy his mate. "
                "He has been promised something and watches it taken away. "
                "That moment of witnessing is the one Shelley lingers on — "
                "not the destruction itself but the creature's face."
            ),
            "word_count": 44,
            "passage_id": "passage_2",
            "book_id": "Frankenstein",
        },
    },
    # ── The Great Gatsby ──────────────────────────────────────────────────────
    {
        "user_a": "Marco Rossi",
        "user_b": "Leon Bauer",
        "book": "The Great Gatsby",
        "passage_id": "passage_1",
        "moment_a": {
            "character_name": "Marco Rossi",
            "interpretation": (
                "The green light is Gatsby's delusion given form. It's not hope — "
                "it's obsession. The beauty Fitzgerald gives it is the lie Gatsby "
                "tells himself. The tragedy is that Gatsby knows this on some level "
                "and reaches anyway."
            ),
            "word_count": 44,
            "passage_id": "passage_1",
            "book_id": "The Great Gatsby",
        },
        "moment_b": {
            "character_name": "Leon Bauer",
            "interpretation": (
                "I read the green light as Fitzgerald's comment on the American Dream "
                "itself — something always just out of reach by design. Gatsby isn't "
                "a tragic individual, he's a symbol of a collective delusion. "
                "The light can never be reached because that's the point of it."
            ),
            "word_count": 48,
            "passage_id": "passage_1",
            "book_id": "The Great Gatsby",
        },
    },
    {
        "user_a": "Emma Chen",
        "user_b": "Marco Rossi",
        "book": "The Great Gatsby",
        "passage_id": "passage_2",
        "moment_a": {
            "character_name": "Emma Chen",
            "interpretation": (
                "Nick's narration here feels unreliable to me in a specific way — "
                "he's drawn to Gatsby's myth even as he describes the mechanics "
                "of how it works. He romanticises what he's supposedly exposing. "
                "Fitzgerald makes the reader complicit in the same way."
            ),
            "word_count": 46,
            "passage_id": "passage_2",
            "book_id": "The Great Gatsby",
        },
        "moment_b": {
            "character_name": "Marco Rossi",
            "interpretation": (
                "Nick is not unreliable — he is honest about being seduced. "
                "That's different. He tells us exactly what is happening to him. "
                "The distinction matters because Fitzgerald wants us to understand "
                "how the myth operates, not just that it operates."
            ),
            "word_count": 44,
            "passage_id": "passage_2",
            "book_id": "The Great Gatsby",
        },
    },
    # ── Pride and Prejudice ───────────────────────────────────────────────────
    {
        "user_a": "James Park",
        "user_b": "Sofia Ali",
        "book": "Pride and Prejudice",
        "passage_id": "passage_1",
        "moment_a": {
            "character_name": "James Park",
            "interpretation": (
                "Elizabeth's refusal isn't just personal — it's political. "
                "She refuses the social script that says women must accept "
                "any proposal that improves their situation. Austen gives her "
                "a kind of freedom that feels almost anachronistic."
            ),
            "word_count": 42,
            "passage_id": "passage_1",
            "book_id": "Pride and Prejudice",
        },
        "moment_b": {
            "character_name": "Sofia Ali",
            "interpretation": (
                "What I notice is Elizabeth's clarity. She doesn't hesitate, "
                "doesn't soften it. Austen writes her refusal as though it costs "
                "nothing — when both characters know it costs a great deal. "
                "That gap between performance and reality is the whole scene."
            ),
            "word_count": 45,
            "passage_id": "passage_1",
            "book_id": "Pride and Prejudice",
        },
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="validation_results.json",
                        help="Path to write validation results JSON")
    args = parser.parse_args()

    print(f"Running validation pipeline on {len(VALIDATION_PAIRS)} pairs...")
    print("Pipeline: decompose → score → aggregate (real Gemini calls)\n")

    results = []
    failed  = 0

    for i, pair in enumerate(VALIDATION_PAIRS):
        print(f"[{i+1}/{len(VALIDATION_PAIRS)}] {pair['user_a']} × {pair['user_b']} "
              f"— {pair['book']} / {pair['passage_id']}")
        try:
            result = run_compatibility_pipeline(
                user_a=pair["user_a"],
                user_b=pair["user_b"],
                book=pair["book"],
                passage_id=pair["passage_id"],
                moment_a=pair["moment_a"],
                moment_b=pair["moment_b"],
            )

            if "error" in result:
                print(f"  ✗ Agent error: {result['error']}")
                failed += 1
                continue

            results.append(result)
            print(f"  ✓ confidence={result.get('confidence'):.2f}  "
                  f"think={result.get('think')}  feel={result.get('feel')}")

        except Exception as e:
            print(f"  ✗ Exception: {e}")
            failed += 1

    if failed > 0:
        print(f"\n⚠  {failed}/{len(VALIDATION_PAIRS)} pairs failed")

    if not results:
        print("✗ No results produced — cannot validate")
        sys.exit(1)

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_pairs":  len(results),
        "failed_pairs": failed,
        "pairs":        results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ {len(results)} results written to {args.output}")


if __name__ == "__main__":
    main()