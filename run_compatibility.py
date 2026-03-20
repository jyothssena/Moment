import json
import numpy as np # type: ignore
import faiss # type: ignore
import anthropic # type: ignore
from collections import Counter
from instrumentation import log_agent_call # type: ignore

MOMENTS_FILE   = "data/processed/moments_embedded.json"
PORTRAITS_FILE = "data/processed/user_portraits.json"
INDEX_FILE     = "data/processed/moments.index"
MAP_FILE       = "data/processed/index_map.json"
OUTPUT_FILE    = "data/processed/compatibility_results.json"

client = anthropic.Anthropic()

# Load everything
with open(MOMENTS_FILE) as f:
    moments = json.load(f)
with open(PORTRAITS_FILE) as f:
    portraits = json.load(f)
with open(MAP_FILE) as f:
    index_map = json.load(f)

index = faiss.read_index(INDEX_FILE)

vectors = np.array([m["embedding"] for m in moments], dtype="float32")
faiss.normalize_L2(vectors)

# Group moments by book
moments_by_book = {}
for m in moments:
    book = m["book_id"]
    if book not in moments_by_book:
        moments_by_book[book] = []
    moments_by_book[book].append(m)

print("Books and moment counts:")
for book, mlist in moments_by_book.items():
    print(f"  {book}: {len(mlist)} moments")

# Find candidate pairs — only within the same book
print("\nFinding candidate pairs (same book only, similarity >= 0.82)...")
candidate_pairs = set()

for i, moment in enumerate(moments):
    query = vectors[i:i+1]
    distances, indices = index.search(query, 20)

    for idx, score in zip(indices[0], distances[0]):
        if idx == i:
            continue
        if score < 0.82:
            continue

        other = moments[idx]

        # Must be same book
        if other["book_id"] != moment["book_id"]:
            continue

        # Must be different user
        if other["user_id"] == moment["user_id"]:
            continue

        pair = tuple(sorted([moment["user_id"], other["user_id"]]))
        candidate_pairs.add(pair)

print(f"Found {len(candidate_pairs)} unique candidate pairs.\n")

# Run Compatibility Investigator for each pair
results = []

for i, (user_a_id, user_b_id) in enumerate(candidate_pairs):
    print(f"  [{i+1}/{len(candidate_pairs)}] {user_a_id[:35]} vs {user_b_id[:35]}")

    portrait_a = portraits.get(user_a_id, {})
    portrait_b = portraits.get(user_b_id, {})

    # Get moments for each user, grouped by book
    moments_a = [m for m in moments if m["user_id"] == user_a_id]
    moments_b = [m for m in moments if m["user_id"] == user_b_id]

    books_a = set(m["book_id"] for m in moments_a)
    books_b = set(m["book_id"] for m in moments_b)
    shared_books = books_a & books_b

    # Build overlapping moments — same book, both users
    overlapping_moments = []
    for book in shared_books:
        for ma in moments_a:
            if ma["book_id"] == book:
                overlapping_moments.append({
                    "user": user_a_id,
                    "book_id": book,
                    "book_title": ma.get("book_title", book),
                    "interpretation": ma["cleaned_interpretation"]
                })
        for mb in moments_b:
            if mb["book_id"] == book:
                overlapping_moments.append({
                    "user": user_b_id,
                    "book_id": book,
                    "book_title": mb.get("book_title", book),
                    "interpretation": mb["cleaned_interpretation"]
                })

    if not overlapping_moments:
        print(f"    Skipping — no overlapping moments found")
        continue

    prompt = f"""You are evaluating intellectual compatibility between two readers of the same book.

READER A PORTRAIT:
{json.dumps(portrait_a, indent=2)}

READER B PORTRAIT:
{json.dumps(portrait_b, indent=2)}

THEIR MOMENTS ON THE SAME BOOK:
{json.dumps(overlapping_moments, indent=2)}

Reason through this in steps:

Step 1 - Portrait comparison: Compare the two reader portraits. What dimensions align? What diverges? Are the differences generative or incompatible? If fundamentally incompatible, go straight to Step 3.

Step 2 - Moment-level evidence: Look at their actual interpretations of the same book. What does each reader's response reveal about how they relate to the text? Are they arriving at similar conclusions by different paths? Asking the same question but answering it differently?

Step 3 - Verdict: Classify this pair as exactly one of:
- resonance: aligned stance and emotional register
- mirror: opposing sides of the same dynamic
- contradiction: genuinely different frameworks that create productive tension
- no_match: fundamentally incompatible, no generative connection

Respond in this exact JSON format with no text outside the JSON:
{{
  "step1_portrait_comparison": "...",
  "step2_moment_evidence": "...",
  "match_type": "resonance",
  "confidence": 0.85,
  "evidence_summary": "..."
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        print(f"    Skipping — malformed JSON response")
        continue

    verdict["user_a_id"] = user_a_id
    verdict["user_b_id"] = user_b_id

    log_agent_call(
        agent="compatibility_investigator",
        user_a_id=user_a_id,
        user_b_id=user_b_id,
        input_portrait_a=portrait_a,
        input_portrait_b=portrait_b,
        input_moments=overlapping_moments,
        reasoning_trace=raw,
        verdict={
            "match_type": verdict.get("match_type"),
            "confidence": verdict.get("confidence"),
            "evidence_summary": verdict.get("evidence_summary")
        }
    )

    results.append(verdict)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Saved {len(results)} results to {OUTPUT_FILE}")

types = Counter(r["match_type"] for r in results)
print("Match type distribution:")
for k, v in types.items():
    print(f"  {k}: {v}")