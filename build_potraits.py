import json
import anthropic # type: ignore

INPUT_FILE  = "data/processed/moments_embedded.json"
OUTPUT_FILE = "data/processed/user_portraits.json"

client = anthropic.Anthropic()

with open(INPUT_FILE, "r") as f:
    moments = json.load(f)

# Group moments by user_id
users = {}
for m in moments:
    uid = m["user_id"]
    if uid not in users:
        users[uid] = []
    users[uid].append(m["cleaned_interpretation"])

print(f"Found {len(users)} users. Generating portraits...")

portraits = {}

for i, (user_id, interpretations) in enumerate(users.items()):
    print(f"  [{i+1}/{len(users)}] {user_id}")

    interpretations_text = "\n\n".join(
        f"Interpretation {j+1}:\n{text}"
        for j, text in enumerate(interpretations)
    )

    prompt = f"""You are analyzing a reader's intellectual character based on their written reflections on literary passages.

Here are all of this reader's interpretations:

{interpretations_text}

Based only on what is written above, produce a structured reader portrait with exactly these four fields:

1. thematic_preoccupations: What subjects, questions, or ideas does this reader keep returning to? Be specific — name the actual themes.
2. emotional_register: How does this reader engage emotionally? Pick the dominant mode from: analyze / empathize / argue / surrender. Explain briefly.
3. engagement_style: How does this reader approach a text? Pick the dominant mode from: structure / feeling / friction / identification. Explain briefly.
4. reflection_density: How precise, layered, and developed are their responses? Rate as low / medium / high and explain what you see.

Respond in valid JSON only. No preamble, no explanation outside the JSON. Format:
{{
  "thematic_preoccupations": "...",
  "emotional_register": "...",
  "engagement_style": "...",
  "reflection_density": "..."
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    portrait = json.loads(raw)
    portrait["user_id"] = user_id
    portrait["moment_count"] = len(interpretations)
    portraits[user_id] = portrait

with open(OUTPUT_FILE, "w") as f:
    json.dump(portraits, f, indent=2)

print(f"\nDone. Saved {len(portraits)} portraits to {OUTPUT_FILE}")