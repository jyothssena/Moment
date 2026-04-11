import json

with open("interpretations_test.json", "r") as f:
    data = json.load(f)

with open("final_test_interpretations.jsonl", "w") as f:
    for record in data:
        f.write(json.dumps(record) + "\n")