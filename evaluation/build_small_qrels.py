import json

# Load original qrels
with open('data/qrels_original.json') as f:
    qrels_original = json.load(f)

# Remove q18 (has docs in 1810)
qrels_small = [q for q in qrels_original if q['query_id'] != 'q18']

# Save
with open('data/qrels_small.json', 'w') as f:
    json.dump(qrels_small, f, indent=2)

print(f"Saved {len(qrels_small)} queries to qrels_small.json")
for q in qrels_small:
    print(f"  {q['query_id']}: {q['query_text'][:50]}...")