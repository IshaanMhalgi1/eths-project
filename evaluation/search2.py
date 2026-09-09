import json, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

chunks = []
with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.jsonl')) as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

def search(terms, year=None, top=6):
    hits = []
    for c in chunks:
        text = c['text'].lower()
        if all(t in text for t in terms):
            if year and c.get('publication_year') != year:
                continue
            hits.append(c)
    return hits[:top]

def show(label, hits):
    print(f"\n{'='*70}")
    print(f"{label}  ({len(hits)} hits)")
    for h in hits:
        print(f"  [{h['parent_doc_id']}] yr={h.get('publication_year')}")
        print(f"  {h['text'][:250].replace(chr(10),' ')}")
        print()

# Lost pocket book
show("pocket", search(["pocket"]))
show("reward + lost", search(["reward", "lost"]))
show("strayed reward", search(["strayed", "reward"]))
show("runaway reward", search(["runaway", "reward"])[:3])
