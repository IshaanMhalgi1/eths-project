"""
Manual corpus search for the 4 problem queries.
Searches raw chunk text without any retrieval scoring.
"""
import os, sys, json, re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.jsonl')
chunks = []
with open(CHUNKS_PATH) as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

def search_corpus(terms, year_range=None, top_n=10):
    """Find chunks containing ALL of the given terms (case-insensitive)."""
    results = []
    for c in chunks:
        text = c['text'].lower()
        if all(t.lower() in text for t in terms):
            yr = c.get('publication_year')
            if year_range and yr:
                if not (year_range[0] <= yr <= year_range[1]):
                    continue
            results.append(c)
    return results[:top_n]

def show(label, results):
    print(f"\n{'='*70}")
    print(f"QUERY: {label}  ({len(results)} hits)")
    print(f"{'='*70}")
    for r in results[:8]:
        print(f"  [{r['parent_doc_id']}] year={r.get('publication_year')} | {r['text'][:200].replace(chr(10),' ')}...")
        print()

# ── 1. Lost pocket book reward ───────────────────────────────────────────────
show("lost pocket book reward",
     search_corpus(["pocket", "lost"]))

show("lost pocket book reward (alt: reward + lost)",
     search_corpus(["reward", "lost"]))

# ── 2. Obituaries deaths 1806 ────────────────────────────────────────────────
show("obituaries deaths 1806",
     search_corpus(["obituary"], year_range=(1806, 1806)))

show("obituaries deaths 1806 (died / departed this life)",
     search_corpus(["died"], year_range=(1806, 1806)))

# ── 3. Military officer appointments ─────────────────────────────────────────
show("military officer appointments (appoint + officer)",
     search_corpus(["appointed", "officer"]))

show("military officer appointments (commission + officer)",
     search_corpus(["commission", "officer"]))

# ── 4. Ship arrivals cargo from Europe 1803 ───────────────────────────────────
show("ship arrivals cargo from Europe 1803 (arrived + ship + cargo)",
     search_corpus(["arrived", "ship", "cargo"], year_range=(1803, 1803)))

show("ship arrivals cargo from Europe 1803 (arrived + brig)",
     search_corpus(["arrived", "brig"], year_range=(1803, 1803)))

show("ship arrivals cargo from Europe 1803 (arrived + vessel)",
     search_corpus(["arrived", "vessel"], year_range=(1803, 1803)))
