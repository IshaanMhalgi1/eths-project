import os
import sys
import json
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from retrieval.bm25 import search_bm25
from temporal.temporal_parser import TemporalParser

parser = TemporalParser()

# Load qrels
with open('data/qrels.json') as f:
    qrels = json.load(f)

# Load chunks for year lookup
chunk_year = {}
with open('data/chunks.jsonl') as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            chunk_year[c['parent_doc_id']] = c.get('publication_year')

# Analyze 1800s queries in detail
for q in qrels:
    if not q.get('relevant_doc_ids'):
        continue
    intent = parser.parse(q['query_text'])
    start_y = intent.get('start_year')
    query_decade = (start_y // 10) * 10 if start_y else None
    
    if query_decade == 1800:
        rel_decades = []
        for doc_id in q['relevant_doc_ids']:
            yr = chunk_year.get(doc_id)
            if yr and 1800 <= yr < 1900:
                rel_decades.append((yr // 10) * 10)
        dec_counts = Counter(rel_decades)
        match = "MATCH" if dec_counts.most_common(1)[0][0] == 1800 else "MISMATCH"
        print(f"{q['query_id']}: {q['query_text'][:60]}... | {match} | dist={dict(dec_counts)}")
        
        # Run BM25 and check top results
        res = search_bm25(q['query_text'], size=50)
        same_dec = [r for r in res if r.get('publication_year') and 1800 <= r['publication_year'] < 1810]
        other_dec = [r for r in res if r.get('publication_year') and not (1800 <= r['publication_year'] < 1810)]
        
        # Check if relevant docs appear in same-decade vs other
        rel_set = set(q['relevant_doc_ids'])
        same_hits = [r for r in same_dec if r['parent_doc_id'] in rel_set]
        other_hits = [r for r in other_dec if r['parent_doc_id'] in rel_set]
        
        print(f"  Same-decade results: {len(same_dec)}, relevant in same: {len(same_hits)}")
        print(f"  Other-decade results: {len(other_dec)}, relevant in other: {len(other_hits)}")
        if same_hits:
            print(f"    Same-decade hits: {[r['parent_doc_id'] + '(' + str(r['publication_year']) + ')' for r in same_hits[:5]]}")
        if other_hits:
            print(f"    Other-decade hits: {[r['parent_doc_id'] + '(' + str(r['publication_year']) + ')' for r in other_hits[:5]]}")
        print()