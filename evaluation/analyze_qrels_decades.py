"""
Analyze qrels: query decade vs relevant document decades
"""
import os
import sys
import json
from collections import defaultdict, Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from temporal.temporal_parser import TemporalParser

parser = TemporalParser()

# Load qrels
with open('data/qrels.json') as f:
    qrels = json.load(f)

# Load chunks
chunk_year = {}
with open('data/chunks.jsonl') as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            chunk_year[c['parent_doc_id']] = c.get('publication_year')

# Analyze each query
by_query_decade = defaultdict(list)

for q in qrels:
    if not q.get('relevant_doc_ids'):
        continue
    intent = parser.parse(q['query_text'])
    start_y = intent.get('start_year')
    query_decade = (start_y // 10) * 10 if start_y else None
    
    # Get relevant doc decades
    rel_decades = []
    for doc_id in q['relevant_doc_ids']:
        yr = chunk_year.get(doc_id)
        if yr and 1800 <= yr < 1900:
            rel_decades.append((yr // 10) * 10)
    
    if rel_decades:
        dec_counts = Counter(rel_decades)
        most_common = dec_counts.most_common(1)[0]
        match = "MATCH" if most_common[0] == query_decade else "MISMATCH"
        by_query_decade[query_decade].append({
            'qid': q['query_id'],
            'query_decade': query_decade,
            'top_rel_decade': most_common[0],
            'match': match,
            'n_rel': len(q['relevant_doc_ids']),
            'n_in_top': most_common[1],
            'dist': dict(dec_counts)
        })
        print(f"{q['query_id']}: query_decade={query_decade}s, top_rel_decade={most_common[0]}s ({most_common[1]}/{len(q['relevant_doc_ids'])}), {match}")
    else:
        print(f"{q['query_id']}: query_decade={query_decade}s, NO REL DOCS WITH YEAR")

print("\n=== SUMMARY BY QUERY DECADE ===")
for qdec in sorted(by_query_decade):
    items = by_query_decade[qdec]
    matches = sum(1 for i in items if i['match'] == 'MATCH')
    print(f"{qdec}s: {len(items)} queries, {matches} match, {len(items)-matches} mismatch")

print("\n=== DETAILED MISMATCHES ===")
for qdec in sorted(by_query_decade):
    for i in by_query_decade[qdec]:
        if i['match'] == 'MISMATCH':
            print(f"  {i['qid']}: query={i['query_decade']}s -> rel={i['top_rel_decade']}s ({i['n_in_top']}/{i['n_rel']}), dist={i['dist']}")