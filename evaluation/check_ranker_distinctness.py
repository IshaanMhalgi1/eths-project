"""Check whether the 'hybrid' and 'temporal' rankers are actually distinct.

The measured metrics for Hybrid+Temporal, BM25+Temporal and TemporalOnly are
identical to four decimals on all 88 clean queries, and Hybrid matches BM25
exactly. Either that is a coincidence or the rankers are producing the same
ordering. This compares the actual ranked document lists.
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ablation_rankers import get_ranker
from ranking.shared_hybrid import SHARED_FETCH_SIZE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(REPO, 'data', 'qrels_expanded.json'), encoding='utf-8') as f:
    qrels = [q for q in json.load(f) if q.get('relevant_doc_ids')]

GROUPS = [
    ('BM25', 'Hybrid'),
    ('Hybrid+Temporal', 'BM25+Temporal'),
    ('Hybrid+Temporal', 'TemporalOnly'),
    ('BM25+Temporal', 'TemporalOnly'),
    ('BM25', 'TemporalOnly'),
]

results = {}
for name in sorted({n for g in GROUPS for n in g}):
    fn = get_ranker(name, corpus='expanded')
    results[name] = [
        [r['parent_doc_id'] for r in fn(q['query_text'], size=SHARED_FETCH_SIZE)]
        for q in qrels
    ]

print(f"queries compared: {len(qrels)}\n")
for a, b in GROUPS:
    same = sum(1 for x, y in zip(results[a], results[b]) if x == y)
    top10 = sum(1 for x, y in zip(results[a], results[b]) if x[:10] == y[:10])
    print(f"{a:<18} vs {b:<18} identical full top-{SHARED_FETCH_SIZE}: "
          f"{same}/{len(qrels)}   identical top-10: {top10}/{len(qrels)}")
