"""Sanity check on dense retrieval.

Dense Recall@10 on the expanded corpus measured 0.0711 against BM25's 0.2711,
which is low enough that the retriever itself may be defective rather than the
qrels being unkind. This inspects raw output for a query whose relevant
documents are known, to separate "dense returns sensible nearest neighbours"
from "dense returns nonsense".

Checks:
  1. Score range. Embeddings are L2-normalised, so cosine similarity should sit
     in [-1, 1] and cluster in the 0.3-0.9 band, not collapse near zero.
  2. Score ordering is strictly decreasing (scores must actually rank).
  3. Returned parent_doc_ids resolve to real corpus documents.
  4. Where a known relevant document appears, its rank.
  5. Text overlap between query and returned documents.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json

from retrieval.dense import search_dense
from retrieval.bm25 import search_bm25
from retrieval import dense as dense_mod

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(REPO, 'data', 'qrels_expanded.json'), encoding='utf-8') as f:
    qrels = {q['query_id']: q for q in json.load(f) if q.get('relevant_doc_ids')}

QID = 'q1'
query = qrels[QID]['query_text']
rel = set(qrels[QID]['relevant_doc_ids'])

print(f"query   : {QID}  {query!r}")
print(f"relevant: {sorted(rel)}\n")

for name, fn in (('DENSE', search_dense), ('BM25', search_bm25)):
    res = fn(query, size=10)
    print(f"=== {name} top-10 ===")
    for i, r in enumerate(res, 1):
        pid = r['parent_doc_id']
        mark = ' <-- RELEVANT' if pid in rel else ''
        txt = (r.get('text') or '')[:78].replace('\n', ' ')
        print(f"{i:>3}. score={r['score']:.4f} year={r.get('publication_year')} "
              f"id={pid}{mark}")
        print(f"     {txt}")
    print()

print("=== DENSE diagnostics ===")
res = search_dense(query, size=50)
scores = [r['score'] for r in res]
print(f"  n returned              : {len(res)}")
print(f"  score min / max         : {min(scores):.4f} / {max(scores):.4f}")
print(f"  score spread            : {max(scores) - min(scores):.4f}")
mono = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
print(f"  scores monotonically sorted: {mono}")
tied = sum(1 for i in range(len(scores) - 1) if abs(scores[i] - scores[i + 1]) < 1e-6)
print(f"  adjacent ties (<1e-6)   : {tied}")
ids = [r['parent_doc_id'] for r in res]
print(f"  unique parent docs      : {len(set(ids))} / {len(ids)}")
if rel:
    ranks = [(ids.index(d) + 1) for d in rel if d in ids]
    print(f"  relevant docs retrieved : {len(ranks)} / {len(rel)} at ranks {ranks}")
else:
    print("  relevant docs retrieved : n/a")

# How large is the candidate pool the searcher is allowed to choose from?
print()
print("=== retrieval internals ===")
for attr in ('DENSE_INDEX', 'DENSE_PATH', 'INDEX_PATH', 'DOC_PATH', 'CHUNKS_PATH'):
    if hasattr(dense_mod, attr):
        print(f"  {attr} = {getattr(dense_mod, attr)}")
