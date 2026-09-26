"""Does the reported n=90 include the 20 zero-reference-doc queries?

Ground truth from the data and the runner's own filter, not from prose.
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'data')


def load(n):
    d = json.load(open(os.path.join(DATA, n)))
    return d['queries'] if isinstance(d, dict) else d


qs = load('qrels_expanded.json')
ZERO = {'q28', 'q36', 'q81', 'q82', 'q83', 'q84', 'q85', 'q86', 'q87', 'q88',
        'q90', 'q91', 'q92', 'q93', 'q94', 'q95', 'q96', 'q97', 'q99', 'q100'}

# The exact filter from run_ablation_matrix_v2.py line 22
kept = [q for q in qs if q.get('relevant_doc_ids')]
dropped = [q for q in qs if not q.get('relevant_doc_ids')]

print("qrels_expanded.json total queries      :", len(qs))
print("after `if q.get('relevant_doc_ids')`   :", len(kept))
print("dropped by that filter                 :", len(dropped))
print("dropped ids match the 20 zero-doc set  :",
      {q['query_id'] for q in dropped} == ZERO)
print()
print("=> the n=90 reported in the master reports is 110 - 20 =", len(kept))
print("=> the 20 zero-reference-doc queries are EXCLUDED, not included.")

# What relevant-doc counts survive into the n=90?
from collections import Counter
c = Counter(len(q['relevant_doc_ids']) for q in kept)
print()
print("relevant-doc counts inside the n=90    :", dict(sorted(c.items())))
print("queries with exactly 5 (the cap)        :", c[5])
print("queries with exactly 1                 :", c[1])

# same check for the small corpus
qs_small = load('qrels_small.json')
kept_small = [q for q in qs_small if q.get('relevant_doc_ids')]
print()
print("small corpus: total", len(qs_small), "-> kept", len(kept_small),
      "counts", dict(sorted(Counter(len(q['relevant_doc_ids']) for q in kept_small).items())))
