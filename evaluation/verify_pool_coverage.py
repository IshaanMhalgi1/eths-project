"""Verify pool construction: is every retriever represented in the judged pool?"""
import json
import os

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')

blind = json.load(open(os.path.join(DATA, 'annotation_pools.json')))
scored = json.load(open(os.path.join(DATA, 'annotation_pools_with_scores.json')))
sc = {q['query_id']: q for q in scored['queries']}

print("=" * 84)
print("POOL COVERAGE BY RETRIEVER (does each retriever's output actually get judged?)")
print("=" * 84)
print(f"{'query':<7}{'pool':<7}{'BM25 in':<10}{'Dense in':<11}{'Hybrid in':<11}{'all three':<12}")
tot = {'bm25': 0, 'dense': 0, 'hybrid': 0, 'all': 0, 'n': 0}
for q in blind['queries']:
    qid = q['query_id']
    s = sc[qid]
    cands = [c['canonical_parent_doc_id'] for c in q['candidates']]
    n = len(cands)
    inb = sum(1 for c in s['candidates'] if c['retriever_scores'].get('bm25') is not None)
    ind = sum(1 for c in s['candidates'] if c['retriever_scores'].get('dense') is not None)
    inh = sum(1 for c in s['candidates'] if c['retriever_scores'].get('hybrid') is not None)
    allthree = sum(1 for c in s['candidates']
                   if all(c['retriever_scores'].get(k) is not None for k in ('bm25', 'dense', 'hybrid')))
    print(f"{qid:<7}{n:<7}{inb:<10}{ind:<11}{inh:<11}{allthree:<12}")
    tot['bm25'] += inb; tot['dense'] += ind; tot['hybrid'] += inh
    tot['all'] += allthree; tot['n'] += n

print(f"\n  TOTAL pool = {tot['n']}")
print(f"    contributed by BM25:   {tot['bm25']} ({100*tot['bm25']/tot['n']:.0f}%)")
print(f"    contributed by Dense:  {tot['dense']} ({100*tot['dense']/tot['n']:.0f}%)")
print(f"    contributed by Hybrid: {tot['hybrid']} ({100*tot['hybrid']/tot['n']:.0f}%)")
print(f"    found by all three:    {tot['all']} ({100*tot['all']/tot['n']:.0f}%)")

print("\n  v1 comparison (top-20 of merged-by-raw-score list, from the old pools):")
print("    BM25 dominated the pool because scores ~5-14 sorted above Dense ~0.7")
print("    and Hybrid ~0.016. Dense's own top-20 was largely absent.")
print("    That made any Dense document outside the pool unfindable, biasing")
print("    every v1 Dense measurement AGAINST Dense.")
