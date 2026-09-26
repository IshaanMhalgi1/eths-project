"""Check for systematic annotator bias and disagreement structure."""
import json
import os
import sys
from collections import Counter

a1 = json.load(open(sys.argv[1]))
a2 = json.load(open(sys.argv[2]))

q1 = {q['query_id']: q for q in a1['queries']}
q2 = {q['query_id']: q for q in a2['queries']}

pairs = []
for qid in q1:
    c1 = {c['canonical_parent_doc_id']: c['relevance_annotator1']
          for c in q1[qid]['candidates']}
    c2 = {c['canonical_parent_doc_id']: c['relevance_annotator2']
          for c in q2[qid]['candidates']}
    for d in set(c1) & set(c2):
        if c1[d] is not None and c2[d] is not None:
            pairs.append((qid, d, c1[d], c2[d]))

print("=" * 66)
print("ANNOTATOR SCORE DISTRIBUTIONS")
print("=" * 66)
c_a1 = Counter(p[2] for p in pairs)
c_a2 = Counter(p[3] for p in pairs)
n = len(pairs)
print(f"{'Score':<8}{'Annotator 1':<16}{'Annotator 2':<16}")
for s in [0, 1, 2]:
    print(f"{s:<8}{c_a1[s]:>4} ({100*c_a1[s]/n:>5.1f}%){'':<4}{c_a2[s]:>4} ({100*c_a2[s]/n:>5.1f}%)")

mean1 = sum(p[2] for p in pairs) / n
mean2 = sum(p[3] for p in pairs) / n
print(f"\n  Mean score:  A1={mean1:.3f}  A2={mean2:.3f}  diff={mean1-mean2:+.3f}")

# Asymmetry: when they disagree, who is higher?
higher_a2 = sum(1 for p in pairs if p[3] > p[2])
higher_a1 = sum(1 for p in pairs if p[2] > p[3])
dis = higher_a1 + higher_a2
print(f"\n  On disagreement ({dis} cases):")
print(f"    A2 more generous: {higher_a2} ({100*higher_a2/dis:.0f}%)")
print(f"    A1 more generous: {higher_a1} ({100*higher_a1/dis:.0f}%)")

# Distance distribution
print("\n" + "=" * 66)
print("DISAGREEMENT MAGNITUDE")
print("=" * 66)
dists = Counter(abs(p[2] - p[3]) for p in pairs)
for d in sorted(dists):
    print(f"  |diff|={d}: {dists[d]:>4} ({100*dists[d]/n:>5.1f}%)")
adjacent = sum(v for k, v in dists.items() if k == 1)
extreme = dists.get(2, 0)
print(f"\n  Adjacent (0v1 or 1v2): {adjacent} ({100*adjacent/n:.0f}%) — gradation disagreement")
print(f"  Extreme (0 vs 2):      {extreme} ({100*extreme/n:.0f}%) — binary disagreement")

# Per-query, does a1 or a2 label more relevant?
print("\n" + "=" * 66)
print("PER-QUERY RELEVANT RATE (>=1 means partially-or-better)")
print("=" * 66)
print(f"{'Query':<7}{'n':<5}{'A1 rel%':<12}{'A2 rel%':<12}{'Delta':<10}")
for qid in q1:
    qp = [p for p in pairs if p[0] == qid]
    if not qp:
        continue
    r1 = sum(1 for p in qp if p[2] >= 1) / len(qp)
    r2 = sum(1 for p in qp if p[3] >= 1) / len(qp)
    print(f"{qid:<7}{len(qp):<5}{100*r1:>5.0f}%{'':<7}{100*r2:>5.0f}%{'':<7}{100*(r1-r2):>+5.0f}pp")

# Candidate pool coverage limitation
print("\n" + "=" * 66)
print("POOLING LIMITATION")
print("=" * 66)
print("  Qrels are built from retrieved candidates (union of BM25/Dense/Hybrid top-20).")
print("  A relevant document outside the pool is unjudged, so Recall@10 is a")
print("  POOL-RELATIVE estimate, not absolute recall (standard TREC pooling bias).")
print(f"\n  Pool sizes: {', '.join(f'{qid}={sum(1 for p in pairs if p[0]==qid)}' for qid in q1)}")
