"""
Single-annotator v1-vs-v2 comparison against independent term-overlap qrels.

PURPOSE
-------
The v1 pilot produced weak inter-annotator agreement (63.9%, kappa=0.452)
with a known confound: the v1 Dense candidate pool was built by merging
retrievers on incompatible raw scores and truncating, and annotators could
see retriever names/scores. That left two competing explanations:

    (a) the broken pool made annotators disagree, or
    (b) the annotators genuinely disagree about relevance.

With only annotator 1 available for v2 so far, inter-annotator kappa cannot
be computed. But explanation (a) is still testable, because it predicts that
fixing the pool should move annotator 1 CLOSER to an independent,
non-lexical ground truth. Explanation (b) predicts it should not.

This script holds the annotator and the reference qrels fixed and varies only
the pool generation, on the queries the two pools share.

THIS IS PRELIMINARY. It uses one annotator and is not a reliability estimate.
It cannot be cited as agreement.
"""
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'data')


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def cohen_kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None
    cats = sorted({l for p in pairs for l in p})
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = [0] * k, [0] * k
    for a, b in pairs:
        ca[idx[a]] += 1
        cb[idx[b]] += 1
    pe = sum((ca[i] / n) * (cb[i] / n) for i in range(k))
    if pe == 1.0:
        return None
    return (po - pe) / (1 - pe)


def labels_by_doc(export, a):
    """{(qid, doc): grade} for one annotator's export."""
    key = f'relevance_annotator{a}'
    out = {}
    for q in export['queries']:
        for c in q['candidates']:
            v = c.get(key)
            if v is not None:
                out[(q['query_id'], c['canonical_parent_doc_id'])] = v
    return out


def binarize(g):
    """Match qrels semantics: partial (1) and relevant (2) both count."""
    return 1 if g >= 1 else 0


def main():
    v1 = load('annotation_results_v1_annotator1.json') if os.path.exists(
        os.path.join(DATA, 'annotation_results_v1_annotator1.json')) else None
    v2 = load('annotation_v2_annotator1.json')

    # Independent reference: term-overlap qrels are binary.
    # These files are stored as a bare list of query objects.
    ref = load('qrels_small.json')
    ref_queries = ref['queries'] if isinstance(ref, dict) else ref
    ref_map = defaultdict(set)
    for q in ref_queries:
        for d in q['relevant_doc_ids']:
            ref_map[q['query_id']].add(d)

    if v1 is None:
        print("v1 annotator-1 export not found in data/; skipping the delta.")
        print("Place it as data/annotation_results_v1_annotator1.json to enable.")
        v1 = {'queries': []}

    l1 = labels_by_doc(v1, 1)
    l2 = labels_by_doc(v2, 1)

    q1 = {q['query_id'] for q in v1['queries']}
    q2 = {q['query_id'] for q in v2['queries']}
    shared = sorted(q1 & q2)
    print("=" * 74)
    print("PRELIMINARY: annotator 1 vs term-overlap reference, by pool generation")
    print("=" * 74)
    print(f"  v1 queries: {len(q1)}   v2 queries: {len(q2)}   shared: {len(shared)}")
    print(f"  shared: {shared}\n")

    print("Binary agreement (partial counts as relevant, matching qrels semantics)")
    print("-" * 74)
    print(f"{'query':<8} {'v1 n':>5} {'v1 agree':>9} {'v1 kappa':>9} "
          f"{'v2 n':>5} {'v2 agree':>9} {'v2 kappa':>9}  delta")
    print("-" * 74)

    agg = {1: [], 2: []}
    for qid in shared:
        row = []
        for gen, labs in ((1, l1), (2, l2)):
            pairs = []
            for (q, d), g in labs.items():
                if q != qid:
                    continue
                if d not in ref_map[qid]:
                    continue  # only score docs the reference has an opinion on
                pairs.append((binarize(g), 1))
            row.append(pairs)
        cells = []
        for pairs in row:
            n = len(pairs)
            agree = sum(1 for a, _ in pairs if a == 1)
            pct = 100 * agree / n if n else float('nan')
            k = cohen_kappa(pairs)
            cells.append((n, pct, k))
        agg[1].extend(row[0])
        agg[2].extend(row[1])

        d = (cells[1][1] - cells[0][1]) if cells[0][1] == cells[0][1] else float('nan')
        ds = f"{d:+.1f}pp" if d == d else "n/a"
        k0 = f"{cells[0][2]:.3f}" if cells[0][2] is not None else "n/a"
        k1 = f"{cells[1][2]:.3f}" if cells[1][2] is not None else "n/a"
        print(f"{qid:<8} {cells[0][0]:>5} {cells[0][1]:>8.1f}% {k0:>9} "
              f"{cells[1][0]:>5} {cells[1][1]:>8.1f}% {k1:>9}  {ds}")

    print("-" * 74)
    print("Pool precision: share of pool candidates the reference calls relevant")
    print("-" * 74)
    for gen, labs, name in ((1, l1, 'v1'), (2, l2, 'v2')):
        hit = tot = 0
        for qid in shared:
            for (q, d), g in labs.items():
                if q != qid or d not in ref_map[qid]:
                    continue
                tot += 1
                hit += 1 if binarize(g) == 1 else 0
        print(f"  {name}: {hit}/{tot} = {100*hit/tot:.1f}%" if tot else f"  {name}: n/a")

    print()
    print("=" * 74)
    print("READ THIS CORRECTLY")
    print("=" * 74)
    print("""  This compares ONE annotator against the term-overlap reference.
  It is NOT an inter-annotator agreement number and must not replace the
  kappa the v2 study is designed to produce.

  It also only scores candidates the reference already has an opinion on,
  so it says nothing about pool COVERAGE (v1's real defect). A broken pool
  can agree with the reference on the docs it happens to contain while
  missing the relevant ones entirely.

  The v2 kappa remains blocked on annotator 2's blind export.
""")


if __name__ == '__main__':
    main()
