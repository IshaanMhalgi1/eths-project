"""Behavioural equivalence classes for the 10 ablation rankers, per corpus.

Two rankers are placed in the same class when they are not distinguishable as
systems: they return the IDENTICAL ranked top-50 document list for EVERY scored
query, and their per-query metric vectors are identical. This mirrors the method
of check_ranker_distinctness.py (which compared selected pairs on the expanded
corpus only) but covers all 10 rankers on both corpora, and additionally
requires metric identity.

Exact top-50 identity is a strict behavioural criterion: two rankers that agree
on all 107 top-50 lists are the same function on this query set, regardless of
how they are implemented. That is the right granularity for significance testing
-- comparing two labels for the same behaviour produces a duplicate test that
inflates the multiple-comparison burden without adding information.

Writes evaluation/equivalence_classes.json:
  {corpus: {n_queries, classes: [{class_id, members, representative}],
            per_query lists are NOT stored (too large); only membership}}
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ablation_rankers import get_ranker
from ranking.shared_hybrid import SHARED_FETCH_SIZE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'evaluation', 'equivalence_classes.json')
ABLATION = os.path.join(REPO, 'evaluation', 'ablation_matrix_results.json')

RANKERS = ["BM25", "Dense", "Hybrid", "Hybrid+Temporal", "Hybrid+Metadata",
           "Final", "BM25+Temporal", "Dense+Temporal", "TemporalOnly",
           "MetadataOnly"]

QRELS = {"small": "data/qrels_small.json", "expanded": "data/qrels_expanded.json"}


def load_scored(path):
    with open(os.path.join(REPO, path), encoding='utf-8') as f:
        return [q for q in json.load(f) if q.get('relevant_doc_ids')]


def metric_identity(a, b, metrics):
    """True if the two rankers' per-query vectors are identical for r and mrr.

    metrics is ablation['metrics'][corpus], i.e. keyed by ranker name, each
    value containing a 'per_query' dict keyed by metric name.
    """
    qa, qb = metrics[a]['per_query'], metrics[b]['per_query']
    return qa['r'] == qb['r'] and qa['mrr'] == qb['mrr']


def main():
    with open(ABLATION, encoding='utf-8') as f:
        ablation = json.load(f)

    out = {'method': 'identical top-50 for every scored query AND identical '
                     'per-query Recall@10/MRR vectors',
           'fetch_size': SHARED_FETCH_SIZE, 'corpora': {}}

    for corpus in ('small', 'expanded'):
        queries = load_scored(QRELS[corpus])
        n = len(queries)
        metrics = ablation['metrics'][corpus]
        n_metric = len(metrics['BM25']['per_query']['r'])
        assert n_metric == n, f"{corpus}: qrels n={n} != metrics n={n_metric}"

        print(f"\n=== {corpus.upper()} (n={n}) : fetching top-{SHARED_FETCH_SIZE} "
              f"for {len(RANKERS)} rankers ===")

        top = {}
        for name in RANKERS:
            fn = get_ranker(name, corpus=corpus)
            top[name] = [[r['parent_doc_id'] for r in
                          fn(q['query_text'], size=SHARED_FETCH_SIZE)]
                         for q in queries]
            ident_self = sum(1 for L in top[name] if L)
            print(f"  {name:<18} fetched, {ident_self}/{n} queries returned a list")

        # ---- equivalence: identical top-50 on every query ----
        # Build classes by exact-match grouping (equality is transitive).
        classes = []
        assigned = {}
        for name in RANKERS:
            placed = False
            for c in classes:
                rep = c['representative']
                if top[name] == top[rep]:
                    c['members'].append(name)
                    assigned[name] = c['class_id']
                    placed = True
                    break
            if not placed:
                cid = len(classes)
                classes.append({'class_id': cid, 'members': [name],
                                'representative': name})
                assigned[name] = cid

        # ---- cross-check: metric identity must agree ----
        for c in classes:
            rep = c['representative']
            for m in c['members']:
                if m == rep:
                    continue
                same_top = top[m] == top[rep]
                same_met = metric_identity(m, rep, metrics)
                c.setdefault('checks', {})[m] = {
                    'identical_top50': same_top, 'identical_metrics': same_met}
                assert same_top and same_met, (
                    f"{corpus}: {m} grouped with {rep} but "
                    f"top50={same_top} metrics={same_met}")

        # pairwise detail for the class membership table
        for i, ci in enumerate(classes):
            for cj in classes[i + 1:]:
                a, b = ci['representative'], cj['representative']
                same_q = sum(1 for x, y in zip(top[a], top[b]) if x == y)
                same_top10 = sum(1 for x, y in zip(top[a], top[b]) if x[:10] == y[:10])
                ci.setdefault('vs', {})[f"{cj['class_id']}"] = {
                    'identical_top50_queries': same_q,
                    'identical_top10_queries': same_top10, 'n': n}

        out['corpora'][corpus] = {
            'n_queries': n,
            'n_classes': len(classes),
            'classes': [{'class_id': c['class_id'],
                         'members': sorted(c['members']),
                         'size': len(c['members']),
                         'representative': c['representative']} for c in classes],
            'assign': assigned,
            'class_pair_matrix': [{'class_id': ci['class_id'],
                                   'vs': ci.get('vs', {})} for ci in classes],
        }

        print(f"  -> {len(classes)} behavioural equivalence classes:")
        for c in classes:
            print(f"     C{c['class_id']} (rep {c['representative']}): "
                  f"{sorted(c['members'])}")

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote evaluation/{os.path.basename(OUT)}")


if __name__ == '__main__':
    main()
