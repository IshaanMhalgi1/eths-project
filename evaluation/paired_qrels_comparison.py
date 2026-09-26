"""
Paired comparison: same 10 queries, evaluated under BOTH qrels types.

This isolates the qrels effect from the query-set effect. Without this, a
difference between "term-overlap results on 90 queries" and "human results
on 10 queries" is confounded by n, by query mix, and by qrels construction.
"""
import os
import sys
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.run_human_qrels_eval import (
    per_query_metrics, RANKERS, K, DATA, index_for_query, corpus_config_for_query
)
from evaluation.ablation_rankers import get_ranker
import evaluation.ablation_rankers as ar

HUMAN = os.path.join(DATA, 'qrels_human_annotated.json')
SMALL = os.path.join(DATA, 'qrels_small.json')
EXPANDED = os.path.join(DATA, 'qrels_expanded.json')


def load(path, corpus_key=None):
    d = json.load(open(path))
    qs = d['queries'] if 'queries' in d else d
    out = []
    for q in qs:
        if corpus_key:
            out.append({**q, 'corpus': corpus_key})
        else:
            out.append(q)
    return out


def metrics_on(rankers, queries, k=K):
    res = {}
    for name in rankers:
        pq = per_query_metrics(name, queries, k)
        res[name] = {
            'P@10': float(np.mean([x['P@10'] for x in pq])),
            'Recall@10': float(np.mean([x['Recall@10'] for x in pq])),
            'MRR': float(np.mean([x['MRR'] for x in pq])),
        }
    return res


def main():
    human = load(HUMAN)
    # Same 10 queries, but with term-overlap ground truth
    h_ids = {q['query_id']: q['corpus'] for q in human}
    term = []
    for path, key in [(SMALL, 'small'), (EXPANDED, 'expanded')]:
        for q in load(path, key):
            if q['query_id'] in h_ids:
                term.append(q)

    # Small corpus qrels lack q101; expanded lacks q5/q14 scope differences.
    # Use the union available for each query_id, preferring the corpus-matched file.
    term = {q['query_id']: q for q in term}
    term_queries = []
    for q in human:
        qid = q['query_id']
        # small-corpus query -> use small qrels; else expanded qrels
        if q['corpus'] == 'small' and qid in term:
            src = term[qid]
            if src.get('corpus') == 'small':
                term_queries.append(src); continue
        if qid in term:
            term_queries.append({**term[qid], 'corpus': q['corpus']})
        else:
            term_queries.append({**q, 'relevant_doc_ids': []})

    print("=" * 80)
    print("PAIRED COMPARISON: SAME 10 QUERIES, TWO QRELS TYPES")
    print("=" * 80)
    print(f"  queries: {len(human)}   human n_judgments: {json.load(open(HUMAN))['n_judgments']}")
    n_term_rel = sum(len(q['relevant_doc_ids']) for q in term_queries)
    n_hum_rel = sum(len(q['relevant_doc_ids']) for q in human)
    print(f"  relevant docs — term-overlap: {n_term_rel}   human: {n_hum_rel}")

    hr = metrics_on(RANKERS, human)
    tr = metrics_on(RANKERS, term_queries)

    print(f"\n{'':<20}{'--- TERM-OVERLAP QRELS ---':^30}{'--- HUMAN QRELS ---':^30}")
    print(f"{'Ranker':<20}{'R@10':>10}{'MRR':>10}{'R@10':>10}{'MRR':>10}{'dR@10':>10}{'dMRR':>10}")
    for name in RANKERS:
        t, h = tr[name], hr[name]
        print(f"{name:<20}"
              f"{t['Recall@10']:>10.3f}{t['MRR']:>10.3f}"
              f"{h['Recall@10']:>10.3f}{h['MRR']:>10.3f}"
              f"{h['Recall@10']-t['Recall@10']:>+10.3f}"
              f"{h['MRR']-t['MRR']:>+10.3f}")

    print(f"\n{'=' * 80}")
    print("KEY CONTRASTS")
    print("=" * 80)
    for a, b, label in [('BM25', 'Dense', 'BM25 vs Dense'),
                        ('Hybrid+Temporal', 'Hybrid', 'Temporal gain over Hybrid'),
                        ('Final', 'BM25', 'Final vs BM25')]:
        for rname, res in [('term-overlap', tr), ('human', hr)]:
            d_r = res[a]['Recall@10'] - res[b]['Recall@10']
            d_m = res[a]['MRR'] - res[b]['MRR']
            print(f"  {label:<30} [{rname:<12}] "
                  f"dR@10={d_r:+.3f}  dMRR={d_m:+.3f}")
        print()

    # Rank-order stability
    print("=" * 80)
    print("RANKER ORDERING BY MRR")
    print("=" * 80)
    for rname, res in [('term-overlap', tr), ('human', hr)]:
        order = sorted(RANKERS, key=lambda n: -res[n]['MRR'])
        print(f"  {rname:<12}: {' > '.join(order)}")

    json.dump({'term_overlap': tr, 'human': hr},
              open(os.path.join(os.path.dirname(__file__), 'paired_qrels_comparison.json'), 'w'),
              indent=2)


if __name__ == '__main__':
    main()
