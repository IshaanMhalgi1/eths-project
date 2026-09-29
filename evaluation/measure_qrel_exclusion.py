"""Final retrieval measurement on the expanded corpus, n=88.

Query set
---------
data/qrels_expanded.json holds 108 entries after removing q89 and q98. Of the
remaining, 88 carry reference documents; the other 20 are q28, q36 and the 18
late-decade queries whose term-overlap match produced nothing once the corpus
was understood to end at 1869. The evaluation filters to the 88.

q89 ("women suffrage movement 1870s") and q98 ("bicycle craze transportation
1880s") were dropped because both are range-type queries with no parseable
year, so qrel generation fell back to raw term overlap and labelled documents
from 1802-1867 as relevant to 1870s/1880s topics.

Baseline
--------
The stored ablation_matrix_results_v2.json was computed on n=90 (the 88 plus
q89/q98). Both are reported here so the effect of dropping the two is visible.

Metric definitions are copied verbatim from evaluation/rrf_sweep.py.
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ablation_rankers import get_ranker
from ranking.shared_hybrid import SHARED_FETCH_SIZE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QRELS = os.path.join(REPO, 'data', 'qrels_expanded.json')

# Stored in evaluation/ablation_matrix_results_v2.json, measured on n=90.
STORED_N90 = {
    'BM25': (0.32229276895943565, 0.2711111111111111),
    'Hybrid': (0.32229276895943565, 0.2711111111111111),
    'Hybrid+Temporal': (0.4914814814814815, 0.36666666666666664),
    'BM25+Temporal': (0.4914814814814815, 0.36666666666666664),
    'TemporalOnly': (0.4914814814814815, 0.36666666666666664),
    'Final': (0.45791005291005293, 0.3377777777777778),
}

RANKERS = ['BM25', 'Hybrid', 'Hybrid+Temporal', 'BM25+Temporal',
           'TemporalOnly', 'Final']


def evaluate_method(candidates, true_docs, k=10):
    """Verbatim from evaluation/rrf_sweep.py."""
    seen = set()
    dedup = []
    for r in candidates:
        pid = r['parent_doc_id']
        if pid not in seen:
            seen.add(pid)
            dedup.append(pid)
        if len(dedup) == k:
            break

    hits = [1 if d in true_docs else 0 for d in dedup]
    n_rel = max(len(true_docs), 1)

    p_at_k = sum(hits) / k
    recall = sum(hits) / n_rel

    mrr = 0
    for rank, d in enumerate(dedup):
        if d in true_docs:
            mrr = 1.0 / (rank + 1)
            break

    return {"P@10": p_at_k, "Recall@10": recall, "MRR": mrr}


def main():
    with open(QRELS, encoding='utf-8') as f:
        raw = json.load(f)
    qrels = [q for q in raw if q.get('relevant_doc_ids')]

    noref = [q['query_id'] for q in raw if not q.get('relevant_doc_ids')]
    print(f"qrels file entries      : {len(raw)}")
    print(f"scored queries (n)      : {len(qrels)}")
    print(f"skipped, no references  : {len(noref)}")
    print(f"  {', '.join(noref)}")
    assert not ({"q89", "q98"} & {q['query_id'] for q in raw}), \
        "q89/q98 should have been dropped"

    print(f"\n{'='*72}\nFINAL expanded-corpus results (n={len(qrels)})\n{'='*72}")
    print(f"{'Ranker':<18}{'MRR':>9}{'dMRR':>9}{'R@10':>9}{'dR@10':>9}{'P@10':>9}")
    print('-' * 72)

    out = {}
    for name in RANKERS:
        fn = get_ranker(name, corpus='expanded')
        acc = {"P@10": 0.0, "Recall@10": 0.0, "MRR": 0.0}
        for q in qrels:
            m = evaluate_method(fn(q['query_text'], size=SHARED_FETCH_SIZE),
                                set(q['relevant_doc_ids']), k=10)
            for kk in acc:
                acc[kk] += m[kk]
        n = len(qrels)
        row = {kk: acc[kk] / n for kk in acc}
        base_mrr, base_rec = STORED_N90[name]
        out[name] = row
        print(f"{name:<18}{row['MRR']:>9.4f}{row['MRR']-base_mrr:>+9.4f}"
              f"{row['Recall@10']:>9.4f}{row['Recall@10']-base_rec:>+9.4f}"
              f"{row['P@10']:>9.4f}")
    print('-' * 72)
    print("dMRR / dR@10 are against the stored n=90 baseline (includes q89/q98).")

    with open(os.path.join(REPO, 'evaluation', 'qrel_exclusion_impact.json'),
              'w', encoding='utf-8') as f:
        json.dump({'n_scored': len(qrels), 'n_file_entries': len(raw),
                   'skipped_no_references': noref,
                   'dropped_as_unsatisfiable': ['q89', 'q98'],
                   'final_n88': out, 'stored_n90_baseline': STORED_N90},
                  f, indent=2)
    print("\nwrote evaluation/qrel_exclusion_impact.json")


if __name__ == '__main__':
    main()
