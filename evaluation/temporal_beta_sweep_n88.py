"""Temporal beta sweep on the corrected 70-year corpus, n=88.

Re-measures the central claim of TEMPORAL_RERANKING_FINDING_REPORT.md — that
temporal reranking flips from net-negative on a 10-year corpus to positive on
the larger corpus — on the corrected query set (n=88, q89/q98 removed) and
against a genuine hybrid base (RRF alpha=0.5, dense actually contributing),
which the ablation matrix does not use because its expanded config sets
alpha=1.0.

Reports bootstrap CIs so the claim is testable rather than asserted.
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ablation_rankers import (
    rank_bm25, rank_hybrid, rank_hybrid_temporal, SMALL_INDEX, DEFAULT_INDEX,
)
from ranking.shared_hybrid import SHARED_FETCH_SIZE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
N_BOOT = 10000


def evaluate(candidates, true_docs, k=10):
    seen, dedup = set(), []
    for r in candidates:
        p = r['parent_doc_id']
        if p not in seen:
            seen.add(p)
            dedup.append(p)
        if len(dedup) == k:
            break
    hits = [1 for d in dedup if d in true_docs]
    n_rel = max(len(true_docs), 1)
    mrr = 0.0
    for i, d in enumerate(dedup):
        if d in true_docs:
            mrr = 1.0 / (i + 1)
            break
    return sum(hits) / n_rel, mrr


def boot_ci(vals, n_boot=N_BOOT):
    rng = np.random.default_rng(42)
    arr = np.array(vals)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return arr.mean(), lo, hi


def run(corpus, index, qrels, alpha_rrf):
    print(f"\n{'='*76}\n{corpus}  (n={len(qrels)}, RRF alpha={alpha_rrf})\n{'='*76}")
    print(f"{'Method':<26}{'R@10 [95% CI]':>26}{'MRR [95% CI]':>26}")
    print('-' * 76)

    rows = {}

    def record(label, fn):
        rec, mrr = [], []
        for q in qrels:
            r, m = evaluate(fn(q['query_text']), set(q['relevant_doc_ids']))
            rec.append(r)
            mrr.append(m)
        rm, rl, rh = boot_ci(rec)
        mm, ml, mh = boot_ci(mrr)
        rows[label] = {'recall': rm, 'recall_ci': [rl, rh],
                       'mrr': mm, 'mrr_ci': [ml, mh],
                       'per_query': {'recall': rec, 'mrr': mrr}}
        print(f"{label:<26}{rm:>8.3f} [{rl:.3f}, {rh:.3f}]{mm:>15.3f} [{ml:.3f}, {mh:.3f}]")

    record('BM25', lambda t: rank_bm25(t, size=SHARED_FETCH_SIZE, index_name=index))
    record(f'Hybrid (alpha={alpha_rrf})',
           lambda t: rank_hybrid(t, size=SHARED_FETCH_SIZE,
                                 index_name=index, alpha=alpha_rrf))
    for b in BETAS:
        if b == 0.0:
            continue
        record(f'Hybrid+Temporal (b={b})',
               lambda t, b=b: rank_hybrid_temporal(
                   t, size=SHARED_FETCH_SIZE, index_name=index,
                   alpha_hybrid=0.7, beta_temporal=b, alpha_rrf=alpha_rrf))
    return rows


def main():
    with open(os.path.join(REPO, 'data', 'qrels_expanded.json'), encoding='utf-8') as f:
        exp = [q for q in json.load(f) if q.get('relevant_doc_ids')]
    with open(os.path.join(REPO, 'data', 'qrels_small.json'), encoding='utf-8') as f:
        small = [q for q in json.load(f) if q.get('relevant_doc_ids')]
    print(f"expanded scored queries: {len(exp)}   small scored queries: {len(small)}")

    out = {
        'expanded_1800_1869': run('EXPANDED CORPUS (1800-1869)', DEFAULT_INDEX, exp, 0.5),
        'small_1800_1810': run('SMALL CORPUS (1800-1810)', SMALL_INDEX, small, 0.7),
    }
    with open(os.path.join(REPO, 'evaluation', 'temporal_beta_sweep_n88.json'),
              'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nwrote evaluation/temporal_beta_sweep_n88.json")


if __name__ == '__main__':
    main()
