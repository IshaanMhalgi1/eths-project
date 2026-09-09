"""
Grid search over alpha_hybrid / beta_temporal.
Key insight: pre-compute all hybrid candidates per query ONCE,
then do temporal re-scoring entirely in Python — no re-querying.
"""
import os, sys, json, itertools, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from retrieval.hybrid import search_hybrid
from ranking.temporal_ranker import calculate_temporal_score
from temporal.temporal_parser import TemporalParser

parser = TemporalParser()

QRELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')
with open(QRELS_PATH) as f:
    qrels = json.load(f)
qrels = [q for q in qrels if q.get('relevant_doc_ids')]   # skip empty

# ── Step 1: Pre-fetch hybrid candidates for every query (one round trip) ──
print("Pre-fetching hybrid candidates for all queries…")
cached = []
for i, q in enumerate(qrels):
    candidates = search_hybrid(q['query_text'], size=100, alpha=0.5)
    intent = parser.parse(q['query_text'])
    cached.append({
        'query': q['query_text'],
        'true_docs': q['relevant_doc_ids'],
        'query_start': intent.get('start_year'),
        'query_end': intent.get('end_year'),
        'candidates': candidates,
    })
    print(f"  [{i+1}/{len(qrels)}] {q['query_text'][:50]}")

print("Done. Running grid search…\n")


def evaluate_cached(cached_data, alpha_hybrid, beta_temporal, k=10):
    p_at_k_list, recall_list, mrr_list = [], [], []

    for entry in cached_data:
        candidates = entry['candidates']
        true_docs = entry['true_docs']
        qs = entry['query_start']
        qe = entry['query_end']

        # Re-score with current weights — no network calls
        if qs is None or qe is None:
            # No temporal constraint: just sort by hybrid base score
            scored = sorted(candidates, key=lambda r: r['score'], reverse=True)
        else:
            # Collect raw scores
            for r in candidates:
                t, _ = calculate_temporal_score(r.get('historical_start'), r.get('historical_end'), qs, qe)
                r['_raw_t'] = t

            # Normalize both within the batch
            h_scores = [r['score'] for r in candidates]
            t_scores = [r['_raw_t'] for r in candidates]

            min_h, max_h = min(h_scores), max(h_scores)
            min_t, max_t = min(t_scores), max(t_scores)

            for r in candidates:
                nh = (r['score'] - min_h) / (max_h - min_h) if max_h > min_h else r['score']
                nt = (r['_raw_t'] - min_t) / (max_t - min_t) if max_t > min_t else r['_raw_t']
                r['_final'] = alpha_hybrid * nh + beta_temporal * nt

            scored = sorted(candidates, key=lambda r: r['_final'], reverse=True)

        # Deduplicate by parent_doc_id
        seen = set()
        dedup = []
        for r in scored:
            pid = r['parent_doc_id']
            if pid not in seen:
                seen.add(pid)
                dedup.append(pid)
            if len(dedup) == k:
                break

        hits = [1 if d in true_docs else 0 for d in dedup]
        n_rel = max(len(true_docs), 1)
        p_at_k_list.append(sum(hits) / k)
        recall_list.append(sum(hits) / n_rel)
        mrr = 0
        for rank, d in enumerate(dedup):
            if d in true_docs:
                mrr = 1.0 / (rank + 1)
                break
        mrr_list.append(mrr)

    return {
        "P@10": float(np.mean(p_at_k_list)),
        "Recall@10": float(np.mean(recall_list)),
        "MRR": float(np.mean(mrr_list)),
    }


# ── Step 2: Grid search ──
alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
betas  = [0.1, 0.2, 0.3, 0.4, 0.5]

best_mrr = -1
best_combo = None
rows = []

for alpha, beta in itertools.product(alphas, betas):
    m = evaluate_cached(cached, alpha, beta)
    rows.append((alpha, beta, m))
    if m['MRR'] > best_mrr:
        best_mrr = m['MRR']
        best_combo = (alpha, beta, m)

# ── Step 3: Print table ──
print(f"{'alpha_h':>8} {'beta_t':>8} {'P@10':>8} {'R@10':>8} {'MRR':>8}")
print("-" * 48)
for alpha, beta, m in rows:
    marker = " <-- BEST MRR" if (alpha, beta) == (best_combo[0], best_combo[1]) else ""
    print(f"{alpha:>8.1f} {beta:>8.1f} {m['P@10']:>8.3f} {m['Recall@10']:>8.3f} {m['MRR']:>8.3f}{marker}")

print()
print(f"Best: alpha_hybrid={best_combo[0]}, beta_temporal={best_combo[1]}")
print(f"  P@10={best_combo[2]['P@10']:.3f}  Recall@10={best_combo[2]['Recall@10']:.3f}  MRR={best_combo[2]['MRR']:.3f}")
