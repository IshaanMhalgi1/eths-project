import os, sys, json
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ranking.shared_hybrid import get_hybrid_candidates, clear_hybrid_cache
from ranking.temporal_ranker import calculate_temporal_score
from temporal.temporal_parser import TemporalParser
from ranking.normalization_stats import NORM_STATS

parser = TemporalParser()

def z_score_normalize(val, mean, std):
    if std > 0:
        return (val - mean) / std
    return 0.0

QRELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')
with open(QRELS_PATH) as f:
    qrels = json.load(f)
qrels = [q for q in qrels if q.get('relevant_doc_ids')]

# Pre-fetch hybrid candidates
clear_hybrid_cache()
cached = []
for q in qrels:
    candidates = get_hybrid_candidates(q['query_text'])
    intent = parser.parse(q['query_text'])
    cached.append({
        'query': q['query_text'],
        'true_docs': q.get('relevant_doc_ids', []),
        'query_start': intent.get('start_year'),
        'query_end': intent.get('end_year'),
        'candidates': candidates,
    })

def evaluate_cached(cached_data, beta_temporal, k=10):
    p_at_k_list, recall_list, mrr_list = [], [], []
    HYBRID_ALPHA = 0.5
    hybrid_std = NORM_STATS['dense_std']

    for entry in cached_data:
        candidates = entry['candidates']
        true_docs = entry['true_docs']
        qs = entry['query_start']
        qe = entry['query_end']

        if qs is not None and qe is not None:
            for r in candidates:
                t, _ = calculate_temporal_score(r.get('historical_start'), r.get('historical_end'), qs, qe)
                r['_raw_t'] = t
        else:
            for r in candidates:
                r['_raw_t'] = 0.5

        for r in candidates:
            norm_temp = z_score_normalize(
                r['_raw_t'], 
                NORM_STATS['temporal_mean'], 
                NORM_STATS['temporal_std']
            )
            temp_adj = norm_temp * hybrid_std
            r['_final'] = HYBRID_ALPHA * r['score'] + beta_temporal * temp_adj

        scored = sorted(candidates, key=lambda r: r['_final'], reverse=True)

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
        'P@10': float(np.mean(p_at_k_list)),
        'Recall@10': float(np.mean(recall_list)),
        'MRR': float(np.mean(mrr_list)),
    }

# Extended grid search
betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print('Extended beta_temporal grid search:')
print('beta_t     P@10     R@10      MRR')
print('-' * 36)

best_mrr = -1
best_combo = None
for beta in betas:
    m = evaluate_cached(cached, beta)
    marker = ' <-- BEST' if m['MRR'] > best_mrr else ''
    if m['MRR'] > best_mrr:
        best_mrr = m['MRR']
        best_combo = (beta, m)
    print('{:>8.1f} {:>8.3f} {:>8.3f} {:>8.3f}{}'.format(beta, m['P@10'], m['Recall@10'], m['MRR'], marker))

print()
print('Best: beta_temporal={}'.format(best_combo[0]))
print('  P@10={:.3f}  Recall@10={:.3f}  MRR={:.3f}'.format(
    best_combo[1]['P@10'], best_combo[1]['Recall@10'], best_combo[1]['MRR']))