"""
Bootstrap significance testing for retrieval metrics.
Uses shared hybrid pool for consistent candidate pool.
"""
import json
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense
from ranking.shared_hybrid import get_hybrid_candidates, clear_hybrid_cache
from ranking.temporal_ranker import search_temporal
from ranking.ranker import final_search

qrels_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')

with open(qrels_path, 'r') as f:
    qrels = json.load(f)

methods = {
    'BM25': search_bm25,
    'Dense': search_dense,
    'Hybrid': lambda q, size=10: get_hybrid_candidates(q)[:size],
    'Temporal': lambda q, size=10: search_temporal(q, size=size, alpha_hybrid=0.7, beta_temporal=0.3),
    'Final': lambda q, size=10: final_search(q, size=size),
}

def compute_query_metrics(query_text, true_docs, retrieval_fn, k=10):
    results = retrieval_fn(query_text, size=k * 5)
    retrieved_docs = [res['parent_doc_id'] for res in results]
    
    seen = set()
    dedup_docs = []
    for d in retrieved_docs:
        if d not in seen:
            seen.add(d)
            dedup_docs.append(d)
            
    dedup_docs = dedup_docs[:k]
    
    hits = [1 if d in true_docs else 0 for d in dedup_docs]
    
    p_at_k = sum(hits) / k if k > 0 else 0
    recall_at_k = sum(hits) / max(len(true_docs), 1) if true_docs else 0
    
    mrr = 0
    for rank, d in enumerate(dedup_docs):
        if d in true_docs:
            mrr = 1.0 / (rank + 1)
            break
    
    return {'p_at_k': p_at_k, 'recall_at_k': recall_at_k, 'mrr': mrr}

# Pre-compute per-query metrics for all methods
print("Pre-computing per-query metrics...")
query_metrics = {}
for name, fn in methods.items():
    query_metrics[name] = []
    for q in qrels:
        query_text = q['query_text']
        true_docs = q.get('relevant_doc_ids', [])
        if 'relevant_doc_id' in q:
            true_docs.append(q['relevant_doc_id'])
        metrics = compute_query_metrics(query_text, true_docs, fn, k=10)
        query_metrics[name].append(metrics)

# Convert to numpy arrays
for name in methods:
    query_metrics[name] = {
        'recall': np.array([m['recall_at_k'] for m in query_metrics[name]]),
        'mrr': np.array([m['mrr'] for m in query_metrics[name]])
    }

print("Pre-computation complete.\n")

def bootstrap_ci(recalls, mrrs, n_bootstrap=10000, ci=95):
    """Compute bootstrap confidence intervals from pre-computed query metrics."""
    n = len(recalls)
    boot_recalls = []
    boot_mrrs = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_recalls.append(np.mean(recalls[idx]))
        boot_mrrs.append(np.mean(mrrs[idx]))
    
    boot_recalls = np.array(boot_recalls)
    boot_mrrs = np.array(boot_mrrs)
    
    alpha = (100 - ci) / 2
    recall_ci = np.percentile(boot_recalls, [alpha, 100 - alpha])
    mrr_ci = np.percentile(boot_mrrs, [alpha, 100 - alpha])
    
    return {
        'recall_mean': np.mean(boot_recalls),
        'recall_ci': recall_ci,
        'mrr_mean': np.mean(boot_mrrs),
        'mrr_ci': mrr_ci
    }

print("=" * 80)
print("BOOTSTRAP SIGNIFICANCE TESTING (10000 resamples, 95% CI)")
print("=" * 80)

results = {}
for name in methods:
    print(f"\n{name}:")
    ci = bootstrap_ci(query_metrics[name]['recall'], query_metrics[name]['mrr'], n_bootstrap=10000)
    results[name] = ci
    print(f"  Recall@10: {ci['recall_mean']:.3f} [{ci['recall_ci'][0]:.3f}, {ci['recall_ci'][1]:.3f}]")
    print(f"  MRR:       {ci['mrr_mean']:.3f} [{ci['mrr_ci'][0]:.3f}, {ci['mrr_ci'][1]:.3f}]")

# Pairwise comparison
print("\n" + "=" * 80)
print("PAIRWISE COMPARISONS (Recall@10)")
print("=" * 80)

method_names = list(methods.keys())
for i in range(len(method_names)):
    for j in range(i+1, len(method_names)):
        m1, m2 = method_names[i], method_names[j]
        diff = results[m1]['recall_mean'] - results[m2]['recall_mean']
        
        ci1 = results[m1]['recall_ci']
        ci2 = results[m2]['recall_ci']
        overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
        
        significance = "NOT significant" if overlap else "SIGNIFICANT"
        print(f"{m1} vs {m2}: diff={diff:+.3f} [{significance}]")
        print(f"  {m1}: [{ci1[0]:.3f}, {ci1[1]:.3f}]")
        print(f"  {m2}: [{ci2[0]:.3f}, {ci2[1]:.3f}]")