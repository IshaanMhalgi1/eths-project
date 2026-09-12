"""
Specific significance tests for the two headline claims:
1. BM25 vs Hybrid (alpha=0.5)
2. Temporal (beta=0.5) vs Hybrid (beta=0) -- i.e., temporal reranking benefit
"""
import os, sys, json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25
from retrieval.hybrid import search_hybrid
from ranking.shared_hybrid import get_hybrid_candidates, clear_hybrid_cache
from ranking.temporal_ranker import search_temporal
from ranking.normalization_stats import NORM_STATS

def z_score_normalize(val, mean, std):
    if std > 0:
        return (val - mean) / std
    return 0.0

QRELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')
with open(QRELS_PATH) as f:
    qrels = json.load(f)

# Only use queries with relevant docs
qrels = [q for q in qrels if q.get('relevant_doc_ids')]

def evaluate_ranker(candidates, true_docs, k=10):
    """Evaluate deduplicated candidates against true docs."""
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

# Pre-fetch candidates for each query
print("Pre-fetching candidates...")
bm25_cache = {}
hybrid_cache = {}
temporal_beta0_cache = {}  # beta=0 (pure hybrid base)
temporal_beta5_cache = {}  # beta=0.5 (temporal reranked)

from retrieval.dense import search_dense
from ranking.temporal_ranker import calculate_temporal_score
from temporal.temporal_parser import TemporalParser

parser = TemporalParser()

for q in qrels:
    query = q['query_text']
    
    # BM25
    bm25_cache[query] = search_bm25(query, size=50)
    
    # Hybrid (shared pool)
    clear_hybrid_cache()
    hybrid_cache[query] = get_hybrid_candidates(query)
    
    # Temporal beta=0 (just hybrid base)
    # Use the same candidates, just sort by hybrid score
    temporal_beta0_cache[query] = hybrid_cache[query]
    
    # Temporal beta=0.5 (with temporal reranking)
    intent = parser.parse(query)
    qs = intent.get('start_year')
    qe = intent.get('end_year')
    
    candidates = hybrid_cache[query]
    if qs is not None and qe is not None:
        for r in candidates:
            t, _ = calculate_temporal_score(r.get('historical_start'), r.get('historical_end'), qs, qe)
            r['_raw_t'] = t
    else:
        for r in candidates:
            r['_raw_t'] = 0.5
    
    # Apply temporal scoring with beta=0.5
    hybrid_std = NORM_STATS['dense_std']
    for r in candidates:
        norm_temp = z_score_normalize(
            r['_raw_t'], 
            NORM_STATS['temporal_mean'], 
            NORM_STATS['temporal_std']
        )
        temp_adj = norm_temp * hybrid_std
        # temporal ranker: alpha=0.7 * hybrid + beta * temp_adj
        r['_temporal_score'] = 0.7 * r['score'] + 0.5 * temp_adj
    
    scored = sorted(candidates, key=lambda r: r['_temporal_score'], reverse=True)
    temporal_beta5_cache[query] = scored

# Evaluate all methods per query
print("Evaluating...")
bm25_recalls, bm25_mrrs = [], []
hybrid_recalls, hybrid_mrrs = [], []
temp0_recalls, temp0_mrrs = [], []
temp5_recalls, temp5_mrrs = [], []

for q in qrels:
    query = q['query_text']
    true_docs = q['relevant_doc_ids']
    
    bm25_res = evaluate_ranker(bm25_cache[query], true_docs)
    hybrid_res = evaluate_ranker(hybrid_cache[query], true_docs)
    temp0_res = evaluate_ranker(temporal_beta0_cache[query], true_docs)
    temp5_res = evaluate_ranker(temporal_beta5_cache[query], true_docs)
    
    bm25_recalls.append(bm25_res['Recall@10'])
    bm25_mrrs.append(bm25_res['MRR'])
    hybrid_recalls.append(hybrid_res['Recall@10'])
    hybrid_mrrs.append(hybrid_res['MRR'])
    temp0_recalls.append(temp0_res['Recall@10'])
    temp0_mrrs.append(temp0_res['MRR'])
    temp5_recalls.append(temp5_res['Recall@10'])
    temp5_mrrs.append(temp5_res['MRR'])

print(f"\nPoint estimates (n={len(qrels)}):")
print(f"  BM25:           R@10={np.mean(bm25_recalls):.3f}, MRR={np.mean(bm25_mrrs):.3f}")
print(f"  Hybrid (alpha=0.5): R@10={np.mean(hybrid_recalls):.3f}, MRR={np.mean(hybrid_mrrs):.3f}")
print(f"  Temporal beta=0:   R@10={np.mean(temp0_recalls):.3f}, MRR={np.mean(temp0_mrrs):.3f}")
print(f"  Temporal beta=0.5: R@10={np.mean(temp5_recalls):.3f}, MRR={np.mean(temp5_mrrs):.3f}")

# Bootstrap significance
n_resamples = 10000
n = len(qrels)
indices = np.arange(n)

def bootstrap_ci(values_a, values_b, metric_name):
    """Bootstrap CI for difference in means (a - b)."""
    diffs = []
    for _ in range(n_resamples):
        sample_idx = np.random.choice(indices, size=n, replace=True)
        mean_a = np.mean([values_a[i] for i in sample_idx])
        mean_b = np.mean([values_b[i] for i in sample_idx])
        diffs.append(mean_a - mean_b)
    
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    point_diff = np.mean(values_a) - np.mean(values_b)
    significant = ci_low > 0 or ci_high < 0  # doesn't include 0
    return point_diff, ci_low, ci_high, significant

print("\n" + "="*70)
print("PAIRWISE SIGNIFICANCE TESTS (10000 bootstrap resamples, 95% CI)")
print("="*70)

# 1. BM25 vs Hybrid (alpha=0.5) - THE headline claim
print("\n1. BM25 vs Hybrid (alpha=0.5) [BM25 - Hybrid]:")
for metric, bm25_vals, hybrid_vals in [
    ("Recall@10", bm25_recalls, hybrid_recalls),
    ("MRR", bm25_mrrs, hybrid_mrrs)
]:
    diff, ci_low, ci_high, sig = bootstrap_ci(bm25_vals, hybrid_vals, metric)
    print(f"  {metric}: diff={diff:+.3f}, 95% CI=[{ci_low:+.3f}, {ci_high:+.3f}] {'**SIGNIFICANT**' if sig else 'not significant'}")

# 2. Temporal beta=0.5 vs beta=0 (temporal reranking benefit) - THE other headline claim
print("\n2. Temporal beta=0.5 vs beta=0 [beta=0.5 - beta=0]:")
for metric, temp5_vals, temp0_vals in [
    ("Recall@10", temp5_recalls, temp0_recalls),
    ("MRR", temp5_mrrs, temp0_mrrs)
]:
    diff, ci_low, ci_high, sig = bootstrap_ci(temp5_vals, temp0_vals, metric)
    print(f"  {metric}: diff={diff:+.3f}, 95% CI=[{ci_low:+.3f}, {ci_high:+.3f}] {'**SIGNIFICANT**' if sig else 'not significant'}")

# 3. Also BM25 vs Temporal beta=0.5 for completeness
print("\n3. BM25 vs Temporal beta=0.5 [BM25 - beta=0.5]:")
for metric, bm25_vals, temp5_vals in [
    ("Recall@10", bm25_recalls, temp5_recalls),
    ("MRR", bm25_mrrs, temp5_mrrs)
]:
    diff, ci_low, ci_high, sig = bootstrap_ci(bm25_vals, temp5_vals, metric)
    print(f"  {metric}: diff={diff:+.3f}, 95% CI=[{ci_low:+.3f}, {ci_high:+.3f}] {'**SIGNIFICANT**' if sig else 'not significant'}")

# 4. Temporal beta=0.5 vs Hybrid (alpha=0.5)
print("\n4. Temporal beta=0.5 vs Hybrid (alpha=0.5) [beta=0.5 - Hybrid]:")
for metric, temp5_vals, hybrid_vals in [
    ("Recall@10", temp5_recalls, hybrid_recalls),
    ("MRR", temp5_mrrs, hybrid_mrrs)
]:
    diff, ci_low, ci_high, sig = bootstrap_ci(temp5_vals, hybrid_vals, metric)
    print(f"  {metric}: diff={diff:+.3f}, 95% CI=[{ci_low:+.3f}, {ci_high:+.3f}] {'**SIGNIFICANT**' if sig else 'not significant'}")

# Per-query details for debugging
print("\n" + "="*70)
print("PER-QUERY DETAILS (Recall@10)")
print("="*70)
for i, q in enumerate(qrels):
    print(f"  {q['query_id']}: BM25={bm25_recalls[i]:.2f}, Hybrid={hybrid_recalls[i]:.2f}, beta=0={temp0_recalls[i]:.2f}, beta=0.5={temp5_recalls[i]:.2f}  [{q['query_text'][:50]}]")