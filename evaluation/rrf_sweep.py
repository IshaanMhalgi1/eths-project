"""
RRF parameter sweep over k and alpha at fixed fetch_size.
Tests whether BM25-alone still beats Hybrid across parameter range.
"""
import os, sys, json, itertools
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25
from retrieval.hybrid import search_hybrid
from retrieval.dense import search_dense
from ranking.shared_hybrid import get_hybrid_candidates, clear_hybrid_cache
from temporal.temporal_parser import TemporalParser

parser = TemporalParser()

QRELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')
with open(QRELS_PATH) as f:
    qrels = json.load(f)

# Only use queries with relevant docs
qrels = [q for q in qrels if q.get('relevant_doc_ids')]

def evaluate_method(candidates, true_docs, k=10):
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

# Sweep parameters
k_values = [10, 20, 40, 60, 80, 100]
alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# alpha=0.0 means pure Dense, alpha=1.0 means pure BM25

FETCH_SIZE = 50

print(f"RRF Parameter Sweep (fetch_size={FETCH_SIZE})")
print(f"Queries: {len(qrels)}")
print(f"k values: {k_values}")
print(f"alpha values: {alpha_values}")
print()

# Pre-fetch BM25 and Dense candidates for each query
print("Pre-fetching BM25 and Dense candidates...")
bm25_cache = {}
dense_cache = {}
for q in qrels:
    query = q['query_text']
    bm25_cache[query] = search_bm25(query, size=FETCH_SIZE)
    dense_cache[query] = search_dense(query, size=FETCH_SIZE)

# Also get pure BM25 baseline
print("Evaluating pure BM25 baseline...")
bm25_results = []
for q in qrels:
    res = evaluate_method(bm25_cache[q['query_text']], q['relevant_doc_ids'])
    bm25_results.append(res)
bm25_avg = {k: float(np.mean([r[k] for r in bm25_results])) for k in ["P@10", "Recall@10", "MRR"]}
print(f"  BM25: P@10={bm25_avg['P@10']:.3f}, R@10={bm25_avg['Recall@10']:.3f}, MRR={bm25_avg['MRR']:.3f}")

# Sweep
results = {}
best_mrr = -1
best_params = None

for k in k_values:
    for alpha in alpha_values:
        all_results = []
        for q in qrels:
            bm25_res = bm25_cache[q['query_text']]
            dense_res = dense_cache[q['query_text']]
            
            # RRF fusion
            # Rank documents in each list
            bm25_ranks = {r['parent_doc_id']: i+1 for i, r in enumerate(bm25_res)}
            dense_ranks = {r['parent_doc_id']: i+1 for i, r in enumerate(dense_res)}
            
            # Get all unique parent_doc_ids
            all_pids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
            
            # Compute RRF scores
            rrf_scores = {}
            for pid in all_pids:
                bm25_rank = bm25_ranks.get(pid, FETCH_SIZE + 1)
                dense_rank = dense_ranks.get(pid, FETCH_SIZE + 1)
                
                bm25_score = alpha / (k + bm25_rank)
                dense_score = (1 - alpha) / (k + dense_rank)
                rrf_scores[pid] = bm25_score + dense_score
            
            # Sort by RRF score
            sorted_pids = sorted(rrf_scores.keys(), key=lambda p: rrf_scores[p], reverse=True)
            
            # Convert to candidate format for evaluation
            candidates = [{'parent_doc_id': pid, 'score': rrf_scores[pid]} for pid in sorted_pids]
            
            res = evaluate_method(candidates, q['relevant_doc_ids'])
            all_results.append(res)
        
        avg = {k: float(np.mean([r[k] for r in all_results])) for k in ["P@10", "Recall@10", "MRR"]}
        results[(k, alpha)] = avg
        
        marker = " <-- BEST" if avg['MRR'] > best_mrr else ""
        if avg['MRR'] > best_mrr:
            best_mrr = avg['MRR']
            best_params = (k, alpha)
        
        print(f"k={k:3d} alpha={alpha:.1f}  P@10={avg['P@10']:.3f}  R@10={avg['Recall@10']:.3f}  MRR={avg['MRR']:.3f}{marker}")

print()
print(f"Best: k={best_params[0]}, alpha={best_params[1]}")
print(f"  P@10={results[best_params]['P@10']:.3f}, R@10={results[best_params]['Recall@10']:.3f}, MRR={results[best_params]['MRR']:.3f}")
print()
print(f"Pure BM25 baseline: MRR={bm25_avg['MRR']:.3f}")
print(f"Best Hybrid vs BM25: MRR diff = {best_mrr - bm25_avg['MRR']:+.3f}")

# Also show alpha=0 (pure Dense) and alpha=1 (pure BM25) for each k
print("\n--- Pure BM25 (alpha=1.0) vs Pure Dense (alpha=0.0) at each k ---")
for k in k_values:
    bm25_only = results[(k, 1.0)]
    dense_only = results[(k, 0.0)]
    print(f"k={k:3d}  BM25: MRR={bm25_only['MRR']:.3f}  Dense: MRR={dense_only['MRR']:.3f}")

# Save results
output = {
    "fetch_size": FETCH_SIZE,
    "k_values": k_values,
    "alpha_values": alpha_values,
    "best": {"k": best_params[0], "alpha": best_params[1], "metrics": results[best_params]},
    "bm25_baseline": bm25_avg,
    "all_results": {f"k={k}_alpha={alpha}": v for (k, alpha), v in results.items()}
}

out_path = os.path.join(os.path.dirname(__file__), 'rrf_sweep_results.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")