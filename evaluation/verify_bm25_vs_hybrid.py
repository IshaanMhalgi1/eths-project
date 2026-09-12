"""
Verify BM25 vs Hybrid evaluation consistency at fetch_size=50.
Check candidate pools, RRF parameters, and score comparisons.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25
from retrieval.hybrid import search_hybrid
from ranking.shared_hybrid import get_hybrid_candidates, clear_hybrid_cache

with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')) as f:
    import json
    qrels = json.load(f)

query = qrels[8]['query_text']  # "political news congress 1800 to 1805"

print(f"Query: '{query}'\n")

# 1. Check BM25 candidate pool at fetch_size=50
clear_hybrid_cache()
bm25_res = search_bm25(query, size=50)
print(f"BM25 (size=50): {len(bm25_res)} candidates")
print(f"  Top-10: {[(r['parent_doc_id'], round(r['score'], 4)) for r in bm25_res[:10]]}")

# 2. Check Hybrid candidate pool at fetch_size=50
clear_hybrid_cache()
hybrid_res = search_hybrid(query, size=50, alpha=0.5)
print(f"\nHybrid (size=50, alpha=0.5): {len(hybrid_res)} candidates")
print(f"  Top-10: {[(r['parent_doc_id'], round(r['score'], 6), round(r['bm25_score'], 6), round(r['dense_score'], 6)) for r in hybrid_res[:10]]}")

# 3. Check shared_hybrid pool
clear_hybrid_cache()
shared_res = get_hybrid_candidates(query)
print(f"\nShared pool (fetch_size=50): {len(shared_res)} candidates")
print(f"  Top-10: {[(r['parent_doc_id'], round(r['score'], 6), round(r['bm25_score'], 6), round(r['dense_score'], 6)) for r in shared_res[:10]]}")

# 4. Compare BM25 scores for same docs across pools
print("\n" + "=" * 60)
print("BM25 SCORE COMPARISON FOR SHARED DOCS")
print("=" * 60)

# Get all parent_doc_ids from BM25
bm25_docs = {r['parent_doc_id']: r['score'] for r in bm25_res}
shared_docs = {r['parent_doc_id']: r['bm25_score'] for r in shared_res}

common = set(bm25_docs.keys()) & set(shared_docs.keys())
print(f"Common docs: {len(common)}")
for pid in sorted(common)[:10]:
    bm25_s = bm25_docs[pid]
    shared_s = shared_docs[pid]
    print(f"  {pid}: BM25={bm25_s:.6f}, Shared BM25={shared_s:.6f}, diff={abs(bm25_s-shared_s):.6f}")

# 5. Full eval: BM25 vs Hybrid MRR at fetch_size=50 for all queries
print("\n" + "=" * 60)
print("FULL EVAL: BM25 vs HYBRID AT FETCH_SIZE=50 (ALL QUERIES)")
print("=" * 60)

total_bm25_mrr = 0
total_hybrid_mrr = 0
total_bm25_r10 = 0
total_hybrid_r10 = 0
n = 0

for q in qrels:
    query_text = q['query_text']
    true_docs = q.get('relevant_doc_ids', [])
    if 'relevant_doc_id' in q:
        true_docs.append(q['relevant_doc_id'])
    
    clear_hybrid_cache()
    bm25_cand = search_bm25(query_text, size=50)
    clear_hybrid_cache()
    hybrid_cand = search_hybrid(query_text, size=50, alpha=0.5)
    
    # BM25
    seen = set()
    bm25_dedup = []
    for r in bm25_cand:
        pid = r['parent_doc_id']
        if pid not in seen:
            seen.add(pid)
            bm25_dedup.append(pid)
        if len(bm25_dedup) == 10:
            break
    
    # Hybrid
    seen = set()
    hybrid_dedup = []
    for r in hybrid_cand:
        pid = r['parent_doc_id']
        if pid not in seen:
            seen.add(pid)
            hybrid_dedup.append(pid)
        if len(hybrid_dedup) == 10:
            break
    
    bm25_hits = [1 if d in true_docs else 0 for d in bm25_dedup]
    hybrid_hits = [1 if d in true_docs else 0 for d in hybrid_dedup]
    
    # MRR
    bm25_mrr = 0
    for rank, d in enumerate(bm25_dedup):
        if d in true_docs:
            bm25_mrr = 1.0 / (rank + 1)
            break
    
    hybrid_mrr = 0
    for rank, d in enumerate(hybrid_dedup):
        if d in true_docs:
            hybrid_mrr = 1.0 / (rank + 1)
            break
    
    total_bm25_mrr += bm25_mrr
    total_hybrid_mrr += hybrid_mrr
    
    bm25_r10 = sum(bm25_hits) / max(len(true_docs), 1)
    hybrid_r10 = sum(hybrid_hits) / max(len(true_docs), 1)
    
    total_bm25_r10 += bm25_r10
    total_hybrid_r10 += hybrid_r10
    n += 1

print(f"\nAcross {n} queries at fetch_size=50:")
print(f"  BM25:       MRR={total_bm25_mrr/n:.4f}, R@10={total_bm25_r10/n:.4f}")
print(f"  Hybrid:     MRR={total_hybrid_mrr/n:.4f}, R@10={total_hybrid_r10/n:.4f}")
print(f"  Difference: MRR={total_hybrid_mrr/n - total_bm25_mrr/n:+.4f}, R@10={total_hybrid_r10/n - total_bm25_r10/n:+.4f}")

# 6. Check RRF parameters
print("\n" + "=" * 60)
print("RRF PARAMETER CHECK")
print("=" * 60)
# k=60, alpha=0.5 means equal weight
# At fetch_size=50, max rank = 50
# RRF scores: 1/(60+1) = 0.0164 down to 1/(60+50) = 0.0091
# Range = 0.0073
print("RRF k=60, alpha=0.5")
print(f"  Max score (rank 1): 1/(60+1) = 0.01639")
print(f"  Min score (rank 50): 1/(60+50) = 0.00909")
print(f"  Range: 0.00730")
print(f"  Alpha=0.5 means equal weight to BM25 and Dense ranks")
print(f"  At this fetch size, RRF scores are tightly clustered")