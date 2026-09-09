import os
import sys

# Ensure module imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense

def search_hybrid(query: str, size: int = 10, alpha: float = 0.5):
    """
    Combines BM25 and Dense scores with a tunable alpha.
    Final Score = alpha * BM25_norm + (1 - alpha) * Dense_norm
    """
    # Fetch more candidates to ensure good overlap
    fetch_size = size * 2
    bm25_res = search_bm25(query, size=fetch_size)
    dense_res = search_dense(query, size=fetch_size)
    
    # Assign ranks instead of min-max scaling
    bm25_ranks = {r['chunk_id']: rank + 1 for rank, r in enumerate(bm25_res)}
    dense_ranks = {r['chunk_id']: rank + 1 for rank, r in enumerate(dense_res)}
    
    k = 60 # RRF constant
                
    # Combine scores using RRF
    combined = {}
    for r in bm25_res:
        rank_bm25 = bm25_ranks[r['chunk_id']]
        rrf_bm25 = 1.0 / (k + rank_bm25)
        combined[r['chunk_id']] = {
            "chunk_id": r['chunk_id'],
            "parent_doc_id": r['parent_doc_id'],
            "text": r['text'],
            "historical_start": r.get('historical_start'),
            "historical_end": r.get('historical_end'),
            "historical_period": r.get('historical_period'),
            "location": r.get('location'),
            "bm25_score": rrf_bm25,
            "dense_score": 0.0,
            "score": alpha * rrf_bm25
        }
    
    for r in dense_res:
        rank_dense = dense_ranks[r['chunk_id']]
        rrf_dense = 1.0 / (k + rank_dense)
        
        if r['chunk_id'] in combined:
            combined[r['chunk_id']]['dense_score'] = rrf_dense
            combined[r['chunk_id']]['score'] += (1 - alpha) * rrf_dense
        else:
            combined[r['chunk_id']] = {
                "chunk_id": r['chunk_id'],
                "parent_doc_id": r['parent_doc_id'],
                "text": r['text'],
                "historical_start": r.get('historical_start'),
                "historical_end": r.get('historical_end'),
                "historical_period": r.get('historical_period'),
                "location": r.get('location'),
                "bm25_score": 0.0,
                "dense_score": rrf_dense,
                "score": (1 - alpha) * rrf_dense
            }
            
    # Sort and return top K
    sorted_res = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
    
    # Explicit deduplication by chunk_id (safety net)
    seen = set()
    deduped = []
    for r in sorted_res:
        cid = r['chunk_id']
        if cid not in seen:
            seen.add(cid)
            deduped.append(r)
    return deduped[:size]

if __name__ == '__main__':
    # Test
    res = search_hybrid("what happened in washington")
    for r in res[:2]:
        print(f"[{r['score']}] {r['parent_doc_id']}: {r['text'][:50]}...")
