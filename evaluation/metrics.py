import os
import json
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense
from ranking.shared_hybrid import get_hybrid_candidates, clear_hybrid_cache

def compute_metrics(qrels_path, retrieval_fn, k=10):
    with open(qrels_path, 'r') as f:
        qrels = json.load(f)

    p_at_k_list = []
    recall_at_k_list = []
    mrr_list = []
    
    for q in qrels:
        query_text = q['query_text']
        true_docs = q.get('relevant_doc_ids', [])
        if 'relevant_doc_id' in q:
            true_docs.append(q['relevant_doc_id'])
            
        # Retrieve - use shared hybrid pool for consistency
        if retrieval_fn.__name__ == 'search_hybrid':
            clear_hybrid_cache()
            results = get_hybrid_candidates(query_text)
        else:
            results = retrieval_fn(query_text, size=k * 5)
        retrieved_docs = [res['parent_doc_id'] for res in results]
        
        # We consider it a hit if ANY chunk from the parent_doc_id is in the top K.
        # Deduplicate retrieved parent_doc_ids preserving order.
        seen = set()
        dedup_docs = []
        for d in retrieved_docs:
            if d not in seen:
                seen.add(d)
                dedup_docs.append(d)
                
        dedup_docs = dedup_docs[:k]
        
        # Metrics
        hits = [1 if d in true_docs else 0 for d in dedup_docs]
        
        p_at_k = sum(hits) / k if k > 0 else 0
        recall_at_k = sum(hits) / max(len(true_docs), 1) if true_docs else 0
        
        mrr = 0
        for rank, d in enumerate(dedup_docs):
            if d in true_docs:
                mrr = 1.0 / (rank + 1)
                break
                
        p_at_k_list.append(p_at_k)
        recall_at_k_list.append(recall_at_k)
        mrr_list.append(mrr)
        
    return {
        "P@10": np.mean(p_at_k_list),
        "Recall@10": np.mean(recall_at_k_list),
        "MRR": np.mean(mrr_list)
    }

if __name__ == '__main__':
    qrels_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')
    print("Evaluating BM25 Baseline...")
    bm25_metrics = compute_metrics(qrels_path, search_bm25, k=10)
    print(json.dumps(bm25_metrics, indent=2))
    
    print("\nEvaluating Dense Baseline...")
    dense_metrics = compute_metrics(qrels_path, search_dense, k=10)
    print(json.dumps(dense_metrics, indent=2))
    
    from retrieval.hybrid import search_hybrid
    print("\nEvaluating Hybrid Baseline (alpha=0.5)...")
    hybrid_metrics = compute_metrics(qrels_path, search_hybrid, k=10)
    print(json.dumps(hybrid_metrics, indent=2))
    
    from ranking.temporal_ranker import search_temporal
    print("\nEvaluating Temporal Ranker (alpha=0.7, beta=0.3)...")
    temporal_metrics = compute_metrics(qrels_path, search_temporal, k=10)
    print(json.dumps(temporal_metrics, indent=2))
    
    from ranking.ranker import final_search
    print("\nEvaluating Final Ranker (w_bm25=0.3, w_dense=0.3, w_temp=0.3, w_meta=0.1)...")
    final_metrics = compute_metrics(qrels_path, final_search, k=10)
    print(json.dumps(final_metrics, indent=2))
