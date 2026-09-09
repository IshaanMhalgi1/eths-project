import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ranking.ranker import final_search

def explain(query: str, size: int = 5):
    """
    Returns results enriched with a human-readable feature contribution breakdown.
    No LLM — explanations are derived directly from ranking feature scores.
    """
    results = final_search(query, size=size)
    explanations = []
    
    # Detect low-discrimination temporal and metadata cases
    temporal_scores = [r.get('temporal_score', 0) for r in results if r.get('temporal_score') is not None]
    temporal_range = max(temporal_scores) - min(temporal_scores) if temporal_scores else 0
    temporal_non_discriminating = temporal_range < 0.05
    
    metadata_scores = [r.get('metadata_score', 0) for r in results if r.get('metadata_score') is not None]
    metadata_range = max(metadata_scores) - min(metadata_scores) if metadata_scores else 0
    metadata_non_discriminating = metadata_range < 0.05
    
    for r in results:
        contrib = {
            "bm25": round((r.get('bm25_score') or 0) * 100, 1),
            "dense": round((r.get('dense_score') or 0) * 100, 1),
        }
        
        if temporal_non_discriminating:
            contrib["temporal"] = "not distinguishing"
        else:
            contrib["temporal"] = round((r.get('temporal_score') or 0) * 100, 1)
        
        if metadata_non_discriminating:
            contrib["metadata"] = "not distinguishing (neutral)"
        else:
            contrib["metadata"] = round((r.get('metadata_score') or 0) * 100, 1)
        
        explanations.append({
            "chunk_id": r.get("chunk_id"),
            "parent_doc_id": r.get("parent_doc_id"),
            "snippet": (r.get("text") or "")[:120] + "...",
            "final_score": round(r.get("final_score") or 0, 4),
            "feature_contributions_%": contrib,
            "temporal_explanation": r.get("temporal_explanation"),
            "metadata_explanation": r.get("metadata_explanation"),
            "historical_start": r.get("historical_start"),
            "historical_end": r.get("historical_end"),
            "temporal_non_discriminating": temporal_non_discriminating,
            "metadata_non_discriminating": metadata_non_discriminating,
        })
    return explanations


if __name__ == '__main__':
    import json
    for e in explain("European news during the Jeffersonian Era"):
        print(json.dumps(e, indent=2))
