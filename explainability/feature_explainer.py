import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ranking.ranker import final_search, z_score_normalize
from ranking.normalization_stats import NORM_STATS

def explain(query: str, size: int = 5):
    """
    Returns results enriched with a human-readable feature contribution breakdown.
    No LLM — explanations are derived directly from ranking feature scores.
    """
    # Fetch more candidates for discrimination check (same as final_search internal fetch)
    results = final_search(query, size=size)
    
    # For discrimination check, we need the raw temporal scores from a larger pool
    # Re-fetch with larger size to get full candidate pool
    from retrieval.hybrid import search_hybrid
    from ranking.temporal_ranker import calculate_temporal_score
    from temporal.temporal_parser import TemporalParser
    
    parser = TemporalParser()
    temporal_intent = parser.parse(query)
    query_start = temporal_intent.get('start_year')
    query_end = temporal_intent.get('end_year')
    
    fetch_size = size * 5
    hybrid_res = search_hybrid(query, size=fetch_size, alpha=0.5)
    
    temporal_raw = []
    for r in hybrid_res:
        doc_start = r.get('historical_start')
        doc_end = r.get('historical_end')
        if query_start is None or query_end is None:
            t_score = 0.5
        else:
            t_score, _ = calculate_temporal_score(doc_start, doc_end, query_start, query_end)
        temporal_raw.append(t_score)
    
    # Z-score normalize for discrimination check
    if temporal_raw:
        temporal_z = [z_score_normalize(t, NORM_STATS['temporal_mean'], NORM_STATS['temporal_std']) for t in temporal_raw]
        temporal_range = max(temporal_z) - min(temporal_z)
    else:
        temporal_range = 0
    temporal_non_discriminating = temporal_range < 0.05
    
    # Metadata is always neutral in current corpus
    metadata_non_discriminating = True
    
    explanations = []
    
    for r in results:
        # Compute weighted contributions as percentages of final_score
        final_score = r.get('final_score', 0) or 1e-9
        
        hybrid_contrib = 0.7 * r.get('hybrid_score', 0)
        temp_contrib = 0.2 * r.get('temporal_score', 0)
        meta_contrib = 0.1 * r.get('metadata_score', 0)
        
        # Convert to percentages: relative weight of each discriminating feature
        # Only include discriminating features in the percentage calculation
        discriminating = []
        if not temporal_non_discriminating:
            discriminating.append(("temporal", abs(temp_contrib)))
        if not metadata_non_discriminating:
            discriminating.append(("metadata", abs(meta_contrib)))
        discriminating.append(("hybrid", abs(hybrid_contrib)))  # hybrid always discriminates
        
        total_mag = sum(v for _, v in discriminating)
        if total_mag > 1e-9:
            contrib = {name: round(100 * mag / total_mag, 1) for name, mag in discriminating}
        else:
            contrib = {name: 0.0 for name, _ in discriminating}
        
        if temporal_non_discriminating:
            contrib["temporal"] = "not distinguishing"
        if metadata_non_discriminating:
            contrib["metadata"] = "not distinguishing (neutral)"
        
        explanations.append({
            "chunk_id": r.get("chunk_id"),
            "parent_doc_id": r.get("parent_doc_id"),
            "snippet": (r.get("text") or "")[:120] + "...",
            "final_score": round(final_score, 6),
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
