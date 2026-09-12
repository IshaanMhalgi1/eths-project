import os
import sys
import json

# Ensure module imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from temporal.temporal_parser import TemporalParser
from ranking.shared_hybrid import get_hybrid_candidates

parser = TemporalParser()

# Load global normalization stats
with open(os.path.join(os.path.dirname(__file__), 'normalization_stats.json'), 'r') as f:
    NORM_STATS = json.load(f)

def z_score_normalize(val, mean, std):
    """Global z-score normalization."""
    if std > 0:
        return (val - mean) / std
    return 0.0

def calculate_temporal_score(doc_start: int, doc_end: int, query_start: int, query_end: int):
    # Missing document dates -> no penalty, treated as neutral
    if doc_start is None or doc_end is None:
        return 0.5, "neutral"
        
    # Full exact overlap
    if doc_start >= query_start and doc_end <= query_end:
        return 1.0, "exact_overlap"
    
    # Contains
    if doc_start <= query_start and doc_end >= query_end:
        return 0.8, "contains"
        
    # Partial overlap
    if doc_start <= query_end and doc_end >= query_start:
        return 0.6, "partial_overlap"
        
    # Adjacent (within 10 years buffer)
    if abs(doc_start - query_end) <= 10 or abs(query_start - doc_end) <= 10:
        return 0.2, "adjacent"
        
    # No overlap
    return 0.0, "no_overlap"

def search_temporal(query: str, size: int = 10, alpha_hybrid: float = 0.7, beta_temporal: float = 0.3):
    """
    Retrieves using hybrid search and boosts based on temporal constraints.
    Uses shared hybrid candidate pool for consistent results across rankers.
    """
    # Parse temporal intent
    temporal_intent = parser.parse(query)
    query_start = temporal_intent.get('start_year')
    query_end = temporal_intent.get('end_year')
    
    # Get shared hybrid candidates
    hybrid_res = get_hybrid_candidates(query)
    
    # If no temporal constraint, just use hybrid scores
    if query_start is None or query_end is None:
        for r in hybrid_res:
            r['temporal_score'] = 0.0  # zero adjustment
            r['temporal_explanation'] = "no_constraint"
            r['final_score'] = r['score']  # Pure hybrid
        return sorted(hybrid_res, key=lambda x: x['final_score'], reverse=True)[:size]
        
    # Apply temporal scoring
    for r in hybrid_res:
        doc_start = r.get('historical_start')
        doc_end = r.get('historical_end')
        
        t_score, explanation = calculate_temporal_score(doc_start, doc_end, query_start, query_end)
        
        r['raw_temporal_score'] = t_score
        r['temporal_explanation'] = explanation
    
    hybrid_std = NORM_STATS['dense_std']
    
    # Global z-score normalization for temporal, scaled to hybrid RRF scale
    for r in hybrid_res:
        norm_temp = z_score_normalize(
            r['raw_temporal_score'], 
            NORM_STATS['temporal_mean'], 
            NORM_STATS['temporal_std']
        )
        
        temp_adj = norm_temp * hybrid_std
        
        r['temporal_score'] = temp_adj
        r['hybrid_score'] = r['score']
        
        r['final_score'] = alpha_hybrid * r['score'] + beta_temporal * temp_adj
        
    sorted_res = sorted(hybrid_res, key=lambda x: x['final_score'], reverse=True)
    
    # Deduplicate by parent_doc_id for user-facing output
    seen_parent = set()
    deduped = []
    for r in sorted_res:
        pid = r['parent_doc_id']
        if pid not in seen_parent:
            seen_parent.add(pid)
            deduped.append(r)
    return deduped[:size]

if __name__ == '__main__':
    # Test
    res = search_temporal("real estate sales in 1805")
    for r in res[:2]:
        print(f"[{r['final_score']:.6f}] (Temp: {r['temporal_explanation']}) {r['parent_doc_id']}: {r['text'][:50]}...")
