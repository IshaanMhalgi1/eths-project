import os
import sys
import json

# Ensure module imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from temporal.temporal_parser import TemporalParser
from retrieval.hybrid import search_hybrid

parser = TemporalParser()

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

def search_temporal(query: str, size: int = 10, alpha_hybrid: float = 0.5, beta_temporal: float = 0.3):
    """
    Retrieves using hybrid search and boosts based on temporal constraints.
    """
    # Parse temporal intent
    temporal_intent = parser.parse(query)
    query_start = temporal_intent.get('start_year')
    query_end = temporal_intent.get('end_year')
    
    # Fetch more candidates from hybrid because temporal might re-rank heavily
    fetch_size = size * 5
    hybrid_res = search_hybrid(query, size=fetch_size, alpha=0.5)
    
    # If no temporal constraint, temporal score is neutral (1.0), and we just use hybrid scores
    if query_start is None or query_end is None:
        for r in hybrid_res:
            r['temporal_score'] = 1.0
            r['temporal_explanation'] = "no_constraint"
            r['final_score'] = r['score'] # Keep hybrid score
        return sorted(hybrid_res, key=lambda x: x['final_score'], reverse=True)[:size]
        
    # Apply temporal scoring
    for r in hybrid_res:
        doc_start = r.get('historical_start')
        doc_end = r.get('historical_end')
        
        t_score, explanation = calculate_temporal_score(doc_start, doc_end, query_start, query_end)
        
        r['raw_temporal_score'] = t_score
        r['temporal_explanation'] = explanation
        
    # Min-max normalize hybrid base (RRF) scores
    hybrid_scores = [r['score'] for r in hybrid_res]
    min_h = min(hybrid_scores) if hybrid_scores else 0
    max_h = max(hybrid_scores) if hybrid_scores else 1
    
    # Min-max normalize temporal scores
    temp_scores = [r['raw_temporal_score'] for r in hybrid_res]
    min_t = min(temp_scores) if temp_scores else 0
    max_t = max(temp_scores) if temp_scores else 1
    
    for r in hybrid_res:
        norm_h = (r['score'] - min_h) / (max_h - min_h) if max_h > min_h else 1.0
        norm_t = (r['raw_temporal_score'] - min_t) / (max_t - min_t) if max_t > min_t else 1.0
        
        r['temporal_score'] = norm_t
        r['final_score'] = alpha_hybrid * norm_h + beta_temporal * norm_t
        
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
        print(f"[{r['final_score']:.3f}] (Temp: {r['temporal_explanation']}) {r['parent_doc_id']}: {r['text'][:50]}...")
