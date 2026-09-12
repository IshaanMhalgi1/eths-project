import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from temporal.temporal_parser import TemporalParser
from ranking.shared_hybrid import get_hybrid_candidates
from ranking.temporal_ranker import calculate_temporal_score

parser = TemporalParser()

# Load global normalization stats
with open(os.path.join(os.path.dirname(__file__), 'normalization_stats.json'), 'r') as f:
    NORM_STATS = json.load(f)

def z_score_normalize(val, mean, std):
    """Global z-score normalization."""
    if std > 0:
        return (val - mean) / std
    return 0.0

def extract_metadata_intent(query: str):
    # Very simple MVP metadata intent extractor.
    temporal_intent = parser.parse(query)
    return {
        "periods": temporal_intent.get('periods', []),
        "locations": [] # Skip location parsing for MVP since subset lacks locations
    }

def calculate_metadata_score(doc_period: str, doc_location: str, query_periods: list, query_locations: list):
    score = 0.0
    explanation_parts = []
    
    if query_periods and doc_period:
        if any(p.lower() in doc_period.lower() for p in query_periods):
            score += 0.5
            explanation_parts.append("period_match")
            
    if query_locations and doc_location:
        if any(l.lower() in doc_location.lower() for l in query_locations):
            score += 0.5
            explanation_parts.append("location_match")
            
    if not query_periods and not query_locations:
        return 0.5, "neutral"
        
    return score, (",".join(explanation_parts) if explanation_parts else "no_metadata_match")

def final_search(query: str, size: int = 10, w_hybrid=0.7, w_temp=0.2, w_meta=0.1):
    """
    Final ranker with global z-score normalization (respecting RRF precedent).
    
    Architecture:
    - Hybrid RRF score is the base (already calibrated rank-fusion of BM25+Dense).
    - Temporal and Metadata are global z-score normalized and added as adjustments.
    - The hybrid RRF score is kept in its natural scale [0, ~0.016].
    - Temporal z-scores are scaled by hybrid_std to be comparable adjustments.
    - Weights: hybrid=0.7, temporal=0.2, metadata=0.1 (sum=1.0)
    """
    temporal_intent = parser.parse(query)
    query_start = temporal_intent.get('start_year')
    query_end = temporal_intent.get('end_year')
    meta_intent = extract_metadata_intent(query)
    
    # Get shared hybrid candidates
    hybrid_res = get_hybrid_candidates(query)
    
    # Calculate raw temporal and metadata scores
    for r in hybrid_res:
        doc_start = r.get('historical_start')
        doc_end = r.get('historical_end')
        
        # Temporal
        if query_start is None or query_end is None:
            t_score, t_exp = 0.5, "neutral"
        else:
            t_score, t_exp = calculate_temporal_score(doc_start, doc_end, query_start, query_end)
            
        r['raw_temporal_score'] = t_score
        r['temporal_explanation'] = t_exp
        
        # Metadata
        doc_period = r.get('historical_period')
        doc_location = r.get('location')
        
        m_score, m_exp = calculate_metadata_score(doc_period, doc_location, meta_intent['periods'], meta_intent['locations'])
        r['raw_metadata_score'] = m_score
        r['metadata_explanation'] = m_exp

    # Hybrid RRF score is the base (already calibrated, range [0, ~0.0164])
    hybrid_std = NORM_STATS['dense_std']  # ~0.0051
    
    for r in hybrid_res:
        # Normalize temporal globally, then scale to hybrid scale
        norm_temp = z_score_normalize(
            r['raw_temporal_score'], 
            NORM_STATS['temporal_mean'], 
            NORM_STATS['temporal_std']
        )
        
        # Normalize metadata globally, then scale to hybrid scale  
        norm_meta = z_score_normalize(
            r['raw_metadata_score'], 
            NORM_STATS['metadata_mean'], 
            NORM_STATS['metadata_std']
        )
        
        # Scale z-scores to hybrid RRF scale for comparable adjustments
        temp_adj = norm_temp * hybrid_std
        meta_adj = norm_meta * hybrid_std
        
        r['temporal_score'] = temp_adj
        r['metadata_score'] = meta_adj
        
        # Keep hybrid in its natural scale
        r['hybrid_score'] = r['score']
        r['bm25_score'] = r.get('bm25_score', 0)
        r['dense_score'] = r.get('dense_score', 0)
        
        # Final blend: hybrid base (70%) + temporal adj (20%) + metadata adj (10%)
        r['final_score'] = (
            w_hybrid * r['score'] +
            w_temp * temp_adj +
            w_meta * meta_adj
        )
    
    # Explicit deduplication by parent_doc_id for user-facing output
    seen_parent = set()
    deduped = []
    for r in sorted(hybrid_res, key=lambda x: x['final_score'], reverse=True):
        pid = r['parent_doc_id']
        if pid not in seen_parent:
            seen_parent.add(pid)
            deduped.append(r)
    return deduped[:size]


if __name__ == '__main__':
    res = final_search("real estate sales in 1805")
    for r in res[:2]:
        print(f"[{r['final_score']:.6f}] {r['parent_doc_id']}: {r['text'][:50]}...")
