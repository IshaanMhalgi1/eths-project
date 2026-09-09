import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from temporal.temporal_parser import TemporalParser
from retrieval.hybrid import search_hybrid
from ranking.temporal_ranker import calculate_temporal_score

parser = TemporalParser()

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

def final_search(query: str, size: int = 10, w_bm25=0.3, w_dense=0.3, w_temp=0.3, w_meta=0.1):
    temporal_intent = parser.parse(query)
    query_start = temporal_intent.get('start_year')
    query_end = temporal_intent.get('end_year')
    meta_intent = extract_metadata_intent(query)
    
    fetch_size = size * 5
    # Get base hybrid scores normalized internally
    hybrid_res = search_hybrid(query, size=fetch_size, alpha=0.5)
    
    # 1. Calculate raw scores
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

    # 2. Extract arrays for normalization
    bm25_scores = [r.get('bm25_score', 0) for r in hybrid_res]
    dense_scores = [r.get('dense_score', 0) for r in hybrid_res]
    temp_scores = [r.get('raw_temporal_score', 0) for r in hybrid_res]
    meta_scores = [r.get('raw_metadata_score', 0) for r in hybrid_res]
    
    # 3. Min-Max Helpers
    def get_norm(val, arr):
        if not arr: return 0.0
        min_v, max_v = min(arr), max(arr)
        if max_v > min_v:
            return (val - min_v) / (max_v - min_v)
        return val

    # 4. Normalize and blend
    for r in hybrid_res:
        norm_bm25 = get_norm(r.get('bm25_score', 0), bm25_scores)
        norm_dense = get_norm(r.get('dense_score', 0), dense_scores)
        norm_temp = get_norm(r.get('raw_temporal_score', 0), temp_scores)
        norm_meta = get_norm(r.get('raw_metadata_score', 0), meta_scores)
        
        r['bm25_score'] = norm_bm25
        r['dense_score'] = norm_dense
        r['temporal_score'] = norm_temp
        r['metadata_score'] = norm_meta
        
        r['final_score'] = (
            w_bm25 * norm_bm25 +
            w_dense * norm_dense +
            w_temp * norm_temp +
            w_meta * norm_meta
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
        print(f"[{r['final_score']:.3f}] {r['parent_doc_id']}: {r['text'][:50]}...")
