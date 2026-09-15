import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25, DEFAULT_INDEX, SMALL_INDEX
from retrieval.dense import search_dense
from retrieval.hybrid import search_hybrid
from ranking.shared_hybrid import get_hybrid_candidates, clear_hybrid_cache, SHARED_FETCH_SIZE
from ranking.temporal_ranker import calculate_temporal_score
from temporal.temporal_parser import TemporalParser

with open(os.path.join(os.path.dirname(__file__), '..', 'ranking', 'normalization_stats.json'), 'r') as f:
    NORM_STATS = json.load(f)

parser = TemporalParser()

def z_score_normalize(val, mean, std):
    if std > 0:
        return (val - mean) / std
    return 0.0


def rank_bm25(query, size=10, index_name=DEFAULT_INDEX):
    res = search_bm25(query, size=SHARED_FETCH_SIZE, index_name=index_name)
    for r in res:
        r['final_score'] = r['score']
    return sorted(res, key=lambda x: x['final_score'], reverse=True)[:size]


def rank_dense(query, size=10, index_name=DEFAULT_INDEX):
    res = search_dense(query, size=SHARED_FETCH_SIZE, index_name=index_name)
    for r in res:
        r['final_score'] = r['score']
    return sorted(res, key=lambda x: x['final_score'], reverse=True)[:size]


def rank_hybrid(query, size=10, index_name=DEFAULT_INDEX, alpha=0.5):
    clear_hybrid_cache()
    res = search_hybrid(query, size=SHARED_FETCH_SIZE, alpha=alpha, index_name=index_name)
    for r in res:
        r['final_score'] = r['score']
    return sorted(res, key=lambda x: x['final_score'], reverse=True)[:size]


def rank_hybrid_temporal(query, size=10, index_name=DEFAULT_INDEX, alpha_hybrid=0.7, beta_temporal=0.3, alpha_rrf=0.5):
    temporal_intent = parser.parse(query)
    query_start = temporal_intent.get('start_year')
    query_end = temporal_intent.get('end_year')
    clear_hybrid_cache()
    hybrid_res = search_hybrid(query, size=SHARED_FETCH_SIZE, alpha=alpha_rrf, index_name=index_name)
    if query_start is None or query_end is None:
        for r in hybrid_res:
            r['final_score'] = r['score']
        return sorted(hybrid_res, key=lambda x: x['final_score'], reverse=True)[:size]
    for r in hybrid_res:
        doc_start = r.get('historical_start')
        doc_end = r.get('historical_end')
        t_score, _ = calculate_temporal_score(doc_start, doc_end, query_start, query_end)
        r['raw_temporal_score'] = t_score
    hybrid_std = NORM_STATS['dense_std']
    for r in hybrid_res:
        norm_temp = z_score_normalize(r['raw_temporal_score'], NORM_STATS['temporal_mean'], NORM_STATS['temporal_std'])
        temp_adj = norm_temp * hybrid_std
        r['temporal_score'] = temp_adj
        r['hybrid_score'] = r['score']
        r['final_score'] = alpha_hybrid * r['score'] + beta_temporal * temp_adj
    return sorted(hybrid_res, key=lambda x: x['final_score'], reverse=True)[:size]


def rank_hybrid_metadata(query, size=10, index_name=DEFAULT_INDEX, alpha_hybrid=0.7, gamma_metadata=0.1, alpha_rrf=0.5):
    temporal_intent = parser.parse(query)
    meta_intent = {"periods": temporal_intent.get('periods', []), "locations": []}
    clear_hybrid_cache()
    hybrid_res = search_hybrid(query, size=SHARED_FETCH_SIZE, alpha=alpha_rrf, index_name=index_name)
    for r in hybrid_res:
        doc_period = r.get('historical_period')
        doc_location = r.get('location')
        score = 0.0
        if meta_intent['periods'] and doc_period:
            if any(p.lower() in doc_period.lower() for p in meta_intent['periods']):
                score += 0.5
        if meta_intent['locations'] and doc_location:
            if any(l.lower() in doc_location.lower() for l in meta_intent['locations']):
                score += 0.5
        if not meta_intent['periods'] and not meta_intent['locations']:
            score = 0.5
        r['raw_metadata_score'] = score
    hybrid_std = NORM_STATS['dense_std']
    for r in hybrid_res:
        norm_meta = z_score_normalize(r['raw_metadata_score'], NORM_STATS['metadata_mean'], NORM_STATS['metadata_std'])
        meta_adj = norm_meta * hybrid_std
        r['metadata_score'] = meta_adj
        r['hybrid_score'] = r['score']
        r['final_score'] = alpha_hybrid * r['score'] + gamma_metadata * meta_adj
    return sorted(hybrid_res, key=lambda x: x['final_score'], reverse=True)[:size]


def rank_final(query, size=10, index_name=DEFAULT_INDEX, w_hybrid=0.7, w_temp=0.2, w_meta=0.1, alpha_rrf=0.5):
    temporal_intent = parser.parse(query)
    query_start = temporal_intent.get('start_year')
    query_end = temporal_intent.get('end_year')
    meta_intent = {"periods": temporal_intent.get('periods', []), "locations": []}
    clear_hybrid_cache()
    hybrid_res = search_hybrid(query, size=SHARED_FETCH_SIZE, alpha=alpha_rrf, index_name=index_name)
    for r in hybrid_res:
        doc_start = r.get('historical_start')
        doc_end = r.get('historical_end')
        if query_start is None or query_end is None:
            t_score = 0.5
        else:
            t_score, _ = calculate_temporal_score(doc_start, doc_end, query_start, query_end)
        r['raw_temporal_score'] = t_score
        doc_period = r.get('historical_period')
        doc_location = r.get('location')
        m_score = 0.0
        if meta_intent['periods'] and doc_period:
            if any(p.lower() in doc_period.lower() for p in meta_intent['periods']):
                m_score += 0.5
        if meta_intent['locations'] and doc_location:
            if any(l.lower() in doc_location.lower() for l in meta_intent['locations']):
                m_score += 0.5
        if not meta_intent['periods'] and not meta_intent['locations']:
            m_score = 0.5
        r['raw_metadata_score'] = m_score
    hybrid_std = NORM_STATS['dense_std']
    for r in hybrid_res:
        norm_temp = z_score_normalize(r['raw_temporal_score'], NORM_STATS['temporal_mean'], NORM_STATS['temporal_std'])
        norm_meta = z_score_normalize(r['raw_metadata_score'], NORM_STATS['metadata_mean'], NORM_STATS['metadata_std'])
        temp_adj = norm_temp * hybrid_std
        meta_adj = norm_meta * hybrid_std
        r['temporal_score'] = temp_adj
        r['metadata_score'] = meta_adj
        r['hybrid_score'] = r['score']
        r['final_score'] = w_hybrid * r['score'] + w_temp * temp_adj + w_meta * meta_adj
    return sorted(hybrid_res, key=lambda x: x['final_score'], reverse=True)[:size]


def rank_bm25_temporal(query, size=10, index_name=DEFAULT_INDEX, alpha_bm25=0.7, beta_temporal=0.3):
    temporal_intent = parser.parse(query)
    query_start = temporal_intent.get('start_year')
    query_end = temporal_intent.get('end_year')
    bm25_res = search_bm25(query, size=SHARED_FETCH_SIZE, index_name=index_name)
    if query_start is None or query_end is None:
        for r in bm25_res:
            r['final_score'] = r['score']
        return sorted(bm25_res, key=lambda x: x['final_score'], reverse=True)[:size]
    
    # Convert BM25 to RRF scale (same as hybrid does for BM25 component)
    k = 60
    bm25_ranks = {r['chunk_id']: rank + 1 for rank, r in enumerate(bm25_res)}
    for r in bm25_res:
        rank_bm25 = bm25_ranks[r['chunk_id']]
        r['rrf_score'] = 1.0 / (k + rank_bm25)
    
    for r in bm25_res:
        doc_start = r.get('historical_start')
        doc_end = r.get('historical_end')
        t_score, _ = calculate_temporal_score(doc_start, doc_end, query_start, query_end)
        r['raw_temporal_score'] = t_score
    
    # Use RRF scale std for temporal adjustment (consistent with hybrid)
    rrf_std = NORM_STATS['dense_std']
    for r in bm25_res:
        norm_temp = z_score_normalize(r['raw_temporal_score'], NORM_STATS['temporal_mean'], NORM_STATS['temporal_std'])
        temp_adj = norm_temp * rrf_std
        r['temporal_score'] = temp_adj
        r['bm25_score'] = r['rrf_score']  # Use RRF-normalized BM25
        r['final_score'] = alpha_bm25 * r['rrf_score'] + beta_temporal * temp_adj
    return sorted(bm25_res, key=lambda x: x['final_score'], reverse=True)[:size]


def rank_dense_temporal(query, size=10, index_name=DEFAULT_INDEX, alpha_dense=0.7, beta_temporal=0.3):
    temporal_intent = parser.parse(query)
    query_start = temporal_intent.get('start_year')
    query_end = temporal_intent.get('end_year')
    dense_res = search_dense(query, size=SHARED_FETCH_SIZE, index_name=index_name)
    if query_start is None or query_end is None:
        for r in dense_res:
            r['final_score'] = r['score']
        return sorted(dense_res, key=lambda x: x['final_score'], reverse=True)[:size]
    for r in dense_res:
        doc_start = r.get('historical_start')
        doc_end = r.get('historical_end')
        t_score, _ = calculate_temporal_score(doc_start, doc_end, query_start, query_end)
        r['raw_temporal_score'] = t_score
    dense_std = NORM_STATS['dense_std']
    for r in dense_res:
        norm_temp = z_score_normalize(r['raw_temporal_score'], NORM_STATS['temporal_mean'], NORM_STATS['temporal_std'])
        temp_adj = norm_temp * dense_std
        r['temporal_score'] = temp_adj
        r['dense_score'] = r['score']
        r['final_score'] = alpha_dense * r['score'] + beta_temporal * temp_adj
    return sorted(dense_res, key=lambda x: x['final_score'], reverse=True)[:size]


def rank_temporal_only(query, size=10, index_name=DEFAULT_INDEX):
    temporal_intent = parser.parse(query)
    query_start = temporal_intent.get('start_year')
    query_end = temporal_intent.get('end_year')
    candidates = search_bm25(query, size=SHARED_FETCH_SIZE, index_name=index_name)
    if query_start is None or query_end is None:
        for r in candidates:
            r['final_score'] = 0.0
        return candidates[:size]
    for r in candidates:
        doc_start = r.get('historical_start')
        doc_end = r.get('historical_end')
        t_score, _ = calculate_temporal_score(doc_start, doc_end, query_start, query_end)
        r['raw_temporal_score'] = t_score
        r['final_score'] = t_score
    return sorted(candidates, key=lambda x: x['final_score'], reverse=True)[:size]


def rank_metadata_only(query, size=10, index_name=DEFAULT_INDEX):
    temporal_intent = parser.parse(query)
    meta_intent = {"periods": temporal_intent.get('periods', []), "locations": []}
    candidates = search_bm25(query, size=SHARED_FETCH_SIZE, index_name=index_name)
    for r in candidates:
        doc_period = r.get('historical_period')
        doc_location = r.get('location')
        score = 0.0
        if meta_intent['periods'] and doc_period:
            if any(p.lower() in doc_period.lower() for p in meta_intent['periods']):
                score += 0.5
        if meta_intent['locations'] and doc_location:
            if any(l.lower() in doc_location.lower() for l in meta_intent['locations']):
                score += 0.5
        if not meta_intent['periods'] and not meta_intent['locations']:
            score = 0.5
        r['final_score'] = score
    return sorted(candidates, key=lambda x: x['final_score'], reverse=True)[:size]


CORPUS_CONFIG = {
    'small': {
        'name': '1800-1810',
        'index_name': SMALL_INDEX,
        'qrels_path': os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels_small.json'),
        'hybrid': {'alpha': 0.7},
        'hybrid_temporal': {'alpha_hybrid': 0.7, 'beta_temporal': 0.3, 'alpha_rrf': 0.7},
        'hybrid_metadata': {'alpha_hybrid': 0.7, 'gamma_metadata': 0.1, 'alpha_rrf': 0.7},
        'final': {'w_hybrid': 0.7, 'w_temp': 0.2, 'w_meta': 0.1, 'alpha_rrf': 0.7},
        'bm25_temporal': {'alpha_bm25': 0.7, 'beta_temporal': 0.3},
        'dense_temporal': {'alpha_dense': 0.7, 'beta_temporal': 0.3},
    },
    'expanded': {
        'name': '1800-1900',
        'index_name': DEFAULT_INDEX,
        'qrels_path': os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels_expanded.json'),
        'hybrid': {'alpha': 1.0},
        'hybrid_temporal': {'alpha_hybrid': 0.7, 'beta_temporal': 0.5, 'alpha_rrf': 1.0},
        'hybrid_metadata': {'alpha_hybrid': 0.7, 'gamma_metadata': 0.1, 'alpha_rrf': 1.0},
        'final': {'w_hybrid': 0.7, 'w_temp': 0.2, 'w_meta': 0.1, 'alpha_rrf': 1.0},
        'bm25_temporal': {'alpha_bm25': 0.7, 'beta_temporal': 0.5},
        'dense_temporal': {'alpha_dense': 0.7, 'beta_temporal': 0.5},
    }
}


def get_ranker(name, corpus='expanded'):
    config = CORPUS_CONFIG[corpus]
    idx = config['index_name']
    if name == 'BM25':
        return lambda q, size=10: rank_bm25(q, size, idx)
    elif name == 'Dense':
        return lambda q, size=10: rank_dense(q, size, idx)
    elif name == 'Hybrid':
        c = config['hybrid']
        return lambda q, size=10: rank_hybrid(q, size, idx, c['alpha'])
    elif name == 'Hybrid+Temporal':
        c = config['hybrid_temporal']
        return lambda q, size=10: rank_hybrid_temporal(q, size, idx, c['alpha_hybrid'], c['beta_temporal'], c['alpha_rrf'])
    elif name == 'Hybrid+Metadata':
        c = config['hybrid_metadata']
        return lambda q, size=10: rank_hybrid_metadata(q, size, idx, c['alpha_hybrid'], c['gamma_metadata'], c['alpha_rrf'])
    elif name == 'Final':
        c = config['final']
        return lambda q, size=10: rank_final(q, size, idx, c['w_hybrid'], c['w_temp'], c['w_meta'], c['alpha_rrf'])
    elif name == 'BM25+Temporal':
        c = config['bm25_temporal']
        return lambda q, size=10: rank_bm25_temporal(q, size, idx, c['alpha_bm25'], c['beta_temporal'])
    elif name == 'Dense+Temporal':
        c = config['dense_temporal']
        return lambda q, size=10: rank_dense_temporal(q, size, idx, c['alpha_dense'], c['beta_temporal'])
    elif name == 'TemporalOnly':
        return lambda q, size=10: rank_temporal_only(q, size, idx)
    elif name == 'MetadataOnly':
        return lambda q, size=10: rank_metadata_only(q, size, idx)
    else:
        raise ValueError("Unknown ranker: " + name)
