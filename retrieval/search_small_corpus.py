import os
import yaml
from opensearchpy import OpenSearch

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

INDEX_NAME = cfg.get('index_name', 'ethsearch_chunks')
SMALL_INDEX_NAME = "ethsearch_chunks_small"

client = OpenSearch(
    hosts=[{'host': cfg['opensearch']['host'], 'port': cfg['opensearch']['port']}],
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False
)

def search_bm25(query: str, size: int = 10, index_name: str = INDEX_NAME):
    body = {
        "size": size,
        "query": {
            "match": {
                "text": query
            }
        }
    }
    response = client.search(index=index_name, body=body)
    hits = response['hits']['hits']
    results = []
    for hit in hits:
        results.append({
            "chunk_id": hit['_id'],
            "parent_doc_id": hit['_source']['parent_doc_id'],
            "score": hit['_score'],
            "text": hit['_source']['text'],
            "publication_year": hit['_source'].get('publication_year'),
            "historical_start": hit['_source'].get('historical_start'),
            "historical_end": hit['_source'].get('historical_end'),
            "historical_period": hit['_source'].get('historical_period'),
            "location": hit['_source'].get('location')
        })
    return results

def search_dense(query: str, size: int = 10, index_name: str = INDEX_NAME):
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = cfg.get('embedding_model', 'all-MiniLM-L6-v2')
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    
    body = {
        "size": size,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": size
                }
            }
        }
    }
    
    response = client.search(index=index_name, body=body)
    hits = response['hits']['hits']
    results = []
    for hit in hits:
        results.append({
            "chunk_id": hit['_id'],
            "parent_doc_id": hit['_source']['parent_doc_id'],
            "score": hit['_score'],
            "text": hit['_source']['text'],
            "publication_year": hit['_source'].get('publication_year'),
            "historical_start": hit['_source'].get('historical_start'),
            "historical_end": hit['_source'].get('historical_end'),
            "historical_period": hit['_source'].get('historical_period'),
            "location": hit['_source'].get('location')
        })
    return results

def search_hybrid(query: str, size: int = 10, alpha: float = 0.5, index_name: str = INDEX_NAME):
    from retrieval.bm25 import search_bm25 as bm25_search
    from retrieval.dense import search_dense as dense_search
    
    fetch_size = size * 2
    bm25_res = bm25_search(query, size=fetch_size, index_name=index_name)
    dense_res = dense_search(query, size=fetch_size, index_name=index_name)
    
    bm25_ranks = {r['chunk_id']: rank + 1 for rank, r in enumerate(bm25_res)}
    dense_ranks = {r['chunk_id']: rank + 1 for rank, r in enumerate(dense_res)}
    
    k = 60
    
    combined = {}
    for r in bm25_res:
        rank_bm25 = bm25_ranks[r['chunk_id']]
        rrf_bm25 = 1.0 / (k + rank_bm25)
        combined[r['chunk_id']] = {
            "chunk_id": r['chunk_id'],
            "parent_doc_id": r['parent_doc_id'],
            "text": r['text'],
            "publication_year": r.get('publication_year'),
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
                "publication_year": r.get('publication_year'),
                "historical_start": r.get('historical_start'),
                "historical_end": r.get('historical_end'),
                "historical_period": r.get('historical_period'),
                "location": r.get('location'),
                "bm25_score": 0.0,
                "dense_score": rrf_dense,
                "score": (1 - alpha) * rrf_dense
            }
            
    sorted_res = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
    
    seen = set()
    deduped = []
    for r in sorted_res:
        cid = r['chunk_id']
        if cid not in seen:
            seen.add(cid)
            deduped.append(r)
    return deduped[:size]


if __name__ == '__main__':
    res = search_bm25("what happened in washington")
    for r in res[:2]:
        print(f"[{r['score']}] {r['parent_doc_id']}: {r['text'][:50]}...")