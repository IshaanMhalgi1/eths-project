import os
import sys
import yaml
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import year_range_filter

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

DEFAULT_INDEX = cfg.get('index_name', 'ethsearch_chunks')
SMALL_INDEX = "ethsearch_chunks_small"
EMBEDDING_MODEL = cfg.get('embedding_model', 'all-MiniLM-L6-v2')

client = OpenSearch(
    hosts=[{'host': cfg['opensearch']['host'], 'port': cfg['opensearch']['port']}],
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False
)

# Load the model
model = SentenceTransformer(EMBEDDING_MODEL)

def search_dense(query: str, size: int = 10, index_name: str = DEFAULT_INDEX,
                 year_start=None, year_end=None):
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    knn_clause = {
        "knn": {"embedding": {"vector": query_embedding, "k": size}}
    }
    rng = year_range_filter(year_start, year_end)

    if rng is None:
        body = {"size": size, "query": knn_clause}
    else:
        # kNN with an inline filter, so the year range is applied to the vector
        # search itself rather than truncating an already-collected neighbour
        # list. Filtering after the fact would silently return fewer than `size`
        # results purely because of where the in-range documents happened to
        # rank, which reads as "the range has little in it" when it may not.
        body = {
            "size": size,
            "query": {
                "bool": {"must": [knn_clause], "filter": [rng]}
            },
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

if __name__ == '__main__':
    # Test
    res = search_dense("what happened in washington")
    for r in res[:2]:
        print(f"[{r['score']}] {r['parent_doc_id']}: {r['text'][:50]}...")
