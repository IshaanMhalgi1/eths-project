import os
import yaml
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

INDEX_NAME = cfg.get('index_name', 'ethsearch_chunks')
EMBEDDING_MODEL = cfg.get('embedding_model', 'all-MiniLM-L6-v2')

client = OpenSearch(
    hosts=[{'host': cfg['opensearch']['host'], 'port': cfg['opensearch']['port']}],
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False
)

# Load the model
model = SentenceTransformer(EMBEDDING_MODEL)

def search_dense(query: str, size: int = 10):
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
    
    response = client.search(index=INDEX_NAME, body=body)
    hits = response['hits']['hits']
    results = []
    for hit in hits:
        results.append({
            "chunk_id": hit['_id'],
            "parent_doc_id": hit['_source']['parent_doc_id'],
            "score": hit['_score'],
            "text": hit['_source']['text'],
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
