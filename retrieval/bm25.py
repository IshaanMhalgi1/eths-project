import os
import yaml
from opensearchpy import OpenSearch

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

DEFAULT_INDEX = cfg.get('index_name', 'ethsearch_chunks')
SMALL_INDEX = "ethsearch_chunks_small"

client = OpenSearch(
    hosts=[{'host': cfg['opensearch']['host'], 'port': cfg['opensearch']['port']}],
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False
)

def year_range_filter(year_start=None, year_end=None):
    """Range clause on publication_year, or None when no constraint is given.

    Filtering on publication_year (rather than the historical window) keeps the
    constraint consistent with the year the timeline plots for each result. A
    document with no publication year cannot be confirmed to sit inside a
    selected range, so a hard filter excludes it rather than quietly including
    something the user cannot see positioned.
    """
    if year_start is None and year_end is None:
        return None
    bounds = {}
    if year_start is not None:
        bounds['gte'] = int(year_start)
    if year_end is not None:
        bounds['lte'] = int(year_end)
    return {"range": {"publication_year": bounds}}


def search_bm25(query: str, size: int = 10, index_name: str = DEFAULT_INDEX,
                year_start=None, year_end=None):
    text_clause = {"match": {"text": query}}
    rng = year_range_filter(year_start, year_end)
    if rng is None:
        body = {"size": size, "query": text_clause}
    else:
        body = {
            "size": size,
            "query": {
                "bool": {"must": [text_clause], "filter": [rng]}
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
    res = search_bm25("what happened in washington")
    for r in res[:2]:
        print(f"[{r['score']}] {r['parent_doc_id']}: {r['text'][:50]}...")
