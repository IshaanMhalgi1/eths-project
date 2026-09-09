"""
BM25 Diagnostic:
1. Print raw BM25 top-10 for 4 eval queries — see what chunk_id / parent_doc_id looks like
2. Check whether the qrels relevant_doc_id values match any parent_doc_id in the index
3. Print a sample of parent_doc_id values from OpenSearch to see the actual ID scheme
"""
import os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
from opensearchpy import OpenSearch

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

INDEX_NAME = cfg.get('index_name', 'ethsearch_chunks')
client = OpenSearch(
    hosts=[{'host': cfg['opensearch']['host'], 'port': cfg['opensearch']['port']}],
    use_ssl=False, verify_certs=False, ssl_show_warn=False
)

QRELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')
with open(QRELS_PATH) as f:
    qrels = json.load(f)

# ── 1. Sample real parent_doc_id values from the index ──────────────────────
print("=" * 60)
print("STEP 1 — Sample parent_doc_id values actually in OpenSearch")
print("=" * 60)
resp = client.search(index=INDEX_NAME, body={"size": 5, "query": {"match_all": {}}})
for hit in resp['hits']['hits']:
    print(f"  chunk_id (doc _id): {hit['_id']}")
    print(f"  parent_doc_id:      {hit['_source'].get('parent_doc_id')}")
    print(f"  publication_year:   {hit['_source'].get('publication_year')}")
    print(f"  text[:80]:          {hit['_source']['text'][:80]}")
    print()

# ── 2. Check if qrels IDs exist in the index ────────────────────────────────
print("=" * 60)
print("STEP 2 — Check if qrels relevant_doc_id values exist in OpenSearch")
print("=" * 60)
qrel_ids = [q['relevant_doc_id'] for q in qrels]
for qid in qrel_ids[:6]:
    resp = client.search(index=INDEX_NAME, body={
        "size": 1,
        "query": {"term": {"parent_doc_id": qid}}
    })
    count = resp['hits']['total']['value']
    print(f"  parent_doc_id='{qid}'  =>  {count} docs in index")

# ── 3. Raw BM25 top-10 for 4 sample queries ─────────────────────────────────
print()
print("=" * 60)
print("STEP 3 — Raw BM25 top-10 results for 4 sample queries")
print("=" * 60)
sample_queries = qrels[:4]
for q in sample_queries:
    query_text = q['query_text']
    expected = q['relevant_doc_id']
    resp = client.search(index=INDEX_NAME, body={
        "size": 10,
        "query": {"match": {"text": query_text}}
    })
    print(f"\nQuery: '{query_text}'  (expected parent_doc_id: '{expected}')")
    print(f"  Total hits in index: {resp['hits']['total']['value']}")
    for hit in resp['hits']['hits']:
        matched = "MATCH" if hit['_source'].get('parent_doc_id') == expected else ""
        print(f"  score={hit['_score']:.3f}  chunk_id={hit['_id']}  parent_doc_id={hit['_source'].get('parent_doc_id')}  {matched}")
        print(f"    text[:80]: {hit['_source']['text'][:80]}")

# ── 4. Check text field and mapping ─────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4 — Index mapping for 'text' field")
print("=" * 60)
mapping = client.indices.get_mapping(index=INDEX_NAME)
text_mapping = mapping[INDEX_NAME]['mappings']['properties'].get('text')
print(json.dumps(text_mapping, indent=2))
