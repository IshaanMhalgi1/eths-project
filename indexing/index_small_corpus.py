import os
import json
import yaml
from opensearchpy import OpenSearch, helpers

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
MAPPING_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'index_mapping.json')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.jsonl')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

OS_HOST = cfg['opensearch']['host']
OS_PORT = cfg['opensearch']['port']

client = OpenSearch(
    hosts=[{'host': OS_HOST, 'port': OS_PORT}],
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False
)

SMALL_INDEX_NAME = "ethsearch_chunks_small"
SMALL_YEAR_START = 1800
SMALL_YEAR_END = 1810  # exclusive, so 1800-1809

with open(MAPPING_PATH, 'r') as f:
    mapping = json.load(f)

# Delete existing small index if exists
if client.indices.exists(index=SMALL_INDEX_NAME):
    print(f"Deleting existing index: {SMALL_INDEX_NAME}")
    client.indices.delete(index=SMALL_INDEX_NAME)

client.indices.create(index=SMALL_INDEX_NAME, body=mapping)
print(f"Created index: {SMALL_INDEX_NAME}")

# Index only 1800-1809 chunks
def doc_generator():
    count = 0
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            yr = doc.get('publication_year')
            if yr is not None and SMALL_YEAR_START <= yr < SMALL_YEAR_END:
                yield {
                    "_index": SMALL_INDEX_NAME,
                    "_id": doc["chunk_id"],
                    "_source": doc
                }
                count += 1
    print(f"Total chunks to index: {count}")

success, failed = helpers.bulk(client, doc_generator(), chunk_size=500)
print(f"Successfully indexed {success} documents.")
if failed:
    print(f"Failed to index {failed} documents.")

# Verify
result = client.search(index=SMALL_INDEX_NAME, body={"size": 0, "query": {"match_all": {}}})
print(f"Index count: {result['hits']['total']['value']}")