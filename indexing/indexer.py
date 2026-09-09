import os
import json
import yaml
from opensearchpy import OpenSearch, helpers

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
MAPPING_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'index_mapping.json')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.jsonl')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

INDEX_NAME = cfg.get('index_name', 'ethsearch_chunks')
OS_HOST = cfg['opensearch']['host']
OS_PORT = cfg['opensearch']['port']

# Security is disabled in our dev docker-compose
client = OpenSearch(
    hosts=[{'host': OS_HOST, 'port': OS_PORT}],
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False
)

def create_index():
    with open(MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
    
    if client.indices.exists(index=INDEX_NAME):
        print(f"Index {INDEX_NAME} already exists. Deleting it.")
        client.indices.delete(index=INDEX_NAME)
    
    client.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Created index: {INDEX_NAME}")

def index_data():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        return

    def doc_generator():
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                yield {
                    "_index": INDEX_NAME,
                    "_id": doc["chunk_id"],
                    "_source": doc
                }
    
    success, failed = helpers.bulk(client, doc_generator(), chunk_size=500)
    print(f"Successfully indexed {success} documents.")
    if failed:
        print(f"Failed to index {failed} documents.")

if __name__ == "__main__":
    create_index()
    index_data()
