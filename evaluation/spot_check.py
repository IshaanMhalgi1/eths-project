import os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from opensearchpy import OpenSearch
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

INDEX = cfg.get('index_name', 'ethsearch_chunks')
client = OpenSearch(
    hosts=[{'host': cfg['opensearch']['host'], 'port': cfg['opensearch']['port']}],
    use_ssl=False, verify_certs=False, ssl_show_warn=False
)

QRELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')
with open(QRELS_PATH) as f:
    qrels = json.load(f)

from retrieval.dense import search_dense

print("MANUAL SPOT CHECK (Lexical Ground Truth vs Dense Misses)\n")
print("=" * 80)

# Pick 8 diverse queries
queries_to_check = qrels[0:8] 

for q in queries_to_check:
    query = q['query_text']
    rel_ids = q['relevant_doc_ids']
    
    print(f"\nQUERY: '{query}'")
    print(f"--- QRELS GROUND TRUTH (Top {len(rel_ids)}) ---")
    
    for rid in rel_ids:
        # Fetch actual text
        resp = client.search(index=INDEX, body={"size":1, "query": {"match": {"parent_doc_id": rid}}})
        if resp['hits']['hits']:
            text = resp['hits']['hits'][0]['_source']['text']
            print(f"  [QREL] {rid}: {text[:150].replace(chr(10), ' ')}...")
            
    print(f"--- DENSE RETRIEVAL (Potential Semantic Misses) ---")
    dense_res = search_dense(query, size=15)
    
    shown = 0
    for r in dense_res:
        pid = r['parent_doc_id']
        if pid not in rel_ids:
            print(f"  [DENSE MISS] {pid}: {r['text'][:150].replace(chr(10), ' ')}...")
            shown += 1
        if shown >= 3:
            break
            
    print("=" * 80)
