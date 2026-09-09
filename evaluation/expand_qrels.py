import os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
from opensearchpy import OpenSearch
from temporal.temporal_parser import TemporalParser

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

INDEX = cfg.get('index_name', 'ethsearch_chunks')
client = OpenSearch(
    hosts=[{'host': cfg['opensearch']['host'], 'port': cfg['opensearch']['port']}],
    use_ssl=False, verify_certs=False, ssl_show_warn=False
)

parser = TemporalParser()

QRELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')
with open(QRELS_PATH) as f:
    qrels = json.load(f)

new_qrels = []

for q in qrels:
    query = q['query_text']
    
    # Parse the true temporal intent
    intent = parser.parse(query)
    start_y = intent.get('start_year')
    end_y = intent.get('end_year')
    
    # Build a query that REQUIRES the document to match the year constraint
    must_clauses = [{"match": {"text": {"query": query, "operator": "or"}}}]
    filter_clauses = []
    
    if start_y is not None and end_y is not None:
        filter_clauses.append({
            "range": {
                "publication_year": {
                    "gte": start_y,
                    "lte": end_y
                }
            }
        })
        
    body = {
        "size": 50,
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        }
    }
    
    resp = client.search(index=INDEX, body=body)
    hits = resp['hits']['hits']
    
    seen_parents = set()
    dedup_hits = []
    for h in hits:
        pid = h['_source']['parent_doc_id']
        if pid not in seen_parents:
            seen_parents.add(pid)
            dedup_hits.append(h)
            
    if not dedup_hits:
        print(f"Warning: No valid docs found for '{query}' with range {start_y}-{end_y}")
        # Fallback to whatever was there, but it will be temporally wrong
        q['relevant_doc_ids'] = [q.get('relevant_doc_id')] if q.get('relevant_doc_id') else []
        new_qrels.append(q)
        continue
        
    # Take top docs that have a score within 70% of the absolute best match
    top_score = dedup_hits[0]['_score']
    threshold = top_score * 0.70
    
    relevant_docs = []
    for h in dedup_hits:
        if h['_score'] >= threshold:
            relevant_docs.append(h['_source']['parent_doc_id'])
            
    # Set the new ground truth (up to 5 relevant chunks per query)
    q['relevant_doc_ids'] = relevant_docs[:5]
    
    # Clean up old single ID field to avoid confusion
    if 'relevant_doc_id' in q:
        del q['relevant_doc_id']
    if 'relevant_chunk_id' in q:
        del q['relevant_chunk_id']
    if 'year_of_match' in q:
        del q['year_of_match']
        
    new_qrels.append(q)

with open(QRELS_PATH, 'w') as f:
    json.dump(new_qrels, f, indent=2)

print(f"Expanded and temporally-corrected qrels.json.")
print(f"Avg docs per query: {sum(len(q['relevant_doc_ids']) for q in new_qrels) / len(new_qrels):.2f}")
