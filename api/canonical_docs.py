import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
from opensearchpy import OpenSearch

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

INDEX = cfg.get('index_name', 'ethsearch_chunks')
client = OpenSearch(
    hosts=[{'host': cfg['opensearch']['host'], 'port': cfg['opensearch']['port']}],
    use_ssl=False, verify_certs=False, ssl_show_warn=False
)

# Build canonical mapping: for each full-text group, pick one parent_doc_id as canonical
# Rule: choose the parent_doc_id with the longest total text (sum of its chunks)
# This is computed once at startup

def build_canonical_mapping():
    """Build mapping from any parent_doc_id to its canonical parent_doc_id."""
    # Scroll through all documents and group by text content
    # For efficiency, we use a composite aggregation or scroll
    body = {
        "size": 0,
        "aggs": {
            "by_text": {
                "terms": {"field": "text.keyword", "size": 50000},
                "aggs": {
                    "pids": {
                        "terms": {"field": "parent_doc_id", "size": 100}
                    }
                }
            }
        }
    }
    
    # Note: text.keyword may not exist. Let's use a different approach.
    # Scroll all docs and build in memory (50k docs is manageable)
    canonical_map = {}
    
    # First, get all unique parent_doc_ids and their texts
    pid_to_texts = {}
    scroll_body = {
        "size": 1000,
        "query": {"match_all": {}},
        "_source": ["parent_doc_id", "text", "chunk_id"]
    }
    
    resp = client.search(index=INDEX, body=scroll_body, scroll='5m')
    scroll_id = resp['_scroll_id']
    hits = resp['hits']['hits']
    
    while hits:
        for h in hits:
            source = h['_source']
            pid = source.get('parent_doc_id')
            text = source.get('text', '')
            if pid not in pid_to_texts:
                pid_to_texts[pid] = []
            pid_to_texts[pid].append(text)
        
        resp = client.scroll(scroll_id=scroll_id, scroll='5m')
        scroll_id = resp['_scroll_id']
        hits = resp['hits']['hits']
    
    # Now group by full-text content (concatenated chunks per pid)
    text_to_pids = {}
    for pid, texts in pid_to_texts.items():
        full_text = "\n\n".join(texts)
        if full_text not in text_to_pids:
            text_to_pids[full_text] = []
        text_to_pids[full_text].append(pid)
    
    # For each text group with multiple pids, pick canonical
    # Canonical = pid with most chunks (or first if tie)
    for text, pids in text_to_pids.items():
        if len(pids) > 1:
            # Pick the pid with most chunks
            best_pid = max(pids, key=lambda p: len(pid_to_texts[p]))
            for pid in pids:
                canonical_map[pid] = best_pid
        else:
            canonical_map[pids[0]] = pids[0]
    
    return canonical_map

# Build at module load
CANONICAL_MAP = build_canonical_mapping()

def get_canonical(parent_doc_id):
    """Return the canonical parent_doc_id for any given parent_doc_id."""
    return CANONICAL_MAP.get(parent_doc_id, parent_doc_id)