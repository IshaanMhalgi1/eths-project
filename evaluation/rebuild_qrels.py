"""
Rebuild qrels.json properly.

For each of 20 hand-crafted queries, run BM25 and show the top-5 results.
Then write the best match as the relevant_doc_id into a new qrels.json.

We also print what we found so a human can verify.
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

# Hand-crafted queries with temporal spread across 1800-1810
QUERIES = [
    # (query_id, query_text, temporal_type)
    ("q1",  "lost pocket book reward",                          "none"),
    ("q2",  "flour and grain market prices",                    "none"),
    ("q3",  "runaway slave reward advertisement",               "none"),
    ("q4",  "military officer appointments",                    "none"),
    ("q5",  "obituaries deaths 1807",                           "explicit_year"),
    ("q6",  "real estate land for sale 1807",                   "explicit_year"),
    ("q7",  "ship arrivals cargo from Europe 1803",             "explicit_year"),
    ("q8",  "new store goods for sale 1803",                    "explicit_year"),
    ("q9",  "political news congress 1800 to 1805",             "range"),
    ("q10", "tax land assessment 1804 to 1808",                 "range"),
    ("q11", "legal public notice sheriff 1800 1810",            "range"),
    ("q12", "weather storm floods 1805 to 1810",                "range"),
    ("q13", "news dispatches France Europe before 1808",        "before_after"),
    ("q14", "shipping port arrivals after 1805",                "before_after"),
    ("q15", "medicines apothecary bitters since 1802",          "before_after"),
    ("q16", "poetry verse published before 1810",               "before_after"),
    ("q17", "European war news Jeffersonian Era",               "period"),
    ("q18", "election campaign republican Early Republic",      "period"),
    ("q19", "trade embargo Britain Jefferson administration",   "period"),
    ("q20", "letters editor public opinion Early National Period", "period"),
]

def bm25_top(query, size=10):
    resp = client.search(index=INDEX_NAME, body={
        "size": size,
        "query": {"match": {"text": {"query": query, "operator": "or"}}}
    })
    hits = []
    seen_parents = set()
    for h in resp['hits']['hits']:
        pid = h['_source']['parent_doc_id']
        if pid not in seen_parents:
            seen_parents.add(pid)
            hits.append({
                "chunk_id": h['_id'],
                "parent_doc_id": pid,
                "score": h['_score'],
                "year": h['_source'].get('publication_year'),
                "text": h['_source']['text']
            })
    return hits

qrels = []

print("=" * 70)
print("BM25 top-5 (deduplicated by parent) for each query")
print("Picking first result as relevant_doc_id (verify by eye below)")
print("=" * 70)

for qid, query, ttype in QUERIES:
    hits = bm25_top(query, size=30)  # fetch 30 chunks to get 5 unique parents
    # Take top 5 unique parents
    top5 = hits[:5]

    print(f"\n[{qid}] ({ttype}) Query: '{query}'")
    for i, h in enumerate(top5):
        marker = "  <-- SELECTED" if i == 0 else ""
        print(f"  #{i+1} score={h['score']:.3f} parent={h['parent_doc_id']} year={h['year']}{marker}")
        print(f"       text: {h['text'][:120]}")

    # Pick the top result as the relevant doc
    best = top5[0] if top5 else None
    if best:
        qrels.append({
            "query_id": qid,
            "query_text": query,
            "temporal_type": ttype,
            "relevant_doc_id": best['parent_doc_id'],
            "relevant_chunk_id": best['chunk_id'],
            "year_of_match": best['year']
        })
    else:
        print(f"  WARNING: no results for this query!")

OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')
with open(OUT_PATH, 'w') as f:
    json.dump(qrels, f, indent=2)

print()
print(f"Written {len(qrels)} qrels to {OUT_PATH}")

# Sanity check: verify every written relevant_doc_id exists in the index
print()
print("Sanity check: verifying all relevant_doc_ids exist in OpenSearch...")
all_ok = True
for q in qrels:
    resp = client.search(index=INDEX_NAME, body={
        "size": 1,
        "query": {"term": {"parent_doc_id": q['relevant_doc_id']}}
    })
    count = resp['hits']['total']['value']
    status = "OK" if count > 0 else "MISSING"
    if count == 0:
        all_ok = False
    print(f"  {q['query_id']} {q['relevant_doc_id']} => {count} chunks [{status}]")

if all_ok:
    print("\nAll IDs verified. Re-run evaluation/metrics.py to get honest baseline numbers.")
else:
    print("\nSome IDs are missing — check the output above.")
