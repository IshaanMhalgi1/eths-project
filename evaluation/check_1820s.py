import os
import sys
import json
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from temporal.temporal_parser import TemporalParser

parser = TemporalParser()

# Load qrels
with open('data/qrels.json') as f:
    qrels = json.load(f)

# Load chunks
chunk_year = {}
with open('data/chunks.jsonl') as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            chunk_year[c['parent_doc_id']] = c.get('publication_year')

# Check 1820s queries
for q in qrels:
    if not q.get('relevant_doc_ids'):
        continue
    intent = parser.parse(q['query_text'])
    start_y = intent.get('start_year')
    query_decade = (start_y // 10) * 10 if start_y else None
    
    if query_decade == 1820:
        rel_decades = []
        for doc_id in q['relevant_doc_ids']:
            yr = chunk_year.get(doc_id)
            if yr and 1800 <= yr < 1900:
                rel_decades.append((yr // 10) * 10)
        dec_counts = Counter(rel_decades)
        print(q['query_id'] + ": " + q['query_text'][:50] + "...")
        print("  relevant docs: " + str(len(q['relevant_doc_ids'])) + ", decade dist: " + str(dict(dec_counts)))
        print()