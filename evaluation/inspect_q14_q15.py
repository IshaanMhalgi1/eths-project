import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from temporal.temporal_parser import TemporalParser

parser = TemporalParser()

# Load qrels
with open('data/qrels.json') as f:
    qrels = json.load(f)

# Load chunks for text lookup
chunk_text = {}
chunk_year = {}
with open('data/chunks.jsonl') as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            chunk_text[c['parent_doc_id']] = c['text'][:200]
            chunk_year[c['parent_doc_id']] = c.get('publication_year')

# Check q14 and q15
for q in qrels:
    if q['query_id'] in ('q14', 'q15'):
        print(f"=== {q['query_id']}: {q['query_text']} ===")
        intent = parser.parse(q['query_text'])
        print(f"  Parsed intent: {intent}")
        print(f"  Relevant docs: {q['relevant_doc_ids']}")
        for doc_id in q['relevant_doc_ids']:
            yr = chunk_year.get(doc_id)
            txt = chunk_text.get(doc_id, 'NOT FOUND')
            print(f"    {doc_id} (year={yr}): {txt}...")
        print()