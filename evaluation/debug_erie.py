import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from retrieval.bm25 import search_bm25

# Check one query in detail
q = 'Erie Canal completion 1825'
res = search_bm25(q, size=50)

# Get relevant doc IDs from qrels
with open('data/qrels.json') as f:
    qrels = json.load(f)
rel = []
for qq in qrels:
    if qq['query_text'] == q:
        rel = qq['relevant_doc_ids']
        print('Relevant docs:', rel)
        break

print('\nTop 20 BM25 results:')
for i, r in enumerate(res[:20]):
    yr = r.get('publication_year')
    dec = (yr // 10) * 10 if yr else 'NA'
    in_rel = '*' if r['parent_doc_id'] in rel else ' '
    print(in_rel + ' ' + str(i+1) + '. ' + r['parent_doc_id'] + ' (' + str(dec) + '): ' + r['text'][:80] + '...')