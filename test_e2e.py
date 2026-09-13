import requests
import json

# Test search
r = requests.post('http://127.0.0.1:8000/search', json={'query': 'ship arrivals cargo from Europe 1803', 'size': 5})
results = r.json()['results']
print('Search results:')
for res in results[:3]:
    pid = res['parent_doc_id']
    txt = res['text'][:80]
    print('  {}: {}...'.format(pid, txt))

# Test document view for first result
if results:
    doc_id = results[0]['parent_doc_id']
    r2 = requests.get('http://127.0.0.1:8000/document/{}'.format(doc_id))
    doc = r2.json()
    print('\nDocument view for {}:'.format(doc_id))
    print('  Chunks: {}'.format(len(doc['chunks'])))
    print('  Full text length: {}'.format(len(doc['full_text'])))
    print('  First chunk: {}'.format(doc['chunks'][0]['chunk_id']))