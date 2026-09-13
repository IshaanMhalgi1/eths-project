import requests
import json

# Test two known duplicate parent_doc_ids
for pid in ['train_2821', 'train_2822']:
    r = requests.get('http://127.0.0.1:8000/document/{}'.format(pid))
    doc = r.json()
    print('{}:'.format(pid))
    print('  chunks: {}'.format(len(doc['chunks'])))
    print('  full_text length: {}'.format(len(doc['full_text'])))
    print('  preview: {}'.format(doc['full_text'][:100]))
    print()