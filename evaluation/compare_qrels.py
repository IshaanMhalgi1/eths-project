import json

with open('data/qrels.json') as f:
    q1 = json.load(f)
with open('data/qrels_expanded.json') as f:
    q2 = json.load(f)

# Compare relevant docs for each query
for i in range(len(q1)):
    q1i = q1[i]
    q2i = q2[i]
    if q1i.get('relevant_doc_ids') != q2i.get('relevant_doc_ids'):
        print('DIFF {}:'.format(q1i['query_id']))
        print('  qrels: {}'.format(q1i.get('relevant_doc_ids')))
        print('  expanded: {}'.format(q2i.get('relevant_doc_ids')))