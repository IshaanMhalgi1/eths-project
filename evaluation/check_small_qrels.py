import json
# Check small corpus qrels - how many have relevant docs in 1800-1810?
with open('data/qrels.json') as f:
    qrels = json.load(f)

# Load chunk years
chunk_year = {}
with open('data/chunks.jsonl') as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            chunk_year[c['parent_doc_id']] = c.get('publication_year')

count_valid = 0
count_total = 0
for q in qrels:
    if not q.get('relevant_doc_ids'):
        continue
    count_total += 1
    has_in_small = False
    for doc_id in q['relevant_doc_ids']:
        yr = chunk_year.get(doc_id)
        if yr and 1800 <= yr < 1810:
            has_in_small = True
            break
    if has_in_small:
        count_valid += 1
    else:
        print('NO SMALL CORPUS REL: {} - {}... -> {}'.format(
            q['query_id'], q['query_text'][:50], [chunk_year.get(d) for d in q['relevant_doc_ids']]))

print('Total queries with relevant: {}'.format(count_total))
print('Have at least 1 relevant in 1800-1810: {}'.format(count_valid))
print('Zero relevant in 1800-1810: {}'.format(count_total - count_valid))