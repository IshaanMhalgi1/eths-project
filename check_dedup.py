import json
from collections import defaultdict

chunks = []
with open('data/chunks.jsonl') as f:
    for line in f:
        chunks.append(json.loads(line))

# Parent-doc-ID duplicates with years
parent_to_years = defaultdict(set)
for c in chunks:
    pid = c['parent_doc_id']
    yr = c.get('publication_year')
    if yr:
        parent_to_years[pid].add(yr)

print('Parent-doc-ID duplicates with cross-decade:')
for pid, years in parent_to_years.items():
    if len(years) > 1:
        print(f'  {pid}: years={sorted(years)}')

# Full-text duplicates with years
text_to_info = defaultdict(lambda: {'count': 0, 'years': set(), 'pids': set()})
for c in chunks:
    text = c['text']
    yr = c.get('publication_year')
    pid = c['parent_doc_id']
    text_to_info[text]['count'] += 1
    if yr:
        text_to_info[text]['years'].add(yr)
    text_to_info[text]['pids'].add(pid)

print()
cross_decade = 0
for text, info in text_to_info.items():
    if info['count'] > 1:
        if len(info['years']) > 1:
            cross_decade += 1
            if cross_decade <= 20:
                yrs = sorted(info['years'])
                print(f'  years={yrs}, count={info["count"]}, pids={len(info["pids"])}')
print(f'Total cross-decade full-text duplicates: {cross_decade}')

same_decade = 0
for text, info in text_to_info.items():
    if info['count'] > 1 and len(info['years']) == 1:
        same_decade += 1
print(f'Same-decade full-text duplicates: {same_decade}')