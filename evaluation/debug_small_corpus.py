import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense

# Test BM25 on small corpus query
query = "obituaries deaths 1806"
print("Query:", query)
print()

# BM25 - no year filtering in search_bm25
res = search_bm25(query, size=50)
print("BM25 top 10 (full index):")
for i, r in enumerate(res[:10]):
    yr = r.get('publication_year')
    print("  {}. {} (year={}): {}...".format(i+1, r['parent_doc_id'], yr, r['text'][:60]))

print()
print("Years in top 20:")
from collections import Counter
years = []
for r in res[:20]:
    yr = r.get('publication_year')
    if yr:
        years.append(yr)
c = Counter(years)
for yr in sorted(c):
    print("  {}: {}".format(yr, c[yr]))