import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25, DEFAULT_INDEX, SMALL_INDEX

query = "obituaries deaths 1806"
print("Query:", query)
print()

# Full index
res_full = search_bm25(query, size=10, index_name=DEFAULT_INDEX)
print("Full index top 5:")
for i, r in enumerate(res_full[:5]):
    print("  {}. {} (year={})".format(i+1, r['parent_doc_id'], r.get('publication_year')))

print()
# Small index
res_small = search_bm25(query, size=10, index_name=SMALL_INDEX)
print("Small index top 5:")
for i, r in enumerate(res_small[:5]):
    print("  {}. {} (year={})".format(i+1, r['parent_doc_id'], r.get('publication_year')))
    assert 1800 <= r.get('publication_year', 0) < 1810, "OUT OF RANGE!"

years = [r.get('publication_year') for r in res_small]
print("\nSmall index years:", sorted(years))
assert all(1800 <= y < 1810 for y in years), "FOUND OUT OF RANGE YEARS"
print("PASS: all small-index results within 1800-1809")