"""Verify the year-range constraint and the density artifact against real data.

Run from the repo root with the project venv.
"""
import os
import sys

sys.path.append(os.path.abspath('.'))

from retrieval.bm25 import search_bm25
from retrieval.hybrid import search_hybrid
from ranking.ranker import final_search
from api.corpus_stats import decade_density

Q = 'cotton prices'


def yrs(rows, key='publication_year'):
    return sorted({r.get(key) for r in rows if r.get(key) is not None})


print('=== 1. BM25 with and without a year range ===')
base = search_bm25(Q, size=10)
print(f'  unfiltered years : {yrs(base)}')
filt = search_bm25(Q, size=10, year_start=1850, year_end=1860)
print(f'  1850-1860 years  : {yrs(filt)}')
bad = [r for r in filt if not (1850 <= (r.get("publication_year") or 0) <= 1860)]
print(f'  out-of-range hits: {len(bad)} {"OK" if not bad else "LEAK -> BUG"}')

print('\n=== 2. Hybrid with a year range (kNN filtering path) ===')
hyb = search_hybrid(Q, size=10, year_start=1850, year_end=1860)
print(f'  returned {len(hyb)} rows, years: {yrs(hyb)}')
bad2 = [r for r in hyb if not (1850 <= (r.get("publication_year") or 0) <= 1860)]
print(f'  out-of-range hits: {len(bad2)} {"OK" if not bad2 else "LEAK -> BUG"}')

print('\n=== 3. Cache isolation: filtered vs unfiltered same query ===')
f1 = final_search(Q, size=10, year_start=1850, year_end=1860)
f2 = final_search(Q, size=10)
y1, y2 = yrs(f1), yrs(f2)
print(f'  filtered   years: {y1}')
print(f'  unfiltered years: {y2}')
leak = [r for r in f1 if not (1850 <= (r.get("publication_year") or 0) <= 1860)]
print(f'  filter leaked into unfiltered call: {"YES -> BUG" if sorted(y1) == sorted(y2) else "no"}')
print(f'  filtered call contains out-of-range: {len(leak)} {"LEAK -> BUG" if leak else "no"}')

print('\n=== 4. Temporal scoring uses the selected range ===')
t = [r.get('temporal_explanation') for r in f1]
print(f'  temporal explanations in-range: {sorted(set(t))}')

print('\n=== 5. Density artifact ===')
for corpus in ('expanded', 'small'):
    d = decade_density(corpus)
    if d is None:
        print(f'  {corpus}: NOT BUILT')
        continue
    print(f"  {corpus}: range {d['range']['start']}-{d['range']['end']}, "
          f"{len(d['decades'])} bands, {d['total_documents']} docs")
    for b in d['decades']:
        flag = '' if b['complete'] else '  (partial decade)'
        print(f"    {b['label']}  docs={b['documents']:>6}  share={b['share']*100:5.1f}%{flag}")

print('\n=== 6. Density range vs configured range ===')
import yaml
with open('configs/config.yaml', encoding='utf-8') as fh:
    cfg = yaml.safe_load(fh)
d = decade_density('expanded')
cfg_start = cfg['corpus_start_year']
# corpus_end_year is an EXCLUSIVE bound: the filter is
# CORPUS_START_YEAR <= yr < CORPUS_END_YEAR, so a config value of 1870 means
# the corpus covers 1800-1869 inclusive. Compare against end-1, not end.
cfg_end_inclusive = cfg['corpus_end_year'] - 1
print(f"  config says     : {cfg_start}-{cfg['corpus_end_year']} "
      f"(exclusive upper bound; covers {cfg_start}-{cfg_end_inclusive} inclusive)")
print(f"  data actually has: {d['range']['start']}-{d['range']['end']}")
if cfg_start == d['range']['start'] and cfg_end_inclusive == d['range']['end']:
    print('  OK: config matches data exactly')
elif cfg_end_inclusive < d['range']['end']:
    print(f"  WARNING: config stops at {cfg_end_inclusive} but data has documents "
          f"up to {d['range']['end']}")
else:
    print(f"  MISMATCH: config claims coverage to {cfg_end_inclusive} but the last "
          f"document is from {d['range']['end']}. Either the corpus is short or "
          f"the config overstates the range.")
