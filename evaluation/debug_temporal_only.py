import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ablation_rankers import get_ranker, rank_temporal_only

# Test TemporalOnly on real queries
queries = [
    "obituaries deaths 1806",
    "real estate land for sale 1807",
    "ship arrivals cargo from Europe 1803",
    "War of 1812 navy battles",
    "Treaty of Ghent peace negotiations 1814",
    "Jackson Bank War veto 1832",
]

for q in queries:
    print("Query:", q)
    res = rank_temporal_only(q, size=5)
    for i, r in enumerate(res[:3]):
        yr = r.get('publication_year')
        start = r.get('historical_start')
        end = r.get('historical_end')
        exp = r.get('temporal_explanation', 'N/A')
        print("  {}. {} (year={}, hist=[{},{}], exp={}): {}...".format(
            i+1, r['parent_doc_id'], yr, start, end, exp, r['text'][:80]))
    print()