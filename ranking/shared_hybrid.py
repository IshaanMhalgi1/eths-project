"""
Shared hybrid candidate pool for all rankers.
Ensures consistent candidate pools across ranker calls.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.hybrid import search_hybrid
from retrieval.bm25 import DEFAULT_INDEX

# Fixed candidate pool size for all rankers (matches size * 5 for size=10)
SHARED_FETCH_SIZE = 50

# Cache for hybrid candidates per query (optional optimization)
_hybrid_cache = {}

def get_hybrid_candidates(query: str, fetch_size: int = SHARED_FETCH_SIZE, alpha: float = 0.5,
                          index_name: str = DEFAULT_INDEX, year_start=None, year_end=None):
    """
    Get hybrid candidates from a fixed large pool.
    All rankers should use this to ensure consistent candidate pools.

    The year range is part of the cache key. It has to be: a filtered search
    draws from a different candidate pool than an unfiltered one, and sharing a
    key would let a range-filtered query return unfiltered candidates, which is
    exactly the silent misrepresentation the timeline must avoid.
    """
    # Use cache key
    cache_key = (query, fetch_size, alpha, index_name, year_start, year_end)
    if cache_key in _hybrid_cache:
        return _hybrid_cache[cache_key]

    candidates = search_hybrid(query, size=fetch_size, alpha=alpha,
                               index_name=index_name,
                               year_start=year_start, year_end=year_end)
    _hybrid_cache[cache_key] = candidates
    return candidates

def clear_hybrid_cache():
    """Clear the hybrid candidate cache."""
    global _hybrid_cache
    _hybrid_cache = {}