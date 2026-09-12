"""
Shared hybrid candidate pool for all rankers.
Ensures consistent candidate pools across ranker calls.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.hybrid import search_hybrid

# Fixed candidate pool size for all rankers
SHARED_FETCH_SIZE = 200

# Cache for hybrid candidates per query (optional optimization)
_hybrid_cache = {}

def get_hybrid_candidates(query: str, fetch_size: int = SHARED_FETCH_SIZE, alpha: float = 0.5):
    """
    Get hybrid candidates from a fixed large pool.
    All rankers should use this to ensure consistent candidate pools.
    """
    # Use cache key
    cache_key = (query, fetch_size, alpha)
    if cache_key in _hybrid_cache:
        return _hybrid_cache[cache_key]
    
    candidates = search_hybrid(query, size=fetch_size, alpha=alpha)
    _hybrid_cache[cache_key] = candidates
    return candidates

def clear_hybrid_cache():
    """Clear the hybrid candidate cache."""
    global _hybrid_cache
    _hybrid_cache = {}