"""Prefix-matched autocomplete suggestions drawn from the corpus vocabulary.

The index (data/autocomplete_index.json) is built by
evaluation/build_autocomplete_index.py directly from the indexed document text,
so every suggestion corresponds to something genuinely present in the corpus --
this never proposes a query the retrieval system cannot answer.

Lookup is a binary search over alphabetically sorted term and phrase lists, so a
query is microseconds once the index is loaded. The index is loaded lazily on
first use and then held in memory, keeping request latency well under the
200 ms target.
"""

import bisect
import json
import os
import threading

INDEX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'autocomplete_index.json'
)

MIN_PREFIX_CHARS = 2
DEFAULT_LIMIT = 8
MAX_LIMIT = 20

# Ranking weights. An exact match on the whole prefix is what the user most
# likely meant; multi-word completions are preferred slightly over bare terms
# because they make a better query, not just a longer word.
EXACT_MATCH_BOOST = 3.0
PHRASE_BOOST = 1.25

_lock = threading.Lock()
_index = None


def _load():
    """Load and memoise the index. Returns None if it has not been built."""
    global _index
    if _index is not None:
        return _index
    with _lock:
        if _index is not None:
            return _index
        if not os.path.exists(INDEX_PATH):
            return None
        with open(INDEX_PATH, encoding='utf-8') as fh:
            raw = json.load(fh)
        _index = {
            "terms": [t for t, _ in raw.get("terms", [])],
            "term_counts": dict(raw.get("terms", [])),
            "phrases": [p for p, _ in raw.get("phrases", [])],
            "phrase_counts": dict(raw.get("phrases", [])),
            "years": raw.get("years", {}),
            "meta": raw.get("meta", {}),
        }
        return _index


def _prefix_matches(words, counts, prefix, exact_boost, phrase_boost, out):
    """Collect scored candidates whose spelling starts with `prefix`."""
    if not words:
        return
    i = bisect.bisect_left(words, prefix)
    n = 0
    while i < len(words) and words[i].startswith(prefix):
        w = words[i]
        c = counts.get(w, 0)
        score = float(c) * phrase_boost
        if w == prefix:
            score *= exact_boost
        out.append((score, w))
        i += 1
        n += 1
        if n >= 400:   # a pathological prefix should not scan the whole index
            break


def suggest(prefix, limit=DEFAULT_LIMIT):
    """Return ranked completion strings for a partially typed query.

    Handles multi-word input by matching on the final token and prepending the
    already-typed words, so "gold " completes to "gold rush" rather than just
    "rush". Numeric input of three or more characters is matched against the
    years observed in the corpus, ordered numerically.
    """
    idx = _load()
    if idx is None:
        return []

    limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    raw = (prefix or "").strip().lower()
    if len(raw) < MIN_PREFIX_CHARS:
        return []

    # Split into the fixed head and the token being completed.
    parts = raw.split()
    token = parts[-1]
    head = " ".join(parts[:-1])
    # A single trailing character is only meaningful once earlier words narrow
    # the search ("gold r"), so it is allowed when a head is present.
    min_token = 1 if head else MIN_PREFIX_CHARS
    if len(token) < min_token:
        return []

    # Numeric input is a date expression. Years read naturally in ascending
    # order, so they bypass the frequency ranking entirely.
    if token.isdigit() and len(token) >= 3:
        years = sorted(y for y in idx["years"] if y.startswith(token))
        out = [f"{head} {y}" if head else y for y in years[:limit]]
        return out

    candidates = []
    _prefix_matches(idx["terms"], idx["term_counts"], token, EXACT_MATCH_BOOST, 1.0, candidates)
    _prefix_matches(idx["phrases"], idx["phrase_counts"], token, EXACT_MATCH_BOOST, PHRASE_BOOST, candidates)

    # Rank: score first, then shorter completions, then alphabetical, so the
    # ordering is stable and predictable between identical-scoring candidates.
    candidates.sort(key=lambda sc: (-sc[0], len(sc[1]), sc[1]))

    seen = set()
    out = []
    for _, word in candidates:
        full = f"{head} {word}" if head else word
        if full in seen:
            continue
        seen.add(full)
        out.append(full)
        if len(out) >= limit:
            break
    return out


def index_meta():
    idx = _load()
    return idx["meta"] if idx else {"available": False}
