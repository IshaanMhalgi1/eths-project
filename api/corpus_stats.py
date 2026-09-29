"""Corpus density statistics for the timeline visualisation.

Counts come from data/corpus_decade_density.json, built by
evaluation/build_decade_density.py. Nothing here aggregates over the index at
request time: the whole point of the background band is that it is cheap,
stable context rather than another expensive query competing with the search
it is meant to frame.

The axis range is taken from the years actually present in the corpus, not from
the configured bounds in configs/config.yaml. Those disagree -- the config
claims a 1800-1900 window while the indexed data stops at 1869 -- and an axis
stretched to 1900 would render three decades of empty space as though it were
measured emptiness rather than simply uncollected years.

The small corpus is a strict year window over the same counts, so its figures
are derived by summation rather than stored twice.
"""

import json
import os
import threading

DENSITY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data',
    'corpus_decade_density.json'
)

# Named corpora the retrieval layer can be pointed at. Each is a year window
# over the same source counts.
SMALL = 'small'
EXPANDED = 'expanded'

_lock = threading.Lock()
_index = None


def _load():
    """Load and memoise the density index. Returns None if it has not been built."""
    global _index
    if _index is not None:
        return _index
    with _lock:
        if _index is not None:
            return _index
        if not os.path.exists(DENSITY_PATH):
            return None
        with open(DENSITY_PATH, encoding='utf-8') as fh:
            raw = json.load(fh)
        _index = {
            'year_counts': {int(k): v for k, v in raw.get('year_counts', {}).items()},
            'year_chunk_counts': {
                int(k): v for k, v in raw.get('year_chunk_counts', {}).items()
            },
            'meta': raw.get('meta', {}),
        }
        return _index


def _window(corpus):
    """Resolve a corpus name to an inclusive year window, or None for 'all'."""
    idx = _load()
    if idx is None:
        return None
    if corpus == SMALL:
        r = idx['meta'].get('small_corpus_range') or {}
        start, end = r.get('start'), r.get('end')
        if start is None or end is None:
            return None
        # The small corpus is 1800 <= year < 1810, so the window is half-open.
        return (start, end - 1)
    return None


def decade_density(corpus=EXPANDED):
    """Decade-banded document and chunk counts for the requested corpus.

    Returns a payload whose range reflects the years actually observed inside
    the window, so a corpus that stops at 1869 gets an axis that stops at 1869.
    """
    idx = _load()
    if idx is None:
        return None

    window = _window(corpus)

    if window is None:
        years = sorted(idx['year_counts'])
    else:
        lo, hi = window
        years = [y for y in sorted(idx['year_counts']) if lo <= y <= hi]

    if not years:
        return None

    decades = []
    for y in years:
        dec = (y // 10) * 10
        if not decades or decades[-1]['decade'] != dec:
            decades.append({
                'decade': dec,
                'label': f'{dec}s',
                'documents': 0,
                'chunks': 0,
                'years_present': 0,
            })
        decades[-1]['documents'] += idx['year_counts'][y]
        decades[-1]['chunks'] += idx['year_chunk_counts'].get(y, 0)
        decades[-1]['years_present'] += 1

    total = sum(d['documents'] for d in decades)
    for d in decades:
        d['share'] = round(d['documents'] / total, 6) if total else 0.0
        # A band that spans years the corpus does not cover would render its
        # gaps as measured emptiness; this says so explicitly instead.
        d['complete'] = d['years_present'] == 10

    return {
        'corpus': corpus if corpus == SMALL else EXPANDED,
        'band': 'decade',
        'range': {'start': years[0], 'end': years[-1]},
        'decades': decades,
        'total_documents': total,
        'max_documents': max(d['documents'] for d in decades),
        'meta': idx['meta'],
    }
