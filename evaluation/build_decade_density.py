"""Precompute per-year and per-decade corpus document density.

The timeline visualisation needs to answer "how many documents does the corpus
hold in each decade?" alongside "where did this search's results land?". Those
are two different questions, and answering the first by aggregating over the
index on every request would make a context bar outnumber the thing it is
supposed to contextualise -- both in cost and in visual weight. So the counts
are computed once, here, and read from data/corpus_decade_density.json.

Counts are per unique parent_doc_id rather than per chunk, because the timeline
compares against *documents*; a 40-chunk paper and a 1-chunk paper are one
document each. Chunk counts are kept alongside, since the retrieval system
actually scores chunks, and the two together make the bar honest.

Per-year counts are stored rather than only per-decade so that any coarser or
finer axis can be derived by summation -- in particular the small 1800-1810
corpus, whose single decade is a strict subset of the full range's 1800s band.

Usage: python evaluation/build_decade_density.py
"""

import json
import os
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data'
)
SOURCE = os.path.join(DATA, 'chunks.jsonl')
OUT = os.path.join(DATA, 'corpus_decade_density.json')

# The small corpus is a strict year window of the same source file, recorded
# here so the API can report an axis matching whichever corpus is active rather
# than assuming 1800-1900 everywhere.
SMALL_CORPUS_RANGE = (1800, 1810)


def main():
    if not os.path.exists(SOURCE):
        raise SystemExit(f'missing source: {SOURCE}')

    # year -> set(parent_doc_id), so a document spanning several chunks is
    # counted once. Documents with no usable year are counted separately and
    # excluded from the axis, since they cannot be placed on a time scale.
    docs_by_year = defaultdict(set)
    chunks_by_year = defaultdict(int)
    undated_documents = set()
    total_chunks = 0

    with open(SOURCE, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total_chunks += 1
            pid = rec.get('parent_doc_id')
            year = rec.get('publication_year')

            if year is None:
                # Fall back to the historical window only when the document
                # points at a single year, so an undated-but-scoped document
                # is still placeable without inventing a publication date.
                hs, he = rec.get('historical_start'), rec.get('historical_end')
                if hs is not None and hs == he:
                    year = hs

            if year is None or not isinstance(year, int):
                if pid:
                    undated_documents.add(pid)
                continue

            chunks_by_year[year] += 1
            if pid:
                docs_by_year[year].add(pid)

    years = sorted(docs_by_year)
    if not years:
        raise SystemExit('no dated documents found; refusing to write an empty index')

    # A document can carry several chunks, so union across years to get the
    # true distinct-document total for the corpus.
    all_docs = set()
    for s in docs_by_year.values():
        all_docs |= s

    year_counts = {str(y): len(docs_by_year[y]) for y in years}
    year_chunk_counts = {str(y): chunks_by_year[y] for y in years}

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
        decades[-1]['documents'] += len(docs_by_year[y])
        decades[-1]['chunks'] += chunks_by_year[y]
        decades[-1]['years_present'] += 1

    # Shares are of dated documents only; undated ones are reported separately
    # so the denominator is never quietly wrong.
    dated_docs = len(all_docs)
    for d in decades:
        d['share'] = round(d['documents'] / dated_docs, 6) if dated_docs else 0.0

    payload = {
        'year_counts': year_counts,
        'year_chunk_counts': year_chunk_counts,
        'decades': decades,
        'range': {'start': years[0], 'end': years[-1]},
        'meta': {
            'built_from': os.path.basename(SOURCE),
            'total_chunks': total_chunks,
            'dated_documents': dated_docs,
            'undated_documents': len(undated_documents),
            'years_present': len(years),
            'small_corpus_range': {'start': SMALL_CORPUS_RANGE[0], 'end': SMALL_CORPUS_RANGE[1]},
        },
    }

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)

    print(f'wrote {OUT}')
    print(f"  dated documents   {dated_docs} (undated: {len(undated_documents)})")
    print(f"  chunks            {total_chunks}")
    print(f"  years present     {len(years)}  ({years[0]}-{years[-1]})")
    print('  decade distribution (documents / share):')
    for d in decades:
        print(f"    {d['label']}  {d['documents']:>6}  {d['share'] * 100:5.1f}%")


if __name__ == '__main__':
    main()
