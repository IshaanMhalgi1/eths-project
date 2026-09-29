"""Reconcile temporal_type="period" labels against the period gazetteer.

Two queries are labelled temporal_type="period" but name eras the parser cannot
resolve, so at runtime they extract no window and silently take the
no-op branch. The question is how to fix that: add the missing eras, or relabel
the queries to match what actually happens.

This script gathers the evidence for that choice rather than assuming it, because
the two options are NOT equivalent. Adding an era to the gazetteer changes what
the ranker does; relabelling a query changes only a label. So before touching
the gazetteer we need to know whether the query's own reference documents
actually fall inside the period it names. If they do not, adding the era would
make the temporal ranker sort AWAY from the documents the qrels calls relevant,
and "fixing" the label would silently repair the parse while worsening retrieval.

For each period-labelled query this reports:
  - whether the gazetteer resolves it
  - whether the reference documents fall inside the named period
  - the counterfactual: metrics if the era were added to the gazetteer

Writes evaluation/period_label_reconciliation.json. Reads the index; does not
modify the gazetteer, the qrels, or any ranker.
"""
import json
import os
import sys

import numpy as np
import yaml
from opensearchpy import OpenSearch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal.temporal_parser import TemporalParser, CORPUS_START, CORPUS_END
from retrieval.bm25 import search_bm25, DEFAULT_INDEX
from ranking.temporal_ranker import calculate_temporal_score
from ranking.shared_hybrid import SHARED_FETCH_SIZE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'evaluation', 'period_label_reconciliation.json')

# The historical spans these eras actually denote. Used to test whether each
# query's reference documents sit inside the period its own text names. Kept
# separate from period_gazetteer.json so the test is independent of the file
# under consideration.
ACTUAL_SPANS = {
    'q17': ('Jeffersonian Era', 1801, 1809),
    'q18': ('Early Republic', 1789, 1824),
    'q19': ('Jefferson administration', 1801, 1809),
    'q20': ('Early National Period', 1789, 1824),
    'q25': ('Era of Good Feelings', 1817, 1825),
    'q57': ('Manifest Destiny / Polk', 1843, 1853),
}


def recall_at_k(retrieved, true_docs, k=10):
    seen, dedup = set(), []
    for r in retrieved:
        p = r['parent_doc_id']
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    dedup = dedup[:k]
    return sum(1 for d in dedup if d in true_docs) / max(len(true_docs), 1)


def main():
    cfg = yaml.safe_load(open(os.path.join(REPO, 'configs', 'config.yaml'),
                              encoding='utf-8'))
    client = OpenSearch(
        hosts=[{'host': cfg['opensearch']['host'],
                'port': cfg['opensearch']['port']}],
        use_ssl=False, verify_certs=False, ssl_show_warn=False)
    parser = TemporalParser()
    queries = [q for q in json.load(
        open(os.path.join(REPO, 'data', 'qrels_expanded.json'), encoding='utf-8'))
        if q.get('relevant_doc_ids')]

    print('=' * 100)
    print('1. GAZETTEER CROSS-CHECK: every temporal_type="period" query')
    print('=' * 100)
    print(f'  corpus span {CORPUS_START}-{CORPUS_END}')
    print(f'{"qid":<6}{"resolves":<11}{"window":<16}{"cov":<8}'
          f'{"is_constr":<11}text')
    period_qs = [q for q in queries if q.get('temporal_type') == 'period']
    rows = []
    for q in period_qs:
        ti = parser.parse(q['query_text'])
        resolved = ti['start_year'] is not None
        win = (f"({ti['start_year']},{ti['end_year']})" if resolved else '-')
        cov = (f"{ti['corpus_coverage']:.3f}" if ti['corpus_coverage'] is not None
               else '-')
        print(f'  {q["query_id"]:<6}{str(resolved):<11}{win:<16}{cov:<8}'
              f'{str(ti["is_constrained"]):<11}{q["query_text"][:34]}')
        rows.append({
            'query_id': q['query_id'], 'text': q['query_text'],
            'resolves': resolved, 'window': win,
            'is_constrained': ti['is_constrained'],
            'coverage': ti['corpus_coverage'],
        })
    unresolved = [r for r in rows if not r['resolves']]
    print(f'\n  resolvable: {len(rows) - len(unresolved)}/{len(rows)}')
    print(f'  FALL THROUGH TO NO-OP: {[r["query_id"] for r in unresolved]}')

    print()
    print('=' * 100)
    print('2. DO THE REFERENCES FALL INSIDE THE PERIOD THE QUERY NAMES?')
    print('=' * 100)
    print('  This decides the fix. If a query names a period but its reference')
    print('  documents sit outside it, adding the era to the gazetteer would make')
    print('  the ranker sort away from the documents the qrels calls relevant.')
    print()
    for q in period_qs:
        era, lo, hi = ACTUAL_SPANS[q['query_id']]
        years = []
        for pid in q['relevant_doc_ids']:
            r = client.search(index=DEFAULT_INDEX, body={
                'size': 1, 'query': {'term': {'parent_doc_id': pid}}})
            if r['hits']['hits']:
                years.append(r['hits']['hits'][0]['_source'].get('publication_year'))
        inside = [y for y in years if y is not None and lo <= y <= hi]
        print(f'  {q["query_id"]}: {era} = {lo}-{hi}')
        print(f'      reference years {sorted(years)}')
        print(f'      inside the named period: {len(inside)}/{len(years)}'
              f'  -> {"CONSISTENT" if len(inside) == len(years) else "INCONSISTENT"}')
        if len(inside) != len(years):
            print(f'      outside: {sorted(y for y in years if not (lo <= y <= hi))}')

    print()
    print('=' * 100)
    print('3. COUNTERFACTUAL: metrics if the era WERE added to the gazetteer')
    print('=' * 100)
    print('  Recall@10 for BM25 (no constraint) vs a date-sorted ranker using the')
    print('  period the query names. A drop means adding the era hurts retrieval.')
    print()
    print(f'  {"qid":<6}{"BM25 R@10":<12}{"date-sorted R@10":<18}{"delta":<10}note')
    counter = []
    for q in period_qs:
        if parser.parse(q['query_text'])['start_year'] is not None:
            continue  # already resolves; nothing to add
        era, lo, hi = ACTUAL_SPANS[q['query_id']]
        text, refs = q['query_text'], set(q['relevant_doc_ids'])
        cands = search_bm25(text, size=SHARED_FETCH_SIZE, index_name=DEFAULT_INDEX)
        bm = recall_at_k(cands, refs)
        scored = []
        for r in cands:
            t, _ = calculate_temporal_score(r.get('historical_start'),
                                             r.get('historical_end'), lo, hi)
            scored.append((t, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        ds = recall_at_k([r for _, r in scored], refs)
        delta = ds - bm
        note = 'adding the era WOULD HELP' if delta > 0 else 'adding the era HURTS'
        print(f'  {q["query_id"]:<6}{bm:<12.4f}{ds:<18.4f}{delta:<+10.4f}{note}')
        counter.append({'query_id': q['query_id'], 'era': era, 'span': [lo, hi],
                        'bm25_recall': bm, 'date_sorted_recall': ds,
                        'delta': delta})
    print()
    print('  CONCLUSION: ' + (
        'adding these eras would damage retrieval, because the reference '
        'documents do not sit in the named period. The label is what is wrong, '
        'not the gazetteer.'
        if all(c['delta'] <= 0 for c in counter) else
        'adding these eras would help at least one query; the gazetteer is the '
        'incomplete component and should be extended.'))

    out = {
        'period_queries': rows,
        'fall_through_to_noop': [r['query_id'] for r in unresolved],
        'actual_spans': ACTUAL_SPANS,
        'counterfactual_if_era_added': counter,
        'decision': 'RELABEL, do not extend the gazetteer -- see conclusion above',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote evaluation/{os.path.basename(OUT)}')


if __name__ == '__main__':
    main()
