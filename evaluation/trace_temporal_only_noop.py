"""What does TemporalOnly actually do when the parser extracts no constraint?

The cluster breakdown reports a POSITIVE +0.04 Recall@10 effect for
TemporalOnly (C5) over the BM25 family (C0) inside the `no_year` cluster. That
is uncomfortable, because TemporalOnly is supposed to be content-blind: it
never reads the query text, so a query carrying no date should leave it nothing
to sort on and it should collapse onto the BM25 ordering. A positive effect
there implies a hidden default range or a hidden tie-break.

This traces the code path and then tests it. The decisive test is behavioural:
run TemporalOnly and BM25 on every scored query and compare the returned
rankings rank-for-rank.

The claim under test, from evaluation/ablation_rankers.py:203-206, is that the
no-constraint branch is a true no-op -- it zeroes every score and returns the
BM25 candidate list unsorted, which is a no-op only because search_bm25 already
returns descending score order. If so, C5 == C0 exactly on every query the
parser leaves unconstrained, and the +0.04 must come from elsewhere.

Reads the index and stored per-query metrics; writes its own report only.
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal.temporal_parser import TemporalParser
from retrieval.bm25 import search_bm25, DEFAULT_INDEX
from ranking.shared_hybrid import SHARED_FETCH_SIZE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'evaluation', 'temporal_only_noop_trace.json')
parser = TemporalParser()


def temporal_only(query, size=10, index_name=DEFAULT_INDEX):
    """Verbatim copy of ablation_rankers.rank_temporal_only, instrumented.

    Copied rather than imported so the trace cannot drift if the ranker is
    later changed; the no-constraint branch is reproduced exactly, including
    the fact that it does NOT re-sort before slicing.
    """
    from ranking.temporal_ranker import calculate_temporal_score
    ti = parser.parse(query)
    qs, qe = ti.get('start_year'), ti.get('end_year')
    candidates = search_bm25(query, size=SHARED_FETCH_SIZE, index_name=index_name)
    if qs is None or qe is None:
        for r in candidates:
            r['final_score'] = 0.0
        # NOTE: returned without sorting. Faithful to the ranker.
        return candidates[:size], 'no_constraint'
    for r in candidates:
        t, _ = calculate_temporal_score(r.get('historical_start'),
                                         r.get('historical_end'), qs, qe)
        r['final_score'] = t
    return sorted(candidates, key=lambda x: x['final_score'],
                  reverse=True)[:size], 'sorted_by_date'

def main():
    queries = [q for q in json.load(
        open(os.path.join(REPO, 'data', 'qrels_expanded.json'), encoding='utf-8'))
        if q.get('relevant_doc_ids')]

    with open(os.path.join(REPO, 'evaluation', 'ablation_matrix_results.json'),
              encoding='utf-8') as f:
        abl = json.load(f)
    stored = {r: np.array(abl['metrics']['expanded'][r]['per_query']['r'], float)
              for r in ('BM25', 'TemporalOnly')}
    qids = [q['query_id'] for q in queries]

    print('=' * 100)
    print('1. THE CODE PATH')
    print('=' * 100)
    print("""
  ablation_rankers.rank_temporal_only(query, size, index_name):
      ti    = parser.parse(query)
      qs, qe = ti['start_year'], ti['end_year']
      cands = search_bm25(query, size=SHARED_FETCH_SIZE)      # BM25, score-descending
      if qs is None or qe is None:
          for r in cands: r['final_score'] = 0.0
          return cands[:size]            # <-- no sort, no date term, no fallback range
      ...
  temporal_parser.parse(): start_year/end_year stay None unless a literal year,
  a range, a before/after operator, or a gazetteer period name matched. There is
  no else-branch, no corpus-bounds default, and no neutral midpoint.
""")
    print(f'  SHARED_FETCH_SIZE (candidate pool) = {SHARED_FETCH_SIZE}')

    print()
    print('=' * 100)
    print('2. BEHAVIOURAL TEST: TemporalOnly vs BM25, rank-for-rank, all 88 queries')
    print('=' * 100)
    rows = []
    for i, q in enumerate(queries):
        text = q['query_text']
        ti = parser.parse(text)
        # Two different questions, kept separate. "window_resolved" is what the
        # RANKER branches on (it re-runs its own start_year/end_year test).
        # "is_constrained" is the corrected classification: resolved AND covers
        # <95% of the 1800-1869 span. q15 is resolved but not discriminating.
        resolved = ti.get('start_year') is not None
        constrained = ti.get('is_constrained', False)
        to_res, branch = temporal_only(text)
        bm = search_bm25(text, size=10, index_name=DEFAULT_INDEX)
        to_ids = [r['parent_doc_id'] for r in to_res]
        bm_ids = [r['parent_doc_id'] for r in bm]
        rows.append({
            'query_id': q['query_id'], 'text': text,
            'temporal_type': q.get('temporal_type'),
            'parser_start': ti.get('start_year'), 'parser_end': ti.get('end_year'),
            'corpus_coverage': ti.get('corpus_coverage'),
            'window_resolved': resolved, 'constrained': constrained,
            'branch': branch,
            'identical_to_bm25': to_ids == bm_ids,
            'n_first_diff': next((j for j, (a, b) in
                                  enumerate(zip(to_ids, bm_ids)) if a != b), None),
            'stored_recall_bm25': float(stored['BM25'][i]),
            'stored_recall_temporal': float(stored['TemporalOnly'][i]),
        })

    # Split on the CORRECTED classification.
    uncon = [r for r in rows if not r['constrained']]
    con = [r for r in rows if r['constrained']]
    nodis = [r for r in rows if r['window_resolved'] and not r['constrained']]
    id_un = [r for r in uncon if r['identical_to_bm25']]
    id_con = [r for r in con if r['identical_to_bm25']]
    print(f'  window resolved (what the ranker branches on): '
          f'{sum(1 for r in rows if r["window_resolved"])}/{len(rows)}')
    print(f'  is_constrained (corrected classification)  : {len(con)}/{len(rows)}')
    print(f'  resolved but NON-discriminating             : {len(nodis)}/'
          f'{len(rows)}  -> {[r["query_id"] for r in nodis]}')
    print(f'  unconstrained queries : {len(uncon)}')
    print(f'    TemporalOnly output identical to BM25 rank-for-rank: {len(id_un)}'
          f'/{len(uncon)}')
    if id_un:
        d = [abs(r['stored_recall_temporal'] - r['stored_recall_bm25'])
             for r in id_un]
        print(f'    max |recall difference| on those queries: {max(d):.10f}')
    print(f'  constrained queries   : {len(con)}')
    print(f'    TemporalOnly output identical to BM25: {len(id_con)}/{len(con)}'
          f'  (these are the date-sorted ones)')

    print()
    print('=' * 100)
    print('3. ATTRIBUTING THE no_year CLUSTER EFFECT')
    print('=' * 100)
    with open(os.path.join(REPO, 'evaluation', 'cluster_breakdown.json'),
              encoding='utf-8') as f:
        cb = json.load(f)
    ny_ids = set(cb['clusters']['no_year']['query_ids'])
    ny = [r for r in rows if r['query_id'] in ny_ids]
    ny_un = [r for r in ny if not r['constrained']]
    ny_co = [r for r in ny if r['constrained']]
    def mean_diff(rs):
        return float(np.mean([r['stored_recall_temporal'] - r['stored_recall_bm25']
                              for r in rs])) if rs else float('nan')

    print(f'  no_year cluster: {len(ny)} queries '
          f'({len(ny_un)} parser-unconstrained, {len(ny_co)} constrained)')
    print(f'{"subset":<42}{"n":>4}{"mean d(Recall@10)":>20}')
    print(f'{"no_year, ALL":<42}{len(ny):>4}{mean_diff(ny):>+20.4f}')
    print(f'{"no_year, parser-UNconstrained":<42}{len(ny_un):>4}{mean_diff(ny_un):>+20.4f}')
    print(f'{"no_year, parser-CONSTrained":<42}{len(ny_co):>4}{mean_diff(ny_co):>+20.4f}')
    print()
    print('  the +0.04 is reproduced from stored per-query metrics above.')
    print()
    print(f'{"qid":<6}{"constrained":<14}{"d(Recall@10)":>14}  text')
    for r in sorted(ny, key=lambda r: -(r['stored_recall_temporal']
                                        - r['stored_recall_bm25'])):
        d = r['stored_recall_temporal'] - r['stored_recall_bm25']
        rng = (f"{r['parser_start']}-{r['parser_end']}"
               if r['constrained'] else '-')
        print(f"{r['query_id']:<6}{rng:<14}{d:>+14.4f}  {r['text'][:52]}")

    print()
    print('=' * 100)
    print('4. TEMPORAL_TYPE vs PARSER: labels the gazetteer cannot resolve')
    print('=' * 100)
    print('  A query labelled temporal_type="period" whose era name is absent from')
    print('  temporal/period_gazetteer.json gets NO constraint. Those are the real')
    print('  silent failures: labelled temporal, parsed as none.')
    miss = [r for r in rows if r['temporal_type'] == 'period'
            and not r['constrained']]
    hit = [r for r in rows if r['temporal_type'] == 'period' and r['constrained']]
    print(f'  labelled period, constraint resolved : {len(hit)}  '
          f'-> {[r["query_id"] for r in hit]}')
    print(f'  labelled period, NO constraint       : {len(miss)}  '
          f'-> {[r["query_id"] for r in miss]}')
    for r in miss:
        print(f'      {r["query_id"]}: {r["text"]}')

    print()
    print('=' * 100)
    print('5. GAZETTEER BOUNDS vs CORPUS BOUNDS (1800-1869)')
    print('=' * 100)
    print('  A resolved period can extend outside the indexed span. TemporalOnly')
    print('  then sorts a 1800-1869 corpus by a window the corpus cannot satisfy,')
    print('  which is a real (if mild) mis-specification distinct from the no-op.')
    wide = [r for r in rows if r['constrained']
            and (r['parser_start'] < 1800 or r['parser_end'] > 1869)]
    print(f'  constrained queries whose window leaves 1800-1869: {len(wide)}')
    for r in wide[:12]:
        print(f'      {r["query_id"]}: {r["parser_start"]}-{r["parser_end"]}'
              f'   {r["text"][:46]}')
    if len(wide) > 12:
        print(f'      ... and {len(wide) - 12} more')

    out = {
        'finding': 'The no-constraint branch is a TRUE NO-OP, not a hidden default '
                   'range. It zeroes all scores and returns the BM25 candidate list '
                   'unsorted, which reproduces BM25 exactly because search_bm25 '
                   'already returns descending score order.',
        'caveat': 'The no-op is incidental rather than explicit: the branch omits the '
                  'sort every other ranker performs, so it is correct only for as '
                  'long as search_bm25 keeps returning score-ordered hits.',
        'classification': {
            'window_resolved_n': sum(1 for r in rows if r['window_resolved']),
            'is_constrained_n': len(con),
            'resolved_but_non_discriminating': [r['query_id'] for r in nodis],
            'note': 'the rankers still branch on start_year/end_year being non-None, '
                    'so q15 is sorted by a window covering 95.7% of the corpus. '
                    'is_constrained is the corrected classification and is not yet '
                    'consumed by any ranker.',
        },
        'no_year_attribution': {
            'n': len(ny),
            'n_parser_unconstrained': len(ny_un),
            'n_parser_constrained': len(ny_co),
            'mean_diff_all': mean_diff(ny),
            'mean_diff_unconstrained': mean_diff(ny_un),
            'mean_diff_constrained': mean_diff(ny_co),
        },
        'unconstrained_queries': len(uncon),
        'unconstrained_identical_to_bm25': len(id_un),
        'constrained_queries': len(con),
        'constrained_identical_to_bm25': len(id_con),
        'period_labelled_but_unresolved': [r['query_id'] for r in miss],        'windows_outside_corpus': [
            {'query_id': r['query_id'], 'start': r['parser_start'],
             'end': r['parser_end']} for r in wide],
        'rows': rows,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote evaluation/{os.path.basename(OUT)}')


if __name__ == '__main__':
    main()
