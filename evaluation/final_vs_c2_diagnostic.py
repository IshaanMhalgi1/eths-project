"""Why is `Final` a separate equivalence class from C2 on the expanded corpus?

C2 = {BM25+Temporal, Hybrid+Temporal} = {BM25+Temporal, Hybrid+Temporal}
C3 = {Final}

Both blend the same three signals, but with DIFFERENT WEIGHTS (expanded config):

  Hybrid+Temporal : 0.7*hybrid + 0.5*temporal          (beta_temporal = 0.5)
  Final           : 0.7*hybrid + 0.2*temporal + 0.1*meta

Two candidate causes for the behavioural difference:
  (a) the metadata term (the report's standing claim: "metadata is inert")
  (b) the blend weights -- the temporal share of the blend

This script separates them:

  VAR-0   Final as configured                       (0.7, 0.2, 0.1)
  VAR-A   metadata weight 0, remaining renormalised (0.7/0.9, 0.2/0.9) = (0.7778, 0.2222)
  VAR-B   control: weights matched to C2 exactly    (0.7, 0.5, 0.0)

If VAR-A still differs from C2 but VAR-B matches it, the metadata term is NOT the
cause and the blend weights are. Also reports, per query, whether the metadata
raw score actually varies across the candidate pool (i.e. whether metadata is
truly constant), and the per-query effect of renormalising.

Analysis only: reuses the existing rankers and index, writes no ranker state.
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ablation_rankers import (
    rank_hybrid_temporal, rank_bm25_temporal, parser, NORM_STATS,
    z_score_normalize, get_ranker)
from retrieval.hybrid import search_hybrid
from ranking.shared_hybrid import clear_hybrid_cache, SHARED_FETCH_SIZE
from retrieval.bm25 import search_bm25
from ranking.temporal_ranker import calculate_temporal_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'evaluation', 'final_vs_c2_diagnostic.json')
QRELS = os.path.join(REPO, 'data', 'qrels_expanded.json')
INDEX = 'ethsearch_chunks'          # DEFAULT_INDEX

W_H, W_T, W_M = 0.7, 0.2, 0.1
# VAR-A: metadata weight 0, remaining renormalised to sum 1
_A = (W_H / (W_H + W_T), W_T / (W_H + W_T))
# VAR-B: match C2's (0.7, 0.5) exactly, metadata off
VAR_B = (0.7, 0.5, 0.0)


def load_queries():
    with open(QRELS, encoding='utf-8') as f:
        return [q for q in json.load(f) if q.get('relevant_doc_ids')]


def final_variant(query, size, w_h, w_t, w_m):
    """rank_final with explicit blend weights (same code path, params exposed).

    alpha_rrf=1.0 is the EXPANDED-corpus value from CORPUS_CONFIG; calling the
    rank_* functions directly would silently use the function defaults
    (beta=0.3, alpha_rrf=0.5) instead, which is a different system.
    """
    ti = parser.parse(query)
    qs, qe = ti.get('start_year'), ti.get('end_year')
    meta_intent = {"periods": ti.get('periods', []), "locations": []}
    clear_hybrid_cache()
    res = search_hybrid(query, size=SHARED_FETCH_SIZE, alpha=1.0, index_name=INDEX)
    for r in res:
        ds, de = r.get('historical_start'), r.get('historical_end')
        r['raw_temporal_score'] = 0.5 if (qs is None or qe is None) else \
            calculate_temporal_score(ds, de, qs, qe)[0]
        dp, dl = r.get('historical_period'), r.get('location')
        m = 0.0
        if meta_intent['periods'] and dp:
            if any(p.lower() in dp.lower() for p in meta_intent['periods']):
                m += 0.5
        if meta_intent['locations'] and dl:
            if any(l.lower() in dl.lower() for l in meta_intent['locations']):
                m += 0.5
        if not meta_intent['periods'] and not meta_intent['locations']:
            m = 0.5
        r['raw_metadata_score'] = m
    hstd = NORM_STATS['dense_std']
    for r in res:
        nt = z_score_normalize(r['raw_temporal_score'], NORM_STATS['temporal_mean'],
                               NORM_STATS['temporal_std'])
        nm = z_score_normalize(r['raw_metadata_score'], NORM_STATS['metadata_mean'],
                               NORM_STATS['metadata_std'])
        r['final_score'] = w_h * r['score'] + w_t * (nt * hstd) + w_m * (nm * hstd)
    return [r['parent_doc_id'] for r in
            sorted(res, key=lambda x: x['final_score'], reverse=True)[:size]]


def metadata_profile(query):
    """Distribution of the raw metadata score across this query's candidate pool.

    Returns (varies, sorted_distinct_values, parsed_periods, n_docs, n_matched).
    """
    ti = parser.parse(query)
    meta_intent = {"periods": ti.get('periods', []), "locations": []}
    clear_hybrid_cache()
    res = search_hybrid(query, size=SHARED_FETCH_SIZE, alpha=1.0, index_name=INDEX)
    vals, matched, with_field = [], 0, 0
    for r in res:
        dp = r.get('historical_period')
        if dp:
            with_field += 1
        m = 0.0
        if meta_intent['periods'] and dp:
            if any(p.lower() in dp.lower() for p in meta_intent['periods']):
                m += 0.5
        if not meta_intent['periods']:
            m = 0.5
        vals.append(m)
        if m == 0.5:
            matched += 1
    return len(set(vals)) > 1, sorted(set(vals)), list(meta_intent['periods']), \
        len(vals), matched, with_field


def main():
    qs = load_queries()
    n = len(qs)
    print(f"expanded queries: {n}")
    print(f"VAR-0 Final  = ({W_H}, {W_T}, {W_M})")
    print(f"VAR-A no-meta= ({_A[0]:.4f}, {_A[1]:.4f})   [0.7,0.2 renormalised]")
    print(f"VAR-B C2-matc= ({VAR_B[0]}, {VAR_B[1]}, {VAR_B[2]})  [control]")

    # Baselines MUST go through get_ranker so CORPUS_CONFIG['expanded'] is
    # applied (beta_temporal=0.5, alpha_rrf=1.0). Calling the rank_* functions
    # directly would use the function defaults and silently compare different
    # systems.
    ht = get_ranker('Hybrid+Temporal', corpus='expanded')
    bt = get_ranker('BM25+Temporal', corpus='expanded')
    c2_ht = [[r['parent_doc_id'] for r in ht(q['query_text'], size=SHARED_FETCH_SIZE)]
             for q in qs]
    c2_bt = [[r['parent_doc_id'] for r in bt(q['query_text'], size=SHARED_FETCH_SIZE)]
             for q in qs]
    v0 = [final_variant(q['query_text'], SHARED_FETCH_SIZE, W_H, W_T, W_M) for q in qs]
    vA = [final_variant(q['query_text'], SHARED_FETCH_SIZE, _A[0], _A[1], 0.0) for q in qs]
    vB = [final_variant(q['query_text'], SHARED_FETCH_SIZE, *VAR_B) for q in qs]

    def ident(a, b):
        return sum(1 for x, y in zip(a, b) if x == y)

    def top10(a, b):
        return sum(1 for x, y in zip(a, b) if x[:10] == y[:10])

    print("\n=== full top-50 identity vs C2 (Hybrid+Temporal) ===")
    for name, v in (("VAR-0 Final(0.7,0.2,0.1)", v0),
                    ("VAR-A no-meta(0.7778,0.2222)", vA),
                    ("VAR-B match-C2(0.7,0.5,0.0)", vB)):
        print(f"  {name:<30} identical top-50: {ident(v,c2_ht)}/{n}   top-10: {top10(v,c2_ht)}/{n}")
    print(f"  {'C2 internal (BM25+T vs HT)':<30} identical top-50: {ident(c2_bt,c2_ht)}/{n}")

    print("\n=== is metadata actually constant? (per query) ===")
    varying, const, period_q = [], [], []
    for q in qs:
        v, vals, periods, ndoc, nmatch, withfield = metadata_profile(q['query_text'])
        rec = (q['query_id'], vals, periods, ndoc, nmatch, withfield,
               q.get('temporal_type'))
        (varying if v else const).append(rec)
        if periods:
            period_q.append(rec)
    print(f"  metadata CONSTANT across pool : {len(const)}/{n} queries")
    print(f"  metadata VARIES across pool   : {len(varying)}/{n} queries")
    print(f"  parser extracted 'periods' on : {len(period_q)}/{n} queries")
    if period_q:
        print("\n  the 6 'period'-type queries (where metadata is NOT the 0.5 default):")
        for qid, vals, periods, ndoc, nmatch, withfield, tt in period_q:
            print(f"    {qid:<5} type={tt:<10} periods={str(periods):<26} "
                  f"docs={ndoc}  matched={nmatch}  docs_with_period_field={withfield}  "
                  f"raw values={vals}")
    if varying:
        print("\n  metadata VARIES:")
        for r in varying:
            print("   ", r)
    else:
        print("\n  -> metadata NEVER varies within a query's candidate pool.")
    types = {}
    for q in qs:
        ti = parser.parse(q['query_text'])
        t = 'none' if not ti.get('periods') else 'periods'
        types[t] = types.get(t, 0) + 1
    print(f"\n  parser 'periods' extracted: {types}")

    out = {
        'n_queries': n,
        'variants': {
            'VAR-0_final_config': {'weights': [W_H, W_T, W_M]},
            'VAR-A_metadata_zero_renormalised': {'weights': [round(_A[0], 6), round(_A[1], 6), 0.0]},
            'VAR-B_match_C2': {'weights': list(VAR_B)},
        },
        'identity_vs_hybrid_temporal': {
            'VAR-0': {'top50': ident(v0, c2_ht), 'top10': top10(v0, c2_ht), 'n': n},
            'VAR-A': {'top50': ident(vA, c2_ht), 'top10': top10(vA, c2_ht), 'n': n},
            'VAR-B': {'top50': ident(vB, c2_ht), 'top10': top10(vB, c2_ht), 'n': n},
        },
        'c2_internal_identity': {'top50': ident(c2_bt, c2_ht), 'n': n},
        'metadata_constant_queries': len(const),
        'metadata_varies_queries': len(varying),
        'queries_with_parsed_periods': [
            {'query_id': qid, 'parsed_periods': periods, 'n_docs': ndoc,
             'n_matched': nmatch, 'docs_with_period_field': withfield,
             'raw_values': vals, 'temporal_type': tt}
            for qid, vals, periods, ndoc, nmatch, withfield, tt in period_q],
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote evaluation/{os.path.basename(OUT)}")


if __name__ == '__main__':
    main()
