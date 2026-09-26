"""
Evaluate rankers against human-annotated qrels.

Each query is evaluated against the index it was POOLED from (small corpus
queries -> ethsearch_chunks_small, everything else -> ethsearch_chunks),
because qrels can only cover documents in the candidate pool.

Also runs a binarization sensitivity analysis, since annotator agreement is
weak (kappa=0.452) and the >=1.0 threshold is a judgment call.
"""
import os
import sys
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ablation_rankers import get_ranker, CORPUS_CONFIG
from retrieval.bm25 import DEFAULT_INDEX, SMALL_INDEX

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
# Overridable so v1 (two-rater) and v2 (single-rater) qrels can both be
# evaluated without either overwriting the other.
QRELS_PATH = os.environ.get(
    'HUMAN_QRELS', os.path.join(DATA, 'qrels_human_annotated.json'))
AUDIT_PATH = os.environ.get(
    'HUMAN_AUDIT', os.path.join(DATA, 'qrels_human_annotated_audit.json'))

RANKERS = ['BM25', 'Dense', 'Hybrid', 'Hybrid+Temporal', 'Hybrid+Metadata',
           'Final', 'BM25+Temporal', 'Dense+Temporal', 'TemporalOnly', 'MetadataOnly']

K = 10


def index_for_query(q):
    return SMALL_INDEX if q['corpus'] == 'small' else DEFAULT_INDEX


def corpus_config_for_query(q):
    return 'small' if q['corpus'] == 'small' else 'expanded'


def ranker_for_query(ranker_name, q):
    """Build a ranker bound to the index this query was pooled from."""
    corpus = corpus_config_for_query(q)
    idx = index_for_query(q)
    from evaluation import ablation_rankers as ar
    fn = ar.get_ranker(ranker_name, corpus)
    return lambda text, size=K: fn(text, size)


def per_query_metrics(ranker_name, queries, k=K):
    out = []
    for q in queries:
        fn = ranker_for_query(ranker_name, q)
        results = fn(q['query_text'], k)
        retrieved = [r['parent_doc_id'] for r in results]
        seen, dedup = set(), []
        for d in retrieved:
            if d not in seen:
                seen.add(d)
                dedup.append(d)
        dedup = dedup[:k]
        true_docs = set(q.get('relevant_doc_ids', []))
        hits = [1 if d in true_docs else 0 for d in dedup]
        mrr = 0.0
        for rank, d in enumerate(dedup):
            if d in true_docs:
                mrr = 1.0 / (rank + 1)
                break
        out.append({
            'query_id': q['query_id'],
            'P@10': sum(hits) / k,
            'Recall@10': sum(hits) / max(len(true_docs), 1) if true_docs else 0.0,
            'MRR': mrr,
            'n_relevant': len(true_docs),
            'n_hits': sum(hits),
        })
    return out


def bootstrap_ci(per_q, metric, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    vals = np.array([q[metric] for q in per_q])
    n = len(vals)
    means = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rebuild_qrels_at_threshold(threshold):
    """Rebuild binarized qrels at a different mean-score threshold."""
    audit = json.load(open(AUDIT_PATH))
    base = json.load(open(QRELS_PATH))
    by_q = {}
    for j in audit['judgments']:
        by_q.setdefault(j['query_id'], []).append(j)
    out = []
    for q in base['queries']:
        rel = sorted(j['doc_id'] for j in by_q[q['query_id']] if j['mean'] >= threshold)
        out.append({**q, 'relevant_doc_ids': rel})
    return out


def main():
    data = json.load(open(QRELS_PATH))
    queries = data['queries']

    small = [q for q in queries if q['corpus'] == 'small']
    expanded = [q for q in queries if q['corpus'] != 'small']

    print("=" * 78)
    print("EVALUATION AGAINST HUMAN-ANNOTATED QRELS")
    print("=" * 78)
    print(f"  n_queries: {len(queries)}  "
          f"(small corpus: {len(small)}, expanded corpus: {len(expanded)})")
    print(f"  n_judgments: {data['n_judgments']}  "
          f"percent_agreement: {data['percent_agreement']}%  "
          f"cohen_kappa: {data['cohen_kappa']}")
    print(f"  resolution: {data['resolution_method']}")

    # Split evaluation: small vs expanded
    for corpus_label, subset in [('SMALL CORPUS (1800-1810)', small),
                                 ('EXPANDED CORPUS (1800-1900)', expanded)]:
        if not subset:
            continue
        print(f"\n{'=' * 78}")
        print(f"{corpus_label}  n={len(subset)} queries")
        print("=" * 78)
        all_pq = {}
        print(f"{'Ranker':<20}{'P@10':>8}{'R@10':>8}{'R@10 95% CI':>22}{'MRR':>8}{'MRR 95% CI':>22}")
        for name in RANKERS:
            pq = per_query_metrics(name, subset)
            all_pq[name] = pq
            p = np.mean([x['P@10'] for x in pq])
            r = np.mean([x['Recall@10'] for x in pq])
            m = np.mean([x['MRR'] for x in pq])
            rlo, rhi = bootstrap_ci(pq, 'Recall@10')
            mlo, mhi = bootstrap_ci(pq, 'MRR')
            print(f"{name:<20}{p:>8.3f}{r:>8.3f}   [{rlo:.3f}, {rhi:.3f}]{m:>8.3f}   [{mlo:.3f}, {mhi:.3f}]")

        # Pooled (all queries)
        print(f"\n  --- POOLED (all {len(queries)} queries) ---")
        print(f"{'Ranker':<20}{'P@10':>8}{'R@10':>8}{'MRR':>8}")
        pooled = {}
        for name in RANKERS:
            pq = per_query_metrics(name, queries)
            pooled[name] = pq
            print(f"{name:<20}"
                  f"{np.mean([x['P@10'] for x in pq]):>8.3f}"
                  f"{np.mean([x['Recall@10'] for x in pq]):>8.3f}"
                  f"{np.mean([x['MRR'] for x in pq]):>8.3f}")

    # Headline claim checks
    print(f"\n{'=' * 78}")
    print("HEADLINE CLAIM DIRECTIONAL CHECK (pooled, n=10)")
    print("=" * 78)
    base_pq = per_query_metrics('BM25', queries)
    base_r = np.mean([x['Recall@10'] for x in base_pq])
    base_m = np.mean([x['MRR'] for x in base_pq])
    dense_pq = per_query_metrics('Dense', queries)
    dense_r = np.mean([x['Recall@10'] for x in dense_pq])
    dense_m = np.mean([x['MRR'] for x in dense_pq])
    hyb_t_pq = per_query_metrics('Hybrid+Temporal', queries)
    hyb_t_r = np.mean([x['Recall@10'] for x in hyb_t_pq])
    hyb_t_m = np.mean([x['MRR'] for x in hyb_t_pq])

    print(f"  BM25 vs Dense:            R@10 {base_r:.3f} vs {dense_r:.3f} "
          f"(delta {base_r-dense_r:+.3f})  MRR {base_m:.3f} vs {dense_m:.3f} "
          f"(delta {base_m-dense_m:+.3f})")
    print(f"    term-overlap qrels:  BM25 >> Dense (0.271 vs 0.071 R@10, "
          f"alpha=1.0 optimal)")
    print(f"    -> human qrels:      "
          f"{'REPLICATES' if base_r > dense_r else 'DOES NOT REPLICATE'} "
          f"(BM25 {'still dominates' if base_r > dense_r else 'no longer dominates'})")

    print(f"\n  Hybrid+Temporal vs BM25:  R@10 {hyb_t_r:.3f} vs {base_r:.3f} "
          f"(delta {hyb_t_r-base_r:+.3f})  MRR {hyb_t_m:.3f} vs {base_m:.3f} "
          f"(delta {hyb_t_m-base_m:+.3f})")
    print(f"    term-overlap qrels:  temporal significantly positive on expanded corpus")
    print(f"    -> human qrels:      "
          f"{'REPLICATES' if hyb_t_r > base_r else 'DOES NOT REPLICATE'}")

    # Per-query hybrid+temporal delta (corpus-span sensitivity)
    print(f"\n  Per-query Hybrid+Temporal minus BM25 (R@10 delta):")
    for pq_b, pq_t, q in zip(base_pq, hyb_t_pq, queries):
        d = pq_t['Recall@10'] - pq_b['Recall@10']
        print(f"    {q['query_id']:>5} ({q['corpus']:<8}) {q['query_text'][:38]:<38} "
              f"BM25={pq_b['Recall@10']:.2f} -> HT={pq_t['Recall@10']:.2f}  {d:+.2f}")

    # Binarization sensitivity
    print(f"\n{'=' * 78}")
    print("BINARIZATION SENSITIVITY (mean-score threshold)")
    print("=" * 78)
    print(f"{'Threshold':<14}{'n_rel_docs':<14}{'BM25 R@10':<14}{'BM25 MRR':<12}"
          f"{'Dense R@10':<14}{'Dense MRR':<12}{'BM25-Dense R@10':<18}")
    for thr in [0.5, 1.0, 1.5, 2.0]:
        qb = rebuild_qrels_at_threshold(thr)
        n_rel = sum(len(q['relevant_doc_ids']) for q in qb)
        pb = per_query_metrics('BM25', qb)
        pd = per_query_metrics('Dense', qb)
        br, dr = np.mean([x['Recall@10'] for x in pb]), np.mean([x['Recall@10'] for x in pd])
        bm, dm = np.mean([x['MRR'] for x in pb]), np.mean([x['MRR'] for x in pd])
        print(f"{thr:<14.1f}{n_rel:<14}{br:<14.3f}{bm:<12.3f}{dr:<14.3f}{dm:<12.3f}{br-dr:<+18.3f}")

    out = {
        'n_queries': len(queries),
        'n_judgments': data['n_judgments'],
        'percent_agreement': data['percent_agreement'],
        'cohen_kappa': data['cohen_kappa'],
        'per_query': {name: per_query_metrics(name, queries) for name in RANKERS},
    }
    p = os.path.join(os.path.dirname(__file__), 'human_qrels_results.json')
    json.dump(out, open(p, 'w'), indent=2)
    print(f"\nSaved per-query results to {p}")


if __name__ == '__main__':
    main()
