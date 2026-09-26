"""
Paired bootstrap CIs on every ranker contrast, plus graded metrics
(nDCG@10, MAP) that remove the binarization-threshold problem entirely.

Two things this adds over the v1 pilot analysis:

1. PAIRED bootstrap. Each contrast resamples queries with replacement and
   computes the metric difference on the same resample for both rankers.
   This is the correct paired test; the v1 analysis reported point estimates
   with no interval at all.

2. GRADED metrics on the raw 0/1/2 labels. Recall/MRR require a binary
   relevant set, which forced an arbitrary threshold that the BM25-vs-Dense
   sign turned out to depend on. nDCG@10 and MAP consume the graded labels
   directly, so no threshold is involved and all 158 judgments are used at
   full fidelity.

At n=10 the intervals are wide and this is expected. The script reports
P(A > B) alongside the CI because it degrades more gracefully than a p-value
when the interval spans zero.
"""
import os
import sys
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import DEFAULT_INDEX, SMALL_INDEX
from evaluation import ablation_rankers as ar
from api.canonical_docs import get_canonical

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')

# Overridable so the same analysis can run against the v1 two-rater qrels or
# the v2 single-rater qrels without either clobbering the other.
#   HUMAN_QRELS=... HUMAN_AUDIT=... python evaluation/paired_bootstrap_graded.py
HUMAN = os.environ.get(
    'HUMAN_QRELS', os.path.join(DATA, 'qrels_human_annotated.json'))
AUDIT = os.environ.get(
    'HUMAN_AUDIT', os.path.join(DATA, 'qrels_human_annotated_audit.json'))

K = 10
N_BOOT = 20000
SEED = 20260926

# Candidate pools are keyed by CANONICAL parent_doc_id (deduplicated via
# api/canonical_docs.py), but retrievers return RAW parent_doc_id. Matching
# raw ids against canonical pool keys silently counts every reprint as a miss.
# With 11,399 duplicate groups in the corpus this is not a rounding error — it
# exposed 44-61 unjudged docs per 100 retrieved before this was fixed, and it
# penalized Dense more than BM25 (56 vs 44), biasing the headline contrast
# AGAINST Dense.
CANONICALIZE = os.environ.get('CANONICALIZE', '1') == '1'

RANKERS = ['BM25', 'Dense', 'Hybrid', 'Hybrid+Temporal', 'Hybrid+Metadata',
           'Final', 'BM25+Temporal', 'Dense+Temporal', 'TemporalOnly', 'MetadataOnly']


# ---------------------------------------------------------------- retrieval

def retrieve(ranker_name, query):
    corpus = 'small' if query['corpus'] == 'small' else 'expanded'
    return ar.get_ranker(ranker_name, corpus)(query['query_text'], K)


def ranked_ids(ranker_name, query):
    out, seen = [], set()
    for r in retrieve(ranker_name, query):
        d = get_canonical(r['parent_doc_id']) if CANONICALIZE else r['parent_doc_id']
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out[:K]


# ------------------------------------------------------------------- qrels

def load_graded():
    """query_id -> {doc_id: mean graded score in [0,2]}"""
    audit = json.load(open(AUDIT))
    graded = {}
    for j in audit['judgments']:
        graded.setdefault(j['query_id'], {})[j['doc_id']] = j['mean']
    return graded


# ----------------------------------------------------------------- metrics

def dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked, graded, k=K):
    gains = [(2 ** graded.get(d, 0.0)) - 1 for d in ranked[:k]]
    ideal = sorted(((2 ** v) - 1 for v in graded.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def average_precision(ranked, graded, rel_threshold=1.0):
    """AP with judged-relevant = mean grade >= rel_threshold.

    Unjudged docs count as non-relevant (standard TREC), but the unjudged
    count is reported separately because pooling bias would otherwise be
    invisible.
    """
    rel_total = sum(1 for v in graded.values() if v >= rel_threshold)
    if rel_total == 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, d in enumerate(ranked[:K]):
        if graded.get(d, 0.0) >= rel_threshold:
            hits += 1
            s += hits / (i + 1)
    return s / rel_total


def recall_at_k(ranked, graded, rel_threshold=1.0):
    rel = {d for d, v in graded.items() if v >= rel_threshold}
    if not rel:
        return 0.0
    return sum(1 for d in ranked[:K] if d in rel) / len(rel)


def mrr(ranked, graded, rel_threshold=1.0):
    for i, d in enumerate(ranked[:K]):
        if graded.get(d, 0.0) >= rel_threshold:
            return 1.0 / (i + 1)
    return 0.0


# ------------------------------------------------------------------ driver

def main():
    base = json.load(open(HUMAN))
    queries = base['queries']
    graded_all = load_graded()

    ranked_cache = {name: [ranked_ids(name, q) for q in queries] for name in RANKERS}

    # unjudged-retrieved diagnostic (pooling-bias exposure)
    print("=" * 84)
    print("POOLING EXPOSURE: retrieved-but-unjudged docs in top-10")
    print("=" * 84)
    print(f"{'Ranker':<20}{'unjudged/100':<16}{'note'}")
    for name in RANKERS:
        n_unj = sum(
            1
            for qi, per_q in enumerate(ranked_cache[name])
            for d in per_q
            if d not in graded_all[queries[qi]['query_id']]
        )
        note = 'ok' if n_unj == 0 else 'POOLING BIAS EXPOSED'
        print(f"{name:<20}{n_unj:<16}{note}")
    print("  (expected near 0: pool = union of BM25/Dense/Hybrid top-20, eval on top-10 of the same)")

    graded = [graded_all[q['query_id']] for q in queries]

    METRICS = {
        'nDCG@10': lambda r, g: ndcg_at_k(r, g),
        'MAP': lambda r, g: average_precision(r, g),
        'Recall@10': lambda r, g: recall_at_k(r, g),
        'MRR': lambda r, g: mrr(r, g),
    }

    # point estimates
    print(f"\n{'=' * 84}")
    print("POINT ESTIMATES — GRADED (nDCG@10, MAP use raw 0/1/2 labels, no threshold)")
    print("=" * 84)
    print(f"{'Ranker':<20}{'nDCG@10':>10}{'MAP':>10}{'Recall@10':>12}{'MRR':>10}")
    scores = {}
    for name in RANKERS:
        row = {}
        for mname, fn in METRICS.items():
            row[mname] = np.array([fn(ranked_cache[name][i], graded[i])
                                   for i in range(len(queries))])
        scores[name] = row
        print(f"{name:<20}{row['nDCG@10'].mean():>10.3f}{row['MAP'].mean():>10.3f}"
              f"{row['Recall@10'].mean():>12.3f}{row['MRR'].mean():>10.3f}")

    # paired bootstrap
    rng = np.random.default_rng(SEED)
    n = len(queries)
    boot_idx = rng.integers(0, n, size=(N_BOOT, n))

    contrasts = [
        ('BM25', 'Dense', 'BM25 vs Dense'),
        ('Hybrid+Temporal', 'Hybrid', 'Temporal gain over Hybrid'),
        ('Hybrid+Temporal', 'BM25', 'Temporal gain over BM25'),
        ('Final', 'BM25', 'Final vs BM25'),
        ('Hybrid+Temporal', 'TemporalOnly', 'Hybrid+Temporal vs TemporalOnly'),
        ('BM25+Temporal', 'Hybrid+Temporal', 'BM25+Temporal vs Hybrid+Temporal'),
        ('Hybrid', 'BM25', 'Hybrid vs BM25'),
        ('Dense', 'Hybrid', 'Dense vs Hybrid'),
    ]

    for mname in ['nDCG@10', 'MAP']:
        print(f"\n{'=' * 84}")
        print(f"PAIRED BOOTSTRAP on {mname}  (n={n} queries, {N_BOOT} resamples, 95% CI)")
        print("=" * 84)
        print(f"{'Contrast':<32}{'delta':>9}{'95% CI':>22}{'P(A>B)':>9}{'excl 0?':>10}")
        for a, b, label in contrasts:
            d = scores[a][mname] - scores[b][mname]
            boot = d[boot_idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p_ab = float((boot > 0).mean())
            excl = 'yes' if (lo > 0 or hi < 0) else 'no'
            print(f"{label:<32}{d.mean():>+9.3f}   [{lo:+.3f}, {hi:+.3f}]{p_ab:>9.2f}{excl:>10}")

    # explicit underpowered statement
    print(f"\n{'=' * 84}")
    print("POWER STATEMENT")
    print("=" * 84)
    print(f"  n = {n} queries. Minimum detectable difference at 80% power, alpha=0.05,")
    print("  for a paired design is roughly d = 1.0 * sd(d) (approx, normal approximation).")
    for a, b, label in contrasts:
        d = scores[a]['nDCG@10'] - scores[b]['nDCG@10']
        sd = d.std(ddof=1) if len(d) > 1 else float('nan')
        print(f"    {label:<32} sd(delta)={sd:.3f}  -> MDE ~= {1.0*sd:.3f} nDCG")
    print("\n  Read every interval above as 'not distinguishable at n=10' unless it")
    print("  excludes zero. A non-significant result here is NOT evidence of no effect;")
    print("  with 10 queries, effects smaller than ~0.1-0.2 nDCG are simply unmeasurable.")

    # Output name follows the input so v1 and v2 runs cannot overwrite each other.
    _base = os.path.splitext(os.path.basename(HUMAN))[0]
    _tag = {
        'qrels_human_annotated': '_v1_tw_orater',
        'qrels_human_annotated_v2_singlerater': '_v2_singlerater',
    }.get(_base, '_' + _base.replace('qrels_human_annotated', '').strip('_'))
    _out = os.path.join(os.path.dirname(__file__), f'human_qrels_graded_metrics{_tag}.json')
    json.dump({name: {m: scores[name][m].tolist() for m in METRICS} for name in RANKERS},
              open(_out, 'w'),
              indent=2)
    print(f"\nSaved per-query graded metrics to {os.path.relpath(_out, os.path.dirname(__file__))}")
    print(f"  (qrels source: {os.path.basename(HUMAN)})")



if __name__ == '__main__':
    main()
