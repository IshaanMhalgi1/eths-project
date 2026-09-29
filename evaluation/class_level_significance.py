"""Class-level significance with clustered sensitivity.

Stage 1 (equivalence_classes.py) collapsed the 10 rankers into behavioural
equivalence classes per corpus. This script runs the statistical analysis at the
CLASS level, so every comparison is between genuinely distinct systems.

Three tests are run on every class-pair contrast, all on the same class
representatives:

  1. Paired bootstrap on per-query differences (the same procedure as
     paired_bootstrap_significance.py). 50,000 resamples, one shared resample
     matrix, two-sided with the +1 correction. This treats queries as
     independent.
  2. Cluster bootstrap: resample QUERY DECADES with replacement (clusters = the
     decade the query text names, plus a 'no_year' bucket for year-less
     queries) and recompute the paired mean difference. This relaxes the
     independence assumption across queries in the same decade -- queries about
     the same decade share topical vocabulary and, critically here, share the
     qrels date-alignment artefact, so a naive query bootstrap can understate
     uncertainty when the effect is decade-clustered.
  3. Paired sign test: on the non-tied per-query differences, two-sided exact
     binomial under H0 that positive and negative are equally likely. This is
     distribution-free and driven only by the per-query SIGN of the difference,
     so it is immune to the magnitude outliers that the mean-based tests are
     sensitive to.

Bonferroni is applied over the number of DISTINCT class-pair contrasts per
corpus, C(k,2), not over the 45 raw ranker pairs. This removes the duplicate
tests of behaviourally identical rankers, which is the whole point of the
class-level view.

Every contrast reports the effect size (mean paired difference) with a 95% CI
for both the query-level and cluster-level bootstrap, plus the exact sign-test
p, and a flag if the three tests disagree on whether the contrast survives
Bonferroni.

Analysis only: reads stored per-query metrics and the class membership. No
ranker, corpus, or index is touched.
"""
import json
import math
import os
import re
import statistics as st
from itertools import combinations
from math import comb

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATION = os.path.join(REPO, 'evaluation', 'ablation_matrix_results.json')
CLASSES = os.path.join(REPO, 'evaluation', 'equivalence_classes.json')
QRELS = {"small": "data/qrels_small.json", "expanded": "data/qrels_expanded.json"}
OUT = os.path.join(REPO, 'evaluation', 'class_level_significance.json')

N_BOOT = 50_000
SEED = 20260928
STRICT_ALPHA = 0.05 / 90          # the ORIGINAL over-penalising threshold, for reference


def decade_of(qtext):
    """Decade the query text names; 'no_year' if it names none in 1800-1869."""
    ys = [int(y) for y in re.findall(r"\b(1[0-8][0-9]{2})\b", qtext)]
    ys = [y for y in ys if 1800 <= y <= 1869]
    if not ys:
        return "no_year"
    return f"{(int(st.median(ys)) // 10) * 10}s"


def paired_bootstrap(d, idx):
    """(p, ci_lo, ci_hi) for mean(d) under a shared query-resample matrix."""
    boot = d[idx].mean(axis=1)
    n_neg, n_pos = int((boot <= 0).sum()), int((boot >= 0).sum())
    p = 2.0 * min((n_neg + 1) / (len(idx) + 1), (n_pos + 1) / (len(idx) + 1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return min(p, 1.0), float(lo), float(hi)


def cluster_bootstrap(d, cluster_ids, n_clusters, rng, reps=N_BOOT):
    """Resample clusters (decades) with replacement; return (p, ci_lo, ci_hi).

    The p-value tests the NULL mean difference of 0 (fraction of resampled mean
    differences that fall on the wrong side of zero), exactly as the query-level
    paired bootstrap does. It must NOT compare against the observed mean: a
    bootstrap distribution is centred on the observed statistic by construction,
    so testing against `obs` returns p ~ 0.5 (=> 1.0 two-sided) for essentially
    every contrast regardless of effect. The percentile CI is reported alongside
    and is consistent with this null-based p.
    """
    idx_by_cluster = [np.where(cluster_ids == c)[0] for c in range(n_clusters)]
    means = np.empty(reps)
    for b in range(reps):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        rows = np.concatenate([idx_by_cluster[j] for j in pick])
        means[b] = d[rows].mean()
    n_neg = int((means <= 0).sum())
    n_pos = int((means >= 0).sum())
    p = 2.0 * min((n_neg + 1) / (reps + 1), (n_pos + 1) / (reps + 1))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return min(p, 1.0), float(lo), float(hi)


def sign_test_p(d):
    """Two-sided exact binomial sign test on the non-zero differences."""
    nz = d[d != 0]
    m = len(nz)
    if m == 0:
        return 1.0, 0, 0
    k = int((nz > 0).sum())
    lo = min(k, m - k)
    # P(X <= lo) under Binom(m, 0.5), doubled and capped
    tail = sum(comb(m, i) for i in range(lo + 1)) / (2.0 ** m)
    return min(2.0 * tail, 1.0), k, m


def main():
    with open(ABLATION, encoding='utf-8') as f:
        ablation = json.load(f)
    with open(CLASSES, encoding='utf-8') as f:
        classes = json.load(f)

    out = {'n_resamples': N_BOOT, 'seed': SEED,
           'bonferroni': 'over C(k,2) class-pair contrasts per corpus',
           'strict_alpha_reference': STRICT_ALPHA, 'corpora': {}}

    for corpus in ('small', 'expanded'):
        with open(os.path.join(REPO, QRELS[corpus]), encoding='utf-8') as f:
            queries = [q for q in json.load(f) if q.get('relevant_doc_ids')]
        n = len(queries)
        metrics = ablation['metrics'][corpus]

        cls = classes['corpora'][corpus]
        reps = [c['representative'] for c in cls['classes']]
        k = len(reps)
        n_pairs = k * (k - 1) // 2
        alpha = 0.05 / n_pairs

        # cluster ids for the cluster bootstrap
        dec_labels = [decade_of(q['query_text']) for q in queries]
        uniq = sorted(set(dec_labels))
        cmap = {lab: i for i, lab in enumerate(uniq)}
        cluster_ids = np.array([cmap[l] for l in dec_labels])
        n_clusters = len(uniq)
        sizes = {lab: int((cluster_ids == i).sum()) for i, lab in enumerate(uniq)}

        print(f"\n{'='*84}\n{corpus.upper()}  n={n} queries, {k} classes, "
              f"{n_pairs} class-pair contrasts, Bonferroni alpha={alpha:.6f}")
        print(f"  clusters (query decade): {n_clusters} -> {sizes}")
        min_cluster = min(sizes.values())
        degenerate = n_clusters < 5 or min_cluster < 3
        if degenerate:
            print(f"  !! DEGENERATE CLUSTERING: only {n_clusters} clusters, smallest "
                  f"has {min_cluster} quer{'y' if min_cluster == 1 else 'ies'}. "
                  f"The cluster bootstrap is coarse and its attainable p-values are "
                  f"quantised; treat it as a weak lower bound on uncertainty, not a "
                  f"precise test. The sign test and query bootstrap are the "
                  f"informative checks on this corpus.")
        print(f"  class representatives : {reps}")

        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n, size=(N_BOOT, n))

        rows = []
        for ra, rb in combinations(reps, 2):
            row = {'a': ra, 'b': rb, 'metrics': {}}
            for mname, key in (('recall', 'r'), ('mrr', 'mrr')):
                d = (np.array(metrics[ra]['per_query'][key], float)
                     - np.array(metrics[rb]['per_query'][key], float))
                delta = float(d.mean())
                # Report delta in the direction of the winner (win - los >= 0)
                # and record the sign separately, so the printed "win > los,
                # d=+..." can never be read against its own sign.
                better = ra if delta > 0 else (rb if delta < 0 else None)
                delta_pos = abs(delta)
                p_q, lo_q, hi_q = paired_bootstrap(d, idx)
                p_c, lo_c, hi_c = cluster_bootstrap(d, cluster_ids, n_clusters, rng)
                p_s, k_pos, m_nz = sign_test_p(d)
                # Orient the CIs in the direction of the reported effect
                # (win - los), so a reader never sees d=+x with a negative CI.
                # Done AFTER the bootstrap calls, which is where lo/hi are set.
                if delta < 0:
                    lo_q, hi_q = -hi_q, -lo_q
                    lo_c, hi_c = -hi_c, -lo_c
                surv_q = p_q < alpha
                surv_c = p_c < alpha
                surv_s = p_s < alpha
                agree = surv_q == surv_c == surv_s
                row['metrics'][mname] = {
                    'delta': delta,
                    'better': better,
                    'effect': delta_pos,
                    'ci95_query_lo': lo_q, 'ci95_query_hi': hi_q, 'p_query': p_q,
                    'ci95_cluster_lo': lo_c, 'ci95_cluster_hi': hi_c, 'p_cluster': p_c,
                    'sign_pos': k_pos, 'sign_n': m_nz, 'p_sign': p_s,
                    'survive_query': surv_q, 'survive_cluster': surv_c,
                    'survive_sign': surv_s, 'survive_strict_query': bool(p_q < STRICT_ALPHA),
                    'conclusion_changes': not agree,
                }
            rows.append(row)

        changes_recall = [r for r in rows
                          if r['metrics']['recall']['conclusion_changes']]
        changes_mrr = [r for r in rows if r['metrics']['mrr']['conclusion_changes']]

        out['corpora'][corpus] = {
            'n_queries': n, 'n_classes': k, 'n_class_pairs': n_pairs,
            'bonferroni_alpha': alpha,
            'classes': classes['corpora'][corpus]['classes'],
            'cluster_definition': {'unit': 'query decade (median year in query text)',
                                   'labels': uniq, 'sizes': sizes,
                                   'n_clusters': n_clusters,
                                   'min_cluster_size': min_cluster,
                                   'degenerate': bool(degenerate)},
            'contrasts': rows,
            'n_survive_recall_query': sum(r['metrics']['recall']['survive_query'] for r in rows),
            'n_survive_recall_cluster': sum(r['metrics']['recall']['survive_cluster'] for r in rows),
            'n_survive_recall_sign': sum(r['metrics']['recall']['survive_sign'] for r in rows),
            'n_survive_recall_strict': sum(r['metrics']['recall']['survive_strict_query'] for r in rows),
            'n_changes_recall': len(changes_recall),
            'n_survive_mrr_query': sum(r['metrics']['mrr']['survive_query'] for r in rows),
            'n_survive_mrr_cluster': sum(r['metrics']['mrr']['survive_cluster'] for r in rows),
            'n_survive_mrr_sign': sum(r['metrics']['mrr']['survive_sign'] for r in rows),
            'n_changes_mrr': len(changes_mrr),
            'changes_recall': [{'a': r['a'], 'b': r['b'],
                                'query': r['metrics']['recall']['survive_query'],
                                'cluster': r['metrics']['recall']['survive_cluster'],
                                'sign': r['metrics']['recall']['survive_sign']}
                               for r in changes_recall],
        }

        # ---- printed table ----
        print(f"\n  RECALL@10 class-pair contrasts "
              f"(effect [95% CI cluster]; p_query/p_cluster/p_sign):")
        for r in sorted(rows, key=lambda x: -x['metrics']['recall']['effect']):
            m = r['metrics']['recall']
            if m['better'] is None:
                label, eff = f"{r['a']} == {r['b']}".ljust(37), "  0.0000"
            else:
                los = r['b'] if m['better'] == r['a'] else r['a']
                label = f"{m['better']} > {los}".ljust(37)
                eff = f"{m['effect']:+.4f}"
            mark = "  <-- DISAGREEMENT" if m['conclusion_changes'] else ""
            print(f"    {label} d={eff} "
                  f"[{m['ci95_cluster_lo']:+.4f},{m['ci95_cluster_hi']:+.4f}]  "
                  f"p q={m['p_query']:.5f} c={m['p_cluster']:.5f} s={m['p_sign']:.4f}"
                  f"  | surv q/c/s = {int(m['survive_query'])}/{int(m['survive_cluster'])}/"
                  f"{int(m['survive_sign'])} | strict_q={int(m['survive_strict_query'])}"
                  f"{mark}")

        print(f"\n  MRR class-pair contrasts:")
        for r in sorted(rows, key=lambda x: -x['metrics']['mrr']['effect']):
            m = r['metrics']['mrr']
            if m['better'] is None:
                label, eff = f"{r['a']} == {r['b']}".ljust(37), "  0.0000"
            else:
                los = r['b'] if m['better'] == r['a'] else r['a']
                label = f"{m['better']} > {los}".ljust(37)
                eff = f"{m['effect']:+.4f}"
            mark = "  <-- DISAGREEMENT" if m['conclusion_changes'] else ""
            print(f"    {label} d={eff} "
                  f"[{m['ci95_cluster_lo']:+.4f},{m['ci95_cluster_hi']:+.4f}]  "
                  f"p q={m['p_query']:.5f} c={m['p_cluster']:.5f} s={m['p_sign']:.4f}"
                  f"  | surv q/c/s = {int(m['survive_query'])}/{int(m['survive_cluster'])}/"
                  f"{int(m['survive_sign'])}{mark}")

        cr = out['corpora'][corpus]
        print(f"\n  SURVIVORS recall: query={cr['n_survive_recall_query']} "
              f"cluster={cr['n_survive_recall_cluster']} sign={cr['n_survive_recall_sign']} "
              f"(strict-alpha query={cr['n_survive_recall_strict']})  "
              f"disagreements={cr['n_changes_recall']}")
        print(f"  SURVIVORS mrr   : query={cr['n_survive_mrr_query']} "
              f"cluster={cr['n_survive_mrr_cluster']} sign={cr['n_survive_mrr_sign']}  "
              f"disagreements={cr['n_changes_mrr']}")

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote evaluation/{os.path.basename(OUT)}")


if __name__ == '__main__':
    main()
