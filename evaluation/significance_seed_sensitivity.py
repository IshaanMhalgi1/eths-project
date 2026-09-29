"""Seed sensitivity for the paired-bootstrap significance results.

`paired_bootstrap_significance.py` reports p-values from a single resample
matrix (seed 20260928). A contrast whose p lands near the Bonferroni threshold
(0.000556) may be significant under one draw and not another, so a single-draw
verdict on such a contrast is not reproducible.

This re-tests every contrast under several INDEPENDENT resample matrices and
reports, per contrast, the median/min/max p and whether survival is stable.

Purpose: separate "survives" (robust) from "flips" (indeterminate). A contrast
that flips must be reported as NOT established, regardless of what the primary
seed happened to return.

Reads the same per-query metrics as paired_bootstrap_significance.py and makes no
corpus or ranker changes.
"""
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'evaluation', 'ablation_matrix_results.json')
OUT = os.path.join(REPO, 'evaluation', 'significance_seed_sensitivity.json')

ALPHA = 0.05 / 90          # 0.000555...
N_BOOT = 50_000
SEEDS = (1, 2, 3, 4, 5)

RANKERS = ["BM25", "Dense", "Hybrid", "Hybrid+Temporal", "Hybrid+Metadata",
           "Final", "BM25+Temporal", "Dense+Temporal", "TemporalOnly",
           "MetadataOnly"]
METRICS = {"recall": "r", "mrr": "mrr"}


def paired_p(d, idx):
    boot = d[idx].mean(axis=1)
    n_neg = int((boot <= 0).sum())
    n_pos = int((boot >= 0).sum())
    return 2.0 * min((n_neg + 1) / (len(idx) + 1), (n_pos + 1) / (len(idx) + 1))


def main():
    with open(RESULTS, encoding='utf-8') as f:
        data = json.load(f)

    out = {'method': 'paired bootstrap re-tested under independent resample matrices',
           'seeds': list(SEEDS), 'n_resamples': N_BOOT,
           'bonferroni_alpha': ALPHA, 'corpora': {}}

    for corp in ('small', 'expanded'):
        pq = data['metrics'][corp]
        n = len(pq['BM25']['per_query']['r'])

        idxs = [np.random.default_rng(s).integers(0, n, size=(N_BOOT, n))
                for s in SEEDS]

        series = {m: {r: np.array(pq[r]['per_query'][k], dtype=float)
                      for r in RANKERS}
                  for k, m in (('r', 'recall'), ('mrr', 'mrr'))}

        per_metric = {}
        for m, key in METRICS.items():
            rows, stable, unstable = [], [], []
            for i in range(len(RANKERS)):
                for j in range(i + 1, len(RANKERS)):
                    a, b = RANKERS[i], RANKERS[j]
                    d = series[m][a] - series[m][b]
                    if np.allclose(d, 0):
                        continue          # identical rankers: no contrast exists
                    ps = [paired_p(d, idx) for idx in idxs]
                    med = float(np.median(ps))
                    is_stable = all(p < ALPHA for p in ps)
                    flips = any(p < ALPHA for p in ps) and not is_stable
                    rec = {'a': a, 'b': b, 'obs_diff': float(d.mean()),
                           'p_median': med, 'p_min': float(min(ps)),
                           'p_max': float(max(ps)),
                           'stable_survivor': bool(is_stable),
                           'flips_across_seeds': bool(flips)}
                    rows.append(rec)
                    if is_stable:
                        stable.append(rec)
                    elif flips:
                        unstable.append(rec)
            per_metric[m] = {'n_stable_survivors': len(stable),
                             'n_flips_across_seeds': len(unstable),
                             'flipping': unstable,
                             'stable': stable}

        out['corpora'][corp] = {'n_queries': n, 'metrics': per_metric}

        print(f"=== {corp.upper()} (n={n}) — {len(SEEDS)} independent draws ===")
        for m in METRICS:
            pm = per_metric[m]
            print(f"  {m:<7} stable survivors: {pm['n_stable_survivors']:<3} "
                  f"flips across seeds: {pm['n_flips_across_seeds']}")
            for r in pm['flipping']:
                print(f"     FLIP  {r['a']} vs {r['b']}  d={r['obs_diff']:+.4f}  "
                      f"p range [{r['p_min']:.6f}, {r['p_max']:.6f}]  "
                      f"vs alpha {ALPHA:.6f}")
        print()

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"wrote evaluation/{os.path.basename(OUT)}")

    for corp in ('small', 'expanded'):
        for m in METRICS:
            pm = out['corpora'][corp]['metrics'][m]
            print(f"  {corp:<9} {m:<7} robust={pm['n_stable_survivors']:<3} "
                  f"indeterminate={pm['n_flips_across_seeds']}")


if __name__ == '__main__':
    main()
