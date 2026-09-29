"""Paired-bootstrap significance for all 90 ablation contrasts.

Replaces the CI-overlap test used in the ablation matrix with a proper paired
bootstrap on per-query metric differences.

Why the old test was wrong: CI overlap between two independently bootstrapped
means is not a test of their difference. Two CIs can fail to overlap while the
paired difference is not significant, and (more often here) they can overlap
while the paired difference is overwhelmingly significant — because unpaired
resampling discards the within-query correlation that the paired design
preserves. With one query set, the same documents are scored by every ranker,
so the contrast is inherently paired and should be tested that way.

Method:
  - ONE resample matrix of query indices per corpus, generated once and reused
    for every contrast, so all 90 contrasts see identical resamples.
  - For a contrast, the paired difference vector d = metric_A - metric_B is
    formed per query, then bootstrapped: mean(d[idx]).
    mean(A[idx]) - mean(B[idx]) == mean(d[idx]) when the same idx is used,
    so this is exactly the paired difference distribution.
  - Two-sided p with the +1 correction, p = 2*min((count+1)/(B+1)) style, so a
    zero count cannot report p = 0.
  - Bonferroni across the full 90-contrast family: alpha = 0.05/90.

No ranker or corpus changes: this reads per-query metrics already computed by
run_ablation_matrix.py and re-tests them.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'evaluation', 'ablation_matrix_results.json')
OUT = os.path.join(REPO, 'evaluation', 'paired_bootstrap_significance.json')

N_CONTRASTS = 45          # C(10, 2)
N_CORPORA = 2
FAMILY = N_CONTRASTS * N_CORPORA      # 90
ALPHA = 0.05 / FAMILY                # 0.000555...
N_BOOT = 50_000
SEED = 20260928

RANKERS = ["BM25", "Dense", "Hybrid", "Hybrid+Temporal", "Hybrid+Metadata",
           "Final", "BM25+Temporal", "Dense+Temporal", "TemporalOnly",
           "MetadataOnly"]
METRICS = {"recall": "Recall@10", "mrr": "MRR"}


def paired_p(a, b, idx, min_p):
    """Two-sided paired bootstrap p for mean(a) - mean(b) under shared resamples."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(d, 0):
        return 1.0, 0.0
    boot = d[idx].mean(axis=1)
    n_neg = int((boot <= 0).sum())
    n_pos = int((boot >= 0).sum())
    p = 2.0 * min((n_neg + 1) / (len(idx) + 1), (n_pos + 1) / (len(idx) + 1))
    return min(p, 1.0), float(boot.mean())


def main():
    with open(RESULTS, encoding='utf-8') as f:
        data = json.load(f)

    print(f"Bonferroni alpha over {FAMILY} contrasts = {ALPHA:.6f}")
    print(f"Resamples per contrast = {N_BOOT} (shared across all contrasts)\n")

    out = {
        'method': 'paired bootstrap on per-query metric differences',
        'n_contrasts': FAMILY,
        'n_contrasts_per_corpus': N_CONTRASTS,
        'n_resamples': N_BOOT,
        'seed': SEED,
        'bonferroni_alpha': ALPHA,
        'test': 'two-sided, +1 correction',
        'corpora': {},
    }

    for corp in ('small', 'expanded'):
        pq = data['metrics'][corp]
        n_q = len(pq['BM25']['per_query']['r'])
        print("=" * 78)
        print(f"{corp.upper()} CORPUS  (n={n_q} queries)")
        print("=" * 78)

        # One resample matrix, reused for every contrast in this corpus.
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n_q, size=(N_BOOT, n_q))
        min_p = 2.0 / (N_BOOT + 1)
        print(f"  shared resample matrix: {idx.shape}, min resolvable p = {min_p:.2e}\n")

        series = {m: {r: np.array(pq[r]['per_query'][k], dtype=float)
                      for r in RANKERS}
                  for k, m in (('r', 'recall'), ('mrr', 'mrr'))}

        old = data['pairwise'][corp]
        rows, survivors = [], {m: [] for m in METRICS}
        changes = []

        for i in range(len(RANKERS)):
            for j in range(i + 1, len(RANKERS)):
                a, b = RANKERS[i], RANKERS[j]
                key = f"('{a}', '{b}')"
                rec = {'a': a, 'b': b}
                for m in METRICS:
                    p, obs = paired_p(series[m][a], series[m][b], idx, min_p)
                    rec[m] = {'p': p, 'obs_diff': obs,
                              'survives_bonferroni': bool(p < ALPHA)}
                    # obs_diff is the bootstrap mean of (a - b). Record the
                    # winner explicitly so consumers never have to infer
                    # direction from the pair ordering.
                    if np.allclose(obs, 0):
                        rec[m]['better'], rec[m]['worse'] = None, None
                    elif obs > 0:
                        rec[m]['better'], rec[m]['worse'] = a, b
                    else:
                        rec[m]['better'], rec[m]['worse'] = b, a
                    if p < ALPHA:
                        survivors[m].append((a, b, obs, p))
                # CI-overlap status from the stored matrix (recall only)
                prev = old.get(key, {})
                rec['ci_overlap_significant'] = bool(prev.get('significant', False))
                rec['ci_overlap_diff'] = prev.get('diff')
                rec['changed'] = bool(
                    rec['recall']['survives_bonferroni'] != rec['ci_overlap_significant'])
                if rec['changed']:
                    changes.append({
                        'contrast': key,
                        'ci_overlap': rec['ci_overlap_significant'],
                        'paired_bootstrap': rec['recall']['survives_bonferroni'],
                        'p': rec['recall']['p'],
                        'obs_diff': rec['recall']['obs_diff'],
                    })
                rows.append(rec)

        out['corpora'][corp] = {'n_queries': n_q, 'contrasts': rows,
                                'survivors_recall': survivors['recall'],
                                'survivors_mrr': survivors['mrr'],
                                'status_changes': changes}

        n_contr = len(rows)
        print(f"  contrasts tested            : {n_contr}")
        print(f"  survive Bonferroni (Recall) : {len(survivors['recall'])}")
        print(f"  survive Bonferroni (MRR)    : {len(survivors['mrr'])}")
        print(f"  CI-overlap said significant : "
              f"{sum(1 for r in rows if r['ci_overlap_significant'])}")
        print(f"  status changes              : {len(changes)}\n")

        print("  --- surviving Recall@10 contrasts (paired bootstrap) ---")
        for a, b, obs, p in sorted(survivors['recall'], key=lambda x: -x[2]):
            # obs is the bootstrap mean of (a - b): positive => a is better.
            # Do not assume the pair order (a, b) reflects the direction.
            if obs > 0:
                win, los, mag = a, b, obs
            else:
                win, los, mag = b, a, -obs
            print(f"    {win} > {los:<22} d={mag:+.4f}  p={p:.6f}")
        if changes:
            print("\n  --- STATUS CHANGES vs CI-overlap ---")
            for c in sorted(changes, key=lambda x: -x['p']):
                direction = ("now SURVIVES" if c['paired_bootstrap']
                             else "no longer survives")
                print(f"    {c['contrast']:<48} d={c['obs_diff']:+.4f} "
                      f"p={c['p']:.6f}  -> {direction}")
        print()

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"wrote evaluation/{os.path.basename(OUT)}")

    total_new = sum(len(out['corpora'][c]['survivors_recall']) for c in out['corpora'])
    total_old = sum(1 for c in out['corpora'] for r in out['corpora'][c]['contrasts']
                    if r['ci_overlap_significant'])
    total_chg = sum(len(out['corpora'][c]['status_changes']) for c in out['corpora'])
    print(f"\nTOTALS  CI-overlap: {total_old}   paired: {total_new}   "
          f"changed: {total_chg}")


if __name__ == '__main__':
    main()
