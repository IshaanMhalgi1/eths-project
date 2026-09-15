"""
Apply Bonferroni correction to pairwise comparisons.
90 total comparisons (45 per corpus x 2 corpora).
Alpha = 0.05 / 90 = 0.000556
Need 99.944% CIs to not overlap (or equivalent p < 0.000556)
"""
import json

# Load results
with open('evaluation/ablation_matrix_results_v2.json') as f:
    results = json.load(f)

# The bootstrap CI results are in results['bootstrap_ci']
# But we need the per-query metrics to recompute with adjusted alpha
# Let's approximate: if 95% CI don't overlap, p ≈ 0.05 (conservative)
# For Bonferroni p < 0.000556, need ~99.94% CI non-overlap
# Equivalent: mean difference > ~3.3 * pooled SE (instead of 2.0)

# Simpler: count how many "SIGNIFICANT" results have large effect sizes
# Effect size = diff / pooled_CI_width

RANKERS = ['BM25', 'Dense', 'Hybrid', 'Hybrid+Temporal', 'Hybrid+Metadata',
           'Final', 'BM25+Temporal', 'Dense+Temporal', 'TemporalOnly', 'MetadataOnly']

def check_bonferroni(corpus_name, ci_results):
    print(f"\n=== {corpus_name} Corpus - Bonferroni Analysis ===")
    print(f"90 total tests, alpha = 0.05/90 = 0.000556")
    print(f"Need ~99.94% CI non-overlap (equivalent to mean_diff > ~3.3 * pooled_SE)")
    print()
    
    survivors = []
    for i in range(len(RANKERS)):
        for j in range(i+1, len(RANKERS)):
            m1, m2 = RANKERS[i], RANKERS[j]
            ci1 = ci_results[m1]['recall_ci']
            ci2 = ci_results[m2]['recall_ci']
            diff = ci_results[m1]['recall_mean'] - ci_results[m2]['recall_mean']
            
            # Pooled CI half-width as proxy for SE
            hw1 = (ci1[1] - ci1[0]) / 2
            hw2 = (ci2[1] - ci2[0]) / 2
            pooled_hw = (hw1 + hw2) / 2
            
            # Bonferroni threshold: diff > 3.3 * pooled_hw (approx)
            threshold = 3.3 * pooled_hw
            survives = abs(diff) > threshold
            
            # Also check 95% CI overlap (current "SIGNIFICANT")
            overlap_95 = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
            sig_95 = not overlap_95
            
            if sig_95:
                status = "SURVIVES" if survives else "FAILS Bonferroni"
                print(f"  {m1:>20} vs {m2:<20}: diff={diff:+.3f}, hw={pooled_hw:.3f}, thresh={threshold:.3f} -> {status}")

# Small corpus
print("SMALL CORPUS (n=19, wide CIs)")
ci_small = results['bootstrap_ci']['small']
check_bonferroni('small', ci_small)

# Expanded corpus
print("\nEXPANDED CORPUS (n=90, tighter CIs)")
ci_expanded = results['bootstrap_ci']['expanded']
check_bonferroni('expanded', ci_expanded)

print("\n=== Summary ===")
print("Small corpus: n=19 -> wide CIs -> few survive Bonferroni")
print("Expanded corpus: n=90 -> tighter CIs -> more survive")