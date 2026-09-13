"""
Full ablation matrix evaluation - runs all 10 rankers on both corpora
through both metrics.py and bootstrap_significance.py paths.
"""
import os
import sys
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ablation_rankers import get_ranker, CORPUS_CONFIG, RANKERS

# Load qrels for both corpora
with open('data/qrels.json') as f:
    qrels_small = [q for q in json.load(f) if q.get('relevant_doc_ids')]

with open('data/qrels_expanded.json') as f:
    qrels_expanded = [q for q in json.load(f) if q.get('relevant_doc_ids')]

QRELS = {
    'small': qrels_small,
    'expanded': qrels_expanded,
}

RANKER_NAMES = [
    'BM25', 'Dense', 'Hybrid', 'Hybrid+Temporal', 'Hybrid+Metadata',
    'Final', 'BM25+Temporal', 'Dense+Temporal', 'TemporalOnly', 'MetadataOnly'
]

K = 10
FETCH_MULT = 5  # size * 5 for candidate pool


def compute_query_metrics(query_text, true_docs, retrieval_fn, k=K):
    """Compute per-query metrics for a retrieval function."""
    results = retrieval_fn(query_text, size=k * FETCH_MULT)
    retrieved_docs = [res['parent_doc_id'] for res in results]
    
    seen = set()
    dedup_docs = []
    for d in retrieved_docs:
        if d not in seen:
            seen.add(d)
            dedup_docs.append(d)
    dedup_docs = dedup_docs[:k]
    
    hits = [1 if d in true_docs else 0 for d in dedup_docs]
    
    p_at_k = sum(hits) / k if k > 0 else 0
    recall_at_k = sum(hits) / max(len(true_docs), 1) if true_docs else 0
    
    mrr = 0
    for rank, d in enumerate(dedup_docs):
        if d in true_docs:
            mrr = 1.0 / (rank + 1)
            break
    
    return {'p_at_k': p_at_k, 'recall_at_k': recall_at_k, 'mrr': mrr}


def evaluate_corpus(corpus_name, qrels, verbose=True):
    """Evaluate all rankers on a single corpus through metrics path."""
    if verbose:
        print("\n" + "=" * 80)
        print("EVALUATION: " + CORPUS_CONFIG[corpus_name]['name'].upper() + " CORPUS (" + str(len(qrels)) + " queries)")
        print("=" * 80)
    
    results = {}
    
    for name in RANKER_NAMES:
        if verbose:
            print("\nEvaluating " + name + "...")
        fn = get_ranker(name, corpus_name)
        
        p_list, r_list, mrr_list = [], [], []
        for q in qrels:
            metrics = compute_query_metrics(q['query_text'], q['relevant_doc_ids'], fn)
            p_list.append(metrics['p_at_k'])
            r_list.append(metrics['recall_at_k'])
            mrr_list.append(metrics['mrr'])
        
        results[name] = {
            'P@10': float(np.mean(p_list)),
            'Recall@10': float(np.mean(r_list)),
            'MRR': float(np.mean(mrr_list)),
            'per_query': {
                'p': p_list, 'r': r_list, 'mrr': mrr_list
            }
        }
        
        if verbose:
            print("  P@10=" + "{:.3f}".format(results[name]['P@10']) + 
                  "  R@10=" + "{:.3f}".format(results[name]['Recall@10']) + 
                  "  MRR=" + "{:.3f}".format(results[name]['MRR']))
    
    return results


def bootstrap_ci(recalls, mrrs, n_bootstrap=10000, ci=95):
    """Compute bootstrap confidence intervals."""
    n = len(recalls)
    boot_recalls = []
    boot_mrrs = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_recalls.append(np.mean(np.array(recalls)[idx]))
        boot_mrrs.append(np.mean(np.array(mrrs)[idx]))
    
    boot_recalls = np.array(boot_recalls)
    boot_mrrs = np.array(boot_mrrs)
    
    alpha = (100 - ci) / 2
    recall_ci = np.percentile(boot_recalls, [alpha, 100 - alpha])
    mrr_ci = np.percentile(boot_mrrs, [alpha, 100 - alpha])
    
    return {
        'recall_mean': float(np.mean(boot_recalls)),
        'recall_ci': [float(recall_ci[0]), float(recall_ci[1])],
        'mrr_mean': float(np.mean(boot_mrrs)),
        'mrr_ci': [float(mrr_ci[0]), float(mrr_ci[1])]
    }


def run_bootstrap_significance(corpus_name, qrels, metrics_results):
    """Run bootstrap significance testing for all pairwise comparisons."""
    if True:
        print("\n" + "=" * 80)
        print("BOOTSTRAP SIGNIFICANCE: " + CORPUS_CONFIG[corpus_name]['name'].upper() + " CORPUS")
        print("=" * 80)
    
    # Pre-compute per-query metrics for bootstrap
    query_metrics = {}
    for name in RANKER_NAMES:
        fn = get_ranker(name, corpus_name)
        query_metrics[name] = {
            'recall': [],
            'mrr': []
        }
        for q in qrels:
            m = compute_query_metrics(q['query_text'], q['relevant_doc_ids'], fn)
            query_metrics[name]['recall'].append(m['recall_at_k'])
            query_metrics[name]['mrr'].append(m['mrr'])
        query_metrics[name]['recall'] = np.array(query_metrics[name]['recall'])
        query_metrics[name]['mrr'] = np.array(query_metrics[name]['mrr'])
    
    # Compute CIs
    ci_results = {}
    for name in RANKER_NAMES:
        ci = bootstrap_ci(query_metrics[name]['recall'], query_metrics[name]['mrr'])
        ci_results[name] = ci
        if True:
            print("\n" + name + ":")
            print("  Recall@10: {:.3f} [{:.3f}, {:.3f}]".format(
                ci['recall_mean'], ci['recall_ci'][0], ci['recall_ci'][1]))
            print("  MRR:       {:.3f} [{:.3f}, {:.3f}]".format(
                ci['mrr_mean'], ci['mrr_ci'][0], ci['mrr_ci'][1]))
    
    # Pairwise comparisons
    if True:
        print("\n" + "=" * 80)
        print("PAIRWISE COMPARISONS (Recall@10) - " + CORPUS_CONFIG[corpus_name]['name'].upper())
        print("=" * 80)
    
    pairwise = {}
    for i in range(len(RANKER_NAMES)):
        for j in range(i+1, len(RANKER_NAMES)):
            m1, m2 = RANKER_NAMES[i], RANKER_NAMES[j]
            diff = ci_results[m1]['recall_mean'] - ci_results[m2]['recall_mean']
            
            ci1 = ci_results[m1]['recall_ci']
            ci2 = ci_results[m2]['recall_ci']
            overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
            
            significance = "SIGNIFICANT" if not overlap else "NOT significant"
            
            pairwise[(m1, m2)] = {
                'diff': diff,
                'm1_ci': ci1,
                'm2_ci': ci2,
                'significant': not overlap
            }
            
            if True:
                print(m1 + " vs " + m2 + ": diff=" + "{:+.3f}".format(diff) + " [" + significance + "]")
                print("  " + m1 + ": [" + "{:.3f}".format(ci1[0]) + ", " + "{:.3f}".format(ci1[1]) + "]")
                print("  " + m2 + ": [" + "{:.3f}".format(ci2[0]) + ", " + "{:.3f}".format(ci2[1]) + "]")
    
    return ci_results, pairwise


def main():
    print("FULL ABLATION MATRIX EVALUATION")
    print("=================================")
    
    all_results = {}
    all_ci = {}
    all_pairwise = {}
    
    for corpus_name in ['small', 'expanded']:
        qrels = QRELS[corpus_name]
        
        # Path 1: metrics.py evaluation
        metrics_results = evaluate_corpus(corpus_name, qrels)
        
        # Path 2: bootstrap significance
        ci_results, pairwise = run_bootstrap_significance(corpus_name, qrels, metrics_results)
        
        all_results[corpus_name] = metrics_results
        all_ci[corpus_name] = ci_results
        all_pairwise[corpus_name] = pairwise
    
    # Cross-corpus comparison
    print("\n" + "=" * 80)
    print("CROSS-CORPUS COMPARISON")
    print("=" * 80)
    
    print("\n{:<20} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(
        "Ranker", "Sm R@10", "Ex R@10", "Delta R@10", "Sm MRR", "Ex MRR", "Delta MRR"))
    print("-" * 100)
    
    for name in RANKER_NAMES:
        sm_r = all_results['small'][name]['Recall@10']
        ex_r = all_results['expanded'][name]['Recall@10']
        sm_m = all_results['small'][name]['MRR']
        ex_m = all_results['expanded'][name]['MRR']
        print("{:<20} | {:>10.3f} | {:>10.3f} | {:>+10.3f} | {:>10.3f} | {:>10.3f} | {:>+10.3f}".format(
            name, sm_r, ex_r, ex_r - sm_r, sm_m, ex_m, ex_m - sm_m))
    
    # Check for direction flips (like Temporal did)
    print("\nDirection flips (small > expanded vs small < expanded):")
    for name in RANKER_NAMES:
        sm_r = all_results['small'][name]['Recall@10']
        ex_r = all_results['expanded'][name]['Recall@10']
        sm_m = all_results['small'][name]['MRR']
        ex_m = all_results['expanded'][name]['MRR']
        r_flip = (sm_r > ex_r) != (sm_m > ex_m)  # If ranking by R@10 vs MRR disagrees on direction
        if sm_r != ex_r:
            print("  " + name + ": R@10 {:.3f} -> {:.3f} (delta={:+.3f})".format(sm_r, ex_r, ex_r - sm_r))
    
    # Save full results
    output = {
        'metrics': all_results,
        'bootstrap_ci': all_ci,
        'pairwise': all_pairwise,
        'cross_corpus': {
            name: {
                'small_recall': all_results['small'][name]['Recall@10'],
                'expanded_recall': all_results['expanded'][name]['Recall@10'],
                'small_mrr': all_results['small'][name]['MRR'],
                'expanded_mrr': all_results['expanded'][name]['MRR'],
            } for name in RANKER_NAMES
        }
    }
    
    out_path = os.path.join(os.path.dirname(__file__), 'ablation_matrix_results.json')
    with open(out_path, 'w') as f:
        # Convert numpy types and tuple keys
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    if isinstance(k, tuple):
                        new_dict[str(k)] = convert(v)
                    else:
                        new_dict[k] = convert(v)
                return new_dict
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(output), f, indent=2)
    
    print("\nResults saved to " + out_path)


if __name__ == '__main__':
    main()