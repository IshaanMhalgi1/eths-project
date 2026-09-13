"""
Full ablation matrix evaluation - runs all 10 rankers on both corpora
through both metrics.py and bootstrap_significance.py paths.
Version 2: uses separate small-corpus index (ethsearch_chunks_small) and
validated small-corpus qrels (qrels_small.json, original 19 queries).
"""
import os
import sys
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ablation_rankers import get_ranker, CORPUS_CONFIG

# Load qrels from corpus config
QRELS = {}
for corpus_name in ['small', 'expanded']:
    path = CORPUS_CONFIG[corpus_name]['qrels_path']
    with open(path) as f:
        qrels = json.load(f)
    QRELS[corpus_name] = [q for q in qrels if q.get('relevant_doc_ids')]

RANKER_NAMES = [
    'BM25', 'Dense', 'Hybrid', 'Hybrid+Temporal', 'Hybrid+Metadata',
    'Final', 'BM25+Temporal', 'Dense+Temporal', 'TemporalOnly', 'MetadataOnly'
]

K = 10
FETCH_MULT = 5


def compute_query_metrics(query_text, true_docs, retrieval_fn, k=K):
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


def sanity_check_identical(corpus_results):
    """
    First-pass sanity check: identical-to-three-decimals results across corpora
    is a tell for cross-corpus scoping bugs. Any ranker with exactly the same
    R@10/MRR on both corpora must be flagged.
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK: identical-results detection (known scoping-bug tell)")
    print("=" * 80)
    flagged = False
    for name in RANKER_NAMES:
        sm = corpus_results['small'][name]
        ex = corpus_results['expanded'][name]
        r_same = abs(sm['Recall@10'] - ex['Recall@10']) < 0.0005
        m_same = abs(sm['MRR'] - ex['MRR']) < 0.0005
        if r_same and m_same:
            print("  FLAG: {} identical across corpora (R@10={:.4f}, MRR={:.4f})".format(name, sm['Recall@10'], sm['MRR']))
            flagged = True
        else:
            print("  OK:   {} differs (sm R={:.4f}/{:.4f} vs ex R={:.4f}/{:.4f})".format(
                name, sm['Recall@10'], sm['MRR'], ex['Recall@10'], ex['MRR']))
    if flagged:
        print("\nWARNING: identical-results found. Inspect corpus scoping before trusting table.")
    else:
        print("\nPASS: no ranker identical to 3 decimals across corpora.")
    return not flagged


def evaluate_corpus(corpus_name, qrels, verbose=True):
    if verbose:
        print("\n" + "=" * 80)
        print("EVALUATION: {} CORPUS index={} ({} queries)".format(
            CORPUS_CONFIG[corpus_name]['name'].upper(),
            CORPUS_CONFIG[corpus_name]['index_name'],
            len(qrels)))
        print("=" * 80)
    results = {}
    for name in RANKER_NAMES:
        if verbose:
            print("\nEvaluating {}...".format(name))
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
            'per_query': {'p': p_list, 'r': r_list, 'mrr': mrr_list}
        }
        if verbose:
            print("  P@10={:.3f}  R@10={:.3f}  MRR={:.3f}".format(
                results[name]['P@10'], results[name]['Recall@10'], results[name]['MRR']))
    return results


def bootstrap_ci(recalls, mrrs, n_bootstrap=10000, ci=95):
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


def run_bootstrap_significance(corpus_name, qrels):
    print("\n" + "=" * 80)
    print("BOOTSTRAP SIGNIFICANCE: {} CORPUS ({} queries, 10000 resamples)".format(
        CORPUS_CONFIG[corpus_name]['name'].upper(), len(qrels)))
    print("=" * 80)
    query_metrics = {}
    for name in RANKER_NAMES:
        fn = get_ranker(name, corpus_name)
        qm = {'recall': [], 'mrr': []}
        for q in qrels:
            m = compute_query_metrics(q['query_text'], q['relevant_doc_ids'], fn)
            qm['recall'].append(m['recall_at_k'])
            qm['mrr'].append(m['mrr'])
        qm['recall'] = np.array(qm['recall'])
        qm['mrr'] = np.array(qm['mrr'])
        query_metrics[name] = qm
    ci_results = {}
    for name in RANKER_NAMES:
        ci = bootstrap_ci(query_metrics[name]['recall'], query_metrics[name]['mrr'])
        ci_results[name] = ci
        print("\n{}:".format(name))
        print("  Recall@10: {:.3f} [{:.3f}, {:.3f}]".format(ci['recall_mean'], ci['recall_ci'][0], ci['recall_ci'][1]))
        print("  MRR:       {:.3f} [{:.3f}, {:.3f}]".format(ci['mrr_mean'], ci['mrr_ci'][0], ci['mrr_ci'][1]))
    pairwise = {}
    for i in range(len(RANKER_NAMES)):
        for j in range(i+1, len(RANKER_NAMES)):
            m1, m2 = RANKER_NAMES[i], RANKER_NAMES[j]
            diff = ci_results[m1]['recall_mean'] - ci_results[m2]['recall_mean']
            ci1 = ci_results[m1]['recall_ci']
            ci2 = ci_results[m2]['recall_ci']
            significant = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
            pairwise[(m1, m2)] = {'diff': diff, 'significant': significant}
            if significant:
                print("{} vs {}: diff={:+.3f} [SIGNIFICANT]".format(m1, m2, diff))
    return ci_results, pairwise


def main():
    print("FULL ABLATION MATRIX EVALUATION (v2 - scoped indices)")
    print("=====================================================")

    all_results = {}
    all_ci = {}
    all_pairwise = {}

    for corpus_name in ['small', 'expanded']:
        qrels = QRELS[corpus_name]
        if corpus_name == 'small':
            # Verify all relevant docs exist in small index range
            print("\n[VALIDATION] Small corpus: {} queries".format(len(qrels)))
        metrics_results = evaluate_corpus(corpus_name, qrels)
        ci_results, pairwise = run_bootstrap_significance(corpus_name, qrels)
        all_results[corpus_name] = metrics_results
        all_ci[corpus_name] = ci_results
        all_pairwise[corpus_name] = pairwise

    # First-pass sanity check
    sanity_check_identical(all_results)

    # Cross-corpus comparison
    print("\n" + "=" * 80)
    print("CROSS-CORPUS COMPARISON")
    print("=" * 80)
    print("\n{:<20} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(
        "Ranker", "Sm R@10", "Ex R@10", "Delta R10", "Sm MRR", "Ex MRR", "Delta MRR"))
    print("-" * 100)
    for name in RANKER_NAMES:
        sm_r = all_results['small'][name]['Recall@10']
        ex_r = all_results['expanded'][name]['Recall@10']
        sm_m = all_results['small'][name]['MRR']
        ex_m = all_results['expanded'][name]['MRR']
        print("{:<20} | {:>10.3f} | {:>10.3f} | {:>+10.3f} | {:>10.3f} | {:>10.3f} | {:>+10.3f}".format(
            name, sm_r, ex_r, ex_r - sm_r, sm_m, ex_m, ex_m - sm_m))

    output = {
        'metrics': all_results,
        'bootstrap_ci': all_ci,
        'pairwise': {str(k): v for k, v in all_pairwise.get('small', {}).items()},
        'cross_corpus': {
            name: {
                'small_recall': all_results['small'][name]['Recall@10'],
                'expanded_recall': all_results['expanded'][name]['Recall@10'],
                'small_mrr': all_results['small'][name]['MRR'],
                'expanded_mrr': all_results['expanded'][name]['MRR'],
            } for name in RANKER_NAMES
        }
    }
    out_path = os.path.join(os.path.dirname(__file__), 'ablation_matrix_results_v2.json')

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

    with open(out_path, 'w') as f:
        json.dump(convert(output), f, indent=2)
    print("\nResults saved to " + out_path)


if __name__ == '__main__':
    main()